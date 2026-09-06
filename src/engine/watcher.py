"""ytint // Continuous Live Streaming Watcher & Auto-Poller Daemon (src/engine/watcher.py)

Autonomous monitoring daemon that polls active YouTube channels and uploads,
adapts polling frequency based on incoming comment velocity, executes automated
incremental pipeline sweeps (runner.py --incremental), scans for threat outbreaks,
and persists telemetry for the Streamlit dashboard and CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Safe standard output configuration for Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure src directory is in sys.path
_src_dir = str(pathlib.Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from engine.config_loader import get_paths, load_config

logger = logging.getLogger("ytint.watcher")


# ==============================================================================
# 1. DATA STRUCTURES & SCHEMAS
# ==============================================================================

@dataclass
class WatchTarget:
    """A monitored YouTube video, live stream, or channel target."""
    target_type: str  # "video", "channel", "live_stream"
    target_id: str
    target_name: str
    last_polled_at: Optional[str] = None
    last_comment_count: int = 0
    comment_velocity_per_min: float = 0.0
    current_polling_interval_sec: int = 300
    status: str = "active"  # "active", "idle", "completed", "error"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WatcherPollCycleResult:
    """Summary record of a single watcher polling cycle."""
    cycle_index: int
    timestamp: str
    targets_checked: int
    new_comments_found: int
    updated_counters_found: int
    pipeline_sync_triggered: bool
    pipeline_sync_duration_sec: float
    threats_detected: int
    alerts_dispatched: int
    quota_used: int
    next_poll_in_sec: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WatcherDaemonState:
    """Full persistent telemetry state for the watcher daemon."""
    daemon_id: str
    status: str  # "running", "idle", "paused", "stopped"
    started_at: str
    last_heartbeat: str
    total_cycles: int
    total_comments_ingested: int
    total_syncs_executed: int
    total_alerts_dispatched: int
    total_quota_consumed: int
    monitored_targets: List[WatchTarget]
    recent_events: List[WatcherPollCycleResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "daemon_id": self.daemon_id,
            "status": self.status,
            "started_at": self.started_at,
            "last_heartbeat": self.last_heartbeat,
            "total_cycles": self.total_cycles,
            "total_comments_ingested": self.total_comments_ingested,
            "total_syncs_executed": self.total_syncs_executed,
            "total_alerts_dispatched": self.total_alerts_dispatched,
            "total_quota_consumed": self.total_quota_consumed,
            "monitored_targets": [t.to_dict() for t in self.monitored_targets],
            "recent_events": [e.to_dict() for e in self.recent_events],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ==============================================================================
# 2. WATCHER ENGINE DAEMON
# ==============================================================================

class YouTubeWatcherDaemon:
    """Continuous polling and automated pipeline synchronization daemon."""

    def __init__(
        self,
        interim_dir: Optional[pathlib.Path] = None,
        output_dir: Optional[pathlib.Path] = None,
        raw_db: Optional[pathlib.Path] = None,
        api_key: Optional[str] = None,
        base_interval_sec: int = 300,
        min_interval_sec: int = 60,
        max_interval_sec: int = 1800,
        adaptive: bool = True,
        auto_sync: bool = True,
        auto_alert: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or load_config()
        self.interim_dir = pathlib.Path(interim_dir) if interim_dir else pathlib.Path(self.config["paths"]["interim_dir"])
        self.output_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path(self.config["paths"]["output_dir"])
        self.raw_db = pathlib.Path(raw_db) if raw_db else pathlib.Path(self.config["paths"]["raw_db"])

        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
        self.base_interval_sec = base_interval_sec
        self.min_interval_sec = min_interval_sec
        self.max_interval_sec = max_interval_sec
        self.adaptive = adaptive
        self.auto_sync = auto_sync
        self.auto_alert = auto_alert

        self.state_file = self.output_dir / "watcher_state.json"
        self.targets: Dict[str, WatchTarget] = {}
        self.cycle_index = 0
        self.total_ingested = 0
        self.total_syncs = 0
        self.total_alerts = 0
        self.total_quota = 0
        self.recent_events: List[WatcherPollCycleResult] = []
        self.daemon_id = f"daemon_{int(time.time())}"
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.status = "idle"

        # Load existing state if available
        self._load_state_from_disk()

    def _load_state_from_disk(self) -> None:
        """Loads existing state from disk if present."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.cycle_index = data.get("total_cycles", 0)
                self.total_ingested = data.get("total_comments_ingested", 0)
                self.total_syncs = data.get("total_syncs_executed", 0)
                self.total_alerts = data.get("total_alerts_dispatched", 0)
                self.total_quota = data.get("total_quota_consumed", 0)

                for t_dict in data.get("monitored_targets", []):
                    t = WatchTarget(**t_dict)
                    self.targets[t.target_id] = t

                for e_dict in data.get("recent_events", [])[-20:]:
                    self.recent_events.append(WatcherPollCycleResult(**e_dict))
            except Exception as e:
                logger.warning(f"Could not load previous watcher state: {e}")

    def save_state(self) -> None:
        """Persists current state to JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state = self.get_state()
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                f.write(state.to_json())
        except Exception as e:
            logger.warning(f"Could not save watcher state: {e}")

    def get_state(self) -> WatcherDaemonState:
        """Constructs current daemon state snapshot."""
        return WatcherDaemonState(
            daemon_id=self.daemon_id,
            status=self.status,
            started_at=self.started_at,
            last_heartbeat=datetime.now(timezone.utc).isoformat(),
            total_cycles=self.cycle_index,
            total_comments_ingested=self.total_ingested,
            total_syncs_executed=self.total_syncs,
            total_alerts_dispatched=self.total_alerts,
            total_quota_consumed=self.total_quota,
            monitored_targets=list(self.targets.values()),
            recent_events=self.recent_events[-30:],
        )

    def add_target(self, target_type: str, target_id: str, target_name: Optional[str] = None) -> WatchTarget:
        """Registers a target video or channel for continuous monitoring."""
        name = target_name or target_id
        target = WatchTarget(
            target_type=target_type,
            target_id=target_id,
            target_name=name,
            current_polling_interval_sec=self.base_interval_sec,
            status="active",
        )
        self.targets[target_id] = target
        self.save_state()
        return target

    def auto_discover_targets(self, max_videos: int = 5) -> List[WatchTarget]:
        """Automatically selects recent videos from videos_clean.parquet or SQLite."""
        discovered: List[WatchTarget] = []
        parquet_videos = self.interim_dir / "videos_clean.parquet"

        if parquet_videos.exists():
            try:
                df_v = pd.read_parquet(parquet_videos)
                if not df_v.empty:
                    # Sort by published_at descending if available
                    if "published_at" in df_v.columns:
                        df_v = df_v.sort_values("published_at", ascending=False)
                    for _, row in df_v.head(max_videos).iterrows():
                        v_id = str(row["video_id"])
                        v_title = str(row.get("title", v_id))
                        t = self.add_target("video", v_id, v_title)
                        discovered.append(t)
                    return discovered
            except Exception as e:
                logger.warning(f"Failed reading video catalog for auto-discovery: {e}")

        # Fallback to mock default targets if empty
        if not self.targets:
            mock_targets = [
                ("video", "WpbN3D5oQBo", "Geopolitical War Room Briefing (Live Watch)"),
                ("video", "6h547XdZYiQ", "Field Analysis: Tactical Deployments & Narrative"),
                ("video", "k8n4L3m9P1a", "Weekly SitRep: Community Open Floor Q&A"),
            ]
            for t_type, t_id, t_name in mock_targets[:max_videos]:
                discovered.append(self.add_target(t_type, t_id, t_name))

        return discovered

    def calculate_adaptive_interval(self, velocity: float) -> int:
        """Computes next polling interval based on velocity scaling."""
        if not self.adaptive:
            return self.base_interval_sec

        # If velocity is high, shorten interval; if velocity is near zero, lengthen interval
        # Interval formula: base / (1 + log2(1 + velocity))
        scaling_factor = 1.0 + math.log2(1.0 + max(0.0, velocity))
        interval = int(self.base_interval_sec / scaling_factor)
        return max(self.min_interval_sec, min(self.max_interval_sec, interval))

    def poll_once(self, mock: bool = False) -> WatcherPollCycleResult:
        """Executes a single polling cycle across all registered targets."""
        self.cycle_index += 1
        self.status = "running"
        now_str = datetime.now(timezone.utc).isoformat()
        cycle_start_time = time.time()

        if not self.targets:
            self.auto_discover_targets(max_videos=3)

        new_comments_found = 0
        updated_counters_found = 0
        quota_cycle = 0
        error_msg: Optional[str] = None

        logger.info(f"⚡ [Cycle #{self.cycle_index}] Polling {len(self.targets)} targets (mock={mock})...")

        for target_id, target in list(self.targets.items()):
            try:
                if mock or not self.api_key:
                    # Synthetic arrival generator
                    # Deterministic but pseudo-random comments arrival
                    arrival_delta = max(1, (self.cycle_index * 7 + hash(target_id)) % 24)
                    velocity = round(arrival_delta / (max(target.current_polling_interval_sec, 60) / 60.0), 2)
                    target.last_comment_count += arrival_delta
                    target.comment_velocity_per_min = velocity
                    target.current_polling_interval_sec = self.calculate_adaptive_interval(velocity)
                    target.last_polled_at = now_str
                    target.status = "active"
                    new_comments_found += arrival_delta
                    quota_cycle += 1
                else:
                    # Live YouTube Data API v3 fetch
                    from engine.youtube_api import YouTubeClient, SQLiteCommentsuiteStore

                    client = YouTubeClient(self.api_key)
                    store = SQLiteCommentsuiteStore(self.raw_db)

                    if target.target_type == "video":
                        top_comments, replies = client.fetch_comment_threads(target.target_id, max_comments=100)
                        quota_cycle += client.quota_units_used
                        cnt_new = store.upsert_comments(top_comments) + store.upsert_comments(replies)
                        new_comments_found += cnt_new

                        # Estimate velocity
                        time_diff_min = (
                            max(target.current_polling_interval_sec, 60) / 60.0
                        )
                        target.comment_velocity_per_min = round(cnt_new / time_diff_min, 2)
                        target.last_comment_count += cnt_new
                        target.current_polling_interval_sec = self.calculate_adaptive_interval(target.comment_velocity_per_min)
                        target.last_polled_at = now_str
                        target.status = "active"

            except Exception as poll_err:
                logger.warning(f"Error polling target {target_id}: {poll_err}")
                target.status = "error"
                error_msg = str(poll_err)

        self.total_ingested += new_comments_found
        self.total_quota += quota_cycle

        # Check if automated sync should run
        sync_triggered = False
        sync_duration = 0.0
        threats_count = 0
        alerts_dispatched = 0

        if (new_comments_found > 0 or mock) and self.auto_sync:
            sync_triggered = True
            t_sync_start = time.time()
            logger.info(f"🔄 Delta detected (+{new_comments_found} comments). Triggering incremental pipeline sweep...")
            try:
                if not mock:
                    from pipeline.runner import PipelineRunner
                    runner = PipelineRunner()
                    runner.run(incremental=True)
                sync_duration = round(time.time() - t_sync_start, 2)
                self.total_syncs += 1
            except Exception as sync_err:
                logger.error(f"Incremental pipeline sweep failed: {sync_err}")
                error_msg = f"Sync failed: {sync_err}"

        # Check if automated threat scan should run
        if sync_triggered and self.auto_alert:
            try:
                from engine.alerting import AlertManager
                alert_mgr = AlertManager(self.output_dir, self.interim_dir, self.config)
                report = alert_mgr.scan_threats(severity_threshold="WARNING")
                threats_count = len(report.active_threats)
                if threats_count > 0 and self.config.get("alerting", {}).get("webhook_url"):
                    res = alert_mgr.dispatch_webhook()
                    if res.get("status") == "dispatched":
                        alerts_dispatched += 1
                        self.total_alerts += 1
            except Exception as alert_err:
                logger.debug(f"Threat scan note: {alert_err}")

        # Compute next poll time based on the minimum interval among targets
        active_intervals = [t.current_polling_interval_sec for t in self.targets.values() if t.status == "active"]
        next_poll_sec = min(active_intervals) if active_intervals else self.base_interval_sec

        result = WatcherPollCycleResult(
            cycle_index=self.cycle_index,
            timestamp=now_str,
            targets_checked=len(self.targets),
            new_comments_found=new_comments_found,
            updated_counters_found=updated_counters_found,
            pipeline_sync_triggered=sync_triggered,
            pipeline_sync_duration_sec=sync_duration,
            threats_detected=threats_count,
            alerts_dispatched=alerts_dispatched,
            quota_used=quota_cycle,
            next_poll_in_sec=next_poll_sec,
            error=error_msg,
        )

        self.recent_events.append(result)
        self.status = "idle"
        self.save_state()

        logger.info(
            f"✅ [Cycle #{self.cycle_index}] Checked {len(self.targets)} targets | "
            f"+{new_comments_found} comments | Sync: {sync_triggered} ({sync_duration}s) | "
            f"Next in {next_poll_sec}s"
        )
        return result

    def run_loop(self, max_iterations: Optional[int] = None, mock: bool = False) -> None:
        """Executes the continuous monitoring daemon loop."""
        print("\n" + "=" * 80)
        print("⚡ ytint-watch // Continuous Live Streaming Watcher & Auto-Poller Daemon")
        print("=" * 80)
        print(f"  • Base Polling Interval: {self.base_interval_sec}s (Min: {self.min_interval_sec}s, Max: {self.max_interval_sec}s)")
        print(f"  • Adaptive Velocity Scaling: {self.adaptive}")
        print(f"  • Automated Incremental Sync: {self.auto_sync}")
        print(f"  • Automated Threat Alerting: {self.auto_alert}")
        print(f"  • Operating Mode: {'MOCK SIMULATION' if mock else 'LIVE YOUTUBE API'}")
        print(f"  • Registered Targets: {len(self.targets)}")
        print("=" * 80 + "\n")

        iter_count = 0
        try:
            while True:
                iter_count += 1
                res = self.poll_once(mock=mock)

                if max_iterations is not None and iter_count >= max_iterations:
                    logger.info(f"Target iteration count ({max_iterations}) reached. Exiting cleanly.")
                    break

                sleep_time = res.next_poll_in_sec
                logger.info(f"⏳ Sleeping {sleep_time}s until next poll cycle (Ctrl+C to stop)...")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n🛑 Watcher daemon stopped by operator.")
        finally:
            self.status = "stopped"
            self.save_state()
            print("💾 Final daemon state persisted.")


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================

def main() -> None:
    """CLI entrypoint for ytint-watch."""
    parser = argparse.ArgumentParser(
        prog="ytint-watch",
        description="Continuous Live Streaming Watcher & Auto-Poller Daemon // ytint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ytint-watch --once --mock
  ytint-watch --status
  ytint-watch --mock --iterations 3
  ytint-watch --channel @mkbhd --interval 120 --adaptive
  ytint-watch --video WpbN3D5oQBo --auto-sync --auto-alert
        """,
    )
    parser.add_argument("--channel", type=str, default=None, help="Channel ID or handle to monitor.")
    parser.add_argument("--video", type=str, default=None, help="Specific video ID to monitor.")
    parser.add_argument("--interval", type=int, default=300, help="Base polling interval in seconds (default: 300).")
    parser.add_argument("--min-interval", type=int, default=60, help="Minimum adaptive interval in seconds (default: 60).")
    parser.add_argument("--max-interval", type=int, default=1800, help="Maximum adaptive interval in seconds (default: 1800).")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive velocity-based interval scaling.")
    parser.add_argument("--no-sync", action="store_true", help="Disable automated incremental pipeline sweeps on deltas.")
    parser.add_argument("--no-alert", action="store_true", help="Disable automated threat scanning and webhooks.")
    parser.add_argument("--iterations", type=int, default=None, help="Run N poll cycles then exit (useful for cron/tests).")
    parser.add_argument("--once", action="store_true", help="Run exactly one poll cycle then exit.")
    parser.add_argument("--mock", action="store_true", help="Run in mock simulation mode (no YouTube API quota used).")
    parser.add_argument("--status", action="store_true", help="Print active watcher daemon telemetry from disk and exit.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging.")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    # Status check mode
    if args.status:
        cfg = load_config()
        out_dir = pathlib.Path(cfg["paths"]["output_dir"])
        state_file = out_dir / "watcher_state.json"
        print("\n" + "=" * 80)
        print("📊 ytint-watch // Active Daemon Status & Telemetry")
        print("=" * 80)
        if not state_file.exists():
            print("⚠️ No active or previous watcher daemon state found on disk.")
            print(f"State file target: {state_file}")
            print("=" * 80 + "\n")
            return

        try:
            with open(state_file, "r", encoding="utf-8") as f:
                s = json.load(f)
            print(f"  • Daemon ID: {s.get('daemon_id')}")
            print(f"  • Current Status: {s.get('status', 'unknown').upper()}")
            print(f"  • Started At: {s.get('started_at')}")
            print(f"  • Last Heartbeat: {s.get('last_heartbeat')}")
            print(f"  • Total Cycles Executed: {s.get('total_cycles', 0):,}")
            print(f"  • Total Comments Ingested: {s.get('total_comments_ingested', 0):,}")
            print(f"  • Total Incremental Syncs: {s.get('total_syncs_executed', 0):,}")
            print(f"  • Total Quota Consumed: {s.get('total_quota_consumed', 0):,} units")

            targets = s.get("monitored_targets", [])
            print(f"\n🎯 Monitored Targets ({len(targets)}):")
            for t in targets:
                print(f"  - [{t.get('target_type')}] {t.get('target_name')} (ID: {t.get('target_id')}): Velocity={t.get('comment_velocity_per_min')} cmds/min, Next Interval={t.get('current_polling_interval_sec')}s, Status={t.get('status')}")

            events = s.get("recent_events", [])
            if events:
                print(f"\n📋 Recent Polling Events (Last {len(events)}):")
                for e in events[-5:]:
                    print(f"  - Cycle #{e.get('cycle_index')}: +{e.get('new_comments_found')} comments, Sync={e.get('pipeline_sync_triggered')} ({e.get('pipeline_sync_duration_sec')}s), Next in {e.get('next_poll_in_sec')}s at {e.get('timestamp')}")
        except Exception as err:
            print(f"❌ Error reading state file: {err}")
        print("=" * 80 + "\n")
        return

    # Normal daemon run mode
    daemon = YouTubeWatcherDaemon(
        base_interval_sec=args.interval,
        min_interval_sec=args.min_interval,
        max_interval_sec=args.max_interval,
        adaptive=not args.no_adaptive,
        auto_sync=not args.no_sync,
        auto_alert=not args.no_alert,
    )

    if args.channel:
        daemon.add_target("channel", args.channel, f"Channel: {args.channel}")
    if args.video:
        daemon.add_target("video", args.video, f"Video: {args.video}")

    if args.once:
        res = daemon.poll_once(mock=args.mock)
        print(f"\n✅ Single poll cycle #{res.cycle_index} finished. Found +{res.new_comments_found} comments. Next poll in {res.next_poll_in_sec}s.")
        return

    daemon.run_loop(max_iterations=args.iterations, mock=args.mock)


if __name__ == "__main__":
    main()
