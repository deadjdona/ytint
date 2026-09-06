"""Real-Time Chronological Event Replay & Crisis Simulation Engine (ytint-replay).

Replays and simulates YouTube comment cascades in discrete chronological frames,
tracking instantaneous arrival velocity, rolling sentiment, toxicity outbreaks,
flame-war escalation, and automated crisis flashpoint alerts.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ytint_replay")

AlertType = Literal[
    "VELOCITY_SURGE",
    "TOXICITY_OUTBREAK",
    "SENTIMENT_CRASH",
    "FLAME_WAR_OUTBREAK",
    "VIRAL_CASCADE",
]


@dataclass
class FlashpointAlert:
    """Represents a discrete community crisis or anomaly trigger at a point in time."""

    timestamp: str
    minute_offset: float
    alert_type: AlertType
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    title: str
    description: str
    trigger_value: float
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayFrame:
    """A discrete temporal snapshot of the community state at simulation time t."""

    step_index: int
    timestamp: str
    minute_offset: float
    hours_elapsed: float
    frame_new_comments: int
    cumulative_comments: int
    arrival_velocity: float  # comments / hour
    velocity_accel: float  # change in arrival_velocity from previous frame
    frame_sentiment: float  # instantaneous sentiment in this frame
    frame_toxicity: float  # instantaneous toxicity in this frame
    rolling_sentiment: float  # -1.0 to +1.0 (smoothed EWMA)
    rolling_toxicity: float  # 0.0 to 1.0 (smoothed EWMA)
    active_authors_count: int
    reply_ratio: float  # proportion of comments that are replies
    flame_war_risk_score: float  # 0.0 to 100.0
    active_flashpoints: List[FlashpointAlert] = field(default_factory=list)
    top_comments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["active_flashpoints"] = [fp.to_dict() if isinstance(fp, FlashpointAlert) else fp for fp in self.active_flashpoints]
        return d


@dataclass
class ReplayChronicle:
    """Complete chronological event record of a video or channel comment sequence."""

    video_id: str
    video_title: str
    published_at: str
    total_comments_replayed: int
    step_minutes: int
    total_frames: int
    frames: List[ReplayFrame] = field(default_factory=list)
    flashpoints_summary: List[FlashpointAlert] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts frame sequence to a tabular DataFrame for plotting or analysis."""
        rows = []
        for f in self.frames:
            rows.append({
                "step_index": f.step_index,
                "timestamp": f.timestamp,
                "minute_offset": f.minute_offset,
                "hours_elapsed": f.hours_elapsed,
                "frame_new_comments": f.frame_new_comments,
                "cumulative_comments": f.cumulative_comments,
                "arrival_velocity": f.arrival_velocity,
                "velocity_accel": f.velocity_accel,
                "frame_sentiment": f.frame_sentiment,
                "frame_toxicity": f.frame_toxicity,
                "rolling_sentiment": f.rolling_sentiment,
                "rolling_toxicity": f.rolling_toxicity,
                "active_authors_count": f.active_authors_count,
                "reply_ratio": f.reply_ratio,
                "flame_war_risk_score": f.flame_war_risk_score,
                "flashpoint_count": len(f.active_flashpoints),
                "flashpoint_titles": "; ".join([fp.title for fp in f.active_flashpoints]),
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "published_at": self.published_at,
            "total_comments_replayed": self.total_comments_replayed,
            "step_minutes": self.step_minutes,
            "total_frames": self.total_frames,
            "frames": [f.to_dict() for f in self.frames],
            "flashpoints_summary": [fp.to_dict() for fp in self.flashpoints_summary],
        }

    def export(self, path: Union[str, Path], export_format: str = "json") -> str:
        """Exports the chronicle to a JSON or CSV file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if export_format.lower() == "csv":
            df = self.to_dataframe()
            df.to_csv(out_path, index=False, encoding="utf-8")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Exported replay chronicle ({len(self.frames)} frames) to {out_path}")
        return str(out_path)


class EventReplayEngine:
    """Chronological comment replay and crisis simulation engine."""

    def __init__(
        self,
        comments_path: str = "data/interim/comments_clean.parquet",
        videos_path: str = "data/interim/videos_clean.parquet",
    ) -> None:
        self.comments_path = Path(comments_path)
        self.videos_path = Path(videos_path)

    def get_available_videos(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns top videos sorted by comment count with titles."""
        if not self.videos_path.exists():
            return [
                {"video_id": "MOCK_VID_001", "title": "Synthesized Demo Video (Mock)", "total_comments": 1500}
            ]
        try:
            df_videos = pd.read_parquet(
                self.videos_path,
                columns=["video_id", "title", "total_comments", "published_at"],
            )
            df_videos = df_videos.sort_values("total_comments", ascending=False).head(limit)
            return df_videos.to_dict(orient="records")
        except Exception as e:
            logger.warning(f"Could not load videos catalog: {e}")
            return []

    def load_video_timeline(
        self,
        video_id: Optional[str] = None,
        step_minutes: int = 30,
        max_hours: Optional[float] = None,
        mock: bool = False,
    ) -> ReplayChronicle:
        """Generates a structured replay chronicle for a video."""
        if mock or not self.comments_path.exists():
            return self.generate_mock_chronicle(
                video_id=video_id or "MOCK_VID_001",
                step_minutes=step_minutes,
                max_hours=max_hours or 48.0,
            )

        # If no video_id specified, pick the most active video
        if not video_id:
            top_videos = self.get_available_videos(limit=1)
            video_id = top_videos[0]["video_id"] if top_videos else "UNKNOWN"

        # Resolve video title & published_at
        video_title = f"Video {video_id}"
        video_pub_time = datetime.now(timezone.utc).isoformat()
        if self.videos_path.exists():
            try:
                df_v = pd.read_parquet(
                    self.videos_path,
                    columns=["video_id", "title", "published_at"],
                    filters=[[("video_id", "==", video_id)]],
                )
                if not df_v.empty:
                    video_title = str(df_v.iloc[0].get("title", video_title))
                    pub_val = df_v.iloc[0].get("published_at")
                    if pd.notna(pub_val):
                        video_pub_time = str(pub_val)
            except Exception as e:
                logger.debug(f"Could not resolve video metadata for {video_id}: {e}")

        # Load video comments
        cols = [
            "video_id",
            "comment_id",
            "author_display_name",
            "published_at",
            "minutes_since_upload",
            "vader_compound",
            "toxicity",
            "like_count",
            "is_reply",
            "text",
        ]
        try:
            df_c = pd.read_parquet(
                self.comments_path,
                columns=cols,
                filters=[[("video_id", "==", video_id)]],
            )
        except Exception as e:
            logger.warning(f"Failed to read comments for {video_id}: {e}. Generating mock chronicle.")
            return self.generate_mock_chronicle(video_id=video_id, step_minutes=step_minutes)

        if df_c.empty:
            logger.warning(f"No comments found for video_id {video_id}.")
            return self.generate_mock_chronicle(video_id=video_id, step_minutes=step_minutes)

        # Ensure minutes_since_upload is numeric and positive
        if "minutes_since_upload" not in df_c.columns or df_c["minutes_since_upload"].isna().all():
            if "published_at" in df_c.columns and pd.api.types.is_datetime64_any_dtype(df_c["published_at"]):
                t_min = df_c["published_at"].min()
                df_c["minutes_since_upload"] = (df_c["published_at"] - t_min).dt.total_seconds() / 60.0
            else:
                df_c["minutes_since_upload"] = np.linspace(0, 1440, len(df_c))
        else:
            df_c["minutes_since_upload"] = pd.to_numeric(df_c["minutes_since_upload"], errors="coerce").fillna(0.0)
            # Clip negative offsets to 0.0
            df_c["minutes_since_upload"] = df_c["minutes_since_upload"].clip(lower=0.0)

        # Sort comments strictly chronologically
        df_c = df_c.sort_values("minutes_since_upload").reset_index(drop=True)

        if max_hours is not None:
            max_mins = max_hours * 60.0
            df_c = df_c[df_c["minutes_since_upload"] <= max_mins]

        total_mins = float(df_c["minutes_since_upload"].max()) if not df_c.empty else 0.0
        step_minutes = max(1, step_minutes)
        num_steps = max(1, int(math.ceil(total_mins / step_minutes)))

        frames: List[ReplayFrame] = []
        all_flashpoints: List[FlashpointAlert] = []

        cumulative_comments = 0
        prev_velocity = 0.0
        rolling_sentiment_ewma = 0.0
        rolling_toxicity_ewma = 0.0
        baseline_velocity_history: List[float] = []

        # Process each discrete time bucket
        for step_idx in range(num_steps):
            window_start = step_idx * step_minutes
            window_end = (step_idx + 1) * step_minutes
            hours_elapsed = round(window_end / 60.0, 2)

            sub = df_c[
                (df_c["minutes_since_upload"] >= window_start)
                & (df_c["minutes_since_upload"] < window_end)
            ]
            frame_new = len(sub)
            cumulative_comments += frame_new

            # Arrival velocity in comments per hour
            velocity = (frame_new / step_minutes) * 60.0
            velocity_accel = velocity - prev_velocity
            prev_velocity = velocity

            # Metrics
            if frame_new > 0:
                frame_sentiment = float(sub["vader_compound"].mean()) if "vader_compound" in sub.columns else 0.0
                frame_toxicity = float(sub["toxicity"].mean()) if "toxicity" in sub.columns else 0.0
                authors_count = int(sub["author_display_name"].nunique()) if "author_display_name" in sub.columns else frame_new
                reply_ratio = float((sub["is_reply"] == 1).mean()) if "is_reply" in sub.columns else 0.0

                # Adaptive volume-weighted EWMA smoothing
                alpha = max(0.35, min(0.85, frame_new / max(1.0, (frame_new + 20.0))))
                if step_idx == 0:
                    rolling_sentiment_ewma = frame_sentiment
                    rolling_toxicity_ewma = frame_toxicity
                else:
                    rolling_sentiment_ewma = alpha * frame_sentiment + (1 - alpha) * rolling_sentiment_ewma
                    rolling_toxicity_ewma = alpha * frame_toxicity + (1 - alpha) * rolling_toxicity_ewma
            else:
                frame_sentiment = rolling_sentiment_ewma
                frame_toxicity = rolling_toxicity_ewma
                authors_count = 0
                reply_ratio = 0.0

            # Composite Flame-War Risk Score (0 - 100%)
            # Higher when toxicity is high, sentiment is negative, reply ratio is high, and velocity is surging
            effective_tox = max(rolling_toxicity_ewma, frame_toxicity if frame_new >= 3 else 0.0)
            effective_sent = min(rolling_sentiment_ewma, frame_sentiment if frame_new >= 3 else 0.0)
            tox_factor = min(1.0, effective_tox / 0.35) * 40.0
            neg_sent_factor = max(0.0, -effective_sent) * 25.0
            reply_factor = reply_ratio * 20.0
            accel_factor = min(15.0, max(0.0, velocity_accel / 5.0))
            flame_war_risk = round(float(np.clip(tox_factor + neg_sent_factor + reply_factor + accel_factor, 0.0, 100.0)), 1)

            # Flashpoint detection
            frame_flashpoints: List[FlashpointAlert] = []
            frame_ts_str = f"T+{hours_elapsed:.1f}h"

            # 1. Velocity surge
            baseline_vel = np.mean(baseline_velocity_history[-5:]) if len(baseline_velocity_history) >= 2 else 5.0
            if velocity >= 3.0 * baseline_vel and velocity >= 25.0:
                alert = FlashpointAlert(
                    timestamp=frame_ts_str,
                    minute_offset=window_end,
                    alert_type="VELOCITY_SURGE",
                    severity="WARNING" if velocity < 100.0 else "CRITICAL",
                    title="Sudden Comment Influx Spike",
                    description=f"Comment arrival surged to {velocity:.1f} comments/hr ({velocity / max(1.0, baseline_vel):.1f}x baseline).",
                    trigger_value=round(velocity, 1),
                    threshold=round(float(3.0 * baseline_vel), 1),
                )
                frame_flashpoints.append(alert)
                all_flashpoints.append(alert)

            # 2. Toxicity Outbreak
            trigger_tox = max(rolling_toxicity_ewma, frame_toxicity)
            if trigger_tox >= 0.22 and frame_new >= 3:
                alert = FlashpointAlert(
                    timestamp=frame_ts_str,
                    minute_offset=window_end,
                    alert_type="TOXICITY_OUTBREAK",
                    severity="CRITICAL" if trigger_tox >= 0.35 else "WARNING",
                    title="Hostility & Toxicity Outbreak",
                    description=f"Toxicity reached {trigger_tox:.3f} across {frame_new} comments in window.",
                    trigger_value=round(trigger_tox, 3),
                    threshold=0.22,
                )
                frame_flashpoints.append(alert)
                all_flashpoints.append(alert)

            # 3. Sentiment Crash
            trigger_sent = min(rolling_sentiment_ewma, frame_sentiment)
            if trigger_sent <= -0.30 and frame_new >= 3:
                alert = FlashpointAlert(
                    timestamp=frame_ts_str,
                    minute_offset=window_end,
                    alert_type="SENTIMENT_CRASH",
                    severity="WARNING",
                    title="Severe Negative Sentiment Dive",
                    description=f"Community sentiment dropped sharply to {trigger_sent:.2f}.",
                    trigger_value=round(trigger_sent, 2),
                    threshold=-0.30,
                )
                frame_flashpoints.append(alert)
                all_flashpoints.append(alert)

            # 4. Flame-War Outbreak
            if flame_war_risk >= 65.0 and frame_new >= 4:
                alert = FlashpointAlert(
                    timestamp=frame_ts_str,
                    minute_offset=window_end,
                    alert_type="FLAME_WAR_OUTBREAK",
                    severity="CRITICAL",
                    title="Contagious Flame-War Escalation",
                    description=f"Flame-war escalation index reached {flame_war_risk}% with high reply intensity ({reply_ratio:.0%}).",
                    trigger_value=flame_war_risk,
                    threshold=65.0,
                )
                frame_flashpoints.append(alert)
                all_flashpoints.append(alert)

            baseline_velocity_history.append(velocity)

            # Extract top exemplar comments in window
            top_comments = []
            if frame_new > 0 and "like_count" in sub.columns:
                exemplars = sub.sort_values("like_count", ascending=False).head(3)
                for _, r in exemplars.iterrows():
                    top_comments.append({
                        "author": str(r.get("author_display_name", "Anonymous")),
                        "likes": int(r.get("like_count", 0)),
                        "sentiment": round(float(r.get("vader_compound", 0.0)), 2),
                        "toxicity": round(float(r.get("toxicity", 0.0)), 3),
                        "text": str(r.get("text", "")).strip()[:180],
                    })

            frame = ReplayFrame(
                step_index=step_idx,
                timestamp=frame_ts_str,
                minute_offset=window_end,
                hours_elapsed=hours_elapsed,
                frame_new_comments=frame_new,
                cumulative_comments=cumulative_comments,
                arrival_velocity=round(velocity, 2),
                velocity_accel=round(velocity_accel, 2),
                frame_sentiment=round(frame_sentiment, 3),
                frame_toxicity=round(frame_toxicity, 3),
                rolling_sentiment=round(rolling_sentiment_ewma, 3),
                rolling_toxicity=round(rolling_toxicity_ewma, 3),
                active_authors_count=authors_count,
                reply_ratio=round(reply_ratio, 3),
                flame_war_risk_score=flame_war_risk,
                active_flashpoints=frame_flashpoints,
                top_comments=top_comments,
            )
            frames.append(frame)

        return ReplayChronicle(
            video_id=video_id,
            video_title=video_title,
            published_at=video_pub_time,
            total_comments_replayed=cumulative_comments,
            step_minutes=step_minutes,
            total_frames=len(frames),
            frames=frames,
            flashpoints_summary=all_flashpoints,
        )

    def generate_mock_chronicle(
        self,
        video_id: str = "MOCK_VID_001",
        step_minutes: int = 30,
        max_hours: float = 48.0,
    ) -> ReplayChronicle:
        """Synthesizes a realistic crisis simulation chronicle for testing and demonstrations."""
        num_steps = max(1, int(round((max_hours * 60.0) / step_minutes)))
        frames: List[ReplayFrame] = []
        all_flashpoints: List[FlashpointAlert] = []

        cum_comments = 0
        prev_vel = 0.0
        rolling_sent = 0.25
        rolling_tox = 0.05

        for i in range(num_steps):
            mins = (i + 1) * step_minutes
            hrs = round(mins / 60.0, 2)
            ts_str = f"T+{hrs:.1f}h"

            # Model a realistic viral outbreak curve with a controversy dip
            # Hours 0-4: steady growth
            # Hours 4-10: viral spike
            # Hours 10-18: controversy & flame-war
            # Hours 18+: steady decay
            if hrs < 4:
                base_arrival = np.random.randint(15, 30)
                sent = 0.35 + np.random.uniform(-0.05, 0.05)
                tox = 0.04 + np.random.uniform(0.0, 0.02)
                replies = 0.15
            elif 4 <= hrs < 10:
                # Viral Surge
                base_arrival = np.random.randint(90, 180)
                sent = 0.20 + np.random.uniform(-0.1, 0.1)
                tox = 0.08 + np.random.uniform(0.0, 0.04)
                replies = 0.30
            elif 10 <= hrs < 18:
                # Controversy & Flame War
                base_arrival = np.random.randint(70, 140)
                sent = -0.42 + np.random.uniform(-0.1, 0.1)
                tox = 0.34 + np.random.uniform(0.0, 0.08)
                replies = 0.65
            else:
                # Post-crisis cooldown
                decay_factor = math.exp(-(hrs - 18) / 12.0)
                base_arrival = max(5, int(50 * decay_factor + np.random.randint(0, 10)))
                sent = 0.10 + np.random.uniform(-0.05, 0.05)
                tox = 0.07 + np.random.uniform(0.0, 0.02)
                replies = 0.20

            cum_comments += base_arrival
            vel = (base_arrival / step_minutes) * 60.0
            accel = vel - prev_vel
            prev_vel = vel

            # Smooth metrics
            rolling_sent = 0.35 * sent + 0.65 * rolling_sent
            rolling_tox = 0.35 * tox + 0.65 * rolling_tox

            flame_war_risk = round(
                float(np.clip(
                    (rolling_tox / 0.35) * 45.0 + max(0.0, -rolling_sent) * 30.0 + replies * 20.0,
                    0.0,
                    100.0,
                )),
                1,
            )

            frame_flashpoints: List[FlashpointAlert] = []
            if hrs == 5.0:
                alert = FlashpointAlert(
                    timestamp=ts_str,
                    minute_offset=mins,
                    alert_type="VELOCITY_SURGE",
                    severity="WARNING",
                    title="Viral Comment Velocity Surge",
                    description=f"Arrival velocity spiked to {vel:.1f} comments/hr as video hit algorithmic recommendations.",
                    trigger_value=round(vel, 1),
                    threshold=50.0,
                )
                frame_flashpoints.append(alert)
                all_flashpoints.append(alert)
            elif hrs == 11.0:
                alert = FlashpointAlert(
                    timestamp=ts_str,
                    minute_offset=mins,
                    alert_type="SENTIMENT_CRASH",
                    severity="WARNING",
                    title="Sentiment Crash Following Hot Take",
                    description="Rolling sentiment dropped below -0.30 following debate trigger at minute 8:12.",
                    trigger_value=round(rolling_sent, 2),
                    threshold=-0.30,
                )
                frame_flashpoints.append(alert)
                all_flashpoints.append(alert)
            elif hrs == 12.0:
                alert = FlashpointAlert(
                    timestamp=ts_str,
                    minute_offset=mins,
                    alert_type="TOXICITY_OUTBREAK",
                    severity="CRITICAL",
                    title="Toxic Flame-War Cascade Triggered",
                    description=f"Hostile exchanges exceeded 0.30 toxicity threshold (risk: {flame_war_risk}%).",
                    trigger_value=round(rolling_tox, 3),
                    threshold=0.25,
                )
                frame_flashpoints.append(alert)
                all_flashpoints.append(alert)

            top_comments = [
                {
                    "author": f"User_{np.random.randint(100, 999)}",
                    "likes": np.random.randint(10, 250),
                    "sentiment": round(rolling_sent, 2),
                    "toxicity": round(rolling_tox, 3),
                    "text": "This part of the video completely changes the discussion. We need to talk about this.",
                }
            ]

            frame = ReplayFrame(
                step_index=i,
                timestamp=ts_str,
                minute_offset=mins,
                hours_elapsed=hrs,
                frame_new_comments=base_arrival,
                cumulative_comments=cum_comments,
                arrival_velocity=round(vel, 2),
                velocity_accel=round(accel, 2),
                frame_sentiment=round(sent, 3),
                frame_toxicity=round(tox, 3),
                rolling_sentiment=round(rolling_sent, 3),
                rolling_toxicity=round(rolling_tox, 3),
                active_authors_count=int(base_arrival * 0.85),
                reply_ratio=round(replies, 3),
                flame_war_risk_score=flame_war_risk,
                active_flashpoints=frame_flashpoints,
                top_comments=top_comments,
            )
            frames.append(frame)

        return ReplayChronicle(
            video_id=video_id,
            video_title="Simulated Event Cascade & Controversy Replay (Demo)",
            published_at=datetime.now(timezone.utc).isoformat(),
            total_comments_replayed=cum_comments,
            step_minutes=step_minutes,
            total_frames=len(frames),
            frames=frames,
            flashpoints_summary=all_flashpoints,
        )

    def stream_playback(
        self,
        chronicle: ReplayChronicle,
        delay_seconds: float = 0.05,
        callback: Optional[Callable[[ReplayFrame], None]] = None,
    ) -> None:
        """Streams chronicle frames to terminal or a consumer callback."""
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

        print(f"\n[REPLAY] Starting Chronological Replay: '{chronicle.video_title}' ({chronicle.video_id})")
        print(f"   Resolution: {chronicle.step_minutes} mins/frame | Total Frames: {chronicle.total_frames}\n")
        print(f"{'STEP':<6} {'TIME':<8} {'ARRIVALS':<10} {'CUMULATIVE':<12} {'VELOCITY':<12} {'SENTIMENT':<10} {'TOXICITY':<10} {'RISK':<8} {'ALERTS'}")
        print("-" * 95)

        for frame in chronicle.frames:
            if callback:
                callback(frame)
            else:
                alert_str = f"[!] {frame.active_flashpoints[0].title}" if frame.active_flashpoints else ""
                sent_color = "+" if frame.rolling_sentiment >= 0 else ""
                print(
                    f"{frame.step_index:<6} {frame.timestamp:<8} {frame.frame_new_comments:<10} "
                    f"{frame.cumulative_comments:<12} {frame.arrival_velocity:<12.1f} "
                    f"{sent_color}{frame.rolling_sentiment:<9.2f} {frame.rolling_toxicity:<9.3f} "
                    f"{frame.flame_war_risk_score:<7.0f}% {alert_str}"
                )
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        print("\n[OK] Replay finished. Total comments replayed:", chronicle.total_comments_replayed)
        print(f"Total Flashpoints Detected: {len(chronicle.flashpoints_summary)}")


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for ytint-replay."""
    parser = argparse.ArgumentParser(
        prog="ytint-replay",
        description="Real-Time Chronological Event Replay & Crisis Simulation Engine for ytint.",
    )
    parser.add_argument("--video-id", type=str, default=None, help="Target YouTube video ID.")
    parser.add_argument("--step-mins", type=int, default=30, help="Temporal bucket resolution in minutes (default: 30).")
    parser.add_argument("--max-hours", type=float, default=None, help="Maximum post-upload hours to replay.")
    parser.add_argument("--speed", type=float, default=0.0, help="Playback delay in seconds per frame (0 for instant).")
    parser.add_argument("--export", type=str, default=None, help="File path to export chronicle (JSON or CSV).")
    parser.add_argument("--mock", action="store_true", help="Run simulation in synthetic mock mode.")
    parsed = parser.parse_args(args)

    engine = EventReplayEngine()
    chronicle = engine.load_video_timeline(
        video_id=parsed.video_id,
        step_minutes=parsed.step_mins,
        max_hours=parsed.max_hours,
        mock=parsed.mock,
    )

    if parsed.speed > 0 or parsed.export is None:
        engine.stream_playback(chronicle, delay_seconds=parsed.speed)

    if parsed.export:
        fmt = "csv" if parsed.export.endswith(".csv") else "json"
        chronicle.export(parsed.export, export_format=fmt)

    return 0


if __name__ == "__main__":
    sys.exit(main())
