"""ytint // Automated Webhook Anomaly & Threat Alerting Daemon (src/engine/alerting.py)

Scans the analytical artifacts produced across the 53-stage pipeline to detect:
1. Coordinated Inauthentic Behavior (CIB) astroturfing rings (Stage 25)
2. Creator Impersonation threats (Stage 23)
3. Viral Attention & Comment Volume spikes (Stage 28)
4. Extreme Negative Sentiment & Toxicity shocks (Stage 15 & 44)
5. Flame-War Instigator / Troll Catalysts (Stage 15)
6. Suspicious Like Inflation & Astroturfed Upvotes (Stage 27)
7. Unnatural Thematic Hijacking & Topic Injections (Stage 51)

Formats and dispatches structured threat notifications to:
- Discord Webhooks (rich embeds with color tiers and metric badges)
- Slack Webhooks (Block Kit markdown sections and metric grids)
- Telegram Bot API (formatted HTML messages)
- Generic JSON Webhooks / Local JSON Digests (offline mock and test mode)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Ensure src is in sys.path
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from engine.config_loader import get_paths, load_config

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

logger = logging.getLogger("ytint_alerting")


@dataclass
class AlertIncident:
    """Individual anomaly or forensic incident record."""
    rule_name: str
    title: str
    severity: str  # CRITICAL, WARNING, INFO
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    action_recommendation: str = ""


@dataclass
class ThreatReport:
    """Consolidated threat intelligence report generated from pipeline scans."""
    generated_at: str
    total_incidents: int
    critical_count: int
    warning_count: int
    info_count: int
    max_severity: str
    incidents: List[AlertIncident] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "total_incidents": self.total_incidents,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "max_severity": self.max_severity,
            "incidents": [asdict(inc) for inc in self.incidents]
        }


class AlertManager:
    """Orchestrates threat detection scans, multi-platform formatting, and webhook dispatching."""

    SEVERITY_ORDER = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        interim_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.cfg = config or load_config()
        _, interim, output = get_paths(self.cfg)
        self.output_dir = Path(output_dir or output)
        self.interim_dir = Path(interim_dir or interim)
        self.alerts_dir = self.output_dir / "alerts"
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        self.alert_cfg = self.cfg.get("alerting", {})
        self.rules_cfg = self.alert_cfg.get("rules", {})

    def _read_parquet_safe(self, filename: str) -> pd.DataFrame:
        """Loads a parquet table from output_dir or interim_dir gracefully."""
        p = self.output_dir / filename
        if not p.exists():
            p = self.interim_dir / filename
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception as e:
                logger.warning(f"Could not read {filename}: {e}")
        return pd.DataFrame()

    def scan_threats(self, severity_threshold: Optional[str] = None) -> ThreatReport:
        """Evaluates all forensic, epidemiological, and viral signals across pipeline artifacts."""
        incidents: List[AlertIncident] = []
        min_sev = severity_threshold or self.alert_cfg.get("severity_threshold", "WARNING")
        min_prio = self.SEVERITY_ORDER.get(min_sev.upper(), 1)

        # 1. Coordinated Inauthentic Behavior (CIB) Rings
        cib_cfg = self.rules_cfg.get("cib_rings", {})
        if cib_cfg.get("enabled", True):
            df_cib = self._read_parquet_safe("cib_rings.parquet")
            if not df_cib.empty and "ring_size" in df_cib.columns:
                min_size = cib_cfg.get("min_ring_size", 2)
                suspect_rings = df_cib[df_cib["ring_size"] >= min_size]
                if not suspect_rings.empty:
                    top_ring = suspect_rings.sort_values("ring_size", ascending=False).iloc[0]
                    total_rings = len(suspect_rings)
                    max_size = int(top_ring.get("ring_size", 0))
                    sync_events = int(top_ring.get("total_synchronized_events", 0))
                    incidents.append(AlertIncident(
                        rule_name="cib_rings",
                        title="🚨 Coordinated Inauthentic Behavior (CIB) Rings Detected",
                        severity=cib_cfg.get("severity", "CRITICAL"),
                        summary=f"Detected {total_rings:,} astroturfing ring(s). Largest ring contains {max_size:,} synchronized accounts with {sync_events:,} coordinated interactions (delta-t <= 120s).",
                        details={
                            "total_rings": total_rings,
                            "largest_ring_id": str(top_ring.get("ring_id", "RING_001")),
                            "largest_ring_size": max_size,
                            "synchronized_events": sync_events,
                            "avg_interval_seconds": float(round(top_ring.get("avg_interval_seconds", 0), 2))
                        },
                        action_recommendation="Inspect Tab 4 CIB Ring Severity Map and consider submitting member channel IDs for platform review."
                    ))

        # 2. Creator Impersonation Threats
        imp_cfg = self.rules_cfg.get("impersonation", {})
        if imp_cfg.get("enabled", True):
            df_imp = self._read_parquet_safe("impersonation_detection.parquet")
            if not df_imp.empty:
                count = len(df_imp)
                names = df_imp["author_display_name"].dropna().unique().tolist()[:5]
                incidents.append(AlertIncident(
                    rule_name="impersonation",
                    title="⚠️ Potential Creator Impersonation Accounts Flagged",
                    severity=imp_cfg.get("severity", "CRITICAL"),
                    summary=f"Identified {count} suspected clone or duplicate accounts imitating creator display names: {', '.join(names)}.",
                    details={
                        "impersonator_count": count,
                        "flagged_names": names
                    },
                    action_recommendation="Review author channels in Tab 4 Forensics to prevent viewer phishing or scam replies."
                ))

        # 3. Viral Volume Spikes
        viral_cfg = self.rules_cfg.get("viral_spikes", {})
        if viral_cfg.get("enabled", True):
            df_viral = self._read_parquet_safe("viral_events.parquet")
            if not df_viral.empty and "z_score" in df_viral.columns:
                cutoff = viral_cfg.get("min_z_score", 3.0)
                intense_spikes = df_viral[df_viral["z_score"] >= cutoff]
                if not intense_spikes.empty:
                    max_spike = intense_spikes.sort_values("z_score", ascending=False).iloc[0]
                    z_val = float(round(max_spike["z_score"], 2))
                    c_count = int(max_spike.get("comment_count", 0))
                    spike_date = str(max_spike.get("date", "Recent"))
                    incidents.append(AlertIncident(
                        rule_name="viral_spikes",
                        title="📈 Viral Engagement / Influx Spike Detected",
                        severity=viral_cfg.get("severity", "WARNING"),
                        summary=f"Comment influx anomaly peaked at {z_val}σ on {spike_date} with {c_count:,} comments arriving in a single period.",
                        details={
                            "peak_z_score": z_val,
                            "comment_count": c_count,
                            "spike_date": spike_date,
                            "total_spikes_over_threshold": len(intense_spikes)
                        },
                        action_recommendation="Check Tab 2 Temporal Dynamics to verify if this correlates with external media features or viral sharing."
                    ))

        # 4. Negative Sentiment Anomaly Spikes
        sent_cfg = self.rules_cfg.get("sentiment_negativity_spike", {})
        if sent_cfg.get("enabled", True):
            df_sent = self._read_parquet_safe("sentiment_anomalies.parquet")
            if not df_sent.empty and "z_score" in df_sent.columns:
                cutoff = sent_cfg.get("min_negative_z_score", 2.5)
                neg_anomalies = df_sent[df_sent["z_score"] >= cutoff]
                if not neg_anomalies.empty:
                    peak_neg = neg_anomalies.sort_values("z_score", ascending=False).iloc[0]
                    z_val = float(round(peak_neg["z_score"], 2))
                    incidents.append(AlertIncident(
                        rule_name="sentiment_negativity_spike",
                        title="🚨 Extreme Negative Sentiment Outbreak",
                        severity=sent_cfg.get("severity", "WARNING"),
                        summary=f"Detected severe community sentiment negativity drop ({z_val}σ deviation) across recent comment windows.",
                        details={
                            "z_score": z_val,
                            "affected_video_id": str(peak_neg.get("video_id", "Channel-wide")),
                            "mean_sentiment": float(round(peak_neg.get("mean_sentiment", 0.0), 3))
                        },
                        action_recommendation="Review Tab 2 and Tab 3 Flame-War Summaries to address viewer discontent."
                    ))

        # 5. Toxicity Contagion & Troll Catalysts
        tox_cfg = self.rules_cfg.get("toxicity_outbreak", {})
        if tox_cfg.get("enabled", True):
            df_tox = self._read_parquet_safe("toxicity_contagion_summary.parquet")
            if not df_tox.empty:
                r0 = float(df_tox.iloc[0].get("toxicity_reproduction_number_r0", 0.0))
                share = float(df_tox.iloc[0].get("toxic_comment_share_pct", 0.0))
                min_r0 = tox_cfg.get("min_r0", 1.0)
                min_share = tox_cfg.get("min_toxic_share", 15.0)  # pct
                if r0 >= min_r0 or share >= min_share:
                    incidents.append(AlertIncident(
                        rule_name="toxicity_outbreak",
                        title="🔥 Elevated Toxicity Contagion Level",
                        severity=tox_cfg.get("severity", "WARNING"),
                        summary=f"Community toxicity reproduction rate R0 is {r0:.2f} with toxic comment share at {share:.1f}%.",
                        details={
                            "toxicity_reproduction_r0": round(r0, 3),
                            "toxic_comment_share_pct": round(share, 2),
                            "total_toxic_comments": int(df_tox.iloc[0].get("total_toxic_comments", 0))
                        },
                        action_recommendation="Implement early pinning of positive comments and moderate top troll sparks."
                    ))

        # 6. Troll Sparks / Catalysts
        troll_cfg = self.rules_cfg.get("troll_catalysts", {})
        if troll_cfg.get("enabled", True):
            df_trolls = self._read_parquet_safe("troll_catalysts.parquet")
            if not df_trolls.empty and "catalyst_score" in df_trolls.columns:
                cutoff = troll_cfg.get("min_catalyst_score", 10.0)
                severe_trolls = df_trolls[df_trolls["catalyst_score"] >= cutoff]
                if not severe_trolls.empty:
                    top_troll = severe_trolls.sort_values("catalyst_score", ascending=False).iloc[0]
                    name = str(top_troll.get("display_name", top_troll.get("author_channel_id", "Unknown")))
                    score = float(round(top_troll.get("catalyst_score", 0.0), 1))
                    replies_sparked = int(top_troll.get("total_replies_sparked", 0))
                    incidents.append(AlertIncident(
                        rule_name="troll_catalysts",
                        title="⚡ Severe Flame-War Catalyst Instigator Flagged",
                        severity=troll_cfg.get("severity", "WARNING"),
                        summary=f"Top flame catalyst '{name}' triggered {replies_sparked:,} contentious replies (Impact Score: {score}).",
                        details={
                            "catalyst_author": name,
                            "catalyst_score": score,
                            "replies_sparked": replies_sparked,
                            "total_flagged_catalysts": len(severe_trolls)
                        },
                        action_recommendation="Review author comment history in Tab 4 Audience Forensics."
                    ))

        # 7. Suspicious Like Inflation
        inf_cfg = self.rules_cfg.get("like_inflation", {})
        if inf_cfg.get("enabled", True):
            df_inf = self._read_parquet_safe("like_inflation.parquet")
            if not df_inf.empty and "is_suspicious" in df_inf.columns:
                suspicious_count = int((df_inf["is_suspicious"] == True).sum())
                min_count = inf_cfg.get("min_suspicious_count", 5)
                if suspicious_count >= min_count:
                    incidents.append(AlertIncident(
                        rule_name="like_inflation",
                        title="🤖 Suspicious Upvote / Like Inflation Detected",
                        severity=inf_cfg.get("severity", "WARNING"),
                        summary=f"Detected {suspicious_count:,} comments with anomalous like-to-reply ratios indicating artificial engagement boosts.",
                        details={
                            "suspicious_comments_count": suspicious_count,
                            "sample_comment_id": str(df_inf[df_inf["is_suspicious"] == True].iloc[0].get("comment_id", ""))
                        },
                        action_recommendation="Examine like-to-reply distribution in Tab 4 to identify artificial upvote vendors."
                    ))

        # 8. Thematic Hijacking / Topic Injections
        topic_cfg = self.rules_cfg.get("topic_injection", {})
        if topic_cfg.get("enabled", True):
            df_inj = self._read_parquet_safe("topic_injection_anomalies.parquet")
            if not df_inj.empty and "js_divergence" in df_inj.columns:
                min_jsd = topic_cfg.get("min_jsd", 0.35)
                anomalies = df_inj[df_inj["js_divergence"] >= min_jsd]
                if not anomalies.empty:
                    top_inj = anomalies.sort_values("js_divergence", ascending=False).iloc[0]
                    jsd_val = float(round(top_inj["js_divergence"], 3))
                    vid_id = str(top_inj.get("video_id", "Unknown"))
                    incidents.append(AlertIncident(
                        rule_name="topic_injection",
                        title="🎯 Unnatural Thematic Shift / Topic Injection",
                        severity=topic_cfg.get("severity", "INFO"),
                        summary=f"Video '{vid_id}' experienced an unnatural topic distribution shift (JS Divergence: {jsd_val}).",
                        details={
                            "video_id": vid_id,
                            "js_divergence": jsd_val,
                            "window_comments": int(top_inj.get("window_comments", 0))
                        },
                        action_recommendation="Check Tab 3 Semantic Drift to inspect whether external brigading drove off-topic discourse."
                    ))

        # Filter incidents by severity threshold
        filtered_incidents = [
            inc for inc in incidents
            if self.SEVERITY_ORDER.get(inc.severity.upper(), 1) >= min_prio
        ]

        crit = sum(1 for inc in filtered_incidents if inc.severity.upper() == "CRITICAL")
        warn = sum(1 for inc in filtered_incidents if inc.severity.upper() == "WARNING")
        info = sum(1 for inc in filtered_incidents if inc.severity.upper() == "INFO")

        if crit > 0:
            max_sev = "CRITICAL"
        elif warn > 0:
            max_sev = "WARNING"
        elif info > 0:
            max_sev = "INFO"
        else:
            max_sev = "CLEAN"

        return ThreatReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_incidents=len(filtered_incidents),
            critical_count=crit,
            warning_count=warn,
            info_count=info,
            max_severity=max_sev,
            incidents=filtered_incidents
        )

    def format_discord_payload(self, report: ThreatReport) -> Dict[str, Any]:
        """Builds a rich Discord Embed notification payload."""
        color_map = {
            "CRITICAL": 0xFF0055,  # Crimson Red
            "WARNING": 0xFFA500,   # Amber Orange
            "INFO": 0x00F0FF,      # Cyber Cyan
            "CLEAN": 0x00E599      # Emerald Green
        }
        color = color_map.get(report.max_severity, 0x0066FE)

        fields = [
            {
                "name": "📊 Threat Summary",
                "value": f"**Status:** {report.max_severity}\n**Critical:** {report.critical_count} | **Warnings:** {report.warning_count} | **Info:** {report.info_count}",
                "inline": False
            }
        ]

        for inc in report.incidents[:6]:  # Discord embed limit
            sev_badge = "🚨" if inc.severity == "CRITICAL" else ("⚠️" if inc.severity == "WARNING" else "ℹ️")
            fields.append({
                "name": f"{sev_badge} {inc.title}",
                "value": f"{inc.summary}\n*Action:* `{inc.action_recommendation}`",
                "inline": False
            })

        embed = {
            "title": f"🛡️ ytint Community Threat Intelligence Report — {report.max_severity}",
            "description": f"Automated analytical audit scanned across 53 pipeline layers at `{report.generated_at[:19]}Z`.",
            "color": color,
            "fields": fields,
            "footer": {
                "text": "ytint Forensic Alerting Engine • High-Dimensional YouTube Intelligence"
            },
            "timestamp": report.generated_at
        }

        return {
            "username": "ytint Threat Radar",
            "avatar_url": "https://raw.githubusercontent.com/deadj/ytint/main/docs/assets/logo.png",
            "embeds": [embed]
        }

    def format_slack_payload(self, report: ThreatReport) -> Dict[str, Any]:
        """Builds a Slack Block Kit notification payload."""
        blocks: List[Dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🛡️ ytint Threat Intelligence Digest: {report.max_severity}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total Incidents:* {report.total_incidents}"},
                    {"type": "mrkdwn", "text": f"*Max Severity:* `{report.max_severity}`"},
                    {"type": "mrkdwn", "text": f"*Critical Threats:* {report.critical_count}"},
                    {"type": "mrkdwn", "text": f"*Warnings:* {report.warning_count}"}
                ]
            },
            {"type": "divider"}
        ]

        for inc in report.incidents[:8]:
            sev_badge = "🚨 *CRITICAL*" if inc.severity == "CRITICAL" else ("⚠️ *WARNING*" if inc.severity == "WARNING" else "ℹ️ *INFO*")
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{sev_badge}: *{inc.title}*\n{inc.summary}\n> 💡 *Action:* {inc.action_recommendation}"
                }
            })

        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"ytint Automated Threat Daemon • Scanned at {report.generated_at[:19]}Z"
                }
            ]
        })

        return {"blocks": blocks}

    def format_telegram_payload(self, report: ThreatReport, chat_id: Optional[str] = None) -> Dict[str, Any]:
        """Builds a Telegram HTML formatted message."""
        lines = [
            f"<b>🛡️ ytint Community Threat Intelligence</b>",
            f"<b>Status:</b> <code>{report.max_severity}</code>",
            f"<b>Incidents:</b> Critical: {report.critical_count} | Warnings: {report.warning_count} | Info: {report.info_count}",
            f"<i>Timestamp: {report.generated_at[:19]}Z</i>",
            "",
            "<b>Detected Anomalies:</b>"
        ]

        for inc in report.incidents[:6]:
            badge = "🚨" if inc.severity == "CRITICAL" else ("⚠️" if inc.severity == "WARNING" else "ℹ️")
            lines.append(f"{badge} <b>{inc.title}</b>")
            lines.append(f"{inc.summary}")
            lines.append(f"👉 <i>Action: {inc.action_recommendation}</i>\n")

        return {
            "chat_id": chat_id or self.alert_cfg.get("telegram_chat_id", ""),
            "text": "\n".join(lines),
            "parse_mode": "HTML"
        }

    @staticmethod
    def detect_webhook_type(url: str) -> str:
        """Heuristically infers webhook provider from URL."""
        if not url:
            return "generic"
        lower_url = url.lower()
        if "discord.com/api/webhooks" in lower_url or "discordapp.com/api/webhooks" in lower_url:
            return "discord"
        if "hooks.slack.com" in lower_url:
            return "slack"
        if "api.telegram.org" in lower_url:
            return "telegram"
        return "generic"

    def dispatch(
        self,
        report: ThreatReport,
        webhook_url: Optional[str] = None,
        webhook_type: str = "auto",
        chat_id: Optional[str] = None,
        dry_run: bool = False,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Dispatches payload to remote webhook or records dry-run digest."""
        target_url = webhook_url or os.getenv("YTINT_WEBHOOK_URL") or self.alert_cfg.get("webhook_url", "")
        resolved_type = webhook_type if webhook_type != "auto" else self.detect_webhook_type(target_url)

        # Build appropriate payload
        if resolved_type == "discord":
            payload = self.format_discord_payload(report)
        elif resolved_type == "slack":
            payload = self.format_slack_payload(report)
        elif resolved_type == "telegram":
            payload = self.format_telegram_payload(report, chat_id)
        else:
            payload = report.to_dict()

        # Always save latest alert digest locally for auditability
        default_save = save_path or (self.alerts_dir / "latest_alert.json")
        default_save.parent.mkdir(parents=True, exist_ok=True)
        with open(default_save, "w", encoding="utf-8") as f:
            json.dump({
                "threat_report": report.to_dict(),
                "rendered_payload": payload,
                "platform": resolved_type,
                "dispatched_at": datetime.now(timezone.utc).isoformat()
            }, f, indent=2, ensure_ascii=False)

        # If dry-run or no URL provided, complete without network I/O
        if dry_run or not target_url:
            logger.info(f"✅ Dry-run / mock alert digest serialized to: {default_save}")
            return {
                "success": True,
                "mode": "DRY_RUN",
                "platform": resolved_type,
                "saved_to": str(default_save),
                "total_incidents": report.total_incidents,
                "max_severity": report.max_severity,
                "payload": payload
            }

        # Remote HTTP Dispatch via urllib
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            target_url,
            data=data_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "ytint-alert-daemon/1.0"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=7) as resp:
                status_code = resp.getcode()
                logger.info(f"✅ Dispatched alert to {resolved_type.upper()} ({status_code})")
                return {
                    "success": True,
                    "mode": "DISPATCHED",
                    "platform": resolved_type,
                    "status_code": status_code,
                    "total_incidents": report.total_incidents,
                    "max_severity": report.max_severity
                }
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP Error during webhook dispatch: {e.code} - {e.reason}")
            return {
                "success": False,
                "mode": "FAILED",
                "error": f"HTTP {e.code}: {e.reason}",
                "status_code": e.code
            }
        except Exception as e:
            logger.error(f"Error during webhook dispatch: {e}")
            return {
                "success": False,
                "mode": "FAILED",
                "error": str(e)
            }


def main():
    parser = argparse.ArgumentParser(description="ytint Automated Webhook Threat & Anomaly Alerting Daemon")
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help="Target Webhook URL (Discord, Slack, Telegram, or generic JSON)"
    )
    parser.add_argument(
        "--webhook-type",
        choices=["auto", "discord", "slack", "telegram", "generic"],
        default="auto",
        help="Webhook payload format (default: auto-detected from URL)"
    )
    parser.add_argument(
        "--severity",
        choices=["INFO", "WARNING", "CRITICAL"],
        default="WARNING",
        help="Minimum severity threshold to report (default: WARNING)"
    )
    parser.add_argument(
        "--chat-id",
        type=str,
        default=None,
        help="Telegram Chat ID (if using Telegram webhook)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute detection scan and preview formatted payload without sending HTTP requests"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to pipeline output directory"
    )
    args = parser.parse_args()

    print("🛡️ Initializing ytint Automated Threat & Anomaly Daemon...")
    manager = AlertManager(output_dir=Path(args.output_dir) if args.output_dir else None)

    print(f"🔍 Scanning forensic and anomaly layers (Threshold >= {args.severity})...")
    report = manager.scan_threats(severity_threshold=args.severity)

    print(f"\n📊 Threat Scan Results:")
    print(f"   Max Severity: {report.max_severity}")
    print(f"   Total Incidents: {report.total_incidents}")
    print(f"   - Critical: {report.critical_count}")
    print(f"   - Warnings: {report.warning_count}")
    print(f"   - Info:     {report.info_count}")

    if report.incidents:
        print("\n🚨 Triggered Incidents:")
        for i, inc in enumerate(report.incidents, 1):
            print(f"   [{i}] {inc.severity} // {inc.title}")
            print(f"       Summary: {inc.summary}")
            print(f"       Action:  {inc.action_recommendation}")

    # Dispatch or dry-run
    res = manager.dispatch(
        report=report,
        webhook_url=args.webhook_url,
        webhook_type=args.webhook_type,
        chat_id=args.chat_id,
        dry_run=args.dry_run
    )

    if res.get("mode") == "DRY_RUN":
        print(f"\n📁 Dry-Run Digest saved to: {res.get('saved_to')}")
    elif res.get("success"):
        print(f"\n🚀 Alert successfully dispatched to {res.get('platform', '').upper()}!")
    else:
        print(f"\n⚠️ Webhook dispatch failed: {res.get('error')}")


if __name__ == "__main__":
    main()
