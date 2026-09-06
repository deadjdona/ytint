"""Tests for Automated Webhook Anomaly & Threat Alerting Engine (src/engine/alerting.py)."""

import json
from pathlib import Path
import pandas as pd
import pytest

from engine.alerting import AlertIncident, AlertManager, ThreatReport


@pytest.fixture
def mock_alert_env(tmp_path):
    """Creates a temporary workspace with synthetic analytical output artifacts."""
    output_dir = tmp_path / "output"
    interim_dir = tmp_path / "interim"
    output_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)

    # 1. Synthetic CIB rings
    df_cib = pd.DataFrame([
        {
            "ring_id": "RING_001",
            "ring_size": 25,
            "author_count": 25,
            "total_synchronized_events": 150,
            "avg_interval_seconds": 45.2,
            "member_channel_ids": "UC1,UC2,UC3"
        }
    ])
    df_cib.to_parquet(output_dir / "cib_rings.parquet")

    # 2. Synthetic Impersonation
    df_imp = pd.DataFrame([
        {
            "author_display_name": "OfficialCreator",
            "unique_accounts": 3,
            "total_comments": 12
        }
    ])
    df_imp.to_parquet(output_dir / "impersonation_detection.parquet")

    # 3. Synthetic Viral Spikes
    df_viral = pd.DataFrame([
        {
            "date": "2026-03-01",
            "comment_count": 5000,
            "z_score": 5.4
        }
    ])
    df_viral.to_parquet(output_dir / "viral_events.parquet")

    # 4. Synthetic Toxicity
    df_tox = pd.DataFrame([
        {
            "total_analyzed_comments": 10000,
            "total_toxic_comments": 2000,
            "toxic_comment_share_pct": 20.0,
            "avg_replies_to_toxic_root": 3.5,
            "avg_replies_to_neutral_root": 1.2,
            "toxicity_reproduction_number_r0": 1.8,
            "high_toxicity_threshold": 0.5
        }
    ])
    df_tox.to_parquet(output_dir / "toxicity_contagion_summary.parquet")

    # 5. Synthetic Troll Catalysts
    df_trolls = pd.DataFrame([
        {
            "author_channel_id": "UC_TROLL_1",
            "display_name": "BadActor99",
            "catalyst_score": 45.5,
            "total_replies_sparked": 180,
            "total_comments": 20,
            "toxic_comments_count": 15
        }
    ])
    df_trolls.to_parquet(output_dir / "troll_catalysts.parquet")

    # 6. Synthetic Like Inflation
    df_inf = pd.DataFrame([
        {"comment_id": f"c_{i}", "is_suspicious": True, "inflation_ratio": 25.0}
        for i in range(10)
    ])
    df_inf.to_parquet(output_dir / "like_inflation.parquet")

    # 7. Synthetic Topic Injection
    df_inj = pd.DataFrame([
        {
            "video_id": "vid_promo",
            "js_divergence": 0.42,
            "window_comments": 200,
            "dominant_topic": "crypto_scam"
        }
    ])
    df_inj.to_parquet(output_dir / "topic_injection_anomalies.parquet")

    return output_dir, interim_dir


def test_alert_manager_init(tmp_path):
    """Verifies AlertManager initializes directories properly."""
    manager = AlertManager(output_dir=tmp_path / "out", interim_dir=tmp_path / "int")
    assert manager.alerts_dir.exists()
    assert manager.alerts_dir == tmp_path / "out" / "alerts"


def test_threat_scan_detection(mock_alert_env):
    """Verifies that all 7 synthetic threat layers are correctly scanned and categorized."""
    output_dir, interim_dir = mock_alert_env
    manager = AlertManager(output_dir=output_dir, interim_dir=interim_dir)

    report = manager.scan_threats(severity_threshold="INFO")
    assert isinstance(report, ThreatReport)
    assert report.total_incidents >= 7
    assert report.critical_count >= 2  # CIB + Impersonation
    assert report.warning_count >= 4   # Viral, Toxicity, Trolls, Like Inflation
    assert report.info_count >= 1      # Topic injection
    assert report.max_severity == "CRITICAL"

    # Check incident structure
    rule_names = {inc.rule_name for inc in report.incidents}
    assert "cib_rings" in rule_names
    assert "impersonation" in rule_names
    assert "viral_spikes" in rule_names
    assert "toxicity_outbreak" in rule_names
    assert "troll_catalysts" in rule_names
    assert "like_inflation" in rule_names
    assert "topic_injection" in rule_names


def test_severity_filtering(mock_alert_env):
    """Verifies that setting a higher threshold filters lower-severity incidents."""
    output_dir, interim_dir = mock_alert_env
    manager = AlertManager(output_dir=output_dir, interim_dir=interim_dir)

    # Threshold CRITICAL only
    report_crit = manager.scan_threats(severity_threshold="CRITICAL")
    assert report_crit.max_severity == "CRITICAL"
    assert report_crit.warning_count == 0
    assert report_crit.info_count == 0
    assert report_crit.total_incidents == report_crit.critical_count


def test_discord_payload_formatting(mock_alert_env):
    """Verifies Discord embed formatting and color coding."""
    output_dir, interim_dir = mock_alert_env
    manager = AlertManager(output_dir=output_dir, interim_dir=interim_dir)
    report = manager.scan_threats()

    payload = manager.format_discord_payload(report)
    assert "embeds" in payload
    assert len(payload["embeds"]) == 1
    embed = payload["embeds"][0]
    assert embed["color"] == 0xFF0055  # Critical red
    assert "fields" in embed
    assert len(embed["fields"]) >= 2
    assert "timestamp" in embed


def test_slack_payload_formatting(mock_alert_env):
    """Verifies Slack Block Kit payload structure."""
    output_dir, interim_dir = mock_alert_env
    manager = AlertManager(output_dir=output_dir, interim_dir=interim_dir)
    report = manager.scan_threats()

    payload = manager.format_slack_payload(report)
    assert "blocks" in payload
    assert len(payload["blocks"]) >= 4
    block_types = [b["type"] for b in payload["blocks"]]
    assert "header" in block_types
    assert "divider" in block_types
    assert "section" in block_types


def test_telegram_payload_formatting(mock_alert_env):
    """Verifies Telegram HTML formatted payload."""
    output_dir, interim_dir = mock_alert_env
    manager = AlertManager(output_dir=output_dir, interim_dir=interim_dir)
    report = manager.scan_threats()

    payload = manager.format_telegram_payload(report, chat_id="123456789")
    assert payload["chat_id"] == "123456789"
    assert payload["parse_mode"] == "HTML"
    assert "<b>🛡️ ytint Community Threat Intelligence</b>" in payload["text"]
    assert "CRITICAL" in payload["text"]


def test_detect_webhook_type():
    """Verifies URL-to-provider detection heuristics."""
    assert AlertManager.detect_webhook_type("https://discord.com/api/webhooks/123/abc") == "discord"
    assert AlertManager.detect_webhook_type("https://discordapp.com/api/webhooks/123/abc") == "discord"
    assert AlertManager.detect_webhook_type("https://hooks.slack.com/services/T00/B00/X00") == "slack"
    assert AlertManager.detect_webhook_type("https://api.telegram.org/bot123:abc/sendMessage") == "telegram"
    assert AlertManager.detect_webhook_type("https://example.com/webhook") == "generic"
    assert AlertManager.detect_webhook_type("") == "generic"


def test_dry_run_dispatch(mock_alert_env):
    """Verifies dry-run execution serializes JSON digest without network calls."""
    output_dir, interim_dir = mock_alert_env
    manager = AlertManager(output_dir=output_dir, interim_dir=interim_dir)
    report = manager.scan_threats()

    result = manager.dispatch(report, webhook_url="https://discord.com/api/webhooks/fake/123", dry_run=True)
    assert result["success"] is True
    assert result["mode"] == "DRY_RUN"
    assert result["platform"] == "discord"

    saved_file = Path(result["saved_to"])
    assert saved_file.exists()

    with open(saved_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "threat_report" in data
    assert "rendered_payload" in data
    assert data["threat_report"]["total_incidents"] == report.total_incidents


def test_empty_corpus_resilience(tmp_path):
    """Verifies scanner does not crash when artifacts directory is empty."""
    empty_out = tmp_path / "empty_out"
    empty_int = tmp_path / "empty_int"
    empty_out.mkdir()
    empty_int.mkdir()

    manager = AlertManager(output_dir=empty_out, interim_dir=empty_int)
    report = manager.scan_threats()

    assert report.total_incidents == 0
    assert report.critical_count == 0
    assert report.warning_count == 0
    assert report.max_severity == "CLEAN"

    # Formatting still works on clean report
    discord_payload = manager.format_discord_payload(report)
    assert discord_payload["embeds"][0]["color"] == 0x00E599  # Clean emerald

    slack_payload = manager.format_slack_payload(report)
    assert "CLEAN" in slack_payload["blocks"][0]["text"]["text"]
