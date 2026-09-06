"""Unit tests for Real-Time Chronological Event Replay & Crisis Simulation Engine (ytint-replay)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from engine.event_replay import (
    EventReplayEngine,
    FlashpointAlert,
    ReplayChronicle,
    ReplayFrame,
    main,
)


def test_dataclasses_serialization() -> None:
    alert = FlashpointAlert(
        timestamp="T+5.0h",
        minute_offset=300.0,
        alert_type="VELOCITY_SURGE",
        severity="WARNING",
        title="Arrival Spike",
        description="Spike detected",
        trigger_value=120.5,
        threshold=50.0,
    )
    assert alert.to_dict()["alert_type"] == "VELOCITY_SURGE"

    frame = ReplayFrame(
        step_index=0,
        timestamp="T+0.5h",
        minute_offset=30.0,
        hours_elapsed=0.5,
        frame_new_comments=15,
        cumulative_comments=15,
        arrival_velocity=30.0,
        velocity_accel=30.0,
        frame_sentiment=0.45,
        frame_toxicity=0.03,
        rolling_sentiment=0.45,
        rolling_toxicity=0.03,
        active_authors_count=12,
        reply_ratio=0.1,
        flame_war_risk_score=12.0,
        active_flashpoints=[alert],
        top_comments=[{"author": "Alice", "likes": 10, "sentiment": 0.5, "toxicity": 0.01, "text": "Hello"}],
    )
    frame_dict = frame.to_dict()
    assert frame_dict["cumulative_comments"] == 15
    assert len(frame_dict["active_flashpoints"]) == 1

    chronicle = ReplayChronicle(
        video_id="TEST_VID",
        video_title="Test Video Title",
        published_at="2026-09-06T12:00:00Z",
        total_comments_replayed=15,
        step_minutes=30,
        total_frames=1,
        frames=[frame],
        flashpoints_summary=[alert],
    )
    df = chronicle.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["arrival_velocity"] == 30.0
    assert df.iloc[0]["flashpoint_count"] == 1


def test_mock_chronicle_generation() -> None:
    engine = EventReplayEngine()
    chronicle = engine.generate_mock_chronicle(
        video_id="MOCK_TEST",
        step_minutes=30,
        max_hours=24.0,
    )
    assert chronicle.video_id == "MOCK_TEST"
    assert chronicle.total_frames == 48  # 24 hours / 0.5h
    assert len(chronicle.frames) == 48
    assert chronicle.total_comments_replayed > 0
    assert len(chronicle.flashpoints_summary) >= 1

    # Verify monotonic cumulative count
    cum_counts = [f.cumulative_comments for f in chronicle.frames]
    for i in range(1, len(cum_counts)):
        assert cum_counts[i] >= cum_counts[i - 1]


def test_export_chronicle_json_and_csv(tmp_path: Path) -> None:
    engine = EventReplayEngine()
    chronicle = engine.generate_mock_chronicle(video_id="MOCK_EXP", max_hours=6.0, step_minutes=60)

    # Test JSON export
    json_path = tmp_path / "replay.json"
    exported_json = chronicle.export(json_path, export_format="json")
    assert Path(exported_json).exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["video_id"] == "MOCK_EXP"
    assert len(data["frames"]) == 6

    # Test CSV export
    csv_path = tmp_path / "replay.csv"
    exported_csv = chronicle.export(csv_path, export_format="csv")
    assert Path(exported_csv).exists()
    df_read = pd.read_csv(csv_path)
    assert len(df_read) == 6
    assert "arrival_velocity" in df_read.columns
    assert "flame_war_risk_score" in df_read.columns


def test_empty_and_missing_files_resilience(tmp_path: Path) -> None:
    nonexistent_c = tmp_path / "missing_c.parquet"
    nonexistent_v = tmp_path / "missing_v.parquet"
    engine = EventReplayEngine(
        comments_path=str(nonexistent_c),
        videos_path=str(nonexistent_v),
    )
    # Available videos should return safe fallback mock
    videos = engine.get_available_videos()
    assert len(videos) >= 1
    assert videos[0]["video_id"] == "MOCK_VID_001"

    # Loading timeline should safely generate mock without raising exception
    chronicle = engine.load_video_timeline("UNKNOWN_VID", step_minutes=60, max_hours=12.0)
    assert isinstance(chronicle, ReplayChronicle)
    assert chronicle.total_frames == 12


def test_metrics_calculation_and_flashpoint_triggers(tmp_path: Path) -> None:
    # Build synthetic comments Parquet with known crisis surge
    comments = []
    # 20 comments in first hour (calm)
    for i in range(20):
        comments.append({
            "video_id": "TEST_CRISIS",
            "comment_id": f"c_{i}",
            "author_display_name": f"User_{i}",
            "published_at": pd.Timestamp("2026-09-01 10:00:00") + pd.Timedelta(minutes=i),
            "minutes_since_upload": float(i),
            "vader_compound": 0.4,
            "toxicity": 0.02,
            "like_count": 1,
            "is_reply": 0,
            "text": "Great video!",
        })
    # 150 hostile comments in hour 2 (surge + toxicity)
    for i in range(150):
        comments.append({
            "video_id": "TEST_CRISIS",
            "comment_id": f"c_crisis_{i}",
            "author_display_name": f"Troll_{i % 5}",
            "published_at": pd.Timestamp("2026-09-01 11:00:00") + pd.Timedelta(minutes=i // 3),
            "minutes_since_upload": 60.0 + float(i // 3),
            "vader_compound": -0.6,
            "toxicity": 0.45,
            "like_count": 5,
            "is_reply": 1,
            "text": "This is completely wrong and dishonest!",
        })

    df_synth = pd.DataFrame(comments)
    comments_file = tmp_path / "comments_synth.parquet"
    df_synth.to_parquet(comments_file, index=False)

    videos_file = tmp_path / "videos_synth.parquet"
    pd.DataFrame([{
        "video_id": "TEST_CRISIS",
        "title": "Crisis Test Video",
        "published_at": "2026-09-01 10:00:00",
        "total_comments": 170,
    }]).to_parquet(videos_file, index=False)

    engine = EventReplayEngine(
        comments_path=str(comments_file),
        videos_path=str(videos_file),
    )
    chronicle = engine.load_video_timeline("TEST_CRISIS", step_minutes=60, max_hours=4.0)
    assert chronicle.video_id == "TEST_CRISIS"
    assert chronicle.total_comments_replayed == 170
    assert len(chronicle.frames) >= 2

    frame_0 = chronicle.frames[0]
    assert frame_0.frame_new_comments == 20
    assert frame_0.arrival_velocity == 20.0
    assert frame_0.rolling_sentiment > 0.0

    frame_1 = chronicle.frames[1]
    assert frame_1.frame_new_comments == 150
    assert frame_1.arrival_velocity == 150.0
    assert frame_1.frame_sentiment < 0.0
    assert frame_1.frame_toxicity > 0.30
    assert frame_1.flame_war_risk_score > 50.0

    # Verify at least one crisis flashpoint was detected
    assert len(chronicle.flashpoints_summary) >= 1
    alert_types = [a.alert_type for a in chronicle.flashpoints_summary]
    assert any(t in alert_types for t in ["VELOCITY_SURGE", "TOXICITY_OUTBREAK", "SENTIMENT_CRASH", "FLAME_WAR_OUTBREAK"])


def test_stream_playback_with_callback() -> None:
    engine = EventReplayEngine()
    chronicle = engine.generate_mock_chronicle(max_hours=2.0, step_minutes=30)
    received_frames: List[ReplayFrame] = []

    def callback(frame: ReplayFrame) -> None:
        received_frames.append(frame)

    engine.stream_playback(chronicle, delay_seconds=0.0, callback=callback)
    assert len(received_frames) == chronicle.total_frames
    assert received_frames[0].step_index == 0


def test_cli_main_mock_and_export(tmp_path: Path) -> None:
    export_file = tmp_path / "cli_replay_export.json"
    code = main([
        "--mock",
        "--video-id", "CLI_TEST",
        "--step-mins", "30",
        "--max-hours", "4.0",
        "--speed", "0",
        "--export", str(export_file),
    ])
    assert code == 0
    assert export_file.exists()
    with open(export_file, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["video_id"] == "CLI_TEST"
    assert doc["total_frames"] == 8
