"""Unit tests for Temporal Narrative Scene Reaction Forensics Engine (ytint-narrative)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engine.narrative import (
    ConfusionHotspot,
    NarrativeForensicsEngine,
    SceneReactionCluster,
    TimestampQuote,
    VideoNarrativeReport,
    classify_comment_reaction,
    extract_timestamps_from_text,
    format_seconds,
    main,
)


def test_format_seconds() -> None:
    """Verifies that integer seconds format properly into MM:SS and HH:MM:SS."""
    assert format_seconds(45) == "00:45"
    assert format_seconds(125) == "02:05"
    assert format_seconds(3665) == "01:01:05"
    assert format_seconds(0) == "00:00"


def test_timestamp_regex_extraction() -> None:
    """Verifies parsing of timestamps from plain text, @mentions, and YouTube URLs."""
    text1 = "Look at this part: 03:42 and then 12:15!"
    assert extract_timestamps_from_text(text1) == [222, 735]

    text2 = "Check out the intro at @0:30 and the finale at 1:05:20"
    assert extract_timestamps_from_text(text2) == [30, 3920]

    text3 = 'Timestamp link: <a href="https://www.youtube.com/watch?v=xyz&t=450s">7:30</a>'
    assert extract_timestamps_from_text(text3) == [450]

    assert extract_timestamps_from_text("No timestamps here") == []
    assert extract_timestamps_from_text("") == []


def test_classify_comment_reaction() -> None:
    """Verifies taxonomy classification across different emotional and stylistic cues."""
    assert classify_comment_reaction("ROFL this is hilarious 😂 lmao") == "humor_laughter"
    assert classify_comment_reaction("OMG what a plot twist!! 😱 wtf") == "shock_surprise"
    assert classify_comment_reaction("This scene made me cry so hard ❤️ wholesome") == "emotional_touching"
    assert classify_comment_reaction("Wait why did he do that? Doesn't make sense 🤔") == "critique_analytical"
    assert classify_comment_reaction("Song name in the intro? 🎵") == "chapter_navigation"
    assert classify_comment_reaction("Just a regular opinion about the video.") == "general_reaction"


def test_dataclasses_serialization() -> None:
    """Verifies serialization of report dataclasses."""
    quote = TimestampQuote(
        comment_id="c_1",
        author_name="@Tester",
        second=90,
        formatted_time="01:30",
        like_count=5,
        sentiment=0.5,
        toxicity=0.01,
        reaction_type="humor_laughter",
        text_snippet="Hilarious scene"
    )
    cluster = SceneReactionCluster(
        bin_index=0,
        start_sec=0,
        end_sec=30,
        formatted_time="00:00 - 00:30",
        comment_count=1,
        dominant_reaction="humor_laughter",
        reaction_label="😂 Humor",
        reaction_color="#00e599",
        mean_sentiment=0.5,
        mean_toxicity=0.01,
        is_confusion_hotspot=False,
        is_humor_peak=True,
        is_outrage_spark=False,
        reaction_counts={"humor_laughter": 1},
        quotes=[quote]
    )
    d = cluster.to_dict()
    assert d["bin_index"] == 0
    assert len(d["quotes"]) == 1

    report = VideoNarrativeReport(
        video_id="v_test",
        video_title="Test Video",
        total_timestamp_comments=1,
        total_scenes=1,
        video_duration_sec=30,
        dominant_reaction="humor_laughter",
        reaction_breakdown={"humor_laughter": 1},
        scenes=[cluster],
        confusion_hotspots=[],
        humor_peaks=[cluster]
    )
    df = report.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["dominant_reaction"] == "humor_laughter"


def test_mock_narrative_generation() -> None:
    """Verifies that the mock generator synthesizes a complete, valid narrative report."""
    engine = NarrativeForensicsEngine()
    report = engine.generate_mock_narrative(video_id="MOCK_TEST")
    assert report.video_id == "MOCK_TEST"
    assert report.total_scenes > 0
    assert report.total_timestamp_comments > 0
    assert len(report.confusion_hotspots) > 0
    assert len(report.humor_peaks) > 0

    d = report.to_dict()
    assert "scenes" in d
    assert "confusion_hotspots" in d


def test_narrative_engine_video_discovery() -> None:
    """Verifies that the engine discovers existing videos with timestamp mentions."""
    engine = NarrativeForensicsEngine()
    vids = engine.get_available_videos(limit=10)
    assert isinstance(vids, list)
    if vids:
        assert "video_id" in vids[0]
        assert "timestamp_comments" in vids[0]
        assert vids[0]["timestamp_comments"] >= vids[-1]["timestamp_comments"]


def test_analyze_video_narrative_real_or_fallback() -> None:
    """Tests narrative analysis on a discovered video or mock fallback."""
    engine = NarrativeForensicsEngine()
    vids = engine.get_available_videos(limit=1)
    vid_id = vids[0]["video_id"] if vids else "MOCK_VIDEO_001"
    
    report = engine.analyze_video_narrative(video_id=vid_id, bin_seconds=30)
    assert report.video_id == vid_id
    assert report.total_scenes > 0
    assert report.video_duration_sec > 0


def test_export_report(tmp_path: Path) -> None:
    """Verifies exporting narrative reports to CSV and JSON."""
    engine = NarrativeForensicsEngine()
    report = engine.generate_mock_narrative(video_id="MOCK_EXPORT")

    csv_file = tmp_path / "narrative.csv"
    ok_csv, msg_csv = engine.export_report(report, csv_file)
    assert ok_csv is True
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    json_file = tmp_path / "narrative.json"
    ok_json, msg_json = engine.export_report(report, json_file)
    assert ok_json is True
    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["video_id"] == "MOCK_EXPORT"


def test_cli_execution(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verifies that CLI commands execute cleanly."""
    # 1. List videos
    code = main(["--list-videos"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Discovered" in captured.out or "No video" in captured.out

    # 2. Mock mode execution
    code = main(["--mock", "--video-id", "TEST_VID"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Narrative Scene Reaction Forensics" in captured.out
    assert "Time Window" in captured.out

    # 3. Hotspots mode
    code = main(["--mock", "--hotspots"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Detected Viewer Confusion Hotspots" in captured.out

    # 4. Export via CLI
    exp_file = tmp_path / "cli_narrative.json"
    code = main(["--mock", "--export", str(exp_file)])
    assert code == 0
    assert exp_file.exists()
