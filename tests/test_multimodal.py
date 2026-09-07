"""Unit and integration tests for Multimodal Transcript & Vision Alignment Engine (src/engine/multimodal.py)."""

import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from engine.multimodal import (
    TranscriptSegment,
    VisualKeyframe,
    AlignedSceneWindow,
    SynthesizedChapter,
    MultimodalReport,
    MultimodalEngine,
    format_seconds,
    parse_timestamp_str,
    extract_keywords,
    parse_vtt_or_srt,
    parse_whisper_json,
    generate_mock_transcript,
    generate_mock_keyframes,
    SCENE_TYPES,
    main as cli_main
)


def test_time_formatting_and_parsing():
    assert format_seconds(0) == "00:00"
    assert format_seconds(65) == "01:05"
    assert format_seconds(3665) == "01:01:05"
    
    assert parse_timestamp_str("01:25") == 85.0
    assert parse_timestamp_str("01:02:03") == 3723.0
    assert parse_timestamp_str("invalid") == 0.0


def test_keyword_extraction():
    text = "The quick brown fox jumps over the lazy dog and optimizes neural embeddings"
    kws = extract_keywords(text, top_n=3)
    assert len(kws) <= 3
    assert "quick" in kws or "brown" in kws or "fox" in kws
    assert "the" not in kws
    assert "and" not in kws


def test_vtt_and_srt_parsing():
    vtt_sample = """WEBVTT

00:00:01.000 --> 00:00:04.500
Hello and welcome to the deep dive on vector databases.

00:00:05.000 --> 00:00:08.200
Here is the core mathematical formulation.
"""
    segments = parse_vtt_or_srt(vtt_sample)
    assert len(segments) == 2
    assert segments[0].start == 1.0
    assert segments[0].duration == 3.5
    assert "vector databases" in segments[0].text
    assert segments[1].start == 5.0


def test_whisper_json_parsing():
    data = {
        "segments": [
            {"start": 0.0, "end": 4.0, "text": "Testing Whisper speech recognition.", "confidence": 0.98},
            {"start": 4.5, "end": 9.0, "text": "Notice the attention heads.", "confidence": 0.92}
        ]
    }
    segments = parse_whisper_json(data)
    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].duration == 4.0
    assert "speech recognition" in segments[0].text
    assert segments[1].confidence == 0.92


def test_mock_transcript_and_keyframe_generation():
    segments = generate_mock_transcript("vid1", "Test Title", 120, ["Embeddings"])
    assert len(segments) >= 3
    assert segments[0].start == 0.0
    assert all(s.duration > 0 for s in segments)
    
    keyframes = generate_mock_keyframes("vid1", 120, step_sec=30)
    assert len(keyframes) == 4
    assert keyframes[0].scene_type in SCENE_TYPES
    assert 0.0 <= keyframes[0].visual_complexity <= 1.0


def test_multimodal_engine_analysis_and_export():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        interim_dir = tmp_path / "interim"
        output_dir = tmp_path / "output"
        interim_dir.mkdir()
        output_dir.mkdir()
        
        # Write dummy comments and videos Parquet files
        comments_data = {
            "comment_id": ["c1", "c2", "c3"],
            "video_id": ["test_v1", "test_v1", "test_v1"],
            "text": ["Great explanation at 00:45!", "Wait why did you do that at 1:15? Confused 🤔", "Awesome music at 02:00"],
            "vader_compound": [0.8, -0.4, 0.6],
            "toxicity": [0.01, 0.05, 0.01],
            "like_count": [12, 5, 2]
        }
        pd.DataFrame(comments_data).to_parquet(interim_dir / "comments_clean.parquet")
        
        videos_data = {
            "video_id": ["test_v1"],
            "title": ["Distributed Systems Tutorial"],
            "duration": [180],
            "comment_count": [3]
        }
        pd.DataFrame(videos_data).to_parquet(interim_dir / "videos_clean.parquet")
        
        config = {
            "paths": {
                "interim_dir": str(interim_dir),
                "output_dir": str(output_dir)
            }
        }
        
        engine = MultimodalEngine(config)
        vids = engine.get_available_videos()
        assert len(vids) >= 1
        assert vids[0]["video_id"] == "test_v1"
        
        report = engine.analyze_video("test_v1", step_secs=30)
        assert report.video_id == "test_v1"
        assert report.duration_secs == 180
        assert report.total_windows == 6  # 180 / 30
        assert len(report.aligned_windows) == 6
        assert len(report.chapters) >= 1
        assert 0.0 <= report.overall_coherence_score <= 1.0
        
        # Verify Parquet exports
        align_pq = output_dir / "multimodal_alignment.parquet"
        chap_pq = output_dir / "multimodal_chapters.parquet"
        assert align_pq.exists()
        assert chap_pq.exists()
        
        df_align = pd.read_parquet(align_pq)
        assert len(df_align) == 6
        assert "coherence_score" in df_align.columns
        assert "visual_scene_type" in df_align.columns
        
        # Test JSON export
        json_out = output_dir / "report.json"
        engine.export_json(report, json_out)
        assert json_out.exists()
        loaded = json.loads(json_out.read_text(encoding="utf-8"))
        assert loaded["video_id"] == "test_v1"


def test_cli_execution(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        interim_dir = tmp_path / "interim"
        output_dir = tmp_path / "output"
        interim_dir.mkdir()
        output_dir.mkdir()
        
        monkeypatch.setattr(
            "engine.multimodal.load_config",
            lambda: {"paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}}
        )
        
        # 1. Test --status
        monkeypatch.setattr("sys.argv", ["ytint-multimodal", "--status"])
        cli_main()
        out1 = capsys.readouterr().out
        assert "Multimodal Alignment Engine Status" in out1
        
        # 2. Test --mock --chapters --align --coherence --hotspots
        export_file = str(output_dir / "cli_report.json")
        monkeypatch.setattr(
            "sys.argv",
            ["ytint-multimodal", "--mock", "--chapters", "--align", "--coherence", "--hotspots", "--export", export_file]
        )
        cli_main()
        out2 = capsys.readouterr().out
        assert "MULTIMODAL DOSSIER" in out2
        assert "SYNTHESIZED YOUTUBE CHAPTERS" in out2
        assert "SCENE WINDOW ALIGNMENT LEDGER" in out2
        assert Path(export_file).exists()
