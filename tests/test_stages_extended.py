"""Unit tests for pipeline stages s07, s08, s15, s30, s31, s33, s34."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import pandas as pd
from pipeline.s15_ner import run_ner
from pipeline.s30_code_switching import run_code_switching
from pipeline.s31_cross_video_comparisons import run_cross_video
from pipeline.s33_author_fingerprints import run_author_fingerprints
from pipeline.s34_impersonation_detection import run_impersonation_detection


def test_s15_ner_run(tmp_path, monkeypatch):
    """Verify named entity extraction pipeline stage execution."""
    data_dir = tmp_path / "data"
    interim_dir = data_dir / "interim"
    output_dir = data_dir / "output"
    interim_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    df_c = pd.DataFrame({
        "comment_id": ["c1", "c2"],
        "text": ["Google and OpenAI are located in California.", "Elon Musk speaks at MIT."]
    })
    df_c.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    monkeypatch.setattr("pipeline.s15_ner.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })

    run_ner()
    assert (output_dir / "named_entities.parquet").exists()


def test_s30_code_switching_run(tmp_path, monkeypatch):
    """Verify code-switching language classification stage execution."""
    data_dir = tmp_path / "data"
    interim_dir = data_dir / "interim"
    output_dir = data_dir / "output"
    interim_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    df_c = pd.DataFrame({
        "comment_id": ["c1", "c2"],
        "text": ["This is pure English text for testing.", "Привет world! Это mix English и Русский text."],
        "like_count": [10, 25]
    })
    df_c.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    monkeypatch.setattr("pipeline.s30_code_switching.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })

    run_code_switching()
    assert (output_dir / "code_switching_impact.parquet").exists()


def test_s31_cross_video_comparisons_mock(tmp_path, monkeypatch):
    """Verify cross-video comparative profile generation."""
    data_dir = tmp_path / "data"
    interim_dir = data_dir / "interim"
    output_dir = data_dir / "output"
    interim_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    df_c = pd.DataFrame({
        "video_id": ["v1"] * 60 + ["v2"] * 60,
        "comment_id": [f"c{i}" for i in range(120)],
        "published_at": pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC"),
        "text": ["Great content!"] * 120,
        "sentiment_label": ["POSITIVE"] * 40 + ["NEGATIVE"] * 80,
        "vader_compound": [0.8] * 40 + [-0.8] * 80,
        "like_count": [5] * 120,
        "parent_id": [None] * 120
    })
    df_c.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    monkeypatch.setattr("pipeline.s31_cross_video_comparisons.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })

    run_cross_video()
    assert (output_dir / "video_profile_radar.parquet").exists()
    assert (output_dir / "controversy_impact.parquet").exists()


def test_s33_author_fingerprints_mock(tmp_path, monkeypatch):
    """Verify super-fan author behavioral radar fingerprinting."""
    data_dir = tmp_path / "data"
    interim_dir = data_dir / "interim"
    output_dir = data_dir / "output"
    interim_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    df_c = pd.DataFrame({
        "author_channel_id": ["a1"] * 10 + ["a2"] * 5,
        "comment_id": [f"c{i}" for i in range(15)],
        "text": ["Great video!"] * 15,
        "sentiment_label": ["POSITIVE"] * 15,
        "like_count": [2] * 15,
        "parent_id": [None] * 15
    })
    df_c.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    monkeypatch.setattr("pipeline.s33_author_fingerprints.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })

    run_author_fingerprints()
    assert (output_dir / "author_fingerprints.parquet").exists()


def test_s34_impersonation_detection_mock(tmp_path, monkeypatch):
    """Verify impersonation & creator spoofing detection."""
    data_dir = tmp_path / "data"
    interim_dir = data_dir / "interim"
    output_dir = data_dir / "output"
    interim_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    df_c = pd.DataFrame({
        "author_display_name": ["Official Channel", "Official Channel", "Normal User"],
        "author_channel_id": ["c1", "c2", "c3"],
        "text": ["Thanks for watching!", "Message me on Telegram for prizes!", "Nice content!"]
    })
    df_c.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    monkeypatch.setattr("pipeline.s34_impersonation_detection.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })

    run_impersonation_detection()
    assert (output_dir / "impersonation_detection.parquet").exists()
