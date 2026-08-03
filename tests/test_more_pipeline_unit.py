"""Additional unit tests to expand coverage for s02_topics.py, s04_aggregation.py, and s01_enrich.py."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from pipeline.s02_topics import compute_semantic_drift
from pipeline.s04_aggregation import aggregate_data


def test_compute_semantic_drift(tmp_path):
    """Verify semantic drift calculation function runs without exception."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    texts = ["hello world machine learning python code"] * 100
    df_comments = pd.DataFrame({"published_at": dates, "text": texts})

    compute_semantic_drift(df_comments, out_dir, n_slices=3)


def test_aggregate_data_mock(tmp_path, monkeypatch):
    """Test aggregate_data execution on synthetic interim comments with >=4 author profiles."""
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    df_comments = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3", "c4"],
        "author_channel_id": ["a1", "a2", "a3", "a4"],
        "author_display_name": ["User One", "User Two", "User Three", "User Four"],
        "video_id": ["v1", "v1", "v2", "v2"],
        "published_at": pd.date_range("2026-01-01", periods=4, freq="D"),
        "like_count": [10, 5, 2, 0],
        "vader_compound": [0.5, 0.2, -0.1, 0.0],
        "minutes_since_upload": [100, 200, 300, 400]
    })
    df_comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    mock_config = {
        "paths": {
            "raw_db": str(tmp_path / "raw.db"),
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir),
        },
        "_root_dir": tmp_path
    }
    monkeypatch.setattr("pipeline.s04_aggregation.load_config", lambda: mock_config)

    aggregate_data()

    assert (output_dir / "authors_final.parquet").exists()
    assert (output_dir / "videos_final.parquet").exists()

    df_auth = pd.read_parquet(output_dir / "authors_final.parquet")
    assert not df_auth.empty
    assert "rfm_cohort" in df_auth.columns
    assert "is_single_video_fan" in df_auth.columns
