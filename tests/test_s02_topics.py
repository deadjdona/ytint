"""Unit tests for Stage 02: Topic Modeling and Progress Visibility."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import pandas as pd
import numpy as np

from pipeline.s02_topics import _format_duration, _format_size, compute_semantic_drift, run_topic_modeling


def test_format_helpers():
    """Verify helper formatting functions for duration and byte size."""
    assert "ms" in _format_duration(0.45)
    assert "s" in _format_duration(12.34)
    assert "m" in _format_duration(125.0)

    assert "B" in _format_size(500)
    assert "KB" in _format_size(2048)
    assert "MB" in _format_size(1024 * 1024 * 5)


def test_compute_semantic_drift_progress(tmp_path, capsys):
    """Verify compute_semantic_drift runs with progress tracking and outputs parquet."""
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    texts = [
        "great video tutorial python data science analytics",
        "awesome coding pipeline deep learning machine intelligence",
        "neural network classification bertopic natural language"
    ] * 20
    df_comments = pd.DataFrame({"published_at": dates, "text": texts})

    compute_semantic_drift(df_comments, out_dir, n_slices=3)
    captured = capsys.readouterr().out
    assert "Temporal Semantic Drift Tracking" in captured


def test_run_topic_modeling_mock(tmp_path, monkeypatch, capsys):
    """Verify run_topic_modeling runs end-to-end with progress visibility phases."""
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic comments with root and reply comments
    df_comments = pd.DataFrame({
        "comment_id": [f"c{i}" for i in range(30)],
        "parent_id": [None if i % 3 != 0 else "c0" for i in range(30)],
        "text": [
            "video content tutorial python pandas analysis" if i % 2 == 0 
            else "neural network deep learning artificial intelligence" 
            for i in range(30)
        ],
        "published_at": pd.date_range("2026-01-01", periods=30, freq="D")
    })
    df_comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    # Pre-computed dummy embeddings to speed up test execution
    dummy_embeddings = np.random.randn(20, 384).astype(np.float32)
    np.save(interim_dir / "topic_embeddings.npy", dummy_embeddings)

    mock_config = {
        "paths": {
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir),
        },
        "stage_02_topics": {
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "min_topic_size": 2,
        },
        "_root_dir": tmp_path
    }
    monkeypatch.setattr("pipeline.s02_topics.load_config", lambda: mock_config)

    run_topic_modeling()

    captured = capsys.readouterr().out
    assert "[Phase 1/6]" in captured
    assert "[Phase 2/6]" in captured
    assert "[Phase 3/6]" in captured
    assert "[Phase 4/6]" in captured
    assert "[Phase 5/6]" in captured
    assert "[Phase 6/6]" in captured
    assert "STAGE 02 COMPLETE" in captured

    assert (output_dir / "topic_metadata.parquet").exists()
    assert (interim_dir / "comments_clean.parquet").exists()
