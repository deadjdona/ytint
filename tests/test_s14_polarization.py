"""Unit tests for Stage 14: Thread Polarization Dynamics (s14_polarization.py)."""

import pandas as pd
import pytest
from pathlib import Path
from pipeline.s14_polarization import run_polarization_dynamics


def test_s14_polarization_with_missing_comment_depth(tmp_path, monkeypatch):
    """Verify s14 calculates comment_depth dynamically and succeeds when absent."""
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic comments WITHOUT 'comment_depth'
    # Tree structure:
    # c1 (root, depth 0) -> c2 (reply to c1, depth 1) -> c3 (reply to c2, depth 2)
    # c4 (root, depth 0)
    comments = pd.DataFrame([
        {
            "comment_id": "c1",
            "parent_id": None,
            "vader_compound": 0.5,
            "published_at": "2026-01-01 10:00:00"
        },
        {
            "comment_id": "c2",
            "parent_id": "c1",
            "vader_compound": -0.2,
            "published_at": "2026-01-01 10:30:00"
        },
        {
            "comment_id": "c3",
            "parent_id": "c2",
            "vader_compound": -0.8,
            "published_at": "2026-01-01 11:00:00"
        },
        {
            "comment_id": "c4",
            "parent_id": None,
            "vader_compound": 0.9,
            "published_at": "2026-01-01 12:00:00"
        }
    ])
    comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    fake_config = {
        "paths": {
            "interim_dir": interim_dir,
            "output_dir": output_dir
        }
    }
    monkeypatch.setattr("pipeline.s14_polarization.load_config", lambda: fake_config)

    run_polarization_dynamics()

    out_file = output_dir / "thread_polarization_corpus.parquet"
    assert out_file.exists(), "thread_polarization_corpus.parquet was not created!"

    df_out = pd.read_parquet(out_file)
    assert not df_out.empty
    assert "comment_depth" in df_out.columns
    assert "mean_sentiment" in df_out.columns
    assert "comment_count" in df_out.columns

    # Depths 0, 1, 2 should all be present in the output
    depths = set(df_out["comment_depth"].tolist())
    assert 0 in depths
    assert 1 in depths
    assert 2 in depths


def test_s14_polarization_with_precomputed_comment_depth(tmp_path, monkeypatch):
    """Verify s14 utilizes existing comment_depth when already present."""
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    comments = pd.DataFrame([
        {
            "comment_id": "c1",
            "parent_id": None,
            "comment_depth": 0,
            "vader_compound": 0.4,
            "published_at": "2026-01-01 10:00:00"
        },
        {
            "comment_id": "c2",
            "parent_id": "c1",
            "comment_depth": 1,
            "vader_compound": -0.1,
            "published_at": "2026-01-01 10:15:00"
        }
    ])
    comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    fake_config = {
        "paths": {
            "interim_dir": interim_dir,
            "output_dir": output_dir
        }
    }
    monkeypatch.setattr("pipeline.s14_polarization.load_config", lambda: fake_config)

    run_polarization_dynamics()

    out_file = output_dir / "thread_polarization_corpus.parquet"
    assert out_file.exists()
    df_out = pd.read_parquet(out_file)
    assert len(df_out) == 2


def test_s14_polarization_empty_comments(tmp_path, monkeypatch):
    """Verify s14 handles empty comments without raising an unhandled exception."""
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    comments = pd.DataFrame(columns=["comment_id", "parent_id", "vader_compound"])
    comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    fake_config = {
        "paths": {
            "interim_dir": interim_dir,
            "output_dir": output_dir
        }
    }
    monkeypatch.setattr("pipeline.s14_polarization.load_config", lambda: fake_config)

    # Should exit gracefully
    run_polarization_dynamics()
