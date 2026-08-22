"""Unit tests for Stage 24: Bot & Spammer Heuristics Classifier."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import pandas as pd
import numpy as np

from pipeline.s24_bot_heuristics import run_bot_heuristics, _safe_read_parquet


def test_safe_read_parquet(tmp_path):
    """Verify _safe_read_parquet only loads available columns and does not crash on missing ones."""
    test_file = tmp_path / "test.parquet"
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    df.to_parquet(test_file, index=False)

    # Requesting a mix of existing and non-existing columns
    res = _safe_read_parquet(test_file, ["col_a", "col_nonexistent", "col_missing"])
    assert list(res.columns) == ["col_a"]

    # Requesting only non-existing columns falls back to all columns without crashing
    res_fallback = _safe_read_parquet(test_file, ["nonexistent_1", "nonexistent_2"])
    assert "col_a" in res_fallback.columns and "col_b" in res_fallback.columns


def test_run_bot_heuristics_without_is_bot_suspect_column(tmp_path, monkeypatch):
    """Verify run_bot_heuristics executes cleanly when authors_final.parquet lacks is_bot_suspect."""
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic comments
    df_comments = pd.DataFrame({
        "author_channel_id": ["author_spam"] * 15 + ["author_human"] * 10,
        "text": ["buy crypto fast now at scam link"] * 15 + [f"unique comment {i}" for i in range(10)],
        "lexical_richness": [0.1] * 15 + [0.8] * 10
    })
    df_comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    # 2. authors_final.parquet without is_bot_suspect (reproducing upstream s19 output)
    df_authors = pd.DataFrame({
        "author_channel_id": ["author_spam", "author_human"],
        "is_display_name_reused": [True, False],
        "pagerank": [0.01, 0.05]
    })
    df_authors.to_parquet(output_dir / "authors_final.parquet", index=False)

    mock_config = {
        "paths": {
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir),
        },
        "stage_24_bot_heuristics": {
            "tier1_min_comments": 10,
            "tier1_max_unique_ratio": 0.2,
            "tier2_min_comments": 50,
            "tier2_max_unique_ratio": 0.5,
            "reused_name_min_comments": 5,
            "reused_name_max_unique": 0.3
        }
    }
    monkeypatch.setattr("pipeline.s24_bot_heuristics.load_config", lambda: mock_config)

    run_bot_heuristics()

    bot_file = output_dir / "bot_classifications.parquet"
    assert bot_file.exists()
    df_out = pd.read_parquet(bot_file)
    assert not df_out.empty
    assert "is_bot" in df_out.columns
    assert "classification" in df_out.columns

    spam_row = df_out[df_out['author_channel_id'] == 'author_spam'].iloc[0]
    human_row = df_out[df_out['author_channel_id'] == 'author_human'].iloc[0]

    assert spam_row['is_bot'] == True
    assert human_row['is_bot'] == False


def test_run_bot_heuristics_with_is_bot_suspect_column(tmp_path, monkeypatch):
    """Verify run_bot_heuristics integrates is_bot_suspect when present in authors_final.parquet."""
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_comments = pd.DataFrame({
        "author_channel_id": ["author_flagged", "author_normal"],
        "text": ["normal comment 1", "normal comment 2"]
    })
    df_comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    df_authors = pd.DataFrame({
        "author_channel_id": ["author_flagged", "author_normal"],
        "is_display_name_reused": [False, False],
        "pagerank": [0.01, 0.02],
        "is_bot_suspect": [True, False]
    })
    df_authors.to_parquet(output_dir / "authors_final.parquet", index=False)

    mock_config = {
        "paths": {
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir),
        }
    }
    monkeypatch.setattr("pipeline.s24_bot_heuristics.load_config", lambda: mock_config)

    run_bot_heuristics()

    bot_file = output_dir / "bot_classifications.parquet"
    assert bot_file.exists()
    df_out = pd.read_parquet(bot_file)

    flagged_row = df_out[df_out['author_channel_id'] == 'author_flagged'].iloc[0]
    assert flagged_row['is_bot'] == True
