"""Unit tests for the src/engine/ package modules."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import numpy as np
import pandas as pd
from engine.config_loader import load_config, get_paths, _find_project_root
from engine.language import detect_corpus_languages, build_stopword_set, corpus_stopwords
from engine.changepoint import select_pelt_penalty
from engine.stats import calculate_rolling_z_scores, detect_volume_anomalies, segment_thematic_eras
from engine.metrics import calculate_aging_scores, calculate_controversy_index


def test_config_loader():
    """Verify load_config and get_paths correctly resolve absolute paths."""
    config = load_config()
    assert "paths" in config
    assert isinstance(config["_root_dir"], Path)

    raw_db, interim_dir, output_dir = get_paths(config)
    assert isinstance(raw_db, Path)
    assert isinstance(interim_dir, Path)
    assert isinstance(output_dir, Path)


def test_language_detection():
    """Verify corpus language detection and stopword merging."""
    docs = [
        "This is a fantastic YouTube video about machine learning.",
        "I really love this tutorial, super clear and helpful!",
        "Great content, keep up the amazing work!"
    ]
    langs = detect_corpus_languages(docs, sample_size=10)
    assert "en" in langs

    stopwords = build_stopword_set({"en"})
    assert "the" in stopwords
    assert "http" in stopwords

    sw, detected = corpus_stopwords(docs, sample_size=10)
    assert "en" in detected
    assert len(sw) > 0


def test_select_pelt_penalty():
    """Verify PELT penalty selection with synthetic step signal."""
    np.random.seed(42)
    signal = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(10, 1, 50)])
    penalty, n_bkps = select_pelt_penalty(signal, model="l2")
    assert isinstance(penalty, float)
    assert len(n_bkps) > 0


def test_stats_rolling_z_scores():
    """Verify rolling z-score calculation."""
    s = pd.Series([10.0] * 10 + [100.0] + [10.0] * 10)
    z = calculate_rolling_z_scores(s, window_size=5, min_periods=1)
    assert len(z) == len(s)
    assert z.iloc[10] > 1.0


def test_detect_volume_anomalies():
    """Verify volume anomaly spike detection."""
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    counts = [10] * 29 + [1000]
    df_comments = pd.DataFrame({"published_at": dates, "comment_id": range(30)})
    df_comments = df_comments.loc[df_comments.index.repeat(counts)].reset_index(drop=True)

    anomalies = detect_volume_anomalies(df_comments, z_threshold=2.0, window_days=7)
    assert not anomalies.empty
    assert "z_score" in anomalies.columns


def test_segment_thematic_eras():
    """Verify thematic era segmentation."""
    dates = pd.date_range("2026-01-01", periods=30, freq="W")
    df_comments = pd.DataFrame({
        "published_at": dates,
        "topic": [1] * 15 + [2] * 15
    })
    eras = segment_thematic_eras(df_comments, min_duration_weeks=4, penalty_modifier=1.0)
    assert isinstance(eras, list)


def test_calculate_aging_scores():
    """Verify aging scores calculation across strata."""
    df_comments = pd.DataFrame({
        "video_id": ["v1"] * 4,
        "days_since_upload": [1, 2, 400, 405],
        "sentiment_label": ["negative", "negative", "positive", "positive"]
    })
    aging_df = calculate_aging_scores(df_comments)
    assert not aging_df.empty
    assert "aging_score" in aging_df.columns
    # Went from -1.0 to +1.0 => aging_score = +2.0
    assert aging_df.loc[0, "aging_score"] == pytest.approx(2.0)


def test_calculate_controversy_index():
    """Verify controversy index calculation."""
    df_comments = pd.DataFrame({
        "video_id": ["v1", "v1", "v1"],
        "comment_id": ["c1", "c2", "c3"],
        "parent_id": [None, "c1", "c1"],
        "sentiment_label": ["negative", "negative", "negative"]
    })
    controversy_df = calculate_controversy_index(df_comments)
    assert not controversy_df.empty
    assert "controversy_score" in controversy_df.columns
    assert controversy_df.loc[0, "controversy_score"] > 0
