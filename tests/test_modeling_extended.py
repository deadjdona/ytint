"""Unit tests for s05_modeling.py predictive and statistical models."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import pandas as pd
import numpy as np
from pipeline.s05_modeling import (
    fit_power_law,
    category_benchmarking,
    detect_poisson_bursts,
    survival_analysis,
    detect_near_duplicates
)


def test_fit_power_law():
    """Verify Power-Law MLE distribution estimation."""
    # Synthetic Pareto power-law distribution
    likes = (np.random.pareto(a=1.5, size=100) + 1).astype(int)
    df = pd.DataFrame({"like_count": likes})
    
    fit_df = fit_power_law(df)
    assert not fit_df.empty
    assert "alpha" in fit_df.columns
    assert fit_df.iloc[0]["alpha"] > 0


def test_category_benchmarking():
    """Verify Kruskal-Wallis & Dunn's post-hoc pairwise rank testing."""
    df = pd.DataFrame({
        "video_id": ["v1"] * 15 + ["v2"] * 15 + ["v3"] * 15,
        "like_count": list(range(15)) + list(range(10, 25)) + list(range(5, 20))
    })
    
    stat, p_val, dunn_df = category_benchmarking(df)
    assert stat is not None
    assert p_val is not None
    assert dunn_df is not None
    assert dunn_df.shape == (3, 3)


def test_detect_poisson_bursts():
    """Verify Poisson process burst detection."""
    dates = pd.date_range("2026-01-01", periods=100, freq="15min")
    counts = [1] * 95 + [500] * 5
    records = []
    for d, c in zip(dates, counts):
        records.extend([{"published_at": d}] * c)
        
    df = pd.DataFrame(records)
    bursts = detect_poisson_bursts(df, window="15T")
    assert isinstance(bursts, pd.DataFrame)


def test_survival_analysis_with_logrank():
    """Verify Kaplan-Meier thread survival estimation and Log-Rank test."""
    dates = pd.date_range("2026-01-01", periods=20, freq="h")
    df = pd.DataFrame({
        "comment_id": [f"c{i}" for i in range(20)],
        "parent_id": [None] * 5 + ["c0", "c0", "c1", "c1", "c2"] + [None] * 10,
        "published_at": dates,
        "reply_latency_seconds": [0.0] * 5 + [3600.0, 7200.0, 1800.0, 5400.0, 900.0] + [0.0] * 10,
        "like_count": [100 if i % 2 == 0 else 5 for i in range(20)]
    })
    
    survival_df = survival_analysis(df)
    assert not survival_df.empty
    assert "survival_probability" in survival_df.columns


def test_detect_near_duplicates():
    """Verify MinHash LSH spam near-duplicate detection."""
    df = pd.DataFrame({
        "text": [
            "Check out my channel for free gift cards and giveaways! http://example.com",
            "Check out my channel for free gift cards and giveaways! http://example.com",
            "Check out my channel for free gift cards and giveaways! http://example.com",
            "This video was really informative and well presented."
        ]
    })
    df_out = detect_near_duplicates(df)
    assert "is_spam_duplicate" in df_out.columns
    assert df_out.loc[0, "is_spam_duplicate"] == True
    assert df_out.loc[3, "is_spam_duplicate"] == False
