"""Unit tests for the src/pipeline/ functions."""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from pipeline.s01_enrich import calculate_linguistic_features
from pipeline.s16_network import compute_author_metrics, compute_bipartite_graph, compute_cocommenting_jaccard
from pipeline.s19_aggregation import gini, compute_half_life
from pipeline.s38_modeling import detect_anomalies, survival_analysis, stl_decomposition, detect_near_duplicates, detect_poisson_bursts
from pipeline.s99_visualize import (
    plot_kaplan_meier, plot_umap_semantics, plot_author_pareto, plot_diurnal_heatmap,
    plot_plutchik_radar, plot_sentiment_divergence, plot_toxicity_heatmap, plot_reply_depth_distribution, plot_lorenz_curve
)


def test_calculate_linguistic_features():
    """Test linguistic feature calculations on sample text."""
    features = calculate_linguistic_features("WOAH THIS IS AMAZING! Check out https://youtube.com #awesome @user 1:23")
    assert features["char_count"] > 0
    assert features["word_count"] > 0
    assert features["all_caps_ratio"] > 0
    assert "#awesome" in features["hashtags"]
    assert "@user" in features["mentions"]
    assert "1:23" in features["video_timestamps"]


def test_gini_coefficient():
    """Verify Gini coefficient calculation."""
    # Equal distribution -> Gini = 0
    assert gini(np.array([10, 10, 10, 10])) == pytest.approx(0.0)
    # Complete inequality -> Gini close to 1
    assert gini(np.array([0, 0, 0, 100])) > 0.6


def test_compute_half_life():
    """Verify exponential half-life fitting."""
    df_comments = pd.DataFrame({
        "minutes_since_upload": [i * 1440.0 for i in range(10)]
    })
    hl = compute_half_life(df_comments)
    assert isinstance(hl, float)


def test_network_metrics():
    """Verify author graph metrics calculation."""
    df_comments = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3"],
        "parent_id": [None, "c1", "c2"],
        "author_channel_id": ["a1", "a2", "a3"],
        "video_id": ["v1", "v1", "v2"]
    })
    df_authors, G_auth = compute_author_metrics(df_comments)
    assert not df_authors.empty
    assert "pagerank" in df_authors.columns
    assert "community_id" in df_authors.columns

    B = compute_bipartite_graph(df_comments)
    assert isinstance(B, nx.Graph)
    assert B.number_of_nodes() > 0

    J_G = compute_cocommenting_jaccard(df_comments, max_authors=10, min_jaccard=0.0)
    assert isinstance(J_G, nx.Graph)


def test_detect_anomalies_isolation_forest():
    """Verify Isolation Forest bot detection on synthetic author profiles."""
    df_authors = pd.DataFrame({
        "author_channel_id": [f"a{i}" for i in range(100)],
        "frequency": [1] * 99 + [10000],
        "recency": [10] * 99 + [0],
        "monetary": [5] * 99 + [50000]
    })
    df_out, _ = detect_anomalies(df_authors)
    assert "is_bot_suspect" in df_out.columns
    assert df_out["is_bot_suspect"].sum() >= 1


def test_survival_analysis():
    """Verify Kaplan-Meier thread survival estimation."""
    df_comments = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3"],
        "parent_id": [None, "c1", "c1"],
        "published_at": pd.date_range("2026-01-01", periods=3, freq="h"),
        "reply_latency_seconds": [0, 3600, 7200]
    })
    survival_df = survival_analysis(df_comments)
    assert isinstance(survival_df, pd.DataFrame)


def test_stl_decomposition():
    """Verify STL decomposition."""
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    df_comments = pd.DataFrame({"published_at": dates})
    stl_df, diurnal = stl_decomposition(df_comments)
    assert not stl_df.empty
    assert not diurnal.empty


def test_detect_near_duplicates():
    """Verify MinHash LSH near duplicate spam detection."""
    df_comments = pd.DataFrame({
        "text": [
            "buy cheap crypto now at http://scam.com",
            "buy cheap crypto now at http://scam.com",
            "buy cheap crypto now at http://scam.com",
            "This is a legitimate unique user comment."
        ]
    })
    res_df = detect_near_duplicates(df_comments, threshold=0.7)
    assert "is_spam_duplicate" in res_df.columns
    assert res_df["is_spam_duplicate"].iloc[0] == True


def test_detect_poisson_bursts():
    """Verify Poisson burst detection."""
    dates = list(pd.date_range("2026-01-01", periods=10, freq="min")) + [pd.Timestamp("2026-01-01 00:05:00")] * 500
    df_comments = pd.DataFrame({"published_at": dates})
    burst_df = detect_poisson_bursts(df_comments, window="5min")
    assert isinstance(burst_df, pd.DataFrame)


def test_visualizations_headless_rendering(tmp_path):
    """Verify plot generation functions run headlessly without error."""
    out_dir = tmp_path / "plots"
    out_dir.mkdir()

    # Kaplan Meier
    df_survival = pd.DataFrame({"timeline_hours": [0, 1, 2], "survival_probability": [1.0, 0.8, 0.5]})
    plot_kaplan_meier(df_survival, out_dir)
    assert (out_dir / "kaplan_meier_survival.png").exists()

    # UMAP
    df_semantics = pd.DataFrame({"umap_x": [0.1, 0.2], "umap_y": [0.3, 0.4], "intent_label": ["praise", "question"]})
    plot_umap_semantics(df_semantics, out_dir)
    assert (out_dir / "umap_semantics.png").exists()

    # Author Pareto
    df_authors = pd.DataFrame({"frequency": [100, 50, 10]})
    plot_author_pareto(df_authors, out_dir)
    assert (out_dir / "author_pareto.png").exists()

    # Lorenz Curve
    plot_lorenz_curve(df_authors, out_dir)
    assert (out_dir / "lorenz_curve.png").exists()
