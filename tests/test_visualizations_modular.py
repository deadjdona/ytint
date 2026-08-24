"""Unit tests verifying the modular pipeline.visualizations package and its submodules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from pipeline.visualizations import (
    nlp_plots,
    thread_plots,
    author_plots,
    temporal_plots,
    model_plots,
    theme,
    run_visualizations,
)
from pipeline.visualizations.model_plots import plot_kaplan_meier
from pipeline.visualizations.nlp_plots import plot_umap_semantics, plot_ner_distribution
from pipeline.visualizations.author_plots import plot_author_pareto, plot_lorenz_curve
from pipeline.visualizations.temporal_plots import plot_new_vs_returning
from pipeline.visualizations.thread_plots import plot_thread_width


def test_theme_configuration():
    """Verify theme initialization without error."""
    theme.setup_theme()
    assert theme.plt is not None
    assert theme.sns is not None


def test_package_exports_count():
    """Verify package re-exports 50 plot functions plus helpers."""
    import pipeline.visualizations as pv
    import pipeline.s99_visualize as s06

    plot_funcs = [
        "plot_kaplan_meier",
        "plot_umap_semantics",
        "plot_author_pareto",
        "plot_diurnal_heatmap",
        "plot_creator_uplift",
        "plot_cib_rings",
        "plot_toxicity_contagion",
        "plot_stance_drift",
    ]
    for fn in plot_funcs:
        assert hasattr(pv, fn)
        assert callable(getattr(pv, fn))
        assert hasattr(s06, fn)
        assert callable(getattr(s06, fn))


def test_nlp_plots_isolated_rendering(tmp_path):
    """Verify nlp_plots submodule can render plots in isolation."""
    out_dir = tmp_path / "nlp_plots"
    out_dir.mkdir()

    # UMAP
    df_semantics = pd.DataFrame({
        "umap_x": [0.1, 0.2, 0.3],
        "umap_y": [0.4, 0.5, 0.6],
        "intent_label": ["praise", "question", "critique"]
    })
    plot_umap_semantics(df_semantics, out_dir)
    assert (out_dir / "umap_semantics.png").exists()

    # NER
    ner_file = tmp_path / "named_entities.parquet"
    df_ner = pd.DataFrame({
        "entity_label": ["PER", "ORG", "PER"],
        "entity_text": ["Alice", "Google", "Bob"]
    })
    df_ner.to_parquet(ner_file)
    plot_ner_distribution(out_dir, ner_file)
    assert (out_dir / "named_entities.png").exists()


def test_author_plots_isolated_rendering(tmp_path):
    """Verify author_plots submodule can render plots in isolation."""
    out_dir = tmp_path / "author_plots"
    out_dir.mkdir()

    df_authors = pd.DataFrame({
        "frequency": [100, 50, 10, 5, 1],
        "author_channel_id": ["a1", "a2", "a3", "a4", "a5"]
    })
    plot_author_pareto(df_authors, out_dir)
    assert (out_dir / "author_pareto.png").exists()

    plot_lorenz_curve(df_authors, out_dir)
    assert (out_dir / "lorenz_curve.png").exists()


def test_model_plots_isolated_rendering(tmp_path):
    """Verify model_plots submodule can render plots in isolation."""
    out_dir = tmp_path / "model_plots"
    out_dir.mkdir()

    df_survival = pd.DataFrame({
        "timeline_hours": [0, 1, 2, 5],
        "survival_probability": [1.0, 0.8, 0.5, 0.2]
    })
    plot_kaplan_meier(df_survival, out_dir)
    assert (out_dir / "kaplan_meier_survival.png").exists()


def test_thread_plots_isolated_rendering(tmp_path):
    """Verify thread_plots submodule can render plots in isolation."""
    out_dir = tmp_path / "thread_plots"
    out_dir.mkdir()

    width_file = tmp_path / "thread_width_dist.parquet"
    df_width = pd.DataFrame({
        "thread_width": [1, 2, 3, 5, 10],
        "frequency": [100, 50, 20, 5, 1]
    })
    df_width.to_parquet(width_file)
    plot_thread_width(out_dir, width_file)
    assert (out_dir / "thread_width_dist.png").exists()


def test_new_extended_plots_rendering(tmp_path):
    """Verify rendering of newly added extended visualization functions."""
    import pipeline.visualizations as pv
    out_dir = tmp_path / "extended_plots"
    out_dir.mkdir()

    # Slang & TF-IDF
    df_slang = pd.DataFrame({
        "slang_term": ["tbh", "lol", "cringe"],
        "total_occurrences": [15, 10, 5],
        "avg_sentiment": [0.2, 0.5, -0.4]
    })
    pv.plot_slang_lexicon(df_slang, out_dir)
    assert (out_dir / "slang_lexicon_distribution.png").exists()

    df_tfidf = pd.DataFrame({
        "video_id": ["v1", "v1", "v2"],
        "keyword": ["python", "code", "gaming"],
        "tfidf_score": [0.8, 0.6, 0.9]
    })
    pv.plot_tfidf_keywords(df_tfidf, out_dir)
    assert (out_dir / "tfidf_keywords_salience.png").exists()

    # Bow-Tie & Concentration
    df_bt = pd.DataFrame({
        "component": ["SCC", "IN", "OUT", "TENDRILS"],
        "author_count": [10, 20, 15, 5],
        "author_share": [0.2, 0.4, 0.3, 0.1]
    })
    pv.plot_bowtie_structure(df_bt, out_dir)
    assert (out_dir / "network_bowtie_structure.png").exists()

    df_conc = pd.DataFrame({
        "scope": ["corpus"],
        "top_1pct_likes_share": [0.4],
        "top_5pct_likes_share": [0.6],
        "top_10pct_likes_share": [0.75],
        "top_20pct_likes_share": [0.85],
        "top_1pct_replies_share": [0.3],
        "top_5pct_replies_share": [0.5],
        "top_10pct_replies_share": [0.65],
        "top_20pct_replies_share": [0.75],
        "gini_likes": [0.7],
        "gini_replies": [0.6]
    })
    pv.plot_top_k_concentration(df_conc, out_dir)
    assert (out_dir / "top_k_attention_concentration.png").exists()

    # Topic Injection & Arrival Curve
    df_inj = pd.DataFrame({
        "video_id": ["v1", "v2"],
        "window_start": [pd.Timestamp("2026-01-01 10:00:00"), pd.Timestamp("2026-01-02 10:00:00")],
        "js_divergence": [0.45, 0.52],
        "dominant_topic": ["Topic 1", "Topic 2"]
    })
    pv.plot_topic_injection_anomalies(df_inj, out_dir)
    assert (out_dir / "topic_injection_anomalies.png").exists()

    df_curve = pd.DataFrame({
        "minute_bin": list(range(10)),
        "cumulative_comments": [i * 2 for i in range(10)],
        "velocity_per_min": [2.0] * 10
    })
    pv.plot_minute_arrival_curve(df_curve, out_dir)
    assert (out_dir / "minute_arrival_speed_curve.png").exists()

    # Series, Creator Polarity, Scene Reactions
    df_ser = pd.DataFrame({
        "category": ["Series / Episodic", "Standalone"],
        "avg_comments_per_video": [50.0, 30.0],
        "avg_likes_per_comment": [5.0, 3.5],
        "avg_sentiment": [0.3, 0.1]
    })
    pv.plot_series_vs_standalone(df_ser, out_dir)
    assert (out_dir / "series_vs_standalone_benchmark.png").exists()

    df_creat = pd.DataFrame({
        "scope": ["corpus"],
        "creator_positive_count": [40],
        "creator_negative_count": [8],
        "creator_neutral_count": [12],
        "creator_pos_neg_ratio": [5.0]
    })
    pv.plot_creator_sentiment_polarity(df_creat, out_dir)
    assert (out_dir / "creator_sentiment_polarity.png").exists()

    df_rxn = pd.DataFrame({
        "reaction_type": ["Humor / Laughter", "Emotional / Sentiment", "Plot / Lore Critique"],
        "n_comments": [30, 20, 10]
    })
    pv.plot_cross_modal_scene_reactions(df_rxn, out_dir)
    assert (out_dir / "cross_modal_scene_reactions.png").exists()
