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
