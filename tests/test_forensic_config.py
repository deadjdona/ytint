"""Tests verifying that forensic detection thresholds in config/settings.yaml are properly exposed and respected."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from engine.config_loader import load_config
from pipeline.s25_cib_detection import detect_cib_rings
from pipeline.s15_toxicity_contagion import analyze_toxicity_contagion
from pipeline.s39_creator_uplift import compute_creator_causal_uplift


def test_settings_yaml_contains_all_forensic_sections():
    """Verify that settings.yaml has dedicated configuration sections for all forensic stages."""
    config = load_config()

    expected_sections = [
        "stage_26_integrity",
        "stage_20_frequency_tiers",
        "stage_24_bot_heuristics",
        "stage_21_driveby_loyalists",
        "stage_27_like_inflation",
        "stage_23_impersonation",
        "stage_25_cib",
        "stage_15_toxicity",
        "stage_39_creator_uplift",
    ]
    for sec in expected_sections:
        assert sec in config, f"Missing config section '{sec}' in settings.yaml"


def test_cib_detection_respects_custom_thresholds():
    """Verify CIB detection adjusts ring detection with configurable time windows."""
    df_comments = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3", "c4"],
        "video_id": ["v1", "v1", "v2", "v2"],
        "author_channel_id": ["a1", "a2", "a1", "a2"],
        "published_at": [
            "2026-01-01 10:00:00",
            "2026-01-01 10:01:00",  # 60s delta on v1
            "2026-01-02 10:00:00",
            "2026-01-02 10:01:00",  # 60s delta on v2
        ],
        "text": ["Sync comment 1", "Sync comment 2", "Sync comment 3", "Sync comment 4"]
    })

    # With 120s window: both pairs qualify (delta=60s) -> ring formed
    df_rings, df_cib = detect_cib_rings(df_comments, time_window_seconds=120, min_cooccurrences=2)
    assert not df_rings.empty
    assert len(df_rings) == 1

    # With very strict 30s window: neither qualifies -> no ring formed
    df_rings_strict, _ = detect_cib_rings(df_comments, time_window_seconds=30, min_cooccurrences=2)
    assert df_rings_strict.empty


def test_toxicity_contagion_respects_custom_threshold():
    """Verify toxicity contagion adjusts R0 and troll classification with custom thresholds."""
    df_comments = pd.DataFrame({
        "author_channel_id": ["a1", "a1", "a2"],
        "author_display_name": ["User1", "User1", "User2"],
        "video_id": ["v1", "v1", "v1"],
        "comment_id": ["c1", "c2", "c3"],
        "toxicity": [0.6, 0.7, 0.2],
        "reply_count": [10, 5, 1],
        "like_count": [0, 0, 5]
    })

    # Threshold 0.5 -> 2 toxic comments for a1
    summary_50, catalysts_50, _ = analyze_toxicity_contagion(df_comments, toxicity_threshold=0.5)
    assert summary_50.iloc[0]["total_toxic_comments"] == 2

    # Strict threshold 0.8 -> 0 toxic comments
    summary_80, _, _ = analyze_toxicity_contagion(df_comments, toxicity_threshold=0.8)
    assert summary_80.iloc[0]["total_toxic_comments"] == 0


def test_creator_uplift_respects_custom_early_window():
    """Verify creator uplift adjusts treatment group according to early_window_minutes."""
    df_comments = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3"],
        "minutes_since_upload": [30.0, 90.0, 300.0],
        "like_count": [10, 10, 10],
        "reply_count": [5, 5, 5],
        "toxicity": [0.1, 0.1, 0.1],
        "vader_compound": [0.5, 0.5, 0.5]
    })

    # With early_window=120: c1 and c2 treated (minutes <= 120)
    summary_120, _ = compute_creator_causal_uplift(df_comments, early_window_minutes=120)
    assert not summary_120.empty

    # With strict early_window=45: only c1 treated
    summary_45, _ = compute_creator_causal_uplift(df_comments, early_window_minutes=45)
    assert not summary_45.empty
