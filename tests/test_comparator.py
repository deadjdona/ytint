"""Unit and integration tests for ytint Multi-Channel & Playlist Competitive Intelligence Engine."""

import json
import os
import pathlib
import sys
import pytest
import pandas as pd
import numpy as np

from engine.comparator import (
    CompetitiveIntelligenceEngine,
    ChannelCohortProfile,
    AudienceOverlapMetrics,
    ComparativeIntelligenceReport,
    RADAR_DIMENSIONS,
    main as comparator_main,
)


def test_channel_cohort_profile_serialization():
    radar = {dim: 80.0 for dim in RADAR_DIMENSIONS}
    profile = ChannelCohortProfile(
        cohort_id="test_ch_01",
        cohort_name="Test Channel 01",
        cohort_type="channel",
        total_videos=25,
        total_comments=5000,
        unique_authors=1200,
        comments_per_video=200.0,
        avg_likes_per_comment=4.5,
        reply_ratio=0.25,
        sentiment_pos_ratio=0.55,
        sentiment_neg_ratio=0.15,
        avg_vader_compound=0.32,
        mean_toxicity=0.08,
        toxicity_r0=0.85,
        avg_author_loyalty=4.16,
        champion_author_ratio=0.06,
        vocab_entropy=7.2,
        top_emojis=["🔥", "👍"],
        diurnal_peak_utc=18,
        radar_scores=radar,
    )
    d = profile.to_dict()
    assert d["cohort_id"] == "test_ch_01"
    assert d["cohort_name"] == "Test Channel 01"
    assert d["total_videos"] == 25
    assert d["total_comments"] == 5000
    assert d["radar_scores"]["Discussion Volume"] == 80.0


def test_audience_overlap_metrics_serialization():
    overlap = AudienceOverlapMetrics(
        cohort_a_id="ch_a",
        cohort_b_id="ch_b",
        cohort_a_name="Channel A",
        cohort_b_name="Channel B",
        authors_a=1000,
        authors_b=500,
        shared_authors=200,
        jaccard_similarity=0.1538,
        overlap_coefficient=0.40,
        sentiment_differential=-0.12,
        toxicity_differential=+0.02,
        top_migrated_commenters=[
            {"author_id": "u1", "author_name": "User 1", "comments_in_a": 10, "comments_in_b": 5, "total_comments": 15, "sentiment_in_a": 0.5, "sentiment_in_b": 0.2, "sentiment_delta": -0.3}
        ],
    )
    d = overlap.to_dict()
    assert d["cohort_a_id"] == "ch_a"
    assert d["shared_authors"] == 200
    assert d["jaccard_similarity"] == 0.1538
    assert len(d["top_migrated_commenters"]) == 1


def test_mock_report_generation():
    engine = CompetitiveIntelligenceEngine()
    report = engine.generate_mock_report()

    assert isinstance(report, ComparativeIntelligenceReport)
    assert len(report.profiles) == 3
    assert len(report.pairwise_overlaps) == 2
    assert len(report.radar_dimensions) == 6

    d = report.to_dict()
    assert "profiles" in d
    assert "pairwise_overlaps" in d
    assert "radar_dimensions" in d
    assert "ch_alpha_osint" in d["profiles"]


def test_discover_available_channels():
    engine = CompetitiveIntelligenceEngine()
    channels = engine.discover_available_channels()
    assert isinstance(channels, list)
    if channels:
        first = channels[0]
        assert "channel_id" in first
        assert "video_count" in first
        assert "comment_count" in first


def test_discover_available_cohorts():
    engine = CompetitiveIntelligenceEngine()
    cohorts_q = engine.discover_available_cohorts(group_by="quarter")
    assert isinstance(cohorts_q, list)

    cohorts_y = engine.discover_available_cohorts(group_by="year")
    assert isinstance(cohorts_y, list)


def test_build_profile_from_slice_empty():
    engine = CompetitiveIntelligenceEngine()
    empty_df = pd.DataFrame()
    profile = engine.build_profile_from_slice("c_empty", "Empty Cohort", "test", 0, empty_df)
    assert profile.total_comments == 0
    assert profile.unique_authors == 0
    assert len(profile.radar_scores) == 6


def test_build_profile_from_slice_synthetic():
    engine = CompetitiveIntelligenceEngine()
    synthetic_df = pd.DataFrame({
        "author_channel_id": ["u1", "u2", "u1", "u3", "u1", "u1", "u1"],
        "like_count": [10, 5, 2, 0, 1, 3, 4],
        "is_reply": [False, True, False, False, True, False, True],
        "sentiment_label": ["POSITIVE", "NEGATIVE", "POSITIVE", "NEUTRAL", "POSITIVE", "POSITIVE", "NEGATIVE"],
        "vader_compound": [0.6, -0.4, 0.5, 0.0, 0.4, 0.7, -0.3],
        "toxicity": [0.02, 0.25, 0.01, 0.05, 0.15, 0.03, 0.20],
        "text": ["Great video! 🔥", "Bad analysis", "Loved the breakdown", "Interesting point", "Yes indeed", "Awesome work", "Horrible"],
        "published_at": ["2026-03-01T12:00:00Z"] * 7
    })
    profile = engine.build_profile_from_slice("c_synth", "Synthetic Cohort", "test", 2, synthetic_df)
    assert profile.total_comments == 7
    assert profile.unique_authors == 3
    assert profile.comments_per_video == 3.5
    assert profile.reply_ratio == pytest.approx(3 / 7, abs=0.01)
    assert profile.champion_author_ratio > 0.0
    assert len(profile.top_emojis) > 0
    for dim in RADAR_DIMENSIONS:
        assert 0.0 <= profile.radar_scores[dim] <= 100.0


def test_compute_pairwise_overlap_synthetic():
    engine = CompetitiveIntelligenceEngine()
    df_a = pd.DataFrame({
        "author_channel_id": ["u1", "u2", "u3"],
        "author_display_name": ["User1", "User2", "User3"],
        "vader_compound": [0.5, 0.2, -0.1],
        "toxicity": [0.05, 0.02, 0.20],
    })
    df_b = pd.DataFrame({
        "author_channel_id": ["u2", "u3", "u4"],
        "author_display_name": ["User2", "User3", "User4"],
        "vader_compound": [0.1, -0.4, 0.6],
        "toxicity": [0.08, 0.35, 0.01],
    })
    prof_a = engine.build_profile_from_slice("pa", "Prof A", "test", 1, df_a)
    prof_b = engine.build_profile_from_slice("pb", "Prof B", "test", 1, df_b)

    overlap = engine.compute_pairwise_overlap(prof_a, prof_b, df_a, df_b)
    assert overlap.shared_authors == 2  # u2, u3
    # Jaccard = 2 / 4 = 0.5
    assert overlap.jaccard_similarity == pytest.approx(0.5, abs=0.01)
    assert len(overlap.top_migrated_commenters) == 2


def test_compare_channels_or_cohorts():
    engine = CompetitiveIntelligenceEngine()
    channels = engine.discover_available_channels()
    if len(channels) >= 2:
        cids = [channels[0]["channel_id"], channels[1]["channel_id"]]
        report = engine.compare_channels(cids)
        assert len(report.profiles) == 2
        assert len(report.pairwise_overlaps) == 1
    else:
        report = engine.compare_cohorts_by_quarter()
        assert len(report.profiles) >= 2


def test_export_report(tmp_path):
    engine = CompetitiveIntelligenceEngine()
    report = engine.generate_mock_report()

    # JSON export
    json_path = tmp_path / "comparative_report.json"
    ok, msg = engine.export_report(report, json_path, export_format="json")
    assert ok is True
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert "profiles" in loaded
        assert len(loaded["pairwise_overlaps"]) == 2

    # CSV export
    csv_path = tmp_path / "comparative_profiles.csv"
    ok, msg = engine.export_report(report, csv_path, export_format="csv")
    assert ok is True
    assert csv_path.exists()
    assert csv_path.stat().st_size > 100


def test_cli_execution(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ytint-compare", "--mock"])
    comparator_main()
    captured = capsys.readouterr()
    assert "Multi-Channel & Playlist Competitive Intelligence Engine" in captured.out
    assert "Benchmark Profiles" in captured.out
    assert "Normalized Radar Dimensions" in captured.out
    assert "Audience Overlap & Commenter Migration" in captured.out


def test_cli_list_channels(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ytint-compare", "--list-channels"])
    comparator_main()
    captured = capsys.readouterr()
    assert "Discovered Channels" in captured.out
