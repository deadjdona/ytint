"""Unit tests for extended pipeline stages (s09, s36, s41 - s51)."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import pandas as pd
import numpy as np

from pipeline.s09_thread_width import run_thread_width
from pipeline.s36_arrival_speed import run_arrival_speed
from pipeline.s41_tfidf_keywords import run_tfidf_keywords
from pipeline.s42_polarity_engagement import run_polarity_engagement
from pipeline.s43_emoji_signatures import run_emoji_signatures
from pipeline.s44_sentiment_anomalies import run_sentiment_anomalies
from pipeline.s45_corpus_quality import run_corpus_quality
from pipeline.s46_thread_topic_drift import run_thread_topic_drift
from pipeline.s47_slang_lexicon import run_slang_lexicon
from pipeline.s48_bowtie_concentration import run_bowtie_concentration
from pipeline.s49_series_creator_sentiment import run_series_creator_sentiment
from pipeline.s50_cross_modal_reactions import run_cross_modal_reactions
from pipeline.s51_topic_injection import run_topic_injection


@pytest.fixture
def mock_pipeline_dirs(tmp_path):
    """Create isolated temporary input/output directory structure with base clean datasets."""
    data_dir = tmp_path / "data"
    interim_dir = data_dir / "interim"
    output_dir = data_dir / "output"
    interim_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    # Base comments clean dataframe
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=50, freq="min", tz="UTC")
    df_comments = pd.DataFrame({
        "comment_id": [f"c{i}" for i in range(50)],
        "parent_id": [None if i % 3 == 0 else f"c{(i // 3) * 3}" for i in range(50)],
        "video_id": ["v1"] * 25 + ["v2"] * 25,
        "author_channel_id": [f"author_{i % 10}" for i in range(50)],
        "author_display_name": [f"User {i % 10}" for i in range(50)],
        "text": [
            "This video is absolute cringe tbh but lowkey hilarious lol! 😂" if i % 2 == 0
            else "Awesome episode, creator did amazing work. What timestamp is that song?"
            for i in range(50)
        ],
        "published_at": timestamps,
        "like_count": [i * 3 for i in range(50)],
        "reply_count": [5 if i % 3 == 0 else 0 for i in range(50)],
        "sentiment_compound": [0.8 if i % 2 == 0 else -0.5 for i in range(50)],
        "language": ["en" if i % 2 == 0 else "ru" for i in range(50)],
        "extracted_timestamp": [120 if i % 4 == 0 else None for i in range(50)],
        "topic_id": [i % 5 for i in range(50)],
        "topic": [f"Topic_{i % 5}" for i in range(50)]
    })
    df_comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    # Videos metadata
    df_videos = pd.DataFrame({
        "video_id": ["v1", "v2"],
        "title": ["Epic Series Part 1: The Beginning", "Standalone Guide to Python"],
        "published_at": [pd.Timestamp("2026-01-01 09:00:00", tz="UTC"), pd.Timestamp("2026-01-01 09:00:00", tz="UTC")],
        "total_comments": [25, 25],
        "total_likes": [300, 300],
        "channel_id": ["creator_chan", "creator_chan"]
    })
    df_videos.to_parquet(interim_dir / "videos_clean.parquet", index=False)
    df_videos.to_parquet(output_dir / "videos_final.parquet", index=False)

    return interim_dir, output_dir


def test_s09_thread_width_and_attention_transfer(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s09_thread_width.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_thread_width()
    assert (output_dir / "thread_width_dist.parquet").exists()
    assert (output_dir / "attention_transfer.parquet").exists()
    df_att = pd.read_parquet(output_dir / "attention_transfer.parquet")
    assert not df_att.empty
    assert "attention_transfer_ratio" in df_att.columns


def test_s36_arrival_speed_and_minute_curve(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s36_arrival_speed.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_arrival_speed()
    assert (output_dir / "arrival_speed.parquet").exists()
    assert (output_dir / "minute_arrival_curve.parquet").exists()
    df_curve = pd.read_parquet(output_dir / "minute_arrival_curve.parquet")
    assert not df_curve.empty
    assert "cumulative_comments" in df_curve.columns


def test_s41_tfidf_keywords(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s41_tfidf_keywords.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_tfidf_keywords()
    assert (output_dir / "tfidf_keywords.parquet").exists()


def test_s42_polarity_engagement(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s42_polarity_engagement.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_polarity_engagement()
    assert (output_dir / "polarity_engagement.parquet").exists()


def test_s43_emoji_signatures(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s43_emoji_signatures.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_emoji_signatures()
    assert (output_dir / "emoji_signatures.parquet").exists()


def test_s44_sentiment_anomalies(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s44_sentiment_anomalies.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_sentiment_anomalies()
    assert (output_dir / "sentiment_anomalies.parquet").exists()


def test_s45_corpus_quality(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s45_corpus_quality.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_corpus_quality()
    assert (output_dir / "corpus_quality.parquet").exists()


def test_s46_thread_topic_drift(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s46_thread_topic_drift.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_thread_topic_drift()
    assert (output_dir / "thread_topic_drift.parquet").exists()


def test_s47_slang_lexicon(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s47_slang_lexicon.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_slang_lexicon()
    assert (output_dir / "slang_lexicon_frequency.parquet").exists()
    df_slang = pd.read_parquet(output_dir / "slang_lexicon_frequency.parquet")
    assert not df_slang.empty
    assert "slang_term" in df_slang.columns
    assert "total_occurrences" in df_slang.columns


def test_s48_bowtie_concentration(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s48_bowtie_concentration.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_bowtie_concentration()
    assert (output_dir / "network_bowtie_structure.parquet").exists()
    assert (output_dir / "top_k_concentration.parquet").exists()
    df_bt = pd.read_parquet(output_dir / "network_bowtie_structure.parquet")
    assert not df_bt.empty
    assert "component" in df_bt.columns


def test_s49_series_creator_sentiment(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s49_series_creator_sentiment.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_series_creator_sentiment()
    assert (output_dir / "series_vs_standalone.parquet").exists()
    assert (output_dir / "creator_sentiment_polarity.parquet").exists()
    df_pol = pd.read_parquet(output_dir / "creator_sentiment_polarity.parquet")
    assert not df_pol.empty
    assert "creator_pos_neg_ratio" in df_pol.columns


def test_s50_cross_modal_reactions(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s50_cross_modal_reactions.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_cross_modal_reactions()
    assert (output_dir / "cross_modal_scene_reactions.parquet").exists()
    assert (output_dir / "spoiler_detections.parquet").exists()


def test_s51_topic_injection(mock_pipeline_dirs, monkeypatch):
    interim_dir, output_dir = mock_pipeline_dirs
    monkeypatch.setattr("pipeline.s51_topic_injection.load_config", lambda: {
        "paths": {"interim_dir": str(interim_dir), "output_dir": str(output_dir)}
    })
    run_topic_injection()
    assert (output_dir / "topic_injection_anomalies.parquet").exists()
