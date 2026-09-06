"""Tests for Neural Semantic Vector Search & Feedback Cluster Engine (src/engine/semantic_search.py)."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from engine.semantic_search import FeedbackCluster, SearchResultItem, SearchResultSet, SemanticSearchEngine


@pytest.fixture
def mock_search_env(tmp_path):
    """Creates a temporary workspace with synthetic embeddings and comments metadata."""
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    n_samples = 40
    dim = 384

    # 1. Synthetic Embeddings
    rng = np.random.RandomState(42)
    embs = rng.randn(n_samples, dim).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-12)
    np.save(interim_dir / "topic_embeddings.npy", embs)

    # 2. Synthetic Comments
    comments_data = []
    tiers = ["Champions", "Loyalists", "Regular", "Casual", "Drive-by"]
    topics = [
        "Audio is buzzing and microphone has static noise",
        "Could you make a video explaining machine learning basics?",
        "Loved the animation at 04:20, fantastic editing",
        "The background music is way too loud, hard to hear voice",
        "Please do a part two on this topic next week",
        "Disagreed with your second argument, it was inaccurate",
        "Sound quality on this upload was significantly better",
        "Subscribed since 2021, amazing work as always"
    ]

    for i in range(n_samples):
        comments_data.append({
            "comment_id": f"c_{i:03d}",
            "video_id": f"vid_{i % 3}",
            "parent_id": None,
            "author_channel_id": f"UC_{i:03d}",
            "author_display_name": f"User_{i}",
            "text": topics[i % len(topics)],
            "like_count": (i * 3) % 25,
            "reply_count": i % 5,
            "published_at": "2026-02-15T12:00:00Z",
            "vader_compound": 0.4 if i % 2 == 0 else -0.3,
            "toxicity": 0.05 if i % 2 == 0 else 0.45
        })

    df_comments = pd.DataFrame(comments_data)
    df_comments.to_parquet(interim_dir / "comments_clean.parquet")

    # 3. Synthetic Videos
    df_vids = pd.DataFrame([
        {"video_id": "vid_0", "title": "Episode 1: The Beginning"},
        {"video_id": "vid_1", "title": "Episode 2: Deep Dive"},
        {"video_id": "vid_2", "title": "Episode 3: The Finale"}
    ])
    df_vids.to_parquet(interim_dir / "videos_clean.parquet")

    # 4. Synthetic Authors
    authors_data = [
        {"author_channel_id": f"UC_{i:03d}", "rfm_tier": tiers[i % len(tiers)]}
        for i in range(n_samples)
    ]
    pd.DataFrame(authors_data).to_parquet(output_dir / "authors_final.parquet")

    return interim_dir, output_dir


def test_search_engine_init(tmp_path):
    """Verifies SemanticSearchEngine initialization."""
    engine = SemanticSearchEngine(interim_dir=tmp_path / "int", output_dir=tmp_path / "out")
    assert engine.embeddings_path == tmp_path / "int" / "topic_embeddings.npy"


def test_mock_query_encoding():
    """Verifies mock query vector normalization and dimension."""
    engine = SemanticSearchEngine()
    vec = engine.encode_query("microphone sound", use_mock=True)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_vector_similarity_search_synthetic(mock_search_env):
    """Verifies vector dot product search and top-K ranking."""
    interim_dir, output_dir = mock_search_env
    engine = SemanticSearchEngine(interim_dir=interim_dir, output_dir=output_dir)

    results = engine.search("audio microphone issues", top_k=10, use_mock=True)
    assert isinstance(results, SearchResultSet)
    assert results.results_count == 10
    assert results.total_searched == 40
    assert results.query_latency_ms >= 0.0

    # Ensure sorted by similarity score descending
    scores = [r.similarity_score for r in results.results]
    assert scores == sorted(scores, reverse=True)

    # Check first item fields
    first = results.results[0]
    assert isinstance(first, SearchResultItem)
    assert first.video_title.startswith("Episode")
    assert first.rfm_tier in ["Champions", "Loyalists", "Regular", "Casual", "Drive-by"]
    assert len(first.text) > 0


def test_metadata_filtering(mock_search_env):
    """Verifies metadata constraints (likes, loyalty tier, video ID)."""
    interim_dir, output_dir = mock_search_env
    engine = SemanticSearchEngine(interim_dir=interim_dir, output_dir=output_dir)

    # 1. Min likes filter
    res_likes = engine.search("audio", top_k=20, min_likes=10, use_mock=True)
    assert all(r.like_count >= 10 for r in res_likes.results)

    # 2. Loyalty tier filter
    res_tier = engine.search("audio", top_k=20, loyalty_tier="Champions", use_mock=True)
    assert all(r.rfm_tier == "Champions" for r in res_tier.results)

    # 3. Video ID filter
    res_vid = engine.search("audio", top_k=20, video_id="vid_1", use_mock=True)
    assert all(r.video_id == "vid_1" for r in res_vid.results)


def test_feedback_clustering(mock_search_env):
    """Verifies grouping of search results into thematic clusters."""
    interim_dir, output_dir = mock_search_env
    engine = SemanticSearchEngine(interim_dir=interim_dir, output_dir=output_dir)

    results = engine.search("sound editing animation", top_k=20, use_mock=True)
    clusters = engine.cluster_feedback(results, n_clusters=3)

    assert isinstance(clusters, list)
    assert len(clusters) > 0
    total_clustered = sum(c.comment_count for c in clusters)
    assert total_clustered == len(results.results)

    for c in clusters:
        assert isinstance(c, FeedbackCluster)
        assert len(c.theme_label) > 0
        assert len(c.exemplar_quote) > 0
        assert c.comment_count >= 1


def test_serialization(mock_search_env):
    """Verifies DataFrame and JSON serialization of search results."""
    interim_dir, output_dir = mock_search_env
    engine = SemanticSearchEngine(interim_dir=interim_dir, output_dir=output_dir)

    results = engine.search("requests", top_k=5, use_mock=True)
    engine.cluster_feedback(results, n_clusters=2)

    df = results.to_dataframe()
    assert len(df) == 5
    assert "video_title" in df.columns
    assert "similarity_score" in df.columns

    json_str = results.to_json()
    data = json.loads(json_str)
    assert data["results_count"] == 5
    assert len(data["clusters"]) > 0


def test_empty_corpus_resilience(tmp_path):
    """Verifies search handles missing or empty corpus files without crashing."""
    engine = SemanticSearchEngine(interim_dir=tmp_path / "empty_int", output_dir=tmp_path / "empty_out")
    results = engine.search("test query", top_k=10, use_mock=True)

    assert results.results_count == 0
    assert results.total_searched == 0
    assert results.to_dataframe().empty
    assert engine.cluster_feedback(results) == []
