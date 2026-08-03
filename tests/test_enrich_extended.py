"""Unit tests for s01_enrich.py lexical richness and enrichment functions."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import pandas as pd
from pipeline.s01_enrich import (
    compute_mattr, 
    compute_mtld, 
    compute_yules_k, 
    calculate_linguistic_features
)


def test_compute_mattr():
    """Verify MATTR moving average type-token ratio calculation."""
    tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] * 10
    mattr = compute_mattr(tokens, window_size=20)
    assert 0.0 <= mattr <= 1.0

    # Short tokens list smaller than window
    short_tokens = ["hello", "world"]
    mattr_short = compute_mattr(short_tokens, window_size=50)
    assert mattr_short == 1.0


def test_compute_mtld():
    """Verify MTLD lexical diversity calculation."""
    tokens = ["cat", "dog", "bird", "fish", "elephant", "lion", "tiger"] * 15
    mtld = compute_mtld(tokens, threshold=0.72)
    assert mtld > 0

    # Empty list fallback
    assert compute_mtld([]) == 0.0


def test_compute_yules_k():
    """Verify Yule's K vocabulary richness calculation."""
    tokens = ["alpha", "beta", "gamma", "delta", "alpha", "beta"]
    k = compute_yules_k(tokens)
    assert isinstance(k, float)
    assert k >= 0.0

    # Single token list fallback
    assert compute_yules_k(["test"]) == 0.0


def test_calculate_linguistic_features():
    """Verify linguistic feature dictionary extraction from comment text."""
    res = calculate_linguistic_features("Hello world! This is a test comment.")
    assert "word_count" in res
    assert "all_caps_ratio" in res
    assert "mattr" in res
    assert "mtld" in res
    assert "yules_k" in res


def test_list_dataset():
    """Verify ListDataset PyTorch Dataset wrapper."""
    from pipeline.s01_enrich import ListDataset
    data = ["text 1", "text 2", "text 3"]
    dataset = ListDataset(data)
    assert len(dataset) == 3
    assert dataset[0] == "text 1"
    assert dataset[2] == "text 3"


def test_gigatoken_bpe_encoding():
    """Verify gigatoken BPE tokenizer functionality and batch encoding."""
    import gigatoken as gt
    tokenizer = gt.Tokenizer("openai-community/gpt2")
    assert tokenizer is not None
    encoded = tokenizer.encode("Hello world")
    assert len(encoded) > 0

    batch_encoded = tokenizer.encode_batch_list(["First comment", "Second comment"])
    assert len(batch_encoded) == 2
    assert isinstance(batch_encoded[0], list)


def test_calculate_linguistic_features_dataframe_no_duplicate_columns():
    """Verify DataFrame feature calculation prevents duplicate columns."""
    df_in = pd.DataFrame({
        'comment_id': ['c1', 'c2'],
        'text': ['Hello world', 'Testing duplicates'],
        'char_count': [11, 18],  # Pre-existing column
        'word_count': [2, 2]
    })
    res_df = calculate_linguistic_features(df_in)
    assert res_df.columns.has_duplicates is False
    assert len(res_df) == 2


def test_enrich_checkpoint_resumption(tmp_path, monkeypatch):
    """Verify chunk checkpoint creation and resumption logic."""
    from pipeline import s01_enrich

    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    comments_df = pd.DataFrame({
        'comment_id': [f"c{i}" for i in range(10)],
        'video_id': ['v1'] * 10,
        'parent_id': [None] * 10,
        'text': ['Hello world test comment'] * 10,
        'published_at': ['2026-01-01 10:00:00'] * 10,
        'like_count': [5] * 10
    })
    comments_file = interim_dir / "comments_clean.parquet"
    comments_df.to_parquet(comments_file, index=False)

    fake_config = {
        "paths": {
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir)
        },
        "stage_01_enrich": {
            "sentiment_model": "blanchefort/rubert-base-cased-sentiment",
            "batch_size": 32
        }
    }
    monkeypatch.setattr(s01_enrich, "load_config", lambda: fake_config)

    # Checkpoint dir test
    checkpoint_dir = interim_dir / "enrich_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = checkpoint_dir / "chunk_0.parquet"

    # Pre-create checkpoint chunk
    pre_chunk = comments_df.copy()
    pre_chunk['sentiment_label'] = 'neutral'
    pre_chunk['sentiment_confidence'] = 0.9
    pre_chunk['emotion_1'] = 'neutral'
    pre_chunk['emotion_2'] = 'neutral'
    pre_chunk['emotion_3'] = 'neutral'
    pre_chunk['toxicity'] = 0.01
    pre_chunk['severe_toxicity'] = 0.0
    pre_chunk['insult'] = 0.0
    pre_chunk['obscene'] = 0.0
    pre_chunk['char_count'] = 24
    pre_chunk['word_count'] = 4
    pre_chunk['bpe_token_count'] = 4
    pre_chunk.to_parquet(chunk_file, index=False)

    assert chunk_file.exists()

    # Run enrich_comments, should restore chunk_0 instantly from checkpoint
    s01_enrich.enrich_comments()
    
    assert comments_file.exists()
    final_df = pd.read_parquet(comments_file)
    assert len(final_df) == 10
    assert final_df.columns.has_duplicates is False
    assert 'sentiment_label' in final_df.columns

