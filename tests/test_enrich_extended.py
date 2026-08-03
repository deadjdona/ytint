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
