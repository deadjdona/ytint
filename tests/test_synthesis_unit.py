"""Unit tests for Stage 04 Synthesis (s04_synthesis.py)."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd

from pipeline.s40_synthesis import compile_ui_metrics, get_project_root


def test_get_project_root():
    """Verify get_project_root resolves valid workspace anchor."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert (root / "src").is_dir()


def test_compile_ui_metrics_mock(tmp_path, monkeypatch):
    """Test compile_ui_metrics synthesis with mock topic and comment datasets."""
    data_dir = tmp_path / "data"
    interim_dir = data_dir / "interim"
    output_dir = data_dir / "output"
    interim_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    df_comments = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3"],
        "topic": [1, 1, 2],
        "like_count": [10, 20, 5],
        "sentiment_label": ["positive", "negative", "neutral"],
        "sentiment_confidence": [0.9, 0.8, 0.95]
    })
    df_comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    df_topics = pd.DataFrame({
        "Topic": [1, 2],
        "Name": ["Cluster 1", "Cluster 2"],
        "Count": [2, 1]
    })
    df_topics.to_parquet(output_dir / "topic_metadata.parquet", index=False)

    monkeypatch.setattr("pipeline.s40_synthesis.get_project_root", lambda: tmp_path)

    compile_ui_metrics()

    result_topics = pd.read_parquet(output_dir / "topic_metadata.parquet")
    assert "total_likes" in result_topics.columns
    assert "pct_positive" in result_topics.columns
    assert "avg_confidence" in result_topics.columns
