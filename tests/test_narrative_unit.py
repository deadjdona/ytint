"""Unit tests for Stage 03 Narrative Synthesis (s03_narrative.py)."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd

from pipeline.s28_narrative import compile_narrative


def test_compile_narrative_mock_pipeline(tmp_path, monkeypatch):
    """Test compile_narrative execution on synthetic interim comments parquet file."""
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir()
    output_dir.mkdir()

    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df_comments = pd.DataFrame({
        "comment_id": [f"c{i}" for i in range(30)],
        "published_at": dates,
        "text": ["hello"] * 30
    })
    df_comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    mock_config = {
        "paths": {
            "raw_db": str(tmp_path / "raw.db"),
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir),
        },
        "stage_03_narrative": {
            "change_point_penalty": "auto",
            "z_threshold": 2.5
        },
        "_root_dir": tmp_path
    }
    monkeypatch.setattr("pipeline.s28_narrative.load_config", lambda: mock_config)

    compile_narrative()

    assert (output_dir / "historical_timeline.parquet").exists()
    assert (output_dir / "viral_events.parquet").exists()

    timeline_df = pd.read_parquet(output_dir / "historical_timeline.parquet")
    assert not timeline_df.empty
    assert "z_score" in timeline_df.columns
