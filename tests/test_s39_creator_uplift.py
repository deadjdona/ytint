import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pytest
import pandas as pd
from pipeline.s39_creator_uplift import compute_creator_causal_uplift, run_creator_uplift

def test_compute_creator_causal_uplift_synthetic():
    df = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3", "c4"],
        "video_id": ["v1", "v1", "v2", "v2"],
        "author_channel_id": ["a1", "a2", "a3", "a4"],
        "like_count": [150, 2, 200, 5],
        "reply_count": [25, 0, 30, 1],
        "minutes_since_upload": [15, 300, 20, 500],
        "vader_compound": [0.8, 0.1, 0.9, 0.0],
        "toxicity": [0.02, 0.25, 0.01, 0.30]
    })

    df_summary, df_threads = compute_creator_causal_uplift(df)

    assert not df_summary.empty
    assert len(df_summary) >= 4
    
    reply_row = df_summary[df_summary["dimension"] == "Thread Reply Volume"].iloc[0]
    assert reply_row["treated_mean"] > reply_row["control_mean"]
    assert reply_row["relative_lift_pct"] > 0

    assert not df_threads.empty
    assert len(df_threads) == 2

def test_run_creator_uplift_execution(monkeypatch, tmp_path):
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir()
    output_dir.mkdir()

    df = pd.DataFrame({
        "comment_id": ["c1", "c2"],
        "video_id": ["v1", "v1"],
        "author_channel_id": ["a1", "a2"],
        "like_count": [50, 1],
        "reply_count": [10, 0],
        "minutes_since_upload": [30, 100],
        "vader_compound": [0.5, 0.2],
        "toxicity": [0.05, 0.15]
    })
    df.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    fake_config = {
        "paths": {
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir)
        }
    }
    monkeypatch.setattr("pipeline.s39_creator_uplift.load_config", lambda: fake_config)

    run_creator_uplift()

    assert (output_dir / "creator_causal_uplift.parquet").exists()
    assert (output_dir / "creator_intervention_threads.parquet").exists()
