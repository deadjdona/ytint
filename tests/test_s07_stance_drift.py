import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pytest
import pandas as pd
from pipeline.s07_stance_drift import classify_comment_stance, compute_stance_analysis, run_stance_analysis

def test_classify_comment_stance():
    assert classify_comment_stance("I completely agree with this analysis, awesome work!") == "Favor"
    assert classify_comment_stance("Полностью согласен, автор молодец!") == "Favor"
    assert classify_comment_stance("This is absolute nonsense and fake lies!", -0.8, 0.7) == "Against"
    assert classify_comment_stance("Это полный бред и вранье!", -0.8, 0.7) == "Against"
    assert classify_comment_stance("What camera settings did you use here?", 0.0, 0.0) == "Neutral"

def test_compute_stance_analysis_synthetic():
    df = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "video_id": ["v1", "v1", "v1", "v1", "v1", "v1"],
        "author_channel_id": ["a1", "a2", "a3", "a4", "a5", "a6"],
        "parent_id": ["", "c1", "c1", "", "c4", "c4"],
        "text": [
            "Awesome video, totally agree!",
            "You are wrong, this is fake!",
            "I agree with the first point",
            "Neutral observation about timestamps",
            "This makes no sense, liar",
            "Great job, loved it"
        ],
        "vader_compound": [0.8, -0.6, 0.5, 0.0, -0.7, 0.9],
        "toxicity": [0.01, 0.65, 0.02, 0.05, 0.70, 0.01]
    })

    df_summary, df_drift, df_threads = compute_stance_analysis(df)

    assert not df_summary.empty
    assert len(df_summary) == 1
    row = df_summary.iloc[0]
    assert row["total_comments"] == 6
    assert row["favor_pct"] > 0
    assert row["against_pct"] > 0
    assert row["polarization_index"] > 0

    assert not df_drift.empty
    assert len(df_drift) >= 2

def test_run_stance_analysis_execution(monkeypatch, tmp_path):
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir()
    output_dir.mkdir()

    df = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "video_id": ["v1", "v1", "v1", "v1", "v1", "v1"],
        "author_channel_id": ["a1", "a2", "a3", "a4", "a5", "a6"],
        "parent_id": ["", "c1", "", "c3", "", "c5"],
        "text": ["agree", "disagree", "agree", "disagree", "neutral", "agree"],
        "vader_compound": [0.5, -0.5, 0.5, -0.5, 0.0, 0.5],
        "toxicity": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0]
    })
    df.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    fake_config = {
        "paths": {
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir)
        }
    }
    monkeypatch.setattr("pipeline.s07_stance_drift.load_config", lambda: fake_config)

    run_stance_analysis()

    assert (output_dir / "stance_summary.parquet").exists()
    assert (output_dir / "stance_depth_drift.parquet").exists()
    assert (output_dir / "polarized_threads.parquet").exists()
