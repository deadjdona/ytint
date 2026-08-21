import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pytest
import pandas as pd
from pipeline.s15_toxicity_contagion import analyze_toxicity_contagion, run_toxicity_contagion

def test_analyze_toxicity_contagion_synthetic():
    df = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3", "c4"],
        "video_id": ["v1", "v1", "v2", "v2"],
        "author_channel_id": ["troll_1", "user_2", "troll_1", "user_3"],
        "author_display_name": ["Troll", "Alice", "Troll", "Bob"],
        "toxicity": [0.85, 0.05, 0.90, 0.10],
        "reply_count": [15, 1, 20, 0],
        "like_count": [2, 10, 1, 5]
    })

    df_summary, df_catalysts, df_videos = analyze_toxicity_contagion(df, toxicity_threshold=0.5)

    assert not df_summary.empty
    assert df_summary.iloc[0]["total_toxic_comments"] == 2
    assert df_summary.iloc[0]["toxicity_reproduction_number_r0"] > 1.0

    assert not df_catalysts.empty
    top_troll = df_catalysts.iloc[0]
    assert top_troll["author_channel_id"] == "troll_1"
    assert top_troll["toxic_comments_count"] == 2
    assert "High Catalyst" in top_troll["catalyst_tier"]

    assert not df_videos.empty
    assert len(df_videos) == 2

def test_run_toxicity_contagion_execution(monkeypatch, tmp_path):
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir()
    output_dir.mkdir()

    df = pd.DataFrame({
        "comment_id": ["c1"],
        "video_id": ["v1"],
        "author_channel_id": ["a1"],
        "toxicity": [0.7],
        "reply_count": [5],
        "like_count": [1]
    })
    df.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    fake_config = {
        "paths": {
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir)
        }
    }
    monkeypatch.setattr("pipeline.s15_toxicity_contagion.load_config", lambda: fake_config)

    run_toxicity_contagion()

    assert (output_dir / "toxicity_contagion_summary.parquet").exists()
    assert (output_dir / "troll_catalysts.parquet").exists()
    assert (output_dir / "video_toxicity_contagion.parquet").exists()
