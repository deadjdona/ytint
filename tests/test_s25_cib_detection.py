import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pytest
import pandas as pd
from pipeline.s25_cib_detection import detect_cib_rings, run_cib_detection

def test_detect_cib_rings_synthetic():
    # Construct 2 synchronized accounts commenting within 30s across 2 videos
    df = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3", "c4", "c5"],
        "video_id": ["v1", "v1", "v2", "v2", "v3"],
        "author_channel_id": ["bot_A", "bot_B", "bot_A", "bot_B", "human_C"],
        "published_at": [
            "2026-01-01 10:00:00",
            "2026-01-01 10:00:25", # 25s apart on v1
            "2026-01-02 12:00:00",
            "2026-01-02 12:00:40", # 40s apart on v2
            "2026-01-03 14:00:00"
        ],
        "text_original": ["spam 1", "spam 2", "spam 3", "spam 4", "normal user"]
    })

    df_rings, df_flagged = detect_cib_rings(df, time_window_seconds=60, min_cooccurrences=2)

    assert not df_rings.empty
    assert len(df_rings) == 1
    assert df_rings.iloc[0]["ring_size"] == 2
    assert "bot_A" in df_rings.iloc[0]["member_channel_ids"]
    assert "bot_B" in df_rings.iloc[0]["member_channel_ids"]
    assert len(df_flagged) == 4

def test_run_cib_detection_execution(monkeypatch, tmp_path):
    interim_dir = tmp_path / "interim"
    output_dir = tmp_path / "output"
    interim_dir.mkdir()
    output_dir.mkdir()

    df = pd.DataFrame({
        "comment_id": ["c1", "c2"],
        "video_id": ["v1", "v1"],
        "author_channel_id": ["bot1", "bot2"],
        "published_at": ["2026-01-01 10:00:00", "2026-01-01 10:00:10"],
        "text_original": ["hi", "hi"]
    })
    df.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    fake_config = {
        "paths": {
            "interim_dir": str(interim_dir),
            "output_dir": str(output_dir)
        }
    }
    monkeypatch.setattr("pipeline.s25_cib_detection.load_config", lambda: fake_config)

    run_cib_detection()

    assert (output_dir / "cib_rings.parquet").exists()
    assert (output_dir / "cib_coordinated_comments.parquet").exists()
