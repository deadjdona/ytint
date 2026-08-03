"""Unit tests for Stage 00 Ingestion (s00_ingest.py)."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import sqlite3
import pandas as pd

from pipeline.s00_ingest import parse_comment_dates, migrate_from_commentsuite
from engine.config_loader import load_config


def test_parse_comment_dates_seconds_and_ms():
    """Verify auto-detection of seconds vs milliseconds in integer timestamps."""
    # Test seconds scale
    sec_series = pd.Series([1600000000, 1600000100])
    parsed_sec = parse_comment_dates(sec_series)
    assert parsed_sec.dt.year.iloc[0] == 2020

    # Test milliseconds scale
    ms_series = pd.Series([1600000000000, 1600000100000])
    parsed_ms = parse_comment_dates(ms_series)
    assert parsed_ms.dt.year.iloc[0] == 2020

    # Test empty series
    empty_parsed = parse_comment_dates(pd.Series([], dtype=object))
    assert empty_parsed.empty


def test_migrate_from_commentsuite_mock_db(tmp_path, monkeypatch):
    """Test full SQLite ingestion pipeline on a synthetic SQLite database."""
    raw_db_file = tmp_path / "commentsuite.sqlite3"
    interim_dir = tmp_path / "interim"

    # Create dummy SQLite database
    conn = sqlite3.connect(raw_db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE comments (
            comment_id TEXT,
            video_id TEXT,
            parent_id TEXT,
            channel_id TEXT,
            comment_text TEXT,
            comment_likes INTEGER,
            comment_date INTEGER
        )
    """)
    cursor.execute("""
        INSERT INTO comments VALUES ('c1', 'v1', NULL, 'a1', 'Hello world', 5, 1600000000000)
    """)
    conn.commit()
    conn.close()

    # Mock load_config to point to temp directories
    mock_config = {
        "paths": {
            "raw_db": str(raw_db_file),
            "interim_dir": str(interim_dir),
            "output_dir": str(tmp_path / "output"),
        },
        "_root_dir": tmp_path
    }
    monkeypatch.setattr("pipeline.s00_ingest.load_config", lambda: mock_config)

    migrate_from_commentsuite()

    assert (interim_dir / "comments_clean.parquet").exists()
    assert (interim_dir / "videos_clean.parquet").exists()

    df_c = pd.read_parquet(interim_dir / "comments_clean.parquet")
    assert len(df_c) == 1
    assert df_c.iloc[0]["comment_id"] == "c1"
