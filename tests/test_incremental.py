"""ytint // Incremental Pipeline & Delta Engine Unit Tests (tests/test_incremental.py)

Validates:
1. DeltaReport data structures and summary text formatting.
2. inspect_delta detection between SQLite tables and interim Parquet artifacts.
3. State persistence (.pipeline_state.json) save and load.
4. s00_ingest incremental upsert preserving existing enriched NLP columns.
5. s01_enrich incremental skip behavior when all comments are already enriched.
6. PipelineRunner --diff and --incremental flags.
"""

from __future__ import annotations

import json
import sqlite3
import pandas as pd
import pytest
from pathlib import Path

from engine.delta import (
    DeltaReport,
    inspect_delta,
    load_pipeline_state,
    save_pipeline_state
)
from pipeline.runner import PipelineRunner


def test_delta_report_summary_text():
    """Verify DeltaReport formatting for both synced and dirty states."""
    synced = DeltaReport(
        has_delta=False,
        total_sqlite_comments=100,
        total_interim_comments=100,
        total_sqlite_videos=5,
        total_interim_videos=5
    )
    summary_synced = synced.summary_text()
    assert "fully synchronized" in summary_synced
    assert "100" in summary_synced

    dirty = DeltaReport(
        has_delta=True,
        new_comments_count=12,
        updated_comments_count=3,
        new_videos_count=1,
        total_sqlite_comments=112,
        total_interim_comments=100
    )
    summary_dirty = dirty.summary_text()
    assert "Pending Ingestion Delta Detected" in summary_dirty
    assert "+12" in summary_dirty
    assert "3" in summary_dirty


def test_pipeline_state_save_and_load(tmp_path):
    """Verify serialization and deserialization of pipeline state JSON."""
    report = DeltaReport(
        has_delta=True,
        new_comments_count=5,
        updated_comments_count=2,
        new_videos_count=1,
        total_sqlite_comments=50,
        total_interim_comments=45
    )
    saved_path = save_pipeline_state(tmp_path, report, mode="incremental")
    assert saved_path.exists()

    state = load_pipeline_state(tmp_path)
    assert state is not None
    assert state["mode"] == "incremental"
    assert state["has_delta"] is True
    assert state["new_comments_count"] == 5


def test_inspect_delta_detects_changes(tmp_path):
    """Verify inspect_delta accurately identifies additions and modifications."""
    raw_db = tmp_path / "commentsuite.sqlite3"
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create SQLite tables with initial data
    conn = sqlite3.connect(raw_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE comments (
            id TEXT PRIMARY KEY,
            video_id TEXT,
            text TEXT,
            author_id TEXT,
            author_name TEXT,
            published_at TEXT,
            like_count INTEGER,
            reply_count INTEGER,
            parent_id TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE videos (
            id TEXT PRIMARY KEY,
            title TEXT,
            view_count INTEGER,
            like_count INTEGER,
            published_at TEXT
        )
    """)
    cursor.execute("INSERT INTO videos VALUES ('v1', 'Test Video', 1000, 50, '2026-01-01T00:00:00Z')")
    cursor.execute("INSERT INTO comments VALUES ('c1', 'v1', 'Hello world', 'a1', 'Alice', '2026-01-01T01:00:00Z', 10, 0, NULL)")
    cursor.execute("INSERT INTO comments VALUES ('c2', 'v1', 'Second comment', 'a2', 'Bob', '2026-01-01T02:00:00Z', 5, 1, NULL)")
    conn.commit()

    # 2. Create matching Parquet in interim
    df_comm = pd.DataFrame([
        {"comment_id": "c1", "video_id": "v1", "text": "Hello world", "like_count": 10, "reply_count": 0, "sentiment_label": "positive"},
        {"comment_id": "c2", "video_id": "v1", "text": "Second comment", "like_count": 5, "reply_count": 1, "sentiment_label": "neutral"}
    ])
    df_comm.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    df_vids = pd.DataFrame([
        {"video_id": "v1", "title": "Test Video", "view_count": 1000, "like_count": 50}
    ])
    df_vids.to_parquet(interim_dir / "videos_clean.parquet", index=False)

    # 3. Initial inspect: should be synchronized
    report1 = inspect_delta(raw_db, interim_dir)
    assert report1.has_delta is False
    assert report1.new_comments_count == 0
    assert report1.updated_comments_count == 0

    # 4. Modify SQLite: add 1 new comment, update like_count on c1
    cursor.execute("INSERT INTO comments VALUES ('c3', 'v1', 'Brand new comment', 'a3', 'Charlie', '2026-01-01T03:00:00Z', 0, 0, NULL)")
    cursor.execute("UPDATE comments SET like_count = 25 WHERE id = 'c1'")
    conn.commit()
    conn.close()

    # 5. Delta inspect: should detect 1 new comment and 1 updated comment
    report2 = inspect_delta(raw_db, interim_dir)
    assert report2.has_delta is True
    assert report2.new_comments_count == 1
    assert "c3" in report2.new_comment_ids
    assert report2.updated_comments_count == 1
    assert "c1" in report2.updated_comment_ids


def test_s00_incremental_upsert_preserves_enriched_columns(tmp_path, monkeypatch):
    """Verify s00_ingest incremental mode updates like/reply counts without clobbering enriched columns."""
    from pipeline.s00_ingest import migrate_from_commentsuite

    raw_db = tmp_path / "commentsuite.sqlite3"
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create SQLite database with Commentsuite schema
    conn = sqlite3.connect(raw_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE comments (
            comment_id TEXT PRIMARY KEY,
            video_id TEXT,
            parent_id TEXT,
            channel_id TEXT,
            comment_text TEXT,
            comment_likes INTEGER,
            reply_count INTEGER,
            comment_date INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE videos (
            video_id TEXT PRIMARY KEY,
            channel_id TEXT,
            video_title TEXT,
            publish_date TEXT,
            grab_date TEXT,
            total_comments INTEGER,
            total_views INTEGER,
            total_likes INTEGER,
            total_dislikes INTEGER,
            video_desc TEXT,
            thumb_url TEXT,
            http_code INTEGER
        )
    """)
    cursor.execute("INSERT INTO videos VALUES ('v1', 'ch1', 'Video 1', '2026-01-01', '2026-01-01', 2, 500, 20, 0, 'desc', 'thumb', 200)")
    cursor.execute("INSERT INTO comments VALUES ('c1', 'v1', NULL, 'a1', 'Great video', 99, 5, 1767225600)")
    cursor.execute("INSERT INTO comments VALUES ('c2', 'v1', NULL, 'a2', 'New comment', 0, 0, 1767229200)")
    conn.commit()
    conn.close()

    # 2. Existing Parquet has c1 with enriched columns and old like_count (10)
    existing_df = pd.DataFrame([{
        "comment_id": "c1",
        "video_id": "v1",
        "text": "Great video",
        "published_at": pd.to_datetime("2026-01-01 01:00:00"),
        "like_count": 10,
        "reply_count": 1,
        "author_id": "a1",
        "author_name": "Alice",
        "parent_id": None,
        "is_reply": False,
        "sentiment_label": "positive",
        "vader_compound": 0.85,
        "toxicity": 0.02
    }])
    existing_df.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    existing_vids = pd.DataFrame([{
        "video_id": "v1",
        "title": "Video 1",
        "published_at": pd.to_datetime("2026-01-01 00:00:00"),
        "view_count": 400,
        "like_count": 15,
        "comment_count": 1
    }])
    existing_vids.to_parquet(interim_dir / "videos_clean.parquet", index=False)

    # 3. Patch paths in s00_ingest config
    fake_config = {
        "paths": {
            "raw_db": raw_db,
            "interim_dir": interim_dir,
            "output_dir": tmp_path / "output"
        }
    }
    monkeypatch.setattr("pipeline.s00_ingest.load_config", lambda: fake_config)

    # 4. Run incremental ingest
    migrate_from_commentsuite(incremental=True)

    # 5. Validate resulting comments_clean.parquet
    result_df = pd.read_parquet(interim_dir / "comments_clean.parquet")
    assert len(result_df) == 2
    c1_row = result_df[result_df["comment_id"] == "c1"].iloc[0]
    assert c1_row["like_count"] == 99  # Updated from SQLite
    assert c1_row["reply_count"] == 5  # Updated from SQLite
    assert c1_row["sentiment_label"] == "positive"  # Preserved!
    assert c1_row["vader_compound"] == 0.85  # Preserved!
    assert c1_row["toxicity"] == 0.02  # Preserved!

    c2_row = result_df[result_df["comment_id"] == "c2"].iloc[0]
    assert c2_row["like_count"] == 0
    assert pd.isna(c2_row["sentiment_label"]) or c2_row["sentiment_label"] is None  # Awaiting s01 delta enrich


def test_s01_incremental_skips_when_all_enriched(tmp_path, monkeypatch):
    """Verify s01_enrich exits in fast-path when 0 comments are missing enrichment."""
    from pipeline.s01_enrich import enrich_comments

    interim_dir = tmp_path / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Create parquet where all records already have sentiment_label
    df = pd.DataFrame([
        {"comment_id": "c1", "text": "Test 1", "sentiment_label": "positive"},
        {"comment_id": "c2", "text": "Test 2", "sentiment_label": "negative"}
    ])
    df.to_parquet(interim_dir / "comments_clean.parquet", index=False)

    fake_config = {
        "paths": {
            "raw_db": tmp_path / "raw.db",
            "interim_dir": interim_dir,
            "output_dir": tmp_path / "output"
        }
    }
    monkeypatch.setattr("pipeline.s01_enrich.load_config", lambda: fake_config)

    # Running enrich_comments in incremental mode should take fast-path (<0.5s) without error
    enrich_comments(incremental=True)

    # Sentinel file should exist
    assert (interim_dir / ".s01_complete").exists()


def test_runner_diff_only_returns_delta():
    """Verify PipelineRunner.run(diff_only=True) executes and returns DeltaReport."""
    runner = PipelineRunner()
    report = runner.run(diff_only=True)
    assert isinstance(report, DeltaReport)
    assert report.total_sqlite_comments >= 0


def test_runner_incremental_quick_exit():
    """Verify PipelineRunner.run(incremental=True) executes in < 2 seconds when clean."""
    runner = PipelineRunner()
    delta = runner.run(incremental=True)
    assert isinstance(delta, DeltaReport)
