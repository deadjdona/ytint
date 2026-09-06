"""Unit tests for the Live YouTube Data API v3 Ingest Connector (src/engine/youtube_api.py)."""

import pytest
import sqlite3
from pathlib import Path

from engine.youtube_api import (
    YouTubeClient,
    MockYouTubeClient,
    SQLiteCommentsuiteStore,
    ingest_youtube_data
)


def test_mock_client_resolve_channel():
    """Verify mock client resolves channel handle and returns uploads playlist."""
    client = MockYouTubeClient()
    ch = client.resolve_channel(handle="@3blue1brown")

    assert "channel_id" in ch
    assert "uploads_playlist_id" in ch
    assert ch["uploads_playlist_id"].startswith("UU_")
    assert ch["video_count"] > 0
    assert ch["title"].startswith("Intelligence Channel")


def test_mock_client_fetch_channel_videos():
    """Verify mock client fetches requested video item count."""
    client = MockYouTubeClient()
    videos = client.fetch_channel_videos("UU_test_uploads", max_videos=5)

    assert len(videos) == 5
    for v in videos:
        assert "video_id" in v
        assert "title" in v
        assert "published_at" in v


def test_mock_client_fetch_comment_threads():
    """Verify mock client retrieves top comments and threaded replies."""
    client = MockYouTubeClient()
    top_c, reps = client.fetch_comment_threads("mock_vid_001", max_comments=10)

    assert len(top_c) == 10
    assert len(reps) > 0

    for c in top_c:
        assert c["is_reply"] is False
        assert c["parent_id"] is None
        assert "comment_id" in c
        assert "comment_text" in c

    for r in reps:
        assert r["is_reply"] is True
        assert r["parent_id"] is not None


def test_sqlite_store_schema_and_upsert(tmp_path):
    """Verify SQLiteCommentsuiteStore creates tables and upserts records cleanly."""
    db_path = tmp_path / "test_commentsuite.sqlite3"
    store = SQLiteCommentsuiteStore(db_path)

    # Verify tables exist
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    assert "channels" in tables
    assert "videos" in tables
    assert "comments" in tables

    # Upsert channel
    store.upsert_channel({
        "channel_id": "UC_TEST_1",
        "title": "Test Channel",
        "thumb_url": "https://example.com/ch.jpg"
    })

    # Upsert videos
    store.upsert_videos([{
        "video_id": "vid_001",
        "channel_id": "UC_TEST_1",
        "published_at": "2026-08-01T12:00:00Z",
        "title": "Test Video 1",
        "total_comments": 10,
        "total_views": 1000,
        "total_likes": 50
    }])

    # Upsert comments
    store.upsert_comments([
        {
            "comment_id": "c_1",
            "channel_id": "UC_AUTHOR_1",
            "author_display_name": "Author One",
            "video_id": "vid_001",
            "comment_date": 1722513600,
            "comment_likes": 5,
            "reply_count": 0,
            "is_reply": False,
            "parent_id": None,
            "comment_text": "Great test comment!"
        }
    ])

    cur.execute("SELECT COUNT(*) FROM channels WHERE channel_id='UC_TEST_1'")
    assert cur.fetchone()[0] == 1

    cur.execute("SELECT COUNT(*) FROM videos WHERE video_id='vid_001'")
    assert cur.fetchone()[0] == 1

    cur.execute("SELECT COUNT(*) FROM comments WHERE comment_id='c_1'")
    assert cur.fetchone()[0] == 1

    conn.close()


def test_youtube_client_missing_key():
    """Verify YouTubeClient raises ValueError if initialized with no API key."""
    client = YouTubeClient(api_key=None)
    client.api_key = None  # Ensure no key
    with pytest.raises(ValueError, match="YouTube API Key is missing"):
        client.resolve_channel(handle="@test")


def test_mock_ingest_workflow(tmp_path, monkeypatch):
    """Verify end-to-end ingest_youtube_data using MockYouTubeClient."""
    mock_db = tmp_path / "mock_commentsuite.sqlite3"
    mock_interim = tmp_path / "interim"
    mock_interim.mkdir()

    # Monkeypatch get_paths to point to test tmp_path
    monkeypatch.setattr(
        "engine.youtube_api.get_paths",
        lambda cfg: (mock_db, mock_interim, tmp_path / "output")
    )

    res = ingest_youtube_data(
        handle="@test_creator",
        max_videos=2,
        max_comments_per_video=15,
        use_mock=True,
        run_migration=False
    )

    assert res["status"] == "success"
    assert res["videos_ingested"] == 2
    assert res["comments_ingested"] > 0
    assert Path(res["raw_db_path"]).exists()

    conn = sqlite3.connect(mock_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM videos")
    assert cur.fetchone()[0] == 2
    cur.execute("SELECT COUNT(*) FROM comments")
    assert cur.fetchone()[0] > 0
    conn.close()
