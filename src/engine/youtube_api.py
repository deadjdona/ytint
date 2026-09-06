"""ytint // Live YouTube Data API v3 Ingest Connector (src/engine/youtube_api.py)

Direct, resilient ingestion connector fetching channel uploads, video metadata,
comment threads, and replies via the official YouTube Data API v3.
Persists data directly to canonical SQLite / Parquet layers for downstream intelligence modeling.
"""

from __future__ import annotations

import sys
import os
import time
import sqlite3
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import requests
import pandas as pd

# Ensure src directory is in sys.path
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from engine.config_loader import load_config, get_paths

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ytint_ingest")

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


class YouTubeClient:
    """Production client for the YouTube Data API v3 with quota tracking and retry backoff."""

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            # Try to read from settings.yaml
            try:
                cfg = load_config()
                api_key = cfg.get("youtube_api", {}).get("api_key")
            except Exception:
                pass

        self.api_key = api_key
        self.session = requests.Session()
        self.quota_units_used = 0

    def _request(self, endpoint: str, params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Performs an HTTP GET request to YouTube API with exponential backoff and quota tracking."""
        if not self.api_key:
            raise ValueError(
                "❌ YouTube API Key is missing! Set YOUTUBE_API_KEY environment variable "
                "or pass --api-key / specify in config/settings.yaml."
            )

        url = f"{YOUTUBE_API_BASE}/{endpoint}"
        req_params = dict(params)
        req_params["key"] = self.api_key

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, params=req_params, timeout=15)
                if resp.status_code == 200:
                    # Estimate quota cost
                    cost_map = {
                        "channels": 1,
                        "videos": 1,
                        "playlistItems": 1,
                        "commentThreads": 1,
                        "comments": 1,
                        "search": 100
                    }
                    self.quota_units_used += cost_map.get(endpoint, 1)
                    return resp.json()

                if resp.status_code == 403:
                    err_json = resp.json().get("error", {})
                    errors = err_json.get("errors", [])
                    reasons = [e.get("reason") for e in errors]
                    if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
                        raise RuntimeError("❌ YouTube API Quota Exceeded (403 quotaExceeded). Please try again tomorrow or upgrade quota.")
                    raise RuntimeError(f"❌ YouTube API Forbidden (403): {err_json.get('message', resp.text)}")

                if resp.status_code in [429, 500, 502, 503, 504]:
                    sleep_time = attempt * 2.0
                    logger.warning(f"HTTP {resp.status_code} on {endpoint}. Retrying in {sleep_time}s (attempt {attempt}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue

                resp.raise_for_status()
            except requests.RequestException as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Network error querying YouTube API endpoint '{endpoint}': {e}")
                time.sleep(attempt * 2.0)

        raise RuntimeError(f"Failed to fetch {endpoint} after {max_retries} attempts.")

    def resolve_channel(self, channel_id: Optional[str] = None, handle: Optional[str] = None) -> Dict[str, Any]:
        """Resolves a channel ID or @handle to channel metadata and uploads playlist ID."""
        params: Dict[str, Any] = {
            "part": "snippet,contentDetails,statistics"
        }
        if handle:
            norm_handle = handle if handle.startswith("@") else f"@{handle}"
            params["forHandle"] = norm_handle
        elif channel_id:
            params["id"] = channel_id
        else:
            raise ValueError("Either channel_id or handle must be provided.")

        data = self._request("channels", params)
        items = data.get("items", [])
        if not items:
            identifier = handle or channel_id
            raise ValueError(f"Channel '{identifier}' could not be resolved on YouTube.")

        ch = items[0]
        snippet = ch.get("snippet", {})
        content_details = ch.get("contentDetails", {})
        statistics = ch.get("statistics", {})

        uploads_id = content_details.get("relatedPlaylists", {}).get("uploads")
        return {
            "channel_id": ch.get("id"),
            "title": snippet.get("title", "Unknown Channel"),
            "custom_url": snippet.get("customUrl"),
            "description": snippet.get("description", ""),
            "thumb_url": snippet.get("thumbnails", {}).get("default", {}).get("url"),
            "uploads_playlist_id": uploads_id,
            "video_count": int(statistics.get("videoCount", 0)),
            "view_count": int(statistics.get("viewCount", 0)),
            "subscriber_count": int(statistics.get("subscriberCount", 0))
        }

    def fetch_channel_videos(self, uploads_playlist_id: str, max_videos: int = 10) -> List[Dict[str, Any]]:
        """Fetches video IDs and basic details from the channel's uploads playlist."""
        videos = []
        page_token = None

        while len(videos) < max_videos:
            fetch_count = min(50, max_videos - len(videos))
            params: Dict[str, Any] = {
                "part": "snippet,contentDetails",
                "playlistId": uploads_playlist_id,
                "maxResults": fetch_count
            }
            if page_token:
                params["pageToken"] = page_token

            data = self._request("playlistItems", params)
            items = data.get("items", [])
            if not items:
                break

            for item in items:
                v_id = item.get("contentDetails", {}).get("videoId")
                snippet = item.get("snippet", {})
                if v_id:
                    videos.append({
                        "video_id": v_id,
                        "title": snippet.get("title", ""),
                        "published_at": snippet.get("publishedAt", ""),
                        "description": snippet.get("description", ""),
                        "thumb_url": snippet.get("thumbnails", {}).get("high", {}).get("url", "")
                    })

            page_token = data.get("nextPageToken")
            if not page_token or len(items) == 0:
                break

        return videos[:max_videos]

    def fetch_video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch fetches full video statistics and content details for a list of video IDs."""
        details = []
        # YouTube API allows up to 50 video IDs per request
        chunk_size = 50
        for i in range(0, len(video_ids), chunk_size):
            chunk = video_ids[i:i + chunk_size]
            params = {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(chunk)
            }
            data = self._request("videos", params)
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                details.append({
                    "video_id": item.get("id"),
                    "channel_id": snippet.get("channelId"),
                    "title": snippet.get("title"),
                    "published_at": snippet.get("publishedAt"),
                    "description": snippet.get("description", ""),
                    "thumb_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    "total_views": int(stats.get("viewCount", 0)),
                    "total_likes": int(stats.get("likeCount", 0)),
                    "total_comments": int(stats.get("commentCount", 0))
                })
        return details

    def fetch_comment_threads(
        self,
        video_id: str,
        max_comments: int = 1000
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fetches all top-level comments and replies for a given video ID."""
        top_comments = []
        replies = []
        page_token = None

        while len(top_comments) < max_comments:
            fetch_count = min(100, max_comments - len(top_comments))
            params: Dict[str, Any] = {
                "part": "snippet,replies",
                "videoId": video_id,
                "maxResults": fetch_count,
                "textFormat": "plainText"
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                data = self._request("commentThreads", params)
            except RuntimeError as e:
                # Video comments may be disabled
                if "commentsDisabled" in str(e):
                    logger.warning(f"Comments are disabled on video {video_id}.")
                    break
                raise

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                snippet = item.get("snippet", {})
                top_c = snippet.get("topLevelComment", {}).get("snippet", {})
                c_id = item.get("id")

                pub_at = top_c.get("publishedAt")
                # Parse timestamp to unix seconds
                ts = int(datetime.fromisoformat(pub_at.replace("Z", "+00:00")).timestamp()) if pub_at else int(time.time())

                top_record = {
                    "comment_id": c_id,
                    "channel_id": top_c.get("authorChannelId", {}).get("value", ""),
                    "author_display_name": top_c.get("authorDisplayName", ""),
                    "author_avatar_url": top_c.get("authorProfileImageUrl", ""),
                    "video_id": video_id,
                    "comment_date": ts,
                    "comment_likes": int(top_c.get("likeCount", 0)),
                    "reply_count": int(snippet.get("totalReplyCount", 0)),
                    "is_reply": False,
                    "parent_id": None,
                    "comment_text": top_c.get("textOriginal", "")
                }
                top_comments.append(top_record)

                # Extract inline replies if returned by commentThreads
                replies_obj = item.get("replies", {})
                inline_replies = replies_obj.get("comments", [])
                for r in inline_replies:
                    r_snip = r.get("snippet", {})
                    r_pub = r_snip.get("publishedAt")
                    r_ts = int(datetime.fromisoformat(r_pub.replace("Z", "+00:00")).timestamp()) if r_pub else int(time.time())
                    replies.append({
                        "comment_id": r.get("id"),
                        "channel_id": r_snip.get("authorChannelId", {}).get("value", ""),
                        "author_display_name": r_snip.get("authorDisplayName", ""),
                        "author_avatar_url": r_snip.get("authorProfileImageUrl", ""),
                        "video_id": video_id,
                        "comment_date": r_ts,
                        "comment_likes": int(r_snip.get("likeCount", 0)),
                        "reply_count": 0,
                        "is_reply": True,
                        "parent_id": c_id,
                        "comment_text": r_snip.get("textOriginal", "")
                    })

                # If totalReplyCount > len(inline_replies), fetch deep replies via comments.list
                total_replies = int(snippet.get("totalReplyCount", 0))
                if total_replies > len(inline_replies):
                    deep_replies = self._fetch_replies_for_thread(c_id, video_id, max_needed=total_replies - len(inline_replies))
                    replies.extend(deep_replies)

            page_token = data.get("nextPageToken")
            if not page_token or len(items) == 0:
                break

        return top_comments, replies

    def _fetch_replies_for_thread(self, parent_id: str, video_id: str, max_needed: int = 100) -> List[Dict[str, Any]]:
        """Paginates comments.list to retrieve remaining replies for a large thread."""
        extra_replies = []
        page_token = None

        while len(extra_replies) < max_needed:
            params: Dict[str, Any] = {
                "part": "snippet",
                "parentId": parent_id,
                "maxResults": min(100, max_needed - len(extra_replies)),
                "textFormat": "plainText"
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                data = self._request("comments", params)
            except Exception as e:
                logger.warning(f"Could not fetch full replies for thread {parent_id}: {e}")
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                snip = item.get("snippet", {})
                pub_at = snip.get("publishedAt")
                ts = int(datetime.fromisoformat(pub_at.replace("Z", "+00:00")).timestamp()) if pub_at else int(time.time())
                extra_replies.append({
                    "comment_id": item.get("id"),
                    "channel_id": snip.get("authorChannelId", {}).get("value", ""),
                    "author_display_name": snip.get("authorDisplayName", ""),
                    "author_avatar_url": snip.get("authorProfileImageUrl", ""),
                    "video_id": video_id,
                    "comment_date": ts,
                    "comment_likes": int(snip.get("likeCount", 0)),
                    "reply_count": 0,
                    "is_reply": True,
                    "parent_id": parent_id,
                    "comment_text": snip.get("textOriginal", "")
                })

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return extra_replies


class MockYouTubeClient(YouTubeClient):
    """Synthesizes high-fidelity mock responses for offline testing, demos, and CI."""

    def __init__(self, api_key: Optional[str] = "MOCK_KEY"):
        super().__init__(api_key=api_key)

    def resolve_channel(self, channel_id: Optional[str] = None, handle: Optional[str] = None) -> Dict[str, Any]:
        cid = channel_id or "UC_MOCK_CHANNEL_12345"
        hname = handle or "@ytint_creator"
        return {
            "channel_id": cid,
            "title": f"Intelligence Channel ({hname})",
            "custom_url": hname,
            "description": "High-dimensional conversational analytics showcase channel.",
            "thumb_url": "https://example.com/channel_avatar.jpg",
            "uploads_playlist_id": f"UU_{cid.lstrip('UC')}",
            "video_count": 42,
            "view_count": 1250000,
            "subscriber_count": 85000
        }

    def fetch_channel_videos(self, uploads_playlist_id: str, max_videos: int = 10) -> List[Dict[str, Any]]:
        videos = []
        for i in range(1, max_videos + 1):
            videos.append({
                "video_id": f"mock_vid_{i:03d}",
                "title": f"Deep Dive Discussion #{i:02d} // Machine Learning and Society",
                "published_at": f"2026-08-{i:02d}T14:00:00Z",
                "description": f"Comprehensive exploration episode {i} on community networks.",
                "thumb_url": f"https://example.com/thumb_{i:03d}.jpg"
            })
        return videos

    def fetch_video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        details = []
        for vid in video_ids:
            details.append({
                "video_id": vid,
                "channel_id": "UC_MOCK_CHANNEL_12345",
                "title": f"Video Title for {vid}",
                "published_at": "2026-08-15T12:00:00Z",
                "description": "Detailed video description with timestamps.",
                "thumb_url": "https://example.com/thumb.jpg",
                "total_views": 45000,
                "total_likes": 2800,
                "total_comments": 350
            })
        return details

    def fetch_comment_threads(
        self,
        video_id: str,
        max_comments: int = 1000
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        top_comments = []
        replies = []
        num_top = min(max_comments, 25)
        base_ts = int(time.time()) - 86400 * 5

        for i in range(1, num_top + 1):
            c_id = f"c_{video_id}_{i:03d}"
            top_comments.append({
                "comment_id": c_id,
                "channel_id": f"UC_user_{i:04d}",
                "author_display_name": f"CommunityUser_{i}",
                "author_avatar_url": f"https://example.com/user_{i}.jpg",
                "video_id": video_id,
                "comment_date": base_ts + i * 3600,
                "comment_likes": (i * 7) % 45,
                "reply_count": 2 if i % 3 == 0 else 0,
                "is_reply": False,
                "parent_id": None,
                "comment_text": f"This is top-level comment #{i} regarding the video findings."
            })
            if i % 3 == 0:
                for r_idx in range(1, 3):
                    r_id = f"r_{c_id}_{r_idx}"
                    replies.append({
                        "comment_id": r_id,
                        "channel_id": f"UC_reply_user_{i}_{r_idx}",
                        "author_display_name": f"Responder_{i}_{r_idx}",
                        "author_avatar_url": f"https://example.com/ruser_{i}_{r_idx}.jpg",
                        "video_id": video_id,
                        "comment_date": base_ts + i * 3600 + r_idx * 600,
                        "comment_likes": r_idx * 3,
                        "reply_count": 0,
                        "is_reply": True,
                        "parent_id": c_id,
                        "comment_text": f"Reply #{r_idx} answering comment {c_id} with further detail."
                    })

        return top_comments, replies


class SQLiteCommentsuiteStore:
    """Manages upserting and persisting ingested YouTube data into commentsuite.sqlite3."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def ensure_schema(self):
        """Creates the canonical Commentsuite tables and indexes if not already present."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY,
            channel_name TEXT,
            channel_profile_url TEXT,
            download_profile BOOLEAN DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            channel_id TEXT,
            grab_date INTEGER,
            publish_date INTEGER,
            video_title TEXT,
            total_comments INTEGER,
            total_views INTEGER,
            total_likes INTEGER,
            total_dislikes INTEGER DEFAULT 0,
            video_desc TEXT,
            thumb_url TEXT,
            http_code INTEGER DEFAULT 200,
            FOREIGN KEY (channel_id) REFERENCES channels (channel_id)
        );

        CREATE TABLE IF NOT EXISTS comments (
            comment_id TEXT PRIMARY KEY,
            channel_id TEXT,
            video_id TEXT,
            comment_date INTEGER,
            comment_likes INTEGER,
            reply_count INTEGER,
            is_reply BOOLEAN,
            parent_id TEXT,
            comment_text TEXT,
            FOREIGN KEY (channel_id) REFERENCES channels (channel_id),
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        );

        CREATE INDEX IF NOT EXISTS idx_comments_video ON comments(video_id);
        CREATE INDEX IF NOT EXISTS idx_comments_author ON comments(channel_id);
        CREATE INDEX IF NOT EXISTS idx_comments_parent ON comments(parent_id);
        """)
        conn.commit()
        conn.close()

    def upsert_channel(self, channel_data: Dict[str, Any]):
        """Upserts a channel record into SQLite."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO channels (channel_id, channel_name, channel_profile_url, download_profile)
            VALUES (?, ?, ?, ?)
        """, (
            channel_data["channel_id"],
            channel_data.get("title", ""),
            channel_data.get("thumb_url", ""),
            1
        ))
        conn.commit()
        conn.close()

    def upsert_videos(self, videos_data: List[Dict[str, Any]]):
        """Upserts a batch of video records into SQLite."""
        if not videos_data:
            return
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        now_ts = int(time.time())
        rows = []
        for v in videos_data:
            pub = v.get("published_at")
            pub_ts = int(datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()) if isinstance(pub, str) and pub else now_ts
            rows.append((
                v["video_id"],
                v.get("channel_id", ""),
                now_ts,
                pub_ts,
                v.get("title", ""),
                v.get("total_comments", 0),
                v.get("total_views", 0),
                v.get("total_likes", 0),
                0,
                v.get("description", ""),
                v.get("thumb_url", ""),
                200
            ))
        cur.executemany("""
            INSERT OR REPLACE INTO videos (
                video_id, channel_id, grab_date, publish_date, video_title,
                total_comments, total_views, total_likes, total_dislikes,
                video_desc, thumb_url, http_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        conn.close()

    def upsert_comments(self, comments_data: List[Dict[str, Any]]):
        """Upserts a batch of comment records (and author channels) into SQLite."""
        if not comments_data:
            return
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        channel_rows = []
        comment_rows = []

        for c in comments_data:
            ch_id = c.get("channel_id") or "UNKNOWN_AUTHOR"
            ch_name = c.get("author_display_name") or ch_id
            avatar = c.get("author_avatar_url") or ""
            channel_rows.append((ch_id, ch_name, avatar, 0))

            comment_rows.append((
                c["comment_id"],
                ch_id,
                c["video_id"],
                int(c.get("comment_date", int(time.time()))),
                int(c.get("comment_likes", 0)),
                int(c.get("reply_count", 0)),
                1 if c.get("is_reply") else 0,
                c.get("parent_id"),
                c.get("comment_text", "")
            ))

        # Insert channel stubs first to satisfy foreign key relationships
        cur.executemany("""
            INSERT OR IGNORE INTO channels (channel_id, channel_name, channel_profile_url, download_profile)
            VALUES (?, ?, ?, ?)
        """, channel_rows)

        cur.executemany("""
            INSERT OR REPLACE INTO comments (
                comment_id, channel_id, video_id, comment_date, comment_likes,
                reply_count, is_reply, parent_id, comment_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, comment_rows)

        conn.commit()
        conn.close()


def ingest_youtube_data(
    channel_id: Optional[str] = None,
    handle: Optional[str] = None,
    video_id: Optional[str] = None,
    api_key: Optional[str] = None,
    max_videos: int = 10,
    max_comments_per_video: int = 1000,
    use_mock: bool = False,
    run_migration: bool = True,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """High-level ingestion workflow coordinating YouTube API queries, SQLite storage, and Parquet migration."""
    config = load_config()
    raw_db_path, interim_dir, _ = get_paths(config)

    store = SQLiteCommentsuiteStore(raw_db_path)
    client: YouTubeClient = MockYouTubeClient(api_key=api_key) if use_mock else YouTubeClient(api_key=api_key)

    total_videos_ingested = 0
    total_comments_ingested = 0

    if video_id:
        # Ingest single video directly
        logger.info(f"Targeting single video: {video_id}")
        v_details = client.fetch_video_details([video_id])
        if not v_details:
            raise ValueError(f"Could not retrieve details for video '{video_id}'.")

        store.upsert_videos(v_details)
        top_c, reps = client.fetch_comment_threads(video_id, max_comments=max_comments_per_video)
        store.upsert_comments(top_c + reps)

        total_videos_ingested = 1
        total_comments_ingested = len(top_c) + len(reps)
        if progress_callback:
            progress_callback(1, 1, v_details[0].get("title", video_id), total_comments_ingested)

    else:
        # Ingest channel uploads
        if not channel_id and not handle:
            raise ValueError("Either channel_id, handle, or video_id must be provided.")

        ch_meta = client.resolve_channel(channel_id=channel_id, handle=handle)
        store.upsert_channel(ch_meta)
        logger.info(f"Resolved Channel: '{ch_meta['title']}' (ID: {ch_meta['channel_id']})")

        uploads_id = ch_meta.get("uploads_playlist_id")
        if not uploads_id:
            raise ValueError("No uploads playlist found for this channel.")

        videos_list = client.fetch_channel_videos(uploads_id, max_videos=max_videos)
        logger.info(f"Fetched {len(videos_list)} video candidates from uploads playlist.")

        video_ids = [v["video_id"] for v in videos_list]
        full_details = client.fetch_video_details(video_ids)
        store.upsert_videos(full_details)
        total_videos_ingested = len(full_details)

        for idx, vid_info in enumerate(full_details, start=1):
            vid = vid_info["video_id"]
            v_title = vid_info.get("title", vid)
            logger.info(f"[{idx}/{total_videos_ingested}] Ingesting comments for: '{v_title}' ({vid})...")

            top_c, reps = client.fetch_comment_threads(vid, max_comments=max_comments_per_video)
            all_c = top_c + reps
            store.upsert_comments(all_c)
            total_comments_ingested += len(all_c)

            if progress_callback:
                progress_callback(idx, total_videos_ingested, v_title, len(all_c))

    # Run s00 migration from SQLite to clean Parquet layers
    if run_migration:
        logger.info("Triggering Stage 00 (SQLite -> Parquet) migration...")
        from pipeline.s00_ingest import migrate_from_commentsuite
        migrate_from_commentsuite()

    logger.info(f"🎉 Ingestion complete: {total_comments_ingested:,} comments across {total_videos_ingested} videos.")
    return {
        "status": "success",
        "videos_ingested": total_videos_ingested,
        "comments_ingested": total_comments_ingested,
        "raw_db_path": str(raw_db_path)
    }


def main():
    """CLI Entrypoint for ytint-ingest."""
    parser = argparse.ArgumentParser(description="ytint Live YouTube Data API v3 Ingest Connector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--handle", type=str, help="YouTube channel handle (e.g. @mkbhd or @3blue1brown)")
    group.add_argument("--channel-id", type=str, help="YouTube channel ID (e.g. UC...)")
    group.add_argument("--video-id", type=str, help="Direct YouTube video ID to fetch")

    parser.add_argument("--api-key", type=str, default=None, help="Google Cloud YouTube Data API v3 key (or set YOUTUBE_API_KEY env)")
    parser.add_argument("--max-videos", type=int, default=10, help="Maximum number of uploads to retrieve (default: 10)")
    parser.add_argument("--max-comments", type=int, default=1000, help="Maximum comments to fetch per video (default: 1000)")
    parser.add_argument("--mock", action="store_true", help="Use built-in mock responses without calling the live API")
    parser.add_argument("--no-pipeline", action="store_true", help="Do not trigger SQLite to Parquet migration after ingestion")
    args = parser.parse_args()

    try:
        res = ingest_youtube_data(
            channel_id=args.channel_id,
            handle=args.handle,
            video_id=args.video_id,
            api_key=args.api_key,
            max_videos=args.max_videos,
            max_comments_per_video=args.max_comments,
            use_mock=args.mock,
            run_migration=not args.no_pipeline
        )
        print(f"\n🎉 Successfully ingested {res['comments_ingested']:,} comments across {res['videos_ingested']} videos into:\n{res['raw_db_path']}\n")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
