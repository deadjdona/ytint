"""ytint // Pipeline Delta Tracker & State Manager (src/engine/delta.py)

Inspects changes between the source SQLite database (commentsuite.sqlite3) and
interim Parquet artifacts (comments_clean.parquet, videos_clean.parquet) to support
fast incremental processing, change detection, and state persistence.
"""

from __future__ import annotations

import os
import sys
import json
import sqlite3
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple

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
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger("ytint_delta")


@dataclass
class DeltaReport:
    """Structured summary of differences between source SQLite and interim Parquet artifacts."""
    has_delta: bool = False
    new_comments_count: int = 0
    new_comment_ids: List[str] = field(default_factory=list)
    updated_comments_count: int = 0
    updated_comment_ids: List[str] = field(default_factory=list)
    new_videos_count: int = 0
    new_video_ids: List[str] = field(default_factory=list)
    total_sqlite_comments: int = 0
    total_interim_comments: int = 0
    total_sqlite_videos: int = 0
    total_interim_videos: int = 0
    sqlite_mtime: float = 0.0
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary_text(self) -> str:
        """Formatted human-readable summary of the detected delta."""
        if not self.has_delta:
            return (
                f"✅ Pipeline is fully synchronized (0 pending deltas).\n"
                f"   • Interim clean comments: {self.total_interim_comments:,}\n"
                f"   • Interim clean videos: {self.total_interim_videos:,}\n"
                f"   • Source SQLite comments: {self.total_sqlite_comments:,}"
            )

        lines = [
            f"⚡ Pending Ingestion Delta Detected:",
            f"   • New comments to ingest: +{self.new_comments_count:,}",
            f"   • Comments with updated like/reply counters: {self.updated_comments_count:,}",
            f"   • New videos to ingest: +{self.new_videos_count:,}",
            f"   • Current interim comments: {self.total_interim_comments:,} -> Target SQLite: {self.total_sqlite_comments:,}"
        ]
        return "\n".join(lines)


def inspect_delta(
    raw_db_path: Optional[Path] = None,
    interim_dir: Optional[Path] = None
) -> DeltaReport:
    """Compares source SQLite against interim Parquet to identify new and updated records."""
    if raw_db_path is None or interim_dir is None:
        cfg = load_config()
        raw_db, interim, _ = get_paths(cfg)
        raw_db_path = raw_db_path or raw_db
        interim_dir = interim_dir or interim
    else:
        raw_db_path = Path(raw_db_path)
        interim_dir = Path(interim_dir)

    if not raw_db_path.exists():
        logger.warning(f"Source database does not exist at {raw_db_path}")
        return DeltaReport()

    sqlite_mtime = os.path.getmtime(raw_db_path)
    comments_file = interim_dir / "comments_clean.parquet"
    videos_file = interim_dir / "videos_clean.parquet"

    # 1. Query source SQLite comment IDs and counts
    sqlite_comments: Dict[str, Tuple[int, int]] = {}
    sqlite_video_ids: Set[str] = set()

    try:
        conn = sqlite3.connect(raw_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = set(r[0] for r in cursor.fetchall())

        if "comments" in tables:
            cursor.execute("PRAGMA table_info(comments);")
            c_cols = {r[1] for r in cursor.fetchall()}
            id_col = "comment_id" if "comment_id" in c_cols else ("id" if "id" in c_cols else None)
            likes_col = "comment_likes" if "comment_likes" in c_cols else ("like_count" if "like_count" in c_cols else None)
            replies_col = "reply_count" if "reply_count" in c_cols else None

            if id_col:
                q_cols = [id_col]
                q_cols.append(likes_col if likes_col else "0")
                q_cols.append(replies_col if replies_col else "0")
                cursor.execute(f"SELECT {q_cols[0]}, {q_cols[1]}, {q_cols[2]} FROM comments")
                for cid, clikes, creplies in cursor.fetchall():
                    if cid is not None:
                        sqlite_comments[str(cid)] = (int(clikes or 0), int(creplies or 0))

        if "videos" in tables:
            cursor.execute("PRAGMA table_info(videos);")
            v_cols = {r[1] for r in cursor.fetchall()}
            vid_col = "video_id" if "video_id" in v_cols else ("id" if "id" in v_cols else None)
            if vid_col:
                cursor.execute(f"SELECT {vid_col} FROM videos")
                for (vid,) in cursor.fetchall():
                    if vid is not None:
                        sqlite_video_ids.add(str(vid))

        conn.close()
    except Exception as e:
        logger.error(f"Error querying SQLite database at {raw_db_path}: {e}")
        return DeltaReport(sqlite_mtime=sqlite_mtime)

    # 2. Inspect existing Parquet artifacts
    interim_comments: Dict[str, Tuple[int, int]] = {}
    interim_video_ids: Set[str] = set()

    if comments_file.exists():
        try:
            # Read only relevant identification columns for high speed
            cols_to_load = ["comment_id"]
            try:
                import pyarrow.parquet as pq
                avail_cols = pq.read_schema(comments_file).names
            except Exception:
                avail_cols = pd.read_parquet(comments_file).head(0).columns.tolist()

            if "like_count" in avail_cols:
                cols_to_load.append("like_count")
            if "reply_count" in avail_cols:
                cols_to_load.append("reply_count")
            
            df_existing = pd.read_parquet(comments_file, columns=cols_to_load)
            c_ids = df_existing["comment_id"].astype(str).tolist()
            likes = df_existing["like_count"].fillna(0).astype(int).tolist() if "like_count" in df_existing.columns else [0] * len(c_ids)
            replies = df_existing["reply_count"].fillna(0).astype(int).tolist() if "reply_count" in df_existing.columns else [0] * len(c_ids)
            interim_comments = dict(zip(c_ids, zip(likes, replies)))
        except Exception as e:
            logger.warning(f"Could not read existing comments_clean.parquet: {e}")

    if videos_file.exists():
        try:
            df_v_existing = pd.read_parquet(videos_file, columns=["video_id"])
            interim_video_ids = set(df_v_existing["video_id"].astype(str))
        except Exception as e:
            logger.warning(f"Could not read existing videos_clean.parquet: {e}")

    # 3. Compute Delta
    sqlite_c_set = set(sqlite_comments.keys())
    interim_c_set = set(interim_comments.keys())

    new_comment_ids = list(sqlite_c_set - interim_c_set)
    new_video_ids = list(sqlite_video_ids - interim_video_ids)

    # Check for count updates on common comments
    updated_comment_ids = []
    common_ids = sqlite_c_set.intersection(interim_c_set)
    for cid in common_ids:
        s_likes, s_replies = sqlite_comments[cid]
        i_likes, i_replies = interim_comments[cid]
        if s_likes != i_likes or s_replies != i_replies:
            updated_comment_ids.append(cid)

    has_delta = bool(new_comment_ids or updated_comment_ids or new_video_ids)

    return DeltaReport(
        has_delta=has_delta,
        new_comments_count=len(new_comment_ids),
        new_comment_ids=new_comment_ids,
        updated_comments_count=len(updated_comment_ids),
        updated_comment_ids=updated_comment_ids,
        new_videos_count=len(new_video_ids),
        new_video_ids=new_video_ids,
        total_sqlite_comments=len(sqlite_comments),
        total_interim_comments=len(interim_comments),
        total_sqlite_videos=len(sqlite_video_ids),
        total_interim_videos=len(interim_video_ids),
        sqlite_mtime=sqlite_mtime
    )


def load_pipeline_state(interim_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Loads state manifest from data/interim/.pipeline_state.json if present."""
    if interim_dir is None:
        cfg = load_config()
        _, interim, _ = get_paths(cfg)
        interim_dir = interim
    else:
        interim_dir = Path(interim_dir)

    state_path = interim_dir / ".pipeline_state.json"
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read pipeline state file: {e}")
    return {}


def save_pipeline_state(
    interim_dir: Optional[Path] = None,
    delta_report: Optional[DeltaReport] = None,
    mode: str = "incremental"
) -> Path:
    """Persists pipeline execution state to data/interim/.pipeline_state.json."""
    if interim_dir is None:
        cfg = load_config()
        _, interim, _ = get_paths(cfg)
        interim_dir = interim
    else:
        interim_dir = Path(interim_dir)

    state_path = interim_dir / ".pipeline_state.json"
    data = load_pipeline_state(interim_dir)

    data["last_execution_utc"] = datetime.now(timezone.utc).isoformat()
    data["execution_mode"] = mode
    data["mode"] = mode

    if delta_report:
        data["has_delta"] = delta_report.has_delta
        data["new_comments_count"] = delta_report.new_comments_count
        data["updated_comments_count"] = delta_report.updated_comments_count
        data["new_videos_count"] = delta_report.new_videos_count
        data["last_delta"] = {
            "has_delta": delta_report.has_delta,
            "new_comments_count": delta_report.new_comments_count,
            "updated_comments_count": delta_report.updated_comments_count,
            "new_videos_count": delta_report.new_videos_count,
            "total_comments": delta_report.total_interim_comments + delta_report.new_comments_count,
            "total_videos": delta_report.total_interim_videos + delta_report.new_videos_count,
            "sqlite_mtime": delta_report.sqlite_mtime
        }

    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write pipeline state file: {e}")

    return state_path
