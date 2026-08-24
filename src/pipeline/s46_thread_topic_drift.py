"""Stage 46: Within-Thread Topic Drift Detection (s46_thread_topic_drift.py)

For threads with >= 3 comments, detects whether replies drift away from the
root comment's topic — measuring how often conversations go off-topic.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from engine.config_loader import load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MIN_THREAD_DEPTH = 3  # minimum replies to compute drift


def run_thread_topic_drift():
    """Main execution entrypoint for Stage 46."""
    print("🌀 Starting Within-Thread Topic Drift Detection (s46)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s46.")
        return

    import pyarrow.parquet as pq
    available = pq.read_schema(comments_file).names
    topic_col = next((c for c in ["topic_id", "topic", "topic_name"] if c in available), None)
    required = {"comment_id", "parent_id", "video_id"}
    if not required.issubset(set(available)) or topic_col is None:
        print(f"⚠️ Missing required columns (comment_id, parent_id, video_id, topic). Skipping s46.")
        return

    cols = list(required) + [topic_col]
    df = pd.read_parquet(comments_file, columns=cols)
    df = df.dropna(subset=[topic_col])
    df["is_reply"] = df["parent_id"].notna() & (df["parent_id"] != "") & (df["parent_id"] != df["comment_id"])

    print(f"  -> Loaded {len(df):,} comments; {df['is_reply'].sum():,} are replies.")

    # Build topic lookup: comment_id -> topic
    topic_lookup = df.set_index("comment_id")[topic_col].to_dict()

    # For each reply, find the root comment's topic
    root_comments = df[~df["is_reply"]].copy()
    reply_comments = df[df["is_reply"]].copy()

    # Map parent topic
    reply_comments["parent_topic"] = reply_comments["parent_id"].map(topic_lookup)

    # Thread root lookup (parent_id for top-level replies is the root comment_id)
    reply_comments["root_topic"] = reply_comments["parent_topic"]

    # Drift = reply is on a different topic than its parent
    reply_comments["topic_drifted"] = (
        reply_comments[topic_col] != reply_comments["parent_topic"]
    ) & reply_comments["parent_topic"].notna()

    # Build per-thread stats: group by parent_id (root comment)
    # Thread = all replies to the same root
    thread_stats = reply_comments.groupby("parent_id").agg(
        n_replies=("comment_id", "count"),
        n_drifted=("topic_drifted", "sum"),
        root_topic=("parent_topic", "first"),
        video_id=("video_id", "first"),
    ).reset_index()
    thread_stats = thread_stats.rename(columns={"parent_id": "root_comment_id"})
    thread_stats = thread_stats[thread_stats["n_replies"] >= MIN_THREAD_DEPTH]
    thread_stats["drift_rate"] = (thread_stats["n_drifted"] / thread_stats["n_replies"]).round(4)

    # Summary stats
    if thread_stats.empty:
        print(f"  -> No threads with >= {MIN_THREAD_DEPTH} replies found.")
    else:
        mean_drift = thread_stats["drift_rate"].mean()
        high_drift = (thread_stats["drift_rate"] > 0.5).sum()
        print(f"  -> Analyzed {len(thread_stats):,} threads. Mean topic drift: {mean_drift:.1%}")
        print(f"  -> {high_drift} threads with >50% reply topic drift")

    # Per-video drift summary
    if not thread_stats.empty:
        video_drift = thread_stats.groupby("video_id").agg(
            n_threads=("root_comment_id", "count"),
            mean_drift_rate=("drift_rate", "mean"),
            high_drift_threads=("drift_rate", lambda x: (x > 0.5).sum()),
        ).reset_index()
        video_drift["mean_drift_rate"] = video_drift["mean_drift_rate"].round(4)
    else:
        video_drift = pd.DataFrame(columns=["video_id", "n_threads", "mean_drift_rate", "high_drift_threads"])

    out_threads = out_dir / "thread_topic_drift.parquet"
    out_video = out_dir / "video_topic_drift_summary.parquet"
    thread_stats.to_parquet(out_threads, index=False) if not thread_stats.empty else \
        pd.DataFrame().to_parquet(out_threads, index=False)
    video_drift.to_parquet(out_video, index=False)

    print(f"✅ Saved {out_threads.name} and {out_video.name}")


if __name__ == "__main__":
    run_thread_topic_drift()
