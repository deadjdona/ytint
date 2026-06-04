import os
import sqlite3
import yaml
import pandas as pd
from pathlib import Path

def load_config(config_path="config/settings.yaml"):
    """Loads central project configurations."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def migrate_from_commentsuite():
    config = load_config()
    raw_db_path = Path(config["paths"]["raw_db"])
    interim_dir = Path(config["paths"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    if not raw_db_path.exists():
        print(f"❌ Source database not found at: {raw_db_path}")
        print("Please ensure your 'commentsuite.sqlite3' file is placed in 'data/raw/'.")
        return

    print(f"🔌 Connecting to source database: {raw_db_path}")
    src_conn = sqlite3.connect(raw_db_path)
    
    print("📥 Reading raw comment layers into memory...")
    # Read directly from your schema
    query = """
    SELECT 
        comment_id,
        video_id,
        parent_id,
        channel_id AS author_channel_id,
        comment_text AS text,
        comment_likes AS like_count,
        comment_date
    FROM comments;
    """
    df_comments = pd.read_sql_query(query, src_conn)
    src_conn.close()

    if df_comments.empty:
        print("⚠️ The source 'comments' table is empty.")
        return

    print(f"🛠️ Processing {len(df_comments)} records...")

    # Handle your INTEGER date (Assuming it's a standard UNIX timestamp in seconds or ms)
    # unit='s' handles standard UNIX. If your dates look like year 1970, change unit to 'ms'
    try:
        df_comments['published_at'] = pd.to_datetime(df_comments['comment_date'], unit='s')
    except Exception:
        print("🔄 Timestamp parsing issue, attempting default string conversion...")
        df_comments['published_at'] = pd.to_datetime(df_comments['comment_date'])

    # Drop the raw unparsed date column to keep parquet files lightweight
    df_comments = df_comments.drop(columns=['comment_date'])

    # --- Synthesize a Minimal Videos Table ---
    # Since your schema references a 'videos' table that we didn't pull, we can instantly
    # synthesize the video metadata base by grouping your comments. This guarantees the 
    # downstream pipeline functions perfectly even without parsing the separate videos table.
    print("📹 Synthesizing video metadata timeline structures...")
    video_timeline = df_comments.groupby('video_id')['published_at'].min().reset_index()
    video_timeline.columns = ['video_id', 'published_at']
    # Create mock titles for the UI since we only have IDs right now
    video_timeline['title'] = "Video Asset // ID: " + video_timeline['video_id'].astype(str)

    # --- Save optimized analytical layers ---
    print("💾 Archiving clean Parquet structures to interim cache...")
    df_comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)
    video_timeline.to_parquet(interim_dir / "videos_clean.parquet", index=False)

    print(f"✅ Successfully ingested {len(df_comments)} comments across {len(video_timeline)} unique video IDs.")

if __name__ == "__main__":
    migrate_from_commentsuite()