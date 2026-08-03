import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path
from engine.config_loader import load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def parse_comment_dates(comment_date_series):
    """Parse Commentsuite integer timestamps, auto-detecting seconds vs milliseconds."""
    numeric_dates = pd.to_numeric(comment_date_series, errors="coerce")
    sample = numeric_dates.dropna()

    if sample.empty:
        return pd.to_datetime(comment_date_series, errors="coerce")

    median_value = sample.median()
    unit = "ms" if median_value > 1e11 else "s"
    return pd.to_datetime(numeric_dates, unit=unit, errors="coerce")

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
    cursor = src_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = set(row[0] for row in cursor.fetchall())

    print("📥 Ingesting comment & channel metadata...")
    if 'channels' in existing_tables:
        comments_query = """
        SELECT 
            c.comment_id,
            c.video_id,
            c.parent_id,
            c.channel_id AS author_channel_id,
            ch.channel_name AS author_display_name,
            ch.channel_profile_url AS author_avatar_url,
            c.comment_text AS text,
            c.comment_likes AS like_count,
            c.reply_count,
            c.is_reply,
            c.comment_date
        FROM comments c
        LEFT JOIN channels ch ON c.channel_id = ch.channel_id;
        """
    else:
        comments_query = """
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
    df_comments = pd.read_sql_query(comments_query, src_conn)

    df_videos = pd.DataFrame()
    if 'videos' in existing_tables:
        print("📥 Ingesting rich video metadata layer...")
        videos_query = """
        SELECT 
            video_id,
            channel_id,
            video_title AS title,
            publish_date,
            grab_date,
            total_comments,
            total_views,
            total_likes AS video_likes,
            total_dislikes,
            video_desc,
            thumb_url
        FROM videos;
        """
        try:
            df_videos = pd.read_sql_query(videos_query, src_conn)
        except Exception as e:
            print(f"⚠️ Warning reading videos table: {e}")

    src_conn.close()

    if df_comments.empty:
        print("⚠️ The source 'comments' table is empty.")
        return

    print(f"🛠️ Processing {len(df_comments)} records...")

    # Commentsuite stores integer UNIX timestamps. Auto-detect milliseconds vs seconds.
    df_comments['published_at'] = parse_comment_dates(df_comments['comment_date'])

    # Retain canonical text column and text_original alias for downstream pipeline compatibility
    df_comments['text_original'] = df_comments['text']

    # Drop the raw unparsed date column
    df_comments = df_comments.drop(columns=['comment_date'])

    # Process author display names
    if 'author_display_name' not in df_comments.columns:
        df_comments['author_display_name'] = df_comments['author_channel_id']
    else:
        df_comments['author_display_name'] = df_comments['author_display_name'].fillna(df_comments['author_channel_id'])

    # Process video metadata layer
    if not df_videos.empty:
        df_videos['published_at'] = parse_comment_dates(df_videos['publish_date'])
        if 'grab_date' in df_videos.columns:
            df_videos['grab_at'] = parse_comment_dates(df_videos['grab_date'])
        df_videos['title'] = df_videos['title'].fillna("Video Asset // ID: " + df_videos['video_id'].astype(str))
    else:
        print("📹 Synthesizing video metadata timeline structures...")
        df_videos = df_comments.groupby('video_id')['published_at'].min().reset_index()
        df_videos.columns = ['video_id', 'published_at']
        df_videos['title'] = "Video Asset // ID: " + df_videos['video_id'].astype(str)

    # --- Save optimized analytical layers ---
    print("💾 Archiving clean Parquet structures to interim cache...")
    df_comments.to_parquet(interim_dir / "comments_clean.parquet", index=False)
    df_videos.to_parquet(interim_dir / "videos_clean.parquet", index=False)

    print(f"✅ Successfully ingested {len(df_comments)} comments across {len(df_videos)} unique videos.")

if __name__ == "__main__":
    migrate_from_commentsuite()