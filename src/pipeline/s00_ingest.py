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

def migrate_from_commentsuite(incremental: bool = False):
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
        cursor.execute("PRAGMA table_info(comments);")
        c_cols = {row[1] for row in cursor.fetchall()}
        reply_clause = "reply_count," if "reply_count" in c_cols else "0 AS reply_count,"
        is_reply_clause = "is_reply," if "is_reply" in c_cols else ""
        comments_query = f"""
        SELECT 
            comment_id,
            video_id,
            parent_id,
            channel_id AS author_channel_id,
            comment_text AS text,
            comment_likes AS like_count,
            {reply_clause}
            {is_reply_clause}
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
            thumb_url,
            http_code
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

    if 'is_reply' not in df_comments.columns:
        df_comments['is_reply'] = df_comments['parent_id'].notna() & (df_comments['parent_id'] != "")
    if 'reply_count' not in df_comments.columns:
        df_comments['reply_count'] = 0

    # Process video metadata layer
    if not df_videos.empty:
        print("🛠️ Processing video metadata layer...")
        if 'publish_date' in df_videos.columns:
            df_videos['published_at'] = parse_comment_dates(df_videos['publish_date'])
        if 'grab_date' in df_videos.columns:
            df_videos['grab_at'] = parse_comment_dates(df_videos['grab_date'])
        df_videos['title'] = df_videos['title'].fillna("Video Asset // ID: " + df_videos['video_id'].astype(str))

        # Derive corpus quality metrics
        if 'total_views' in df_videos.columns and 'total_comments' in df_videos.columns:
            views = pd.to_numeric(df_videos['total_views'], errors='coerce').fillna(0)
            comments = pd.to_numeric(df_videos['total_comments'], errors='coerce').fillna(0)
            df_videos['comment_rate'] = (comments / (views / 1000)).replace([float('inf'), float('-inf')], 0).fillna(0).round(4)
        if 'http_code' in df_videos.columns:
            df_videos['is_comment_disabled'] = (
                (df_videos['http_code'].fillna(200).astype(int) != 200) |
                (pd.to_numeric(df_videos.get('total_comments', 0), errors='coerce').fillna(0) == 0)
            )
    else:
        print("📹 Synthesizing video metadata timeline structures...")
        df_videos = df_comments.groupby('video_id')['published_at'].min().reset_index()
        df_videos.columns = ['video_id', 'published_at']
        df_videos['title'] = "Video Asset // ID: " + df_videos['video_id'].astype(str)

    # --- Save optimized analytical layers ---
    comments_path = interim_dir / "comments_clean.parquet"
    videos_path = interim_dir / "videos_clean.parquet"

    if incremental and comments_path.exists():
        print("⚡ Incremental mode: Reconciling new and updated records with cached Parquet artifacts...")
        try:
            df_existing_comments = pd.read_parquet(comments_path)
            existing_cids = set(df_existing_comments['comment_id'].astype(str))
            
            # 1. Identify newly added comments
            is_new_comment = ~df_comments['comment_id'].astype(str).isin(existing_cids)
            df_new_comments = df_comments[is_new_comment].copy()
            
            # 2. Update mutable counters (like_count, reply_count) on existing comments without clobbering enriched columns
            if not is_new_comment.all():
                update_cols = ['comment_id']
                if 'like_count' in df_comments.columns:
                    update_cols.append('like_count')
                if 'reply_count' in df_comments.columns:
                    update_cols.append('reply_count')
                df_common = df_comments[~is_new_comment][update_cols].copy()
                df_existing_comments = df_existing_comments.set_index('comment_id')
                df_existing_comments.update(df_common.set_index('comment_id'))
                df_existing_comments = df_existing_comments.reset_index()

            # 3. Append new comments (they will have NaN for enriched columns to be processed by s01)
            if not df_new_comments.empty:
                df_final_comments = pd.concat([df_existing_comments, df_new_comments], ignore_index=True)
                print(f"  ➕ Appended {len(df_new_comments):,} new raw comments to existing {len(df_existing_comments):,} enriched rows.")
            else:
                df_final_comments = df_existing_comments
                print(f"  ✓ 0 new comments detected. Preserved {len(df_existing_comments):,} existing comments with updated counters.")
            
            # 4. Update videos layer
            if videos_path.exists() and not df_videos.empty:
                df_existing_videos = pd.read_parquet(videos_path)
                existing_vids = set(df_existing_videos['video_id'].astype(str))
                df_new_videos = df_videos[~df_videos['video_id'].astype(str).isin(existing_vids)]
                
                df_existing_videos = df_existing_videos.set_index('video_id')
                update_cols = [c for c in ['total_views', 'total_comments', 'video_likes', 'comment_rate', 'is_comment_disabled'] if c in df_videos.columns and c in df_existing_videos.columns]
                if update_cols:
                    df_existing_videos.update(df_videos.set_index('video_id')[update_cols])
                df_final_videos = df_existing_videos.reset_index()
                if not df_new_videos.empty:
                    df_final_videos = pd.concat([df_final_videos, df_new_videos], ignore_index=True)
            if 'published_at' in df_final_comments.columns:
                df_final_comments['published_at'] = pd.to_datetime(df_final_comments['published_at'], errors='coerce')
            if 'published_at' in df_final_videos.columns:
                df_final_videos['published_at'] = pd.to_datetime(df_final_videos['published_at'], errors='coerce')

            print("💾 Archiving updated Parquet structures to interim cache...")
            df_final_comments.to_parquet(comments_path, index=False)
            df_final_videos.to_parquet(videos_path, index=False)
            print(f"✅ Incremental ingestion complete: {len(df_final_comments):,} comments, {len(df_final_videos):,} videos.")
            return
        except Exception as inc_err:
            print(f"⚠️ Incremental reconcile error: {inc_err}. Falling back to full overwrite.")

    print("💾 Archiving clean Parquet structures to interim cache...")
    df_comments.to_parquet(comments_path, index=False)
    df_videos.to_parquet(videos_path, index=False)

    print(f"✅ Successfully ingested {len(df_comments)} comments across {len(df_videos)} unique videos.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 00 Ingest & Synthesize")
    parser.add_argument("--incremental", action="store_true", help="Perform non-destructive incremental upsert preserving enriched fields")
    args = parser.parse_args()
    migrate_from_commentsuite(incremental=args.incremental)

if __name__ == "__main__":
    main()