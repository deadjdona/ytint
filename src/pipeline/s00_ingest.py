import os
import sqlite3
import yaml
import pandas as pd
from pathlib import Path

def load_config():
    """
    Dynamically resolves the project root directory and loads the unified settings.
    Ensures absolute path compatibility across all execution environments.
    """
    # 1. Locate the file system context of the running script
    current_file = Path(__file__).resolve()
    
    # 2. Walk upward until we locate the parent directory containing the 'config' folder
    root_dir = current_file.parent
    while root_dir != root_dir.parent:
        if (root_dir / "config").is_dir():
            break
        root_dir = root_dir.parent
        
    config_path = root_dir / "config" / "settings.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"❌ Critical Configuration Alignment Failure:\n"
            f"Could not locate 'config/settings.yaml'.\n"
            f"Resolved root searched: {root_dir}"
        )
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 3. Dynamic Absolute Translation Layer
    # Automatically convert configured paths to absolute system paths relative to the project root
    config["paths"]["raw_db"] = str(root_dir / config["paths"]["raw_db"])
    config["paths"]["interim_dir"] = str(root_dir / config["paths"]["interim_dir"])
    config["paths"]["output_dir"] = str(root_dir / config["paths"]["output_dir"])
    
    return config

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

    # Commentsuite stores integer UNIX timestamps. Current exports use
    # milliseconds, but older data may use seconds, so auto-detect the scale.
    df_comments['published_at'] = parse_comment_dates(df_comments['comment_date'])

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