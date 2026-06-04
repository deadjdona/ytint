import os
import json
import sqlite3
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_config(config_path="config/settings.yaml"):
    """Loads the central project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_db_connection(db_path):
    """Establishes a connection to the local SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")  # Enforce structural integrity
    return conn

def initialize_database(conn):
    """Creates the structural relational database tables if they do not exist."""
    cursor = conn.cursor()
    
    # 1. Core Videos Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        published_at DATETIME NOT NULL,
        view_count INTEGER DEFAULT 0,
        like_count INTEGER DEFAULT 0
    );
    """)

    # 2. Transcripts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcripts (
        video_id TEXT PRIMARY KEY,
        chunks TEXT, -- Raw JSON string array containing [{text, start, duration}]
        FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    );
    """)

    # 3. Comprehensive Comments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        comment_id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        parent_id TEXT, -- Populated if it is a nested reply thread
        author_channel_id TEXT NOT NULL,
        text TEXT NOT NULL,
        like_count INTEGER DEFAULT 0,
        published_at DATETIME NOT NULL,
        FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    );
    """)

    # 4. Ingestion Tracking Ledger (Enables Incremental Processing / Resumable State)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fetch_state (
        video_id TEXT PRIMARY KEY,
        comments_complete BOOLEAN DEFAULT FALSE,
        transcript_complete BOOLEAN DEFAULT FALSE,
        last_updated DATETIME NOT NULL,
        FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    );
    """)
    
    conn.commit()
    print("✅ Database initialization complete. All tables verified.")

def ingest_raw_json_files(conn, raw_dir):
    """
    Parses downloaded raw video/comment JSON files from external tooling
    and batches them into the local SQLite analytical core.
    """
    cursor = conn.cursor()
    raw_path = Path(raw_dir)
    
    # Target files matching typical video extraction pattern (e.g., video_id.json)
    json_files = list(raw_path.glob("*.json"))
    if not json_files:
        print(f"⚠️ No raw JSON files discovered inside: {raw_dir}")
        print("Place your extracted YouTube data there to parse it.")
        return

    print(f"Found {len(json_files)} raw video artifacts. Parsing contents...")
    
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ Failed to parse malformed JSON file: {file_path.name}")
                continue
            
        video_id = data.get("id") or data.get("video_id")
        if not video_id:
            continue
            
        # Extract metadata fields safely
        title = data.get("title", "Unknown Title")
        published_at = data.get("published_at") or data.get("upload_date")
        view_count = data.get("view_count", 0)
        like_count = data.get("like_count", 0)
        
        # Standardize standard ISO timestamp string formats
        try:
            dt = pd.to_datetime(published_at).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Insert metadata securely via parameterized queries
        cursor.execute("""
            INSERT INTO videos (video_id, title, published_at, view_count, like_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title=excluded.title,
                view_count=excluded.view_count,
                like_count=excluded.like_count;
        """, (video_id, title, dt, view_count, like_count))

        # Handle associated comment arrays
        comments_list = data.get("comments", [])
        for c in comments_list:
            c_id = c.get("comment_id") or c.get("id")
            if not c_id:
                continue
                
            c_text = c.get("text") or c.get("text_display", "")
            c_author = c.get("author_channel_id") or c.get("author", "anonymous")
            c_likes = c.get("like_count", 0)
            c_parent = c.get("parent_id") # NULL if top-level comment
            
            try:
                c_dt = pd.to_datetime(c.get("published_at") or c.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                c_dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute("""
                INSERT INTO comments (comment_id, video_id, parent_id, author_channel_id, text, like_count, published_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(comment_id) DO UPDATE SET
                    text=excluded.text,
                    like_count=excluded.like_count;
            """, (c_id, video_id, c_parent, c_author, c_text, c_likes, c_dt))

        # Log completion to our ingestion ledger system
        cursor.execute("""
            INSERT INTO fetch_state (video_id, comments_complete, last_updated)
            VALUES (?, 1, ?)
            ON CONFLICT(video_id) DO UPDATE SET 
                comments_complete=1, 
                last_updated=excluded.last_updated;
        """, (video_id, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    print("✅ Raw JSON metadata and comments successfully mapped to DB.")

def export_to_interim_parquet(conn, interim_dir):
    """
    Mirrors clean database states into analytical Parquet records.
    Parquet processes faster during bulk tokenization / transformer workloads.
    """
    interim_path = Path(interim_dir)
    interim_path.mkdir(parents=True, exist_ok=True)

    print("Syncing relational data structures to fast-access analytical Parquet tables...")
    
    # Export Normalized Video Dataframe
    df_videos = pd.read_sql_query("SELECT * FROM videos", conn)
    df_videos['published_at'] = pd.to_datetime(df_videos['published_at'])
    df_videos.to_parquet(interim_path / "videos_clean.parquet", index=False)

    # Export Normalized Community Comments Dataframe
    df_comments = pd.read_sql_query("SELECT * FROM comments", conn)
    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'])
    df_comments.to_parquet(interim_path / "comments_clean.parquet", index=False)
    
    print(f"📦 Successfully stored {len(df_comments)} clean comments across {len(df_videos)} historical video files.")

def main():
    config = load_config()
    db_path = config["paths"]["database"]
    raw_dir = config["paths"]["raw_dir"]
    interim_dir = config["paths"]["interim_dir"]

    # Ensure targeted operating system storage locations physically exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # Execute operational sequence pipeline steps
    conn = get_db_connection(db_path)
    try:
        initialize_database(conn)
        ingest_raw_json_files(conn, raw_dir)
        export_to_interim_parquet(conn, interim_dir)
    finally:
        conn.close()

if __name__ == "__main__":
    main()