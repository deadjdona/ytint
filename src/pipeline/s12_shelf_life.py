"""Stage 12: Shelf Life of Likes (s12_shelf_life.py)

Calculates the 'First Mover Advantage' by analyzing how many likes a comment 
typically receives based on how many hours/days after the video upload it was posted.
"""

import sys
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


import pandas as pd
from pathlib import Path
from engine.config_loader import load_config

def run_shelf_life():
    print("📈 Starting Shelf Life of Likes Calculation (s12)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    videos_file = interim_dir / "videos_clean.parquet"
    
    if not comments_file.exists() or not videos_file.exists():
        print("⚠️ Missing comments_clean.parquet or videos_clean.parquet. Skipping s12.")
        return
        
    print("  -> Loading comments and videos data...")
    # Read just the columns we need
    df_comments = pd.read_parquet(comments_file, columns=['comment_id', 'video_id', 'like_count', 'published_at'])
    df_videos = pd.read_parquet(videos_file, columns=['video_id', 'published_at'])
    
    # Rename video published_at to avoid collision
    df_videos = df_videos.rename(columns={'published_at': 'video_published_at'})
    
    # We only care about comments with likes >= 0
    df_comments = df_comments[df_comments['like_count'].notna()].copy()
    
    if df_comments.empty:
        print("⚠️ No comments with likes found. Skipping.")
        return
        
    print("  -> Merging and computing age relative to video upload...")
    df = df_comments.merge(df_videos, on='video_id', how='inner')
    
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['video_published_at'] = pd.to_datetime(df['video_published_at'])
    
    # Calculate hours since the video was published (using the first comment as proxy if real upload date missing)
    df['hours_since_upload'] = (df['published_at'] - df['video_published_at']).dt.total_seconds() / 3600.0
    
    # Filter out negatives (due to clock skew or weird data)
    df = df[df['hours_since_upload'] >= 0]
    
    print("  -> Aggregating average likes by hour bin...")
    # Bin by hour
    df['hour_bin'] = df['hours_since_upload'].astype(int)
    
    # We only care about the first 72 hours (3 days) to see the "shelf life" curve clearly
    # But let's compute it for the first 168 hours (1 week)
    df_subset = df[df['hour_bin'] <= 168]
    
    # Group by hour and calculate average likes
    shelf_life = df_subset.groupby('hour_bin').agg(
        avg_likes=('like_count', 'mean'),
        comment_count=('comment_id', 'count')
    ).reset_index()
    
    # Smooth the curve using a rolling average of 3 hours to remove noise
    shelf_life['smoothed_likes'] = shelf_life['avg_likes'].rolling(window=3, min_periods=1, center=True).mean()
    
    out_file = out_dir / "shelf_life_likes.parquet"
    shelf_life.to_parquet(out_file)
    print(f"✅ Generated shelf life curve (First 168 hours).")
    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    run_shelf_life()
