"""Stage 21: Drive-by vs Loyalists Segmentation (s21_driveby_loyalists.py)

Classifies authors based on the breadth of their engagement (unique videos commented on),
distinguishing between those who only ever engaged with a single video (Drive-by)
versus those who return for multiple videos (Loyalists).
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
import numpy as np

def run_driveby_loyalists():
    print("🚗 Starting Drive-by vs Loyalists Analysis (s24)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    authors_file = out_dir / "authors_final.parquet"
    comments_file = interim_dir / "comments_clean.parquet"
    
    if authors_file.exists():
        print("  -> Utilizing precomputed author metrics from authors_final.parquet...")
        df_authors = pd.read_parquet(authors_file, columns=['author_channel_id', 'unique_videos_commented'])
        author_videos = df_authors[['author_channel_id', 'unique_videos_commented']].dropna()
        author_videos.columns = ['author_channel_id', 'unique_videos']
    elif comments_file.exists():
        print("  -> Loading comments...")
        df = pd.read_parquet(comments_file, columns=['author_channel_id', 'video_id'])
        df = df.dropna()
        if df.empty: return
        print("  -> Calculating unique video engagements per author...")
        author_videos = df.groupby('author_channel_id')['video_id'].nunique().reset_index()
        author_videos.columns = ['author_channel_id', 'unique_videos']
    else:
        print("⚠️ Missing authors_final.parquet and comments_clean.parquet. Skipping s24.")
        return
        
    if author_videos.empty:
        return
    
    # Configuration parameters
    stage_cfg = config.get("stage_21_driveby_loyalists", {})
    driveby_max = int(stage_cfg.get("driveby_max_videos", 1))
    casual_max = int(stage_cfg.get("casual_max_videos", 3))

    # Classify
    conditions = [
        author_videos['unique_videos'] <= driveby_max,
        author_videos['unique_videos'] <= casual_max,
        author_videos['unique_videos'] > casual_max
    ]
    choices = [
        f"1. Drive-by ({driveby_max} Video{'s' if driveby_max > 1 else ''})",
        f"2. Casual ({driveby_max + 1}-{casual_max} Videos)",
        f"3. Loyalist ({casual_max + 1}+ Videos)"
    ]
    
    author_videos['classification'] = np.select(conditions, choices, default="Unknown")
    
    distribution = author_videos['classification'].value_counts().reset_index()
    distribution.columns = ['classification', 'author_count']
    distribution = distribution.sort_values('classification')
    
    out_file = out_dir / "driveby_loyalists.parquet"
    distribution.to_parquet(out_file, index=False)
    
    print(f"✅ Generated Drive-by vs Loyalists distribution for {len(author_videos)} authors.")

if __name__ == "__main__":
    run_driveby_loyalists()
