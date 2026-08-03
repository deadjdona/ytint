"""Stage 24: Drive-by vs Loyalists (s24_driveby_loyalists.py)

Classifies authors based on the breadth of their engagement (unique videos commented on),
distinguishing between those who only ever engaged with a single video (Drive-by)
versus those who return for multiple videos (Loyalists).
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config
import numpy as np

def run_driveby_loyalists():
    print("🚗 Starting Drive-by vs Loyalists Analysis (s24)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s24.")
        return
        
    print("  -> Loading comments...")
    df = pd.read_parquet(comments_file, columns=['author_channel_id', 'video_id'])
    df = df.dropna()
    if df.empty: return
    
    print("  -> Calculating unique video engagements per author...")
    author_videos = df.groupby('author_channel_id')['video_id'].nunique().reset_index()
    author_videos.columns = ['author_channel_id', 'unique_videos']
    
    # Classify
    conditions = [
        author_videos['unique_videos'] == 1,
        author_videos['unique_videos'] <= 3,
        author_videos['unique_videos'] > 3
    ]
    choices = [
        "1. Drive-by (1 Video)",
        "2. Casual (2-3 Videos)",
        "3. Loyalist (4+ Videos)"
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
