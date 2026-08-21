"""Stage 34: Cross-Video Audience Overlap (s34_overlap.py)

Calculates the Jaccard similarity (audience overlap) between every pair of videos,
revealing which videos share the most commenters.
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
from itertools import combinations

def run_overlap():
    print("🕸️ Starting Cross-Video Overlap Analysis (s20)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s20.")
        return
        
    print("  -> Loading interactions...")
    df = pd.read_parquet(comments_file, columns=['author_channel_id', 'video_id'])
    df = df.dropna()
    
    if df.empty:
        return
        
    print("  -> Calculating audience intersections...")
    video_authors = df.groupby('video_id')['author_channel_id'].apply(set).to_dict()
    videos = list(video_authors.keys())
    
    if len(videos) > 60:
        print("  -> Truncating to top 60 most commented videos for readability.")
        top_vids = df['video_id'].value_counts().head(60).index.tolist()
        videos = [v for v in videos if v in top_vids]
        video_authors = {v: video_authors[v] for v in videos}
    
    matrix = pd.DataFrame(index=videos, columns=videos, dtype=float)
    
    for v in videos:
        matrix.loc[v, v] = 1.0
        
    for v1, v2 in combinations(videos, 2):
        set1 = video_authors[v1]
        set2 = video_authors[v2]
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        jaccard = intersection / union if union > 0 else 0
        
        matrix.loc[v1, v2] = jaccard
        matrix.loc[v2, v1] = jaccard
        
    out_file = out_dir / "video_overlap_matrix.parquet"
    matrix.reset_index(names=['video_id']).to_parquet(out_file, index=False)
    
    print(f"✅ Generated {len(videos)}x{len(videos)} Cross-Video Overlap Matrix.")

if __name__ == "__main__":
    run_overlap()
