"""Stage 25: Position Bias & Branching Factor (s25_position_bias.py)

Analyzes the "Early Bird" effect by calculating the average likes and replies (branching factor)
a comment receives based on its chronological position in the video's comment section.
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

def run_position_bias():
    print("📍 Starting Position Bias & Branching Factor Analysis (s25)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s25.")
        return
        
    print("  -> Loading comments...")
    df = pd.read_parquet(comments_file, columns=['comment_id', 'video_id', 'parent_id', 'like_count', 'published_at'])
    df = df.dropna(subset=['video_id', 'published_at', 'comment_id'])
    
    if df.empty: return
    
    print("  -> Calculating branching factor (replies per comment)...")
    reply_counts = df['parent_id'].value_counts().reset_index()
    reply_counts.columns = ['comment_id', 'reply_count']
    
    df = df.merge(reply_counts, on='comment_id', how='left')
    df['reply_count'] = df['reply_count'].fillna(0)
    
    # We only want to look at TOP-LEVEL comments for Position Bias
    top_level = df[df['parent_id'].isna() | (df['parent_id'] == '')].copy()
    
    print("  -> Assigning chronological positions...")
    top_level = top_level.sort_values(['video_id', 'published_at'])
    top_level['position'] = top_level.groupby('video_id').cumcount() + 1
    
    print("  -> Aggregating by position (up to top 100)...")
    position_stats = top_level[top_level['position'] <= 100].groupby('position').agg(
        avg_likes=('like_count', 'mean'),
        avg_replies=('reply_count', 'mean'),
        comment_count=('comment_id', 'count')
    ).reset_index()
    
    out_file = out_dir / "position_bias.parquet"
    position_stats.to_parquet(out_file, index=False)
    
    print(f"✅ Generated Position Bias data across {len(position_stats)} chronological positions.")

if __name__ == "__main__":
    run_position_bias()
