"""Stage 14: Thread Polarization Dynamics (s14_polarization.py)

Analyzes sentiment trajectories within deep reply threads to model 
if and how conversations polarize or decay over depth.
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
import numpy as np
from pathlib import Path
from engine.config_loader import load_config
from scipy.stats import linregress

def run_polarization_dynamics():
    print("📉 Starting Thread Polarization Dynamics (s09)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ No comments_clean.parquet found. Skipping s09.")
        return
        
    print("  -> Loading comments data...")
    df = pd.read_parquet(comments_file)
    
    if 'vader_compound' not in df.columns or 'comment_depth' not in df.columns:
        print("⚠️ Missing 'vader_compound' or 'comment_depth'. Cannot run polarization analysis.")
        return

    # 1. Overall Corpus Sentiment by Depth
    print("  -> Calculating corpus-level sentiment by depth...")
    depth_agg = df.groupby('comment_depth').agg(
        mean_sentiment=('vader_compound', 'mean'),
        comment_count=('comment_id', 'count')
    ).reset_index()
    
    # 2. Thread-Level Decay Slope
    # If thread_id is not explicitly defined, we assume all replies belonging to the same root 
    # comment share a common parent tree. YouTube API usually provides a `parent_id`. 
    # For a flat dataset, we'll map top-level comments as the root.
    if 'thread_id' not in df.columns and 'parent_id' in df.columns:
        print("  -> Constructing thread groups...")
        # A simple approximation: For YouTube, typically replies have `parent_id` equal to the root comment ID.
        # So thread_id = parent_id if parent_id is not NA, else comment_id
        df['thread_id'] = df['parent_id'].fillna(df['comment_id'])
    
    if 'thread_id' in df.columns:
        print("  -> Calculating per-thread sentiment decay slopes (for depth >= 3)...")
        # Find threads that have at least depth 3
        max_depths = df.groupby('thread_id')['comment_depth'].max()
        deep_threads = max_depths[max_depths >= 3].index
        
        df_deep = df[df['thread_id'].isin(deep_threads)].copy()
        
        slopes = []
        for thread_id, group in df_deep.groupby('thread_id'):
            # Sort by depth
            group = group.sort_values('comment_depth')
            # If multiple comments at same depth, take mean sentiment for that depth
            group_agg = group.groupby('comment_depth')['vader_compound'].mean().reset_index()
            
            if len(group_agg) >= 3:
                # Calculate linear regression slope
                slope, intercept, r_value, p_value, std_err = linregress(
                    group_agg['comment_depth'], 
                    group_agg['vader_compound']
                )
                slopes.append({
                    'thread_id': thread_id,
                    'decay_slope': slope,
                    'max_depth': group_agg['comment_depth'].max(),
                    'comment_count': len(group)
                })
                
        df_slopes = pd.DataFrame(slopes)
        
        if not df_slopes.empty:
            out_slopes_file = out_dir / "thread_decay_slopes.parquet"
            df_slopes.to_parquet(out_slopes_file)
            print(f"✅ Saved thread decay slopes to {out_slopes_file}")
            
            # Print a quick summary
            avg_slope = df_slopes['decay_slope'].mean()
            print(f"   -> Average thread sentiment slope: {avg_slope:.4f} (Negative = decays into toxicity)")
            
    out_file = out_dir / "thread_polarization_corpus.parquet"
    depth_agg.to_parquet(out_file)
    print(f"✅ Saved corpus depth aggregation to {out_file}")

if __name__ == "__main__":
    run_polarization_dynamics()
