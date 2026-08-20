"""Stage 11: Reply Latency Distribution (s11_reply_latency.py)

Calculates the time difference (latency) between a root comment 
(or parent comment) and its replies to measure conversational velocity.
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

def run_reply_latency():
    print("⏱️ Starting Reply Latency Calculation (s11)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ No comments_clean.parquet found. Skipping s11.")
        return
        
    print("  -> Loading comments data...")
    df = pd.read_parquet(comments_file, columns=['comment_id', 'parent_id', 'published_at'])
    
    # Filter out comments with no parent (root comments have no latency)
    replies = df[df['parent_id'].notna() & (df['parent_id'] != "")].copy()
    out_file = out_dir / "reply_latency.parquet"
    
    if replies.empty:
        print("⚠️ No replies found in dataset. Writing empty latency parquet.")
        pd.DataFrame(columns=['comment_id', 'parent_id', 'latency_minutes', 'latency_hours']).to_parquet(out_file)
        return
        
    # We need the parent's published_at. We can do a self-join.
    # Create a lookup dictionary or dataframe for fast join
    parents = df[['comment_id', 'published_at']].rename(columns={
        'comment_id': 'parent_id', 
        'published_at': 'parent_published_at'
    })
    
    print("  -> Joining replies to parent timestamps...")
    # Merge replies with their parents
    merged = replies.merge(parents, on='parent_id', how='inner')
    
    # Convert to datetime
    merged['published_at'] = pd.to_datetime(merged['published_at'])
    merged['parent_published_at'] = pd.to_datetime(merged['parent_published_at'])
    
    print("  -> Calculating latency in seconds/minutes...")
    # Calculate difference
    merged['latency_seconds'] = (merged['published_at'] - merged['parent_published_at']).dt.total_seconds()
    
    # Filter out invalid negative latencies (e.g. data errors)
    valid_latency = merged[merged['latency_seconds'] >= 0].copy()
    
    valid_latency['latency_minutes'] = valid_latency['latency_seconds'] / 60.0
    valid_latency['latency_hours'] = valid_latency['latency_seconds'] / 3600.0
    
    out_file = out_dir / "reply_latency.parquet"
    # Save just the essential columns for plotting
    valid_latency[['comment_id', 'parent_id', 'latency_minutes', 'latency_hours']].to_parquet(out_file)
    print(f"✅ Calculated latency for {len(valid_latency)} replies.")
    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    run_reply_latency()
