"""Stage 27: Thread Width & Branching (s27_thread_width.py)

Calculates the distribution of thread widths (i.e. how many replies a single parent comment receives).
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

def run_thread_width():
    print("🌲 Starting Thread Width Analysis (s27)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s27.")
        return
        
    print("  -> Loading comments...")
    df = pd.read_parquet(comments_file, columns=['comment_id', 'parent_id'])
    
    # We only care about comments that ARE replies (they have a parent)
    replies = df[df['parent_id'].notna() & (df['parent_id'] != '')]
    if replies.empty: return
    
    print("  -> Calculating branching factors (replies per parent)...")
    # Count how many replies each parent_id gets
    width_counts = replies['parent_id'].value_counts().reset_index()
    width_counts.columns = ['parent_id', 'thread_width']
    
    # We want the distribution of these widths (e.g., how many threads have 1 reply, 2 replies, etc.)
    width_distribution = width_counts['thread_width'].value_counts().reset_index()
    width_distribution.columns = ['thread_width', 'frequency']
    width_distribution = width_distribution.sort_values('thread_width')
    
    out_file = out_dir / "thread_width_dist.parquet"
    width_distribution.to_parquet(out_file, index=False)
    
    print(f"✅ Generated Thread Width Distribution across {len(width_counts)} conversational threads.")

if __name__ == "__main__":
    run_thread_width()
