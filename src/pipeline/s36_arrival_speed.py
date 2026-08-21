"""Stage 36: Arrival Speed Dynamics (s36_arrival_speed.py)

Classifies authors based on their average response time (days since upload).
Distinguishes between "First Responders" (always comment immediately) and "Necromancers" (comment on years-old videos).
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

def run_arrival_speed():
    print("⏱️ Starting Arrival Speed Classification (s26)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s26.")
        return
        
    print("  -> Loading comments...")
    df = pd.read_parquet(comments_file, columns=['author_channel_id', 'minutes_since_upload'])
    df['days_since_upload'] = df['minutes_since_upload'] / 1440.0
    df = df.dropna()
    if df.empty: return
    
    print("  -> Calculating average arrival speeds...")
    author_speed = df.groupby('author_channel_id')['days_since_upload'].mean().reset_index()
    
    # Classify
    conditions = [
        author_speed['days_since_upload'] < 1,
        author_speed['days_since_upload'] <= 7,
        author_speed['days_since_upload'] <= 30,
        author_speed['days_since_upload'] > 30
    ]
    choices = [
        "1. First Responder (<1 Day)",
        "2. On-Time (1-7 Days)",
        "3. Late Arrival (8-30 Days)",
        "4. Necromancer (>30 Days)"
    ]
    
    author_speed['classification'] = np.select(conditions, choices, default="Unknown")
    
    distribution = author_speed['classification'].value_counts().rename_axis('classification').reset_index(name='author_count')
    distribution = distribution.sort_values('classification')
    
    out_file = out_dir / "arrival_speed.parquet"
    distribution.to_parquet(out_file, index=False)
    
    print(f"✅ Generated Arrival Speed distribution for {len(author_speed)} authors.")

if __name__ == "__main__":
    run_arrival_speed()
