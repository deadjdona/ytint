"""Stage 29: Initiator/Response Patterns (s29_initiator_patterns.py)

Classifies authors based on their conversational role:
- Broadcaster: Only starts threads, never replies.
- Responder: Only replies to others, never starts threads.
- Conversationalist: Does both.
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config
import numpy as np

def run_initiator_patterns():
    print("🗣️ Starting Initiator vs Responder Analysis (s29)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s29.")
        return
        
    print("  -> Loading comments...")
    df = pd.read_parquet(comments_file, columns=['author_channel_id', 'parent_id'])
    df = df.dropna(subset=['author_channel_id'])
    if df.empty: return
    
    df['is_reply'] = df['parent_id'].notna() & (df['parent_id'] != '')
    
    print("  -> Categorizing author behaviors...")
    author_stats = df.groupby('author_channel_id').agg(
        total_comments=('is_reply', 'count'),
        replies_made=('is_reply', 'sum')
    ).reset_index()
    
    author_stats['top_level_made'] = author_stats['total_comments'] - author_stats['replies_made']
    
    conditions = [
        (author_stats['top_level_made'] > 0) & (author_stats['replies_made'] == 0),
        (author_stats['top_level_made'] == 0) & (author_stats['replies_made'] > 0),
        (author_stats['top_level_made'] > 0) & (author_stats['replies_made'] > 0)
    ]
    choices = [
        "1. Broadcaster (Only Starts Threads)",
        "2. Responder (Only Replies)",
        "3. Conversationalist (Starts & Replies)"
    ]
    
    author_stats['role'] = np.select(conditions, choices, default="Unknown")
    
    role_dist = author_stats['role'].value_counts().reset_index()
    role_dist.columns = ['role', 'author_count']
    role_dist = role_dist.sort_values('role')
    
    out_file = out_dir / "initiator_patterns.parquet"
    role_dist.to_parquet(out_file, index=False)
    
    print(f"✅ Generated Initiator Patterns for {len(author_stats)} authors.")

if __name__ == "__main__":
    run_initiator_patterns()
