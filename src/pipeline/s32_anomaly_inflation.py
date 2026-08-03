"""Stage 32: Suspicious Like Inflation (s32_anomaly_inflation.py)

Detects "Like Inflation" (botting/astroturfing).
A typical highly-liked YouTube comment organically generates a deep reply thread.
If a comment has massive likes but zero replies, or a mathematically anomalous Like-to-Reply ratio,
it is highly indicative of bot-driven like inflation.
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config

def run_like_inflation():
    print("📈 Starting Suspicious Like Inflation Analysis (s32)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s32.")
        return
        
    print("  -> Loading comments...")
    df = pd.read_parquet(comments_file, columns=['comment_id', 'parent_id', 'like_count'])
    
    # Calculate reply counts
    reply_counts = df[df['parent_id'].notna() & (df['parent_id'] != '')]['parent_id'].value_counts().reset_index()
    reply_counts.columns = ['comment_id', 'reply_count']
    
    # We only care about top-level comments for like inflation
    top_level = df[df['parent_id'].isna() | (df['parent_id'] == '')].copy()
    top_level = top_level.merge(reply_counts, on='comment_id', how='left').fillna({'reply_count': 0})
    
    print("  -> Calculating inflation ratios...")
    # Inflation ratio: Likes per Reply (add 1 to avoid div zero)
    top_level['inflation_ratio'] = top_level['like_count'] / (top_level['reply_count'] + 1)
    
    # We only care if the absolute like_count is non-trivial (e.g. > 5 likes)
    analyzed = top_level[top_level['like_count'] >= 5].copy()
    
    if analyzed.empty:
        print("⚠️ Not enough likes to compute inflation. Skipping.")
        return
        
    # Flag top 1% of inflation ratios as suspicious
    threshold = analyzed['inflation_ratio'].quantile(0.99)
    # Threshold must be at least somewhat high to be considered anomalous (e.g. 50 likes : 0 replies)
    if threshold < 20: threshold = 20 
    
    analyzed['is_suspicious'] = analyzed['inflation_ratio'] >= threshold
    
    out_file = out_dir / "like_inflation.parquet"
    
    # Save a subset to avoid giant files
    plot_data = analyzed[['comment_id', 'like_count', 'reply_count', 'is_suspicious', 'inflation_ratio']]
    plot_data.to_parquet(out_file, index=False)
    
    num_sus = plot_data['is_suspicious'].sum()
    print(f"✅ Flagged {num_sus} comments as exhibiting Suspicious Like Inflation (out of {len(plot_data)} candidates).")

if __name__ == "__main__":
    run_like_inflation()
