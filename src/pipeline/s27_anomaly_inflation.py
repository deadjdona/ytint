"""Stage 27: Suspicious Like Inflation & Astroturfing (s27_anomaly_inflation.py)

Detects "Like Inflation" (botting/astroturfing).
A typical highly-liked YouTube comment organically generates a deep reply thread.
If a comment has massive likes but zero replies, or a mathematically anomalous Like-to-Reply ratio,
it is highly indicative of bot-driven like inflation.
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
    df = pd.read_parquet(comments_file, columns=['comment_id', 'parent_id', 'like_count', 'author_channel_id'])
    
    # Calculate reply counts
    reply_counts = df[df['parent_id'].notna() & (df['parent_id'] != '')]['parent_id'].value_counts().reset_index()
    reply_counts.columns = ['comment_id', 'reply_count']
    
    # We only care about top-level comments for like inflation
    top_level = df[df['parent_id'].isna() | (df['parent_id'] == '')].copy()
    top_level = top_level.merge(reply_counts, on='comment_id', how='left').fillna({'reply_count': 0})
    
    # Cross-reference with bot classifications from upstream stage if available
    bot_file = out_dir / "bot_classifications.parquet"
    if bot_file.exists():
        print("  -> Cross-referencing bot classifications from bot_classifications.parquet...")
        df_bots = pd.read_parquet(bot_file, columns=['author_channel_id', 'is_bot'])
        top_level = top_level.merge(df_bots, on='author_channel_id', how='left')
        top_level['is_bot_author'] = top_level['is_bot'].fillna(False)
    else:
        top_level['is_bot_author'] = False

    print("  -> Calculating inflation ratios...")
    # Inflation ratio: Likes per Reply (add 1 to avoid div zero)
    top_level['inflation_ratio'] = top_level['like_count'] / (top_level['reply_count'] + 1)
    
    # Configuration parameters
    stage_cfg = config.get("stage_27_like_inflation", {})
    min_like_count = int(stage_cfg.get("min_like_count", 5))
    inflation_quantile = float(stage_cfg.get("inflation_quantile", 0.99))
    min_inflation_ratio = float(stage_cfg.get("min_inflation_ratio", 20.0))
    bot_inflation_ratio = float(stage_cfg.get("bot_inflation_ratio", 10.0))

    # We only care if the absolute like_count is non-trivial (e.g. >= min_like_count)
    analyzed = top_level[top_level['like_count'] >= min_like_count].copy()
    
    if analyzed.empty:
        print("⚠️ Not enough likes to compute inflation. Skipping.")
        return
        
    # Flag top anomalous quantile of inflation ratios or bot-authored high likes as suspicious
    threshold = analyzed['inflation_ratio'].quantile(inflation_quantile)
    if threshold < min_inflation_ratio:
        threshold = min_inflation_ratio
    
    analyzed['is_suspicious'] = (analyzed['inflation_ratio'] >= threshold) | (analyzed['is_bot_author'] & (analyzed['inflation_ratio'] >= bot_inflation_ratio))
    
    out_file = out_dir / "like_inflation.parquet"
    
    # Save a subset to avoid giant files
    plot_data = analyzed[['comment_id', 'like_count', 'reply_count', 'is_suspicious', 'inflation_ratio', 'is_bot_author']]
    plot_data.to_parquet(out_file, index=False)
    
    num_sus = plot_data['is_suspicious'].sum()
    print(f"✅ Flagged {num_sus} comments as exhibiting Suspicious Like Inflation (out of {len(plot_data)} candidates).")

if __name__ == "__main__":
    run_like_inflation()
