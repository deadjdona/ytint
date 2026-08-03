"""Stage 28: Conversation Resolution Patterns (s28_resolution_patterns.py)

Analyzes the properties of the "Terminal Comment" (the final reply in any thread)
to understand how conversations typically resolve (e.g. positive vs negative).
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config

def run_resolution_patterns():
    print("🏁 Starting Conversation Resolution Analysis (s28)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s28.")
        return
        
    print("  -> Loading comments...")
    try:
        df = pd.read_parquet(comments_file, columns=['comment_id', 'parent_id', 'published_at', 'sentiment_label'])
    except ValueError:
        print("⚠️ Missing required columns (likely sentiment_label). Run earlier stages first. Skipping s28.")
        return
        
    replies = df[df['parent_id'].notna() & (df['parent_id'] != '')].copy()
    if replies.empty: return
    
    print("  -> Identifying terminal nodes (final word in thread)...")
    replies = replies.sort_values('published_at')
    
    terminal_nodes = replies.drop_duplicates(subset=['parent_id'], keep='last').copy()
    terminal_ids = set(terminal_nodes['comment_id'])
    
    replies['is_terminal'] = replies['comment_id'].isin(terminal_ids)
    
    terminal_dist = replies[replies['is_terminal']]['sentiment_label'].value_counts(normalize=True).reset_index()
    terminal_dist.columns = ['sentiment', 'terminal_ratio']
    
    non_terminal_dist = replies[~replies['is_terminal']]['sentiment_label'].value_counts(normalize=True).reset_index()
    non_terminal_dist.columns = ['sentiment', 'non_terminal_ratio']
    
    if non_terminal_dist.empty:
        non_terminal_dist = pd.DataFrame([{'sentiment': s, 'non_terminal_ratio': 0.0} for s in terminal_dist['sentiment']])
        
    comparison = pd.merge(terminal_dist, non_terminal_dist, on='sentiment', how='outer').fillna(0)
    
    comparison['terminal_ratio'] = comparison['terminal_ratio'] * 100
    comparison['non_terminal_ratio'] = comparison['non_terminal_ratio'] * 100
    
    out_file = out_dir / "resolution_patterns.parquet"
    comparison.to_parquet(out_file, index=False)
    
    print(f"✅ Generated Resolution Patterns across {len(terminal_nodes)} conversation threads.")

if __name__ == "__main__":
    run_resolution_patterns()
