"""Stage 22: Author Stylometric Fingerprints (s22_author_fingerprints.py)

Generates a behavioral fingerprint for the most active authors on the channel.
Profiles them based on Positivity, Negativity, Verbosity, Branching (replies received),
and Reply Rate (replies given).
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config
import numpy as np
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

def run_author_fingerprints():
    print("🕵️ Starting Author Fingerprinting (s33)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s33.")
        return
        
    print("  -> Loading comments...")
    comments_df = pd.read_parquet(comments_file)
    text_col = 'text_original' if 'text_original' in comments_df.columns else ('text' if 'text' in comments_df.columns else None)
    
    df = comments_df.dropna(subset=['author_channel_id']).copy()
    if text_col:
        df['text_original'] = df[text_col]
    else:
        df['text_original'] = ""
    if 'comment_id' not in df.columns:
        df['comment_id'] = [f"c{i}" for i in range(len(df))]
    if 'parent_id' not in df.columns:
        df['parent_id'] = None
    if 'sentiment_label' not in df.columns:
        df['sentiment_label'] = 'NEUTRAL'
    if df.empty: return
    
    # 1. Base counts
    df['comment_length'] = df['text_original'].astype(str).str.len()
    df['is_reply'] = df['parent_id'].notna() & (df['parent_id'] != '')
    
    if 'bpe_token_count' not in df.columns:
        import gigatoken as gt
        try:
            g_tok = gt.Tokenizer("openai-community/gpt2")
            tok_lists = g_tok.encode_batch_list(df['text_original'].astype(str).tolist())
            df['bpe_token_count'] = [len(t) for t in tok_lists]
        except Exception:
            df['bpe_token_count'] = df['comment_length']
    
    author_stats = df.groupby('author_channel_id').agg(
        total_comments=('comment_id', 'count'),
        positive_count=('sentiment_label', lambda x: (x == 'POSITIVE').sum()),
        negative_count=('sentiment_label', lambda x: (x == 'NEGATIVE').sum()),
        avg_length=('comment_length', 'mean'),
        avg_bpe_tokens=('bpe_token_count', 'mean'),
        replies_made=('is_reply', 'sum')
    ).reset_index()
    
    # Calculate Branching (replies received)
    reply_counts = df[df['is_reply']]['parent_id'].value_counts().reset_index()
    reply_counts.columns = ['comment_id', 'replies_received']
    
    # Join back to original df to sum by author
    df_with_rc = df.merge(reply_counts, on='comment_id', how='left').fillna({'replies_received': 0})
    received_stats = df_with_rc.groupby('author_channel_id')['replies_received'].sum().reset_index()
    
    author_stats = author_stats.merge(received_stats, on='author_channel_id', how='left')
    
    # Filter to at least 10 comments to have a stable fingerprint
    valid_authors = author_stats[author_stats['total_comments'] >= 10].copy()
    if valid_authors.empty:
        valid_authors = author_stats.nlargest(5, 'total_comments').copy()
        
    if valid_authors.empty: return
    
    # Generate Ratios
    valid_authors['positivity'] = valid_authors['positive_count'] / valid_authors['total_comments']
    valid_authors['negativity'] = valid_authors['negative_count'] / valid_authors['total_comments']
    valid_authors['reply_rate'] = valid_authors['replies_made'] / valid_authors['total_comments']
    valid_authors['branching_factor'] = valid_authors['replies_received'] / valid_authors['total_comments']
    
    # Take top 5 most active
    top_authors = valid_authors.nlargest(5, 'total_comments').copy()
    
    # Normalize for radar chart (0 to 1)
    for col in ['positivity', 'negativity', 'avg_length', 'reply_rate', 'branching_factor']:
        max_val = valid_authors[col].max()
        if max_val > 0:
            top_authors[f'{col}_norm'] = top_authors[col] / max_val
        else:
            top_authors[f'{col}_norm'] = 0
            
    out_file = out_dir / "author_fingerprints.parquet"
    top_authors.to_parquet(out_file, index=False)
    
    print(f"✅ Generated fingerprints for the top {len(top_authors)} most active authors.")

if __name__ == "__main__":
    run_author_fingerprints()
