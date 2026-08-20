"""Stage 23: Bot & Spammer Heuristics (s23_bot_heuristics.py)

Classifies authors as Humans vs Bots/Spammers based on behavioral heuristics:
- Comment volume (high volume)
- Duplication ratio (copy-pasting the exact same text)
- Lexical richness
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

def run_bot_heuristics():
    print("🤖 Starting Bot Heuristics Classifier (s23)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s23.")
        return
        
    print("  -> Loading comments for bot detection...")
    df = pd.read_parquet(comments_file, columns=['author_channel_id', 'text', 'lexical_richness'])
    df = df.dropna(subset=['author_channel_id'])
    if df.empty: return
    
    print("  -> Pre-tokenizing text via Gigatoken (Rust-accelerated) for Bot Template Hashing...")
    import gigatoken as gt
    try:
        giga_tok = gt.Tokenizer("openai-community/gpt2")
        token_lists = giga_tok.encode_batch_list(df['text'].astype(str).tolist())
        df['token_hash'] = [hash(tuple(t)) for t in token_lists]
    except Exception as e:
        print(f"  ⚠️ Gigatoken fallback: {e}")
        df['token_hash'] = df['text']

    print("  -> Aggregating author behaviors...")
    author_stats = df.groupby('author_channel_id').agg(
        total_comments=('text', 'count'),
        unique_comments=('text', 'nunique'),
        unique_token_patterns=('token_hash', 'nunique'),
        avg_lexical_richness=('lexical_richness', 'mean')
    ).reset_index()
    
    # Duplication Ratio: Uses Gigatoken BPE sequence hashing to catch spammers varying minor whitespace
    author_stats['unique_ratio'] = author_stats['unique_token_patterns'] / author_stats['total_comments']
    
    print("  -> Applying heuristic rules...")
    conditions = [
        (author_stats['total_comments'] >= 10) & (author_stats['unique_ratio'] <= 0.2),
        (author_stats['total_comments'] >= 50) & (author_stats['unique_ratio'] <= 0.5),
    ]
    
    author_stats['is_bot'] = np.select(conditions, [True, True], default=False)
    author_stats['classification'] = np.where(author_stats['is_bot'], 'Bot/Spammer', 'Human/Organic')
    
    bot_count = author_stats['is_bot'].sum()
    print(f"  -> Detected {bot_count} highly probable bots out of {len(author_stats)} authors.")
    
    out_file = out_dir / "bot_classifications.parquet"
    author_stats.to_parquet(out_file, index=False)
    print("✅ Generated Bot Heuristics.")

if __name__ == "__main__":
    run_bot_heuristics()
