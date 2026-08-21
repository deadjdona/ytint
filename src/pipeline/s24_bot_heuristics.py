"""Stage 24: Bot & Spammer Heuristics Classifier (s24_bot_heuristics.py)

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
    
    # Enrich with upstream graph features from authors_final.parquet if present
    authors_file = out_dir / "authors_final.parquet"
    if authors_file.exists():
        print("  -> Integrating author graph centrality & name reuse from authors_final.parquet...")
        df_authors = pd.read_parquet(authors_file, columns=['author_channel_id', 'is_display_name_reused', 'pagerank', 'is_bot_suspect'])
        author_stats = author_stats.merge(df_authors, on='author_channel_id', how='left')
        author_stats['is_display_name_reused'] = author_stats['is_display_name_reused'].fillna(False)
        author_stats['is_bot_suspect'] = author_stats['is_bot_suspect'].fillna(False)
    else:
        author_stats['is_display_name_reused'] = False
        author_stats['is_bot_suspect'] = False

    # Configuration parameters
    stage_cfg = config.get("stage_24_bot_heuristics", {})
    t1_comments = int(stage_cfg.get("tier1_min_comments", 10))
    t1_ratio = float(stage_cfg.get("tier1_max_unique_ratio", 0.2))
    t2_comments = int(stage_cfg.get("tier2_min_comments", 50))
    t2_ratio = float(stage_cfg.get("tier2_max_unique_ratio", 0.5))
    rn_comments = int(stage_cfg.get("reused_name_min_comments", 5))
    rn_ratio = float(stage_cfg.get("reused_name_max_unique", 0.3))

    print("  -> Applying heuristic rules...")
    conditions = [
        (author_stats['total_comments'] >= t1_comments) & (author_stats['unique_ratio'] <= t1_ratio),
        (author_stats['total_comments'] >= t2_comments) & (author_stats['unique_ratio'] <= t2_ratio),
        (author_stats['total_comments'] >= rn_comments) & (author_stats['is_display_name_reused']) & (author_stats['unique_ratio'] <= rn_ratio),
        author_stats['is_bot_suspect']
    ]
    
    author_stats['is_bot'] = np.select(conditions, [True, True, True, True], default=False)
    author_stats['classification'] = np.where(author_stats['is_bot'], 'Bot/Spammer', 'Human/Organic')
    
    bot_count = author_stats['is_bot'].sum()
    print(f"  -> Detected {bot_count} highly probable bots out of {len(author_stats)} authors.")
    
    out_file = out_dir / "bot_classifications.parquet"
    author_stats.to_parquet(out_file, index=False)
    print("✅ Generated Bot Heuristics.")

if __name__ == "__main__":
    run_bot_heuristics()
