"""Stage 23: Bot & Spammer Heuristics (s23_bot_heuristics.py)

Classifies authors as Humans vs Bots/Spammers based on behavioral heuristics:
- Comment volume (high volume)
- Duplication ratio (copy-pasting the exact same text)
- Lexical richness
"""

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
    
    print("  -> Aggregating author behaviors...")
    author_stats = df.groupby('author_channel_id').agg(
        total_comments=('text', 'count'),
        unique_comments=('text', 'nunique'),
        avg_lexical_richness=('lexical_richness', 'mean')
    ).reset_index()
    
    # Duplication Ratio: 1.0 means every comment is unique, 0.01 means massive copy-pasting
    author_stats['unique_ratio'] = author_stats['unique_comments'] / author_stats['total_comments']
    
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
