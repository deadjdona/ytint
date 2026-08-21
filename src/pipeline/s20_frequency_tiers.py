"""Stage 20: Commenting Frequency Tiers (s20_frequency_tiers.py)

Groups authors into behavioral tiers based on their total comment volume,
revealing the distribution between casual viewers and hardcore super-fans.
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

def run_frequency_tiers():
    print("📊 Starting Commenting Frequency Tiers (s22)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    authors_file = out_dir / "authors_final.parquet"
    comments_file = interim_dir / "comments_clean.parquet"
    
    if authors_file.exists():
        print("  -> Utilizing precomputed author comment frequencies from authors_final.parquet...")
        df_authors = pd.read_parquet(authors_file, columns=['frequency']).dropna()
        author_counts = df_authors['frequency']
    elif comments_file.exists():
        print("  -> Loading comments...")
        df = pd.read_parquet(comments_file, columns=['author_channel_id'])
        df = df.dropna()
        if df.empty: return
        print("  -> Calculating user frequencies...")
        author_counts = df['author_channel_id'].value_counts()
    else:
        print("⚠️ Missing authors_final.parquet and comments_clean.parquet. Skipping s22.")
        return
        
    if author_counts.empty:
        return
    
    # Configuration parameters
    stage_cfg = config.get("stage_20_frequency_tiers", {})
    casual_max = int(stage_cfg.get("casual_max", 5))
    regular_max = int(stage_cfg.get("regular_max", 20))
    superfan_max = int(stage_cfg.get("superfan_max", 100))

    # Define Tiers
    def get_tier(count):
        if count == 1:
            return "1. One-and-Done (1)"
        elif count <= casual_max:
            return f"2. Casual (2-{casual_max})"
        elif count <= regular_max:
            return f"3. Regular ({casual_max + 1}-{regular_max})"
        elif count <= superfan_max:
            return f"4. Super-Fan ({regular_max + 1}-{superfan_max})"
        else:
            return f"5. Mega-Fan / Bot ({superfan_max}+)"
            
    tiers = author_counts.apply(get_tier)
    tier_distribution = tiers.value_counts().rename_axis('tier').reset_index(name='author_count')
    tier_distribution = tier_distribution.sort_values('tier')
    
    out_file = out_dir / "frequency_tiers.parquet"
    tier_distribution.to_parquet(out_file, index=False)
    
    print(f"✅ Generated Frequency Tiers for {len(author_counts)} unique authors.")

if __name__ == "__main__":
    run_frequency_tiers()
