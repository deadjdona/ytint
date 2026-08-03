"""Stage 22: Commenting Frequency Tiers (s22_frequency_tiers.py)

Groups authors into behavioral tiers based on their total comment volume,
revealing the distribution between casual viewers and hardcore super-fans.
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config

def run_frequency_tiers():
    print("📊 Starting Commenting Frequency Tiers (s22)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s22.")
        return
        
    print("  -> Loading comments...")
    df = pd.read_parquet(comments_file, columns=['author_channel_id'])
    df = df.dropna()
    
    if df.empty:
        return
        
    print("  -> Calculating user frequencies...")
    author_counts = df['author_channel_id'].value_counts()
    
    # Define Tiers
    def get_tier(count):
        if count == 1:
            return "1. One-and-Done (1)"
        elif count <= 5:
            return "2. Casual (2-5)"
        elif count <= 20:
            return "3. Regular (6-20)"
        elif count <= 100:
            return "4. Super-Fan (21-100)"
        else:
            return "5. Mega-Fan / Bot (100+)"
            
    tiers = author_counts.apply(get_tier)
    tier_distribution = tiers.value_counts().reset_index()
    tier_distribution.columns = ['tier', 'author_count']
    tier_distribution = tier_distribution.sort_values('tier')
    
    out_file = out_dir / "frequency_tiers.parquet"
    tier_distribution.to_parquet(out_file, index=False)
    
    print(f"✅ Generated Frequency Tiers for {len(author_counts)} unique authors.")

if __name__ == "__main__":
    run_frequency_tiers()
