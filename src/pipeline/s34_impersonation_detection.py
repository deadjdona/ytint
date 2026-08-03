"""Stage 34: Impersonation Detection (s34_impersonation_detection.py)

Detects display-name spoofing, a common YouTube fraud tactic where scammers create 
new accounts with the exact same display name as the channel creator or famous users
to deceive commenters in the replies.
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config

def run_impersonation_detection():
    print("🕵️ Starting Impersonation / Spoofing Detection (s34)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s34.")
        return
        
    print("  -> Loading comments...")
    comments_df = pd.read_parquet(comments_file)
    name_col = 'author_display_name' if 'author_display_name' in comments_df.columns else ('author_name' if 'author_name' in comments_df.columns else None)
    if not name_col or 'author_channel_id' not in comments_df.columns:
        print("⚠️ Missing author columns. Skipping s34.")
        return
        
    df = comments_df.dropna(subset=['author_channel_id', name_col]).copy()
    df['author_display_name'] = df[name_col]
    if df.empty: return
    
    # Clean up standard generic names by enforcing minimum length and removing basic names
    df['name_len'] = df['author_display_name'].str.len()
    valid_names = df[df['name_len'] >= 6].copy()
    
    print("  -> Grouping distinct channel IDs by Display Name...")
    impersonation = valid_names.groupby('author_display_name').agg(
        unique_accounts=('author_channel_id', 'nunique'),
        total_comments=('author_channel_id', 'count')
    ).reset_index()
    
    # We are looking for names used by multiple distinct channel IDs
    suspects = impersonation[impersonation['unique_accounts'] > 1].copy()
    
    if suspects.empty:
        print("✅ No impersonation detected in this dataset.")
        return
        
    suspects = suspects.sort_values(by=['unique_accounts', 'total_comments'], ascending=[False, False])
    
    out_file = out_dir / "impersonation_detection.parquet"
    
    # Limit to Top 50 to avoid massive files for simple bar charts
    plot_data = suspects.head(50)
    plot_data.to_parquet(out_file, index=False)
    
    print(f"✅ Found {len(suspects)} display names that are being shared by multiple distinct accounts.")
    print(f"   Top spoofed name: '{suspects.iloc[0]['author_display_name']}' with {suspects.iloc[0]['unique_accounts']} accounts.")

if __name__ == "__main__":
    run_impersonation_detection()
