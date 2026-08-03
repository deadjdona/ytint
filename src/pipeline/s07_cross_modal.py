"""Stage 7: Cross-Modal Reaction Mapping (s07_cross_modal.py)

Extracts video timestamps from comments (e.g., "1:24") and aggregates 
sentiment/toxicity metrics at specific playback seconds for each video.
"""

import re
import pandas as pd
from pathlib import Path
from engine.config_loader import load_config

def parse_timestamp_to_seconds(ts: str) -> int:
    """Parses 'MM:SS' or 'H:MM:SS' strings into total integer seconds."""
    parts = ts.split(':')
    try:
        if len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return m * 60 + s
        elif len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + s
    except ValueError:
        return -1
    return -1

def run_cross_modal():
    print("🎬 Starting Cross-Modal Reaction Mapping (s07)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    
    if not comments_file.exists():
        print("⚠️ No comments_clean.parquet found. Skipping s07.")
        return
        
    print("  -> Loading comments data...")
    df = pd.read_parquet(comments_file)
    
    text_col = 'text' if 'text' in df.columns else ('text_original' if 'text_original' in df.columns else None)
    if not text_col or 'video_id' not in df.columns or 'comment_id' not in df.columns:
        print("⚠️ Missing required text or metadata columns. Skipping s07.")
        return
            
    print("  -> Extracting inline timestamps from comments...")
    # Regex for MM:SS or H:MM:SS
    pattern = r'\b(\d{1,2}:\d{2}(?::\d{2})?)\b'
    
    # Extract timestamps into a list for each comment
    df['extracted_ts'] = df[text_col].astype(str).apply(lambda x: re.findall(pattern, x))
    
    # Filter to only comments that actually have timestamps
    df_ts = df[df['extracted_ts'].str.len() > 0].copy()
    
    if df_ts.empty:
        print("⚠️ No timestamps found in dataset. Outputting empty reaction map.")
        empty_df = pd.DataFrame(columns=['video_id', 'second', 'vader_compound', 'toxicity', 'comment_count'])
        empty_df.to_parquet(out_dir / "video_reaction_map.parquet")
        return
        
    # Explode so each timestamp gets its own row
    df_exploded = df_ts.explode('extracted_ts')
    
    print("  -> Parsing timestamps to absolute seconds...")
    df_exploded['second'] = df_exploded['extracted_ts'].apply(parse_timestamp_to_seconds)
    df_valid = df_exploded[df_exploded['second'] >= 0].copy()
    
    # Group by Video and Second
    print("  -> Aggregating sentiment and toxicity per video second...")
    agg_dict = {'comment_id': 'count'}
    if 'vader_compound' in df_valid.columns:
        agg_dict['vader_compound'] = 'mean'
    if 'toxicity' in df_valid.columns:
        agg_dict['toxicity'] = 'mean'
        
    df_agg = df_valid.groupby(['video_id', 'second']).agg(agg_dict).reset_index()
    df_agg.rename(columns={'comment_id': 'comment_count'}, inplace=True)
    
    out_file = out_dir / "video_reaction_map.parquet"
    df_agg.to_parquet(out_file)
    print(f"✅ Extracted {len(df_agg)} reaction points across {df_agg['video_id'].nunique()} videos.")
    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    run_cross_modal()
