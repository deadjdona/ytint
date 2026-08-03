"""Stage 8: Integrity & Bot Detection (s08_integrity.py)

Uses MinHash LSH for near-duplicate spam cluster detection and rolling 
time-windows for temporal bot burst/brigading detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from engine.config_loader import load_config
from tqdm import tqdm
tqdm.pandas()

def run_integrity_detection():
    print("🛡️ Starting Integrity & Bot Detection (s08)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ No comments_clean.parquet found. Skipping s08.")
        return
        
    print("  -> Loading comments data...")
    df = pd.read_parquet(comments_file)
    text_col = 'text' if 'text' in df.columns else ('text_original' if 'text_original' in df.columns else None)
    if not text_col:
        print("⚠️ Missing text column in comments. Skipping s08.")
        return
    df['published_at'] = pd.to_datetime(df['published_at'])
    
    # --- 1. MinHash LSH Duplicate Clustering ---
    print("  -> Pre-tokenizing text via Gigatoken (Rust-accelerated) for MinHash LSH...")
    import gigatoken as gt
    try:
        giga_tok = gt.Tokenizer("openai-community/gpt2")
        all_token_lists = giga_tok.encode_batch_list(df[text_col].astype(str).tolist())
    except Exception as e:
        print(f"  ⚠️ Gigatoken fallback: {e}")
        all_token_lists = None

    # Using 128 permutations and threshold of 0.8 Jaccard similarity
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    minhashes = {}
    
    def compute_minhash(text, tokens=None):
        m = MinHash(num_perm=128)
        if tokens:
            for tok in tokens:
                m.update(str(tok).encode('utf8'))
        else:
            text_str = str(text).lower()
            for i in range(len(text_str) - 2):
                m.update(text_str[i:i+3].encode('utf8'))
        return m

    # Compute hashes
    for i, row in tqdm(enumerate(df.itertuples()), total=len(df), desc="Hashing"):
        cid = getattr(row, 'comment_id')
        txt = getattr(row, text_col)
        toks = all_token_lists[i] if all_token_lists is not None else None
        m = compute_minhash(txt, tokens=toks)
        minhashes[cid] = m
        lsh.insert(cid, m)
        
    print("  -> Querying LSH for spam clusters...")
    df['spam_cluster_id'] = -1
    cluster_counter = 1
    
    # Keep track of which items are already assigned a cluster
    assigned = set()
    
    spam_cluster_map = {}
    
    for c_id, m in tqdm(minhashes.items(), desc="Querying"):
        if c_id in assigned:
            continue
            
        result = lsh.query(m)
        # If we have a cluster of duplicates (more than 1 identical/near-identical)
        if len(result) > 1:
            for duplicate_id in result:
                spam_cluster_map[duplicate_id] = cluster_counter
                assigned.add(duplicate_id)
            cluster_counter += 1
        else:
            spam_cluster_map[c_id] = -1
            assigned.add(c_id)
            
    df['spam_cluster_id'] = df['comment_id'].map(spam_cluster_map)
    df['is_duplicate'] = df['spam_cluster_id'] != -1
    
    # --- 2. Temporal Burst Detection ---
    print("  -> Detecting temporal bursts and bot brigading...")
    df = df.sort_values(by=['video_id', 'published_at'])
    
    df['brigade_suspect'] = False
    
    # Process each video separately to find isolated bursts
    for video_id, group in df.groupby('video_id'):
        # Resample to 5-minute bins and count comments
        time_series = group.set_index('published_at').resample('5min').size()
        
        # Calculate median volume (excluding zero-bins)
        median_vol = time_series[time_series > 0].median()
        if pd.isna(median_vol) or median_vol == 0:
            median_vol = 1
            
        # Burst threshold: 10x the normal median rate
        threshold = median_vol * 10
        burst_windows = time_series[time_series > threshold].index
        
        # If any bursts found, flag all comments in those 5-minute windows
        if not burst_windows.empty:
            for bw in burst_windows:
                start_time = bw
                end_time = bw + pd.Timedelta(minutes=5)
                
                # Flag the specific rows
                mask = (df['video_id'] == video_id) & \
                       (df['published_at'] >= start_time) & \
                       (df['published_at'] < end_time)
                df.loc[mask, 'brigade_suspect'] = True
                
    # --- Output ---
    out_file = interim_dir / "integrity_flags.parquet"
    out_cols = ['comment_id', 'spam_cluster_id', 'is_duplicate', 'brigade_suspect']
    df[out_cols].to_parquet(out_file)
    
    dupes = df['is_duplicate'].sum()
    brigades = df['brigade_suspect'].sum()
    print(f"✅ Flagged {dupes} duplicate/spam comments.")
    print(f"✅ Flagged {brigades} comments in suspected temporal brigading bursts.")
    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    run_integrity_detection()
