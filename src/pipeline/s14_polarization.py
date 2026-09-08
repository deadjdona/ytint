"""Stage 14: Thread Polarization Dynamics (s14_polarization.py)

Analyzes sentiment trajectories within deep reply threads to model 
if and how conversations polarize or decay over depth.
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
import numpy as np
from pathlib import Path
from engine.config_loader import load_config
from scipy.stats import linregress

def run_polarization_dynamics():
    print("📉 Starting Thread Polarization Dynamics (s14)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ No comments_clean.parquet found. Skipping s14.")
        return
        
    print("  -> Loading comments data...")
    df = pd.read_parquet(comments_file)
    if df.empty:
        print("⚠️ comments_clean.parquet is empty. Skipping s14.")
        return

    # Ensure sentiment compound metric exists
    if 'vader_compound' not in df.columns:
        if 'sentiment_compound' in df.columns:
            df['vader_compound'] = df['sentiment_compound']
        elif 'sentiment_score' in df.columns:
            df['vader_compound'] = df['sentiment_score']
        else:
            df['vader_compound'] = 0.0

    # Ensure comment_depth exists (self-healing DAG walk if running before s16)
    if 'comment_depth' not in df.columns:
        print("  -> 'comment_depth' not precomputed. Deriving thread tree depth dynamically...")
        if 'parent_id' in df.columns and 'comment_id' in df.columns:
            import networkx as nx
            valid_cids = set(df['comment_id'].dropna())
            G = nx.DiGraph()
            G.add_nodes_from(df['comment_id'])
            edges = [
                (row.parent_id, row.comment_id)
                for row in df.itertuples()
                if pd.notna(row.parent_id) and row.parent_id in valid_cids and row.parent_id != row.comment_id
            ]
            G.add_edges_from(edges)
            node_depth = {}
            try:
                for node in nx.topological_sort(G):
                    in_edges = list(G.in_edges(node))
                    if not in_edges:
                        node_depth[node] = 0
                    else:
                        parent = in_edges[0][0]
                        node_depth[node] = node_depth.get(parent, 0) + 1
            except nx.NetworkXUnfeasible:
                node_depth = {n: 0 for n in G.nodes()}
            df['comment_depth'] = df['comment_id'].map(node_depth).fillna(0).astype(int)
        else:
            df['comment_depth'] = 0

    # 1. Overall Corpus Sentiment by Depth
    print("  -> Calculating corpus-level sentiment by depth...")
    depth_agg = df.groupby('comment_depth').agg(
        mean_sentiment=('vader_compound', 'mean'),
        comment_count=('comment_id', 'count')
    ).reset_index()
    
    # 2. Thread-Level Decay Slope
    # If thread_id is not explicitly defined, we assume all replies belonging to the same root 
    # comment share a common parent tree. YouTube API usually provides a `parent_id`. 
    # For a flat dataset, we'll map top-level comments as the root.
    if 'thread_id' not in df.columns and 'parent_id' in df.columns:
        print("  -> Constructing thread groups...")
        # A simple approximation: For YouTube, typically replies have `parent_id` equal to the root comment ID.
        # So thread_id = parent_id if parent_id is not NA, else comment_id
        df['thread_id'] = df['parent_id'].fillna(df['comment_id'])
    
    if 'thread_id' in df.columns:
        print("  -> Calculating per-thread sentiment decay slopes (for multi-comment threads >= 3)...")
        thread_counts = df['thread_id'].value_counts()
        multi_threads = thread_counts[thread_counts >= 3].index
        
        if len(multi_threads) > 0:
            sort_cols = ['thread_id']
            if 'published_at' in df.columns:
                sort_cols.append('published_at')
            elif 'published_time' in df.columns:
                sort_cols.append('published_time')
                
            df_multi = df[df['thread_id'].isin(multi_threads)].sort_values(sort_cols).copy()
            df_multi['seq_idx'] = df_multi.groupby('thread_id').cumcount()
            
            slopes = []
            for thread_id, group in df_multi.groupby('thread_id'):
                if len(group) >= 3:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        group['seq_idx'],
                        group['vader_compound']
                    )
                    slopes.append({
                        'thread_id': thread_id,
                        'decay_slope': float(slope),
                        'comment_count': int(len(group)),
                        'max_depth': int(group['comment_depth'].max()) if 'comment_depth' in group.columns else int(len(group) - 1),
                        'r_squared': float(r_value ** 2)
                    })
                    
            df_slopes = pd.DataFrame(slopes)
            if not df_slopes.empty:
                out_slopes_file = out_dir / "thread_decay_slopes.parquet"
                df_slopes.to_parquet(out_slopes_file, index=False)
                print(f"✅ Saved {len(df_slopes):,} thread decay slopes to {out_slopes_file.name}")
                avg_slope = df_slopes['decay_slope'].mean()
                print(f"   -> Average thread sentiment slope: {avg_slope:.4f} (Negative = decays into toxicity)")
            
    out_file = out_dir / "thread_polarization_corpus.parquet"
    depth_agg.to_parquet(out_file)
    print(f"✅ Saved corpus depth aggregation to {out_file}")

if __name__ == "__main__":
    run_polarization_dynamics()
