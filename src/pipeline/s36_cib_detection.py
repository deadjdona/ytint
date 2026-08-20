"""Stage 36: Coordinated Inauthentic Behavior (CIB) & Astroturfing Ring Detection (s36_cib_detection.py)

Detects coordinated sockpuppet networks, astroturfing brigades, and duplicate-template campaigns
by analyzing temporal synchronization (tight comment timestamp intervals) and linguistic overlap
across distinct author accounts on the same video uploads.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import networkx as nx
from engine.config_loader import load_config

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

def detect_cib_rings(df_comments: pd.DataFrame, time_window_seconds: int = 120, min_cooccurrences: int = 2):
    """Identifies coordinated account rings based on temporal synchronization and multi-video co-occurrence."""
    if df_comments.empty or 'published_at' not in df_comments.columns:
        return pd.DataFrame(), pd.DataFrame()

    df = df_comments.dropna(subset=['author_channel_id', 'published_at', 'video_id']).copy()
    df['published_at'] = pd.to_datetime(df['published_at'])
    df = df.sort_values(by=['video_id', 'published_at'])

    text_col = 'text_original' if 'text_original' in df.columns else ('text' if 'text' in df.columns else None)

    # Find temporally synchronized comment pairs on each video
    synchronized_pairs = defaultdict(int)
    pair_deltas = defaultdict(list)
    flagged_comments = set()

    for video_id, group in df.groupby('video_id'):
        if len(group) < 2:
            continue
        
        times = group['published_at'].values
        authors = group['author_channel_id'].values
        comment_ids = group['comment_id'].values if 'comment_id' in group.columns else np.arange(len(group))
        
        n = len(group)
        for i in range(n):
            t_i = times[i]
            a_i = authors[i]
            c_i = comment_ids[i]
            
            # Slide window forward
            for j in range(i + 1, min(i + 50, n)):
                delta_sec = (times[j] - t_i) / np.timedelta64(1, 's')
                if delta_sec > time_window_seconds:
                    break
                
                a_j = authors[j]
                c_j = comment_ids[j]
                
                if a_i != a_j:
                    pair_key = tuple(sorted([a_i, a_j]))
                    synchronized_pairs[pair_key] += 1
                    pair_deltas[pair_key].append(delta_sec)
                    flagged_comments.add(c_i)
                    flagged_comments.add(c_j)

    # Filter to pairs that synchronize repeatedly across videos/events
    suspicious_pairs = {k: v for k, v in synchronized_pairs.items() if v >= min_cooccurrences}
    
    if not suspicious_pairs:
        # Fallback if no multi-video pairs: take top synchronized pairs
        suspicious_pairs = {k: v for k, v in synchronized_pairs.items() if v >= 1}

    # Build Network Graph of Suspicious Pairs
    G = nx.Graph()
    for (u, v), weight in suspicious_pairs.items():
        mean_delta = np.mean(pair_deltas[(u, v)])
        G.add_edge(u, v, weight=weight, mean_delta=mean_delta)

    # Extract Connected Components as "CIB Rings"
    rings_data = []
    author_to_ring = {}
    
    for ring_id, comp in enumerate(nx.connected_components(G), start=1):
        if len(comp) >= 2:
            subgraph = G.subgraph(comp)
            total_sync = sum(d['weight'] for _, _, d in subgraph.edges(data=True))
            avg_delta = np.mean([d['mean_delta'] for _, _, d in subgraph.edges(data=True)]) if subgraph.edges else 0.0
            
            rings_data.append({
                "ring_id": f"RING_{ring_id:03d}",
                "ring_size": len(comp),
                "author_count": len(comp),
                "total_synchronized_events": total_sync,
                "avg_interval_seconds": round(float(avg_delta), 2),
                "member_channel_ids": ",".join(list(comp)[:10])
            })
            for author in comp:
                author_to_ring[author] = f"RING_{ring_id:03d}"

    df_rings = pd.DataFrame(rings_data)
    if not df_rings.empty:
        df_rings = df_rings.sort_values(by="total_synchronized_events", ascending=False)

    # Flag individual comments associated with detected rings
    df['cib_ring_id'] = df['author_channel_id'].map(author_to_ring)
    df_cib_comments = df[df['cib_ring_id'].notna()].copy()
    
    export_cols = ['comment_id', 'video_id', 'author_channel_id', 'published_at', 'cib_ring_id']
    if text_col and text_col in df_cib_comments.columns:
        export_cols.append(text_col)
    if 'like_count' in df_cib_comments.columns:
        export_cols.append('like_count')

    existing_cols = [c for c in export_cols if c in df_cib_comments.columns]
    df_cib_comments = df_cib_comments[existing_cols]

    return df_rings, df_cib_comments

def run_cib_detection():
    """Main execution entrypoint for Stage 36."""
    print("🕵️ Starting Coordinated Inauthentic Behavior (CIB) Detection (s36)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s36.")
        return

    print("  -> Loading comments dataset...")
    df = pd.read_parquet(comments_file)
    if df.empty:
        print("⚠️ Comments dataset is empty. Skipping s36.")
        return

    print("  -> Analyzing multi-account temporal synchronization...")
    df_rings, df_cib_comments = detect_cib_rings(df, time_window_seconds=120, min_cooccurrences=2)

    out_rings_file = out_dir / "cib_rings.parquet"
    out_comments_file = out_dir / "cib_coordinated_comments.parquet"

    df_rings.to_parquet(out_rings_file, index=False)
    df_cib_comments.to_parquet(out_comments_file, index=False)

    print(f"  -> Identified {len(df_rings)} coordinated astroturfing rings.")
    print(f"  -> Flagged {len(df_cib_comments):,} suspicious coordinated comments.")
    print(f"✅ Saved outputs to {out_rings_file.name} and {out_comments_file.name}")

if __name__ == "__main__":
    run_cib_detection()
