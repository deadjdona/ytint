"""Stage 18: Co-commenting Graph (s18_cocommenting.py)

Constructs a network of authors who frequently comment on the same videos, 
revealing tight-knit communities of users who travel together across content.
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config
from itertools import combinations
from collections import Counter

def run_cocommenting():
    print("🕸️ Starting Co-commenting Network (s18)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s18.")
        return
        
    print("  -> Loading interactions...")
    df = pd.read_parquet(comments_file, columns=['author_channel_id', 'video_id'])
    
    # We only care about unique (author, video) interactions
    edges = df.drop_duplicates(subset=['author_channel_id', 'video_id']).copy()
    edges = edges.dropna()
    
    if edges.empty:
        print("⚠️ No valid interactions found. Skipping.")
        return
        
    print("  -> Filtering for super-users to maintain readability...")
    author_counts = edges['author_channel_id'].value_counts()
    # Keep top 400 most active cross-pollinating authors
    top_authors = author_counts.head(400).index
    edges_filtered = edges[edges['author_channel_id'].isin(top_authors)]
    
    print("  -> Generating co-commenting edges...")
    co_edges = []
    
    # Group by video to find users who commented on the same video
    for video, group in edges_filtered.groupby('video_id'):
        authors_in_video = sorted(group['author_channel_id'].tolist())
        if len(authors_in_video) > 1:
            for pair in combinations(authors_in_video, 2):
                co_edges.append(pair)
                
    if not co_edges:
        print("⚠️ No co-commenting found among top users.")
        pd.DataFrame(columns=['source', 'target', 'weight']).to_parquet(out_dir / "cocomment_edges.parquet", index=False)
        pd.DataFrame(columns=['node', 'freq']).to_parquet(out_dir / "cocomment_nodes.parquet", index=False)
        return
        
    print("  -> Building Graph...")
    edge_counts = Counter(co_edges)
    
    edges_df = pd.DataFrame([
        {'source': u, 'target': v, 'weight': w}
        for (u, v), w in edge_counts.items()
    ])
    
    # Filter to strongest edges if the graph is too dense
    if not edges_df.empty:
        # Keep edges where weight >= 2 (they commented on at least 2 of the same videos)
        strong_edges = edges_df[edges_df['weight'] > 1]
        if len(strong_edges) > 50:
            edges_df = strong_edges
        # Hard cap at top 1000 edges for viz speed
        edges_df = edges_df.sort_values('weight', ascending=False).head(1000)
        
    # Create Nodes DataFrame (node frequency = total unique videos they commented on)
    active_nodes = set(edges_df['source']).union(set(edges_df['target']))
    nodes_df = pd.DataFrame([
        {'node': a, 'freq': author_counts.get(a, 1)}
        for a in active_nodes
    ])
    
    out_edges = out_dir / "cocomment_edges.parquet"
    out_nodes = out_dir / "cocomment_nodes.parquet"
    
    edges_df.to_parquet(out_edges, index=False)
    nodes_df.to_parquet(out_nodes, index=False)
    
    print(f"✅ Generated Co-commenting graph with {len(nodes_df)} authors and {len(edges_df)} edges.")

if __name__ == "__main__":
    run_cocommenting()
