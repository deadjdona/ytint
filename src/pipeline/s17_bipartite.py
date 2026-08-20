"""Stage 17: Author-Video Bipartite Graph (s17_bipartite.py)

Constructs a bipartite network connecting commenters (authors) to the 
videos they commented on, revealing super-fans and cross-video audiences.
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

def run_bipartite():
    print("🕸️ Starting Author-Video Bipartite Network (s17)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s17.")
        return
        
    print("  -> Loading interactions...")
    df = pd.read_parquet(comments_file, columns=['author_channel_id', 'video_id'])
    
    # We only care about unique interactions for the network structure
    edges = df.drop_duplicates(subset=['author_channel_id', 'video_id']).copy()
    edges = edges.dropna()
    
    if edges.empty:
        print("⚠️ No edges found. Skipping.")
        return
        
    print("  -> Filtering for super-users to maintain readability...")
    # Find authors who commented on the most distinct videos (cross-pollinators)
    author_counts = edges['author_channel_id'].value_counts()
    
    # Keep top 400 most active multi-video authors
    top_authors = author_counts.head(400).index
    edges_filtered = edges[edges['author_channel_id'].isin(top_authors)]
    
    # Prepare node metadata (type distinction)
    videos = edges_filtered['video_id'].unique()
    authors = edges_filtered['author_channel_id'].unique()
    
    print(f"  -> Building graph with {len(videos)} videos and {len(authors)} authors...")
    
    # We need full data to calculate weight (total comment volume, not just unique connections)
    # We'll calculate it via value_counts on the full dataframe for speed
    full_vid_counts = df['video_id'].value_counts()
    full_auth_counts = df['author_channel_id'].value_counts()
    
    nodes_data = []
    for v in videos:
        vol = full_vid_counts.get(v, 1)
        nodes_data.append({'node': v, 'type': 'video', 'weight': vol})
        
    for a in authors:
        vol = full_auth_counts.get(a, 1)
        nodes_data.append({'node': a, 'type': 'author', 'weight': vol})
        
    nodes_df = pd.DataFrame(nodes_data)
    
    out_edges = out_dir / "bipartite_edges.parquet"
    out_nodes = out_dir / "bipartite_nodes.parquet"
    
    edges_filtered.rename(columns={'author_channel_id': 'source', 'video_id': 'target'}).to_parquet(out_edges, index=False)
    nodes_df.to_parquet(out_nodes, index=False)
    
    print(f"✅ Generated Bipartite graph edges: {len(edges_filtered)}.")

if __name__ == "__main__":
    run_bipartite()
