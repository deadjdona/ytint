"""Stage 05: Hashtags & Mentions Network (s05_tags_mentions.py)

Extracts #hashtags and @mentions from comments to build a co-occurrence 
network, revealing which communities, channels, or topics are linked.
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
import re
from pathlib import Path
from engine.config_loader import load_config
from itertools import combinations
from collections import Counter

def run_tag_networks():
    print("🏷️ Starting Hashtag/Mention Extraction (s16)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s16.")
        return
        
    print("  -> Loading comments...")
    df = pd.read_parquet(comments_file, columns=['comment_id', 'text'])
    df = df.dropna(subset=['text'])
    
    if df.empty:
        print("⚠️ No valid text found. Skipping.")
        return
        
    print("  -> Extracting hashtags and mentions...")
    # Find all words starting with # or @ (supporting Russian and English characters)
    # \w in regex python 3+ supports unicode word characters (including cyrillic)
    tag_pattern = re.compile(r'([#@]\w+)')
    
    edges_list = []
    node_counts = Counter()
    
    # Process each comment
    for text in df['text']:
        tags = [t.lower() for t in tag_pattern.findall(text)]
        if not tags:
            continue
            
        node_counts.update(tags)
        
        # Create edges for all pairs in the same comment
        unique_tags = sorted(list(set(tags)))
        if len(unique_tags) > 1:
            for pair in combinations(unique_tags, 2):
                edges_list.append(pair)
                
    if not edges_list and not node_counts:
        print("⚠️ No hashtags or mentions found in corpus.")
        pd.DataFrame(columns=['source', 'target', 'weight']).to_parquet(out_dir / "tag_network_edges.parquet", index=False)
        pd.DataFrame(columns=['node', 'freq']).to_parquet(out_dir / "tag_network_nodes.parquet", index=False)
        return
        
    # Aggregate edge weights
    print("  -> Building Graph...")
    edge_counts = Counter(edges_list)
    
    edges_df = pd.DataFrame([
        {'source': u, 'target': v, 'weight': w}
        for (u, v), w in edge_counts.items()
    ])
    
    # Keep top 300 strongest edges
    if not edges_df.empty:
        edges_df = edges_df.sort_values('weight', ascending=False).head(300)
        
    # Create Nodes DataFrame (top 1000 nodes total to save space)
    nodes_df = pd.DataFrame([
        {'node': n, 'freq': f}
        for n, f in node_counts.most_common(1000)
    ])
    
    out_edges = out_dir / "tag_network_edges.parquet"
    out_nodes = out_dir / "tag_network_nodes.parquet"
    
    edges_df.to_parquet(out_edges, index=False)
    nodes_df.to_parquet(out_nodes, index=False)
    
    print(f"✅ Generated Tag/Mention graph with {len(nodes_df)} nodes and {len(edges_df)} edges.")

if __name__ == "__main__":
    run_tag_networks()
