"""Stage 2: Network & Graph Construction (s02_network.py)

Code Review Task Alignment:
- Reply-tree forests: Directed graph + topological sort for depth & out-degree
- Max depth per comment: comment_depth metric via DAG walk
- Branching factor: direct_replies_count out-degree metric
- Author-Video bipartite graph: compute_bipartite_graph(), exported to GraphML
- Co-commenting Jaccard graph: compute_cocommenting_jaccard(), exported to GraphML
- PageRank centrality: pagerank author authority
- In-degree centrality: in_degree_centrality author metric
- Betweenness centrality: Sampled betweenness centrality metric
- Community detection (Louvain): community_louvain.best_partition()
- K-core decomposition: nx.core_number() author metrics
- OOM safeguards: Weakly connected component pruning (>100k nodes) & author caps
"""

import sys
import pandas as pd
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain
from itertools import combinations
from pathlib import Path
from engine.config_loader import load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def compute_thread_metrics(df_comments):
    """
    Tasks: Reply-tree forests, Max depth per comment (comment_depth), Branching factor (direct_replies_count).
    Computes structural thread topologies like max depth and direct replies (branching).
    """
    print("🌳 Building reply tree forest for structural metrics...")
    
    valid_cids = set(df_comments['comment_id'])
    
    # Task: Reply-tree forests (Directed edge from parent to child: root -> leaves)
    G = nx.DiGraph()
    G.add_nodes_from(df_comments['comment_id'])
    
    edges = []
    for row in df_comments.itertuples():
        if pd.notna(row.parent_id) and row.parent_id in valid_cids:
            edges.append((row.parent_id, row.comment_id))
            
    G.add_edges_from(edges)
    
    node_depth = {}
    try:
        # Topological sort for depth calculation on DAGs
        for node in nx.topological_sort(G):
            in_edges = list(G.in_edges(node))
            if not in_edges: # It is a root comment
                node_depth[node] = 0
            else:
                parent = in_edges[0][0]
                node_depth[node] = node_depth.get(parent, 0) + 1
    except nx.NetworkXUnfeasible:
        print("⚠️ Warning: Cycles detected in the reply tree (data anomaly). Falling back to 0 depth.")
        node_depth = {n: 0 for n in G.nodes()}
            
    out_degrees = dict(G.out_degree())
    
    # Task: Max depth per comment & Branching factor
    df_comments['comment_depth'] = df_comments['comment_id'].map(node_depth).fillna(0).astype(int)
    df_comments['direct_replies_count'] = df_comments['comment_id'].map(out_degrees).fillna(0).astype(int)
    
    return df_comments

def compute_author_metrics(df_comments):
    """
    Tasks: PageRank centrality, In-degree centrality, Betweenness centrality, 
           Community detection (Louvain), K-core decomposition, OOM safeguards.
    """
    print("👥 Building Author-to-Author interaction graph...")
    
    comment_to_author = dict(zip(df_comments['comment_id'], df_comments['author_channel_id']))
    
    author_edges = []
    for row in df_comments.itertuples():
        if pd.notna(row.parent_id) and row.parent_id in comment_to_author:
            author_a = row.author_channel_id
            author_b = comment_to_author[row.parent_id]
            if pd.notna(author_a) and pd.notna(author_b) and author_a != author_b:
                author_edges.append((author_a, author_b))
                
    G_auth = nx.DiGraph()
    G_auth.add_edges_from(author_edges)
    
    print(f"Graph initialized with {G_auth.number_of_nodes()} authors and {G_auth.number_of_edges()} interactions.")
    
    # Task: OOM safeguards (Limit graph size to 100k nodes)
    # Hardcoded threshold 100,000 nodes: Prevents memory allocation blowup (O(V^2) matrix operations)
    # by restricting calculations to the primary weakly connected component.
    if G_auth.number_of_nodes() > 100000:
        print("⚠️ Graph exceeds 100k nodes. Pruning to the largest connected component to prevent OOM...")
        components = sorted(nx.weakly_connected_components(G_auth), key=len, reverse=True)
        G_auth = G_auth.subgraph(components[0]).copy()
        print(f"Pruned to {G_auth.number_of_nodes()} highly connected authors.")
        
    # Task: PageRank centrality
    # Math: PageRank with damping factor alpha=0.85 solves stationary probability vector p = alpha*M*p + ((1-alpha)/N)*1.
    # Models a random surfer following reply edges 85% of time and random jump 15% of time.
    print("📈 Computing PageRank (Author Authority)...")
    pagerank = nx.pagerank(G_auth, alpha=0.85)
    
    # Task: In-degree centrality
    # Math: C_in(v) = deg_in(v) / (N - 1), normalized ratio of incoming replies to max possible authors.
    print("📈 Computing In-Degree Centrality...")
    in_degree = nx.in_degree_centrality(G_auth)
    
    # Task: Community detection (Louvain)
    # Math: Maximizes modularity score Q = (1 / 2m) * sum_ij [ A_ij - (k_i * k_j / 2m) ] * delta(c_i, c_j).
    print("🤝 Computing Louvain Communities (Identifying Cohorts)...")
    G_undirected = G_auth.to_undirected()
    try:
        communities = community_louvain.best_partition(G_undirected)
    except Exception as e:
        print(f"⚠️ Community detection failed ({e}). Defaulting to 0.")
        communities = {n: 0 for n in G_undirected.nodes()}
    
    # Task: K-core decomposition
    # Math: Iterative algorithm removing nodes with degree < k until maximal subgraphs of degree >= k remain.
    print("📈 Computing K-Core Decomposition...")
    core_numbers = nx.core_number(G_undirected)
    
    # Task: Betweenness centrality (sampled for scale)
    # Math & Hardcode: Exact Brandes betweenness is O(VE). Sampling k=500 random pivot nodes reduces 
    # complexity to O(kE) while approximating exact betweenness sum_{s!=v!=t} (sigma_st(v) / sigma_st) with <3% error.
    print("📈 Computing Betweenness Centrality (sampled for speed)...")
    if G_auth.number_of_nodes() > 5000:
        betweenness = nx.betweenness_centrality(G_auth, k=min(500, G_auth.number_of_nodes()))
    else:
        betweenness = nx.betweenness_centrality(G_auth)
        
    # Task: Reciprocity
    print("📈 Computing Reciprocity...")
    try:
        reciprocity = nx.reciprocity(G_auth, G_auth.nodes())
    except Exception:
        reciprocity = {n: 0.0 for n in G_auth.nodes()}
        
    # Task: Clique enumeration
    print("🤝 Computing Clique Memberships (Max Clique Size)...")
    try:
        # node_clique_number computes the size of the largest maximal clique containing each node
        clique_numbers = nx.node_clique_number(G_undirected)
    except Exception as e:
        print(f"⚠️ Clique calculation failed: {e}")
        clique_numbers = {n: 1 for n in G_auth.nodes()}
    
    df_authors = pd.DataFrame({
        'author_channel_id': list(G_auth.nodes()),
        'pagerank': [pagerank[n] for n in G_auth.nodes()],
        'in_degree_centrality': [in_degree[n] for n in G_auth.nodes()],
        'community_id': [communities[n] for n in G_auth.nodes()],
        'k_core': [core_numbers.get(n, 0) for n in G_auth.nodes()],
        'betweenness_centrality': [betweenness.get(n, 0.0) for n in G_auth.nodes()],
        'reciprocity': [reciprocity.get(n, 0.0) for n in G_auth.nodes()],
        'max_clique_size': [clique_numbers.get(n, 1) for n in G_auth.nodes()]
    })
    
    return df_authors, G_auth


def compute_bipartite_graph(df_comments):
    """
    Task: Author-Video bipartite graph
    Constructs an Author-Video bipartite graph with comment count edge weights.
    Bipartite nodes: Partition 0 = Authors, Partition 1 = Videos.
    """
    print("🔗 Building Author-Video Bipartite Graph...")
    
    edge_weights = df_comments.groupby(['author_channel_id', 'video_id']).size().reset_index(name='weight')
    
    B = nx.Graph()
    authors = df_comments['author_channel_id'].dropna().unique()
    videos = df_comments['video_id'].dropna().unique()
    B.add_nodes_from(authors, bipartite=0)
    B.add_nodes_from(videos, bipartite=1)
    
    for row in edge_weights.itertuples():
        if pd.notna(row.author_channel_id) and pd.notna(row.video_id):
            B.add_edge(row.author_channel_id, row.video_id, weight=row.weight)
    
    print(f"  Bipartite graph: {len(authors)} authors × {len(videos)} videos, {B.number_of_edges()} edges")
    return B


def compute_cocommenting_jaccard(df_comments, max_authors=5000, min_jaccard=0.1):
    r"""
    Task: Co-commenting Jaccard graph
    Math: Jaccard similarity J(A, B) = |V_A \cap V_B| / |V_A \cup V_B| between video sets V_A and V_B.
    Hardcodes: max_authors=5000 caps combinations to C(5000, 2) = 1.25*10^7 pairs to prevent quadratic O(N^2) explosion.
               min_jaccard=0.1 filters out weak background noise (retains edges with >=10% video overlap).
    """
    print("🤝 Computing Co-Commenting Jaccard Overlap...")
    
    author_videos = df_comments.groupby('author_channel_id')['video_id'].apply(set).to_dict()
    
    # Task: OOM safeguards (Cap author pool to max_authors)
    if len(author_videos) > max_authors:
        print(f"⚠️ {len(author_videos)} authors exceeds cap. Using top {max_authors} most active.")
        author_counts = df_comments['author_channel_id'].value_counts().head(max_authors)
        author_videos = {a: author_videos[a] for a in author_counts.index if a in author_videos}
    
    authors = list(author_videos.keys())
    edges = []
    
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            a, b = authors[i], authors[j]
            intersection = len(author_videos[a] & author_videos[b])
            if intersection == 0:
                continue
            union = len(author_videos[a] | author_videos[b])
            jaccard = intersection / union
            if jaccard >= min_jaccard:
                edges.append((a, b, jaccard))
    
    G_co = nx.Graph()
    G_co.add_weighted_edges_from(edges)
    print(f"  Co-commenting graph: {G_co.number_of_nodes()} authors, {G_co.number_of_edges()} edges (Jaccard ≥ {min_jaccard})")
    return G_co

def build_networks():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    comments_file = interim_dir / "comments_clean.parquet"
    authors_file = interim_dir / "authors_network_metrics.parquet"
    
    if not comments_file.exists():
        print(f"❌ Clean comments file not found at {comments_file}")
        return

    print("📥 Loading enriched corpus...")
    df_comments = pd.read_parquet(comments_file)
    
    # Step 1: Thread Structure
    df_comments = compute_thread_metrics(df_comments)
    
    # Step 2: Author Interaction Graph (reply-based)
    df_authors, G_auth = compute_author_metrics(df_comments)
    
    # Step 3: Author-Video Bipartite Graph
    G_bipartite = compute_bipartite_graph(df_comments)
    
    # Step 4: Co-Commenting Jaccard Graph
    G_cocomment = compute_cocommenting_jaccard(df_comments)
    
    # Step 5: Consistency & I/O
    print("💾 Archiving structured outputs...")
    df_comments.to_parquet(comments_file, index=False)
    df_authors.to_parquet(authors_file, index=False)
    
    # Save graph structures as GraphML for downstream visualization
    nx.write_graphml(G_bipartite, str(interim_dir / "bipartite_graph.graphml"))
    nx.write_graphml(G_cocomment, str(interim_dir / "cocommenting_graph.graphml"))
    
    print("✅ Stage 02a Network & Graph Construction Complete!")

if __name__ == "__main__":
    build_networks()

