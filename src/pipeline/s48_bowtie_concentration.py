"""Stage 48: Network Bow-Tie Structure & Top-K Concentration (s48_bowtie_concentration.py)

Computes:
1. Directed Network Bow-Tie Decomposition (SCC, IN, OUT, Tendrils/Tubes, Disconnected)
2. Top-K Engagement Concentration Ratios (Top-1%, 5%, 10%, 20% likes/replies share)
"""

import sys
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from engine.config_loader import load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def compute_bowtie_decomposition(df_comments):
    """
    Decomposes a directed author-reply network into its canonical Bow-Tie structure:
    - SCC: Core strongly connected component (mutually reachable authors)
    - IN: Authors who reply into SCC but receive no SCC replies
    - OUT: Authors replied to by SCC who never reply back into SCC
    - TENDRILS_TUBES: Peripheral paths connecting to IN/OUT without passing through SCC
    - DISCONNECTED: Isolated subgraphs
    """
    # Create directed author-to-parent-author graph
    roots = df_comments[df_comments["parent_id"].isna() | (df_comments["parent_id"] == "") | (df_comments["parent_id"] == df_comments["comment_id"])].set_index("comment_id")["author_channel_id"].to_dict()
    replies = df_comments[df_comments["parent_id"].notna() & (df_comments["parent_id"] != "") & (df_comments["parent_id"] != df_comments["comment_id"])].copy()

    replies["parent_author"] = replies["parent_id"].map(roots)
    valid_edges = replies.dropna(subset=["author_channel_id", "parent_author"])
    valid_edges = valid_edges[valid_edges["author_channel_id"] != valid_edges["parent_author"]]

    if valid_edges.empty:
        return pd.DataFrame([{
            "component": "DISCONNECTED",
            "node_count": len(df_comments["author_channel_id"].dropna().unique()),
            "percentage": 100.0,
            "description": "Entire network is disconnected or single-node"
        }])

    G = nx.DiGraph()
    for _, row in valid_edges.iterrows():
        G.add_edge(row["author_channel_id"], row["parent_author"])

    total_nodes = G.number_of_nodes()
    if total_nodes == 0:
        return pd.DataFrame()

    from collections import deque

    def multi_source_bfs(graph, source_nodes):
        visited = set(source_nodes)
        queue = deque(source_nodes)
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return visited

    # Find SCCs
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        return pd.DataFrame()

    largest_scc = max(sccs, key=len)
    scc_set = set(largest_scc)

    G_rev = G.reverse(copy=False)

    # Nodes reachable from SCC in G
    desc_from_scc = multi_source_bfs(G, scc_set)
    out_component = desc_from_scc - scc_set

    # Nodes that can reach SCC in G (i.e. reachable from SCC in G_rev)
    anc_to_scc = multi_source_bfs(G_rev, scc_set)
    in_component = anc_to_scc - scc_set

    classified = scc_set | in_component | out_component
    remaining = set(G.nodes()) - classified

    # Tendrils & tubes: reachable from IN or can reach OUT
    reachable_from_in = multi_source_bfs(G, in_component) if in_component else set()
    can_reach_out = multi_source_bfs(G_rev, out_component) if out_component else set()

    tendrils_tubes = remaining & (reachable_from_in | can_reach_out)
    disconnected = remaining - tendrils_tubes

    bowtie_data = [
        {"component": "SCC (Core)", "node_count": len(scc_set), "percentage": round(len(scc_set) / total_nodes * 100, 2), "description": "Core strongly connected reciprocal cluster"},
        {"component": "IN Component", "node_count": len(in_component), "percentage": round(len(in_component) / total_nodes * 100, 2), "description": "Commenters replying to the core without receiving replies"},
        {"component": "OUT Component", "node_count": len(out_component), "percentage": round(len(out_component) / total_nodes * 100, 2), "description": "Influencers/threads receiving replies from the core"},
        {"component": "Tendrils & Tubes", "node_count": len(tendrils_tubes), "percentage": round(len(tendrils_tubes) / total_nodes * 100, 2), "description": "Peripheral branch chains connected to IN or OUT"},
        {"component": "Disconnected", "node_count": len(disconnected), "percentage": round(len(disconnected) / total_nodes * 100, 2), "description": "Isolated dialogue pairs unconnected to main core"},
    ]

    return pd.DataFrame(bowtie_data)


def compute_top_k_concentration(df_comments):
    """
    Computes exact Pareto & Top-K concentration ratios:
    Top 1%, 5%, 10%, 20% of comments share of total likes and replies.
    """
    df = df_comments.copy()
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    df["reply_count"] = pd.to_numeric(df.get("reply_count", 0), errors="coerce").fillna(0)

    rows = []

    def get_concentration(series, name):
        sorted_vals = series.sort_values(ascending=False).values
        total = sorted_vals.sum()
        n = len(sorted_vals)
        if total == 0 or n == 0:
            return {}

        top1_idx = max(1, int(n * 0.01))
        top5_idx = max(1, int(n * 0.05))
        top10_idx = max(1, int(n * 0.10))
        top20_idx = max(1, int(n * 0.20))

        # Gini coefficient
        cum = np.cumsum(np.sort(sorted_vals))
        gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n if cum[-1] > 0 else 0.0

        return {
            f"top_1pct_{name}_share": round(float(sorted_vals[:top1_idx].sum() / total), 4),
            f"top_5pct_{name}_share": round(float(sorted_vals[:top5_idx].sum() / total), 4),
            f"top_10pct_{name}_share": round(float(sorted_vals[:top10_idx].sum() / total), 4),
            f"top_20pct_{name}_share": round(float(sorted_vals[:top20_idx].sum() / total), 4),
            f"gini_{name}": round(float(gini), 4)
        }

    # Corpus-wide concentration
    corpus_likes = get_concentration(df["like_count"], "likes")
    corpus_replies = get_concentration(df["reply_count"], "replies")
    corpus_row = {"scope": "corpus", "video_id": "__ALL__", "n_comments": len(df), **corpus_likes, **corpus_replies}
    rows.append(corpus_row)

    # Per-video concentration
    if "video_id" in df.columns:
        for vid, grp in df.groupby("video_id"):
            if len(grp) < 10:
                continue
            v_likes = get_concentration(grp["like_count"], "likes")
            v_replies = get_concentration(grp["reply_count"], "replies")
            rows.append({"scope": "video", "video_id": str(vid), "n_comments": len(grp), **v_likes, **v_replies})

    return pd.DataFrame(rows)


def run_bowtie_concentration():
    """Main execution entrypoint for Stage 48."""
    print("🎀 Starting Network Bow-Tie Structure & Top-K Concentration (s48)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s48.")
        return

    import pyarrow.parquet as pq
    available = pq.read_schema(comments_file).names
    cols = [c for c in ["comment_id", "parent_id", "author_channel_id", "video_id", "like_count", "reply_count"] if c in available]
    df = pd.read_parquet(comments_file, columns=cols)

    if df.empty:
        print("⚠️ Comments dataset is empty. Skipping s48.")
        return

    print("  -> Decomposing author reply graph into Bow-Tie components...")
    df_bowtie = compute_bowtie_decomposition(df)
    out_bowtie = out_dir / "network_bowtie_structure.parquet"
    df_bowtie.to_parquet(out_bowtie, index=False)
    print(f"  -> Bow-Tie decomposition written to {out_bowtie.name}")

    print("  -> Calculating Top-K attention concentration ratios...")
    df_conc = compute_top_k_concentration(df)
    out_conc = out_dir / "top_k_concentration.parquet"
    df_conc.to_parquet(out_conc, index=False)
    print(f"  -> Top-K concentration written to {out_conc.name}")

    print("✅ Stage 48 Bow-Tie & Concentration Analysis Complete.")


if __name__ == "__main__":
    run_bowtie_concentration()
