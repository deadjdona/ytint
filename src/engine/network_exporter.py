"""ytint // Network Graph & Gephi Exporter Engine (src/engine/network_exporter.py)

Builds, serializes, and visualizes rich structural network topologies:
1. Author Reply Network (Directed): Who replies to whom with Louvain communities, PageRank, and interaction weights.
2. Author-Video Bipartite Network: Multi-modal connection mapping authors to videos they commented on.
3. Co-Commenting Network: Community network of users who co-occur across the same video uploads.

Exports to industry-standard GEXF (Gephi) and GraphML (Cytoscape / yEd),
and renders interactive WebGL force-directed layouts via Plotly.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure src is in sys.path
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from engine.config_loader import get_paths, load_config

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

logger = logging.getLogger("ytint_network_exporter")


class NetworkExporter:
    """Orchestrates network graph construction, attribute attachment, export, and interactive visualization."""

    def __init__(self, interim_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        cfg = load_config()
        _, interim, output = get_paths(cfg)
        self.interim_dir = Path(interim_dir or interim)
        self.output_dir = Path(output_dir or output)
        self.networks_dir = self.output_dir / "networks"
        self.networks_dir.mkdir(parents=True, exist_ok=True)

    def load_data_layers(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads clean comments, videos, authors, and bot classification datasets."""
        comments_file = self.interim_dir / "comments_clean.parquet"
        videos_file = self.interim_dir / "videos_clean.parquet"
        authors_file = self.output_dir / "authors_final.parquet"
        if not authors_file.exists():
            authors_file = self.interim_dir / "authors_network_metrics.parquet"
        bots_file = self.output_dir / "bot_classifications.parquet"

        df_comments = pd.read_parquet(comments_file) if comments_file.exists() else pd.DataFrame()
        df_videos = pd.read_parquet(videos_file) if videos_file.exists() else pd.DataFrame()
        df_authors = pd.read_parquet(authors_file) if authors_file.exists() else pd.DataFrame()
        df_bots = pd.read_parquet(bots_file) if bots_file.exists() else pd.DataFrame()

        return df_comments, df_videos, df_authors, df_bots

    def build_author_reply_network(
        self,
        df_comments: Optional[pd.DataFrame] = None,
        df_authors: Optional[pd.DataFrame] = None,
        df_bots: Optional[pd.DataFrame] = None
    ) -> nx.DiGraph:
        """Constructs directed Author-to-Author interaction graph with rich node/edge metadata."""
        if df_comments is None:
            df_comments, _, df_authors, df_bots = self.load_data_layers()

        G = nx.DiGraph(name="ytint_author_reply_network")
        if df_comments.empty or "author_channel_id" not in df_comments.columns:
            return G

        # 1. Map comments to author identifiers and display names
        comment_to_author = dict(zip(
            df_comments["comment_id"].astype(str),
            df_comments["author_channel_id"].astype(str)
        ))
        
        display_name_map = {}
        if "author_display_name" in df_comments.columns:
            display_name_map = dict(zip(
                df_comments["author_channel_id"].astype(str),
                df_comments["author_display_name"].astype(str)
            ))

        # 2. Extract directed reply edges (reply_author -> parent_author)
        edge_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        valid_replies = df_comments[
            df_comments["parent_id"].notna() &
            (df_comments["parent_id"] != "") &
            (df_comments["parent_id"] != df_comments["comment_id"])
        ]

        has_vader = "vader_compound" in df_comments.columns
        has_toxicity = "toxicity" in df_comments.columns

        for row in valid_replies.itertuples():
            u = str(getattr(row, "author_channel_id", ""))
            parent_cid = str(getattr(row, "parent_id", ""))
            v = comment_to_author.get(parent_cid, "")

            if u and v and u != v:
                pair = (u, v)
                sentiment = float(getattr(row, "vader_compound", 0.0)) if has_vader and pd.notna(getattr(row, "vader_compound", None)) else 0.0
                toxicity = float(getattr(row, "toxicity", 0.0)) if has_toxicity and pd.notna(getattr(row, "toxicity", None)) else 0.0

                if pair not in edge_data:
                    edge_data[pair] = {
                        "weight": 1,
                        "sentiment_sum": sentiment,
                        "toxicity_sum": toxicity
                    }
                else:
                    edge_data[pair]["weight"] += 1
                    edge_data[pair]["sentiment_sum"] += sentiment
                    edge_data[pair]["toxicity_sum"] += toxicity

        for (u, v), m in edge_data.items():
            w = m["weight"]
            G.add_edge(
                u,
                v,
                weight=int(w),
                sentiment_mean=float(round(m["sentiment_sum"] / w, 4)),
                toxicity_mean=float(round(m["toxicity_sum"] / w, 4))
            )

        # 3. Calculate structural graph metrics on G
        if G.number_of_nodes() > 0:
            pagerank = nx.pagerank(G, alpha=0.85)
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())
            
            # Community detection via modularity
            G_undirected = G.to_undirected()
            try:
                import community.community_louvain as community_louvain
                communities = community_louvain.best_partition(G_undirected)
            except Exception:
                communities = {n: 0 for n in G.nodes()}

            # Pre-index author metadata if provided
            auth_meta = {}
            if df_authors is not None and not df_authors.empty and "author_channel_id" in df_authors.columns:
                auth_indexed = df_authors.drop_duplicates("author_channel_id").set_index("author_channel_id")
                auth_meta = auth_indexed.to_dict(orient="index")

            # Pre-index bot classifications if provided
            bot_meta = {}
            if df_bots is not None and not df_bots.empty and "author_channel_id" in df_bots.columns:
                bot_indexed = df_bots.drop_duplicates("author_channel_id").set_index("author_channel_id")
                bot_meta = bot_indexed.to_dict(orient="index")

            # Attach primitive, type-safe attributes to every node
            for node in G.nodes():
                a_info = auth_meta.get(node, {})
                b_info = bot_meta.get(node, {})
                label = display_name_map.get(node, str(node))

                G.nodes[node]["label"] = str(label)
                G.nodes[node]["pagerank"] = float(round(pagerank.get(node, 0.0), 6))
                G.nodes[node]["in_degree"] = int(in_degrees.get(node, 0))
                G.nodes[node]["out_degree"] = int(out_degrees.get(node, 0))
                G.nodes[node]["community_id"] = int(communities.get(node, 0))
                G.nodes[node]["total_comments"] = int(a_info.get("total_comments", 1))
                G.nodes[node]["total_likes"] = int(a_info.get("total_likes_received", 0))
                G.nodes[node]["rfm_tier"] = str(a_info.get("rfm_tier", a_info.get("frequency_tier", "Regular")))
                G.nodes[node]["is_bot_suspect"] = bool(b_info.get("is_bot_suspect", False))

        return G

    def build_bipartite_network(
        self,
        df_comments: Optional[pd.DataFrame] = None,
        df_videos: Optional[pd.DataFrame] = None,
        max_authors: int = 300
    ) -> nx.Graph:
        """Constructs two-mode Author-Video bipartite network."""
        if df_comments is None:
            df_comments, df_videos, _, _ = self.load_data_layers()

        B = nx.Graph(name="ytint_author_video_bipartite")
        if df_comments.empty or "author_channel_id" not in df_comments.columns or "video_id" not in df_comments.columns:
            return B

        # Filter to most active multi-video authors to prevent layout explosion
        edges_df = df_comments.dropna(subset=["author_channel_id", "video_id"]).copy()
        author_vid_counts = edges_df.groupby("author_channel_id")["video_id"].nunique()
        top_authors = set(author_vid_counts.nlargest(max_authors).index)

        edges_filtered = edges_df[edges_df["author_channel_id"].isin(top_authors)]
        edge_weights = edges_filtered.groupby(["author_channel_id", "video_id"]).size().reset_index(name="weight")

        # Video title lookup
        video_title_map = {}
        if df_videos is not None and not df_videos.empty and "video_id" in df_videos.columns:
            title_col = "title" if "title" in df_videos.columns else "video_id"
            video_title_map = dict(zip(df_videos["video_id"].astype(str), df_videos[title_col].fillna(df_videos["video_id"]).astype(str)))

        # Author name lookup
        author_name_map = {}
        if "author_display_name" in df_comments.columns:
            author_name_map = dict(zip(
                df_comments["author_channel_id"].astype(str),
                df_comments["author_display_name"].astype(str)
            ))

        for row in edge_weights.itertuples():
            a = str(row.author_channel_id)
            v = str(row.video_id)
            w = int(row.weight)

            if not B.has_node(a):
                B.add_node(
                    a,
                    bipartite=0,
                    node_type="author",
                    label=str(author_name_map.get(a, a)),
                    weight=int(author_vid_counts.get(a, 1))
                )

            if not B.has_node(v):
                B.add_node(
                    v,
                    bipartite=1,
                    node_type="video",
                    label=str(video_title_map.get(v, v)),
                    weight=1
                )

            B.add_edge(a, v, weight=w)

        return B

    def build_cocommenting_network(
        self,
        df_comments: Optional[pd.DataFrame] = None,
        min_overlap: int = 2,
        max_authors: int = 250
    ) -> nx.Graph:
        """Constructs an Author Co-commenting Network based on shared video engagement."""
        if df_comments is None:
            df_comments, _, _, _ = self.load_data_layers()

        G_co = nx.Graph(name="ytint_cocommenting_network")
        if df_comments.empty or "author_channel_id" not in df_comments.columns or "video_id" not in df_comments.columns:
            return G_co

        # Unique author-video pairs
        pairs = df_comments[["author_channel_id", "video_id"]].dropna().drop_duplicates()
        author_videos = pairs.groupby("author_channel_id")["video_id"].apply(set).to_dict()

        # Keep authors who commented on at least 2 videos
        multi_authors = {a: vids for a, vids in author_videos.items() if len(vids) >= 2}
        if len(multi_authors) > max_authors:
            # Sort by activity volume
            top_keys = sorted(multi_authors.keys(), key=lambda a: len(multi_authors[a]), reverse=True)[:max_authors]
            multi_authors = {k: multi_authors[k] for k in top_keys}

        authors_list = list(multi_authors.keys())
        display_map = {}
        if "author_display_name" in df_comments.columns:
            display_map = dict(zip(
                df_comments["author_channel_id"].astype(str),
                df_comments["author_display_name"].astype(str)
            ))

        edges = []
        for i in range(len(authors_list)):
            for j in range(i + 1, len(authors_list)):
                a1, a2 = authors_list[i], authors_list[j]
                common = len(multi_authors[a1] & multi_authors[a2])
                if common >= min_overlap:
                    union = len(multi_authors[a1] | multi_authors[a2])
                    jaccard = round(common / union, 4) if union > 0 else 0.0
                    edges.append((str(a1), str(a2), int(common), float(jaccard)))

        for u, v, common, jaccard in edges:
            if not G_co.has_node(u):
                G_co.add_node(u, label=str(display_map.get(u, u)), video_count=len(multi_authors[u]))
            if not G_co.has_node(v):
                G_co.add_node(v, label=str(display_map.get(v, v)), video_count=len(multi_authors[v]))
            G_co.add_edge(u, v, weight=common, jaccard=jaccard)

        return G_co

    def export_graph(self, G: nx.Graph, file_stem: str, formats: List[str] = None) -> Dict[str, Path]:
        """Serializes a network graph to GEXF and/or GraphML files in data/output/networks/."""
        formats = formats or ["gexf", "graphml"]
        exported_paths = {}

        if "gexf" in formats:
            gexf_path = self.networks_dir / f"{file_stem}.gexf"
            try:
                nx.write_gexf(G, str(gexf_path), encoding="utf-8")
                exported_paths["gexf"] = gexf_path
                logger.info(f"✅ Exported GEXF: {gexf_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            except Exception as e:
                logger.error(f"Failed to export GEXF for {file_stem}: {e}")

        if "graphml" in formats:
            graphml_path = self.networks_dir / f"{file_stem}.graphml"
            try:
                nx.write_graphml(G, str(graphml_path), encoding="utf-8")
                exported_paths["graphml"] = graphml_path
                logger.info(f"✅ Exported GraphML: {graphml_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            except Exception as e:
                logger.error(f"Failed to export GraphML for {file_stem}: {e}")

        return exported_paths

    def export_all_networks(self, formats: List[str] = None) -> Dict[str, Any]:
        """Generates and exports all three topological graphs."""
        formats = formats or ["gexf", "graphml"]
        df_comments, df_videos, df_authors, df_bots = self.load_data_layers()

        results = {}

        # 1. Author Reply Network
        G_reply = self.build_author_reply_network(df_comments, df_authors, df_bots)
        reply_paths = self.export_graph(G_reply, "author_reply_network", formats)
        results["author_reply_network"] = {
            "nodes": G_reply.number_of_nodes(),
            "edges": G_reply.number_of_edges(),
            "files": {k: str(v) for k, v in reply_paths.items()}
        }

        # 2. Bipartite Network
        G_bipartite = self.build_bipartite_network(df_comments, df_videos)
        bipartite_paths = self.export_graph(G_bipartite, "bipartite_network", formats)
        results["bipartite_network"] = {
            "nodes": G_bipartite.number_of_nodes(),
            "edges": G_bipartite.number_of_edges(),
            "files": {k: str(v) for k, v in bipartite_paths.items()}
        }

        # 3. Co-commenting Network
        G_cocomment = self.build_cocommenting_network(df_comments)
        cocomment_paths = self.export_graph(G_cocomment, "cocommenting_network", formats)
        results["cocommenting_network"] = {
            "nodes": G_cocomment.number_of_nodes(),
            "edges": G_cocomment.number_of_edges(),
            "files": {k: str(v) for k, v in cocomment_paths.items()}
        }

        return results

    @staticmethod
    def generate_interactive_network_figure(
        G: nx.Graph,
        top_k: int = 100,
        color_dimension: str = "community_id"
    ) -> go.Figure:
        """Renders a high-performance Plotly 2D force-directed spring layout graph."""
        if G.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(
                title="Empty Network Topology",
                template="plotly_dark",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return fig

        # 1. Subgraph to top-K most influential nodes by degree or PageRank
        nodes_with_metric = []
        for n in G.nodes():
            pr = G.nodes[n].get("pagerank", None)
            deg = G.degree(n)
            score = pr if pr is not None else deg
            nodes_with_metric.append((n, score))

        nodes_with_metric.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = set(n for n, _ in nodes_with_metric[:top_k])
        sub_G = G.subgraph(selected_nodes).copy()

        # 2. Compute 2D Spring Layout
        pos = nx.spring_layout(sub_G, seed=42, k=0.35, iterations=40)

        # 3. Build Edge Traces
        edge_x = []
        edge_y = []
        for u, v in sub_G.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.8, color="rgba(100, 120, 150, 0.35)"),
            hoverinfo="none",
            mode="lines"
        )

        # 4. Build Node Traces
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []

        for node in sub_G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            n_data = sub_G.nodes[node]
            label = n_data.get("label", node)
            pr = n_data.get("pagerank", 0.0)
            in_deg = n_data.get("in_degree", sub_G.in_degree(node) if hasattr(sub_G, "in_degree") else sub_G.degree(node))
            out_deg = n_data.get("out_degree", sub_G.out_degree(node) if hasattr(sub_G, "out_degree") else 0)
            comm = n_data.get("community_id", 0)
            rfm = n_data.get("rfm_tier", "Regular")
            bot = n_data.get("is_bot_suspect", False)
            node_type = n_data.get("node_type", "author")

            # Size proportional to PageRank or degree
            base_size = max(8, min(35, int(10 + (pr * 500 if pr > 0 else in_deg * 2))))
            node_sizes.append(base_size)

            # Pick color attribute
            if color_dimension == "community_id":
                node_colors.append(comm)
            elif color_dimension == "pagerank":
                node_colors.append(pr)
            elif color_dimension == "in_degree":
                node_colors.append(in_deg)
            elif color_dimension == "is_bot_suspect":
                node_colors.append(1 if bot else 0)
            else:
                node_colors.append(comm)

            hover_html = (
                f"<b>{label}</b><br>"
                f"Type: {node_type.upper()}<br>"
                f"Community Cluster: #{comm}<br>"
                f"In-Degree (Replies Received): {in_deg}<br>"
                f"Out-Degree (Replies Sent): {out_deg}<br>"
                f"PageRank Authority: {pr:.5f}<br>"
                f"Loyalty Tier: {rfm}<br>"
                f"Bot Suspect: {'🚨 YES' if bot else 'No'}"
            )
            node_text.append(hover_html)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text" if len(sub_G) <= 35 else "markers",
            text=[sub_G.nodes[n].get("label", n)[:12] for n in sub_G.nodes()] if len(sub_G) <= 35 else None,
            textposition="top center",
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale="Viridis",
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=12,
                    title=dict(
                        text=color_dimension.replace("_", " ").title(),
                        side="right"
                    ),
                    xanchor="left"
                ),
                line=dict(width=1, color="rgba(255, 255, 255, 0.4)")
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f"<b>Interactive Network Topology</b> // Top {len(sub_G)} Influential Nodes ({sub_G.number_of_edges():,} Active Connections)",
                    font=dict(size=15)
                ),
                showlegend=False,
                hovermode="closest",
                template="plotly_dark",
                margin=dict(b=20, l=10, r=10, t=50),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        return fig


def main():
    parser = argparse.ArgumentParser(description="ytint Network Graph Exporter (GEXF & GraphML)")
    parser.add_argument(
        "--network",
        choices=["author", "bipartite", "cocomment", "all"],
        default="all",
        help="Target network graph to build and export"
    )
    parser.add_argument(
        "--format",
        choices=["gexf", "graphml", "all"],
        default="all",
        help="Export format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory (defaults to config data/output)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all networks in all formats"
    )
    args = parser.parse_args()

    target_network = "all" if args.all else args.network
    target_format = "all" if args.all else args.format
    fmt_list = ["gexf", "graphml"] if target_format == "all" else [target_format]
    exporter = NetworkExporter(output_dir=Path(args.output_dir) if args.output_dir else None)

    print(f"🌐 Initializing ytint Network Graph Exporter...")
    print(f"   Target Network: {target_network.upper()} | Formats: {fmt_list}")

    if args.network in ["all", "author"]:
        print("\n👥 Building Author Reply Network...")
        G = exporter.build_author_reply_network()
        paths = exporter.export_graph(G, "author_reply_network", fmt_list)
        print(f"   Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")
        for k, p in paths.items():
            print(f"   📁 {k.upper()}: {p}")

    if args.network in ["all", "bipartite"]:
        print("\n🔗 Building Author-Video Bipartite Network...")
        G = exporter.build_bipartite_network()
        paths = exporter.export_graph(G, "bipartite_network", fmt_list)
        print(f"   Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")
        for k, p in paths.items():
            print(f"   📁 {k.upper()}: {p}")

    if args.network in ["all", "cocomment"]:
        print("\n🤝 Building Author Co-Commenting Network...")
        G = exporter.build_cocommenting_network()
        paths = exporter.export_graph(G, "cocommenting_network", fmt_list)
        print(f"   Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")
        for k, p in paths.items():
            print(f"   📁 {k.upper()}: {p}")

    print("\n🎉 Network Graph Export Complete!")


if __name__ == "__main__":
    main()
