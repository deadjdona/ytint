"""ytint // Network Exporter Unit Tests (tests/test_network_exporter.py)

Validates:
1. Author Reply Network (Directed) graph generation and node/edge attributes.
2. Author-Video Bipartite Network construction and partitioning.
3. Co-commenting Network construction and Jaccard edge weighting.
4. GEXF (Gephi) and GraphML (Cytoscape) serialization and round-trip deserialization.
5. Interactive WebGL Plotly force-directed figure generation.
6. Empty corpus resilience and error handling.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest
from pathlib import Path
import plotly.graph_objects as go

from engine.network_exporter import NetworkExporter


@pytest.fixture
def mock_network_data():
    """Generates synthetic comments, videos, and authors for network testing."""
    df_comments = pd.DataFrame([
        {
            "comment_id": "c1",
            "video_id": "v1",
            "parent_id": None,
            "author_channel_id": "a_root",
            "author_display_name": "Alice Root",
            "vader_compound": 0.5,
            "toxicity": 0.05
        },
        {
            "comment_id": "c2",
            "video_id": "v1",
            "parent_id": "c1",
            "author_channel_id": "b_reply",
            "author_display_name": "Bob Replier",
            "vader_compound": -0.2,
            "toxicity": 0.1
        },
        {
            "comment_id": "c3",
            "video_id": "v1",
            "parent_id": "c1",
            "author_channel_id": "c_reply",
            "author_display_name": "Charlie Replier",
            "vader_compound": 0.8,
            "toxicity": 0.01
        },
        {
            "comment_id": "c4",
            "video_id": "v2",
            "parent_id": None,
            "author_channel_id": "b_reply",
            "author_display_name": "Bob Replier",
            "vader_compound": 0.1,
            "toxicity": 0.02
        },
        {
            "comment_id": "c5",
            "video_id": "v2",
            "parent_id": "c4",
            "author_channel_id": "c_reply",
            "author_display_name": "Charlie Replier",
            "vader_compound": 0.3,
            "toxicity": 0.0
        }
    ])

    df_videos = pd.DataFrame([
        {"video_id": "v1", "title": "Deep Learning Tutorial"},
        {"video_id": "v2", "title": "Transformer Networks In-Depth"}
    ])

    df_authors = pd.DataFrame([
        {"author_channel_id": "a_root", "total_comments": 10, "total_likes_received": 150, "rfm_tier": "Champion"},
        {"author_channel_id": "b_reply", "total_comments": 8, "total_likes_received": 45, "rfm_tier": "Loyalist"},
        {"author_channel_id": "c_reply", "total_comments": 5, "total_likes_received": 20, "rfm_tier": "Regular"}
    ])

    df_bots = pd.DataFrame([
        {"author_channel_id": "a_root", "is_bot_suspect": False},
        {"author_channel_id": "b_reply", "is_bot_suspect": False},
        {"author_channel_id": "c_reply", "is_bot_suspect": False}
    ])

    return df_comments, df_videos, df_authors, df_bots


def test_exporter_init(tmp_path):
    """Verify NetworkExporter initializes and creates output network directory."""
    interim = tmp_path / "interim"
    output = tmp_path / "output"
    exporter = NetworkExporter(interim_dir=interim, output_dir=output)

    assert exporter.interim_dir == interim
    assert exporter.output_dir == output
    assert exporter.networks_dir.exists()


def test_build_author_reply_network(tmp_path, mock_network_data):
    """Verify directed author reply network construction with node/edge metrics."""
    df_comments, _, df_authors, df_bots = mock_network_data
    exporter = NetworkExporter(interim_dir=tmp_path / "interim", output_dir=tmp_path / "output")

    G = exporter.build_author_reply_network(df_comments, df_authors, df_bots)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() >= 2

    # Verify edge attributes
    edge = G.get_edge_data("b_reply", "a_root")
    assert edge is not None
    assert edge["weight"] == 1
    assert "sentiment_mean" in edge
    assert "toxicity_mean" in edge

    # Verify node attributes
    node_a = G.nodes["a_root"]
    assert node_a["label"] == "Alice Root"
    assert "pagerank" in node_a
    assert node_a["in_degree"] >= 1
    assert "community_id" in node_a
    assert node_a["rfm_tier"] == "Champion"
    assert node_a["is_bot_suspect"] is False


def test_build_bipartite_network(tmp_path, mock_network_data):
    """Verify two-mode bipartite network construction."""
    df_comments, df_videos, _, _ = mock_network_data
    exporter = NetworkExporter(interim_dir=tmp_path / "interim", output_dir=tmp_path / "output")

    B = exporter.build_bipartite_network(df_comments, df_videos)

    assert isinstance(B, nx.Graph)
    assert nx.is_bipartite(B)

    # Verify partitions
    author_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    video_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]

    assert "a_root" in author_nodes
    assert "v1" in video_nodes
    assert B.nodes["v1"]["label"] == "Deep Learning Tutorial"
    assert B.has_edge("a_root", "v1")


def test_build_cocommenting_network(tmp_path, mock_network_data):
    """Verify co-commenting network construction based on shared video engagement."""
    df_comments, _, _, _ = mock_network_data
    exporter = NetworkExporter(interim_dir=tmp_path / "interim", output_dir=tmp_path / "output")

    # b_reply and c_reply both commented on v1 and v2 (2 shared videos)
    G_co = exporter.build_cocommenting_network(df_comments, min_overlap=2)

    assert isinstance(G_co, nx.Graph)
    assert G_co.has_edge("b_reply", "c_reply")
    edge = G_co.get_edge_data("b_reply", "c_reply")
    assert edge["weight"] == 2
    assert edge["jaccard"] > 0.0


def test_export_gexf_and_graphml_roundtrip(tmp_path, mock_network_data):
    """Verify GEXF and GraphML serialization and deserialization."""
    df_comments, _, df_authors, df_bots = mock_network_data
    exporter = NetworkExporter(interim_dir=tmp_path / "interim", output_dir=tmp_path / "output")

    G = exporter.build_author_reply_network(df_comments, df_authors, df_bots)
    exported = exporter.export_graph(G, "test_reply_network", ["gexf", "graphml"])

    assert "gexf" in exported
    assert "graphml" in exported
    assert Path(exported["gexf"]).exists()
    assert Path(exported["graphml"]).exists()

    # Roundtrip validation: deserialize with networkx
    G_gexf = nx.read_gexf(exported["gexf"])
    assert G_gexf.number_of_nodes() == G.number_of_nodes()
    assert G_gexf.number_of_edges() == G.number_of_edges()

    G_graphml = nx.read_graphml(exported["graphml"])
    assert G_graphml.number_of_nodes() == G.number_of_nodes()
    assert G_graphml.number_of_edges() == G.number_of_edges()


def test_generate_interactive_network_figure(mock_network_data):
    """Verify generation of Plotly force-directed network diagram."""
    df_comments, _, df_authors, df_bots = mock_network_data
    exporter = NetworkExporter()

    G = exporter.build_author_reply_network(df_comments, df_authors, df_bots)
    fig = NetworkExporter.generate_interactive_network_figure(G, top_k=50, color_dimension="community_id")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # 1 edge trace + 1 node trace
    assert fig.data[0].mode == "lines"
    assert "markers" in fig.data[1].mode
    assert len(fig.data[1].x) == G.number_of_nodes()


def test_empty_corpus_resilience(tmp_path):
    """Verify graceful handling when dataframes or graphs are empty."""
    exporter = NetworkExporter(interim_dir=tmp_path / "interim", output_dir=tmp_path / "output")

    empty_df = pd.DataFrame()
    G = exporter.build_author_reply_network(empty_df)
    assert G.number_of_nodes() == 0

    fig = NetworkExporter.generate_interactive_network_figure(G)
    assert isinstance(fig, go.Figure)

    exported = exporter.export_graph(G, "empty_net")
    assert "gexf" in exported
    assert Path(exported["gexf"]).exists()
