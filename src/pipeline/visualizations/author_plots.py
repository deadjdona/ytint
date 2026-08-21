"""Phase 3: Author Profiling, Network Topologies, Forensics, and CIB Visualizations."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from .theme import setup_theme


def plot_author_pareto(df_authors, out_dir):
    """
    Task 3: Author Zipf Pareto Distribution Chart
    Generates a Pareto distribution showing the Zipf power-law of super-fans.
    """
    print("📊 Generating Zipf Author Pareto Chart...")
    
    df_plot = df_authors.sort_values('frequency', ascending=False).reset_index(drop=True)
    df_plot['cumulative_freq'] = df_plot['frequency'].cumsum()
    df_plot['cumulative_percent'] = 100 * df_plot['cumulative_freq'] / df_plot['frequency'].sum()
    
    top_n = min(100, len(df_plot))
    df_top = df_plot.head(top_n)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    ax1.bar(range(top_n), df_top['frequency'], color='steelblue', alpha=0.7)
    ax1.set_xlabel(f"Top {top_n} Authors (Ranked by Activity)")
    ax1.set_ylabel("Total Comments Made", color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks([])
    
    ax2 = ax1.twinx()
    ax2.plot(range(top_n), df_top['cumulative_percent'], color='darkorange', marker='.', linewidth=2)
    ax2.set_ylabel("Cumulative Percentage of All Comments (%)", color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 105)
    
    plt.title(f"Author Activity Power Law (Zipf Distribution) - Top {top_n}", pad=15)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Demonstrates the Pareto principle where a tiny fraction of highly active 'super-fans' generate the vast majority of all comments.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / "author_pareto.png")
    plt.close()

def plot_force_directed_network(interim_dir, out_dir, max_nodes=2000):
    """
    Task 8: Force-Directed Network Layout
    Force-directed layout of the author interaction network, colored by community.
    """
    print("🕸️ Generating Force-Directed Author Network...")
    
    authors_file = interim_dir.parent / "output" / "authors_final.parquet"
    if not authors_file.exists():
        print("⚠️ No authors_final.parquet found. Skipping network plot.")
        return
    
    df_auth = pd.read_parquet(authors_file)
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        return
    
    df_c = pd.read_parquet(comments_file, columns=['comment_id', 'parent_id', 'author_channel_id'])
    comment_to_author = dict(zip(df_c['comment_id'], df_c['author_channel_id']))
    
    edges = []
    for row in df_c.itertuples():
        if pd.notna(row.parent_id) and row.parent_id in comment_to_author:
            a, b = row.author_channel_id, comment_to_author[row.parent_id]
            if pd.notna(a) and pd.notna(b) and a != b:
                edges.append((a, b))
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    if G.number_of_nodes() > max_nodes:
        G_und = G.to_undirected()
        core = nx.k_core(G_und, k=3)
        if core.number_of_nodes() > 0:
            G = G.subgraph(core.nodes()).copy()
        else:
            top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            G = G.subgraph([n for n, _ in top_nodes]).copy()
    
    community_map = dict(zip(df_auth['author_channel_id'], df_auth.get('community_id', 0)))
    
    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    colors = [community_map.get(n, 0) for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap='tab20', node_size=15, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.05, arrows=False, ax=ax)
    
    ax.set_title(f"Author Interaction Network (N={G.number_of_nodes()}, colored by community)", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Network graph mapping who replies to whom. Colors indicate distinct conversational echo-chambers or communities.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'author_network_force.png')
    plt.close()

def plot_rfm_3d(df_authors, out_dir):
    """
    Task 9: Interactive 3D RFM Scatter
    3D scatter of RFM segmentation using Plotly.
    """
    print("💎 Generating RFM 3D Scatter...")
    
    try:
        import plotly.express as px
    except ImportError:
        print("⚠️ plotly not installed. Skipping RFM 3D scatter.")
        return
    
    required = ['recency', 'frequency', 'monetary']
    if not all(c in df_authors.columns for c in required):
        print("⚠️ Missing RFM columns. Skipping 3D scatter.")
        return
    
    color_col = 'rfm_cohort' if 'rfm_cohort' in df_authors.columns else None
    
    fig = px.scatter_3d(
        df_authors, x='recency', y='frequency', z='monetary',
        color=color_col, opacity=0.6,
        title='RFM Author Segmentation (3D)<br><sup>Explanation: Segments users by Recency (last active), Frequency (comment volume), and Monetary (engagement/likes).</sup>',
        labels={'recency': 'Recency (days)', 'frequency': 'Frequency', 'monetary': 'Engagement'}
    )
    fig.write_html(str(out_dir / 'rfm_3d.html'))
    print("  → Saved as interactive HTML: rfm_3d.html")

def plot_lorenz_curve(df_authors, out_dir):
    """
    Task 13: Lorenz Curve (Gini)
    Lorenz curve with Gini coefficient for attention inequality.
    """
    print("📐 Generating Lorenz Curve (Engagement Inequality)...")
    
    if 'frequency' not in df_authors.columns:
        return
    
    sorted_freq = np.sort(df_authors['frequency'].values)
    n = len(sorted_freq)
    cum_freq = np.cumsum(sorted_freq) / sorted_freq.sum()
    lorenz_x = np.arange(1, n + 1) / n
    
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    gini = 1 - 2 * trapz_func(cum_freq, lorenz_x)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(lorenz_x, cum_freq, color='crimson', linewidth=2, label=f'Lorenz (Gini={gini:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
    ax.fill_between(lorenz_x, lorenz_x, cum_freq, color='crimson', alpha=0.1)
    
    ax.set_xlabel('Cumulative Share of Authors')
    ax.set_ylabel('Cumulative Share of Comments')
    ax.set_title('Lorenz Curve: Author Engagement Inequality', pad=15)
    ax.legend()
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: A higher Gini coefficient (closer to 1.0) indicates extreme inequality where a few voices dominate the conversation.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'lorenz_curve.png')
    plt.close()

def plot_bipartite_network(out_dir, edges_file, nodes_file):
    print("🕸️ Generating Author-Video Bipartite Graph...")
    edges = pd.read_parquet(edges_file)
    nodes = pd.read_parquet(nodes_file).set_index('node')
    if edges.empty: return
    
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'])
        
    videos = [n for n, attr in nodes.iterrows() if attr['type'] == 'video' and n in G.nodes()]
    
    node_colors = []
    node_sizes = []
    
    max_v_weight = nodes[nodes['type']=='video']['weight'].max() or 1
    max_a_weight = nodes[nodes['type']=='author']['weight'].max() or 1
    
    for n in G.nodes():
        if n in videos:
            node_colors.append('#06d6a0')
            sz = (nodes.loc[n, 'weight'] / max_v_weight) * 3000 + 500
            node_sizes.append(sz)
        else:
            node_colors.append('#118ab2')
            sz = (nodes.loc[n, 'weight'] / max_a_weight) * 300 + 50
            node_sizes.append(sz)
            
    fig, ax = plt.subplots(figsize=(16, 16))
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
    
    video_labels = {v: str(v)[:10] + '...' for v in videos}
    nx.draw_networkx_labels(G, pos, labels=video_labels, font_size=10, font_family="sans-serif", font_weight="bold", ax=ax)
    
    ax.set_title('Author–Video Bipartite Network (Cross-pollination)', pad=15, fontsize=18)
    ax.axis('off')
    
    import matplotlib.lines as mlines
    v_patch = mlines.Line2D([], [], color='#06d6a0', marker='o', linestyle='None', markersize=12, label='Videos')
    a_patch = mlines.Line2D([], [], color='#118ab2', marker='o', linestyle='None', markersize=8, label='Highly Active Authors')
    ax.legend(handles=[v_patch, a_patch], loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Bipartite network connecting the most active commenters to the videos they engage with. Shows audience overlap between different videos.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'bipartite_network.png')
    plt.close()

def plot_cocommenting_network(out_dir, edges_file, nodes_file):
    print("🕸️ Generating Co-commenting Graph...")
    edges = pd.read_parquet(edges_file)
    nodes = pd.read_parquet(nodes_file).set_index('node')
    if edges.empty: return
    
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
        
    node_colors = '#ff9f1c' # Orange for co-commenters
    node_sizes = []
    
    max_freq = nodes['freq'].max() if not nodes.empty else 1
    
    for n in G.nodes():
        sz = (nodes.loc[n, 'freq'] / max_freq) * 800 + 50 if n in nodes.index else 100
        node_sizes.append(sz)
            
    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(G, k=0.4, iterations=50, seed=42)
    
    edge_widths = [d['weight'] for u, v, d in G.edges(data=True)]
    max_w = max(edge_widths) if edge_widths else 1
    edge_widths = [(w/max_w)*4 + 0.5 for w in edge_widths]
    
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=edge_widths, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
    
    ax.set_title('Co-commenting Network (Audience Travel)', pad=15, fontsize=18)
    ax.axis('off')
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Connects authors who comment on the EXACT SAME videos. Thicker lines = more videos in common. Reveals 'traveling squads' of super-users.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'cocomment_network.png')
    plt.close()

def plot_frequency_tiers(out_dir, tiers_file):
    print("📊 Generating Frequency Tiers Distribution...")
    df = pd.read_parquet(tiers_file)
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#ef476f', '#f78c6b', '#ffd166', '#06d6a0', '#118ab2']
    
    wedges, texts, autotexts = ax.pie(df['author_count'], labels=df['tier'], colors=colors, 
                                      autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                      wedgeprops=dict(width=0.4, edgecolor='w'))
                                      
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.setp(texts, size=12)
    
    ax.set_title('Commenting Frequency Tiers', pad=20, fontsize=16)
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Categorizes users by total comment volume. Often reveals a 'Long Tail' where 90% of users leave 1 comment, but 10% generate 90% of engagement.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'frequency_tiers.png')
    plt.close()

def plot_bot_heuristics(out_dir, bot_file):
    print("🤖 Generating Bot Detection Scatter Plot...")
    df = pd.read_parquet(bot_file)
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    humans = df[~df['is_bot']]
    bots = df[df['is_bot']]
    
    ax.scatter(humans['total_comments'], humans['unique_ratio'], alpha=0.3, color='#118ab2', label=f'Human ({len(humans):,})', s=10)
    ax.scatter(bots['total_comments'], bots['unique_ratio'], alpha=0.7, color='#ef476f', label=f'Bot/Spammer ({len(bots):,})', s=30, marker='X')
    
    ax.set_xscale('log')
    ax.set_xlabel('Total Comments (Log Scale)', fontsize=12)
    ax.set_ylabel('Unique Content Ratio (1.0 = All Unique)', fontsize=12)
    ax.set_title('Bot vs Human Heuristics', pad=15, fontsize=16)
    
    ax.axvline(10, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5)
    
    ax.legend()
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Identifies bots/spammers by looking for high-volume accounts that copy-paste identical text. Bots cluster in the bottom-left/right with low uniqueness.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'bot_heuristics.png')
    plt.close()

def plot_driveby_loyalists(out_dir, dist_file):
    print("🚗 Generating Drive-by vs Loyalists Chart...")
    df = pd.read_parquet(dist_file)
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#ef476f', '#ffd166', '#06d6a0']
    
    ax.bar(df['classification'], df['author_count'], color=colors, edgecolor='black', alpha=0.9)
    
    for i, v in enumerate(df['author_count']):
        ax.text(i, v + (max(df['author_count']) * 0.02), f"{v:,}", ha='center', fontsize=11, fontweight='bold')
        
    ax.set_title('Audience Breadth: Drive-by vs Loyalists', pad=15, fontsize=16)
    ax.set_ylabel('Number of Unique Authors')
    
    # Clean up x-labels to just the short name
    labels = [c.split('.')[1].strip() for c in df['classification']]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Categorizes users by the number of unique videos they have commented on, ignoring total comment volume. 'Drive-bys' only ever touch a single video.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'driveby_loyalists.png')
    plt.close()

def plot_like_inflation(out_dir, inflation_file):
    print("📈 Generating Like Inflation Scatter Plot...")
    df = pd.read_parquet(inflation_file)
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    normal = df[df['is_suspicious'] == False]
    sus = df[df['is_suspicious'] == True]
    
    ax.scatter(normal['like_count'], normal['reply_count'], alpha=0.3, color='#118ab2', label='Organic Engagement', s=10)
    ax.scatter(sus['like_count'], sus['reply_count'], alpha=0.8, color='#ef476f', label='Suspicious Inflation (High Likes, No Replies)', s=20)
    
    ax.set_xscale('log')
    ax.set_yscale('symlog')
    
    ax.set_xlabel('Like Count (Log Scale)', fontsize=12)
    ax.set_ylabel('Reply Count (Branching Factor)', fontsize=12)
    ax.set_title('Fraud Analytics: Suspicious Like Inflation', pad=20, fontsize=16)
    
    ax.legend()
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Top-level comments normally generate replies relative to their likes. Dots hugging the X-axis (massive likes, zero replies) indicate botted/astroturfed like inflation.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'like_inflation.png')
    plt.close()

def plot_author_fingerprints(out_dir, fingerprint_file):
    print("🕵️ Generating Author Fingerprint Radar Chart...")
    df = pd.read_parquet(fingerprint_file)
    if df.empty: return
    
    labels = ['Positivity', 'Negativity', 'Avg Length', 'Reply Rate', 'Branching Factor']
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for idx, row in df.iterrows():
        values = [row['positivity_norm'], row['negativity_norm'], row['avg_length_norm'], row['reply_rate_norm'], row['branching_factor_norm']]
        values += values[:1]
        
        # We only use first 8 chars of author ID for display to keep it clean
        ax.plot(angles, values, linewidth=2, label=f"User: {row['author_channel_id'][:8]}")
        ax.fill(angles, values, alpha=0.1)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    
    plt.title('Author Fingerprints: Top 5 Most Active Users', pad=20, fontsize=16)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.figtext(0.5, 0.015, "Explanation: Behavioral footprints of the channel's most active super-fans. Shows if a user acts as a positive conversationalist, an angry troll, or a long-form essayist.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'author_fingerprints.png')
    plt.close()

def plot_impersonation(out_dir, impersonation_file):
    print("🕵️ Generating Impersonation Bar Chart...")
    df = pd.read_parquet(impersonation_file)
    if df.empty: return
    
    # Take top 10 most spoofed names
    top_10 = df.head(10).sort_values(by='unique_accounts', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(top_10['author_display_name'], top_10['unique_accounts'], color='#ef476f', edgecolor='black')
    
    ax.set_xlabel('Number of Unique Channel IDs using this Name', fontsize=12)
    ax.set_title('Fraud Analytics: Display-Name Impersonation / Spoofing', pad=20, fontsize=16)
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: YouTube allows non-unique display names. Scammers exploit this by copying creator names. This chart highlights names used by multiple distinct channel IDs.", ha="center", fontsize=11, color="dimgray", wrap=True)
    
    plt.savefig(out_dir / 'impersonation_detection.png')
    plt.close()

def plot_cib_rings(out_dir, cib_rings_file):
    print("🕸️ Generating Coordinated Inauthentic Behavior (CIB) Graph...")
    df = pd.read_parquet(cib_rings_file)
    if df.empty: return
    
    top_rings = df.head(15).copy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_rings, x='total_synchronized_events', y='ring_id', palette='Reds_r', hue='ring_id', legend=False, ax=ax)
    ax.set_xscale('log')
    ax.set_title('Top Coordinated Inauthentic Behavior (CIB) Rings by Sync Events (Log Scale)', pad=15, fontsize=14)
    ax.set_xlabel('Total Synchronized Event Pairings Across Videos (Log Scale)', fontsize=11)
    ax.set_ylabel('CIB Ring Identifier', fontsize=11)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Identifies multi-account astroturfing rings that repeatedly comment in tightly synchronized time windows across videos.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'cib_rings_graph.png')
    plt.close()
