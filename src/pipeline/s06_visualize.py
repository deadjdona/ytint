"""Stage 6: Visualizations (s06_visualize.py)

Code Review Task Alignment:
- 1. Kaplan-Meier Survival Plot: plot_kaplan_meier() -> kaplan_meier_survival.png
- 2. 2D UMAP Semantic Clusters: plot_umap_semantics() -> umap_semantics.png (Hexbin fallback for >10k pts)
- 3. Author Zipf Pareto Chart: plot_author_pareto() -> author_pareto.png
- 4. SHAP Feature Importance Summary: plot_shap_summary() -> shap_summary.png
- 5. 24×7 Diurnal Activity Heatmap: plot_diurnal_heatmap() -> diurnal_heatmap.png
- 6. Plutchik Emotion Radar: plot_plutchik_radar() -> plutchik_emotion_wheel.png
- 7. Sentiment Ridge Plot (Joyplot): plot_sentiment_ridges() -> sentiment_ridges.png
- 8. Force-Directed Network Layout: plot_force_directed_network() -> author_network_force.png
- 9. Interactive 3D RFM Scatter: plot_rfm_3d() -> rfm_3d.html
- 10. Sentiment Divergence Chart: plot_sentiment_divergence() -> sentiment_divergence.png
- 11. Toxicity Time-of-Day Heatmap: plot_toxicity_heatmap() -> toxicity_heatmap.png
- 12. Reply Depth Distribution: plot_reply_depth_distribution() -> reply_depth_distribution.png
- 13. Lorenz Curve (Gini): plot_lorenz_curve() -> lorenz_curve.png
- 14. STL Decomposition Multi-Panel: plot_stl_decomposition() -> stl_decomposition.png
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


import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import networkx as nx
from engine.config_loader import load_config

# Set high-quality styling for publication-ready outputs
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial', 'Segoe UI Emoji', 'DejaVu Sans']


# ============================================================
# Core Pipeline Visualizations
# ============================================================

def plot_kaplan_meier(df_survival, out_dir):
    """
    Task 1: Kaplan-Meier Survival Plot
    Generates the Thread Lifespan Survival Curve.
    """
    print("📈 Generating Kaplan-Meier Thread Survival Plot...")
    plt.figure()
    
    # Step plot is standard for Kaplan-Meier
    plt.step(df_survival['timeline_hours'], df_survival['survival_probability'], where="post", color="darkred", linewidth=2)
    plt.fill_between(df_survival['timeline_hours'], df_survival['survival_probability'], step="post", color="darkred", alpha=0.1)
    
    plt.title("Kaplan-Meier Survival Estimate of Thread Lifespans", pad=15)
    plt.xlabel("Time Since First Comment (Hours)")
    plt.ylabel("Probability of Receiving More Replies")
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Shows the probability of a thread continuing to receive replies over time. A steeper drop indicates shorter-lived discussions.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / "kaplan_meier_survival.png")
    plt.close() # 🛡️ Prevent Memory Leaks

def plot_umap_semantics(df_semantics, out_dir):
    """
    Task 2: 2D UMAP Semantic Clusters Space (with Hexbin fallback for >10k points)
    Generates a 2D UMAP projection of topics and intents.
    """
    print("🌌 Generating UMAP Semantic Clusters...")
    
    df_plot = df_semantics.dropna(subset=['umap_x', 'umap_y', 'intent_label'])
    
    plt.figure(figsize=(12, 8))
    
    if len(df_plot) > 10000:
        print("⚠️ Large dataset detected. Using Hexbin aggregation to prevent visual overplotting & crashes...")
        hb = plt.hexbin(df_plot['umap_x'], df_plot['umap_y'], gridsize=50, cmap='viridis', mincnt=1)
        cb = plt.colorbar(hb, label='Comment Density')
        plt.title(f"UMAP Semantic Density Space (N={len(df_plot)})", pad=15)
    else:
        sns.scatterplot(
            data=df_plot, 
            x='umap_x', 
            y='umap_y', 
            hue='intent_label', 
            alpha=0.6, 
            s=20,
            linewidth=0
        )
        plt.title(f"UMAP Semantic Clusters by Intent (N={len(df_plot)})", pad=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: 2D projection of comment meanings. Proximity indicates semantic similarity; colors denote inferred user intent.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / "umap_semantics.png")
    plt.close()

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
    
def plot_shap_summary(shap_values, shap_features, out_dir):
    """
    Task 4: SHAP Feature Importance Summary Beeswarm
    Visualizes XGBoost Feature Importance via SHAP Values.
    """
    print("🧠 Generating SHAP XGBoost Attribution Summary...")
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(shap_values, shap_features, show=False)
    
    plt.title("SHAP Feature Importance (Predicting Like-Counts)", pad=15)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Highlights which linguistic or emotional features most strongly drive comment likes. Red dots indicate high feature values.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / "shap_summary.png")
    plt.close()


# ============================================================
# Task Visualizations
# ============================================================

def plot_diurnal_heatmap(out_dir, diurnal_file):
    """
    Task 5: 24×7 Diurnal Activity Heatmap
    24×7 heatmap of comment activity by hour of day × day of week.
    """
    print("🕐 Generating 24×7 Diurnal Activity Heatmap...")
    
    diurnal = pd.read_parquet(diurnal_file)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        diurnal.values if hasattr(diurnal, 'values') else diurnal,
        cmap='YlOrRd', ax=ax, annot=False, linewidths=0.5,
        xticklabels=[f'{h:02d}:00' for h in range(24)],
        yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    )
    ax.set_title('Comment Activity Heatmap (Hour × Day of Week)', pad=15)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Darker colors represent peak engagement periods. Used to determine the optimal time to post or interact.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'diurnal_heatmap.png')
    plt.close()


def plot_plutchik_radar(df_comments, out_dir):
    """
    Task 6: Plutchik Emotion Radar Chart
    Radar chart of aggregated Plutchik emotion categories (from GoEmotions).
    """
    print("🎭 Generating Plutchik Emotion Radar Chart...")
    
    if 'emotion_1' not in df_comments.columns:
        print("⚠️ No GoEmotions data found. Skipping Plutchik wheel.")
        return
    
    # Map GoEmotions 27 labels to 8 Plutchik primary emotions
    plutchik_map = {
        'joy': 'joy', 'amusement': 'joy', 'excitement': 'joy', 'love': 'joy',
        'anger': 'anger', 'annoyance': 'anger', 'disapproval': 'anger',
        'sadness': 'sadness', 'grief': 'sadness', 'remorse': 'sadness',
        'fear': 'fear', 'nervousness': 'fear',
        'surprise': 'surprise', 'realization': 'surprise', 'confusion': 'surprise',
        'disgust': 'disgust',
        'anticipation': 'anticipation', 'desire': 'anticipation', 'curiosity': 'anticipation', 'optimism': 'anticipation',
        'trust': 'trust', 'admiration': 'trust', 'approval': 'trust', 'gratitude': 'trust', 'caring': 'trust', 'pride': 'trust',
    }
    
    all_emotions = pd.concat([df_comments['emotion_1'], df_comments['emotion_2'], df_comments['emotion_3']])
    mapped = all_emotions.map(plutchik_map).dropna()
    counts = mapped.value_counts()
    
    # Ensure all 8 Plutchik emotions present
    primary_emotions = ['joy', 'trust', 'anticipation', 'surprise', 'sadness', 'disgust', 'anger', 'fear']
    values = [counts.get(e, 0) for e in primary_emotions]
    total = sum(values) or 1
    values = [v / total for v in values]  # Normalize
    
    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(primary_emotions), endpoint=False).tolist()
    values += values[:1]  # Close the polygon
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='mediumslateblue', alpha=0.25)
    ax.plot(angles, values, color='mediumslateblue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([e.title() for e in primary_emotions], fontsize=11)
    ax.set_title("Plutchik Emotion Profile (GoEmotions)", pad=20, fontsize=14)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Radar chart showing the aggregate blend of the 8 primary human emotions expressed across the dataset.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'plutchik_emotion_wheel.png')
    plt.close()


def plot_sentiment_ridges(df_comments, out_dir, max_videos=15):
    """
    Task 7: Sentiment Ridge Plot (Joyplot)
    Joyplot / ridge plot of sentiment distributions across videos.
    """
    print("🌊 Generating Sentiment Ridge Plot...")
    
    try:
        import joypy
    except ImportError:
        print("⚠️ joypy not installed. Skipping ridge plot.")
        return
    
    top_videos = df_comments['video_id'].value_counts().head(max_videos).index
    df_plot = df_comments[df_comments['video_id'].isin(top_videos)].copy()
    
    if df_plot.empty or 'vader_compound' not in df_plot.columns:
        return
    
    df_plot['video_short'] = df_plot['video_id'].str[:12]
    
    fig, axes = joypy.joyplot(
        df_plot, by='video_short', column='vader_compound',
        figsize=(10, 12), alpha=0.6, colormap=plt.cm.coolwarm,
        linewidth=1
    )
    plt.title('Sentiment Distribution Across Videos', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Joyplot comparing the density of positive vs. negative sentiment across the most commented videos.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'sentiment_ridges.png')
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


def plot_sentiment_divergence(df_comments, out_dir, max_videos=25):
    """
    Task 10: Sentiment Divergence Chart
    Diverging bar chart: sentiment shift from root comments → replies per video.
    """
    print("↕️ Generating Sentiment Divergence Chart...")
    
    if 'vader_compound' not in df_comments.columns:
        return
    
    roots = df_comments[df_comments['parent_id'].isna()]
    replies = df_comments[df_comments['parent_id'].notna()]
    
    root_sent = roots.groupby('video_id')['vader_compound'].mean()
    reply_sent = replies.groupby('video_id')['vader_compound'].mean()
    delta = (reply_sent - root_sent).dropna().sort_values()
    
    if delta.empty:
        return
    
    delta = pd.concat([delta.head(max_videos // 2), delta.tail(max_videos // 2)])
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(delta) * 0.35)))
    colors = ['#ff3366' if v < 0 else '#00cc66' for v in delta.values]
    y_labels = [vid[:16] + '...' for vid in delta.index]
    
    ax.barh(range(len(delta)), delta.values, color=colors, height=0.7)
    ax.set_yticks(range(len(delta)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Sentiment Shift (Reply Mean − Root Mean)')
    ax.set_title('Sentiment Divergence: Root Comments → Replies', pad=15)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Shows if replies tend to be more negative (red/left) or positive (green/right) than the original root comment.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'sentiment_divergence.png')
    plt.close()


def plot_toxicity_heatmap(df_comments, out_dir):
    """
    Task 11: Toxicity Time-of-Day Heatmap
    Heatmap of toxicity scores by hour of day.
    """
    print("☠️ Generating Toxicity by Time-of-Day Heatmap...")
    
    if 'toxicity' not in df_comments.columns:
        print("⚠️ No toxicity data. Skipping.")
        return
    
    df_comments['hour'] = pd.to_datetime(df_comments['published_at']).dt.hour
    df_comments['weekday'] = pd.to_datetime(df_comments['published_at']).dt.dayofweek
    
    tox_matrix = df_comments.groupby(['weekday', 'hour'])['toxicity'].mean().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        tox_matrix, cmap='Reds', ax=ax, annot=False, linewidths=0.5,
        xticklabels=[f'{h:02d}:00' for h in range(24)],
        yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    )
    ax.set_title('Mean Toxicity Score by Time (Hour × Day)', pad=15)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Identifies specific days and hours where community toxicity or hostility statistically spikes.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'toxicity_heatmap.png')
    plt.close()


def plot_reply_depth_distribution(df_comments, out_dir):
    """
    Task 12: Reply Depth Distribution
    Bar chart of reply depth distribution with exponential decay overlay.
    """
    print("📏 Generating Reply Depth Distribution...")
    
    if 'comment_depth' not in df_comments.columns:
        return
    
    depth_counts = df_comments['comment_depth'].value_counts().sort_index()
    max_depth = min(20, depth_counts.index.max())
    depth_counts = depth_counts.loc[:max_depth]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(depth_counts.index, depth_counts.values, color='teal', alpha=0.8)
    ax.set_xlabel('Reply Depth (0 = Root)')
    ax.set_ylabel('Number of Comments')
    ax.set_title('Reply Depth Distribution', pad=15)
    ax.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Log-scale distribution of how deep conversation threads go before users stop replying.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'reply_depth_distribution.png')
    plt.close()


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


def plot_stl_decomposition(out_dir, stl_file):
    """
    Task 14: STL Decomposition Multi-Panel
    STL decomposition multi-panel chart (trend + seasonal + residual).
    """
    print("📉 Generating STL Decomposition Plot...")
    
    stl_df = pd.read_parquet(stl_file)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    axes[0].plot(stl_df['date'], stl_df['observed'], color='steelblue', linewidth=1)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('STL Decomposition of Daily Comment Volume', fontsize=14)
    
    axes[1].plot(stl_df['date'], stl_df['trend'], color='darkorange', linewidth=2)
    axes[1].set_ylabel('Trend')
    
    axes[2].plot(stl_df['date'], stl_df['seasonal'], color='forestgreen', linewidth=1)
    axes[2].set_ylabel('Seasonal')
    
    axes[3].scatter(stl_df['date'], stl_df['residual'], color='crimson', s=3, alpha=0.6)
    axes[3].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Breaks down activity into overall Trend, repeating Seasonal patterns, and unpredictable Residual noise.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'stl_decomposition.png')
    plt.close()


# ============================================================
# Main orchestration
# ============================================================

def plot_reaction_timeline(out_dir, reaction_file):
    print("⏱️ Generating Cross-Modal Reaction Timeline...")
    df = pd.read_parquet(reaction_file)
    if df.empty: return
    top_video = df['video_id'].value_counts().idxmax()
    df_plot = df[df['video_id'] == top_video].sort_values('second')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_plot['second'], df_plot.get('vader_compound', df_plot['comment_count']), color='blueviolet', linewidth=2)
    ax.fill_between(df_plot['second'], df_plot.get('vader_compound', df_plot['comment_count']), color='blueviolet', alpha=0.2)
    ax.set_title(f'Timeline Reaction Ribbon (Video: {top_video[:12]}...)', pad=15)
    ax.set_xlabel('Video Playback Time (seconds)')
    ax.set_ylabel('Mean Sentiment' if 'vader_compound' in df.columns else 'Engagement')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Maps emotional spikes to specific video playback moments (e.g. at 1m24s).", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'reaction_timeline.png')
    plt.close()

def plot_anomaly_scatter(out_dir, integrity_file, comments_file):
    print("🚨 Generating Integrity Anomaly Scatter...")
    df_int = pd.read_parquet(integrity_file)
    if df_int.empty: return
    df_com = pd.read_parquet(comments_file, columns=['comment_id', 'published_at'])
    df = df_int.merge(df_com, on='comment_id', how='inner')
    df['published_at'] = pd.to_datetime(df['published_at'])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    normal = df[~(df['is_duplicate'] | df['brigade_suspect'])]
    anomalous = df[(df['is_duplicate'] | df['brigade_suspect'])]
    
    ax.scatter(normal['published_at'], range(len(normal)), color='lightgray', s=5, alpha=0.5, label='Normal')
    ax.scatter(anomalous['published_at'], range(len(anomalous)), color='red', s=15, alpha=0.8, label='Bot/Spam Anomaly')
    ax.set_title('Bot Brigading & Spam Detection Timeline', pad=15)
    ax.legend()
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Red clusters indicate coordinated bot brigading or high volumes of duplicated spam templates.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'integrity_scatter.png')
    plt.close()

def plot_thread_polarization(out_dir, slopes_file):
    print("🌪️ Generating Thread Polarization Decay...")
    df = pd.read_parquet(slopes_file)
    if df.empty: return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['decay_slope'], bins=30, kde=True, color='darkorange', ax=ax)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title('Thread Sentiment Decay Slopes (Polarization)', pad=15)
    ax.set_xlabel('Sentiment Decay Slope (Negative = increasingly toxic over depth)')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Shows whether deep reply threads tend to devolve into toxicity (left of zero) or remain polite (right).", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'thread_polarization.png')
    plt.close()

def plot_topic_streamgraph(out_dir, topic_file):
    print("🌊 Generating Topic Evolution Streamgraph...")
    df = pd.read_parquet(topic_file)
    if df.empty: return
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(df.index, df.values.T, labels=df.columns, baseline='wiggle', alpha=0.8)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.set_title('Topic Evolution over Time (Streamgraph)', pad=15)
    plt.tight_layout(rect=[0, 0.04, 0.85, 1]) # Make room for legend
    plt.figtext(0.5, 0.015, "Explanation: Flowing alluvial diagram showing how community conversational focus (topics) drifts across months/years.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'topic_streamgraph.png')
    plt.close()

def plot_reply_latency_distribution(out_dir, latency_file):
    print("⏱️ Generating Reply Latency Distribution...")
    df = pd.read_parquet(latency_file)
    if df.empty: return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['latency_hours'], bins=50, log_scale=(True, False), color='dodgerblue', ax=ax)
    ax.set_title('Reply Latency Distribution (Time to Reply)', pad=15)
    ax.set_xlabel('Hours between Parent and Reply (Log Scale)')
    ax.set_ylabel('Number of Replies')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Log-scale distribution showing how quickly users respond to each other. Peaks indicate the 'half-life' of a conversation.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'reply_latency_distribution.png')
    plt.close()

def plot_shelf_life(out_dir, shelf_life_file):
    print("📈 Generating Shelf Life of Likes Curve...")
    df = pd.read_parquet(shelf_life_file)
    if df.empty: return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['hour_bin'], df['smoothed_likes'], color='crimson', linewidth=2, label='Avg Likes (Smoothed)')
    ax.scatter(df['hour_bin'], df['avg_likes'], color='lightcoral', alpha=0.3, s=10)
    ax.set_title('Shelf Life of Likes (First-Mover Advantage)', pad=15)
    ax.set_xlabel('Hours since Video Upload')
    ax.set_ylabel('Average Likes on Comment')
    ax.set_xlim(left=0, right=df['hour_bin'].max())
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Shows the 'First Mover Advantage'. Comments posted in the first few hours typically harvest the vast majority of likes before the engagement 'shelf life' expires.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'shelf_life_likes.png')
    plt.close()

def plot_topic_video_matrix(out_dir, matrix_file):
    print("🧩 Generating Topic × Video Matrix Heatmap...")
    df = pd.read_parquet(matrix_file)
    if df.empty: return
    if len(df) > 30:
        df = df.loc[df.sum(axis=1).sort_values(ascending=False).head(30).index]
    try:
        g = sns.clustermap(df, cmap="mako", figsize=(14, 10), 
                           dendrogram_ratio=0.1, 
                           cbar_pos=(0.02, 0.8, 0.03, 0.18),
                           linewidths=.5,
                           annot=False)
        g.fig.suptitle("Topic × Video Matrix (Normalized % of Comments)", y=0.98)
        plt.figtext(0.5, 0.015, "Explanation: Clustered heatmap showing which videos focus on which topics. Darker cells mean the topic dominates that video's discussion.", ha="center", fontsize=10, color="dimgray", wrap=True)
        g.savefig(out_dir / 'topic_video_matrix.png')
        plt.close(g.fig)
    except Exception as e:
        print(f"⚠️ Could not generate clustered matrix: {e}")

def plot_cooccurrence_network(out_dir, edges_file, nodes_file):
    print("🕸️ Generating Word Co-occurrence Network...")
    edges = pd.read_parquet(edges_file)
    nodes = pd.read_parquet(nodes_file).set_index('word')
    if edges.empty: return
    
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
        
    node_sizes = [nodes.loc[n, 'freq'] if n in nodes.index else 10 for n in G.nodes()]
    max_size = max(node_sizes) if node_sizes else 1
    node_sizes = [(size/max_size)*2000 + 50 for size in node_sizes]
    
    edge_widths = [d['weight'] for u, v, d in G.edges(data=True)]
    max_w = max(edge_widths) if edge_widths else 1
    edge_widths = [(w/max_w)*4 + 0.5 for w in edge_widths]
    
    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_widths, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='teal', alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif", font_weight="bold", ax=ax)
    
    ax.set_title('Word Co-occurrence Semantic Network', pad=15, fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Concepts that frequently appear in the same comments are linked. Larger nodes = more frequent. Thicker lines = stronger association.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'word_cooccurrence.png')
    plt.close()

def plot_ner_distribution(out_dir, ner_file):
    print("👤 Generating Named Entity Distributions...")
    df = pd.read_parquet(ner_file)
    if df.empty: return
    
    top_per = df[df['entity_label'] == 'PER']['entity_text'].value_counts().head(10)
    top_org = df[df['entity_label'] == 'ORG']['entity_text'].value_counts().head(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if not top_per.empty:
        sns.barplot(x=top_per.values, y=top_per.index, hue=top_per.index, palette='Blues_r', legend=False, ax=axes[0])
        axes[0].set_title('Top Mentioned Persons (PER)')
        axes[0].set_xlabel('Mentions')
        
    if not top_org.empty:
        sns.barplot(x=top_org.values, y=top_org.index, hue=top_org.index, palette='Greens_r', legend=False, ax=axes[1])
        axes[1].set_title('Top Mentioned Organizations (ORG)')
        axes[1].set_xlabel('Mentions')
        
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Automatically extracted entities. Shows which public figures or organizations are the main focus of community discussion.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'named_entities.png')
    plt.close()

def plot_sarcasm_distribution(out_dir, comments_file):
    print("🎭 Generating Sarcasm Distribution...")
    df = pd.read_parquet(comments_file, columns=['is_sarcasm_suspect', 'sentiment_label'])
    if df.empty or 'is_sarcasm_suspect' not in df.columns: return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Overall Proportion
    sarcasm_counts = df['is_sarcasm_suspect'].value_counts()
    axes[0].pie(sarcasm_counts, labels=['Genuine', 'Sarcasm/Irony Suspect'], 
                autopct='%1.1f%%', colors=['#4a4e69', '#fca311'], startangle=90, explode=[0, 0.1])
    axes[0].set_title('Proportion of Sarcastic Comments')
    
    # 2. Sarcasm by Sentiment Label
    if 'sentiment_label' in df.columns:
        sarcasm_df = df[df['is_sarcasm_suspect'] == True]
        sns.countplot(data=sarcasm_df, x='sentiment_label', hue='sentiment_label', order=['positive', 'neutral', 'negative'],
                      palette={'positive': '#2a9d8f', 'neutral': '#e9c46a', 'negative': '#e76f51'}, legend=False, ax=axes[1])
        axes[1].set_title('Sentiment Classification of Sarcastic Comments')
        axes[1].set_xlabel('Sentiment Assigned by XLM-RoBERTa')
        axes[1].set_ylabel('Count')
        
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Sarcasm is identified using lexical markers and structural exaggeration (e.g. excessive punctuation/caps). The right chart shows how the AI sentiment model interprets these ironic comments.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'sarcasm_analysis.png')
    plt.close()

def plot_linguistic_features(out_dir, comments_file):
    print("📝 Generating Linguistic & Stylistic Features...")
    df = pd.read_parquet(comments_file)
    if df.empty: return
    
    required_cols = ['lexical_richness', 'emoji_count', 'sentiment_label', 'all_caps_ratio', 'punctuation_intensity']
    for col in required_cols:
        if col not in df.columns:
            return
            
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Lexical Richness KDE
    sns.kdeplot(data=df, x='lexical_richness', fill=True, color='#264653', ax=axes[0, 0])
    axes[0, 0].set_title('Lexical Richness (Type-Token Ratio)')
    axes[0, 0].set_xlabel('Richness Score')
    
    # 2. Emoji Usage by Sentiment
    sns.boxplot(data=df, x='sentiment_label', y='emoji_count', hue='sentiment_label', order=['positive', 'neutral', 'negative'],
                palette={'positive': '#2a9d8f', 'neutral': '#e9c46a', 'negative': '#e76f51'}, legend=False, ax=axes[0, 1], showfliers=False)
    axes[0, 1].set_title('Emoji Usage across Sentiment')
    axes[0, 1].set_ylabel('Emojis per Comment')
    
    # 3. All Caps vs Punctuation
    sample_df = df.sample(min(10000, len(df)), random_state=42)
    sns.scatterplot(data=sample_df, x='all_caps_ratio', y='punctuation_intensity', 
                    hue='sentiment_label', alpha=0.5, palette={'positive': '#2a9d8f', 'neutral': '#e9c46a', 'negative': '#e76f51'}, ax=axes[1, 0])
    axes[1, 0].set_title('Structural Intensity (Caps vs Punctuation)')
    axes[1, 0].set_xlabel('All-Caps Ratio')
    axes[1, 0].set_ylabel('Punctuation Intensity')
    
    # 4. Readability (if present)
    if 'readability_flesch' in df.columns:
        sns.histplot(data=df, x='readability_flesch', bins=50, color='#e76f51', ax=axes[1, 1])
        axes[1, 1].set_title('Flesch Reading Ease')
        axes[1, 1].set_xlabel('Score (Higher = Easier)')
        
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Linguistic profiling shows lexical diversity, structural exaggeration (caps/punctuation), and how emojis correlate with AI sentiment.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'linguistic_profiling.png')
    plt.close()

def plot_tag_network(out_dir, edges_file, nodes_file):
    print("🏷️ Generating Hashtag & Mention Network...")
    edges = pd.read_parquet(edges_file)
    nodes = pd.read_parquet(nodes_file).set_index('node')
    if edges.empty: return
    
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
        
    node_sizes = []
    node_colors = []
    for n in G.nodes():
        freq = nodes.loc[n, 'freq'] if n in nodes.index else 5
        node_sizes.append((freq / nodes['freq'].max()) * 2000 + 50 if not nodes.empty else 100)
        node_colors.append('#3a86ff' if str(n).startswith('#') else '#ff006e')
        
    edge_widths = [d['weight'] for u, v, d in G.edges(data=True)]
    max_w = max(edge_widths) if edge_widths else 1
    edge_widths = [(w/max_w)*4 + 0.5 for w in edge_widths]
    
    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_widths, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif", font_weight="bold", ax=ax)
    
    ax.set_title('Hashtag & @Mention Co-occurrence Network', pad=15, fontsize=16)
    ax.axis('off')
    
    import matplotlib.lines as mlines
    blue_patch = mlines.Line2D([], [], color='#3a86ff', marker='o', linestyle='None', markersize=10, label='Hashtags (#)')
    pink_patch = mlines.Line2D([], [], color='#ff006e', marker='o', linestyle='None', markersize=10, label='Mentions (@)')
    ax.legend(handles=[blue_patch, pink_patch], loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Networks of cross-referenced channels and topics. Tags/Mentions used in the same comment are linked. Larger nodes = higher frequency.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'tag_network.png')
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

def plot_cohort_retention(out_dir, pct_file, count_file):
    print("👥 Generating Cohort Retention Heatmap...")
    df_pct = pd.read_parquet(pct_file).set_index('cohort')
    df_counts = pd.read_parquet(count_file).set_index('cohort')
    if df_pct.empty: return
    
    video_labels = [f"V{i+1}" for i in range(len(df_pct.columns))]
    df_pct.columns = video_labels
    df_pct.index = video_labels
    
    annot = df_pct.round(1) if len(df_pct) <= 20 else False
    
    fig, ax = plt.subplots(figsize=(max(12, len(df_pct)*0.4), max(10, len(df_pct)*0.4)))
    
    sns.heatmap(df_pct, annot=annot, fmt=".1f", cmap="YlGnBu", 
                cbar_kws={'label': '% of Cohort Retained'}, ax=ax, vmin=0, vmax=100 if len(df_pct) <= 20 else 50)
    
    ax.set_title('Acquisition Cohort Retention Matrix', pad=20, fontsize=16)
    ax.set_ylabel('Acquisition Cohort (First Video Commented On)', fontsize=12)
    ax.set_xlabel('Subsequent Engagement (Video Commented On)', fontsize=12)
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Rows represent users grouped by the first video they ever engaged with. Columns show what % of them returned to comment on later videos.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'cohort_retention.png')
    plt.close()

def plot_video_overlap(out_dir, matrix_file):
    print("🕸️ Generating Cross-Video Overlap Heatmap...")
    df_matrix = pd.read_parquet(matrix_file).set_index('video_id')
    if df_matrix.empty: return
    
    video_labels = [f"V{i+1}" for i in range(len(df_matrix))]
    df_matrix.columns = video_labels
    df_matrix.index = video_labels
    
    df_matrix = df_matrix * 100
    annot = df_matrix.round(0) if len(df_matrix) <= 15 else False
    
    fig, ax = plt.subplots(figsize=(max(12, len(df_matrix)*0.35), max(10, len(df_matrix)*0.35)))
    
    mask = np.triu(np.ones_like(df_matrix, dtype=bool))
    
    sns.heatmap(df_matrix, mask=mask, annot=annot, fmt=".0f", cmap="magma", 
                cbar_kws={'label': 'Jaccard Overlap (%)'}, ax=ax, vmin=0, vmax=20)
    
    ax.set_title('Cross-Video Audience Overlap (Jaccard Similarity)', pad=20, fontsize=16)
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Symmetric matrix showing the exact percentage of shared audience between any two videos. Higher % means the videos attracted the exact same users.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'video_overlap_matrix.png')
    plt.close()

def plot_new_vs_returning(out_dir, nr_file):
    print("👥 Generating New vs Returning Share...")
    df_nr = pd.read_parquet(nr_file)
    if df_nr.empty: return
    
    # Sort chronologically if video_id is chronologically structured, otherwise just plot as is
    # It's already in chronological order from s19
    df_nr.set_index('video_id', inplace=True)
    
    # Calculate percentages
    total = df_nr['new_count'] + df_nr['returning_count']
    df_pct = df_nr.div(total, axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Limit to top 60 videos for readability
    if len(df_pct) > 60:
        df_pct = df_pct.tail(60)
        
    x = np.arange(len(df_pct))
    width = 0.8
    
    ax.bar(x, df_pct['returning_count'], width, label='Returning (Loyalists)', color='#118ab2')
    ax.bar(x, df_pct['new_count'], width, bottom=df_pct['returning_count'], label='New (Acquisition)', color='#06d6a0')
    
    ax.set_ylabel('Audience Share (%)', fontsize=12)
    ax.set_title('New vs Returning Commenter Share Over Time', pad=15, fontsize=16)
    
    # Labels
    ax.set_xticks(x)
    labels = [f"V{i+1}" for i in range(len(df_pct))]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(out_dir / 'new_vs_returning_share.png')
    plt.close()

def plot_topic_cohorts(out_dir, cohorts_file):
    print("📊 Generating Topic-Cohort Distribution...")
    df = pd.read_parquet(cohorts_file)
    if df.empty: return
    
    df = df.sort_values('author_count', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df)*0.4)))
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(df)))
    ax.barh(df['topic_cohort'], df['author_count'], color=colors, edgecolor='black', alpha=0.8)
    
    for i, v in enumerate(df['author_count']):
        ax.text(v + (max(df['author_count']) * 0.01), i, str(v), va='center', fontsize=10, fontweight='bold')
        
    ax.set_title('Audience Topic Affinities (Cohort Sizes)', pad=15, fontsize=16)
    ax.set_xlabel('Number of Authors (Primary Topic Affinity)')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'topic_cohorts.png')
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

def plot_position_bias(out_dir, bias_file):
    print("📍 Generating Position Bias Chart...")
    df = pd.read_parquet(bias_file)
    if df.empty: return
    
    df = df[df['comment_count'] > 10]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#ef476f'
    ax1.set_xlabel('Chronological Position (1 = First Comment on Video)', fontsize=12)
    ax1.set_ylabel('Average Likes', color=color1, fontsize=12)
    ax1.plot(df['position'], df['avg_likes'], color=color1, linewidth=2, label='Avg Likes')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = '#118ab2'
    ax2.set_ylabel('Average Replies (Branching Factor)', color=color2, fontsize=12)
    ax2.plot(df['position'], df['avg_replies'], color=color2, linewidth=2, linestyle='--', label='Avg Replies')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Position Bias: The "Early Bird" Advantage', pad=20, fontsize=16)
    
    fig.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Shows how being the 1st vs 50th commenter impacts the average likes and replies (branching factor) received. The massive spike at position 1 proves the extreme 'Early Bird' positional bias.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'position_bias.png')
    plt.close()

def plot_arrival_speed(out_dir, speed_file):
    print("⏱️ Generating Arrival Speed Chart...")
    df = pd.read_parquet(speed_file)
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#ef476f', '#ffd166', '#06d6a0', '#118ab2']
    
    wedges, texts, autotexts = ax.pie(df['author_count'], labels=df['classification'], colors=colors, 
                                      autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                      wedgeprops=dict(width=0.4, edgecolor='w'))
                                      
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.setp(texts, size=12)
    
    ax.set_title('Author Arrival Speed: First Commenters vs Late Arrivals', pad=20, fontsize=16)
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Categorizes users by how quickly they typically comment after a video is published. Highlights the core 'First Responder' audience vs algorithmic long-tail viewers.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'arrival_speed.png')
    plt.close()

def plot_thread_width(out_dir, width_file):
    print("🌲 Generating Thread Width Distribution Chart...")
    df = pd.read_parquet(width_file)
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(df['thread_width'], df['frequency'], color='#ef476f', edgecolor='black', alpha=0.8)
    
    ax.set_yscale('log')
    if df['thread_width'].max() > 100:
        ax.set_xscale('log')
        
    ax.set_xlabel('Thread Width (Number of Replies to a Single Comment)', fontsize=12)
    ax.set_ylabel('Frequency (Number of Threads) - Log Scale', fontsize=12)
    ax.set_title('Conversational Branching: Thread Width Distribution', pad=20, fontsize=16)
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Shows the distribution of how many replies top-level comments spawn. Most comments get 1 reply, while viral debates spawn massive widths.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'thread_width_dist.png')
    plt.close()

def plot_resolution_patterns(out_dir, pattern_file):
    print("🏁 Generating Conversation Resolution Chart...")
    df = pd.read_parquet(pattern_file)
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df['sentiment']))
    width = 0.35
    
    ax.bar(x - width/2, df['non_terminal_ratio'], width, label='Ongoing Conversation', color='#118ab2')
    ax.bar(x + width/2, df['terminal_ratio'], width, label='Terminal Node (The Last Word)', color='#ef476f')
    
    ax.set_ylabel('Percentage of Replies (%)', fontsize=12)
    ax.set_title('Conversation Resolution Patterns', pad=20, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(df['sentiment'], fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Compares the emotional sentiment of ongoing conversational replies vs the final 'Terminal' reply. Shows whether arguments typically end angrily or positively.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'resolution_patterns.png')
    plt.close()

def plot_initiator_patterns(out_dir, role_file):
    print("🗣️ Generating Initiator/Responder Chart...")
    df = pd.read_parquet(role_file)
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#ef476f', '#118ab2', '#06d6a0']
    
    wedges, texts, autotexts = ax.pie(df['author_count'], labels=df['role'], colors=colors, 
                                      autopct='%1.1f%%', startangle=140, pctdistance=0.85,
                                      wedgeprops=dict(width=0.4, edgecolor='w'))
                                      
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.setp(texts, size=12)
    
    ax.set_title('Initiator/Response Patterns: Conversational Roles', pad=20, fontsize=16)
    
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Classifies users into Broadcasters (only start threads), Responders (only reply), and Conversationalists (do both). Reveals whether the community is driven by dialogue or independent shouts.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'initiator_patterns.png')
    plt.close()

def plot_code_switching(out_dir, cs_file):
    print("🔀 Generating Code-Switching Chart...")
    df = pd.read_parquet(cs_file)
    if df.empty: return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    colors = ['#118ab2', '#ef476f']
    
    # Subplot 1: Prevalence
    ax1.pie(df['comment_count'], labels=df['type'], colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Prevalence of Code-Switching')
    
    # Subplot 2: Engagement Impact
    ax2.bar(df['type'], df['avg_likes'], color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Average Likes')
    ax2.set_title('Engagement Impact: Does Code-Switching work?')
    
    plt.suptitle('Code-Switching Detection (Cyrillic + Latin Mixing)', fontsize=16)
    plt.tight_layout()
    plt.figtext(0.5, 0.015, "Explanation: Detects users who mix Russian and English in the same sentence (e.g. gamer slang). Shows prevalence and whether multilingual flexing increases engagement.", ha="center", fontsize=11, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'code_switching_impact.png')
    plt.close()

def plot_cross_video(out_dir, radar_file, controversy_file):
    print("⚔️ Generating Comparative Cross-Video Charts...")
    
    if radar_file.exists():
        df_radar = pd.read_parquet(radar_file)
        if not df_radar.empty:
            labels = ['Positivity', 'Negativity', 'Volume', 'Branching', 'Avg Length']
            num_vars = len(labels)
            
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            for idx, row in df_radar.iterrows():
                values = [row['positivity_norm'], row['negativity_norm'], row['volume_norm'], row['branching_factor_norm'], row['avg_length_norm']]
                values += values[:1]
                
                ax.plot(angles, values, linewidth=2, label=f"Video: {row['video_id'][:8]}")
                ax.fill(angles, values, alpha=0.1)
                
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), labels)
            
            plt.title('Video Profile Radar: Top 5 Videos', pad=20, fontsize=16)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            
            plt.figtext(0.5, 0.015, "Explanation: Plots the unique personality footprint of the top videos. Some are highly positive & branching, others are toxic & brief.", ha="center", fontsize=11, color="dimgray", wrap=True)
            plt.savefig(out_dir / 'video_profile_radar.png')
            plt.close()
            
    if controversy_file.exists():
        df_controv = pd.read_parquet(controversy_file)
        if not df_controv.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            x = np.arange(2)
            width = 0.25
            
            ax.bar(x - width, df_controv['POSITIVE'] * 100, width, label='Positive', color='#06d6a0')
            ax.bar(x, df_controv['NEUTRAL'] * 100, width, label='Neutral', color='#ffd166')
            ax.bar(x + width, df_controv['NEGATIVE'] * 100, width, label='Negative', color='#ef476f')
            
            ax.set_ylabel('Percentage of Comments (%)', fontsize=12)
            ax.set_title('Before/After Controversy Analysis', pad=20, fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(df_controv['period'], fontsize=12)
            ax.legend()
            
            plt.tight_layout()
            plt.figtext(0.5, 0.015, "Explanation: Auto-detects the most negative video and compares channel-wide sentiment 14 days before vs 14 days after its release.", ha="center", fontsize=11, color="dimgray", wrap=True)
            plt.savefig(out_dir / 'controversy_impact.png')
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

def plot_audience_intent(out_dir, intent_file):
    print("🎯 Generating Audience Demand & Intent Distribution...")
    df = pd.read_parquet(intent_file)
    if df.empty: return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    palette = {'CONTENT_IDEA': '#06d6a0', 'QUESTION_CONFUSION': '#118ab2', 'CRITIQUE_FEEDBACK': '#ffd166', 'APPRECIATION': '#ef476f', 'DEBATE_OPINION': '#073b4c'}
    colors = [palette.get(c, '#888888') for c in df['audience_intent']]
    axes[0].pie(df['comment_count'], labels=df['audience_intent'], autopct='%1.1f%%', colors=colors, startangle=140)
    axes[0].set_title('Audience Conversational Intent Share', pad=15, fontsize=13)
    
    sns.barplot(data=df, x='avg_likes', y='audience_intent', hue='audience_intent', palette=palette, legend=False, ax=axes[1])
    axes[1].set_title('Average Engagement (Likes) by Intent', pad=15, fontsize=13)
    axes[1].set_xlabel('Average Likes per Comment')
    axes[1].set_ylabel('')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Categorizes audience comments into content ideas, questions, critiques, and appreciation. Highlights which types receive the highest community upvotes.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'audience_intent_distribution.png')
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

def run_visualizations():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # File Paths
    authors_file = out_dir / "authors_final.parquet"
    semantics_file = interim_dir / "semantic_topics.parquet"
    survival_file = out_dir / "kaplan_meier_survival.parquet"
    shap_vals_file = out_dir / "shap_values.npy"
    shap_feat_file = out_dir / "shap_features.parquet"
    comments_file = interim_dir / "comments_clean.parquet"
    diurnal_file = out_dir / "diurnal_heatmap.parquet"
    stl_file = out_dir / "stl_decomposition.parquet"
    
    # Phase 3 File Paths
    reaction_file = out_dir / "video_reaction_map.parquet"
    integrity_file = interim_dir / "integrity_flags.parquet"
    slopes_file = out_dir / "thread_decay_slopes.parquet"
    topic_evo_file = out_dir / "topic_evolution.parquet"
    latency_file = out_dir / "reply_latency.parquet"
    shelf_life_file = out_dir / "shelf_life_likes.parquet"
    matrix_file = out_dir / "topic_video_matrix.parquet"
    coocc_edges_file = out_dir / "word_cooccurrence_edges.parquet"
    coocc_nodes_file = out_dir / "word_cooccurrence_nodes.parquet"
    ner_file = out_dir / "named_entities.parquet"
    tag_edges_file = out_dir / "tag_network_edges.parquet"
    tag_nodes_file = out_dir / "tag_network_nodes.parquet"
    bipartite_edges_file = out_dir / "bipartite_edges.parquet"
    bipartite_nodes_file = out_dir / "bipartite_nodes.parquet"
    cocomment_edges_file = out_dir / "cocomment_edges.parquet"
    cocomment_nodes_file = out_dir / "cocomment_nodes.parquet"
    cohort_retention_file = out_dir / "cohort_retention.parquet"
    cohort_retention_pct_file = out_dir / "cohort_retention_pct.parquet"
    new_vs_returning_file = out_dir / "new_vs_returning.parquet"
    video_overlap_file = out_dir / "video_overlap_matrix.parquet"
    topic_cohorts_file = out_dir / "topic_cohorts.parquet"
    frequency_tiers_file = out_dir / "frequency_tiers.parquet"
    bot_file = out_dir / "bot_classifications.parquet"
    driveby_loyalists_file = out_dir / "driveby_loyalists.parquet"
    position_bias_file = out_dir / "position_bias.parquet"
    arrival_speed_file = out_dir / "arrival_speed.parquet"
    thread_width_file = out_dir / "thread_width_dist.parquet"
    resolution_file = out_dir / "resolution_patterns.parquet"
    initiator_file = out_dir / "initiator_patterns.parquet"
    code_switching_file = out_dir / "code_switching_impact.parquet"
    video_radar_file = out_dir / "video_profile_radar.parquet"
    controversy_file = out_dir / "controversy_impact.parquet"
    inflation_file = out_dir / "like_inflation.parquet"
    fingerprint_file = out_dir / "author_fingerprints.parquet"
    impersonation_file = out_dir / "impersonation_detection.parquet"
    intent_summary_file = out_dir / "audience_intent_summary.parquet"
    cib_rings_file = out_dir / "cib_rings.parquet"
    
    print(f"🎨 Initializing Visualization Engine (Outputting to {plots_dir})...")
    
    # === Original Charts ===
    
    # 1. Temporal Survival Plot
    if survival_file.exists():
        df_surv = pd.read_parquet(survival_file)
        plot_kaplan_meier(df_surv, plots_dir)
        
    # 2. Semantic Cluster Scatter/Hexbin
    if semantics_file.exists():
        df_sem = pd.read_parquet(semantics_file)
        plot_umap_semantics(df_sem, plots_dir)
        
    # 3. Author Power-Law Pareto
    if authors_file.exists():
        df_auth = pd.read_parquet(authors_file)
        plot_author_pareto(df_auth, plots_dir)
        
    # 4. XGBoost Feature Importance
    if shap_vals_file.exists() and shap_feat_file.exists():
        shap_vals = np.load(shap_vals_file)
        shap_feat = pd.read_parquet(shap_feat_file)
        plot_shap_summary(shap_vals, shap_feat, plots_dir)
    
    # === NEW Phase 2 Charts ===
    
    # Load comments once for all comment-based visualizations
    df_comments = None
    if comments_file.exists():
        df_comments = pd.read_parquet(comments_file)
    
    # 5. Diurnal Activity Heatmap (24×7)
    if diurnal_file.exists():
        plot_diurnal_heatmap(plots_dir, diurnal_file)
    
    # 6. Plutchik Emotion Radar
    if df_comments is not None:
        plot_plutchik_radar(df_comments, plots_dir)
    
    # 7. Sentiment Ridge Plot
    if df_comments is not None and 'vader_compound' in df_comments.columns:
        plot_sentiment_ridges(df_comments, plots_dir)
    
    # 8. Force-Directed Network
    plot_force_directed_network(interim_dir, plots_dir)
    
    # 9. RFM 3D Scatter
    if authors_file.exists():
        df_auth = pd.read_parquet(authors_file)
        plot_rfm_3d(df_auth, plots_dir)
    
    # 10. Sentiment Divergence
    if df_comments is not None:
        plot_sentiment_divergence(df_comments, plots_dir)
    
    # 11. Toxicity Heatmap
    if df_comments is not None:
        plot_toxicity_heatmap(df_comments, plots_dir)
    
    # 12. Reply Depth Distribution
    if df_comments is not None:
        plot_reply_depth_distribution(df_comments, plots_dir)
    
    # 13. Lorenz Curve
    if authors_file.exists():
        df_auth = pd.read_parquet(authors_file)
        plot_lorenz_curve(df_auth, plots_dir)
    
    # 14. STL Decomposition
    if stl_file.exists():
        plot_stl_decomposition(plots_dir, stl_file)
        
    # === NEW Phase 3 Charts (s07-s15) ===
    if reaction_file.exists():
        plot_reaction_timeline(plots_dir, reaction_file)
    if integrity_file.exists() and comments_file.exists():
        plot_anomaly_scatter(plots_dir, integrity_file, comments_file)
    if slopes_file.exists():
        plot_thread_polarization(plots_dir, slopes_file)
    if topic_evo_file.exists():
        plot_topic_streamgraph(plots_dir, topic_evo_file)
    if latency_file.exists():
        plot_reply_latency_distribution(plots_dir, latency_file)
    if shelf_life_file.exists():
        plot_shelf_life(plots_dir, shelf_life_file)
    if matrix_file.exists():
        plot_topic_video_matrix(plots_dir, matrix_file)
    if coocc_edges_file.exists() and coocc_nodes_file.exists():
        plot_cooccurrence_network(plots_dir, coocc_edges_file, coocc_nodes_file)
    if ner_file.exists():
        plot_ner_distribution(plots_dir, ner_file)
    if comments_file.exists():
        plot_sarcasm_distribution(plots_dir, comments_file)
        plot_linguistic_features(plots_dir, comments_file)
    if tag_edges_file.exists() and tag_nodes_file.exists():
        plot_tag_network(plots_dir, tag_edges_file, tag_nodes_file)
    if bipartite_edges_file.exists() and bipartite_nodes_file.exists():
        plot_bipartite_network(plots_dir, bipartite_edges_file, bipartite_nodes_file)
    if cocomment_edges_file.exists() and cocomment_nodes_file.exists():
        plot_cocommenting_network(plots_dir, cocomment_edges_file, cocomment_nodes_file)
    if cohort_retention_file.exists() and cohort_retention_pct_file.exists():
        plot_cohort_retention(plots_dir, cohort_retention_pct_file, cohort_retention_file)
    if new_vs_returning_file.exists():
        plot_new_vs_returning(plots_dir, new_vs_returning_file)
    if video_overlap_file.exists():
        plot_video_overlap(plots_dir, video_overlap_file)
    if topic_cohorts_file.exists():
        plot_topic_cohorts(plots_dir, topic_cohorts_file)
    if frequency_tiers_file.exists():
        plot_frequency_tiers(plots_dir, frequency_tiers_file)
    if bot_file.exists():
        plot_bot_heuristics(plots_dir, bot_file)
    if driveby_loyalists_file.exists():
        plot_driveby_loyalists(plots_dir, driveby_loyalists_file)
    if position_bias_file.exists():
        plot_position_bias(plots_dir, position_bias_file)
    if arrival_speed_file.exists():
        plot_arrival_speed(plots_dir, arrival_speed_file)
    if thread_width_file.exists():
        plot_thread_width(plots_dir, thread_width_file)
    if resolution_file.exists():
        plot_resolution_patterns(plots_dir, resolution_file)
    if initiator_file.exists():
        plot_initiator_patterns(plots_dir, initiator_file)
    if code_switching_file.exists():
        plot_code_switching(plots_dir, code_switching_file)
    if video_radar_file.exists() or controversy_file.exists():
        plot_cross_video(plots_dir, video_radar_file, controversy_file)
    if inflation_file.exists():
        plot_like_inflation(plots_dir, inflation_file)
    if fingerprint_file.exists():
        plot_author_fingerprints(plots_dir, fingerprint_file)
    if impersonation_file.exists():
        plot_impersonation(plots_dir, impersonation_file)
    if intent_summary_file.exists():
        plot_audience_intent(plots_dir, intent_summary_file)
    if cib_rings_file.exists():
        plot_cib_rings(plots_dir, cib_rings_file)

        
    print("✅ Stage 06 Visualizations Complete! 🎆")

if __name__ == "__main__":
    run_visualizations()
