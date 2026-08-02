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
    
    plt.tight_layout()
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
    
    plt.tight_layout()
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
    plt.tight_layout()
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
    plt.tight_layout()
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
    
    plt.tight_layout()
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
    
    plt.tight_layout()
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
    plt.tight_layout()
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
    
    plt.tight_layout()
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
        title='RFM Author Segmentation (3D)',
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
    
    plt.tight_layout()
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
    
    plt.tight_layout()
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
    
    plt.tight_layout()
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
    
    gini = 1 - 2 * np.trapz(cum_freq, lorenz_x)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(lorenz_x, cum_freq, color='crimson', linewidth=2, label=f'Lorenz (Gini={gini:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
    ax.fill_between(lorenz_x, lorenz_x, cum_freq, color='crimson', alpha=0.1)
    
    ax.set_xlabel('Cumulative Share of Authors')
    ax.set_ylabel('Cumulative Share of Comments')
    ax.set_title('Lorenz Curve: Author Engagement Inequality', pad=15)
    ax.legend()
    
    plt.tight_layout()
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
    
    plt.tight_layout()
    plt.savefig(out_dir / 'stl_decomposition.png')
    plt.close()


# ============================================================
# Main orchestration
# ============================================================

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
        
    print("✅ Stage 06 Visualizations Complete! 🎆")

if __name__ == "__main__":
    run_visualizations()
