"""Phase 2: Conversation Tree, Thread Dynamics, Latency, and Toxicity Contagion Visualizations."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .theme import setup_theme


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

def plot_thread_polarization(out_dir, slopes_file):
    print("🌪️ Generating Thread Polarization Decay Slopes...")
    df = pd.read_parquet(slopes_file)
    if df.empty: return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['decay_slope'], bins=35, kde=True, color='darkorange', ax=ax)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Neutral Equilibrium (Slope = 0)')
    ax.set_title('Thread Sentiment Decay Slopes (Conversational Polarization)', pad=15, fontsize=14)
    ax.set_xlabel('Sentiment Trajectory Slope (Negative = Decays into Toxicity, Positive = Deepening Praise)')
    ax.set_ylabel('Thread Count')
    ax.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Distribution of sentiment trajectory slopes across conversation threads. Negative slopes indicate discussions that devolve into hostility over consecutive replies.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'thread_decay_slopes.png', dpi=150)
    plt.savefig(out_dir / 'thread_polarization.png', dpi=150)
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
    plt.savefig(out_dir / 'reply_latency.png', dpi=150)
    plt.savefig(out_dir / 'reply_latency_distribution.png', dpi=150)
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
    plt.savefig(out_dir / 'thread_width.png', dpi=150)
    plt.savefig(out_dir / 'thread_width_dist.png', dpi=150)
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

def plot_toxicity_contagion(out_dir, catalysts_file):
    print("🔥 Generating Toxicity Contagion & Troll Catalyst Charts...")
    df_cat = pd.read_parquet(catalysts_file)
    if df_cat.empty: return
    
    top_trolls = df_cat.head(10).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    
    tier_series = df_cat['catalyst_tier'].str.replace(r'^[^\w\s]+\s*', '', regex=True)
    tier_counts = tier_series.value_counts()
    colors = ['#ff4d4d', '#ff9933', '#ffcc00', '#66cc66']
    axes[0].pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%', colors=colors[:len(tier_counts)], startangle=180, pctdistance=0.75)
    axes[0].set_title('Community Provocation & Catalyst Spectrum', pad=20, fontsize=13)
    
    sns.barplot(data=top_trolls, x='catalyst_score', y='display_name', palette='Reds_r', hue='display_name', legend=False, ax=axes[1])
    axes[1].set_title('Top Troll Catalysts by Flame-War Spark Score', pad=20, fontsize=13)
    axes[1].set_xlabel('Catalyst Impact Score (Toxicity × Replies Sparked)', fontsize=11)
    axes[1].set_ylabel('Author Display Name', fontsize=11)
    
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    plt.figtext(0.5, 0.015, "Explanation: Identifies flame-war instigators who post provocative comments that disproportionately spawn hostile replies.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'toxicity_contagion.png')
    plt.close()


def plot_thread_sunburst(out_dir, comments_file):
    """
    Thread depth sunburst using Plotly — shows hierarchical comment tree structure.
    Requires plotly.
    """
    print("☀️ Generating Thread Depth Sunburst...")
    try:
        import plotly.express as px
    except ImportError:
        print("  ⚠️ plotly not installed. Skipping thread sunburst.")
        return
    if not Path(comments_file).exists():
        return
    import pyarrow.parquet as pq
    available = pq.read_schema(comments_file).names
    df = pd.read_parquet(comments_file, columns=[
        c for c in ['video_id', 'comment_depth', 'comment_id', 'parent_id']
        if c in available
    ])
    if 'comment_depth' not in df.columns:
        # Infer depth from parent_id presence
        df['comment_depth'] = df['parent_id'].apply(
            lambda x: 1 if (pd.notna(x) and x != '') else 0
        )
    depth_dist = df.groupby(['video_id', 'comment_depth']).size().reset_index(name='count')
    # Take top 10 most active videos
    top_vids = df['video_id'].value_counts().head(10).index
    depth_dist = depth_dist[depth_dist['video_id'].isin(top_vids)]
    depth_dist['depth_label'] = 'Depth ' + depth_dist['comment_depth'].astype(str)
    if depth_dist.empty:
        return
    fig = px.sunburst(
        depth_dist,
        path=['video_id', 'depth_label'],
        values='count',
        title='Thread Depth Sunburst — Top 10 Videos',
        color='comment_depth',
        color_continuous_scale='RdBu',
    )
    fig.update_layout(margin=dict(t=60, l=0, r=0, b=0))
    fig.write_html(str(out_dir / 'thread_sunburst.html'))
    print("  -> Saved thread_sunburst.html")


def plot_corpus_quality(out_dir, quality_file):
    """
    Log-log scatter of views vs comment count per video — corpus quality / scaling check.
    """
    print("📊 Generating Corpus Quality Log-Log Scatter...")
    if not Path(quality_file).exists():
        return
    df = pd.read_parquet(quality_file)
    if 'log_views' not in df.columns or 'log_comments' not in df.columns:
        return
    df = df.dropna(subset=['log_views', 'log_comments'])
    df = df[(df['log_views'] > 0) & (df['log_comments'] > 0)]
    if df.empty:
        return
    import matplotlib.pyplot as plt
    from .theme import setup_theme
    setup_theme()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df['log_views'], df['log_comments'], alpha=0.6,
               color='steelblue', edgecolors='white', linewidth=0.5, s=50)
    # Fit a trend line
    import numpy as np
    z = np.polyfit(df['log_views'], df['log_comments'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['log_views'].min(), df['log_views'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=1.5, label=f'Trend (slope={z[0]:.2f})')
    ax.set_xlabel('log(1 + Views)', fontsize=12)
    ax.set_ylabel('log(1 + Sampled Comments)', fontsize=12)
    ax.set_title('Video Views vs Comment Volume (Log-Log Scale)', fontsize=14, pad=15)
    ax.legend()
    plt.tight_layout()
    plt.figtext(0.5, 0.01, "Explanation: Each dot is a video. Videos above the trend line over-perform in comments relative to views; those below under-perform or have limited sampling.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'corpus_quality_scatter.png', dpi=150)
    plt.close()
