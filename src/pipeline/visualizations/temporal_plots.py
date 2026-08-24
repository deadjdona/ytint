"""Phase 4: Video Dynamics, Cohort Retention, Longitudinal Heatmaps, and Radar Profiles."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .theme import setup_theme


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
    plt.savefig(out_dir / 'reaction_timeline.png', dpi=150)
    plt.savefig(out_dir / 'temporal_clustering.png', dpi=150)
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
    plt.savefig(out_dir / 'integrity_scatter.png', dpi=150)
    plt.close()

def plot_velocity_spikes(out_dir, timeline_file, viral_file):
    print("📈 Generating Viral Velocity Spikes Chart...")
    df_time = pd.read_parquet(timeline_file)
    df_viral = pd.read_parquet(viral_file)
    if df_time.empty or df_viral.empty:
        return
        
    df_time['date'] = pd.to_datetime(df_time['date'])
    df_time = df_time.sort_values('date')
    df_viral['date'] = pd.to_datetime(df_viral['date'])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df_time['date'], df_time['comment_count'], color='#1f77b4', alpha=0.6, linewidth=1.2, label='Daily Comments')
    rolling_avg = df_time['comment_count'].rolling(window=7, min_periods=1, center=True).mean()
    ax.plot(df_time['date'], rolling_avg, color='#0d3b66', linewidth=2, label='7-Day Rolling Trend')

    ax.scatter(df_viral['date'], df_viral['comment_count'], color='#e63946', s=df_viral['z_score'].clip(lower=20, upper=250), alpha=0.9, edgecolor='black', linewidth=0.8, zorder=5, label=f'Viral Velocity Spikes (Z > 3, n={len(df_viral)})')

    top_events = df_viral.nlargest(5, 'comment_count')
    for _, row in top_events.iterrows():
        d_str = row['date'].strftime('%Y-%m-%d')
        cnt = int(row['comment_count'])
        z = float(row['z_score'])
        ax.annotate(
            f"{d_str}\n{cnt:,} comments (Z={z:.1f})",
            xy=(row['date'], row['comment_count']),
            xytext=(0, 20),
            textcoords='offset points',
            ha='center',
            fontsize=8.5,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeb3b', alpha=0.85, edgecolor='gray'),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=0.8)
        )

    ax.set_title('Viral Velocity Spikes & Volume Surge Anomalies Over Time', pad=15, fontsize=14)
    ax.set_xlabel('Timeline (Date)', fontsize=11)
    ax.set_ylabel('Daily Comment Count', fontsize=11)
    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Identifies statistically significant daily comment volume surges (Z > 3.0) over baseline activity, highlighting viral breakouts, external algorithmic pushes, or breaking news events.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'velocity_spikes.png', dpi=150)
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

def plot_video_half_life(out_dir, videos_file):
    print("⏳ Generating Video Attention Half-Life Distribution...")
    df = pd.read_parquet(videos_file)
    if df.empty or 'attention_half_life_days' not in df.columns:
        return
        
    valid_hl = df['attention_half_life_days'].dropna()
    valid_hl = valid_hl[valid_hl > 0]
    if valid_hl.empty:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Distribution
    sns.histplot(valid_hl, bins=30, kde=True, color='royalblue', ax=axes[0])
    med_hl = valid_hl.median()
    mean_hl = valid_hl.mean()
    axes[0].axvline(med_hl, color='crimson', linestyle='--', linewidth=1.5, label=f'Median Half-Life: {med_hl:.2f} days')
    axes[0].axvline(mean_hl, color='darkorange', linestyle=':', linewidth=1.5, label=f'Mean Half-Life: {mean_hl:.2f} days')
    axes[0].set_title('Video Attention Half-Life Distribution', pad=15, fontsize=13)
    axes[0].set_xlabel('Attention Half-Life (Days to 50% Inactivity)')
    axes[0].set_ylabel('Video Upload Count')
    axes[0].legend(loc='upper right')

    # Right: Top 10 Longest Half-Life Videos (Evergreen assets)
    top_evergreen = df.nlargest(10, 'attention_half_life_days')
    sns.barplot(data=top_evergreen, x='attention_half_life_days', y='video_id', palette='Blues_r', hue='video_id', legend=False, ax=axes[1])
    axes[1].set_title('Top 10 Evergreen Videos by Attention Lifespan', pad=15, fontsize=13)
    axes[1].set_xlabel('Half-Life (Days)')
    axes[1].set_ylabel('Video ID')

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    plt.figtext(0.5, 0.015, 'Explanation: Exponential decay parameter modeling the duration (in days) before comment velocity drops by 50%. Separates fast-fading flash topics from evergreen catalog assets.', ha='center', fontsize=10, color='dimgray', wrap=True)
    plt.savefig(out_dir / 'video_half_life.png', dpi=150)
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
        g.fig.subplots_adjust(top=0.93, bottom=0.06)
        plt.figtext(0.5, 0.015, "Explanation: Clustered heatmap showing which videos focus on which topics. Darker cells mean the topic dominates that video's discussion.", ha="center", fontsize=10, color="dimgray", wrap=True)
        g.savefig(out_dir / 'topic_video_matrix.png')
        plt.close(g.fig)
    except Exception as e:
        print(f"⚠️ Could not generate clustered matrix: {e}")

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


def plot_topic_injection_anomalies(df_inj, out_dir):
    """
    Scatter/timeline plot of detected topic injection anomalies by JS divergence score.
    """
    print("💉 Generating Topic Injection Anomalies Plot...")
    if df_inj is None or df_inj.empty or 'js_divergence' not in df_inj.columns:
        return
    setup_theme()
    fig, ax = plt.subplots(figsize=(12, 6))
    df = df_inj.copy()
    df['window_start'] = pd.to_datetime(df['window_start'])

    size = (df['dominant_topic_share'] * 200) if 'dominant_topic_share' in df.columns else 100
    scatter = ax.scatter(df['window_start'], df['js_divergence'],
                         s=size,
                         c=df['js_divergence'], cmap='Reds', alpha=0.8, edgecolors='black')
    ax.axhline(0.40, color='red', linestyle='--', label='Anomaly Threshold (0.40)')
    ax.set_title('Topic Injection & Thematic Shift Anomalies (Jensen-Shannon Divergence)', fontsize=14, pad=15)
    ax.set_ylabel('JS Divergence vs Corpus Baseline')
    ax.set_xlabel('Anomaly Window Start Time')
    plt.colorbar(scatter, ax=ax, label='Divergence Severity')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_dir / 'topic_injection_anomalies.png', dpi=150)
    plt.close()


def plot_minute_arrival_curve(df_curve, out_dir):
    """
    Line plot of minute-level comment accumulation velocity in the first 120 minutes post-upload.
    """
    print("⏱️ Generating Minute-Level Arrival Velocity Curve...")
    if df_curve is None or df_curve.empty or 'minute_bin' not in df_curve.columns:
        return
    setup_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(df_curve['minute_bin'], df_curve['cumulative_comments'], color='#3498db', linewidth=2.5)
    ax1.set_title('Cumulative Comments (First 120 Minutes)', fontsize=13)
    ax1.set_xlabel('Minutes Since Video Upload')
    ax1.set_ylabel('Cumulative Comment Count')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(df_curve['minute_bin'], df_curve['velocity_per_min'], color='#e67e22', linewidth=2)
    ax2.set_title('Comment Velocity (Comments / Minute)', fontsize=13)
    ax2.set_xlabel('Minutes Since Video Upload')
    ax2.set_ylabel('Arrival Velocity (5-min rolling mean)')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / 'minute_arrival_speed_curve.png', dpi=150)
    plt.close()

