"""Phase 1: Semantics, BERTopic, UMAP, NER, Co-occurrence, and Stance Visualizations."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from .theme import setup_theme


def plot_umap_semantics(df_semantics, out_dir):
    """
    Task 2: 2D UMAP Semantic Clusters Space (with Hexbin fallback for >10k points)
    Generates a 2D UMAP projection of topics and intents.
    """
    print("🌌 Generating UMAP Semantic Clusters...")
    
    req_cols = [c for c in ['umap_x', 'umap_y'] if c in df_semantics.columns]
    if len(req_cols) < 2:
        return
        
    color_col = None
    for candidate in ['intent_label', 'topic_name', 'topic', 'sentiment']:
        if candidate in df_semantics.columns:
            color_col = candidate
            break
            
    subset_cols = ['umap_x', 'umap_y']
    if color_col:
        subset_cols.append(color_col)
        
    df_plot = df_semantics.dropna(subset=subset_cols).copy()
    if df_plot.empty:
        return
        
    plt.figure(figsize=(12, 8))
    
    if len(df_plot) > 10000 and not color_col:
        print("⚠️ Large dataset detected. Using Hexbin aggregation to prevent visual overplotting & crashes...")
        hb = plt.hexbin(df_plot['umap_x'], df_plot['umap_y'], gridsize=50, cmap='viridis', mincnt=1)
        cb = plt.colorbar(hb, label='Comment Density')
        plt.title(f"UMAP Semantic Density Space (N={len(df_plot):,})", pad=15)
    elif color_col:
        sample_df = df_plot.sample(min(len(df_plot), 5000), random_state=42) if len(df_plot) > 5000 else df_plot
        sns.scatterplot(
            data=sample_df, 
            x='umap_x', 
            y='umap_y', 
            hue=color_col, 
            alpha=0.65, 
            s=25,
            linewidth=0,
            palette='tab10' if sample_df[color_col].nunique() <= 10 else 'viridis'
        )
        title_suffix = color_col.replace('_', ' ').title()
        plt.title(f"UMAP Semantic Clusters by {title_suffix} (Sample N={len(sample_df):,})", pad=15, fontsize=13)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    else:
        sns.scatterplot(
            data=df_plot, 
            x='umap_x', 
            y='umap_y', 
            alpha=0.6, 
            s=20,
            linewidth=0
        )
        plt.title(f"UMAP Semantic Clusters (N={len(df_plot):,})", pad=15)
        
    plt.xlabel("UMAP Dimension 1", fontsize=11)
    plt.ylabel("UMAP Dimension 2", fontsize=11)
    
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    plt.figtext(0.5, 0.015, "Explanation: 2D projection of comment meanings. Proximity indicates semantic similarity; colors denote inferred topic or user intent.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / "umap_semantics.png", dpi=150)
    plt.close()

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
    
    def _clean_labels(series):
        return [str(s).encode('ascii', 'replace').decode('ascii').replace('?', '').strip() or str(s) for s in series]

    if not top_per.empty:
        per_labels = _clean_labels(top_per.index)
        sns.barplot(x=top_per.values, y=per_labels, hue=per_labels, palette='Blues_r', legend=False, ax=axes[0])
        axes[0].set_title('Top Mentioned Persons (PER)')
        axes[0].set_xlabel('Mentions')
        
    if not top_org.empty:
        org_labels = _clean_labels(top_org.index)
        sns.barplot(x=top_org.values, y=org_labels, hue=org_labels, palette='Greens_r', legend=False, ax=axes[1])
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

def plot_stance_drift(out_dir, stance_file, drift_file):
    print("🎯 Generating Stance Detection & Polarization Drift Charts...")
    df_stance = pd.read_parquet(stance_file)
    df_drift = pd.read_parquet(drift_file)
    if df_stance.empty or df_drift.empty: return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    top_vids = df_stance.head(8).copy()
    stance_melt = top_vids.melt(id_vars=['video_id'], value_vars=['favor_pct', 'against_pct', 'neutral_pct'], var_name='Stance', value_name='Percentage')
    stance_melt['Stance'] = stance_melt['Stance'].map({'favor_pct': 'Favor / Agreement', 'against_pct': 'Against / Opposition', 'neutral_pct': 'Neutral / Objective'})
    
    sns.barplot(data=stance_melt, x='video_id', y='Percentage', hue='Stance', palette=['#00cc66', '#ff3366', '#888888'], ax=axes[0])
    axes[0].set_title('Target Stance Breakdown (Favor vs. Against vs. Neutral)', pad=15, fontsize=13)
    axes[0].set_xlabel('Video Identifier')
    axes[0].set_ylabel('Audience Stance Share (%)')
    axes[0].tick_params(axis='x', rotation=30)
    
    ax2 = axes[1]
    sns.lineplot(data=df_drift, x='depth_label', y='against_pct', marker='o', linewidth=2.5, color='#ff3366', label='Opposition (%)', ax=ax2)
    sns.lineplot(data=df_drift, x='depth_label', y='favor_pct', marker='s', linewidth=2.5, color='#00cc66', label='Favor (%)', ax=ax2)
    
    ax2.set_title('Polarization Drift across Reply Tree Depth', pad=15, fontsize=13)
    ax2.set_xlabel('Conversation Tree Level')
    ax2.set_ylabel('Stance Orientation Share (%)')
    ax2.tick_params(axis='x', rotation=15)
    ax2.legend(loc='upper right')
    
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    plt.figtext(0.5, 0.015, "Explanation: Target-specific stance extraction mapping consensus vs opposition balance, and demonstrating polarization drift across deep debate replies.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'stance_polarization_drift.png')
    plt.close()


def plot_emoji_treemap(out_dir, emoji_file):
    """
    Emoji frequency treemap — top emoji across the corpus.
    Requires squarify: pip install squarify
    """
    print("😊 Generating Emoji Frequency Treemap...")
    try:
        import squarify
    except ImportError:
        print("  ⚠️ squarify not installed. Skipping emoji treemap.")
        return
    if not Path(emoji_file).exists():
        return
    df = pd.read_parquet(emoji_file)
    corpus = df[df['group_type'] == 'sentiment'].groupby('emoji')['count'].sum().reset_index()
    corpus = corpus.nlargest(40, 'count')
    if corpus.empty:
        return
    setup_theme()
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab20.colors
    squarify.plot(sizes=corpus['count'], label=corpus['emoji'],
                  color=colors[:len(corpus)], alpha=0.85, ax=ax,
                  text_kwargs={'fontsize': 14})
    ax.set_title('Emoji Frequency Treemap — Corpus-Wide', fontsize=16, pad=15)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / 'emoji_treemap.png', dpi=150)
    plt.close()


def plot_length_vs_likes_hexbin(df_comments, out_dir):
    """
    Comment length vs like_count hexbin scatter — reveals engagement sweet spots.
    """
    print("📐 Generating Length vs Likes Hexbin...")
    if 'char_count' not in df_comments.columns or 'like_count' not in df_comments.columns:
        return
    df = df_comments[['char_count', 'like_count']].copy()
    df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0)
    df['char_count'] = pd.to_numeric(df['char_count'], errors='coerce').fillna(0)
    df = df[(df['char_count'] > 0) & (df['char_count'] < 2000)]
    if df.empty:
        return
    setup_theme()
    fig, ax = plt.subplots(figsize=(12, 7))
    hb = ax.hexbin(df['char_count'], df['like_count'],
                   gridsize=50, cmap='YlOrRd', mincnt=1,
                   bins='log')
    plt.colorbar(hb, ax=ax, label='Log(Comment Count)')
    ax.set_xlabel('Comment Length (characters)', fontsize=12)
    ax.set_ylabel('Like Count', fontsize=12)
    ax.set_title('Comment Length vs Engagement (Hexbin Density)', fontsize=15, pad=15)
    ax.set_xlim(0, df['char_count'].quantile(0.99))
    ax.set_ylim(0, df['like_count'].quantile(0.99))
    plt.tight_layout()
    plt.figtext(0.5, 0.01, "Explanation: Density map showing where comment length and like-count co-occur most frequently. Warm zones indicate optimal comment length for engagement.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'length_vs_likes_hexbin.png', dpi=150)
    plt.close()


def plot_language_distribution(df_comments, out_dir, top_n=10):
    """
    Stacked/grouped bar of language distribution across the corpus and per top video.
    """
    print("🌍 Generating Language Distribution...")
    if 'language' not in df_comments.columns:
        return
    setup_theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Corpus-wide
    lang_counts = df_comments['language'].fillna('unknown').value_counts().head(top_n)
    axes[0].barh(lang_counts.index[::-1], lang_counts.values[::-1],
                 color=plt.cm.Set2.colors[:len(lang_counts)])
    axes[0].set_title('Corpus-Wide Language Distribution', fontsize=13)
    axes[0].set_xlabel('Number of Comments')

    # Top videos language mix
    if 'video_id' in df_comments.columns:
        top_vids = df_comments['video_id'].value_counts().head(8).index
        df_top = df_comments[df_comments['video_id'].isin(top_vids)]
        top_langs = df_comments['language'].fillna('unknown').value_counts().head(5).index
        df_top = df_top.copy()
        df_top['language'] = df_top['language'].where(df_top['language'].isin(top_langs), 'other')
        pivot = df_top.groupby(['video_id', 'language']).size().unstack(fill_value=0)
        pivot.plot(kind='bar', stacked=True, ax=axes[1],
                   colormap='Set2', width=0.8)
        axes[1].set_title('Language Mix per Top Video', fontsize=13)
        axes[1].set_xlabel('')
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].legend(title='Language', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / 'language_distribution.png', dpi=150)
    plt.close()


def plot_valence_per_topic(df_comments, out_dir, top_n=15):
    """
    Diverging bar chart of mean sentiment score per topic.
    """
    print("↔️ Generating Valence per Topic Diverging Chart...")
    topic_col = next((c for c in ['topic_name', 'topic_id'] if c in df_comments.columns), None)
    sent_col = next((c for c in ['vader_compound', 'sentiment_compound', 'sentiment_score'] if c in df_comments.columns), None)
    if topic_col is None or sent_col is None:
        return
    df = df_comments[[topic_col, sent_col]].dropna()
    df[sent_col] = pd.to_numeric(df[sent_col], errors='coerce').fillna(0)
    topic_sentiment = df.groupby(topic_col)[sent_col].mean().sort_values()
    topic_sentiment = pd.concat([topic_sentiment.head(top_n // 2),
                                 topic_sentiment.tail(top_n // 2)])
    setup_theme()
    fig, ax = plt.subplots(figsize=(12, max(6, len(topic_sentiment) * 0.4)))
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in topic_sentiment.values]
    ax.barh(topic_sentiment.index, topic_sentiment.values, color=colors, edgecolor='white')
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title('Mean Sentiment per Topic (Diverging)', fontsize=14, pad=15)
    ax.set_xlabel('Mean VADER Compound Score')
    ax.set_ylabel('Topic')
    plt.tight_layout()
    plt.figtext(0.5, 0.01, "Explanation: Topics to the right are discussed positively on average; topics to the left attract negative commentary.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'valence_per_topic.png', dpi=150)
    plt.close()


def plot_slang_lexicon(df_slang, out_dir, top_n=15):
    """
    Bar chart of top internet slang / informal register occurrences and their average sentiment.
    """
    print("💬 Generating Slang & Internet-Register Lexicon Plot...")
    if df_slang is None or df_slang.empty or 'slang_term' not in df_slang.columns:
        return
    setup_theme()
    top_df = df_slang.sort_values('total_occurrences', ascending=False).head(top_n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.barh(top_df['slang_term'][::-1], top_df['total_occurrences'][::-1], color='#3498db', edgecolor='white')
    ax1.set_title('Top Slang & Internet-Register Terms (Occurrences)', fontsize=13)
    ax1.set_xlabel('Total Occurrences')

    colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in top_df['avg_sentiment'][::-1]]
    ax2.barh(top_df['slang_term'][::-1], top_df['avg_sentiment'][::-1], color=colors, edgecolor='white')
    ax2.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax2.set_title('Average Sentiment Association', fontsize=13)
    ax2.set_xlabel('Mean Sentiment Compound')

    plt.tight_layout()
    plt.savefig(out_dir / 'slang_lexicon_distribution.png', dpi=150)
    plt.close()


def plot_tfidf_keywords(df_tfidf, out_dir, top_n=20):
    """
    Horizontal bar chart of top TF-IDF keywords extracted per video.
    """
    print("🔑 Generating Top TF-IDF Keywords Plot...")
    if df_tfidf is None or df_tfidf.empty or 'keyword' not in df_tfidf.columns:
        return
    setup_theme()
    top_kw = df_tfidf.groupby('keyword')['tfidf_score'].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top_kw.index[::-1], top_kw.values[::-1], color='#9b59b6', edgecolor='white')
    ax.set_title('Top Corpus-Wide TF-IDF Salient Keywords', fontsize=14, pad=15)
    ax.set_xlabel('Mean TF-IDF Salience Score')
    plt.tight_layout()
    plt.savefig(out_dir / 'tfidf_keywords_salience.png', dpi=150)
    plt.close()
