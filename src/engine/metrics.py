import numpy as np
import pandas as pd

def calculate_aging_scores(df_comments):
    """
    Calculates an 'Aging Score' for every video.
    Measures the divergence between immediate reactions (Layer A) 
    and long-term retrospective sentiment (Layer C).
    
    Positive Score: 'Cult Classic' (Community warms up to it over time)
    Negative Score: 'Aged Poorly' (Community grows cynical over time)
    """
    if 'days_since_upload' not in df_comments.columns or 'sentiment_label' not in df_comments.columns:
        return pd.DataFrame()

    # Map sentiment to a numerical continuous scale for scalar math
    sentiment_weights = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    df_comments['sentiment_score'] = df_comments['sentiment_label'].map(sentiment_weights)

    # Segment comments into Archaeological Strata
    # Layer A: Primary Deposit (0-7 days)
    # Layer C: Retrospective Strata (365+ days)
    layer_a = df_comments[df_comments['days_since_upload'] <= 7]
    layer_c = df_comments[df_comments['days_since_upload'] >= 365]

    if layer_a.empty or layer_c.empty:
        return pd.DataFrame(columns=['video_id', 'aging_score', 'primary_sentiment', 'retrospective_sentiment'])

    # Aggregate average sentiment scores per video for both strata
    mean_a = layer_a.groupby('video_id')['sentiment_score'].mean().rename('primary_sentiment')
    mean_c = layer_c.groupby('video_id')['sentiment_score'].mean().rename('retrospective_sentiment')

    # Join strata metrics
    metrics_df = pd.concat([mean_a, mean_c], axis=1).dropna()
    
    # Aging Score is the direct delta: Long-term sentiment minus short-term sentiment
    metrics_df['aging_score'] = metrics_df['retrospective_sentiment'] - metrics_df['primary_sentiment']
    
    return metrics_df.reset_index()

def calculate_controversy_index(df_comments):
    """
    Calculates a 'Controversy Index' per video.
    Fuses emotional negativity variance with structural thread interaction depth.
    
    High Score: Deep nested argument threads loaded with negative sentiment.
    Low Score: Uncontroversial standalone interactions or peaceful praise.
    """
    if 'parent_id' not in df_comments.columns or 'sentiment_label' not in df_comments.columns:
        return pd.DataFrame()

    # 1. Thread Depth Factor: Count replies per top-level comment parent node
    reply_counts = (
        df_comments[df_comments['parent_id'].notna()]
        .groupby(['video_id', 'parent_id'])
        .size()
        .reset_index(name='reply_depth')
    )
    
    # Capture the max conversational depth spike per video
    max_depth_spike = reply_counts.groupby('video_id')['reply_depth'].max().rename('max_thread_depth')

    # 2. Negativity Ratio: Total percentage of negative interactions per video
    sentiment_counts = pd.crosstab(df_comments['video_id'], df_comments['sentiment_label'], normalize='index')
    negative_ratio = sentiment_counts.get('negative', pd.Series(0.0, index=sentiment_counts.index)).rename('negativity_ratio')

    # 3. Compile Composite Controversy Score
    controversy_df = pd.concat([max_depth_spike, negative_ratio], axis=1).fillna(0)
    
    # Log transformation scaling prevents extreme viral threads from completely breaking graphs
    controversy_df['controversy_score'] = np.log1p(controversy_df['max_thread_depth']) * controversy_df['negativity_ratio']
    
    return controversy_df.reset_index()