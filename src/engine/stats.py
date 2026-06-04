import numpy as np
import pandas as pd
import ruptures as rpt

def calculate_rolling_z_scores(series, window_size=90, min_periods=7):
    """
    Computes a rolling, historical Z-score for a given series.
    Protects against future data leakage by calculating standard deviations 
    using only historical windows.
    """
    # Calculate rolling metrics based on historical context window
    rolling_mean = series.rolling(window=window_size, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window_size, min_periods=min_periods).std()
    
    # Avoid zero division errors on flat or dead timelines
    rolling_std = rolling_std.replace(0, np.nan)
    
    # Compute the number of standard deviations from the rolling mean
    z_scores = (series - rolling_mean) / rolling_std
    return z_scores.fillna(0)

def detect_volume_anomalies(df_comments, z_threshold=3.0, window_days=90):
    """
    Identifies 'Event Artifacts' based on rapid spikes in comment velocity.
    Returns a dataframe containing only the outlier dates that pass the z-score bar.
    """
    # 1. Aggregate daily conversational activity volume
    daily_counts = (
        df_comments.groupby(df_comments['published_at'].dt.date)
        .size()
        .reset_index(name='comment_count')
    )
    daily_counts = daily_counts.sort_values('published_at').reset_index(drop=True)
    
    # 2. Compute timeline volatility metrics
    daily_counts['z_score'] = calculate_rolling_z_scores(
        daily_counts['comment_count'], 
        window_size=window_days
    )
    
    # 3. Filter down to true anomaly spikes
    anomalies = daily_counts[daily_counts['z_score'].abs() >= z_threshold].copy()
    return anomalies

def segment_thematic_eras(df_comments, min_duration_weeks=12, penalty_modifier=2.0):
    """
    Applies change-point detection (Pelt or Window-based algorithm) over shifting 
    weekly topic distributions to split the timeline into long, stable 'Eras'.
    """
    # 1. Pivot text distributions to build a clean temporal matrix (Weeks x Topics)
    df_comments['week'] = df_comments['published_at'].dt.to_period('W').dt.to_timestamp()
    
    # Drop noise records (-1) to look purely at structured narrative trends
    df_structured = df_comments[df_comments['topic_id'] != -1]
    
    if df_structured.empty:
        return []

    # Build matrix: rows are weeks, columns are topic IDs, values are comment counts
    topic_matrix = (
        df_structured.groupby(['week', 'topic_id'])
        .size()
        .unstack(fill_value=0)
    )
    
    # Normalize rows to percentages so volume fluctuations don't mimic topic drift
    row_sums = topic_matrix.sum(axis=1)
    normalized_matrix = topic_matrix.div(row_sums, axis=0).fillna(0).values
    
    if len(normalized_matrix) < min_duration_weeks:
        # Not enough history to calculate segments; return whole timeline as single Era
        return [0, len(topic_matrix)]

    # 2. Configure Ruptures cost function engine (L2 tracking for variance shifts)
    algo = rpt.Window(width=min_duration_weeks, model="l2").fit(normalized_matrix)
    
    # Predict splitting index boundaries based on structural config penalties
    try:
        change_points = algo.predict(pen=penalty_modifier)
        # Convert matrix row index locations back into actual datetime timestamps
        era_date_boundaries = [topic_matrix.index[idx - 1] for idx in change_points if idx < len(topic_matrix)]
        return era_date_boundaries
    except Exception as e:
        print(f"⚠️ Change-point algorithm could not converge: {e}")
        return []