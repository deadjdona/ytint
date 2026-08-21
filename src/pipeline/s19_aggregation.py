"""Stage 19: Author & Video Aggregations (s19_aggregation.py)

Code Review Task Alignment:
- RFM Segmentation: Champions, Loyal, At Risk cohorts via K-Means
- RobustScaler for Zipf correction: Outlier-robust scaling before K-Means
- Exponential decay half-life: compute_half_life() curve fitting with right-censored imputation
- Savitzky-Golay revival spikes: >3σ anomaly detection on smoothed daily volume
- Gini coefficient: Lorenz inequality measure per video
- Fan Loyalty tracking: is_single_video_fan vs multi-video author footprint
- Display Name Reuse risk: is_display_name_reused impersonation / bot detection
- UI Metric Synthesis: Integrated via s04_synthesis.py
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
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import warnings
from engine.config_loader import load_config

# Suppress convergence warnings from curve_fit on weird data distributions
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def gini(x):
    """
    Task: Gini coefficient
    Math: Calculates Gini coefficient G = (n + 1 - 2 * sum(cum_x) / cum_x[-1]) / n.
          Gini coefficient measures inequality of like distribution across comments (0 = equal, 1 = concentrated).
    """
    if len(x) <= 1 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def compute_half_life(df_video_comments):
    """
    Task: Exponential decay half-life
    Math: Fits N(t) = N0 * exp(-lambda * t) using non-linear least squares (curve_fit).
          Attention half-life is computed as t_1/2 = ln(2) / lambda.
    Hardcode: Cap at 365.0 days and fallback to global median if < 7 days data to prevent right-censored mathematical instability.
    """
    df_video_comments['days_since_upload'] = (df_video_comments['minutes_since_upload'] / 1440.0).astype(int)
    daily_vol = df_video_comments.groupby('days_since_upload').size().reset_index(name='volume')
    
    if len(daily_vol) < 3 or daily_vol['days_since_upload'].max() < 7:
        return -1.0 
        
    def exp_decay(t, N0, lam):
        return N0 * np.exp(-lam * t)
        
    try:
        x_data = daily_vol['days_since_upload'].values
        y_data = daily_vol['volume'].values
        
        popt, _ = curve_fit(exp_decay, x_data, y_data, p0=[max(y_data), 0.1], maxfev=1000)
        lam = popt[1]
        
        if lam <= 0:
            return -1.0
            
        half_life = np.log(2) / lam
        return min(half_life, 365.0)
    except RuntimeError:
        return -1.0

def aggregate_data():
    """
    Tasks: RFM Segmentation, RobustScaler for Zipf correction, Savitzky-Golay revival spikes,
           Fan Loyalty tracking, Display Name Reuse risk.
    """
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    comments_file = interim_dir / "comments_clean.parquet"
    authors_file = interim_dir / "authors_network_metrics.parquet"
    semantics_file = interim_dir / "semantic_topics.parquet"
    
    print("📥 Loading cross-stage datasets...")
    df_c = pd.read_parquet(comments_file)
    
    if authors_file.exists():
        df_a = pd.read_parquet(authors_file)
    else:
        df_a = pd.DataFrame()
        
    if semantics_file.exists():
        df_s = pd.read_parquet(semantics_file)
        df_c = df_c.merge(df_s, on='comment_id', how='left')
        
    df_c['published_at'] = pd.to_datetime(df_c['published_at'])
    now = df_c['published_at'].max() 
    
    # --- 1. Author Aggregation (RFM & Cohorts) ---
    print("👤 Aggregating Author Profiles & Computing RFM Cohorts...")
    
    # Tasks: RFM metrics (Recency = days since last comment, Frequency = comment count, Monetary = sum of likes)
    author_agg = df_c.groupby('author_channel_id').agg(
        recency=('published_at', lambda x: (now - x.max()).days),
        frequency=('comment_id', 'count'),
        monetary=('like_count', 'sum'),
        avg_sentiment=('vader_compound', 'mean'),
        unique_videos_commented=('video_id', 'nunique')
    ).reset_index()
    
    # Task: Fan Loyalty tracking (is_single_video_fan vs multi-video author footprint)
    author_agg['is_single_video_fan'] = author_agg['unique_videos_commented'] == 1
    
    # Task: Display Name Reuse risk (flags author_display_names mapped to >1 distinct channel_id)
    if 'author_display_name' in df_c.columns:
        name_counts = df_c.groupby('author_display_name')['author_channel_id'].nunique()
        reused_names = set(name_counts[name_counts > 1].index)
        
        author_names = df_c.groupby('author_channel_id')['author_display_name'].first().to_dict()
        author_agg['author_display_name'] = author_agg['author_channel_id'].map(author_names)
        author_agg['is_display_name_reused'] = author_agg['author_display_name'].isin(reused_names)
    
    if not df_a.empty:
        author_agg = author_agg.merge(df_a, on='author_channel_id', how='left')
        
    # Task: RobustScaler for Zipf correction & K-Means RFM Segmentation
    # Math: RobustScaler scales features via median and Interquartile Range z = (x - median) / IQR.
    # Essential for heavy-tailed Zipf power-law distributions to prevent super-fan outliers from distorting K-Means centroids.
    # Hardcode n_clusters=3: Segments user base into 3 distinct cohorts (Champions, Loyal, At Risk).
    print("🎯 Running K-Means clustering with RobustScaler...")
    features = ['recency', 'frequency', 'monetary']
    scaler = RobustScaler()
    scaled_rfm = scaler.fit_transform(author_agg[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    author_agg['rfm_cluster'] = kmeans.fit_predict(scaled_rfm)
    
    # Dynamically assign RFM labels based on cluster frequency medians
    cluster_medians = author_agg.groupby('rfm_cluster')['frequency'].median()
    sorted_clusters = cluster_medians.sort_values(ascending=False).index
    label_map = {
        sorted_clusters[0]: "Champions",
        sorted_clusters[1]: "Loyal",
        sorted_clusters[2]: "At Risk"
    }
    author_agg['rfm_cohort'] = author_agg['rfm_cluster'].map(label_map)
    author_agg = author_agg.drop(columns=['rfm_cluster'])
    
    # --- 2. Video Aggregation (Attention & Narrative) ---
    print("📺 Aggregating Video Profiles & Attention Dynamics...")
    
    video_records = []
    
    video_groups = df_c.groupby('video_id')
    for vid, group in tqdm(video_groups, desc="Analyzing Videos"):
        total_comments = len(group)
        total_likes = group['like_count'].sum()
        avg_sentiment = group['vader_compound'].mean()
        
        # Task: Gini coefficient per video
        gini_coeff = gini(group['like_count'].values)
        
        # Task: Savitzky-Golay revival spikes (>3σ anomaly detection on smoothed daily volume)
        # Math: Savitzky-Golay fits a degree-2 polynomial over a 7-day window. Residuals r_t = y_t - y_hat_t 
        # exceeding 3 standard deviations (r_t > 3 * std(r)) identify statistically anomalous revival events.
        group['days_since_upload'] = (group['minutes_since_upload'] / 1440.0).astype(int)
        daily = group.groupby('days_since_upload').size()
        
        spikes = 0
        if len(daily) > 7:
            window = min(len(daily) if len(daily) % 2 != 0 else len(daily) - 1, 7)
            if window >= 3:
                smoothed = savgol_filter(daily.values, window, 2)
                residuals = daily.values - smoothed
                std_res = np.std(residuals)
                if std_res > 0:
                    spikes = np.sum(residuals > 3 * std_res)
                    
        # Task: Exponential decay half-life
        half_life = compute_half_life(group.copy())
        
        video_records.append({
            'video_id': vid,
            'total_comments': total_comments,
            'total_likes': total_likes,
            'avg_sentiment': avg_sentiment,
            'gini_coefficient': round(gini_coeff, 4),
            'revival_spikes': int(spikes),
            'attention_half_life_days': round(half_life, 2)
        })
        
    df_video = pd.DataFrame(video_records)
    
    # Impute missing half-lifes with global median
    median_hl = df_video[df_video['attention_half_life_days'] > 0]['attention_half_life_days'].median()
    if pd.isna(median_hl):
        median_hl = 7.0
    df_video['attention_half_life_days'] = df_video['attention_half_life_days'].replace(-1.0, median_hl)
    
    # --- 3. Consistency & I/O ---
    print("💾 Archiving Final Cohort Data to Output Directory...")
    authors_final_path = out_dir / "authors_final.parquet"
    videos_final_path = out_dir / "videos_final.parquet"
    
    author_agg.to_parquet(authors_final_path, index=False)
    df_video.to_parquet(videos_final_path, index=False)
    
    print("✅ Stage 04 Aggregation & Cohort Modeling Complete!")

if __name__ == "__main__":
    aggregate_data()

