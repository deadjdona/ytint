"""Stage 19: Acquisition Cohorts (s19_cohorts.py)

Groups authors into cohorts based on the first video they ever commented on.
Builds a cohort retention matrix to see how many users acquired in Video X
returned to comment on Video Y.
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config
import numpy as np

def run_cohorts():
    print("👥 Starting Cohort Analysis (s19)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s19.")
        return
        
    print("  -> Loading comments...")
    df = pd.read_parquet(comments_file, columns=['author_channel_id', 'video_id', 'published_at'])
    df = df.dropna(subset=['author_channel_id', 'video_id', 'published_at'])
    
    if df.empty:
        return
        
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    
    # 1. Determine Video Chronology (by earliest comment on the video)
    print("  -> Ordering videos chronologically...")
    video_dates = df.groupby('video_id')['published_at'].min().sort_values()
    chronological_videos = video_dates.index.tolist()
    
    if len(chronological_videos) > 60:
        print("  -> Truncating to top 60 most active videos to maintain heatmap readability.")
        top_vids = df['video_id'].value_counts().head(60).index
        chronological_videos = [v for v in chronological_videos if v in top_vids]
    
    # 2. Determine Author Acquisition Cohort
    print("  -> Assigning users to acquisition cohorts...")
    author_first = df.groupby('author_channel_id')['published_at'].min().reset_index()
    author_first = author_first.rename(columns={'published_at': 'first_comment_time'})
    
    acquisition_events = df.merge(author_first, left_on=['author_channel_id', 'published_at'], right_on=['author_channel_id', 'first_comment_time'])
    acquisition_events = acquisition_events.drop_duplicates(subset=['author_channel_id'])
    
    cohort_map = dict(zip(acquisition_events['author_channel_id'], acquisition_events['video_id']))
    df['cohort'] = df['author_channel_id'].map(cohort_map)
    
    # 3. Build Retention Matrix
    print("  -> Building Retention Matrix...")
    df_filtered = df[df['video_id'].isin(chronological_videos) & df['cohort'].isin(chronological_videos)]
    
    cohort_eng = df_filtered.groupby(['cohort', 'video_id'])['author_channel_id'].nunique().reset_index()
    
    matrix = cohort_eng.pivot(index='cohort', columns='video_id', values='author_channel_id').fillna(0)
    matrix = matrix.reindex(index=chronological_videos, columns=chronological_videos, fill_value=0)
    
    cohort_sizes = df_filtered.groupby('cohort')['author_channel_id'].nunique()
    
    matrix_pct = matrix.copy()
    for cohort in matrix_pct.index:
        size = cohort_sizes.get(cohort, 1)
        if size == 0: size = 1
        matrix_pct.loc[cohort] = (matrix.loc[cohort] / size) * 100
        
    out_file = out_dir / "cohort_retention.parquet"
    out_file_pct = out_dir / "cohort_retention_pct.parquet"
    
    matrix.reset_index(names=['cohort']).to_parquet(out_file, index=False)
    matrix_pct.reset_index(names=['cohort']).to_parquet(out_file_pct, index=False)
    
    # 4. New vs Returning Share per Video
    print("  -> Calculating New vs Returning Share...")
    new_returning = []
    for vid in chronological_videos:
        vid_users = df[df['video_id'] == vid]['author_channel_id'].unique()
        if len(vid_users) == 0: continue
        
        # New users have their cohort == this video
        new_users = df[(df['video_id'] == vid) & (df['cohort'] == vid)]['author_channel_id'].nunique()
        returning_users = len(vid_users) - new_users
        
        new_returning.append({
            'video_id': vid,
            'new_count': new_users,
            'returning_count': returning_users
        })
        
    df_nr = pd.DataFrame(new_returning)
    df_nr.to_parquet(out_dir / "new_vs_returning.parquet", index=False)
    
    print(f"✅ Generated Cohort Retention Matrix ({len(chronological_videos)}x{len(chronological_videos)}) and New vs Returning Share.")

if __name__ == "__main__":
    run_cohorts()
