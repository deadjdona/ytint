"""Stage 13: Topic × Video Matrix (s13_topic_matrix.py)

Generates a cross-tabulation matrix showing which topics are most 
prevalent in which specific videos.
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config

def run_topic_matrix():
    print("🧩 Starting Topic × Video Matrix Calculation (s13)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    topic_meta_file = out_dir / "topic_metadata.parquet"
    
    if not comments_file.exists() or not topic_meta_file.exists():
        print("⚠️ Missing comments_clean.parquet or topic_metadata.parquet. Skipping s13.")
        return
        
    print("  -> Loading comments and topic metadata...")
    df = pd.read_parquet(comments_file, columns=['comment_id', 'video_id', 'topic'])
    df_meta = pd.read_parquet(topic_meta_file)
    
    if 'topic' not in df.columns:
        print("⚠️ No 'topic' column found in comments. Run s02_topics.py first. Skipping.")
        return
        
    # Ignore BERTopic outliers (Topic -1)
    df_valid = df[(df['topic'].notna()) & (df['topic'] != -1)].copy()
    
    if df_valid.empty:
        print("⚠️ No valid assigned topics found. Skipping.")
        return
        
    # Merge topic names
    if 'Name' in df_meta.columns:
        topic_map = dict(zip(df_meta['Topic'], df_meta['Name']))
        df_valid['topic_name'] = df_valid['topic'].map(topic_map)
    else:
        df_valid['topic_name'] = "Topic " + df_valid['topic'].astype(str)
        
    print("  -> Aggregating topic volume by Video...")
    # Group by Video and Topic
    df_grouped = df_valid.groupby(['video_id', 'topic_name']).size().reset_index(name='comment_count')
    
    # Pivot into a Matrix: Rows = Video, Columns = Topic
    df_pivot = df_grouped.pivot(index='video_id', columns='topic_name', values='comment_count').fillna(0)
    
    # Normalize per video (What percentage of a video's comments are in this topic?)
    # This prevents highly-commented videos from washing out the heatmap
    df_normalized = df_pivot.div(df_pivot.sum(axis=1), axis=0)
    
    # Filter to top N topics to keep the chart readable (e.g. Top 20 largest topics overall)
    top_topics = df_pivot.sum(axis=0).sort_values(ascending=False).head(20).index
    df_matrix = df_normalized[top_topics]
    
    out_file = out_dir / "topic_video_matrix.parquet"
    df_matrix.to_parquet(out_file)
    print(f"✅ Generated normalized topic-video matrix for {len(top_topics)} topics across {len(df_matrix)} videos.")
    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    run_topic_matrix()
