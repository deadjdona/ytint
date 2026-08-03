"""Stage 10: Topic Evolution over Time (s10_topic_evolution.py)

Maps how semantic intent and topic prevalence shift across the dataset 
over months/years using flowing Streamgraphs or Alluvial Diagrams.
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config

def run_topic_evolution():
    print("🌊 Starting Topic Evolution over Time (s10)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    topic_meta_file = out_dir / "topic_metadata.parquet"
    
    if not comments_file.exists() or not topic_meta_file.exists():
        print("⚠️ Missing comments_clean.parquet or topic_metadata.parquet. Skipping s10.")
        return
        
    print("  -> Loading comments and topic metadata...")
    df = pd.read_parquet(comments_file, columns=['comment_id', 'published_at', 'topic'])
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
        # Create a mapping dictionary from Topic to Name
        # Topic is often integer in df_meta, but check just in case
        topic_map = dict(zip(df_meta['Topic'], df_meta['Name']))
        df_valid['topic_name'] = df_valid['topic'].map(topic_map)
    else:
        df_valid['topic_name'] = "Topic " + df_valid['topic'].astype(str)
        
    print("  -> Aggregating topic volume by Month...")
    df_valid['published_at'] = pd.to_datetime(df_valid['published_at'])
    df_valid['year_month'] = df_valid['published_at'].dt.to_period('M').dt.to_timestamp()
    
    # Group by Year-Month and Topic
    df_grouped = df_valid.groupby(['year_month', 'topic_name']).size().reset_index(name='comment_count')
    
    # Pivot into wide format for Streamgraphs
    # Index = year_month, Columns = topics, Values = comment_count (fill 0)
    df_pivot = df_grouped.pivot(index='year_month', columns='topic_name', values='comment_count').fillna(0)
    
    # Filter to top N topics to keep the chart readable (e.g. Top 15 largest topics overall)
    top_topics = df_pivot.sum().sort_values(ascending=False).head(15).index
    df_pivot_top = df_pivot[top_topics]
    
    out_file = out_dir / "topic_evolution.parquet"
    df_pivot_top.to_parquet(out_file)
    print(f"✅ Generated topic evolution matrix for {len(top_topics)} topics across {len(df_pivot_top)} months.")
    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    run_topic_evolution()
