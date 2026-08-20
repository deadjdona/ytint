"""Stage 21: Topic-Cohort Clustering (s21_topic_cohorts.py)

Segments authors into cohorts based on their primary topic of interest.
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


import pandas as pd
from pathlib import Path
from engine.config_loader import load_config

def run_topic_cohorts():
    print("📊 Starting Topic-Cohort Clustering (s21)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s21.")
        return
        
    print("  -> Loading comments and topics...")
    try:
        df = pd.read_parquet(comments_file, columns=['author_channel_id', 'topic'])
    except ValueError:
        print("⚠️ 'topic' column not found. Run topic modeling (s04) first.")
        return
        
    df = df.dropna()
    if df.empty: return
    
    print("  -> Calculating topic affinities...")
    author_topics = df.groupby(['author_channel_id', 'topic']).size().reset_index(name='count')
    primary_topics = author_topics.sort_values('count', ascending=False).groupby('author_channel_id').first().reset_index()
    
    cohort_sizes = primary_topics['topic'].value_counts().reset_index()
    cohort_sizes.columns = ['topic_cohort', 'author_count']
    
    out_file = out_dir / "topic_cohorts.parquet"
    cohort_sizes.to_parquet(out_file, index=False)
    
    print(f"✅ Generated Topic Cohorts for {len(primary_topics)} authors across {len(cohort_sizes)} topics.")

if __name__ == "__main__":
    run_topic_cohorts()
