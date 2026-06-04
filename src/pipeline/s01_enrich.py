import os
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def load_config(config_path="config/settings.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def enrich_comments():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    comments_file = interim_dir / "comments_clean.parquet"
    videos_file = interim_dir / "videos_clean.parquet"
    
    if not comments_file.exists():
        print(f"❌ Clean comments file not found at {comments_file}. Run ingestion first.")
        return

    print("📥 Loading cleaned comment layers...")
    df_comments = pd.read_parquet(comments_file)
    df_comments['text'] = df_comments['text'].fillna("").astype(str)
    
    # Initialize VADER Lexicon Engine natively supported on Python 3.14
    print("🤖 Initializing Python 3.14 Native Lexicon Engine (VADER)...")
    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        print("📥 Downloading missing VADER lexicon dependency...")
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()

    sentiments = []
    scores = []

    print(f"🧠 Running high-speed sentiment screening across {len(df_comments)} rows...")
    
    # VADER is incredibly fast, so we can process row by row without complex batch tensors
    for text in tqdm(df_comments['text']):
        polarity = sia.polarity_scores(text)
        compound = polarity['compound']
        
        # Standard rules to map continuous VADER compound scores to categorical tags
        if compound >= 0.05:
            sentiments.append("positive")
            scores.append(compound)
        elif compound <= -0.05:
            sentiments.append("negative")
            scores.append(abs(compound))
        else:
            sentiments.append("neutral")
            scores.append(1.0 - abs(compound))

    # Append results cleanly back to the memory frame
    df_comments['sentiment_label'] = sentiments
    df_comments['sentiment_confidence'] = scores
    
    # Calculate retrospective memory latency timelines safely
    if videos_file.exists():
        print("📐 Computing retrospective memory latency timelines...")
        df_videos = pd.read_parquet(videos_file)
        df_merged = df_comments.merge(df_videos[['video_id', 'published_at']], on='video_id', suffixes=('', '_video'))
        
        df_comments['days_since_upload'] = (
            pd.to_datetime(df_merged['published_at']) - pd.to_datetime(df_merged['published_at_video'])
        ).dt.days
    else:
        df_comments['days_since_upload'] = 0

    print("💾 Committing optimized metrics to analytical engine Parquet files...")
    df_comments.to_parquet(comments_file, index=False)
    print("✅ Enrichment phase successfully completed using Python 3.14!")

if __name__ == "__main__":
    enrich_comments()