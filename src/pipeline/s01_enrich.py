import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def enrich_comments():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    comments_file = interim_dir / "comments_clean.parquet"
    videos_file = interim_dir / "videos_clean.parquet"
    
    if not comments_file.exists():
        print("❌ Clean comments file not found.")
        return

    df_comments = pd.read_parquet(comments_file)
    df_comments['text'] = df_comments['text'].fillna("").astype(str)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running on: {device.type.upper()}")
    
    model_name = config["stage_01_enrich"]["sentiment_model"]
    batch_size = config["stage_01_enrich"]["batch_size"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    sentiments, scores = [], []
    label_map = {0: "neutral", 1: "positive", 2: "negative"}

    with torch.no_grad():
        for i in tqdm(range(0, len(df_comments), batch_size)):
            batch_texts = df_comments['text'].iloc[i:i+batch_size].tolist()
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            for prob in probabilities:
                pred = np.argmax(prob)
                sentiments.append(label_map[pred])
                scores.append(float(prob[pred]))

    df_comments['sentiment_label'] = sentiments
    df_comments['sentiment_confidence'] = scores
    
    if videos_file.exists():
        df_videos = pd.read_parquet(videos_file)
        df_merged = df_comments.merge(df_videos[['video_id', 'published_at']], on='video_id', suffixes=('', '_video'))
        df_comments['days_since_upload'] = (pd.to_datetime(df_merged['published_at']) - pd.to_datetime(df_merged['published_at_video'])).dt.days
    else:
        df_comments['days_since_upload'] = 0

    df_comments.to_parquet(comments_file, index=False)
    print("✅ Stage 01 Sentiment Enrichment Complete!")

if __name__ == "__main__":
    enrich_comments()
