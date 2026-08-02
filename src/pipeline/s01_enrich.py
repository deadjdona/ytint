from dotenv import load_dotenv
load_dotenv()
import os
import re
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_config():
    """
    Dynamically resolves the project root directory and loads the unified settings.
    Ensures absolute path compatibility across all execution environments.
    """
    current_file = Path(__file__).resolve()
    
    # Walk upward until we locate the parent directory containing the 'config' folder
    root_dir = current_file.parent
    while root_dir != root_dir.parent:
        if (root_dir / "config").is_dir():
            break
        root_dir = root_dir.parent
        
    config_path = root_dir / "config" / "settings.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"❌ Critical Configuration Alignment Failure:\n"
            f"Could not locate 'config/settings.yaml'.\n"
            f"Resolved root searched: {root_dir}"
        )
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Translate relative paths into absolute paths anchored to the project root
    config["paths"]["raw_db"] = str(root_dir / config["paths"]["raw_db"])
    config["paths"]["interim_dir"] = str(root_dir / config["paths"]["interim_dir"])
    config["paths"]["output_dir"] = str(root_dir / config["paths"]["output_dir"])
    
    return config

def calculate_linguistic_features(text):
    """
    Extracts stylistic and linguistic metrics for the given text.
    Implements Category 4: Linguistic & Stylistic.
    """
    if not text or not isinstance(text, str):
        return {
            'char_count': 0, 'word_count': 0, 'emoji_count': 0, 
            'all_caps_ratio': 0.0, 'punctuation_intensity': 0.0, 'lexical_richness': 0.0
        }

    # Basic counts
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    
    # Emoji detection (simple regex for non-ASCII/Unicode symbols)
    emoji_count = len(re.findall(r'[^\x00-\x7F]+', text))
    
    # All-caps ratio
    upper_chars = sum(1 for c in text if c.isupper())
    all_caps_ratio = upper_chars / char_count if char_count > 0 else 0.0
    
    # Punctuation intensity (count of !, ?, .)
    punctuation_count = len(re.findall(r'[!?.]', text))
    punctuation_intensity = punctuation_count / char_count if char_count > 0 else 0.0
    
    # Lexical richness (Type-Token Ratio)
    unique_words = len(set(text.lower().split()))
    lexical_richness = unique_words / word_count if word_count > 0 else 0.0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'emoji_count': emoji_count,
        'all_caps_ratio': round(all_caps_ratio, 4),
        'punctuation_intensity': round(punctuation_intensity, 4),
        'lexical_richness': round(lexical_richness, 4)
    }

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
    
    # --- 1. Sentiment Inference ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    sentiments, scores = [], []
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    with torch.no_grad():
        for i in tqdm(range(0, len(df_comments), batch_size), desc="🎭 Running Sentiment Inference"):
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
    
    # --- 2. Linguistic & Stylistic Features ---
    print("📊 Extracting Linguistic & Stylistic features...")
    linguistic_data = df_comments['text'].apply(calculate_linguistic_features).apply(pd.Series)
    df_comments = pd.concat([df_comments, linguistic_data], axis=1)

    # --- 3. Temporal Enrichment ---
    if videos_file.exists():
        df_videos = pd.read_parquet(videos_file)
        df_comments = df_comments.merge(
            df_videos[['video_id', 'published_at']], 
            on='video_id', 
            how='left', 
            suffixes=('', '_video')
        )
        
        df_comments['days_since_upload'] = (
            pd.to_datetime(df_comments['published_at']) - 
            pd.to_datetime(df_comments['published_at_video'])
        ).dt.days
        
        df_comments = df_comments.drop(columns=['published_at_video'])
    else:
        df_comments['days_since_upload'] = 0

    df_comments.to_parquet(comments_file, index=False)
    print("✅ Stage 01 Sentiment & Linguistic Enrichment Complete!")

if __name__ == "__main__":
    enrich_comments()