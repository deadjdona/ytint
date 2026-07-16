import os
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
    
    # FIX: Secure, aligned calculations via explicit in-place left merging
    if videos_file.exists():
        df_videos = pd.read_parquet(videos_file)
        
        # Merge the temporary video upload timestamp into df_comments cleanly 
        df_comments = df_comments.merge(
            df_videos[['video_id', 'published_at']], 
            on='video_id', 
            how='left', 
            suffixes=('', '_video')
        )
        
        # Safe vectorized chronological subtraction on the validated index grid
        df_comments['days_since_upload'] = (
            pd.to_datetime(df_comments['published_at']) - 
            pd.to_datetime(df_comments['published_at_video'])
        ).dt.days
        
        # Remove the staging column to keep the file format thin and concise
        df_comments = df_comments.drop(columns=['published_at_video'])
    else:
        df_comments['days_since_upload'] = 0

    df_comments.to_parquet(comments_file, index=False)
    print("✅ Stage 01 Sentiment Enrichment Complete!")

if __name__ == "__main__":
    enrich_comments()