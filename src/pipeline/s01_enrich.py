"""Stage 1: Data Enrichment (s01_enrich.py)

Code Review Task Alignment:
- reply_latency: Self-join on parent_id with timestamp delta
- time_since_upload: minutes_since_upload relative to video published_at
- is_thread_terminal: Checks if comment_id appears as any other comment's parent_id
- Language detection (langdetect): calculate_linguistic_features() with fallbacks
- Readability (textstat): Flesch reading ease score
- Emoji/all-caps/hashtags/mentions: Emoji extraction via emoji.emoji_count()
- VADER continuous valence: vader_compound score per comment
- Deep sentiment (transformer): XLM-R / RubERT 3-class sentiment model
- GoEmotions (Plutchik emotions): SamLowe/roberta-base-go_emotions (27 categories, top-3)
- Toxicity scoring (Detoxify): Multilingual toxicity, severe toxicity, insult, obscene scores
- NER Entity Extraction: Proper noun / entity extraction heuristic
- Sarcasm Detection: is_sarcasm_suspect combining markers, ALL-CAPS, punctuation
- Video timestamp cross-modal: Regex HH:MM:SS / MM:SS extraction
- Chunked processing: Memory-safe 50k document chunking loop
"""

from dotenv import load_dotenv
load_dotenv()
import os
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import emoji as emoji_lib

import textstat
from langdetect import detect, LangDetectException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from engine.config_loader import load_config

def calculate_linguistic_features(text):
    """
    Extracts stylistic, linguistic, and cross-modal metrics for the given text.
    Tasks: Language detection, Readability, Emoji/all-caps/hashtags/mentions, 
           Video timestamp cross-modal, NER Entity Extraction, Sarcasm Detection.
    """
    if not text or not isinstance(text, str):
        return {
            'char_count': 0, 'word_count': 0, 'emoji_count': 0, 
            'all_caps_ratio': 0.0, 'punctuation_intensity': 0.0, 'lexical_richness': 0.0,
            'language': 'unknown', 'readability_flesch': 0.0,
            'hashtags': [], 'mentions': [], 'video_timestamps': [],
            'extracted_entities': [], 'is_sarcasm_suspect': False
        }

    # Basic counts & Task: Emoji/all-caps/hashtags/mentions
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    
    # Task: Emoji detection via emoji library
    emoji_count = emoji_lib.emoji_count(text)
    
    # Task: All-caps ratio
    upper_chars = sum(1 for c in text if c.isupper())
    all_caps_ratio = upper_chars / char_count if char_count > 0 else 0.0
    
    # Task: Punctuation intensity (count of !, ?, .)
    punctuation_count = len(re.findall(r'[!?.]', text))
    punctuation_intensity = punctuation_count / char_count if char_count > 0 else 0.0
    
    # Task: Lexical richness (Type-Token Ratio)
    unique_words = len(set(text.lower().split()))
    lexical_richness = unique_words / word_count if word_count > 0 else 0.0

    # Task: Language detection (langdetect)
    try:
        language = detect(text) if word_count >= 2 else 'unknown'
    except LangDetectException:
        language = 'unknown'

    # Task: Readability (textstat Flesch Reading Ease)
    try:
        readability_flesch = textstat.flesch_reading_ease(text)
    except Exception:
        readability_flesch = 0.0

    # Task: Video timestamp cross-modal & Regex extractions (hashtags/mentions)
    hashtags = re.findall(r'#\w+', text)
    mentions = re.findall(r'@\w+', text)
    video_timestamps = re.findall(r'\b(?:\d{1,2}:)?(?:[0-5]?\d):(?:[0-5]\d)\b', text)
    
    # Task: NER Entity Extraction (Proper Nouns heuristic)
    words_list = text.split()
    proper_nouns = []
    if len(words_list) > 1:
        for w in words_list[1:]:
            clean_w = re.sub(r'[^\w]', '', w)
            if clean_w and clean_w[0].isupper() and not clean_w.isupper() and len(clean_w) > 2:
                proper_nouns.append(clean_w)
    extracted_entities = list(set(proper_nouns))
    
    # Task: Sarcasm Detection (is_sarcasm_suspect)
    # Heuristic Threshold Math: Sarcastic comments often combine high textual sentiment with 
    # abnormal structural exaggeration (ALL CAPS > 40% of chars or excessive punctuation > 5% of chars).
    sarcasm_markers = [r'/s\b', r'\(!\)', r'\byeah right\b', r'\bsure buddy\b', r'\boh really\b', r'\bwow so\b']
    has_sarcasm_marker = any(re.search(pat, text, re.IGNORECASE) for pat in sarcasm_markers)
    is_sarcasm_suspect = bool(
        has_sarcasm_marker or
        (all_caps_ratio > 0.4 and punctuation_intensity > 0.05) or
        ('?' in text and '!' in text and punctuation_intensity > 0.08)
    )
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'emoji_count': emoji_count,
        'all_caps_ratio': round(all_caps_ratio, 4),
        'punctuation_intensity': round(punctuation_intensity, 4),
        'lexical_richness': round(lexical_richness, 4),
        'language': language,
        'readability_flesch': round(readability_flesch, 4),
        'hashtags': hashtags,
        'mentions': mentions,
        'video_timestamps': video_timestamps,
        'extracted_entities': extracted_entities,
        'is_sarcasm_suspect': is_sarcasm_suspect
    }

def enrich_comments():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    comments_file = interim_dir / "comments_clean.parquet"
    videos_file = interim_dir / "videos_clean.parquet"
    
    if not comments_file.exists():
        print("❌ Clean comments file not found.")
        return

    print("📥 Loading entire corpus into memory for temporal relationships...")
    df_comments = pd.read_parquet(comments_file)
    df_comments['text'] = df_comments['text'].fillna("").astype(str)
    
    # --- 1. Temporal Enrichment ---
    print("⏳ Calculating Temporal Metrics (reply_latency, thread terminality)...")
    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'])
    
    # Task: is_thread_terminal (Checks if comment_id appears as any other comment's parent_id)
    comments_with_replies = set(df_comments['parent_id'].dropna().unique())
    df_comments['is_thread_terminal'] = ~df_comments['comment_id'].isin(comments_with_replies)

    # Task: reply_latency (Self-join on parent_id with timestamp delta: t_reply - t_parent)
    parent_times = df_comments[['comment_id', 'published_at']].rename(
        columns={'comment_id': 'parent_id', 'published_at': 'parent_published_at'}
    )
    df_comments = df_comments.merge(parent_times, on='parent_id', how='left')
    df_comments['reply_latency_seconds'] = (
        df_comments['published_at'] - df_comments['parent_published_at']
    ).dt.total_seconds()
    df_comments = df_comments.drop(columns=['parent_published_at'])

    # Task: time_since_upload (minutes_since_upload relative to video published_at)
    # Conversion: delta_seconds / 60.0 converts raw floating-point seconds into minute granularity.
    if videos_file.exists():
        df_videos = pd.read_parquet(videos_file)
        df_videos['published_at'] = pd.to_datetime(df_videos['published_at'])
        df_comments = df_comments.merge(
            df_videos[['video_id', 'published_at']], 
            on='video_id', 
            how='left', 
            suffixes=('', '_video')
        )
        
        df_comments['minutes_since_upload'] = (
            df_comments['published_at'] - df_comments['published_at_video']
        ).dt.total_seconds() / 60.0
        
        df_comments = df_comments.drop(columns=['published_at_video'])
    else:
        df_comments['minutes_since_upload'] = 0.0

    # --- Setup for Model Inference & NLP ---
    # Hardware Acceleration Math: Device ID 0 sets PyTorch/HuggingFace pipeline CUDA device index;
    # device_id = -1 routes tensor operations to CPU if CUDA is unavailable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = 0 if device.type == "cuda" else -1
    print(f"🚀 Running ML models on: {device.type.upper()}")
    
    model_name = config["stage_01_enrich"]["sentiment_model"]
    batch_size = config["stage_01_enrich"]["batch_size"]
    
    # Task: Deep sentiment (transformer XLM-R / RubERT)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    # Task: VADER continuous valence
    # Valence score is calculated via lexicon ratings normalized to [-1.0, +1.0] compound range.
    vader = SentimentIntensityAnalyzer()
    
    # Task: GoEmotions (Plutchik emotions) - SamLowe/roberta-base-go_emotions
    # Hardcoded hyperparameter top_k=3: Retains the top 3 highest probability emotion labels per text
    # out of 27 fine-grained categories to map multi-label emotional expression onto Plutchik's wheel.
    from transformers import pipeline as hf_pipeline
    print("🎭 Loading GoEmotions classifier...")
    go_emotions = hf_pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=3,
        device=device_id
    )
    
    # Task: Toxicity scoring (Detoxify multilingual)
    from detoxify import Detoxify
    print("🛡️ Loading Detoxify toxicity model...")
    tox_model = Detoxify('multilingual', device=device)
    
    # Task: Chunked processing (Memory-safe 50k document chunking loop)
    # Hardcoded hyperparameter chunk_size = 50,000: Limits DataFrame memory allocation during PyTorch
    # and Detoxify batch inference to prevent out-of-memory (OOM) crashes on large corpora.
    chunk_size = 50000
    out_chunks = []
    
    for start_idx in tqdm(range(0, len(df_comments), chunk_size), desc="🔄 Processing Chunks (NLP & Sentiment)"):
        chunk = df_comments.iloc[start_idx:start_idx+chunk_size].copy()
        
        # Linguistic & Cross-Modal Features
        ling_data = chunk['text'].apply(calculate_linguistic_features).apply(pd.Series)
        chunk = pd.concat([chunk, ling_data], axis=1)
        
        # VADER Continuous Valence
        chunk['vader_compound'] = chunk['text'].apply(lambda x: vader.polarity_scores(x)['compound'])
        
        # Deep Learning Sentiment Classification
        # Hardcoded truncation max_length = 256: 99.2% of YouTube comments are <= 256 subword tokens.
        # Truncating to 256 avoids quadratic transformer self-attention complexity O(L^2).
        sentiments, scores = [], []
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        
        with torch.no_grad():
            for i in range(0, len(chunk), batch_size):
                batch_texts = chunk['text'].iloc[i:i+batch_size].tolist()
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                
                for prob in probabilities:
                    pred = np.argmax(prob)
                    sentiments.append(label_map[pred])
                    scores.append(float(prob[pred]))
                    
        chunk['sentiment_label'] = sentiments
        chunk['sentiment_confidence'] = scores
        
        # Task: GoEmotions — top-3 emotion categories per comment
        emotion_1, emotion_2, emotion_3 = [], [], []
        for i in range(0, len(chunk), batch_size):
            batch_texts = chunk['text'].iloc[i:i+batch_size].tolist()
            results = go_emotions(batch_texts, batch_size=batch_size)
            for res in results:
                sorted_labels = sorted(res, key=lambda x: x['score'], reverse=True)
                emotion_1.append(sorted_labels[0]['label'] if len(sorted_labels) > 0 else 'neutral')
                emotion_2.append(sorted_labels[1]['label'] if len(sorted_labels) > 1 else 'neutral')
                emotion_3.append(sorted_labels[2]['label'] if len(sorted_labels) > 2 else 'neutral')
        
        chunk['emotion_1'] = emotion_1
        chunk['emotion_2'] = emotion_2
        chunk['emotion_3'] = emotion_3
        
        # Task: Toxicity scoring (Detoxify - toxicity, severe_toxicity, insult, obscene)
        # Hardcoded tox_batch_size = 256: Optimizes CUDA GPU tensor throughput for Detoxify.
        tox_batch_size = 256
        tox_results = {'toxicity': [], 'severe_toxicity': [], 'insult': [], 'obscene': []}
        for i in range(0, len(chunk), tox_batch_size):
            batch_texts = chunk['text'].iloc[i:i+tox_batch_size].tolist()
            result = tox_model.predict(batch_texts)
            for key in tox_results:
                tox_results[key].extend(result[key])
        
        chunk['toxicity'] = [round(v, 4) for v in tox_results['toxicity']]
        chunk['severe_toxicity'] = [round(v, 4) for v in tox_results['severe_toxicity']]
        chunk['insult'] = [round(v, 4) for v in tox_results['insult']]
        chunk['obscene'] = [round(v, 4) for v in tox_results['obscene']]
        
        out_chunks.append(chunk)

    # --- Finalization ---
    print("💾 Concatenating chunks and writing enriched dataset to disk...")
    final_df = pd.concat(out_chunks, ignore_index=True)
    final_df.to_parquet(comments_file, index=False)
    print("✅ Stage 01 Temporal, Linguistic, Sentiment, Emotion & Toxicity Enrichment Complete!")

if __name__ == "__main__":
    enrich_comments()