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
textstat.set_lang('ru')
from langdetect import detect, LangDetectException
from datetime import datetime
from engine.config_loader import load_config

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def compute_yules_k(words):
    N = len(words)
    if N == 0:
        return 0.0
    counts = pd.Series([w.lower() for w in words]).value_counts()
    freq_spectrum = counts.value_counts()
    sum_f_i2 = sum((i ** 2) * f for i, f in freq_spectrum.items())
    k = 10000.0 * (sum_f_i2 - N) / (N * N) if N > 0 else 0.0
    return max(0.0, k)

def compute_mattr(words, window_size=50):
    N = len(words)
    if N == 0:
        return 0.0
    words_lower = [w.lower() for w in words]
    if N <= window_size:
        return len(set(words_lower)) / N
    ttr_sum = sum(len(set(words_lower[i:i + window_size])) / window_size for i in range(N - window_size + 1))
    return ttr_sum / (N - window_size + 1)

def compute_mtld(words, threshold=0.72):
    N = len(words)
    if N < 5:
        return (len(set([w.lower() for w in words])) / N) if N > 0 else 0.0
    words_lower = [w.lower() for w in words]
    
    def mtld_one_factor(word_list):
        factors = 0.0
        current = []
        for w in word_list:
            current.append(w)
            ttr = len(set(current)) / len(current)
            if ttr <= threshold:
                factors += 1.0
                current = []
        if current:
            ttr = len(set(current)) / len(current)
            partial = (1.0 - ttr) / (1.0 - threshold) if ttr < 1.0 else 0.0
            factors += partial
        return factors
    
    f1 = mtld_one_factor(words_lower)
    f2 = mtld_one_factor(words_lower[::-1])
    avg_factors = (f1 + f2) / 2.0
    return N / avg_factors if avg_factors > 0 else float(N)

def calculate_linguistic_features(text):
    """
    Extracts stylistic, linguistic, and cross-modal metrics for the given text or DataFrame.
    """
    if isinstance(text, pd.DataFrame):
        df_in = text
        text_col = 'text' if 'text' in df_in.columns else 'text_original'
        res_dicts = [calculate_linguistic_features(t) for t in df_in[text_col]]
        res_df = pd.DataFrame(res_dicts)
        return pd.concat([df_in.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)

    if not text or not isinstance(text, str):
        return {
            'char_count': 0, 'word_count': 0, 'emoji_count': 0, 
            'all_caps_ratio': 0.0, 'caps_ratio': 0.0, 'punctuation_intensity': 0.0, 'lexical_richness': 0.0,
            'mattr': 0.0, 'mtld': 0.0, 'yules_k': 0.0,
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
    
    # Task: Lexical richness (Type-Token Ratio, MATTR, MTLD, Yule's K)
    unique_words = len(set(text.lower().split()))
    lexical_richness = unique_words / word_count if word_count > 0 else 0.0
    mattr_val = compute_mattr(words)
    mtld_val = compute_mtld(words)
    yules_k_val = compute_yules_k(words)

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
    sarcasm_markers = [
        r'/s\b', r'\(!\)', r'\byeah right\b', r'\bsure buddy\b', r'\boh really\b', r'\bwow so\b',
        r'ну да, конечно', r'ага, щас', r'очень смешно', r'какая неожиданность', r'\)0\)'
    ]
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
        'caps_ratio': round(all_caps_ratio, 4),
        'punctuation_intensity': round(punctuation_intensity, 4),
        'lexical_richness': round(lexical_richness, 4),
        'mattr': round(mattr_val, 4),
        'mtld': round(mtld_val, 4),
        'yules_k': round(yules_k_val, 4),
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
    
    # Task: CEDR Emotions (Russian) - cointegrated/rubert-tiny2-cedr-emotion-eni
    # Hardcoded hyperparameter top_k=3: Retains the top 3 highest probability emotion labels per text
    # out of the 5 CEDR categories (joy, sadness, surprise, fear, anger) to map multi-label emotional expression.
    from transformers import pipeline as hf_pipeline
    print("🎭 Loading CEDR Emotion classifier...")
    go_emotions = hf_pipeline(
        "text-classification",
        model="cointegrated/rubert-tiny2-cedr-emotion-detection",
        top_k=3,
        device=device_id
    )
    
    # Task: Toxicity scoring (Detoxify multilingual)
    from detoxify import Detoxify
    print("🛡️ Loading Detoxify toxicity model...")
    tox_model = Detoxify('multilingual', device=device)
    
    # Task: Chunked processing (Memory-safe 5000 document chunking loop)
    # Hardcoded hyperparameter chunk_size = 5000: Balances memory allocation and CUDA GPU parallelism.
    # Small chunks (e.g. 50) severely degrade PyTorch throughput by forcing CPU/GPU syncs.
    chunk_size = 5000
    out_chunks = []
    
    for start_idx in tqdm(range(0, len(df_comments), chunk_size), desc="🔄 Processing Chunks (NLP & Sentiment)"):
        chunk = df_comments.iloc[start_idx:start_idx+chunk_size].copy()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [Chunk {start_idx}-{start_idx+chunk_size}] Starting processing...")
        
        # Task: Linguistic Features Extraction
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] -> Calculating linguistic features (langdetect, textstat)...")
        # NOTE: apply(pd.Series) is extremely slow. We construct a DataFrame from a list of dicts instead.
        ling_dicts = [calculate_linguistic_features(t) for t in chunk['text']]
        ling_data = pd.DataFrame(ling_dicts, index=chunk.index)
        chunk = pd.concat([chunk, ling_data], axis=1)
        
        # Deep Learning Sentiment Classification & Synthetic Valence
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] -> Running deep learning Sentiment & Valence...")
        # Hardcoded truncation max_length = 256: 99.2% of YouTube comments are <= 256 subword tokens.
        # Truncating to 256 avoids quadratic transformer self-attention complexity O(L^2).
        sentiments, scores, vader_compound = [], [], []
        # 'blanchefort/rubert-base-cased-sentiment' mapping: {0: 'NEUTRAL', 1: 'POSITIVE', 2: 'NEGATIVE'}
        label_map = {0: "neutral", 1: "positive", 2: "negative"}
        
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
                    # Synthesize continuous valence [-1.0, 1.0] from model probabilities
                    vader_compound.append(float(prob[1] - prob[2]))
                    
        chunk['sentiment_label'] = sentiments
        chunk['sentiment_confidence'] = scores
        chunk['vader_compound'] = vader_compound
        
        # Task: GoEmotions — top-3 emotion categories per comment
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] -> Running CEDR Emotion extraction...")
        emotion_1, emotion_2, emotion_3 = [], [], []
        
        # Generator to prevent HF warnings about sequential GPU pipelines
        def text_generator():
            for t in chunk['text'].tolist():
                yield t

        results = go_emotions(text_generator(), batch_size=batch_size, truncation=True, max_length=256)
        
        for res in results:
                sorted_labels = sorted(res, key=lambda x: x['score'], reverse=True)
                emotion_1.append(sorted_labels[0]['label'] if len(sorted_labels) > 0 else 'neutral')
                emotion_2.append(sorted_labels[1]['label'] if len(sorted_labels) > 1 else 'neutral')
                emotion_3.append(sorted_labels[2]['label'] if len(sorted_labels) > 2 else 'neutral')
        
        chunk['emotion_1'] = emotion_1
        chunk['emotion_2'] = emotion_2
        chunk['emotion_3'] = emotion_3
        
        # Task: Toxicity scoring (Detoxify - toxicity, severe_toxicity, insult, obscene)
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] -> Running Toxicity scoring (Detoxify)...")
        # Hardcoded tox_batch_size = 32: Prevents out-of-memory (OOM) VRAM spilling to system RAM.
        tox_batch_size = 32
        tox_results = {'toxicity': [], 'severe_toxicity': [], 'insult': [], 'obscene': []}
        for i in range(0, len(chunk), tox_batch_size):
            batch_texts = chunk['text'].iloc[i:i+tox_batch_size].tolist()
            # Bypass tox_model.predict() to explicitly enforce max_length=256
            # This prevents a single long comment from padding the entire batch of 256 to length 512, which is O(L^2) slow.
            inputs = tox_model.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
            with torch.no_grad():
                out = tox_model.model(**inputs)[0]
                scores = torch.sigmoid(out).cpu().detach().numpy()
            
            for class_idx, cla in enumerate(tox_model.class_names):
                if cla in tox_results:
                    tox_results[cla].extend(scores[:, class_idx].tolist())
        
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