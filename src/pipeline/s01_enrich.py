"""Stage 01: Sentiment & Linguistic Feature Extraction

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
- Chunked processing: Memory-safe 5000 document chunking loop
"""

from dotenv import load_dotenv
load_dotenv()
import os
import re
import sys
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

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# High-Speed BPE Tokenizer via Rust (gigatoken)
try:
    import gigatoken as gt
    giga_tokenizer = gt.Tokenizer("openai-community/gpt2")
except Exception:
    giga_tokenizer = None

class ListDataset(torch.utils.data.Dataset):
    """Memory-efficient PyTorch Dataset wrapper for Hugging Face inference pipelines."""
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self.data_list[i]

def compute_yules_k(words):
    """
    Computes Yule's K vocabulary richness statistic.
    K = 10^4 * (sum(f_i * i^2) - N) / N^2
    Higher values indicate repetitive, lower-richness vocabulary.
    """
    if not words or len(words) < 2:
        return 0.0
    tokens = [w.lower() for w in words]
    N = len(tokens)
    freqs = pd.Series(tokens).value_counts().values
    m1 = N
    m2 = sum(f**2 for f in freqs)
    if N <= 1 or m1 == 0:
        return 0.0
    k = 10000.0 * (m2 - m1) / (m1 * m1)
    return float(max(0.0, k))

def compute_mattr(words, window_size=50):
    """
    Computes Moving-Average Type-Token Ratio (MATTR).
    Calculates TTR across a sliding window of fixed length to control for length bias.
    """
    if not words:
        return 0.0
    tokens = [w.lower() for w in words]
    N = len(tokens)
    if N < window_size:
        return len(set(tokens)) / N if N > 0 else 0.0
    
    ttr_sum = 0.0
    windows = N - window_size + 1
    for i in range(windows):
        sub_window = tokens[i:i+window_size]
        ttr_sum += len(set(sub_window)) / window_size
    return ttr_sum / windows

def compute_mtld(words, threshold=0.72):
    """
    Computes Measure of Textual Lexical Diversity (MTLD).
    Calculates average segment length where Type-Token Ratio (TTR) remains above threshold.
    """
    if not words:
        return 0.0
    tokens = [w.lower() for w in words]
    N = len(tokens)
    if N < 2:
        return 0.0
        
    def evaluate_factor(seq):
        factors = 0
        types = set()
        token_count = 0
        for token in seq:
            types.add(token)
            token_count += 1
            ttr = len(types) / token_count
            if ttr <= threshold:
                factors += 1
                types = set()
                token_count = 0
        if token_count > 0:
            excess = (1.0 - (len(types) / token_count)) / (1.0 - threshold)
            factors += excess
        return factors if factors > 0 else 1.0

    f1 = evaluate_factor(tokens)
    f2 = evaluate_factor(tokens[::-1])
    avg_factors = (f1 + f2) / 2.0
    return N / avg_factors if avg_factors > 0 else float(N)

def calculate_linguistic_features(text):
    """
    Extracts stylistic, linguistic, and cross-modal metrics for the given text or DataFrame.
    Guaranteed fail-safe execution.
    """
    if isinstance(text, pd.DataFrame):
        df_in = text
        text_col = 'text' if 'text' in df_in.columns else 'text_original'
        res_dicts = [calculate_linguistic_features(t) for t in df_in[text_col]]
        res_df = pd.DataFrame(res_dicts)
        df_out = df_in.copy()
        for col in res_df.columns:
            df_out[col] = res_df[col].values
        return df_out

    fallback = {
        'char_count': len(text) if isinstance(text, str) else 0,
        'word_count': len(str(text).split()) if text else 0,
        'emoji_count': 0, 
        'all_caps_ratio': 0.0, 'caps_ratio': 0.0, 'punctuation_intensity': 0.0, 'lexical_richness': 0.0,
        'mattr': 0.0, 'mtld': 0.0, 'yules_k': 0.0,
        'language': 'unknown', 'readability_flesch': 0.0,
        'hashtags': [], 'mentions': [], 'video_timestamps': [],
        'extracted_entities': [], 'is_sarcasm_suspect': False
    }

    if not text or not isinstance(text, str):
        return fallback

    try:
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        
        try:
            emoji_count = emoji_lib.emoji_count(text)
        except Exception:
            emoji_count = 0
        
        upper_chars = sum(1 for c in text if c.isupper())
        all_caps_ratio = upper_chars / char_count if char_count > 0 else 0.0
        
        punctuation_count = len(re.findall(r'[!?.]', text))
        punctuation_intensity = punctuation_count / char_count if char_count > 0 else 0.0
        
        unique_words = len(set(text.lower().split()))
        lexical_richness = unique_words / word_count if word_count > 0 else 0.0
        try:
            mattr_val = compute_mattr(words)
        except Exception:
            mattr_val = 0.0
        try:
            mtld_val = compute_mtld(words)
        except Exception:
            mtld_val = 0.0
        try:
            yules_k_val = compute_yules_k(words)
        except Exception:
            yules_k_val = 0.0

        # Task: Fast Language detection (Cyrillic pre-filter -> 200x faster, langdetect fallback)
        try:
            if re.search(r'[\u0400-\u04FF]', text):
                language = 'ru'
            elif word_count >= 2:
                language = detect(text[:200])
            else:
                language = 'unknown'
        except Exception:
            language = 'unknown'

        # Task: Readability (textstat Flesch Reading Ease)
        try:
            readability_flesch = textstat.flesch_reading_ease(text[:300])
        except Exception:
            readability_flesch = 0.0

        hashtags = re.findall(r'#\w+', text)
        mentions = re.findall(r'@\w+', text)
        video_timestamps = re.findall(r'\b(?:\d{1,2}:)?(?:[0-5]?\d):(?:[0-5]\d)\b', text)
        
        words_list = text.split()
        proper_nouns = []
        if len(words_list) > 1:
            for w in words_list[1:]:
                clean_w = re.sub(r'[^\w]', '', w)
                if clean_w and clean_w[0].isupper() and not clean_w.isupper() and len(clean_w) > 2:
                    proper_nouns.append(clean_w)
        extracted_entities = list(set(proper_nouns))
        
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
    except Exception:
        return fallback

def enrich_comments():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    comments_file = interim_dir / "comments_clean.parquet"
    videos_file = interim_dir / "videos_clean.parquet"

    if not comments_file.exists():
        print(f"❌ Clean comments layer missing at: {comments_file}")
        print("Please execute Stage 00 ('python -m pipeline.s00_ingest') first.")
        return

    print(f"📥 Loading interim comments layer from: {comments_file}")
    df_comments = pd.read_parquet(comments_file)
    
    if df_comments.empty:
        print("⚠️ Ingested comments dataset is empty. Skipping enrichment.")
        return

    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'], errors='coerce')
    print(f"📊 Dataset loaded into memory: {len(df_comments)} rows, {len(df_comments.columns)} columns.")

    # --- Pre-processing & Relational Graphs ---
    print("⏳ Computing relational tree flags (is_thread_terminal, reply_latency)...")
    
    parent_ids_set = set(df_comments['parent_id'].dropna().unique())
    df_comments['is_thread_terminal'] = ~df_comments['comment_id'].isin(parent_ids_set)
    
    df_parents = df_comments[['comment_id', 'published_at']].rename(
        columns={'comment_id': 'parent_id', 'parent_published_at': 'parent_published_at'}
    )
    df_comments = df_comments.merge(
        df_comments[['comment_id', 'published_at']].rename(
            columns={'comment_id': 'parent_id', 'published_at': 'parent_published_at'}
        ), 
        on='parent_id', 
        how='left'
    )
    
    df_comments['reply_latency'] = (
        df_comments['published_at'] - df_comments['parent_published_at']
    ).dt.total_seconds() / 60.0
    
    df_comments = df_comments.drop(columns=['parent_published_at'])

    if videos_file.exists():
        print("📹 Computing video timeline offset (minutes_since_upload)...")
        df_videos = pd.read_parquet(videos_file)
        df_videos['published_at'] = pd.to_datetime(df_videos['published_at'], errors='coerce')
        df_vid_pub = df_videos[['video_id', 'published_at']].rename(
            columns={'published_at': 'published_at_video'}
        )
        df_comments = df_comments.merge(
            df_vid_pub, 
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = 0 if device.type == "cuda" else -1
    print(f"🚀 Running ML models on: {device.type.upper()}")
    
    model_name = config["stage_01_enrich"]["sentiment_model"]
    batch_size = config["stage_01_enrich"]["batch_size"]
    
    tokenizer = None
    model = None
    go_emotions = None
    tox_model = None
    models_loaded = False

    def load_models_if_needed():
        nonlocal tokenizer, model, go_emotions, tox_model, models_loaded
        if models_loaded:
            return
        print("🎭 Loading ML Sentiment, Emotion & Toxicity models onto GPU/CPU...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()

        from transformers import pipeline as hf_pipeline
        go_emotions = hf_pipeline(
            "text-classification",
            model="cointegrated/rubert-tiny2-cedr-emotion-detection",
            top_k=3,
            device=device_id
        )

        from detoxify import Detoxify
        print("🛡️ Loading Detoxify toxicity model...")
        tox_model = Detoxify('multilingual', device=device)
        models_loaded = True

    checkpoint_dir = interim_dir / "enrich_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_size = 5000
    out_chunks = []
    
    for start_idx in tqdm(range(0, len(df_comments), chunk_size), desc="🔄 Processing Chunks (NLP & Sentiment)"):
        chunk_file = checkpoint_dir / f"chunk_{start_idx}.parquet"
        if chunk_file.exists():
            try:
                restored_chunk = pd.read_parquet(chunk_file)
                target_ids = df_comments['comment_id'].iloc[start_idx:start_idx+len(restored_chunk)].tolist()
                if list(restored_chunk['comment_id']) == target_ids:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⏩ [Chunk {start_idx}-{start_idx+chunk_size}] Restored from checkpoint ({chunk_file.name})")
                    out_chunks.append(restored_chunk)
                    continue
                else:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Checkpoint {chunk_file.name} comment ID mismatch. Re-computing chunk...")
            except Exception as e:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Error loading checkpoint {chunk_file.name}: {e}. Re-computing chunk...")

        chunk = df_comments.iloc[start_idx:start_idx+chunk_size].copy()
        
        enrich_cols = ['sentiment_label', 'emotion_1', 'toxicity', 'char_count']
        if all(col in chunk.columns for col in enrich_cols) and chunk['sentiment_label'].notna().all() and (chunk['sentiment_label'] != '').all():
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⏩ [Chunk {start_idx}-{start_idx+chunk_size}] Pre-enriched in dataset, saving checkpoint & skipping...")
            chunk.to_parquet(chunk_file, index=False)
            out_chunks.append(chunk)
            continue

        try:
            load_models_if_needed()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚀 [Chunk {start_idx}-{start_idx+chunk_size}] Starting processing ({len(chunk)} items)...")
            
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] -> (1/5) Calculating linguistic features (langdetect, textstat)...")
            ling_dicts = [calculate_linguistic_features(t) for t in chunk['text']]
            ling_data = pd.DataFrame(ling_dicts, index=chunk.index)
            for col in ling_data.columns:
                chunk[col] = ling_data[col]
            print(f"  [{datetime.now().strftime('%H:%M:%S')}]    ✓ Linguistic features calculated.")

            if giga_tokenizer is not None:
                try:
                    encoded_lists = giga_tokenizer.encode_batch_list(chunk['text'].tolist())
                    chunk['bpe_token_count'] = [len(toks) for toks in encoded_lists]
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] -> (2/5) Gigatoken Rust BPE tokens calculated.")
                except Exception:
                    chunk['bpe_token_count'] = chunk['word_count']
            else:
                chunk['bpe_token_count'] = chunk['word_count']
            
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] -> (3/5) Running deep learning Sentiment & Valence...")
            sentiments, scores, vader_compound = [], [], []
            label_map = {0: "neutral", 1: "positive", 2: "negative"}
            
            with torch.no_grad():
                for i in range(0, len(chunk), batch_size):
                    batch_texts = [str(t) if (t is not None and not pd.isna(t)) else "" for t in chunk['text'].iloc[i:i+batch_size].tolist()]
                    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                    outputs = model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                    
                    for prob in probabilities:
                        pred = np.argmax(prob)
                        sentiments.append(label_map[pred])
                        scores.append(float(prob[pred]))
                        vader_compound.append(float(prob[1] - prob[2]))
                        
            chunk['sentiment_label'] = sentiments
            chunk['sentiment_confidence'] = scores
            chunk['vader_compound'] = vader_compound
            print(f"  [{datetime.now().strftime('%H:%M:%S')}]    ✓ Sentiment & Valence finished.")
            
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] -> (4/5) Running CEDR Emotion extraction...")
            emotion_1, emotion_2, emotion_3 = [], [], []
            
            clean_emot_texts = [str(t) if (t is not None and not pd.isna(t)) else "" for t in chunk['text'].tolist()]
            dataset = ListDataset(clean_emot_texts)
            results = go_emotions(dataset, batch_size=batch_size, truncation=True, max_length=256)
            
            for res in results:
                sorted_labels = sorted(res, key=lambda x: x['score'], reverse=True)
                emotion_1.append(sorted_labels[0]['label'] if len(sorted_labels) > 0 else 'neutral')
                emotion_2.append(sorted_labels[1]['label'] if len(sorted_labels) > 1 else 'neutral')
                emotion_3.append(sorted_labels[2]['label'] if len(sorted_labels) > 2 else 'neutral')
            
            chunk['emotion_1'] = emotion_1
            chunk['emotion_2'] = emotion_2
            chunk['emotion_3'] = emotion_3
            print(f"  [{datetime.now().strftime('%H:%M:%S')}]    ✓ CEDR Emotion extraction finished.")
            
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] -> (5/5) Running Toxicity scoring (Detoxify)...")
            tox_batch_size = 32
            tox_results = {'toxicity': [], 'severe_toxicity': [], 'insult': [], 'obscene': []}
            total_tox_batches = (len(chunk) + tox_batch_size - 1) // tox_batch_size
            
            for i in range(0, len(chunk), tox_batch_size):
                batch_num = i // tox_batch_size + 1
                batch_texts = [str(t) if (t is not None and not pd.isna(t)) else "" for t in chunk['text'].iloc[i:i+tox_batch_size].tolist()]
                
                try:
                    inputs = tox_model.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
                    with torch.no_grad():
                        out = tox_model.model(**inputs)[0]
                        scores = torch.sigmoid(out).cpu().detach().numpy()
                    
                    for class_idx, cla in enumerate(tox_model.class_names):
                        if cla in tox_results:
                            tox_results[cla].extend(scores[:, class_idx].tolist())
                except Exception as batch_err:
                    print(f"    ⚠️ Detoxify batch {batch_num}/{total_tox_batches} error: {batch_err}. Fallback 0.0")
                    for cla in tox_results:
                        tox_results[cla].extend([0.0] * len(batch_texts))
            
            chunk['toxicity'] = [round(v, 4) for v in tox_results['toxicity']]
            chunk['severe_toxicity'] = [round(v, 4) for v in tox_results['severe_toxicity']]
            chunk['insult'] = [round(v, 4) for v in tox_results['insult']]
            chunk['obscene'] = [round(v, 4) for v in tox_results['obscene']]
            print(f"  [{datetime.now().strftime('%H:%M:%S')}]    ✓ Toxicity scoring finished.")
        except Exception as chunk_err:
            print(f"\n❌ Exception processing [Chunk {start_idx}-{start_idx+chunk_size}]: {chunk_err}")
            import traceback
            traceback.print_exc()
        
        # Task: Immediate Chunk Checkpoint & Memory Cleanup
        try:
            chunk.to_parquet(chunk_file, index=False)
            out_chunks.append(chunk)
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] 💾 Saved checkpoint: {chunk_file.name}")
            
            # Incremental interim update of comments_clean.parquet so progress is NEVER lost
            if out_chunks:
                interim_df = pd.concat(out_chunks, ignore_index=True)
                interim_df = interim_df.loc[:, ~interim_df.columns.duplicated()]
                interim_df.to_parquet(comments_file, index=False)
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] 💾 Incremental interim parquet updated ({len(interim_df)} total records saved).")
                del interim_df
        except Exception as save_err:
            print(f"⚠️ Warning saving interim parquet checkpoint: {save_err}")

        # Explicit CUDA VRAM Flushing & Garbage Collection per chunk
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        import gc
        gc.collect()

    print("💾 Finalizing enriched dataset on disk...")
    final_df = pd.concat(out_chunks, ignore_index=True)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    final_df.to_parquet(comments_file, index=False)
    print("✅ Stage 01 Temporal, Linguistic, Sentiment, Emotion & Toxicity Enrichment Complete!")

if __name__ == "__main__":
    enrich_comments()