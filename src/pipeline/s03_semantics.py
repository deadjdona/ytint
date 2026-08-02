import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from transformers import pipeline
import warnings
from engine.config_loader import load_config

# Suppress verbose UMAP warnings for cleaner CLI logs
warnings.filterwarnings('ignore', category=UserWarning)

def model_semantics():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    comments_file = interim_dir / "comments_clean.parquet"
    semantic_file = interim_dir / "semantic_topics.parquet"
    embeddings_file = interim_dir / "embeddings_cache.npy"
    
    if not comments_file.exists():
        print(f"❌ Clean comments file not found at {comments_file}")
        return

    print("📥 Loading corpus for Semantic Modeling...")
    df_comments = pd.read_parquet(comments_file)
    df_comments['text'] = df_comments['text'].fillna("").astype(str)
    
    # 🛡️ Bullet-Proofing: Filter extremely short comments for BERTopic to avoid sparse matrix degradation
    # We will compute embeddings for all, but only use valid texts for topic model fitting
    def is_valid_for_topic(text):
        words = text.split()
        return len(words) >= 3
        
    df_comments['is_topic_valid'] = df_comments['text'].apply(is_valid_for_topic)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running ML models on: {device.upper()}")
    
    # --- 1. Embeddings ---
    print("🧠 Computing Document Embeddings (sentence-transformers)...")
    if embeddings_file.exists():
        print(f"💾 Found cached embeddings at {embeddings_file}. Loading from disk...")
        embeddings = np.load(embeddings_file)
    else:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        embeddings = embedding_model.encode(
            df_comments['text'].tolist(), 
            show_progress_bar=True,
            batch_size=128
        )
        np.save(embeddings_file, embeddings)
        print("💾 Cached embeddings to disk.")

    # --- 2. UMAP Dimensionality Reduction ---
    print("🌌 Running UMAP for 2D Dimensionality Reduction (Visualization)...")
    umap_model_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    
    # 🛡️ Bullet-Proofing: Prevent OOM on UMAP by fitting on a sample if dataset is massive
    if len(embeddings) > 50000:
        print("⚠️ Large dataset detected. Fitting UMAP on a 50k sample to prevent OOM, then transforming the rest...")
        np.random.seed(42)
        sample_indices = np.random.choice(len(embeddings), 50000, replace=False)
        umap_model_2d.fit(embeddings[sample_indices])
        umap_embeddings = umap_model_2d.transform(embeddings)
    else:
        umap_embeddings = umap_model_2d.fit_transform(embeddings)
        
    df_comments['umap_x'] = umap_embeddings[:, 0]
    df_comments['umap_y'] = umap_embeddings[:, 1]
    
    # --- 3. BERTopic ---
    print("📚 Running BERTopic for Semantic Clustering...")
    
    valid_indices = df_comments[df_comments['is_topic_valid']].index
    valid_texts = df_comments.loc[valid_indices, 'text'].tolist()
    valid_embeddings = embeddings[valid_indices]
    
    # We use a 5-component UMAP strictly for BERTopic's internal HDBSCAN density clustering
    topic_umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    
    topic_model = BERTopic(
        embedding_model=None, # Already computed
        umap_model=topic_umap,
        calculate_probabilities=False
    )
    
    topics, _ = topic_model.fit_transform(valid_texts, valid_embeddings)
    
    # Default invalid comments to -1 (Outlier/Noise class)
    df_comments['topic_id'] = -1
    df_comments.loc[valid_indices, 'topic_id'] = topics
    
    # --- 4. Zero-Shot Intent Taxonomy ---
    print("🎯 Categorizing Intent via Zero-Shot Classification (on Root Comments only)...")
    
    # Only classify root comments to isolate intent extraction to thread initiators
    is_root = pd.isna(df_comments.get('parent_id')) | (df_comments.get('parent_id') == "")
    root_indices = df_comments[is_root].index
    root_texts = df_comments.loc[root_indices, 'text'].tolist()
    
    # Only run zero-shot if we actually have root texts
    if len(root_texts) > 0:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device == "cuda" else -1)
        candidate_labels = ["praise", "complaint", "question", "suggestion", "spam", "discussion"]
        
        intents = []
        batch_size = 64
        for i in tqdm(range(0, len(root_texts), batch_size), desc="Zero-Shot Batch"):
            batch = root_texts[i:i+batch_size]
            results = classifier(batch, candidate_labels, multi_label=False)
            for res in results:
                intents.append(res['labels'][0])
                
        df_comments['intent_label'] = "reply"
        df_comments.loc[root_indices, 'intent_label'] = intents
    else:
        df_comments['intent_label'] = "unknown"
    
    # --- 5. Consistency & I/O ---
    print(f"💾 Saving Semantic Metrics to {semantic_file.name}...")
    
    semantic_columns = ['comment_id', 'topic_id', 'intent_label', 'umap_x', 'umap_y']
    df_semantic = df_comments[semantic_columns].copy()
    
    df_semantic.to_parquet(semantic_file, index=False)
    print("✅ Stage 03 Topic & Semantic Modeling Complete!")

if __name__ == "__main__":
    model_semantics()
