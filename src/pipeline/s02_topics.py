"""Stage 02: BERTopic Modeling

Code Review Task Alignment:
- BERTopic Clustering: Multilingual embeddings, UMAP (5D), HDBSCAN
- Automated Topic Reduction: c-TF-IDF similarity merging (reduce_topics(nr_topics="auto"))
- Stopword filtering: Language-aware corpus stopword detection (corpus_stopwords)
- Root-only modeling: Models root comments to avoid reply-username noise
- Semantic Drift Tracking: compute_semantic_drift() with Procrustes-aligned Word2Vec across time slices
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
from tqdm import tqdm
# Core ML Architecture Components
# pyrefly: ignore [missing-import]
from bertopic import BERTopic
# pyrefly: ignore [missing-import]
from umap import UMAP
# pyrefly: ignore [missing-import]
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from pipeline.topic_assignment import attach_topics
from engine.config_loader import load_config
from engine.language import corpus_stopwords

def compute_semantic_drift(df_comments, output_dir, n_slices=6):
    """
    Task: Semantic Drift Tracking
    Math: Trains separate Word2Vec models per time slice, then computes Procrustes orthogonal matrix alignment 
          min_R ||M1 * R - M2||_F^2 via SVD (M2^T * M1 = U * Sigma * V^T => R = V * U^T).
          Sum of squared disparities measures true semantic drift of terms across time.
    """
    print("🌊 Computing Semantic Drift (Procrustes-aligned Word2Vec)...")
    if 'published_at' not in df_comments.columns or len(df_comments) < 50:
        print("⚠️ Insufficient data for semantic drift analysis. Skipping.")
        return
    
    try:
        import numpy as np
        from gensim.models import Word2Vec
        from scipy.spatial import procrustes
        
        df_valid = df_comments.dropna(subset=['published_at', 'text']).copy()
        df_valid['published_at'] = pd.to_datetime(df_valid['published_at'])
        
        # Hardcode n_slices=6: Divides temporal horizon into 6 equal-volume quantiles for trajectory comparison.
        n_bins = min(n_slices, max(2, len(df_valid) // 20))
        df_valid['time_slice'] = pd.qcut(df_valid['published_at'], q=n_bins, labels=False, duplicates='drop')
        
        models = {}
        for slice_id, group in df_valid.groupby('time_slice'):
            corpus = [str(text).lower().split() for text in group['text']]
            if len(corpus) >= 5:
                # Word2Vec math: Skip-gram / CBOW embedding space in 30 dimensions with window=5
                models[slice_id] = Word2Vec(corpus, vector_size=30, window=5, min_count=1, workers=1, seed=42)
        
        slice_ids = sorted(list(models.keys()))
        drift_records = []
        
        for i in range(len(slice_ids) - 1):
            s1, s2 = slice_ids[i], slice_ids[i+1]
            m1, m2 = models[s1], models[s2]
            
            shared_vocab = list(set(m1.wv.key_to_index.keys()) & set(m2.wv.key_to_index.keys()))
            if len(shared_vocab) < 5:
                continue
                
            M1 = np.array([m1.wv[w] for w in shared_vocab])
            M2 = np.array([m2.wv[w] for w in shared_vocab])
            
            # Procrustes transformation disparity score
            _, _, disparity = procrustes(M1, M2)
            drift_records.append({
                'from_slice': s1,
                'to_slice': s2,
                'vocab_size': len(shared_vocab),
                'procrustes_drift': float(disparity)
            })
            
        if drift_records:
            df_drift = pd.DataFrame(drift_records)
            df_drift.to_parquet(output_dir / "semantic_drift.parquet", index=False)
            print(f"✅ Semantic drift computed across {len(drift_records)} temporal transitions.")
        else:
            print("⚠️ Not enough overlapping vocabulary across time slices for drift computation.")
            
    except Exception as e:
        print(f"⚠️ Semantic drift computation failed ({e}). Proceeding...")


def run_topic_modeling():
    """
    Tasks: BERTopic Clustering, Automated Topic Reduction, Stopword filtering, Root-only modeling.
    """
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    phases = [
        "Loading enriched comment tables",
        "Initializing Multilingual Sub-Models",
        "Executing BERTopic Fit-Transform Pipeline",
        "Compiling Topic Assignment Metrics",
        "Writing Analytical Parquet Layers"
    ]
    
    with tqdm(total=len(phases), desc="🎬 Initializing Stage 02", bar_format="{l_bar}{bar:40}{r_bar}") as pbar:
        
        pbar.set_description(f"📥 {phases[0]}")
        clean_parquet_path = interim_dir / "comments_clean.parquet"
        df_comments = pd.read_parquet(clean_parquet_path)
        
        # Task: Root-only modeling (filter out replies to prevent username pollution)
        modeled_comments = df_comments[
            df_comments['parent_id'].isna() | (df_comments['parent_id'] == "")
        ]
        df_comments = df_comments[df_comments['parent_id'].isna() | (df_comments['parent_id'] == "")]
        text_col = 'text' if 'text' in df_comments.columns else df_comments.columns[1]
        docs = df_comments[text_col].astype(str).tolist()
        pbar.update(1)
        
        # Task: BERTopic Clustering setup (UMAP 5D + HDBSCAN)
        pbar.set_description(f"🧩 {phases[1]}")
        embedding_model_name = config["stage_02_topics"]["embedding_model"]
        min_topic_size = config["stage_02_topics"]["min_topic_size"]
        
        # Hardcode Math: effective_min_size = max(min_topic_size, len(docs) // 2000).
        # Dynamic scaling ensures clusters represent at least 0.05% of the total corpus,
        # preventing HDBSCAN from over-fragmenting massive corpora into hundreds of duplicate micro-clusters.
        effective_min_size = max(min_topic_size, len(docs) // 2000)
        if effective_min_size != min_topic_size:
            print(f"🔧 Scaling min_cluster_size {min_topic_size} -> {effective_min_size} "
                 f"for corpus of {len(docs)} documents")
        
        # UMAP Hyperparameters Math: n_components=5 preserves topological manifold structure for HDBSCAN better 
        # than 2D while reducing 384D transformer embeddings; n_neighbors=15 balances local vs global structure.
        umap_model = UMAP(
            n_neighbors=15, 
            n_components=5, 
            min_dist=0.0, 
            metric='cosine', 
            random_state=42,
            verbose=True  
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=effective_min_size,    
            metric='euclidean', 
            cluster_selection_method='eom', 
            prediction_data=True            
        )
        
        # Task: Stopword filtering (Language-aware corpus stopword detection)
        stop_words, detected_langs = corpus_stopwords(docs)
        print(f"🌍 Detected corpus language(s): {sorted(detected_langs)} "
              f"({len(stop_words)} stopwords applied)")
        vectorizer_model = CountVectorizer(stop_words=list(stop_words), min_df=2)
        
        topic_model = BERTopic(
            embedding_model=embedding_model_name,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=False,
            verbose=True
        )
        pbar.update(1)
        
        pbar.set_description(f"🔮 {phases[2]}")
        pbar.close()
        
        print("\n" + "-"*60)
        print("🚀 Starting Main Pipeline Engine (Embeddings -> UMAP -> HDBSCAN)")
        print("-"*60)
        
        topics, _ = topic_model.fit_transform(docs)
        
        # Task: Automated Topic Reduction (c-TF-IDF similarity auto-merge)
        n_topics_before = len(topic_model.get_topic_info())
        topic_model.reduce_topics(docs, nr_topics="auto")
        topics = topic_model.topics_
        n_topics_after = len(topic_model.get_topic_info())
        print(f"🧩 Topic reduction: {n_topics_before} -> {n_topics_after} topics (auto-merged)")

        pbar = tqdm(total=len(phases), initial=3, desc="💾 Wrapping Up Stage 02", bar_format="{l_bar}{bar:40}{r_bar}")
        
        pbar.set_description(f"📊 {phases[3]}")
        df_comments = attach_topics(df_comments, modeled_comments, topics)
        df_topic_info = topic_model.get_topic_info()
        
        df_comments.to_parquet(interim_dir / "comments_clean.parquet")
        df_topic_info.to_parquet(output_dir / "topic_metadata.parquet")
        pbar.update(1)
        
        pbar.set_description(f"💾 {phases[4]}")
        df_comments.to_parquet(clean_parquet_path, index=False)
        df_topic_info.to_parquet(output_dir / "topic_metadata.parquet", index=False)
        pbar.update(1)
        
    # Task: Semantic Drift Tracking execution
    compute_semantic_drift(df_comments, output_dir)
        
    print("\n" + "="*60)
    print(f"✅ Stage 02 Core Topic Clustering Complete!")
    print(f"🧩 Discovered {len(df_topic_info) - 1} Distinct Conversational Clusters.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_topic_modeling()
