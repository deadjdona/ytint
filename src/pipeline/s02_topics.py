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

import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
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


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f}ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes}m {rem_seconds:.1f}s"


def _format_size(num_bytes: int) -> str:
    """Format byte count into a human-readable size string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def _print_phase_header(phase_idx: int, total_phases: int, title: str, icon: str = "📌"):
    """Render a structured phase banner."""
    print(f"\n{'='*72}")
    print(f" {icon} [Phase {phase_idx}/{total_phases}] {title}")
    print(f"{'='*72}")


def compute_semantic_drift(df_comments, output_dir, n_slices=6):
    """
    Task: Semantic Drift Tracking
    Math: Trains separate Word2Vec models per time slice, then computes Procrustes orthogonal matrix alignment 
          min_R ||M1 * R - M2||_F^2 via SVD (M2^T * M1 = U * Sigma * V^T => R = V * U^T).
          Sum of squared disparities measures true semantic drift of terms across time.
    """
    t_drift_start = time.perf_counter()
    print("\n" + "-"*72)
    print("🌊 [Sub-Task] Temporal Semantic Drift Tracking (Procrustes Alignment)")
    print("-" * 72)
    
    if 'published_at' not in df_comments.columns or len(df_comments) < 50:
        print("  ⚠️ Insufficient data for semantic drift analysis (<50 comments or missing 'published_at'). Skipping.")
        return
    
    try:
        import numpy as np
        from gensim.models import Word2Vec
        from scipy.spatial import procrustes
        
        df_valid = df_comments.dropna(subset=['published_at', 'text']).copy()
        df_valid['published_at'] = pd.to_datetime(df_valid['published_at'])
        
        min_date = df_valid['published_at'].min()
        max_date = df_valid['published_at'].max()
        print(f"  📅 Temporal Horizon: {min_date.strftime('%Y-%m-%d %H:%M')} -> {max_date.strftime('%Y-%m-%d %H:%M')} ({len(df_valid):,} valid comments)")
        
        # Hardcode n_slices=6: Divides temporal horizon into 6 equal-volume quantiles for trajectory comparison.
        n_bins = min(n_slices, max(2, len(df_valid) // 20))
        df_valid['time_slice'] = pd.qcut(df_valid['published_at'], q=n_bins, labels=False, duplicates='drop')
        
        slice_groups = list(df_valid.groupby('time_slice'))
        print(f"  🔪 Partitioned into {len(slice_groups)} equal-volume temporal quantiles:")
        for slice_id, group in slice_groups:
            s_start = group['published_at'].min().strftime('%Y-%m-%d')
            s_end = group['published_at'].max().strftime('%Y-%m-%d')
            print(f"     • Slice {slice_id}: {s_start} to {s_end} ({len(group):,} comments)")
            
        models = {}
        print("\n  🧠 Training Word2Vec embeddings per time slice (vector_size=30, window=5)...")
        for slice_id, group in tqdm(slice_groups, desc="     Training Word2Vec slices", unit="slice", leave=False):
            corpus = [str(text).lower().split() for text in group['text']]
            if len(corpus) >= 5:
                # Word2Vec math: Skip-gram / CBOW embedding space in 30 dimensions with window=5
                models[slice_id] = Word2Vec(corpus, vector_size=30, window=5, min_count=1, workers=1, seed=42)
        
        slice_ids = sorted(list(models.keys()))
        drift_records = []
        
        print(f"  📐 Aligning vector spaces across {len(slice_ids) - 1} transitions via orthogonal Procrustes...")
        transition_indices = range(len(slice_ids) - 1)
        for i in tqdm(transition_indices, desc="     Aligning slice transitions", unit="transition", leave=False):
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
            out_file = output_dir / "semantic_drift.parquet"
            df_drift.to_parquet(out_file, index=False)
            
            print(f"\n  📊 Semantic Drift Trajectory Metrics:")
            print("  " + "─"*60)
            print(f"   {'Transition':<16} │ {'Shared Vocab':<15} │ {'Procrustes Drift':<18}")
            print("  " + "─"*60)
            for rec in drift_records:
                trans_label = f"Slice {rec['from_slice']} -> {rec['to_slice']}"
                print(f"   {trans_label:<16} │ {rec['vocab_size']:<15,} │ {rec['procrustes_drift']:<18.4f}")
            print("  " + "─"*60)
            
            drift_duration = time.perf_counter() - t_drift_start
            print(f"  ✅ Semantic drift saved to {out_file.name} ({len(drift_records)} transitions, {_format_duration(drift_duration)}).")
        else:
            print("  ⚠️ Not enough overlapping vocabulary across time slices for drift computation.")
            
    except Exception as e:
        print(f"  ⚠️ Semantic drift computation failed ({e}). Proceeding...")


def run_topic_modeling():
    """
    Tasks: BERTopic Clustering, Automated Topic Reduction, Stopword filtering, Root-only modeling.
    """
    t_stage_start = time.perf_counter()
    TOTAL_PHASES = 6
    
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_parquet_path = interim_dir / "comments_clean.parquet"
    topic_meta_path = output_dir / "topic_metadata.parquet"
    cached_embeddings_path = interim_dir / "topic_embeddings.npy"
    
    print("\n" + "╔" + "═"*70 + "╗")
    print("║  🎬 STAGE 02: BERTopic Semantic Clustering & Topic Modeling         ║")
    print("╚" + "═"*70 + "╝")
    
    # -------------------------------------------------------------------------
    # Phase 1: Data Ingestion & Root Comment Filtering
    # -------------------------------------------------------------------------
    t_phase_start = time.perf_counter()
    _print_phase_header(1, TOTAL_PHASES, "Ingesting & Filtering Root Comments", icon="📥")
    
    if not clean_parquet_path.exists():
        raise FileNotFoundError(f"Missing required input dataset: {clean_parquet_path}")
        
    file_size_mb = clean_parquet_path.stat().st_size / (1024 * 1024)
    print(f"  • Source File: {clean_parquet_path.name} ({file_size_mb:.2f} MB)")
    
    df_all_comments = pd.read_parquet(clean_parquet_path)
    t_load = time.perf_counter() - t_phase_start
    print(f"  • Loaded {len(df_all_comments):,} total comments in {_format_duration(t_load)}.")
    
    # Task: Root-only modeling (filter out replies to prevent username pollution)
    modeled_comments = df_all_comments[
        df_all_comments['parent_id'].isna() | (df_all_comments['parent_id'] == "")
    ]
    text_col = 'text' if 'text' in df_all_comments.columns else df_all_comments.columns[1]
    docs = modeled_comments[text_col].astype(str).tolist()
    
    n_total = len(df_all_comments)
    n_root = len(docs)
    n_replies = n_total - n_root
    pct_root = (n_root / n_total * 100.0) if n_total > 0 else 0.0
    pct_replies = (n_replies / n_total * 100.0) if n_total > 0 else 0.0
    
    print(f"  • Segmentation Breakdown:")
    print(f"     - Root Comments (Modeled):  {n_root:>9,} ({pct_root:>5.1f}%)")
    print(f"     - Reply Comments (Retained): {n_replies:>9,} ({pct_replies:>5.1f}%)")
    
    # Corpus length diagnostics
    if docs:
        avg_chars = sum(len(d) for d in docs) / len(docs)
        avg_words = sum(len(d.split()) for d in docs) / len(docs)
        print(f"  • Document Text Stats: avg {avg_chars:.1f} chars/doc, avg {avg_words:.1f} words/doc")
        
    print(f"  ⏱️ Phase 1 Complete in {_format_duration(time.perf_counter() - t_phase_start)}.")
    
    # -------------------------------------------------------------------------
    # Phase 2: Language & Stopwords Profiling
    # -------------------------------------------------------------------------
    t_phase_start = time.perf_counter()
    _print_phase_header(2, TOTAL_PHASES, "Multilingual Language & Stopwords Detection", icon="🌍")
    
    # Task: Stopword filtering (Language-aware corpus stopword detection)
    stop_words, detected_langs = corpus_stopwords(docs)
    sorted_langs = sorted(detected_langs)
    print(f"  • Detected Language(s): {', '.join(sorted_langs) if sorted_langs else 'universal'}")
    print(f"  • Stopwords Filter:     {len(stop_words):,} terms loaded into vocabulary filter")
    
    # Show small sample of detected stopwords
    sample_sw = list(stop_words)[:8]
    if sample_sw:
        print(f"  • Stopwords Sample:     {', '.join(sample_sw)}...")
        
    vectorizer_model = CountVectorizer(stop_words=list(stop_words), min_df=2)
    print(f"  • CountVectorizer:      min_df=2, ngram_range=(1, 1)")
    print(f"  ⏱️ Phase 2 Complete in {_format_duration(time.perf_counter() - t_phase_start)}.")
    
    # -------------------------------------------------------------------------
    # Phase 3: SentenceTransformer Embeddings
    # -------------------------------------------------------------------------
    t_phase_start = time.perf_counter()
    _print_phase_header(3, TOTAL_PHASES, "Generating / Loading Vector Embeddings", icon="🧠")
    
    embedding_model_name = config["stage_02_topics"]["embedding_model"]
    embeddings = None
    
    if cached_embeddings_path.exists():
        try:
            cached_emb = np.load(cached_embeddings_path)
            if len(cached_emb) == len(docs):
                embeddings = cached_emb
                cache_size_mb = cached_embeddings_path.stat().st_size / (1024 * 1024)
                print(f"  ⚡ Cache Hit! Loaded {len(embeddings):,} embeddings from {cached_embeddings_path.name}")
                print(f"  • Matrix Shape: {embeddings.shape} ({cache_size_mb:.2f} MB)")
            else:
                print(f"  ⚠️ Cache size mismatch ({len(cached_emb):,} cached vs {len(docs):,} docs). Recomputing...")
        except Exception as e:
            print(f"  ⚠️ Cache read error: {e}. Recomputing...")

    if embeddings is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_label = torch.cuda.get_device_name(0) if device == "cuda" else f"CPU ({torch.get_num_threads()} threads)"
        batch_size = 256
        n_batches = (len(docs) + batch_size - 1) // batch_size
        
        print(f"  • Model Architecture: {embedding_model_name}")
        print(f"  • Compute Target:     {device.upper()} [{device_label}]")
        print(f"  • Batch Dimension:    {batch_size} documents/batch ({n_batches} total batches)")
        print(f"  • Encoding {len(docs):,} root comments (progress bar below):")
        
        t_enc_start = time.perf_counter()
        st_model = SentenceTransformer(embedding_model_name, device=device)
        embeddings = st_model.encode(docs, show_progress_bar=True, batch_size=batch_size)
        t_enc_elapsed = time.perf_counter() - t_enc_start
        
        docs_per_sec = len(docs) / max(0.001, t_enc_elapsed)
        print(f"  ✨ Encoding Complete: {len(embeddings):,} vectors ({embeddings.shape[1]}D) in {_format_duration(t_enc_elapsed)} ({docs_per_sec:.1f} docs/sec)")
        
        try:
            np.save(cached_embeddings_path, embeddings)
            cache_size_mb = cached_embeddings_path.stat().st_size / (1024 * 1024)
            print(f"  💾 Cached embeddings to {cached_embeddings_path.name} ({cache_size_mb:.2f} MB)")
        except Exception as e:
            print(f"  ⚠️ Could not cache embeddings: {e}")
            
    print(f"  ⏱️ Phase 3 Complete in {_format_duration(time.perf_counter() - t_phase_start)}.")
    
    # -------------------------------------------------------------------------
    # Phase 4: BERTopic Dimensionality Reduction & Clustering
    # -------------------------------------------------------------------------
    t_phase_start = time.perf_counter()
    _print_phase_header(4, TOTAL_PHASES, "Executing BERTopic (UMAP 5D + HDBSCAN + c-TF-IDF)", icon="🚀")
    
    min_topic_size = config["stage_02_topics"]["min_topic_size"]
    # Hardcode Math: effective_min_size = max(min_topic_size, len(docs) // 2000).
    effective_min_size = max(min_topic_size, len(docs) // 2000)
    
    umap_epochs = 150 if len(docs) > 100000 else 200
    print(f"  • Hyperparameter Manifest:")
    print(f"     - UMAP:    n_neighbors=15, n_components=5, metric='cosine', min_dist=0.0, epochs={umap_epochs}, n_jobs=-1")
    print(f"     - HDBSCAN: min_cluster_size={effective_min_size}" + (f" (scaled from {min_topic_size})" if effective_min_size != min_topic_size else "") + ", metric='euclidean', selection='eom'")
    print(f"     - c-TF-IDF: CountVectorizer(min_df=2, stopwords={len(stop_words)})")
    
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine', 
        n_epochs=umap_epochs,
        n_jobs=-1,
        verbose=True  
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=effective_min_size,    
        metric='euclidean', 
        cluster_selection_method='eom', 
        core_dist_n_jobs=-1,
        prediction_data=True            
    )
    
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True
    )
    
    print("\n  ⚙️ Fitting UMAP manifold and HDBSCAN density clusters...")
    t_fit_start = time.perf_counter()
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    t_fit_elapsed = time.perf_counter() - t_fit_start
    
    # Cluster discovery metrics
    initial_topic_info = topic_model.get_topic_info()
    n_raw_topics = len(initial_topic_info) - 1 # exclude -1
    outlier_count = (np.array(topics) == -1).sum()
    clustered_count = len(docs) - outlier_count
    pct_clustered = (clustered_count / len(docs) * 100.0) if len(docs) > 0 else 0.0
    pct_outlier = (outlier_count / len(docs) * 100.0) if len(docs) > 0 else 0.0
    
    print(f"\n  🎯 Initial Cluster Discovery ({_format_duration(t_fit_elapsed)}):")
    print(f"     - Raw Clusters Discovered: {n_raw_topics:,} topics")
    print(f"     - Clustered Comments:       {clustered_count:,} ({pct_clustered:.1f}%)")
    print(f"     - Outliers / Noise (-1):    {outlier_count:,} ({pct_outlier:.1f}%)")
    print(f"  ⏱️ Phase 4 Complete in {_format_duration(time.perf_counter() - t_phase_start)}.")
    
    # -------------------------------------------------------------------------
    # Phase 5: Automated Topic Reduction
    # -------------------------------------------------------------------------
    t_phase_start = time.perf_counter()
    _print_phase_header(5, TOTAL_PHASES, "Automated Topic Reduction & Top Topic Synthesis", icon="🧩")
    
    n_topics_before = len(topic_model.get_topic_info())
    print(f"  • Initiating c-TF-IDF similarity merging (nr_topics='auto') across {n_topics_before} initial partitions...")
    
    t_red_start = time.perf_counter()
    topic_model.reduce_topics(docs, nr_topics="auto")
    topics = topic_model.topics_
    t_red_elapsed = time.perf_counter() - t_red_start
    
    df_topic_info = topic_model.get_topic_info()
    n_topics_after = len(df_topic_info)
    merge_reduction_pct = ((n_topics_before - n_topics_after) / max(1, n_topics_before)) * 100.0
    
    print(f"  ✨ Topic Reduction Complete in {_format_duration(t_red_elapsed)}:")
    print(f"     - Hierarchy Compression: {n_topics_before} -> {n_topics_after} topics ({merge_reduction_pct:.1f}% reduction)")
    print(f"     - Distinct Thematic Clusters: {n_topics_after - 1:,} (excluding outlier bin)")
    
    # Render formatted Top Topics Preview Table
    print("\n  📊 Top Discovered Conversational Topics (by document volume):")
    print("  " + "─"*72)
    print(f"   {'Rank':<5} │ {'Topic ID':<10} │ {'Documents':<11} │ {'% Corpus':<10} │ Top Representative Terms")
    print("  " + "─"*72)
    
    # Show top 10 topics sorted by Count descending (excluding -1 if possible or showing it clearly)
    preview_df = df_topic_info[df_topic_info['Topic'] != -1].sort_values(by='Count', ascending=False).head(10)
    for rank, (_, row) in enumerate(preview_df.iterrows(), 1):
        tid = row['Topic']
        cnt = row['Count']
        pct = (cnt / len(docs) * 100.0) if len(docs) > 0 else 0.0
        # Extract representative terms from Name or Representation
        if 'Representation' in row and isinstance(row['Representation'], list):
            rep_str = ", ".join(row['Representation'][:5])
        elif '_' in str(row['Name']):
            rep_str = ", ".join(str(row['Name']).split('_')[1:6])
        else:
            rep_str = str(row['Name'])
        print(f"   {rank:<5} │ Topic {tid:<4} │ {cnt:<11,} │ {pct:>7.1f}%   │ {rep_str}")
    print("  " + "─"*72)
    
    print(f"  ⏱️ Phase 5 Complete in {_format_duration(time.perf_counter() - t_phase_start)}.")
    
    # -------------------------------------------------------------------------
    # Phase 6: Topic Assignment, Corpus Integration & Parquet Serialization
    # -------------------------------------------------------------------------
    t_phase_start = time.perf_counter()
    _print_phase_header(6, TOTAL_PHASES, "Corpus Integration & Parquet Serialization", icon="💾")
    
    print("  • Attaching topic assignments to complete comments corpus (preserving replies)...")
    df_all_comments = attach_topics(df_all_comments, modeled_comments, topics)
    
    # Corpus stats
    assigned_mask = (df_all_comments['topic'] >= 0)
    assigned_count = assigned_mask.sum()
    root_outlier_count = ((df_all_comments['parent_id'].isna() | (df_all_comments['parent_id'] == "")) & (df_all_comments['topic'] == -1)).sum()
    reply_retained_count = (~(df_all_comments['parent_id'].isna() | (df_all_comments['parent_id'] == ""))).sum()
    
    print(f"  • Final Topic Distribution across All Comments ({len(df_all_comments):,} total):")
    print(f"     - Assigned to Topics:        {assigned_count:>9,} ({assigned_count / len(df_all_comments) * 100.0:>5.1f}%)")
    print(f"     - Root Outliers (Topic -1):  {root_outlier_count:>9,} ({root_outlier_count / len(df_all_comments) * 100.0:>5.1f}%)")
    print(f"     - Reply Comments (Topic -1): {reply_retained_count:>9,} ({reply_retained_count / len(df_all_comments) * 100.0:>5.1f}%)")
    
    print(f"\n  • Writing Parquet Artifacts atomically:")
    
    # Persist comments_clean.parquet
    t_write = time.perf_counter()
    temp_clean = clean_parquet_path.with_suffix(".tmp.parquet")
    df_all_comments.to_parquet(temp_clean, engine="pyarrow", compression="snappy", index=False)
    if temp_clean.exists():
        temp_clean.replace(clean_parquet_path)
    clean_size_mb = clean_parquet_path.stat().st_size / (1024 * 1024)
    print(f"     ✅ {clean_parquet_path.name} -> {len(df_all_comments):,} records ({clean_size_mb:.2f} MB, {_format_duration(time.perf_counter() - t_write)})")
    
    # Persist topic_metadata.parquet
    t_write = time.perf_counter()
    temp_topic = topic_meta_path.with_suffix(".tmp.parquet")
    df_topic_info.to_parquet(temp_topic, engine="pyarrow", compression="snappy", index=False)
    if temp_topic.exists():
        temp_topic.replace(topic_meta_path)
    meta_size_kb = topic_meta_path.stat().st_size / 1024
    print(f"     ✅ {topic_meta_path.name} -> {len(df_topic_info):,} topics ({meta_size_kb:.1f} KB, {_format_duration(time.perf_counter() - t_write)})")
    
    print(f"  ⏱️ Phase 6 Complete in {_format_duration(time.perf_counter() - t_phase_start)}.")
    
    # -------------------------------------------------------------------------
    # Sub-Task: Semantic Drift Tracking execution
    # -------------------------------------------------------------------------
    compute_semantic_drift(df_all_comments, output_dir)
        
    t_total_elapsed = time.perf_counter() - t_stage_start
    print("\n" + "═"*72)
    print(" 🎉 STAGE 02 COMPLETE: BERTopic Semantic Modeling Finished")
    print(f" ⏱️ Total Runtime: {_format_duration(t_total_elapsed)}")
    print(f" 🧩 Discovered {len(df_topic_info) - 1:,} Distinct Conversational Clusters")
    print(f" 📊 Processed {len(df_all_comments):,} Comments ({len(docs):,} Root Modeled)")
    print("═"*72 + "\n")


if __name__ == "__main__":
    run_topic_modeling()
