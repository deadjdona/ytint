import yaml
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
import sys
# Make 'engine' importable whether this runs standalone or via runner.py
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))
from engine.language import corpus_stopwords

def load_config():
    """
    Dynamically resolves the project root directory and loads the unified settings.
    Ensures absolute path compatibility across all execution environments.
    """
    # 1. Locate the file system context of the running script
    current_file = Path(__file__).resolve()
    
    # 2. Walk upward until we locate the parent directory containing the 'config' folder
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
        
    # 3. Dynamic Absolute Translation Layer
    # Automatically convert configured paths to absolute system paths relative to the project root
    config["paths"]["raw_db"] = str(root_dir / config["paths"]["raw_db"])
    config["paths"]["interim_dir"] = str(root_dir / config["paths"]["interim_dir"])
    config["paths"]["output_dir"] = str(root_dir / config["paths"]["output_dir"])
    
    return config

def run_topic_modeling():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step-by-step master phases for the tracking engine
    phases = [
        "Loading enriched comment tables",
        "Initializing Multilingual Sub-Models",
        "Executing BERTopic Fit-Transform Pipeline",
        "Compiling Topic Assignment Metrics",
        "Writing Analytical Parquet Layers"
    ]
    
    with tqdm(total=len(phases), desc="🎬 Initializing Stage 02", bar_format="{l_bar}{bar:40}{r_bar}") as pbar:
        
        # Phase 1: Load Data
        pbar.set_description(f"📥 {phases[0]}")
        clean_parquet_path = interim_dir / "comments_clean.parquet"
        df_comments = pd.read_parquet(clean_parquet_path)
        
        # Model only top-level comments to avoid reply-context noise, but retain
        # the complete canonical comments layer when serializing results.
        modeled_comments = df_comments[
            df_comments['parent_id'].isna() | (df_comments['parent_id'] == "")
        ]
        # NEW: Filter out replies to stop username pollution in BERTopic
        df_comments = df_comments[df_comments['parent_id'].isna() | (df_comments['parent_id'] == "")]
        text_col = 'text' if 'text' in df_comments.columns else df_comments.columns[1]
        docs = df_comments[text_col].astype(str).tolist()
        pbar.update(1)
        
        # Phase 2: Initialize ML Components with Native Progress Overrides
        pbar.set_description(f"🧩 {phases[1]}")
        embedding_model_name = config["stage_02_topics"]["embedding_model"]
        min_topic_size = config["stage_02_topics"]["min_topic_size"]
        
        # min_topic_size alone as a flat HDBSCAN min_cluster_size massively
        # over-fragments large corpora: it pulls out every tight near-duplicate
        # micro-cluster (repeated emoji spam, templated reactions) as its own
        # "topic". Validated empirically: min_cluster_size=15 on a 188k-doc
        # corpus with realistic spam contamination produced 316 spurious
        # clusters vs 40 true topics; min_cluster_size~=100 recovered ~30-46.
        # Treat the configured value as a floor and scale it with corpus size.
        effective_min_size = max(min_topic_size, len(docs) // 2000)
        if effective_min_size != min_topic_size:
            print(f"🔧 Scaling min_cluster_size {min_topic_size} -> {effective_min_size} "
                 f"for corpus of {len(docs)} documents")
        
        # Keep verbose=True here—this gives us the great UMAP epoch progress bar!
        umap_model = UMAP(
            n_neighbors=15, 
            n_components=5, 
            min_dist=0.0, 
            metric='cosine', 
            random_state=42,
            verbose=True  
        )
        
        # Removed verbose=True from here to fix the scikit-learn KDTree initialization crash
        hdbscan_model = HDBSCAN(
            min_cluster_size=effective_min_size,    
            metric='euclidean', 
            cluster_selection_method='eom', 
            prediction_data=True            
        )
        
        # Detect corpus language(s) and build a matching stopword set —
        # hardcoding "english" silently disables stopword filtering on
        # non-English corpora (e.g. Russian/Ukrainian comments), polluting
        # every c-TF-IDF topic label with function words.
        stop_words, detected_langs = corpus_stopwords(docs)
        print(f"🌍 Detected corpus language(s): {sorted(detected_langs)} "
              f"({len(stop_words)} stopwords applied)")
        vectorizer_model = CountVectorizer(stop_words=list(stop_words), min_df=2)
        
        # Build unified architecture
        topic_model = BERTopic(
            embedding_model=embedding_model_name,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=False,
            verbose=True  # Keeps BERTopic's main status printouts active
        )
        pbar.update(1)
        
        # Phase 3: Execute Monolithic Fit Transform
        pbar.set_description(f"🔮 {phases[2]}")
        
        # Temporarily pause master progress bar layout so internal ML loops print cleanly
        pbar.close()
        
        print("\n" + "-"*60)
        print("🚀 Starting Main Pipeline Engine (Embeddings -> UMAP -> HDBSCAN)")
        print("-"*60)
        
        topics, _ = topic_model.fit_transform(docs)
        
        # Safety net: even with a scaled min_cluster_size, spam/template
        # clusters and semantically near-identical topics commonly survive
        # HDBSCAN as separate topics. Merge via c-TF-IDF similarity instead
        # of trusting one clustering pass to land on an interpretable count.
        n_topics_before = len(topic_model.get_topic_info())
        topic_model.reduce_topics(docs, nr_topics="auto")
        topics = topic_model.topics_
        n_topics_after = len(topic_model.get_topic_info())
        print(f"🧩 Topic reduction: {n_topics_before} -> {n_topics_after} topics (auto-merged)")

        # Reinitialize master phase tracker for final file serialization tasks
        pbar = tqdm(total=len(phases), initial=3, desc="💾 Wrapping Up Stage 02", bar_format="{l_bar}{bar:40}{r_bar}")
        
        # Phase 4: Compile Assignment Metrics
        pbar.set_description(f"📊 {phases[3]}")
        df_comments = attach_topics(df_comments, modeled_comments, topics)
        df_topic_info = topic_model.get_topic_info()
        
        # VERY IMPORTANT: Save the row-level topic assignments back to disk
        # so s04_synthesis.py can read them!
        df_comments.to_parquet(interim_dir / "comments_clean.parquet")
        df_topic_info.to_parquet(output_dir / "topic_metadata.parquet")
        
        pbar.update(1)
        
        # Phase 5: Overwrite Clean Parquet with new Topic Matrix Features
        pbar.set_description(f"💾 {phases[4]}")
        df_comments.to_parquet(clean_parquet_path, index=False)
        df_topic_info.to_parquet(output_dir / "topic_metadata.parquet", index=False)
        pbar.update(1)
        
    print("\n" + "="*60)
    print(f"✅ Stage 02 Core Topic Clustering Complete!")
    print(f"🧩 Discovered {len(df_topic_info) - 1} Distinct Conversational Clusters.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_topic_modeling()