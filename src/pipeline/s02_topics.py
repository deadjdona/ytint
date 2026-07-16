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
        pbar.update(1)
        
        # Phase 2: Initialize ML Components with Native Progress Overrides
        pbar.set_description(f"🧩 {phases[1]}")
        embedding_model_name = config["stage_02_topics"]["embedding_model"]
        min_topic_size = config["stage_02_topics"]["min_topic_size"]
        
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
            min_cluster_size=min_topic_size, 
            metric='euclidean', 
            cluster_selection_method='eom', 
            prediction_data=True
        )
        
        # Standard NLP vectorizer to filter out noisy stop-words
        vectorizer_model = CountVectorizer(stop_words="english", min_df=2)
        
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
        text_col = 'text' if 'text' in df_comments.columns else df_comments.columns[1]
        docs = df_comments[text_col].astype(str).tolist()
        
        # Temporarily pause master progress bar layout so internal ML loops print cleanly
        pbar.close()
        
        print("\n" + "-"*60)
        print("🚀 Starting Main Pipeline Engine (Embeddings -> UMAP -> HDBSCAN)")
        print("-"*60)
        
        topics, _ = topic_model.fit_transform(docs)
        
        # Reinitialize master phase tracker for final file serialization tasks
        pbar = tqdm(total=len(phases), initial=3, desc="💾 Wrapping Up Stage 02", bar_format="{l_bar}{bar:40}{r_bar}")
        
        # Phase 4: Compile Assignment Metrics
        pbar.set_description(f"📊 {phases[3]}")
        df_comments["topic"] = topics
        
        # Generate clean human-readable summaries of topics discovered
        df_topic_info = topic_model.get_topic_info()
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