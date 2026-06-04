import os
import yaml
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

def load_config(config_path="config/settings.yaml"):
    """Loads the central project configuration configuration profiles."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_topics():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    model_cache_dir = Path(config["paths"]["model_cache"]) / "bertopic_model"
    comments_file = interim_dir / "comments_clean.parquet"
    
    if not comments_file.exists():
        print(f"❌ Clean comments layer not found at {comments_file}. Run ingestion and enrichment stages first.")
        return

    print("📥 Loading enriched comment tables...")
    df_comments = pd.read_parquet(comments_file)
    
    # Filter out empty records and isolate text corpus
    df_comments['text'] = df_comments['text'].fillna("").astype(str)
    docs = df_comments['text'].tolist()
    
    if len(docs) < config["stage_02_topics"]["bertopic"]["min_topic_size"]:
        print("⚠️ Insufficient text documents to cluster meaningful topic groupings.")
        return

    # 1. Initialize Vector Core
    embedding_model_name = config["stage_02_topics"]["embedding_model"]
    print(f"📡 Generating dense semantic vectors using: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # 2. Configure Tokenization Engine to drop noisy internet symbols/stop-words
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2)

    # 3. Instantiate BERTopic Orchestrator using parameters from config
    print("🧬 Initializing BERTopic pipeline clustering architecture...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=config["stage_02_topics"]["bertopic"]["min_topic_size"],
        nr_topics=config["stage_02_topics"]["bertopic"]["nr_topics"],
        calculate_probabilities=config["stage_02_topics"]["bertopic"]["calculate_probabilities"]
    )

    # 4. Execute Clustering Fit Loop
    print("🧩 Mapping conversational geometry and calculating c-TF-IDF profiles...")
    topics, _ = topic_model.fit_transform(docs)

    # 5. Extract and Map Human-Readable Labels
    print("🏷️ Formatting cluster taxonomy labels...")
    topic_info = topic_model.get_topic_info()
    
    # Generate a clean dictionary mapping Topic ID to the top 3 descriptive keywords
    topic_label_dict = {}
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            topic_label_dict[topic_id] = "Unclassified Noise"
        else:
            # Join the top 3 predictive keywords for the timeline UI
            words = [word for word, _ in topic_model.get_topic(topic_id)[:3]]
            topic_label_dict[topic_id] = " | ".join(words)

    # 6. Append Structural Structural Metrics to Local Parquet Core
    df_comments['topic_id'] = topics
    df_comments['topic_label'] = df_comments['topic_id'].map(topic_label_dict)

    print("💾 Committing thematic indices back to interim analytical cache...")
    df_comments.to_parquet(comments_file, index=False)

    # 7. Persist Model State to local cache for standalone analytical recall
    model_cache_dir.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(model_cache_dir, serialization="safetensors", save_ctfidf=True)
    print(f"📦 Model state cleanly serialized to: {model_cache_dir}")
    print(f"✅ Discovered {len(topic_info) - 1} distinct conversational archetypes inside the community.")

if __name__ == "__main__":
    extract_topics()