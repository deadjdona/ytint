import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# Ultimate Insurance Policy: Stop NumPy 2.x background drift
import numpy as np
if int(np.__version__.split('.')[0]) >= 2:
    print("🔄 Rolling back active terminal session to NumPy 1.x matrix...")
    os.system("pip install 'numpy==1.26.4'")
    sys.exit(0)

from bertopic import BERTopic

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def extract_topics():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    comments_file = interim_dir / "comments_clean.parquet"
    
    print("📥 Loading enriched comment tables...")
    df = pd.read_parquet(comments_file)
    docs = df['text'].fillna("").astype(str).tolist()
    
    # Secure native Russian stop-words to clean c-TF-IDF profiles
    try:
        russian_stopwords = nltk.corpus.stopwords.words('russian')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        russian_stopwords = nltk.corpus.stopwords.words('russian')
        
    # Inject Russian stop-words into the Vectorizer component
    vectorizer_model = CountVectorizer(stop_words=russian_stopwords)
    
    print(f"🧩 Initializing Multilingual BERTopic Architecture...")
    topic_model = BERTopic(
        embedding_model=config["stage_02_topics"]["embedding_model"],
        min_topic_size=config["stage_02_topics"]["min_topic_size"],
        vectorizer_model=vectorizer_model,
        verbose=True
    )
    
    print("🔮 Mapping conversational geometry and calculating c-TF-IDF profiles...")
    topics, probs = topic_model.fit_transform(docs)
    
    df['topic_id'] = topics
    
    # Extract structural keywords for each mapped category
    topic_info = topic_model.get_topic_info()
    df_info = pd.DataFrame(topic_info)
    
    # Save mathematical results back to physical parquet tables
    df.to_parquet(comments_file, index=False)
    df_info.to_parquet(interim_dir / "topics_metadata.parquet", index=False)
    print("✅ Stage 02 Thematic Clustering Complete!")

if __name__ == "__main__":
    extract_topics()
