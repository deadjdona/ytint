"""Stage 14: Word Co-occurrence Network (s14_cooccurrence.py)

Extracts top terms from comments and builds a co-occurrence matrix 
to visualize which words/concepts are frequently discussed together.
"""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from engine.config_loader import load_config

def run_cooccurrence():
    print("🕸️ Starting Word Co-occurrence Extraction (s14)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s14.")
        return
        
    print("  -> Loading comments text...")
    df = pd.read_parquet(comments_file, columns=['comment_id', 'text'])
    df = df.dropna(subset=['text'])
    
    if df.empty:
        print("⚠️ No valid text found. Skipping.")
        return
        
    print("  -> Vectorizing text and building co-occurrence matrix...")
    # Restrict to words >= 4 characters to filter out common short Russian/English stop words.
    # Ignore words that appear in >15% of documents, require at least 15 occurrences.
    # Keep top 150 words for network readability.
    vec = CountVectorizer(max_df=0.15, min_df=15, max_features=150, token_pattern=r'(?u)\b\w{4,}\b')
    
    X = vec.fit_transform(df['text'])
    
    # Co-occurrence matrix: X.T * X
    Xc = (X.T * X)
    Xc.setdiag(0)
    
    vocab = vec.get_feature_names_out()
    df_coocc = pd.DataFrame(Xc.toarray(), index=vocab, columns=vocab)
    
    print("  -> Converting to edge list...")
    edges = df_coocc.stack().reset_index()
    edges.columns = ['source', 'target', 'weight']
    
    # Filter out 0 weights
    edges = edges[edges['weight'] > 0]
    
    # Remove duplicates (A->B is same as B->A in undirected graph)
    edges['node1'] = edges[['source', 'target']].min(axis=1)
    edges['node2'] = edges[['source', 'target']].max(axis=1)
    edges = edges.drop_duplicates(subset=['node1', 'node2']).drop(columns=['node1', 'node2'])
    
    # Keep top 300 strongest associations for a readable network graph
    edges = edges.sort_values('weight', ascending=False).head(300)
    
    out_edges = out_dir / "word_cooccurrence_edges.parquet"
    edges.to_parquet(out_edges, index=False)
    
    # Save node frequencies for sizing the bubbles
    node_freq = pd.DataFrame({'word': vocab, 'freq': X.sum(axis=0).A1})
    out_nodes = out_dir / "word_cooccurrence_nodes.parquet"
    node_freq.to_parquet(out_nodes, index=False)
    
    print(f"✅ Generated co-occurrence graph with {len(vocab)} nodes and {len(edges)} edges.")
    print(f"✅ Saved to {out_dir}")

if __name__ == "__main__":
    run_cooccurrence()
