"""Stage 41: TF-IDF Keyword Extraction (s41_tfidf_keywords.py)

Computes per-video TF-IDF keyword importance scores and extracts
top-K keywords for each video in the corpus.
"""

import sys
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from engine.config_loader import load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_tfidf_keywords(top_k=20):
    """Main execution entrypoint for Stage 41."""
    print("🔑 Starting TF-IDF Keyword Extraction (s41)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s41.")
        return

    df = pd.read_parquet(comments_file, columns=["video_id", "text"])
    if df.empty:
        print("⚠️ Comments dataset is empty. Skipping s41.")
        return

    # Aggregate all comment text per video into a single document
    df["text"] = df["text"].fillna("").astype(str)
    video_docs = df.groupby("video_id")["text"].apply(lambda texts: " ".join(texts)).reset_index()
    video_docs.columns = ["video_id", "doc"]

    if len(video_docs) < 2:
        print("⚠️ Need at least 2 videos for TF-IDF. Skipping s41.")
        return

    print(f"  -> Computing TF-IDF across {len(video_docs)} video documents...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(video_docs["doc"])
    feature_names = vectorizer.get_feature_names_out()

    # Extract top-K keywords per video
    rows = []
    for idx, video_id in enumerate(video_docs["video_id"]):
        scores = tfidf_matrix[idx].toarray().flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        for rank, i in enumerate(top_indices, 1):
            if scores[i] > 0:
                rows.append({
                    "video_id": video_id,
                    "keyword": feature_names[i],
                    "tfidf_score": round(float(scores[i]), 6),
                    "rank": rank
                })

    df_keywords = pd.DataFrame(rows)
    out_file = out_dir / "tfidf_keywords.parquet"
    df_keywords.to_parquet(out_file, index=False)

    print(f"  -> Extracted {len(df_keywords):,} keyword entries across {len(video_docs)} videos.")
    print(f"✅ Saved to {out_file.name}")


if __name__ == "__main__":
    run_tfidf_keywords()
