"""Stage 45: Meta & Corpus Quality (s45_corpus_quality.py)

Computes corpus-level quality metrics:
- Comment-volume vs view-count scaling per video
- Engagement rate (comments per 1K views)
- Comment-disabled video flagging
- Language coverage analysis
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from engine.config_loader import load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_corpus_quality():
    """Main execution entrypoint for Stage 45."""
    print("🔬 Starting Meta & Corpus Quality Analysis (s45)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])

    comments_file = interim_dir / "comments_clean.parquet"
    videos_file = interim_dir / "videos_clean.parquet"

    if not comments_file.exists() or not videos_file.exists():
        print("⚠️ Missing comments_clean or videos_clean. Skipping s45.")
        return

    import pyarrow.parquet as pq
    _want_cols = {"video_id", "language", "published_at"}
    _all_cols = set(pq.read_schema(comments_file).names)
    df_comments = pd.read_parquet(comments_file, columns=list(_want_cols & _all_cols))
    df_videos = pd.read_parquet(videos_file)

    # --- 1. Per-video corpus quality metrics ---
    print("  -> Computing per-video corpus quality metrics...")
    video_comment_counts = df_comments.groupby("video_id").size().reset_index(name="sampled_comments")

    quality_cols = ["video_id"]
    for col in ["total_views", "total_comments", "video_likes", "total_dislikes",
                 "comment_rate", "is_comment_disabled", "title"]:
        if col in df_videos.columns:
            quality_cols.append(col)

    df_quality = df_videos[quality_cols].merge(video_comment_counts, on="video_id", how="left")
    df_quality["sampled_comments"] = df_quality["sampled_comments"].fillna(0).astype(int)

    # Sampling bias estimate: what fraction of total comments did we actually capture?
    if "total_comments" in df_quality.columns:
        total = pd.to_numeric(df_quality["total_comments"], errors="coerce").fillna(0)
        df_quality["sampling_fraction"] = np.where(
            total > 0,
            (df_quality["sampled_comments"] / total).clip(0, 1).round(4),
            np.nan
        )

    # Log-log scaling: views vs comments
    if "total_views" in df_quality.columns:
        views = pd.to_numeric(df_quality["total_views"], errors="coerce").fillna(0)
        df_quality["log_views"] = np.log1p(views).round(4)
        df_quality["log_comments"] = np.log1p(df_quality["sampled_comments"]).round(4)

    out_quality_file = out_dir / "corpus_quality.parquet"
    df_quality.to_parquet(out_quality_file, index=False)
    print(f"  -> Quality metrics for {len(df_quality)} videos → {out_quality_file.name}")

    # --- 2. Language coverage ---
    if "language" in df_comments.columns:
        print("  -> Computing language coverage...")
        lang_counts = df_comments["language"].fillna("unknown").value_counts().reset_index()
        lang_counts.columns = ["language", "n_comments"]
        total_comments = len(df_comments)
        lang_counts["share"] = (lang_counts["n_comments"] / total_comments).round(6)
        lang_counts["cumulative_share"] = lang_counts["share"].cumsum().round(6)

        out_lang_file = out_dir / "language_coverage.parquet"
        lang_counts.to_parquet(out_lang_file, index=False)
        print(f"  -> {lang_counts['language'].nunique()} languages detected → {out_lang_file.name}")

        # Summary
        top_lang = lang_counts.head(3)["language"].tolist()
        top_share = lang_counts.head(3)["share"].sum() * 100
        print(f"  -> Top languages: {top_lang} covering {top_share:.1f}% of corpus")

    # --- 3. Disabled/missing video report ---
    if "is_comment_disabled" in df_quality.columns:
        n_disabled = df_quality["is_comment_disabled"].sum()
        print(f"  -> {int(n_disabled)} videos with comments disabled or unreachable")

    print(f"✅ Corpus quality analysis complete.")


if __name__ == "__main__":
    run_corpus_quality()
