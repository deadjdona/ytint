"""Stage 42: Polarity vs Engagement Analysis (s42_polarity_engagement.py)

Analyzes whether negative comments systematically receive more likes/replies
than positive or neutral comments, computed per-video and corpus-wide.
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


def run_polarity_engagement():
    """Main execution entrypoint for Stage 42."""
    print("📊 Starting Polarity vs Engagement Analysis (s42)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s42.")
        return

    cols = ["video_id", "like_count", "reply_count", "sentiment_label", "sentiment_compound", "vader_compound"]
    import pyarrow.parquet as pq
    available_cols = pq.read_schema(comments_file).names
    cols = [c for c in cols if c in available_cols]
    df = pd.read_parquet(comments_file, columns=cols)

    if df.empty:
        print("⚠️ Comments dataset is empty. Skipping s42.")
        return

    if "sentiment_label" not in df.columns:
        s_col = "sentiment_compound" if "sentiment_compound" in df.columns else ("vader_compound" if "vader_compound" in df.columns else None)
        if s_col:
            df["sentiment_label"] = np.where(df[s_col] >= 0.05, "POSITIVE", np.where(df[s_col] <= -0.05, "NEGATIVE", "NEUTRAL"))
        else:
            print("⚠️ No sentiment data available. Skipping s42.")
            return

    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    if "reply_count" in df.columns:
        df["reply_count"] = pd.to_numeric(df["reply_count"], errors="coerce").fillna(0)

    # -- Corpus-wide analysis --
    print("  -> Computing corpus-wide polarity vs engagement...")
    corpus_stats = df.groupby("sentiment_label").agg(
        n_comments=("like_count", "count"),
        mean_likes=("like_count", "mean"),
        median_likes=("like_count", "median"),
        total_likes=("like_count", "sum"),
        p90_likes=("like_count", lambda x: np.percentile(x, 90)),
    ).reset_index()

    if "reply_count" in df.columns:
        reply_stats = df.groupby("sentiment_label")["reply_count"].agg(
            mean_replies="mean", median_replies="median"
        ).reset_index()
        corpus_stats = corpus_stats.merge(reply_stats, on="sentiment_label", how="left")

    corpus_stats["level"] = "corpus"
    corpus_stats["video_id"] = "__ALL__"

    # -- Per-video analysis --
    print("  -> Computing per-video polarity vs engagement...")
    per_video = df.groupby(["video_id", "sentiment_label"]).agg(
        n_comments=("like_count", "count"),
        mean_likes=("like_count", "mean"),
        median_likes=("like_count", "median"),
        total_likes=("like_count", "sum"),
    ).reset_index()

    if "reply_count" in df.columns:
        reply_pv = df.groupby(["video_id", "sentiment_label"])["reply_count"].agg(
            mean_replies="mean", median_replies="median"
        ).reset_index()
        per_video = per_video.merge(reply_pv, on=["video_id", "sentiment_label"], how="left")

    per_video["level"] = "video"

    # -- Combine and save --
    df_out = pd.concat([corpus_stats, per_video], ignore_index=True)
    for col in df_out.select_dtypes(include="float").columns:
        df_out[col] = df_out[col].round(4)

    out_file = out_dir / "polarity_engagement.parquet"
    df_out.to_parquet(out_file, index=False)

    # Quick summary
    neg = corpus_stats[corpus_stats["sentiment_label"] == "negative"]
    pos = corpus_stats[corpus_stats["sentiment_label"] == "positive"]
    if not neg.empty and not pos.empty:
        neg_mean = neg["mean_likes"].values[0]
        pos_mean = pos["mean_likes"].values[0]
        ratio = neg_mean / pos_mean if pos_mean > 0 else float("inf")
        print(f"  -> Negativity engagement ratio: {ratio:.2f}x (neg={neg_mean:.1f} vs pos={pos_mean:.1f} avg likes)")

    print(f"✅ Saved to {out_file.name}")


if __name__ == "__main__":
    run_polarity_engagement()
