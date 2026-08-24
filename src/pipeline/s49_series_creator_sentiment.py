"""Stage 49: Series vs Standalone Comparison & Creator Sentiment Polarity (s49_series_creator_sentiment.py)

Computes:
1. Series vs Standalone Video Performance Benchmarking
2. Creator-Directed Sentiment Polarity Ratio (Positive vs Negative praise ratio)
"""

import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
from engine.config_loader import load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def detect_series_videos(df_videos):
    """
    Identifies whether videos belong to a serialized multi-part series
    or are standalone uploads using title pattern heuristics and prefix clustering.
    """
    df = df_videos.copy()
    if "title" not in df.columns:
        df["is_series"] = False
        df["series_name"] = "Standalone"
        return df

    series_patterns = [
        r"(?:part|часть|эпизод|episode|серия|выпуск|сезон|season|#|№)\s*(\d+)",
        r"(\d+)\s*(?:часть|серия|эпизод|выпуск)",
        r"(?:vol|том|act|акт)\.?\s*(\d+)",
    ]

    is_series = []
    series_names = []

    for title in df["title"].fillna(""):
        title_str = str(title).strip()
        matched = False
        for pat in series_patterns:
            m = re.search(pat, title_str, re.IGNORECASE)
            if m:
                # Extract prefix before the number as series name
                prefix = re.split(pat, title_str, flags=re.IGNORECASE)[0].strip(" -:|#№")
                is_series.append(True)
                series_names.append(prefix if len(prefix) > 2 else "Numbered Series")
                matched = True
                break
        if not matched:
            # Check for colon / dash structured title
            if ":" in title_str or "—" in title_str or " - " in title_str:
                prefix = re.split(r"[:—\-]", title_str)[0].strip()
                if len(prefix) > 3:
                    is_series.append(True)
                    series_names.append(prefix)
                    continue
            is_series.append(False)
            series_names.append("Standalone")

    df["is_series"] = is_series
    df["series_name"] = series_names
    return df


def analyze_creator_sentiment(df_comments):
    """
    Measures creator-specific sentiment polarity:
    Filters comments addressing the creator/author and calculates
    the Creator Positive / Negative praise-to-criticism ratio.
    """
    df = df_comments.copy()
    creator_address_patterns = [
        r"\b(?:автор|автору|автора|автором|ты|тебя|тебе|тобой|бро|чел|красава|красавчик|молодец|уважение|спс|спасибо)\b",
        r"\b(?:creator|author|you|your|bro|man|dude|legend|goat|king|thanks|thank you|love you)\b",
    ]
    combined_pat = re.compile("|".join(creator_address_patterns), re.IGNORECASE)

    df["text_str"] = df["text"].fillna("").astype(str)
    df["is_creator_directed"] = df["text_str"].str.contains(combined_pat, regex=True)

    sentiment_col = "sentiment_compound" if "sentiment_compound" in df.columns else ("vader_compound" if "vader_compound" in df.columns else None)
    label_col = "sentiment_label" if "sentiment_label" in df.columns else None

    if sentiment_col is None:
        df["sentiment_score"] = 0.0
    else:
        df["sentiment_score"] = pd.to_numeric(df[sentiment_col], errors="coerce").fillna(0.0)

    creator_comments = df[df["is_creator_directed"]].copy()

    rows = []

    # Corpus-wide creator sentiment
    pos_count = (creator_comments["sentiment_score"] > 0.05).sum()
    neg_count = (creator_comments["sentiment_score"] < -0.05).sum()
    neu_count = len(creator_comments) - pos_count - neg_count
    ratio = pos_count / neg_count if neg_count > 0 else float(pos_count)

    rows.append({
        "scope": "corpus",
        "video_id": "__ALL__",
        "creator_directed_comments": len(creator_comments),
        "creator_positive_count": int(pos_count),
        "creator_negative_count": int(neg_count),
        "creator_neutral_count": int(neu_count),
        "creator_pos_neg_ratio": round(float(ratio), 2),
        "creator_mean_sentiment": round(float(creator_comments["sentiment_score"].mean()), 4) if len(creator_comments) > 0 else 0.0,
    })

    # Per-video creator sentiment
    if "video_id" in creator_comments.columns:
        for vid, grp in creator_comments.groupby("video_id"):
            p_cnt = (grp["sentiment_score"] > 0.05).sum()
            n_cnt = (grp["sentiment_score"] < -0.05).sum()
            neu_cnt = len(grp) - p_cnt - n_cnt
            r = p_cnt / n_cnt if n_cnt > 0 else float(p_cnt)
            rows.append({
                "scope": "video",
                "video_id": str(vid),
                "creator_directed_comments": len(grp),
                "creator_positive_count": int(p_cnt),
                "creator_negative_count": int(n_cnt),
                "creator_neutral_count": int(neu_cnt),
                "creator_pos_neg_ratio": round(float(r), 2),
                "creator_mean_sentiment": round(float(grp["sentiment_score"].mean()), 4),
            })

    return pd.DataFrame(rows)


def run_series_creator_sentiment():
    """Main execution entrypoint for Stage 49."""
    print("🎬 Starting Series vs Standalone & Creator Sentiment Polarity (s49)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    comments_file = interim_dir / "comments_clean.parquet"
    videos_file = interim_dir / "videos_clean.parquet"

    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s49.")
        return

    df_comments = pd.read_parquet(comments_file)
    df_videos = pd.read_parquet(videos_file) if videos_file.exists() else pd.DataFrame()

    # --- 1. Series vs Standalone Video Comparison ---
    print("  -> Benchmarking series vs standalone uploads...")
    if not df_videos.empty:
        df_videos_tagged = detect_series_videos(df_videos)
        
        # Merge comment metrics
        comm_stats = df_comments.groupby("video_id").agg(
            comment_count=("comment_id", "count"),
            mean_likes=("like_count", "mean"),
            mean_sentiment=("sentiment_compound", "mean") if "sentiment_compound" in df_comments.columns else ("comment_id", lambda x: 0.0)
        ).reset_index()

        series_merged = df_videos_tagged.merge(comm_stats, on="video_id", how="left")
        
        # Summary per group
        summary_rows = []
        for is_ser, grp in series_merged.groupby("is_series"):
            summary_rows.append({
                "category": "Series / Episodic" if is_ser else "Standalone",
                "video_count": len(grp),
                "avg_comments_per_video": round(float(grp["comment_count"].mean()), 1),
                "avg_likes_per_comment": round(float(grp["mean_likes"].mean()), 2),
                "avg_sentiment": round(float(grp["mean_sentiment"].mean()), 4),
            })
        df_series_out = pd.DataFrame(summary_rows)
    else:
        df_series_out = pd.DataFrame(columns=["category", "video_count", "avg_comments_per_video", "avg_likes_per_comment", "avg_sentiment"])

    out_series = out_dir / "series_vs_standalone.parquet"
    df_series_out.to_parquet(out_series, index=False)
    print(f"  -> Series comparison written to {out_series.name}")

    # --- 2. Creator Sentiment Polarity ---
    print("  -> Computing creator-directed praise vs criticism polarity ratio...")
    df_creator = analyze_creator_sentiment(df_comments)
    out_creator = out_dir / "creator_sentiment_polarity.parquet"
    df_creator.to_parquet(out_creator, index=False)
    print(f"  -> Creator sentiment polarity written to {out_creator.name}")

    print("✅ Stage 49 Series & Creator Polarity Analysis Complete.")


if __name__ == "__main__":
    run_series_creator_sentiment()
