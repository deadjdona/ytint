"""Stage 36: Arrival Speed Dynamics (s36_arrival_speed.py)

Classifies authors based on their average response time (days since upload)
and computes minute-level arrival velocity curves for the first N hours post-upload.
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


def run_arrival_speed():
    print("⏱️ Starting Arrival Speed Dynamics (s36)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s36.")
        return

    print("  -> Loading comments...")
    import pyarrow.parquet as pq
    available = pq.read_schema(comments_file).names
    cols = [c for c in ["author_channel_id", "minutes_since_upload", "video_id", "published_at"] if c in available]
    df = pd.read_parquet(comments_file, columns=cols)
    
    if "minutes_since_upload" not in df.columns or df["minutes_since_upload"].isna().all():
        if "published_at" in df.columns:
            df["published_at_dt"] = pd.to_datetime(df["published_at"], errors="coerce")
            videos_file = interim_dir / "videos_clean.parquet"
            if videos_file.exists():
                v_avail = pq.read_schema(videos_file).names
                v_cols = [c for c in ["video_id", "published_at"] if c in v_avail]
                df_v = pd.read_parquet(videos_file, columns=v_cols)
                if "published_at" in df_v.columns:
                    df_v["video_pub_at"] = pd.to_datetime(df_v["published_at"], errors="coerce")
                    df = df.merge(df_v[["video_id", "video_pub_at"]].drop_duplicates("video_id"), on="video_id", how="left")
                    df["minutes_since_upload"] = (df["published_at_dt"] - df["video_pub_at"]).dt.total_seconds() / 60.0
            if "minutes_since_upload" not in df.columns or df["minutes_since_upload"].isna().all():
                min_pub = df.groupby("video_id")["published_at_dt"].transform("min")
                df["minutes_since_upload"] = (df["published_at_dt"] - min_pub).dt.total_seconds() / 60.0
        else:
            df["minutes_since_upload"] = 10.0

    df["minutes_since_upload"] = pd.to_numeric(df["minutes_since_upload"], errors="coerce").fillna(10.0)
    df = df[df["minutes_since_upload"] >= 0]

    if df.empty:
        return

    df["days_since_upload"] = df["minutes_since_upload"] / 1440.0

    print("  -> Calculating average arrival speeds per author...")
    author_speed = df.groupby("author_channel_id")["days_since_upload"].mean().reset_index()

    # Classify authors
    conditions = [
        author_speed["days_since_upload"] < 1,
        author_speed["days_since_upload"] <= 7,
        author_speed["days_since_upload"] <= 30,
        author_speed["days_since_upload"] > 30,
    ]
    choices = [
        "1. First Responder (<1 Day)",
        "2. On-Time (1-7 Days)",
        "3. Late Arrival (8-30 Days)",
        "4. Necromancer (>30 Days)",
    ]

    author_speed["classification"] = np.select(conditions, choices, default="Unknown")
    distribution = (
        author_speed["classification"]
        .value_counts()
        .rename_axis("classification")
        .reset_index(name="author_count")
    )
    distribution = distribution.sort_values("classification")

    out_file = out_dir / "arrival_speed.parquet"
    distribution.to_parquet(out_file, index=False)

    # --- Minute-Level Arrival Velocity Curve (First 120 Minutes) ---
    print("  -> Computing minute-level arrival velocity curve (first 120 minutes)...")
    early_comments = df[df["minutes_since_upload"] <= 120].copy()
    if not early_comments.empty:
        early_comments["minute_bin"] = early_comments["minutes_since_upload"].astype(int)
        minute_curve = early_comments.groupby("minute_bin").size().reset_index(name="comment_count")
        minute_curve["cumulative_comments"] = minute_curve["comment_count"].cumsum()
        minute_curve["velocity_per_min"] = minute_curve["comment_count"].rolling(window=5, min_periods=1).mean().round(2)

        out_curve = out_dir / "minute_arrival_curve.parquet"
        minute_curve.to_parquet(out_curve, index=False)
        print(f"  -> Minute arrival curve written to {out_curve.name}")

    print(f"✅ Generated Arrival Speed distribution for {len(author_speed)} authors.")


if __name__ == "__main__":
    run_arrival_speed()
