"""Stage 44: Sentiment Anomaly Detection (s44_sentiment_anomalies.py)

Identifies time windows where sentiment is unusually negative (> 2.5σ below
the per-video rolling mean), flagging potential controversy or raid events.
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

WINDOW_HOURS = 6        # rolling window size
SIGMA_THRESHOLD = 2.5   # standard deviations below mean to flag


def run_sentiment_anomalies():
    """Main execution entrypoint for Stage 44."""
    print("🚨 Starting Sentiment Anomaly Detection (s44)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s44.")
        return

    import pyarrow.parquet as pq
    available = pq.read_schema(comments_file).names
    sentiment_col = "sentiment_compound" if "sentiment_compound" in available else ("vader_compound" if "vader_compound" in available else None)
    if not sentiment_col or "published_at" not in available or "video_id" not in available:
        print(f"⚠️ Missing required columns (published_at, video_id, sentiment). Skipping s44.")
        return

    cols = ["video_id", "published_at", sentiment_col]
    df = pd.read_parquet(comments_file, columns=cols)
    df = df.rename(columns={sentiment_col: "sentiment_score"})
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at", "sentiment_score"])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0)

    if df.empty:
        print("⚠️ No usable sentiment + timestamp data. Skipping s44.")
        return

    print(f"  -> Scanning {len(df):,} comments across {df['video_id'].nunique()} videos...")

    anomaly_rows = []

    for video_id, grp in df.groupby("video_id"):
        grp = grp.sort_values("published_at").copy()
        if len(grp) < 20:
            continue

        grp = grp.set_index("published_at")
        # Resample to hourly mean sentiment
        hourly = grp["sentiment_score"].resample("1h").mean().dropna()
        if len(hourly) < 5:
            continue

        rolling_mean = hourly.rolling(window=WINDOW_HOURS, min_periods=2).mean()
        rolling_std = hourly.rolling(window=WINDOW_HOURS, min_periods=2).std()

        for ts, score in hourly.items():
            rm = rolling_mean.get(ts, np.nan)
            rs = rolling_std.get(ts, np.nan)
            if pd.isna(rm) or pd.isna(rs) or rs == 0:
                continue
            z_score = (score - rm) / rs
            if z_score < -SIGMA_THRESHOLD:
                anomaly_rows.append({
                    "video_id": video_id,
                    "window_start": ts,
                    "mean_sentiment": round(score, 4),
                    "rolling_mean": round(rm, 4),
                    "rolling_std": round(rs, 4),
                    "z_score": round(z_score, 4),
                    "sigma_threshold": SIGMA_THRESHOLD,
                    "anomaly_type": "negativity_spike",
                })

    if not anomaly_rows:
        print("  -> No sentiment anomalies detected above threshold.")
        df_out = pd.DataFrame(columns=["video_id", "window_start", "mean_sentiment",
                                        "rolling_mean", "rolling_std", "z_score",
                                        "sigma_threshold", "anomaly_type"])
    else:
        df_out = pd.DataFrame(anomaly_rows)
        print(f"  -> Detected {len(df_out)} anomalous windows across {df_out['video_id'].nunique()} videos.")

    out_file = out_dir / "sentiment_anomalies.parquet"
    df_out.to_parquet(out_file, index=False)
    print(f"✅ Saved to {out_file.name}")


if __name__ == "__main__":
    run_sentiment_anomalies()
