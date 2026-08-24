"""Stage 51: Topic Injection & Thematic Hijack Anomaly Detection (s51_topic_injection.py)

Detects unnatural topic injections or sudden thematic hijacking across time windows
by computing rolling Kullback-Leibler (KL) divergence and Jensen-Shannon divergence
of topic distributions relative to the baseline corpus distribution.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from engine.config_loader import load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WINDOW_HOURS = 24
JS_DIVERGENCE_THRESHOLD = 0.40  # Flag windows with JS divergence above baseline threshold


def compute_kl_divergence(p, q, epsilon=1e-7):
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def run_topic_injection():
    """Main execution entrypoint for Stage 51."""
    print("💉 Starting Topic Injection Anomaly Scanner (s51)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s51.")
        return

    import pyarrow.parquet as pq
    available = pq.read_schema(comments_file).names
    topic_col = next((c for c in ["topic_name", "topic_id", "topic"] if c in available), None)
    if not topic_col or "published_at" not in available:
        print("⚠️ Missing topic or published_at column. Skipping s51.")
        return

    cols = ["video_id", "published_at", topic_col]
    df = pd.read_parquet(comments_file, columns=cols)
    df = df.dropna(subset=[topic_col, "published_at"])
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"])

    if df.empty or df[topic_col].nunique() < 2:
        print("⚠️ Insufficient topic diversity for injection analysis. Skipping s51.")
        return

    # Baseline global topic distribution across all topics
    all_topics = sorted(df[topic_col].unique().tolist())
    global_counts = df[topic_col].value_counts().reindex(all_topics, fill_value=0).values
    global_dist = global_counts / global_counts.sum()

    print(f"  -> Scanning {len(df):,} comments across {len(all_topics)} topic categories for divergence anomalies...")

    anomaly_rows = []

    for video_id, grp in df.groupby("video_id"):
        if len(grp) < 10:
            continue
        grp = grp.sort_values("published_at").set_index("published_at")
        
        # Resample to 24H rolling periods
        for start_time in pd.date_range(grp.index.min(), grp.index.max(), freq="12h"):
            end_time = start_time + pd.Timedelta(hours=WINDOW_HOURS)
            window_df = grp[(grp.index >= start_time) & (grp.index <= end_time)]
            
            if len(window_df) < 5:
                continue

            window_counts = window_df[topic_col].value_counts().reindex(all_topics, fill_value=0).values
            window_dist = window_counts / window_counts.sum()

            # Compute JS divergence and KL divergence against global baseline
            js_div = float(jensenshannon(window_dist, global_dist))
            kl_div = compute_kl_divergence(window_dist, global_dist)

            # Dominant topic in this window
            top_topic_idx = np.argmax(window_counts)
            dominant_topic = all_topics[top_topic_idx]
            dominant_share = window_dist[top_topic_idx]

            if js_div >= JS_DIVERGENCE_THRESHOLD or dominant_share > 0.65:
                anomaly_rows.append({
                    "video_id": str(video_id),
                    "window_start": start_time,
                    "window_end": end_time,
                    "window_comments": len(window_df),
                    "dominant_topic": str(dominant_topic),
                    "dominant_topic_share": round(float(dominant_share), 4),
                    "js_divergence": round(float(js_div), 4),
                    "kl_divergence": round(float(kl_div), 4),
                    "anomaly_flag": "topic_injection_suspect"
                })

    if anomaly_rows:
        df_out = pd.DataFrame(anomaly_rows)
    else:
        df_out = pd.DataFrame(columns=[
            "video_id", "window_start", "window_end", "window_comments",
            "dominant_topic", "dominant_topic_share", "js_divergence", "kl_divergence", "anomaly_flag"
        ])

    out_file = out_dir / "topic_injection_anomalies.parquet"
    df_out.to_parquet(out_file, index=False)
    print(f"✅ Saved to {out_file.name}")


if __name__ == "__main__":
    run_topic_injection()
