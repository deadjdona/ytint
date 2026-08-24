"""Stage 09: Thread Width & Branching Factor (s09_thread_width.py)

Calculates the distribution of thread widths (replies per parent comment)
and computes Attention Transfer (downstream like & reply spillover to child comments).
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


def run_thread_width():
    print("🌲 Starting Thread Width & Attention Transfer Analysis (s09)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s09.")
        return

    print("  -> Loading comments...")
    import pyarrow.parquet as pq
    available = pq.read_schema(comments_file).names
    cols = [c for c in ["comment_id", "parent_id", "like_count", "reply_count"] if c in available]
    df = pd.read_parquet(comments_file, columns=cols)
    df["like_count"] = pd.to_numeric(df.get("like_count", 0), errors="coerce").fillna(0)

    # We only care about comments that ARE replies (they have a parent)
    replies = df[df["parent_id"].notna() & (df["parent_id"] != "") & (df["parent_id"] != df["comment_id"])].copy()
    roots = df[df["parent_id"].isna() | (df["parent_id"] == "") | (df["parent_id"] == df["comment_id"])].copy()

    if replies.empty:
        print("⚠️ No reply threads found.")
        return

    print("  -> Calculating branching factors (replies per parent)...")
    width_counts = replies["parent_id"].value_counts().reset_index()
    width_counts.columns = ["parent_id", "thread_width"]

    width_distribution = width_counts["thread_width"].value_counts().reset_index()
    width_distribution.columns = ["thread_width", "frequency"]
    width_distribution = width_distribution.sort_values("thread_width")

    out_file = out_dir / "thread_width_dist.parquet"
    width_distribution.to_parquet(out_file, index=False)

    # --- Attention Transfer (Root Likes -> Child Reply Likes Ripple) ---
    print("  -> Computing Attention Transfer and downstream spillover...")
    root_likes = roots.set_index("comment_id")["like_count"].to_dict()
    reply_like_agg = replies.groupby("parent_id")["like_count"].agg(
        child_total_likes="sum",
        child_mean_likes="mean",
        reply_count="count"
    ).reset_index()

    reply_like_agg["root_likes"] = reply_like_agg["parent_id"].map(root_likes).fillna(0)
    reply_like_agg["attention_transfer_ratio"] = np.where(
        reply_like_agg["root_likes"] > 0,
        (reply_like_agg["child_total_likes"] / reply_like_agg["root_likes"]).round(4),
        0.0
    )

    out_transfer = out_dir / "attention_transfer.parquet"
    reply_like_agg.to_parquet(out_transfer, index=False)
    print(f"  -> Attention transfer written to {out_transfer.name}")

    print(f"✅ Generated Thread Width & Attention Transfer across {len(width_counts)} threads.")


if __name__ == "__main__":
    run_thread_width()
