"""Stage 43: Emoji Signatures & Sentiment Mapping (s43_emoji_signatures.py)

Computes per-topic and per-video emoji frequency distributions,
emoji diversity profiles, and emoji-to-sentiment correlation mapping.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from engine.config_loader import load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_emoji_signatures():
    """Main execution entrypoint for Stage 43."""
    print("😀 Starting Emoji Signatures & Sentiment Mapping (s43)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s43.")
        return

    import pyarrow.parquet as pq
    available = pq.read_schema(comments_file).names
    cols = [c for c in ["video_id", "emoji_list", "emoji_count", "sentiment_label", "topic_id", "topic_name", "text"] if c in available]
    df = pd.read_parquet(comments_file, columns=cols)

    if df.empty:
        print("⚠️ Comments dataset is empty. Skipping s43.")
        return

    if "emoji_list" not in df.columns:
        if "text" in df.columns:
            import emoji
            df["emoji_list"] = df["text"].fillna("").astype(str).apply(lambda t: [m["emoji"] for m in emoji.emoji_list(t)])
            df["emoji_count"] = df["emoji_list"].apply(len)
        else:
            print("⚠️ No emoji_list or text column found. Skipping s43.")
            return

    # Normalise: emoji_list may be stored as string repr or list
    def parse_emoji_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str) and val.startswith("["):
            try:
                import ast
                return ast.literal_eval(val)
            except Exception:
                return []
        return []

    df["emoji_list"] = df["emoji_list"].apply(parse_emoji_list)
    df_with_emoji = df[df["emoji_list"].map(len) > 0].copy()

    if df_with_emoji.empty:
        print("⚠️ No comments with emoji data found. Skipping s43.")
        return

    print(f"  -> {len(df_with_emoji):,} comments contain emoji out of {len(df):,} total.")

    rows = []

    # --- Per-sentiment emoji mapping ---
    print("  -> Computing emoji-sentiment mapping...")
    if "sentiment_label" in df.columns:
        for sentiment, grp in df_with_emoji.groupby("sentiment_label"):
            all_emoji = [e for lst in grp["emoji_list"] for e in lst]
            counter = Counter(all_emoji)
            total = sum(counter.values())
            for emoji, count in counter.most_common(50):
                rows.append({
                    "group_type": "sentiment",
                    "group_value": sentiment,
                    "emoji": emoji,
                    "count": count,
                    "frequency": round(count / total, 6),
                    "video_id": None,
                    "topic_id": None,
                })

    # --- Per-topic emoji signatures ---
    print("  -> Computing per-topic emoji signatures...")
    topic_col = "topic_name" if "topic_name" in df.columns else ("topic_id" if "topic_id" in df.columns else None)
    if topic_col:
        for topic, grp in df_with_emoji.groupby(topic_col):
            all_emoji = [e for lst in grp["emoji_list"] for e in lst]
            counter = Counter(all_emoji)
            total = sum(counter.values())
            for emoji, count in counter.most_common(20):
                rows.append({
                    "group_type": "topic",
                    "group_value": str(topic),
                    "emoji": emoji,
                    "count": count,
                    "frequency": round(count / total, 6),
                    "video_id": None,
                    "topic_id": str(topic) if topic_col == "topic_id" else None,
                })

    # --- Per-video emoji profiles ---
    print("  -> Computing per-video emoji profiles...")
    if "video_id" in df.columns:
        for vid, grp in df_with_emoji.groupby("video_id"):
            all_emoji = [e for lst in grp["emoji_list"] for e in lst]
            counter = Counter(all_emoji)
            total = sum(counter.values())
            for emoji, count in counter.most_common(15):
                rows.append({
                    "group_type": "video",
                    "group_value": str(vid),
                    "emoji": emoji,
                    "count": count,
                    "frequency": round(count / total, 6),
                    "video_id": str(vid),
                    "topic_id": None,
                })

    df_out = pd.DataFrame(rows)
    out_file = out_dir / "emoji_signatures.parquet"
    df_out.to_parquet(out_file, index=False)

    n_emoji = df_out["emoji"].nunique()
    print(f"  -> {len(df_out):,} emoji-group entries covering {n_emoji} unique emoji.")
    print(f"✅ Saved to {out_file.name}")


if __name__ == "__main__":
    run_emoji_signatures()
