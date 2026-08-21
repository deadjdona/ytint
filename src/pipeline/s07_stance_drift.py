"""Stage 07: Target-Specific Stance Detection & Polarization Drift (s07_stance_drift.py)

Classifies audience stance (Favor, Against, Neutral) toward key discussion targets,
tracks stance divergence and polarization across reply tree depths, and computes
video/topic consensus vs. conflict indices.
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
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Lexicon patterns for stance orientation in multilingual contexts (EN / RU)
FAVOR_PATTERNS = [
    r'\b(agree|support|great|awesome|based|correct|true|love|best|bravo|congrats|respect)\b',
    r'\b(согласен|прав|молодец|красава|лучший|респект|поддерживаю|точно|верно|обожаю|топ|красавчик)\b'
]

AGAINST_PATTERNS = [
    r'\b(disagree|wrong|fake|lie|lies|liar|nonsense|bullshit|terrible|hate|trash|scam|idiot)\b',
    r'\b(не согласен|неправ|ложь|вранье|бред|чушь|хрень|фигня|скам|против|отстой|дурак|обман)\b'
]

FAVOR_REGEX = re.compile('|'.join(FAVOR_PATTERNS), re.IGNORECASE)
AGAINST_REGEX = re.compile('|'.join(AGAINST_PATTERNS), re.IGNORECASE)

def classify_comment_stance(text: str, sentiment: float = 0.0, toxicity: float = 0.0) -> str:
    """Classifies comment text into Favor, Against, or Neutral stance."""
    if not isinstance(text, str) or not text.strip():
        return "Neutral"

    text_clean = text.lower()
    has_favor = bool(FAVOR_REGEX.search(text_clean))
    has_against = bool(AGAINST_REGEX.search(text_clean))

    if has_favor and not has_against:
        return "Favor"
    elif has_against and not has_favor:
        return "Against"
    elif has_favor and has_against:
        # Ambivalent, resolve via sentiment
        return "Favor" if sentiment >= 0.1 else ("Against" if sentiment <= -0.1 else "Neutral")
    else:
        # Fallback to polarity and toxicity thresholds
        if sentiment >= 0.35 and toxicity < 0.2:
            return "Favor"
        elif sentiment <= -0.35 or toxicity >= 0.5:
            return "Against"
        else:
            return "Neutral"

def compute_stance_analysis(df_comments: pd.DataFrame):
    """Computes target-level stance distributions, reply depth drift, and polarization indices."""
    if df_comments.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df_comments.copy()

    # Ensure required columns
    for col in ['text', 'vader_compound', 'toxicity', 'parent_id', 'video_id', 'topic']:
        if col not in df.columns:
            if col in ['vader_compound', 'toxicity']:
                df[col] = 0.0
            elif col == 'topic':
                df[col] = 0
            else:
                df[col] = ""

    # 1. Stance Classification
    df['stance'] = [
        classify_comment_stance(t, s, tox)
        for t, s, tox in zip(df['text'], df['vader_compound'], df['toxicity'])
    ]

    # Stance numeric mapping for polarization and drift calculation
    stance_val_map = {"Favor": 1.0, "Neutral": 0.0, "Against": -1.0}
    df['stance_score'] = df['stance'].map(stance_val_map).fillna(0.0)

    # 2. Reply Depth & Debate Intensity Estimation
    if 'parent_id' in df.columns and df['parent_id'].notna().any() and (df['parent_id'] != "").any():
        df['reply_depth'] = 0
        is_reply = df['parent_id'].notna() & (df['parent_id'] != "")
        df.loc[is_reply, 'reply_depth'] = 1
        if 'is_thread_terminal' in df.columns:
            df.loc[is_reply & df['is_thread_terminal'], 'reply_depth'] = 2
        depth_labels_map = {
            0: "Root Comments",
            1: "Direct Replies",
            2: "Deep Debate Threads"
        }
    else:
        df['reply_depth'] = 0
        if 'reply_count' in df.columns:
            df.loc[df['reply_count'] == 0, 'reply_depth'] = 0
            df.loc[(df['reply_count'] >= 1) & (df['reply_count'] <= 3), 'reply_depth'] = 1
            df.loc[(df['reply_count'] >= 4) & (df['reply_count'] <= 10), 'reply_depth'] = 2
            df.loc[df['reply_count'] > 10, 'reply_depth'] = 3
        depth_labels_map = {
            0: "Standalone (0 replies)",
            1: "Light Discussion (1–3 replies)",
            2: "Active Debate (4–10 replies)",
            3: "Deep Debate (10+ replies)"
        }

    # 3. Video & Topic-Level Stance Summary
    group_col = 'video_id' if 'video_id' in df.columns else 'topic'
    stance_groups = []
    
    for g_id, g_df in df.groupby(group_col):
        n_total = len(g_df)
        if n_total < 5:
            continue
        
        counts = g_df['stance'].value_counts()
        favor_cnt = counts.get('Favor', 0)
        against_cnt = counts.get('Against', 0)
        neutral_cnt = counts.get('Neutral', 0)

        favor_pct = round((favor_cnt / n_total) * 100, 2)
        against_pct = round((against_cnt / n_total) * 100, 2)
        neutral_pct = round((neutral_cnt / n_total) * 100, 2)

        # Polarization Index: High when Favor and Against are equally balanced (50/50) with low Neutral
        active_ratio = (favor_cnt + against_cnt) / max(n_total, 1)
        balance = 1.0 - (abs(favor_cnt - against_cnt) / max(favor_cnt + against_cnt, 1))
        polarization_idx = round(balance * active_ratio, 4)

        stance_groups.append({
            group_col: str(g_id),
            "total_comments": n_total,
            "favor_pct": favor_pct,
            "against_pct": against_pct,
            "neutral_pct": neutral_pct,
            "polarization_index": polarization_idx,
            "avg_stance_score": round(float(g_df['stance_score'].mean()), 4),
            "avg_toxicity": round(float(g_df['toxicity'].mean()), 4)
        })

    df_stance_summary = pd.DataFrame(stance_groups)
    if not df_stance_summary.empty:
        df_stance_summary = df_stance_summary.sort_values(by='total_comments', ascending=False)

    # 4. Reply Depth Drift Analysis
    depth_groups = []
    for depth, d_df in df.groupby('reply_depth'):
        d_counts = d_df['stance'].value_counts()
        n_d = len(d_df)
        depth_label = depth_labels_map.get(depth, f"Depth {depth}")
        depth_groups.append({
            "depth_level": depth,
            "depth_label": depth_label,
            "comment_count": n_d,
            "favor_pct": round((d_counts.get('Favor', 0) / max(n_d, 1)) * 100, 2),
            "against_pct": round((d_counts.get('Against', 0) / max(n_d, 1)) * 100, 2),
            "neutral_pct": round((d_counts.get('Neutral', 0) / max(n_d, 1)) * 100, 2),
            "mean_sentiment": round(float(d_df['vader_compound'].mean()), 4),
            "mean_toxicity": round(float(d_df['toxicity'].mean()), 4)
        })

    df_drift = pd.DataFrame(depth_groups).sort_values(by='depth_level')

    # 5. Polarized High-Conflict Threads
    conflict_threads = df[df['toxicity'] >= 0.4].sort_values(by='toxicity', ascending=False).head(100).copy()
    keep_cols = ['comment_id', 'video_id', 'author_channel_id', 'text', 'stance', 'toxicity', 'vader_compound', 'reply_depth']
    avail_cols = [c for c in keep_cols if c in conflict_threads.columns]
    df_polarized_threads = conflict_threads[avail_cols]

    return df_stance_summary, df_drift, df_polarized_threads

def run_stance_analysis():
    """Main execution entrypoint for Stage 39."""
    print("🎯 Starting Target-Specific Stance Detection & Polarization Drift (s39)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s39.")
        return

    print("  -> Loading comments dataset...")
    df = pd.read_parquet(comments_file)
    if df.empty:
        print("⚠️ Comments dataset is empty. Skipping s39.")
        return

    print("  -> Classifying target stances and calculating reply depth drift...")
    df_summary, df_drift, df_threads = compute_stance_analysis(df)

    out_summary = out_dir / "stance_summary.parquet"
    out_drift = out_dir / "stance_depth_drift.parquet"
    out_threads = out_dir / "polarized_threads.parquet"

    df_summary.to_parquet(out_summary, index=False)
    df_drift.to_parquet(out_drift, index=False)
    df_threads.to_parquet(out_threads, index=False)

    print(f"  -> Processed stance distributions across {len(df_summary)} entities/videos.")
    print(f"✅ Saved outputs to {out_summary.name}, {out_drift.name}, and {out_threads.name}")

if __name__ == "__main__":
    run_stance_analysis()
