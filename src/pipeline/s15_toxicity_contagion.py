"""Stage 15: Toxicity Contagion & Troll Catalyst Identification (s15_toxicity_contagion.py)

Analyzes the propagation of toxic discourse, flame wars, and hostility contagion across comment threads.
Identifies and ranks 'Troll Catalysts'—authors who post provocative or toxic comments that disproportionately
spark heated arguments, hostile cascades, and high reply-to-like ratios.
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
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def analyze_toxicity_contagion(
    df_comments: pd.DataFrame,
    toxicity_threshold: float = 0.5,
    catalyst_min_toxic: int = 2,
    catalyst_min_score: float = 10.0
):
    """Computes toxicity contagion dynamics, thread escalation rates, and ranks troll catalysts."""
    if df_comments.empty or 'author_channel_id' not in df_comments.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df_comments.copy()

    # Ensure required metrics exist with sensible fallbacks
    if 'toxicity' not in df.columns:
        if 'vader_compound' in df.columns:
            df['toxicity'] = np.clip((df['vader_compound'] * -0.5) + 0.5, 0.0, 1.0)
        else:
            df['toxicity'] = 0.0

    if 'reply_count' not in df.columns:
        df['reply_count'] = 0

    if 'like_count' not in df.columns:
        df['like_count'] = 0

    df['is_toxic'] = df['toxicity'] >= toxicity_threshold

    # 1. Global Contagion Summary & Reproduction Rate (R0)
    toxic_comments = df[df['is_toxic']]
    neutral_comments = df[~df['is_toxic']]

    mean_replies_toxic = float(toxic_comments['reply_count'].mean()) if not toxic_comments.empty else 0.0
    mean_replies_neutral = float(neutral_comments['reply_count'].mean()) if not neutral_comments.empty else 1.0
    if mean_replies_neutral == 0:
        mean_replies_neutral = 1.0

    toxicity_r0 = round(mean_replies_toxic / mean_replies_neutral, 3)

    summary_data = [{
        "total_analyzed_comments": len(df),
        "total_toxic_comments": int(df['is_toxic'].sum()),
        "toxic_comment_share_pct": round(float(df['is_toxic'].mean() * 100), 2),
        "avg_replies_to_toxic_root": round(mean_replies_toxic, 2),
        "avg_replies_to_neutral_root": round(mean_replies_neutral, 2),
        "toxicity_reproduction_number_r0": toxicity_r0,
        "high_toxicity_threshold": toxicity_threshold
    }]
    df_summary = pd.DataFrame(summary_data)

    # 2. Video-Level Toxicity & Flame-War Vulnerability
    video_summary = df.groupby('video_id').agg(
        total_comments=('comment_id', 'count') if 'comment_id' in df.columns else ('author_channel_id', 'count'),
        toxic_comments=('is_toxic', 'sum'),
        mean_toxicity=('toxicity', 'mean'),
        total_flame_replies=('reply_count', lambda s: df.loc[s.index[df.loc[s.index, 'is_toxic']], 'reply_count'].sum() if df.loc[s.index, 'is_toxic'].any() else 0)
    ).reset_index()
    video_summary['toxic_share_pct'] = round((video_summary['toxic_comments'] / video_summary['total_comments']) * 100, 2)
    video_summary['mean_toxicity'] = round(video_summary['mean_toxicity'], 4)
    video_summary = video_summary.sort_values(by='total_flame_replies', ascending=False)

    # 3. Troll Catalyst & Flame War Instigator Ranking
    author_groups = df.groupby('author_channel_id').agg(
        total_comments=('author_channel_id', 'count'),
        toxic_comments_count=('is_toxic', 'sum'),
        mean_toxicity=('toxicity', 'mean'),
        max_toxicity=('toxicity', 'max'),
        total_likes_received=('like_count', 'sum'),
        total_replies_sparked=('reply_count', 'sum'),
        display_name=('author_display_name', 'first') if 'author_display_name' in df.columns else ('author_channel_id', 'first')
    ).reset_index()

    # Calculate Catalyst Impact Score: high toxicity + high spark of replies
    author_groups['catalyst_score'] = (
        (author_groups['toxic_comments_count'] * author_groups['mean_toxicity'] * 1.5) +
        (author_groups['total_replies_sparked'] * author_groups['mean_toxicity'])
    ).round(2)

    # Assign Catalyst Tier
    def assign_tier(row):
        if row['toxic_comments_count'] >= catalyst_min_toxic and row['catalyst_score'] >= catalyst_min_score:
            return "🔥 High Catalyst (Troll Instigator)"
        elif row['toxic_comments_count'] >= 1 and row['mean_toxicity'] > (toxicity_threshold * 0.8):
            return "⚡ Moderate Flame-Baiter"
        elif row['mean_toxicity'] > (toxicity_threshold * 0.6):
            return "⚠️ Occasional Provocateur"
        else:
            return "🛡️ Constructive Community Member"

    author_groups['catalyst_tier'] = author_groups.apply(assign_tier, axis=1)

    df_catalysts = author_groups[author_groups['toxic_comments_count'] > 0].copy()
    if df_catalysts.empty:
        df_catalysts = author_groups.copy()

    df_catalysts = df_catalysts.sort_values(by='catalyst_score', ascending=False)

    return df_summary, df_catalysts, video_summary

def run_toxicity_contagion():
    """Main execution entrypoint for Stage 37."""
    print("🔥 Starting Toxicity Contagion & Troll Catalyst Analysis (s37)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s37.")
        return

    print("  -> Loading comments dataset...")
    df = pd.read_parquet(comments_file)
    if df.empty:
        print("⚠️ Comments dataset is empty. Skipping s37.")
        return

    stage_cfg = config.get("stage_15_toxicity", {})
    tox_threshold = float(stage_cfg.get("toxicity_threshold", 0.5))
    cat_min_toxic = int(stage_cfg.get("catalyst_min_toxic", 2))
    cat_min_score = float(stage_cfg.get("catalyst_min_score", 10.0))

    print("  -> Computing toxicity contagion rates and ranking troll catalysts...")
    df_summary, df_catalysts, df_videos = analyze_toxicity_contagion(
        df,
        toxicity_threshold=tox_threshold,
        catalyst_min_toxic=cat_min_toxic,
        catalyst_min_score=cat_min_score
    )

    out_summary_file = out_dir / "toxicity_contagion_summary.parquet"
    out_catalysts_file = out_dir / "troll_catalysts.parquet"
    out_videos_file = out_dir / "video_toxicity_contagion.parquet"

    df_summary.to_parquet(out_summary_file, index=False)
    df_catalysts.to_parquet(out_catalysts_file, index=False)
    df_videos.to_parquet(out_videos_file, index=False)

    print(f"  -> Identified {len(df_catalysts):,} authors with toxic provocation profiles.")
    r0 = df_summary.iloc[0]['toxicity_reproduction_number_r0'] if not df_summary.empty else 'N/A'
    print(f"  -> Toxicity Reproduction Number (R0): {r0}")
    print(f"✅ Saved outputs to {out_summary_file.name}, {out_catalysts_file.name}, and {out_videos_file.name}")

if __name__ == "__main__":
    run_toxicity_contagion()
