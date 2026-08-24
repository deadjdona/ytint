"""Stage 47: Slang & Internet-Register Lexicon Analysis (s47_slang_lexicon.py)

Analyzes the frequency, temporal adoption trends, and sentiment associations
of bilingual internet slang and informal register terms across the corpus.
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

# Comprehensive bilingual (English & Russian) internet register and slang lexicon
SLANG_LEXICON = {
    # English terms
    "imo": r"\bimo\b",
    "imho": r"\bimho\b",
    "tbh": r"\btbh\b",
    "afaik": r"\bafaik\b",
    "lol": r"\blol+\b",
    "lmao": r"\blmao+\b",
    "rofl": r"\brofl+\b",
    "cringe": r"\bcringe\b",
    "based": r"\bbased\b",
    "fr": r"\bfr\b",
    "ong": r"\bong\b",
    "rip": r"\brip\b",
    "gg": r"\bgg\b",
    "idk": r"\bidk\b",
    "smh": r"\bsmh\b",
    "goat": r"\bgoat\b",
    "cap": r"\bcap\b",
    "no cap": r"\bno cap\b",
    "w": r"\b(?:massive\s+w|huge\s+w|common\s+w)\b",
    "l": r"\b(?:massive\s+l|huge\s+l|common\s+l)\b",
    "ratio": r"\bratio\b",
    "pov": r"\bpov\b",
    "sus": r"\bsus\b",
    "simp": r"\bsimp\b",
    
    # Russian terms
    "лол": r"\bлол+\b",
    "кек": r"\bкек+\b",
    "кринж": r"\bкринж\w*\b",
    "база": r"\bбаза\b|\bбазированн?\w*\b",
    "рофл": r"\bрофл\w*\b",
    "хз": r"\bхз\b",
    "жиза": r"\bжиз[аеу]\b",
    "рил": r"\bрил\b",
    "топ": r"\bтоп\b|\bтопово\w*\b",
    "годнота": r"\bгоднот\w*\b",
    "чел": r"\bчел\w*\b",
    "краш": r"\bкраш\w*\b",
    "вайб": r"\bвайб\w*\b",
    "душный": r"\bдушн\w*\b",
    "хайп": r"\bхайп\w*\b",
    "пруфы": r"\bпруф\w*\b",
    "имхо": r"\bимхо\b",
    "чилл": r"\bчилл\w*\b",
    "помойка": r"\bпомойк\w*\b",
    "зашквар": r"\bзашквар\w*\b",
}


def run_slang_lexicon():
    """Main execution entrypoint for Stage 47."""
    print("💬 Starting Slang & Internet-Register Lexicon Analysis (s47)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s47.")
        return

    import pyarrow.parquet as pq
    available = pq.read_schema(comments_file).names
    cols = [c for c in ["comment_id", "video_id", "published_at", "text", "like_count", "sentiment_compound", "vader_compound"] if c in available]
    df = pd.read_parquet(comments_file, columns=cols)

    if df.empty:
        print("⚠️ Comments dataset is empty. Skipping s47.")
        return

    df["text_clean"] = df["text"].fillna("").astype(str).str.lower()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    sentiment_col = "sentiment_compound" if "sentiment_compound" in df.columns else None

    print(f"  -> Scanning {len(df):,} comments across {len(SLANG_LEXICON)} slang terms...")

    rows = []
    total_comments = len(df)

    # Compile regexes
    compiled_patterns = {term: re.compile(pat, re.IGNORECASE) for term, pat in SLANG_LEXICON.items()}

    for term, pattern in compiled_patterns.items():
        matches = df["text_clean"].str.contains(pattern, regex=True)
        count = matches.sum()
        if count == 0:
            continue

        matched_df = df[matches]
        mean_likes = matched_df["like_count"].mean()
        median_likes = matched_df["like_count"].median()
        avg_sentiment = matched_df[sentiment_col].mean() if sentiment_col else 0.0

        # Unique videos containing the slang
        unique_videos = matched_df["video_id"].nunique() if "video_id" in matched_df.columns else 0

        # Language of the slang term (heuristic)
        lang = "ru" if re.search(r"[\u0400-\u04FF]", term) else "en"

        rows.append({
            "slang_term": term,
            "language": lang,
            "total_occurrences": int(count),
            "comment_frequency": round(float(count / total_comments), 6),
            "mean_likes": round(float(mean_likes), 2),
            "median_likes": round(float(median_likes), 2),
            "avg_sentiment": round(float(avg_sentiment), 4),
            "unique_videos_present": int(unique_videos),
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("total_occurrences", ascending=False).reset_index(drop=True)
        df_out["rank"] = df_out.index + 1
    else:
        df_out = pd.DataFrame(columns=[
            "rank", "slang_term", "language", "total_occurrences",
            "comment_frequency", "mean_likes", "median_likes",
            "avg_sentiment", "unique_videos_present"
        ])

    out_file = out_dir / "slang_lexicon_frequency.parquet"
    df_out.to_parquet(out_file, index=False)

    print(f"  -> Extracted {len(df_out)} active slang terms across corpus.")
    print(f"✅ Saved to {out_file.name}")


if __name__ == "__main__":
    run_slang_lexicon()
