"""Stage 50: Cross-Modal Scene Reactions & Spoiler Detection (s50_cross_modal_reactions.py)

Computes:
1. Moment-Level Scene Reaction Taxonomy (Humor, Shock, Emotional, Critique, Navigation)
2. Narrative Spoiler & Plot Reference Detection
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

# Scene Reaction Taxonomy Lexicons
REACTION_TAXONOMY = {
    "humor_laughter": [
        r"[😂🤣😆😹]", r"\b(lol+|lmao+|rofl+|смешн\w*|угар\w*|ор\b|ржу|хах\w*|кек\w*|ору)\b"
    ],
    "shock_surprise": [
        r"[😱🤯😳😮]", r"\b(wtf|omg|holy\s+shit|шок\w*|жесть|офигеть|капец|нихера|plot\s*twist)\b"
    ],
    "emotional_touching": [
        r"[❤️🥺😭😍💔]", r"\b(плачу|душевн\w*|слез\w*|слёз\w*|трогательн\w*|wholesome|crying|beautiful|heartwarming)\b"
    ],
    "critique_analytical": [
        r"[🤔🧐]", r"\b(ошибк\w*|ляп\w*|нелогичн\w*|почему|зачем|логика|сюжет|plot\s*hole|mistake|logic)\b"
    ],
    "chapter_navigation": [
        r"[⏱️🕒🎵]", r"\b(таймкод\w*|трек|музык\w*|начало|конец|интро|аутро|intro|outro|music|song|track|timestamp)\b"
    ]
}

SPOILER_PATTERNS = [
    r"\b(спойлер\w*|спойлить|spoiler\w*)\b",
    r"\[spoiler\]|\(спойлер\)",
    r"\b(концовк\w*|финал\w*|умирает|убьют|предатель|злодей|убийца|turns\s+out|dies\s+in\s+the\s+end)\b"
]


def classify_scene_reaction(text):
    text_lower = str(text).lower()
    for reaction, patterns in REACTION_TAXONOMY.items():
        for pat in patterns:
            if re.search(pat, text_lower, re.IGNORECASE):
                return reaction
    return "general_reaction"


def detect_spoilers(text):
    text_lower = str(text).lower()
    return any(re.search(pat, text_lower, re.IGNORECASE) for pat in SPOILER_PATTERNS)


def parse_timestamp_to_seconds(ts_str):
    parts = ts_str.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        return None
    return None


def run_cross_modal_reactions():
    """Main execution entrypoint for Stage 50."""
    print("🎭 Starting Cross-Modal Scene Reactions & Spoiler Detection (s50)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s50.")
        return

    import pyarrow.parquet as pq
    available = pq.read_schema(comments_file).names
    cols = [c for c in ["comment_id", "video_id", "text", "like_count", "published_at", "extracted_timestamp"] if c in available]
    df = pd.read_parquet(comments_file, columns=cols)

    if df.empty:
        print("⚠️ Comments dataset is empty. Skipping s50.")
        return

    ts_regex = re.compile(r"\b(?:\d{1,2}:)?(?:[0-5]?\d):(?:[0-5]\d)\b")
    df["text_str"] = df["text"].fillna("").astype(str)
    
    # Extract timestamps if not already present or augment from text
    extracted_ts = []
    has_ts_mask = []
    for idx, row in df.iterrows():
        t = row["text_str"]
        existing_ts = row.get("extracted_timestamp", None)
        if pd.notna(existing_ts):
            extracted_ts.append(str(existing_ts))
            has_ts_mask.append(True)
        else:
            matches = ts_regex.findall(t)
            if matches:
                extracted_ts.append(matches[0])
                has_ts_mask.append(True)
            else:
                extracted_ts.append(None)
                has_ts_mask.append(False)

    df["extracted_timestamp"] = extracted_ts
    df["has_timestamp"] = has_ts_mask
    df["seconds"] = df["extracted_timestamp"].apply(lambda x: parse_timestamp_to_seconds(x) if x else None)

    # --- 1. Scene Reaction Classification ---
    ts_comments = df[df["has_timestamp"]].copy()
    print(f"  -> Found {len(ts_comments):,} timestamped comments across corpus.")

    if not ts_comments.empty:
        ts_comments["reaction_type"] = ts_comments["text_str"].apply(classify_scene_reaction)
        
        reaction_summary = ts_comments.groupby(["video_id", "reaction_type"]).agg(
            n_comments=("comment_id", "count"),
            mean_likes=("like_count", "mean"),
        ).reset_index()
        reaction_summary["mean_likes"] = reaction_summary["mean_likes"].round(2)
    else:
        reaction_summary = pd.DataFrame(columns=["video_id", "reaction_type", "n_comments", "mean_likes"])

    out_reactions = out_dir / "cross_modal_scene_reactions.parquet"
    reaction_summary.to_parquet(out_reactions, index=False)
    print(f"  -> Scene reactions written to {out_reactions.name}")

    # --- 2. Narrative Spoiler Detection ---
    print("  -> Scanning for narrative spoilers and plot references...")
    df["is_spoiler"] = df["text_str"].apply(detect_spoilers)
    spoiler_comments = df[df["is_spoiler"]].copy()

    if not spoiler_comments.empty:
        spoiler_out = spoiler_comments[[
            "comment_id", "video_id", "text_str", "like_count", "extracted_timestamp", "seconds"
        ]].rename(columns={"text_str": "text"})
    else:
        spoiler_out = pd.DataFrame(columns=["comment_id", "video_id", "text", "like_count", "extracted_timestamp", "seconds"])

    out_spoilers = out_dir / "spoiler_detections.parquet"
    spoiler_out.to_parquet(out_spoilers, index=False)
    print(f"  -> Flagged {len(spoiler_out)} potential spoiler comments → {out_spoilers.name}")

    print("✅ Stage 50 Cross-Modal Reactions & Spoilers Complete.")


if __name__ == "__main__":
    run_cross_modal_reactions()
