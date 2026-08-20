"""Stage 35: Audience Demand & Content Intent Mining (s35_audience_intent.py)

Extracts and categorizes audience intentions (Content Ideas / Video Suggestions,
Questions & Confusion, Critiques & Technical Feedback, and Appreciation)
to generate actionable creator intelligence and a prioritized content roadmap.
"""

import re
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

# Multilingual Regex Heuristic Patterns
RE_CONTENT_IDEA = re.compile(
    r"(?i)\b(сделай(?:те)?\s+(?:видео|ролик|обзор|выпуск|разбор|стрим)|"
    r"сними(?:те)?\s+(?:видео|ролик|обзор|про)|"
    r"расскажи(?:те)?\s+про|"
    r"хотелось\s+бы\s+(?:увидеть|услышать|видео|ролик)|"
    r"было\s+бы\s+круто\s+(?:снять|увидеть|сделать)|"
    r"жду\s+(?:видео|ролик|обзор|выпуск|вторую\s+часть)|"
    r"make\s+a\s+video\s+about|"
    r"can\s+you\s+(?:make|do|cover|review)|"
    r"please\s+(?:do|make|cover|talk\s+about)|"
    r"video\s+idea|"
    r"next\s+video\s+(?:about|on)|"
    r"part\s+2\s+(?:about|on|please))\b"
)

RE_CRITIQUE = re.compile(
    r"(?i)\b((?:очень\s+)?(?:тихий|плохой|ужасный|кривой|странный)\s+(?:звук|микрофон|монтаж)|"
    r"(?:звук|микрофон|монтаж)\s+(?:тихий|плохой|хрипит|лагает|фонит|шумит|ужасный|кривой)|"
    r"сделай(?:те)?\s+(?:звук|микрофон)\s+погромче|"
    r"ошибка\s+на\s+\d+|"
    r"таймкод\s+\d+|"
    r"плохо\s+слышно|"
    r"bad\s+audio|audio\s+is\s+(?:too\s+quiet|low|distorted)|"
    r"fix\s+the\s+(?:mic|audio|sound|volume|editing)|"
    r"too\s+quiet|can\s+barely\s+hear|"
    r"typo\s+at|mistake\s+at)\b"
)

RE_QUESTION = re.compile(
    r"(?i)\b(почему|как\s+(?:так|это|сделать|найти)|что\s+(?:значит|делать|такое)|"
    r"где\s+(?:взять|найти|купить)|когда\s+(?:будет|выйдет)|в\s+чем\s+смысл|"
    r"why\s+(?:did|is|are|do|would)|how\s+(?:to|do|did|can)|what\s+(?:is|does|about)|"
    r"when\s+will|where\s+can)\b"
)

RE_APPRECIATION = re.compile(
    r"(?i)\b(спасибо|благодарю|молодец|лучший|отличн(?:ое|ый|о)|топ|супер|красава|красавчик|"
    r"обожаю|круто|шедевр|респект|"
    r"thank\s+you|thanks|awesome|great\s+video|amazing|legend|best\s+channel|love\s+this)\b"
)

def classify_intent(text: str) -> str:
    """Classifies a comment string into primary intent category."""
    if not isinstance(text, str) or not text.strip():
        return "DEBATE_OPINION"
    
    clean = text.strip()
    
    # 1. Content Ideas have highest priority for creator roadmaps
    if RE_CONTENT_IDEA.search(clean):
        return "CONTENT_IDEA"
    
    # 2. Critiques & Technical Issues
    if RE_CRITIQUE.search(clean):
        return "CRITIQUE_FEEDBACK"
        
    # 3. Direct Questions & Inquiries
    if ("?" in clean and len(clean) > 10) or RE_QUESTION.search(clean):
        return "QUESTION_CONFUSION"
        
    # 4. Gratitude & Praise
    if RE_APPRECIATION.search(clean):
        return "APPRECIATION"
        
    return "DEBATE_OPINION"

def run_audience_intent():
    """Main execution function for Stage 35."""
    print("🎯 Starting Audience Demand & Intent Mining (s35)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print(f"⚠️ Missing {comments_file.name}. Skipping s35.")
        return
        
    print("  -> Loading comments dataset...")
    df = pd.read_parquet(comments_file)
    if df.empty:
        print("⚠️ Comments dataset is empty. Skipping s35.")
        return

    text_col = 'text_original' if 'text_original' in df.columns else ('text' if 'text' in df.columns else None)
    if not text_col:
        print("⚠️ No text column found. Skipping s35.")
        return

    print("  -> Classifying conversational intents...")
    df['audience_intent'] = df[text_col].astype(str).apply(classify_intent)

    # Calculate Intent Summary Aggregations
    sentiment_col = 'vader_compound' if 'vader_compound' in df.columns else None
    like_col = 'like_count' if 'like_count' in df.columns else None

    agg_dict = {'comment_id': 'count'}
    if like_col:
        agg_dict[like_col] = ['sum', 'mean', 'max']
    if sentiment_col:
        agg_dict[sentiment_col] = 'mean'

    summary = df.groupby('audience_intent').agg(
        comment_count=('comment_id', 'count'),
        total_likes=(like_col, 'sum') if like_col else ('comment_id', 'count'),
        avg_likes=(like_col, 'mean') if like_col else ('comment_id', 'count'),
        max_likes=(like_col, 'max') if like_col else ('comment_id', 'count'),
        avg_sentiment=(sentiment_col, 'mean') if sentiment_col else ('comment_id', lambda x: 0.0)
    ).reset_index()

    total_comments = len(df)
    summary['share_pct'] = (summary['comment_count'] / total_comments) * 100

    out_summary_file = out_dir / "audience_intent_summary.parquet"
    summary.to_parquet(out_summary_file, index=False)
    print(f"  -> Saved intent summary matrix to {out_summary_file.name}")

    # Extract Actionable Content Requests & Top Questions
    actionable_df = df[df['audience_intent'].isin(['CONTENT_IDEA', 'QUESTION_CONFUSION', 'CRITIQUE_FEEDBACK'])].copy()
    
    export_cols = ['comment_id', 'video_id', 'author_channel_id', text_col, 'audience_intent']
    if like_col:
        export_cols.append(like_col)
    if 'published_at' in df.columns:
        export_cols.append('published_at')

    existing_export_cols = [c for c in export_cols if c in actionable_df.columns]
    actionable_df = actionable_df[existing_export_cols]
    
    if like_col:
        actionable_df = actionable_df.sort_values(by=like_col, ascending=False)

    out_requests_file = out_dir / "audience_content_requests.parquet"
    actionable_df.head(1000).to_parquet(out_requests_file, index=False)
    print(f"  -> Saved top {min(len(actionable_df), 1000)} prioritized audience requests to {out_requests_file.name}")
    print(f"✅ Audience Demand & Intent Mining (s35) complete. Processed {total_comments:,} records.")

if __name__ == "__main__":
    run_audience_intent()
