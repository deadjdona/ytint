"""Stage 30: Code-Switching Detection (s30_code_switching.py)

Detects "Code-Switching" (mixing languages) by analyzing the ratio of Cyrillic to Latin characters.
In a predominantly Russian corpus, significant presence of both alphabets in a single comment indicates
the user is code-switching (e.g. inserting English gamer slang or terms into Russian sentences).
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config
import re

def run_code_switching():
    print("🔀 Starting Code-Switching Detection (s30)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s30.")
        return
        
    print("  -> Loading comments...")
    comments_df = pd.read_parquet(comments_file)
    text_col = 'text_original' if 'text_original' in comments_df.columns else ('text' if 'text' in comments_df.columns else None)
    if not text_col:
        print("⚠️ Missing text column. Skipping s30.")
        return
        
    df = comments_df.dropna(subset=[text_col]).copy()
    df['text_original'] = df[text_col]
    if 'like_count' not in df.columns:
        df['like_count'] = 0
    if 'comment_id' not in df.columns:
        df['comment_id'] = range(len(df))
    if df.empty: return
    
    print("  -> Applying Cyrillic/Latin heuristic for Code-Switching...")
    
    def is_code_switched(text):
        # Remove URLs so they don't artificially inflate Latin counts
        text_no_url = re.sub(r'http\S+', '', str(text))
        
        cyrillic_matches = len(re.findall(r'[а-яА-ЯёЁ]', text_no_url))
        latin_matches = len(re.findall(r'[a-zA-Z]', text_no_url))
        
        total = cyrillic_matches + latin_matches
        if total < 10: return False # Too short to confidently detect code-switching
        
        # If at least 15% of the text is in the secondary alphabet, and it's at least 3 characters
        if cyrillic_matches >= 3 and latin_matches >= 3:
            if (cyrillic_matches / total >= 0.15) and (latin_matches / total >= 0.15):
                return True
        return False

    df['is_code_switched'] = df[text_col].apply(is_code_switched)
    
    # Calculate Impact
    summary = df.groupby('is_code_switched').agg(
        comment_count=('comment_id', 'count'),
        avg_likes=('like_count', 'mean')
    ).reset_index()
    
    # Map booleans to labels
    summary['type'] = summary['is_code_switched'].map({True: 'Code-Switched (Mixed RU/EN)', False: 'Monolingual'})
    
    out_file = out_dir / "code_switching_impact.parquet"
    summary.to_parquet(out_file, index=False)
    
    cs_df = summary[summary['is_code_switched'] == True]
    num_cs = cs_df['comment_count'].iloc[0] if not cs_df.empty else 0
    print(f"✅ Detected {num_cs} code-switched comments out of {len(df)}.")

if __name__ == "__main__":
    run_code_switching()
