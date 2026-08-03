"""Stage 31: Comparative & Cross-Video (s31_cross_video_comparisons.py)

1. Video Profile Radar: Generates comparative dimensions (Positivity, Negativity, Branching, Volume, Length) 
   for the top videos.
2. Controversy Impact: Auto-detects the most toxic/negative video and compares channel-wide 
   sentiment 7 days before vs 7 days after its release.
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config
import numpy as np
import sys

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

def run_cross_video():
    print("⚔️ Starting Comparative & Cross-Video Analysis (s31)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s31.")
        return
        
    print("  -> Loading comments...")
    comments_df = pd.read_parquet(comments_file)
    text_col = 'text_original' if 'text_original' in comments_df.columns else ('text' if 'text' in comments_df.columns else None)
    
    df = comments_df.dropna(subset=['video_id', 'published_at']).copy()
    if text_col:
        df['text_original'] = df[text_col]
    else:
        df['text_original'] = ""
    if 'parent_id' not in df.columns:
        df['parent_id'] = None
    if 'sentiment_label' not in df.columns:
        df['sentiment_label'] = 'NEUTRAL'
    if 'comment_id' not in df.columns:
        df['comment_id'] = range(len(df))
    
    if df.empty: return
    
    # 1. Video Profile Radar metrics
    print("  -> Computing per-video profiles...")
    
    df['comment_length'] = df[text_col].fillna('').astype(str).str.len()
    
    if 'bpe_token_count' not in df.columns:
        import gigatoken as gt
        try:
            g_tok = gt.Tokenizer("openai-community/gpt2")
            tok_lists = g_tok.encode_batch_list(df[text_col].astype(str).tolist())
            df['bpe_token_count'] = [len(t) for t in tok_lists]
        except Exception:
            df['bpe_token_count'] = df['comment_length']
            
    video_stats = df.groupby('video_id').agg(
        volume=('comment_id', 'count'),
        positive_comments=('sentiment_label', lambda x: (x == 'POSITIVE').sum()),
        negative_comments=('sentiment_label', lambda x: (x == 'NEGATIVE').sum()),
        avg_length=('comment_length', 'mean'),
        avg_bpe_tokens=('bpe_token_count', 'mean')
    ).reset_index()
    
    video_stats['positivity'] = video_stats['positive_comments'] / video_stats['volume']
    video_stats['negativity'] = video_stats['negative_comments'] / video_stats['volume']
    
    replies = df[df['parent_id'].notna() & (df['parent_id'] != '')]
    branching = replies.groupby('video_id').size().reset_index(name='reply_count')
    
    video_stats = video_stats.merge(branching, on='video_id', how='left').fillna(0)
    video_stats['branching_factor'] = video_stats['reply_count'] / video_stats['volume']
    
    top_videos = video_stats.nlargest(5, 'volume').copy()
    
    for col in ['positivity', 'negativity', 'volume', 'branching_factor', 'avg_length']:
        max_val = video_stats[col].max()
        if max_val > 0:
            top_videos[f'{col}_norm'] = top_videos[col] / max_val
        else:
            top_videos[f'{col}_norm'] = 0
            
    top_videos.to_parquet(out_dir / "video_profile_radar.parquet", index=False)
    
    # 2. Controversy Impact
    print("  -> Auto-detecting most controversial video...")
    valid_videos = video_stats[video_stats['volume'] >= 50]
    if not valid_videos.empty:
        most_controversial = valid_videos.loc[valid_videos['negativity'].idxmax()]
        cv_vid = most_controversial['video_id']
        
        vid_comments = df[df['video_id'] == cv_vid]
        release_date = pd.to_datetime(vid_comments['published_at'].min(), utc=True)
        
        print(f"     Found highly negative video {cv_vid} released around {release_date.date()}")
        
        df['pub_date'] = pd.to_datetime(df['published_at'], utc=True)
        
        before_mask = (df['pub_date'] >= release_date - pd.Timedelta(days=14)) & (df['pub_date'] < release_date)
        after_mask = (df['pub_date'] >= release_date) & (df['pub_date'] < release_date + pd.Timedelta(days=14))
        
        before_df = df[before_mask]
        after_df = df[after_mask]
        
        def get_dist(sub_df):
            if sub_df.empty: return {}
            counts = sub_df['sentiment_label'].value_counts(normalize=True)
            return counts.to_dict()
            
        before_dist = get_dist(before_df)
        after_dist = get_dist(after_df)
        
        controversy_data = pd.DataFrame([
            {'period': '14 Days Before', 'POSITIVE': before_dist.get('POSITIVE', 0), 'NEUTRAL': before_dist.get('NEUTRAL', 0), 'NEGATIVE': before_dist.get('NEGATIVE', 0)},
            {'period': '14 Days After', 'POSITIVE': after_dist.get('POSITIVE', 0), 'NEUTRAL': after_dist.get('NEUTRAL', 0), 'NEGATIVE': after_dist.get('NEGATIVE', 0)}
        ])
        
        controversy_data.to_parquet(out_dir / "controversy_impact.parquet", index=False)
        
        # CausalImpact Counterfactual Time-Series Modeling
        try:
            from causalimpact import CausalImpact
            daily_series = df.groupby([df['pub_date'].dt.date, 'video_id']).size().unstack(fill_value=0)
            if cv_vid in daily_series.columns and len(daily_series.columns) >= 2:
                other_vids = [c for c in daily_series.columns if c != cv_vid]
                ci_data = pd.concat([daily_series[[cv_vid]], daily_series[other_vids].mean(axis=1)], axis=1)
                ci_data.columns = ['y', 'x1']
                pre_start = ci_data.index.min()
                pre_end = (release_date - pd.Timedelta(days=1)).date()
                post_start = release_date.date()
                post_end = ci_data.index.max()
                
                if pre_start < pre_end and post_start <= post_end:
                    ci = CausalImpact(ci_data, [str(pre_start), str(pre_end)], [str(post_start), str(post_end)])
                    ci_summary = ci.summary_data
                    ci_summary.to_parquet(out_dir / "causal_impact_summary.parquet")
                    print("  ✅ CausalImpact counterfactual model computed for controversy event.")
        except Exception as e:
            print(f"  CausalImpact model skipped: {e}")
            
        print("✅ Cross-video and controversy data generated.")

if __name__ == "__main__":
    run_cross_video()
