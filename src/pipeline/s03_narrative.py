import os
import yaml
import pandas as pd
from src.engine.metrics import calculate_aging_scores, calculate_controversy_index
from pathlib import Path

# Import our mathematical algorithms directly from the engine layer
from src.engine.stats import detect_volume_anomalies, segment_thematic_eras

def load_config(config_path="config/settings.yaml"):
    """Loads global project configurations."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compile_narrative_spine():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    final_dir = Path(config["paths"]["final_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)
    
    comments_file = interim_dir / "comments_clean.parquet"
    videos_file = interim_dir / "videos_clean.parquet"
    
    if not comments_file.exists() or not videos_file.exists():
        print("❌ Analytical data layers missing. Ensure ingestion, enrichment, and topic stages are complete.")
        return

    print("📥 Loading computational layers for timeline compilation...")
    df_comments = pd.read_parquet(comments_file)
    df_videos = pd.read_parquet(videos_file)
    
    # Ensure uniform datetime formats across datasets
    df_comments['published_at'] = pd.to_datetime(df_comments['published_at'])
    df_videos['published_at'] = pd.to_datetime(df_videos['published_at'])
    
    timeline_start = df_comments['published_at'].min()
    timeline_end = df_comments['published_at'].max()
    print(f"🗓️ Analyzing community trajectory from {timeline_start.strftime('%Y-%m-%d')} to {timeline_end.strftime('%Y-%m-%d')}")

    # ---------------------------------------------------------------------
    # Step 1: Detect Event Pins (Spikes in Engagement Velocity)
    # ---------------------------------------------------------------------
    print("🔍 Scanning for major engagement velocity disruptions (Events)...")
    event_settings = config["stage_03_narrative"]["events"]
    anomalies = detect_volume_anomalies(
        df_comments, 
        z_threshold=event_settings["z_score_threshold"],
        window_days=event_settings["rolling_window_days"]
    )
    
    events_compiled = []
    for _, row in anomalies.iterrows():
        target_date = row['published_at']
        
        # Isolate the exact comments published during this spike day
        day_comments = df_comments[df_comments['published_at'].dt.date == target_date]
        
        # Find the single most impactful comment (highest like count) as the definitive artifact
        top_comment = "No text captured."
        if not day_comments.empty:
            top_comment = day_comments.sort_values(by='like_count', ascending=False).iloc[0]['text']
            
        # Isolate the video released closest to this spike date to establish visual context
        df_videos['time_delta'] = (df_videos['published_at'] - pd.to_datetime(target_date)).abs()
        closest_video = df_videos.sort_values('time_delta').iloc[0]
        
        events_compiled.append({
            "event_date": pd.to_datetime(target_date),
            "comment_volume": row['comment_count'],
            "z_score": row['z_score'],
            "associated_video": closest_video['title'],
            "representative_artifact": top_comment
        })
        
    df_events = pd.DataFrame(events_compiled)
    
    # ---------------------------------------------------------------------
    # Step 2: Segment Historical Strata (Thematic Eras)
    # ---------------------------------------------------------------------
    print("🔀 Running variance segmentation to map core structural shifts (Eras)...")
    era_settings = config["stage_03_narrative"]["eras"]
    
    boundary_dates = segment_thematic_eras(
        df_comments,
        min_duration_weeks=era_settings["min_duration_days"] // 7,
        penalty_modifier=era_settings["penalty"]
    )
    
    # Append outer timeline limits to structure complete chronological blocks
    all_milestones = sorted(list(set([timeline_start] + boundary_dates + [timeline_end])))
    
    eras_compiled = []
    for idx in range(len(all_milestones) - 1):
        start_dt = all_milestones[idx]
        end_dt = all_milestones[idx+1]
        
        # Isolate comments inside this specific era's time window
        era_mask = (df_comments['published_at'] >= start_dt) & (df_comments['published_at'] < end_dt)
        era_comments = df_comments[era_mask]
        
        # Determine the dominant topic dominating this specific era
        dominant_label = "Undefined thematic substrate"
        if not era_comments.empty and not era_comments[era_comments['topic_id'] != -1].empty:
            dominant_label = era_comments[era_comments['topic_id'] != -1]['topic_label'].value_counts().idxmax()
            
        # Calculate the base sentiment makeup inside this time bracket
        sentiment_mix = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        if not era_comments.empty:
            mix_pct = era_comments['sentiment_label'].value_counts(normalize=True).to_dict()
            sentiment_mix.update(mix_pct)
            
        eras_compiled.append({
            "era_id": idx + 1,
            "start_date": start_dt,
            "end_date": end_dt,
            "dominant_theme": dominant_label,
            "pct_positive": sentiment_mix.get("positive", 0.0),
            "pct_neutral": sentiment_mix.get("neutral", 0.0),
            "pct_negative": sentiment_mix.get("negative", 0.0),
        })
        
    df_eras = pd.DataFrame(eras_compiled)
# Add this block inside compile_narrative_spine() right after Step 2 completes
    # ---------------------------------------------------------------------
# Step 2.5: Parse Advanced Memory and Friction Metrics
    # ---------------------------------------------------------------------
    print("📐 Computing deep archaeological memory aging and controversy metrics...")
    df_aging = calculate_aging_scores(df_comments)
    df_controversy = calculate_controversy_index(df_comments)

    # Merge back into final data tables for structural storage lookup
    if not df_aging.empty:
        df_aging.to_parquet(final_dir / "compiled_aging.parquet", index=False)
    if not df_controversy.empty:
        df_controversy.to_parquet(final_dir / "compiled_controversy.parquet", index=False)

    # ---------------------------------------------------------------------
    # Step 3: Persist Compiled Narrative Structures
    # ---------------------------------------------------------------------
    print("💾 Archiving compiled historical landmarks into the final analytical directory...")
    
    if not df_events.empty:
        df_events.to_parquet(final_dir / "compiled_events.parquet", index=False)
    else:
        # Save a clean, empty placeholder framework to avoid breaking the UI layout
        pd.DataFrame(columns=["event_date", "comment_volume", "z_score", "associated_video", "representative_artifact"]).to_parquet(final_dir / "compiled_events.parquet", index=False)
        
    df_eras.to_parquet(final_dir / "compiled_eras.parquet", index=False)
    
    print(f"📦 Historical compilation successful.")
    print(f"   ├─ Detected {len(df_events)} high-impact event spikes.")
    print(f"   └─ Segmented the timeline into {len(df_eras)} distinct thematic eras.")

if __name__ == "__main__":
    compile_narrative_spine()