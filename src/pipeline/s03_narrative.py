import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import ruptures as rpt

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def compile_narrative():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Loading enriched comment tables...")
    df_comments = pd.read_parquet(interim_dir / "comments_clean.parquet")
    
    print("🔧 Examining timestamp scales...")
    # Coerce to datetime format to check the actual calendar years
    dt_series = pd.to_datetime(df_comments['published_at'], errors='coerce')
    sample_years = dt_series.dropna().dt.year
    
    if not sample_years.empty and sample_years.iloc[0] > 3000:
        print("⏱️ Detected year 50000+ anomaly. Recalculating timeline via Epoch-Delta...")
        # Compute exact seconds from 1970 (which matches the original raw millisecond value)
        # This is safe against any internal pandas timezone or resolution configurations
        epoch = pd.Timestamp('1970-01-01', tz=dt_series.dt.tz)
        original_ms_ticks = (dt_series - epoch).dt.total_seconds()
        
        # Re-parse the ticks using the correct millisecond mapping
        df_comments['published_at'] = pd.to_datetime(original_ms_ticks, unit='ms', errors='coerce')
    else:
        # Fallback for raw unparsed numeric tokens
        raw_ticks = pd.to_numeric(df_comments['published_at'], errors='coerce')
        sample_series = raw_ticks.dropna()
        if not sample_series.empty:
            sample_val = sample_series.iloc[0]
            if 1e11 < sample_val < 1e14:
                print("⏱️ Corrected raw millisecond-scale tokens...")
                df_comments['published_at'] = pd.to_datetime(raw_ticks, unit='ms', errors='coerce')
            elif 1e8 < sample_val < 1e11:
                print("⏱️ Timestamps verified as standard second-scale...")
                df_comments['published_at'] = pd.to_datetime(raw_ticks, unit='s', errors='coerce')
            else:
                df_comments['published_at'] = dt_series
        else:
            df_comments['published_at'] = dt_series

    # Print data verification diagnostic
    print(f"📅 Real-world date range detected: {df_comments['published_at'].min()} to {df_comments['published_at'].max()}")
    
    # 🛡️ Guard Rail: Clean up any genuinely corrupted rows outside YouTube's operational window
    initial_count = len(df_comments)
    df_comments = df_comments[
        (df_comments['published_at'] >= '2005-01-01') & 
        (df_comments['published_at'] <= '2030-01-01')
    ]
    dropped_count = initial_count - len(df_comments)
    if dropped_count > 0:
        print(f"🧹 Cleaned up {dropped_count} rows with actual corrupt dates.")
    
    print(f"📊 Ready to process {len(df_comments)} comments on the timeline.")
    
    # Aggregate timelines by day safely
    timeline = df_comments.groupby(df_comments['published_at'].dt.date).size().to_frame(name='comment_count')
    
    if timeline.empty:
        print("❌ Error: No valid dates found after filtering timeline data.")
        return

    # Run Change-Point Detection (Pelt algorithm) to mark community Eras
    print("📉 Calculating structural historical change-points...")
    signal = timeline['comment_count'].values
    algo = rpt.Pelt(model="rbf").fit(signal)
    
    # Pull penalty criteria from settings.yaml configuration
    penalty = config["stage_03_narrative"]["change_point_penalty"]
    change_points = algo.predict(pen=penalty)
    
    # Compute rolling volume Z-scores to isolate anomaly spikes ("Events")
    print("⚡ Evaluating historical volume anomaly thresholds...")
    rolling_mean = timeline['comment_count'].rolling(window=7, min_periods=1).mean()
    rolling_std = timeline['comment_count'].rolling(window=7, min_periods=1).std().fillna(1)
    timeline['z_score'] = (timeline['comment_count'] - rolling_mean) / rolling_std
    
    z_thresh = config["stage_03_narrative"]["z_threshold"]
    events = timeline[timeline['z_score'] > z_thresh].copy()
    
    # Save the analytical outputs out to the distribution folder
    timeline.to_parquet(output_dir / "historical_timeline.parquet")
    events.to_parquet(output_dir / "viral_events.parquet")
    
    print(f"✅ Stage 03 Core Narrative Compilation Complete! Mapped {len(change_points)-1} Eras and {len(events)} Events.")

if __name__ == "__main__":
    compile_narrative()