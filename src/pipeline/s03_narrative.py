import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import ruptures as rpt
from tqdm import tqdm
import time
import sys
# Make 'engine' importable whether this runs standalone or via runner.py
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))
from engine.changepoint import select_pelt_penalty

def load_config():
    """
    Dynamically resolves the project root directory and loads the unified settings.
    Ensures absolute path compatibility across all execution environments.
    """
    current_file = Path(__file__).resolve()
    
    # Walk upward until we locate the parent directory containing the 'config' folder
    root_dir = current_file.parent
    while root_dir != root_dir.parent:
        if (root_dir / "config").is_dir():
            break
        root_dir = root_dir.parent
        
    config_path = root_dir / "config" / "settings.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"❌ Critical Configuration Alignment Failure:\n"
            f"Could not locate 'config/settings.yaml'.\n"
            f"Resolved root searched: {root_dir}"
        )
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Translate relative paths into absolute paths anchored to the project root
    config["paths"]["raw_db"] = str(root_dir / config["paths"]["raw_db"])
    config["paths"]["interim_dir"] = str(root_dir / config["paths"]["interim_dir"])
    config["paths"]["output_dir"] = str(root_dir / config["paths"]["output_dir"])
    
    return config

def compile_narrative():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define pipeline execution phases for the progress tracker
    phases = [
        "Loading enriched tables",
        "Examining timestamp scales",
        "Translating timeline metrics",
        "Filtering date guardrails",
        "Aggregating daily timeline",
        "Calculating structural change-points",
        "Evaluating volume anomalies",
        "Writing analytical layers"
    ]
    
    # Initialize the master pipeline progress bar
    with tqdm(total=len(phases), desc="🎬 Initializing Stage 03", bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}") as pbar:
        
        # Phase 1: Load Data
        pbar.set_description(f"📥 {phases[0]}")
        df_comments = pd.read_parquet(interim_dir / "comments_clean.parquet")
        pbar.update(1)
        
        # Phase 2: Check Scales
        pbar.set_description(f"🔧 {phases[1]}")
        dt_series = pd.to_datetime(df_comments['published_at'], errors='coerce')
        sample_years = dt_series.dropna().dt.year
        pbar.update(1)
        
        # Phase 3: Translate Epoch-Delta Anomaly
        pbar.set_description(f"⏱️ {phases[2]}")
        if not sample_years.empty and sample_years.iloc[0] > 3000:
            epoch = pd.Timestamp('1970-01-01', tz=dt_series.dt.tz)
            original_ms_ticks = (dt_series - epoch).dt.total_seconds()
            df_comments['published_at'] = pd.to_datetime(original_ms_ticks, unit='ms', errors='coerce')
        else:
            raw_ticks = pd.to_numeric(df_comments['published_at'], errors='coerce')
            sample_series = raw_ticks.dropna()
            if not sample_series.empty:
                sample_val = sample_series.iloc[0]
                if 1e11 < sample_val < 1e14:
                    df_comments['published_at'] = pd.to_datetime(raw_ticks, unit='ms', errors='coerce')
                elif 1e8 < sample_val < 1e11:
                    df_comments['published_at'] = pd.to_datetime(raw_ticks, unit='s', errors='coerce')
                else:
                    df_comments['published_at'] = dt_series
            else:
                df_comments['published_at'] = dt_series
        pbar.update(1)
        
        # Phase 4: Guardrails
        pbar.set_description(f"🧹 {phases[3]}")
        df_comments = df_comments[
            (df_comments['published_at'] >= '2005-01-01') & 
            (df_comments['published_at'] <= '2030-01-01')
        ]
        pbar.update(1)
        
        # Phase 5: Aggregation & Serialization Standardization
        pbar.set_description(f"📊 {phases[4]}")
        timeline = df_comments.groupby(df_comments['published_at'].dt.date).size().to_frame(name='comment_count')
        if timeline.empty:
            print("\n❌ Error: No valid dates found after filtering timeline data.")
            return
            
        # Convert index from python date objects to datetime64[ns] and flatten to a named column
        timeline.index = pd.to_datetime(timeline.index)
        timeline = timeline.sort_index().reset_index().rename(columns={'published_at': 'date'})
        pbar.update(1)
        
        # Phase 6: Change-Point Detection (Pelt)
        pbar.set_description(f"📉 {phases[5]}")
        signal = timeline['comment_count'].values
        algo = rpt.Pelt(model="rbf").fit(signal)
        # A fixed penalty only suits one dataset's noise/scale and won't
        # generalize across channels of different comment volume. "auto"
        # calibrates it per-run via the breakpoint-count elbow method;
        # set a numeric value in config to force a fixed penalty instead.
        penalty_cfg = config["stage_03_narrative"]["change_point_penalty"]
        if isinstance(penalty_cfg, str) and penalty_cfg.strip().lower() == "auto":
            penalty, _ = select_pelt_penalty(signal)
            print(f"🎯 Auto-selected change-point penalty: {penalty:.2f}")
        else:
            penalty = float(penalty_cfg)
        change_points = algo.predict(pen=penalty)
        pbar.update(1)
        
        # Phase 7: Volumetric Anomalies (Z-Score scanning)
        pbar.set_description(f"⚡ {phases[6]}")
        # Baseline must exclude the day being scored. A trailing rolling
        # window that includes the current point drags its own mean/std
        # toward the spike, suppressing its own z-score. Validated: a clean
        # 4.5x-baseline spike scored z~2.26 (below the 2.5 threshold, missed)
        # with the self-inclusive window vs z~21-32 (correctly flagged) once
        # the current point is excluded from its own baseline.
        prior = timeline['comment_count'].shift(1)
        rolling_mean = prior.rolling(window=7, min_periods=3).mean()
        rolling_std = prior.rolling(window=7, min_periods=3).std()
        timeline['z_score'] = (timeline['comment_count'] - rolling_mean) / rolling_std
        z_thresh = config["stage_03_narrative"]["z_threshold"]
        events = timeline[timeline['z_score'] > z_thresh].copy()
        pbar.update(1)
        
        # Phase 8: Save Deliverables (Clean Arrow Columns with index flattening)
        pbar.set_description(f"💾 {phases[7]}")
        timeline.to_parquet(output_dir / "historical_timeline.parquet", index=False)
        events.to_parquet(output_dir / "viral_events.parquet", index=False)
        pbar.update(1)
        
        # Final descriptive flag on completion
        pbar.set_description("✅ Stage 03 Complete")

    # Final Summary Diagnostics Printout
    print("\n" + "="*60)
    print(f"📅 Verified Date Range : {df_comments['published_at'].min()} to {df_comments['published_at'].max()}")
    print(f"📊 Processed Records   : {len(df_comments)} comments")
    print(f"🧩 Structural Layout   : Mapped {len(change_points)-1} Eras and {len(events)} Event Spikes.")
    print("="*60)

if __name__ == "__main__":
    compile_narrative()