import streamlit as st
import pandas as pd
import yaml
import pathlib
import plotly.express as px

# 1. Page Configuration Setup
st.set_page_config(
    page_title="ytint // Analytics Engine Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_config():
    """
    Dynamically resolves the project root directory and loads the unified settings.
    Ensures data asset loading remains aligned under any execution context.
    """
    current_file = pathlib.Path(__file__).resolve()
    root_dir = current_file.parent
    while root_dir != root_dir.parent:
        if (root_dir / "config").is_dir():
            break
        root_dir = root_dir.parent
        
    config_path = root_dir / "config" / "settings.yaml"
    if not config_path.exists():
        st.error(f"❌ Could not find configuration file at: {config_path}")
        st.stop()
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Translate relative paths into absolute paths anchored to the project root
    config["paths"]["interim_dir"] = root_dir / config["paths"]["interim_dir"]
    config["paths"]["output_dir"] = root_dir / config["paths"]["output_dir"]
    return config

@st.cache_data
def load_pipeline_data(interim_dir, output_dir):
    """Loads clean master data layers and topic dimensions into memory."""
    try:
        comments = pd.read_parquet(interim_dir / "comments_clean.parquet")
        topics = pd.read_parquet(output_dir / "topic_metadata.parquet")
        return comments, topics
    except Exception as e:
        st.error(f"❌ Missing Data Artifacts: Ensure you run the orchestrator runner completely first. Details: {e}")
        st.stop()

# --- Initialization Layer ---
config = load_config()
interim_path = config["paths"]["interim_dir"]
output_path = config["paths"]["output_dir"]

df_comments_raw, df_topics = load_pipeline_data(interim_path, output_path)

# --- Sidebar Controls & Interaction Layer ---
st.sidebar.header("🎛️ Dashboard Configurations")

# Interactive Thread Segmentation Filter
comment_filter = st.sidebar.selectbox(
    "💬 Comment Layer Segmentation",
    options=["Show All Records", "Top-level Comments Only", "Replies/Responses Only"]
)

st.sidebar.divider()
st.sidebar.markdown("**Backend Architecture Metrics:**")
st.sidebar.info(
    f"**Z-Score Threshold:** {config['stage_03_narrative']['z_threshold']}\n\n"
    f"**Pelt Penalty Value:** {config['stage_03_narrative']['change_point_penalty']}"
)

# --- Dynamic Processing Slice Engine ---
# Segment rows according to parent_id tracking signatures
if comment_filter == "Top-level Comments Only":
    df_filtered = df_comments_raw[df_comments_raw['parent_id'].isna() | (df_comments_raw['parent_id'] == "")]
elif comment_filter == "Replies/Responses Only":
    df_filtered = df_comments_raw[df_comments_raw['parent_id'].notna() & (df_comments_raw['parent_id'] != "")]
else:
    df_filtered = df_comments_raw

# Standardize date types on the filtered dataset slice
if not df_filtered.empty:
    df_filtered = df_filtered.copy()
    df_filtered['published_at'] = pd.to_datetime(df_filtered['published_at'])
    
    # On-the-fly Daily Timeline Aggregation
    df_timeline = df_filtered.groupby(df_filtered['published_at'].dt.date).size().to_frame(name='comment_count')
    df_timeline.index = pd.to_datetime(df_timeline.index)
    df_timeline = df_timeline.sort_index().reset_index().rename(columns={'published_at': 'date'})
    
    # On-the-fly Rolling Z-Score Anomaly Scanner
    rolling_mean = df_timeline['comment_count'].rolling(window=7, min_periods=1).mean()
    rolling_std = df_timeline['comment_count'].rolling(window=7, min_periods=1).std().fillna(1)
    df_timeline['z_score'] = (df_timeline['comment_count'] - rolling_mean) / rolling_std
    
    z_thresh = config["stage_03_narrative"]["z_threshold"]
    df_events = df_timeline[df_timeline['z_score'] > z_thresh].copy()
    
    # On-the-fly Topic Frequency Resizing
    if 'topic' in df_filtered.columns:
        slice_counts = df_filtered['topic'].value_counts().reset_index()
        slice_counts.columns = ['Topic', 'Filtered_Count']
        df_topics_filtered = df_topics.merge(slice_counts, on='Topic', how='inner')
        df_topics_filtered = df_topics_filtered[df_topics_filtered['Topic'] != -1] # Filter noise cluster
        df_topics_filtered = df_topics_filtered.sort_values(by='Filtered_Count', ascending=False)
    else:
        df_topics_filtered = pd.DataFrame(columns=['Topic', 'Count', 'Name', 'Filtered_Count'])
else:
    df_timeline = pd.DataFrame(columns=['date', 'comment_count', 'z_score'])
    df_events = pd.DataFrame(columns=['date', 'comment_count', 'z_score'])
    df_topics_filtered = pd.DataFrame(columns=['Topic', 'Count', 'Name', 'Filtered_Count'])

# --- Header Section ---
st.title("🎬 ytint: YouTube Intelligence Dashboard")
st.markdown(f"Currently displaying: **{comment_filter}** ({len(df_filtered):,} total records matching filters)")
st.divider()

# --- Metric Row ---
total_comments = int(df_timeline['comment_count'].sum()) if not df_timeline.empty else 0
total_topics = len(df_topics_filtered)
total_spikes = len(df_events)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📊 Slice Processed Comments", value=f"{total_comments:,}")
with col2:
    st.metric(label="🧩 Active Semantic Topics", value=f"{total_topics:,}")
with col3:
    st.metric(label="⚡ Filtered Volumetric Spikes", value=f"{total_spikes}")

st.divider()

# --- Temporal Narrative Row ---
st.header("📈 Dynamic Volume Trajectory Map")

if not df_timeline.empty:
    fig_timeline = px.line(
        df_timeline, 
        x='date', 
        y='comment_count', 
        title="Chronological Filtered Conversation Density Timeline",
        labels={'date': 'Timeline Execution Date', 'comment_count': 'Captured Volume'},
        line_shape='spline',
        render_mode='svg'
    )
    fig_timeline.update_traces(line_color='#FF4B4B', line_width=2.5)
    fig_timeline.update_layout(
        hovermode="x unified",
        xaxis_gridcolor="#eeeeee",
        yaxis_gridcolor="#eeeeee",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.warning("⚠️ No data records mapped inside this selected time slice timeline configuration.")

# --- Dual Layout Blocks: Topics vs Anomalies ---
left_col, right_col = st.columns(2)

with left_col:
    st.header("🧩 Targeted Semantic Topic Registry")
    st.markdown("Context groupings discovered inside this layer slice, ordered by volume concentration.")
    
    if not df_topics_filtered.empty:
        display_topics = df_topics_filtered.rename(columns={
            'Topic': 'Cluster ID',
            'Filtered_Count': 'Group Size (This Slice)',
            'Name': 'Primary Representation Keywords'
        })
        st.dataframe(
            display_topics[['Cluster ID', 'Group Size (This Slice)', 'Primary Representation Keywords']], 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("ℹ️ No thematic clusters identified inside this layer partition sequence.")

with right_col:
    st.header("⚡ Anomaly Flashpoint Ledger")
    st.markdown("Chronological logs of structural threshold violations calculated within this context.")
    
    if not df_events.empty:
        # Clean columns for presentation display
        clean_events = df_events.copy().rename(columns={
            'date': 'Spike Date',
            'comment_count': 'Comments Registered',
            'z_score': 'Breach Score Severity'
        })
        st.dataframe(
            clean_events[['Spike Date', 'Comments Registered', 'Breach Score Severity']], 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info(
            f"ℹ️ Smooth Density: Zero active dates crossed the target "
            f"Z-score parameter limits ({config['stage_03_narrative']['z_threshold']}) in this segmented layer context."
        )