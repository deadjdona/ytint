import os
import logging
import pathlib
import streamlit as st
import pandas as pd
import plotly.express as px

# ==============================================================================
# 1. LOGGING & ENVIRONMENT CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ytint_ui")

# Page Configuration
st.set_page_config(
    page_title="ytint Analytical Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust Absolute Path Anchoring
UI_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = UI_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"

logger.info(f"Initializing UI. Root Directory caught at: {ROOT_DIR}")

# ==============================================================================
# 2. CACHED DATA CORE PIPELINE LOADER
# ==============================================================================
@st.cache_data(show_spinner="Parsing analytical parquet layers from disk...")
def load_pipeline_data():
    logger.info("Attempting a fresh load of cached analytical layers.")
    
    paths = {
        "spikes": DATA_DIR / "output" / "viral_events.parquet",
        "topics": DATA_DIR / "output" / "topic_metadata.parquet",
        "timeline": DATA_DIR / "output" / "historical_timeline.parquet",
        "comments": DATA_DIR / "interim" / "comments_clean.parquet"
    }
    
    # Check for file existence before reading to throw crystal-clear logs
    missing_files = [str(p.relative_to(ROOT_DIR)) for name, p in paths.items() if name != "comments" and not p.exists()]
    if missing_files:
        logger.error(f"UI Boot Blocked. Missing target paths: {missing_files}")
        st.error(f"❌ **Data Layer Mismatch:** Missing required files: {', '.join(missing_files)}")
        st.info("💡 Please verify that your test suite passes (`pytest -v`) before booting the UI.")
        st.stop()

    try:
        logger.info("Reading viral_events.parquet...")
        df_spikes = pd.read_parquet(paths["spikes"])
        logger.info(f"Loaded viral_events successfully: {len(df_spikes)} rows found.")

        logger.info("Reading topic_metadata.parquet...")
        df_topics = pd.read_parquet(paths["topics"])
        logger.info(f"Loaded topic_metadata successfully: {len(df_topics)} rows found.")

        logger.info("Reading historical_timeline.parquet...")
        df_timeline = pd.read_parquet(paths["timeline"])
        logger.info(f"Loaded historical_timeline successfully: {len(df_timeline)} rows found.")
        
        # Optional layer: gracefully check if source comments are accessible
        df_comments = None
        if paths["comments"].exists():
            logger.info("Reading comments_clean.parquet for preview granularity...")
            df_comments = pd.read_parquet(paths["comments"])
            logger.info(f"Loaded source comments: {len(df_comments)} rows available.")
        else:
            logger.warning("Source comments_clean.parquet not found. Deep previews will be restricted.")

        return df_spikes, df_topics, df_timeline, df_comments

    except Exception as e:
        logger.exception("Fatal runtime exception encountered while reading parquet matrices.")
        st.error("⚠️ **Parquet Read Engine Failure**")
        st.exception(e)
        st.stop()

# Execution of data layer pull
df_spikes, df_topics, df_timeline, df_comments = load_pipeline_data()

# ==============================================================================
# 3. SIDEBAR PERSISTENT METRICS & FILTERS
# ==============================================================================
st.sidebar.title("🧬 `ytint` Core Engine")
st.sidebar.markdown("---")

st.sidebar.subheader("Pipeline Manifest")
processed_comments = len(df_comments) if df_comments is not None else 0
discovered_topics = len(df_topics[df_topics["Topic"] != -1]) if "Topic" in df_topics.columns else len(df_topics)
st.sidebar.metric(label="Processed Comments", value=f"{processed_comments:,}")
st.sidebar.metric(label="Discovered Topics", value=f"{discovered_topics:,}")
st.sidebar.metric(label="Identified Event Spikes", value=f"{len(df_spikes)}")

st.sidebar.markdown("---")
st.sidebar.caption("System Environment: **Python 3.14** + **CUDA 12.6 Acceleration**")

# ==============================================================================
# 4. MAIN INTERFACE LAYOUT
# ==============================================================================
st.title("📊 Conversational Topic Modeling & Narrative Dashboard")
st.markdown("Exploratory interface parsing high-density cluster structures and event spikes.")

# Layout Tabs
tab_spikes, tab_topics, tab_timeline = st.tabs([
    "🚨 Event Spikes & Volatility", 
    "🧩 Micro-Cluster Explorer", 
    "📅 Macro Timeline Engine"
])

# ------------------------------------------------------------------------------
# TAB 1: EVENT SPIKES
# ------------------------------------------------------------------------------
with tab_spikes:
    st.header(f"{len(df_spikes)} Detected Conversational Event Spikes")
    st.markdown("High-volume temporal anomalies isolated automatically via density analysis pipelines.")
    
    if not df_spikes.empty:
        # Check if timeline columns exist for a line/bar chart visualization
        spikes_for_chart = df_spikes.reset_index()
        date_col = next((c for c in spikes_for_chart.columns if "date" in c.lower() or "timestamp" in c.lower()), None)
        count_col = next((c for c in df_spikes.columns if "count" in c.lower() or "volume" in c.lower() or "size" in c.lower()), None)
        
        if date_col and count_col:
            fig_spikes = px.bar(
                spikes_for_chart, x=date_col, y=count_col, 
                title="Spike Intensity Metric",
                labels={date_col: "Date Vector", count_col: "Volume Weight"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_spikes, use_container_width=True)
        
        st.subheader("Raw Layer Inspection: `viral_events`")
        st.dataframe(df_spikes, use_container_width=True)
    else:
        st.info("The viral events matrix parsed successfully but returned empty rows.")

# ------------------------------------------------------------------------------
# TAB 2: MICRO-CLUSTER EXPLORER
# ------------------------------------------------------------------------------
with tab_topics:
    st.header("Discovered Conversational Clusters")
    st.markdown("Granular semantic pockets grouped via high-speed UMAP dimensionality reduction and HDBSCAN.")
    
    # NEW: Topic Resonance Scatter Plot
    if "avg_likes" in df_topics.columns and not df_topics.empty:
        # Exclude the -1 outlier topic to keep the chart scale readable
        plot_data = df_topics[df_topics["Topic"] != -1] 
        
        fig_resonance = px.scatter(
            plot_data,
            x="Count",
            y="avg_likes",
            size="total_likes",
            hover_name="Name",
            title="Topic Resonance Matrix (Volume vs. Engagement)",
            labels={"Count": "Total Comments in Topic", "avg_likes": "Average Likes per Comment"},
            template="plotly_dark",
            size_max=40
        )
        st.plotly_chart(fig_resonance, use_container_width=True)
            
    # Simple keyword search block
    search_query = st.text_input("🔍 Filter clusters by keyword/topic token:", "").strip().lower()
    
    filtered_topics = df_topics.copy()
    if search_query:
        # Generic string search across text columns
        text_cols = [c for c in filtered_topics.columns if filtered_topics[c].dtype == 'object']
        if text_cols:
            mask = filtered_topics[text_cols].astype(str).apply(lambda x: x.str.lower().str.contains(search_query)).any(axis=1)
            filtered_topics = filtered_topics[mask]
            logger.info(f"Applied keyword filter '{search_query}'. Matches remaining: {len(filtered_topics)}")
            
    st.metric(label="Filtered Cluster Count", value=len(filtered_topics))
    st.dataframe(filtered_topics, use_container_width=True)

# NEW: Emotional Polarity Stacked Bar Chart
    # Ensure the columns exist before rendering to prevent UI crashes on old caches
    if "pct_negative" in df_topics.columns and not df_topics.empty:
        st.subheader("Emotional Polarity & Toxicity Matrix")
        
        # Exclude outlier cluster (-1) and grab the top 20 most toxic topics
        plot_data = df_topics[df_topics["Topic"] != -1].sort_values("pct_negative", ascending=False).head(20)
        
        # Dynamically grab the sentiment columns we generated
        sentiment_cols = [c for c in df_topics.columns if c.startswith("pct_")]
        
        # Force a recognizable color palette (Red=Negative, Gray/Blue=Neutral, Green=Positive)
        color_map = {"pct_negative": "#ef553b", "pct_neutral": "#636efa", "pct_positive": "#00cc96"}
        
        fig_sentiment = px.bar(
            plot_data,
            x="Name",
            y=sentiment_cols,
            title="Top 20 Topics Ranked by Negativity (Toxicity Index)",
            labels={"value": "Percentage (%)", "Name": "Topic Name", "variable": "Sentiment"},
            template="plotly_dark",
            barmode="stack",
            color_discrete_map=color_map
        )
        # Clean up the hover legend layout
        fig_sentiment.update_layout(legend_title_text="Sentiment Type")
        st.plotly_chart(fig_sentiment, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: MACRO TIMELINE ENGINE
# ------------------------------------------------------------------------------
with tab_timeline:
    st.header("Macro Historical Timelines")
    st.markdown("Longitudinal baseline eras across the full data collection scope.")
    
    # NEW: Visualize the macro timeline
    if not df_timeline.empty:
        fig_timeline = px.line(
            df_timeline, 
            x="date", 
            y="comment_count", 
            title="Total Conversational Volume Over Time",
            template="plotly_dark"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.subheader("Raw Layer Inspection: `historical_timeline`")
    st.dataframe(df_timeline, use_container_width=True)