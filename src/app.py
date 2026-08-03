import os
import logging
import pathlib
import streamlit as st
import pandas as pd
import plotly.express as px

from engine.config_loader import load_config as _load_config

# Page Configuration Setup
st.set_page_config(
    page_title="ytint // Analytics Engine Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_config():
    """Cached wrapper around the shared config loader."""
    config = _load_config()
    config["paths"]["interim_dir"] = pathlib.Path(config["paths"]["interim_dir"])
    config["paths"]["output_dir"] = pathlib.Path(config["paths"]["output_dir"])
    return config

@st.cache_data
def load_pipeline_data(interim_dir, output_dir):
    """Loads clean master data layers and topic dimensions into memory."""
    try:
        comments = pd.read_parquet(interim_dir / "comments_clean.parquet")
        topics = pd.read_parquet(output_dir / "topic_metadata.parquet")
        
        extra_layers = {}
        for key, fname in [
            ("author_fingerprints", "author_fingerprints.parquet"),
            ("video_radar", "video_profile_radar.parquet"),
            ("code_switching", "code_switching_impact.parquet"),
            ("topic_evolution", "topic_evolution.parquet"),
            ("reaction_map", "video_reaction_map.parquet"),
            ("dunn_matrix", "dunn_posthoc_matrix.parquet"),
            ("power_law", "power_law_fit.parquet")
        ]:
            fpath = output_dir / fname
            if fpath.exists():
                extra_layers[key] = pd.read_parquet(fpath)
                
        return comments, topics, extra_layers
    except Exception as e:
        st.error(f"❌ Missing Data Artifacts: Ensure you run the orchestrator runner completely first. Details: {e}")
        st.stop()

# Initialization Layer
config = load_config()
interim_path = config["paths"]["interim_dir"]
output_path = config["paths"]["output_dir"]

df_comments_raw, df_topics, extra_layers = load_pipeline_data(interim_path, output_path)

# Sidebar Controls
st.sidebar.header("🎛️ Dashboard Configurations")

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

if "power_law" in extra_layers and not extra_layers["power_law"].empty:
    pl_df = extra_layers["power_law"]
    if "alpha" in pl_df.columns:
        st.sidebar.metric(label="Power-Law Exponent (α)", value=f"{pl_df['alpha'].iloc[0]:.3f}")

# Dynamic Processing Slice
if comment_filter == "Top-level Comments Only":
    df_filtered = df_comments_raw[df_comments_raw['parent_id'].isna() | (df_comments_raw['parent_id'] == "")]
elif comment_filter == "Replies/Responses Only":
    df_filtered = df_comments_raw[df_comments_raw['parent_id'].notna() & (df_comments_raw['parent_id'] != "")]
else:
    df_filtered = df_comments_raw

if not df_filtered.empty:
    df_filtered = df_filtered.copy()
    df_filtered['published_at'] = pd.to_datetime(df_filtered['published_at'])
    
    df_timeline = df_filtered.groupby(df_filtered['published_at'].dt.date).size().to_frame(name='comment_count')
    df_timeline.index = pd.to_datetime(df_timeline.index)
    df_timeline = df_timeline.sort_index().reset_index().rename(columns={'published_at': 'date'})
    
    rolling_mean = df_timeline['comment_count'].rolling(window=7, min_periods=1).mean()
    rolling_std = df_timeline['comment_count'].rolling(window=7, min_periods=1).std().fillna(1)
    df_timeline['z_score'] = (df_timeline['comment_count'] - rolling_mean) / rolling_std
    
    z_thresh = config["stage_03_narrative"]["z_threshold"]
    df_events = df_timeline[df_timeline['z_score'] > z_thresh].copy()
    
    if 'topic' in df_filtered.columns:
        slice_counts = df_filtered['topic'].value_counts().reset_index()
        slice_counts.columns = ['Topic', 'Filtered_Count']
        df_topics_filtered = df_topics.merge(slice_counts, on='Topic', how='inner')
        df_topics_filtered = df_topics_filtered[df_topics_filtered['Topic'] != -1]
        df_topics_filtered = df_topics_filtered.sort_values(by='Filtered_Count', ascending=False)
    else:
        df_topics_filtered = pd.DataFrame(columns=['Topic', 'Count', 'Name', 'Filtered_Count'])
else:
    df_timeline = pd.DataFrame(columns=['date', 'comment_count', 'z_score'])
    df_events = pd.DataFrame(columns=['date', 'comment_count', 'z_score'])
    df_topics_filtered = pd.DataFrame(columns=['Topic', 'Count', 'Name', 'Filtered_Count'])

# Header Section
st.title("🎬 ytint: YouTube Intelligence Dashboard")
st.markdown(f"Currently displaying: **{comment_filter}** ({len(df_filtered):,} total records matching filters)")
st.divider()

# Metric Row
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

# Temporal Narrative Row
st.header("📈 Dynamic Volume Trajectory Map")

if not df_timeline.empty:
    fig_timeline = px.line(
        df_timeline, 
        x='date', 
        y='comment_count', 
        title="Chronological Filtered Conversation Density Timeline",
        labels={'date': 'Timeline Execution Date', 'comment_count': 'Captured Volume'},
        line_shape='spline',
        template="plotly_dark"
    )
    fig_timeline.update_traces(line_color='#FF4B4B', line_width=2.5)
    st.plotly_chart(fig_timeline, width="stretch")
else:
    st.warning("⚠️ No data records mapped inside this selected time slice timeline configuration.")

# Interactive Topic Evolution Area Chart
if "topic_evolution" in extra_layers and not extra_layers["topic_evolution"].empty:
    st.header("📈 Topic Share Evolution Over Time")
    df_te = extra_layers["topic_evolution"]
    if "year_month" in df_te.columns:
        fig_te = px.area(
            df_te,
            x="year_month",
            y=[c for c in df_te.columns if c != "year_month"],
            title="Monthly Topic Volume Dynamics",
            template="plotly_dark"
        )
        st.plotly_chart(fig_te, width="stretch")

# Interactive Author Behavioral Radar
if "author_fingerprints" in extra_layers and not extra_layers["author_fingerprints"].empty:
    st.header("🕸️ Top Author Behavioral Fingerprints")
    df_af = extra_layers["author_fingerprints"]
    norm_cols = [c for c in df_af.columns if c.endswith("_norm")]
    if norm_cols:
        df_af_melt = df_af.melt(
            id_vars=["author_channel_id"], 
            value_vars=norm_cols, 
            var_name="Dimension", 
            value_name="Normalized_Score"
        )
        df_af_melt["Dimension"] = df_af_melt["Dimension"].str.replace("_norm", "")
        fig_af_radar = px.line_polar(
            df_af_melt, 
            r="Normalized_Score", 
            theta="Dimension", 
            color="author_channel_id",
            line_close=True,
            title="Super-Fan Behavioral Footprints",
            template="plotly_dark"
        )
        st.plotly_chart(fig_af_radar, width="stretch")

# Interactive Code-Switching & Dunn's Matrix
left_col, right_col = st.columns(2)

with left_col:
    st.header("🧩 Targeted Semantic Topic Registry")
    if not df_topics_filtered.empty:
        display_topics = df_topics_filtered.rename(columns={
            'Topic': 'Cluster ID',
            'Filtered_Count': 'Group Size (This Slice)',
            'Name': 'Primary Representation Keywords'
        })
        st.dataframe(
            display_topics[['Cluster ID', 'Group Size (This Slice)', 'Primary Representation Keywords']], 
            width="stretch", 
            hide_index=True
        )
    else:
        st.info("ℹ️ No thematic clusters identified inside this layer partition sequence.")

with right_col:
    st.header("⚡ Anomaly Flashpoint Ledger")
    if not df_events.empty:
        clean_events = df_events.copy().rename(columns={
            'date': 'Spike Date',
            'comment_count': 'Comments Registered',
            'z_score': 'Breach Score Severity'
        })
        st.dataframe(
            clean_events[['Spike Date', 'Comments Registered', 'Breach Score Severity']], 
            width="stretch", 
            hide_index=True
        )
    else:
        st.info("ℹ️ Smooth Density: Zero active dates crossed target Z-score limits.")

# Publication-Ready Visual Analytics Gallery
st.divider()
st.header("🖼️ Stage 06 Publication-Ready Visual Analytics Gallery")
st.markdown("All 14 analytical plots and interactive models rendered by `s06_visualize.py`.")

plots_dir = output_path / "plots"
import streamlit.components.v1 as components

def render_plot_card(title, filename, description, is_html=False):
    st.subheader(title)
    st.caption(description)
    filepath = plots_dir / filename
    if filepath.exists():
        if is_html:
            with open(filepath, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=600, scrolling=True)
        else:
            st.image(str(filepath), width="stretch")
    else:
        st.info(f"ℹ️ Artifact `{filename}` not generated yet. Run `python -m pipeline.runner` to render.")

g_col1, g_col2 = st.columns(2)
with g_col1:
    render_plot_card("1. Kaplan-Meier Thread Survival Estimate", "kaplan_meier_survival.png", "Task 1: Predicts conversation death/lifespan probability over time.")
    render_plot_card("3. Author Zipf Pareto Distribution", "author_pareto.png", "Task 3: Zipf power-law distribution of author contribution volume.")
    render_plot_card("5. 24×7 Diurnal Activity Heatmap", "diurnal_heatmap.png", "Task 5: Temporal activity matrix by Hour of Day × Day of Week.")
    render_plot_card("7. Sentiment Ridge Plot (Joyplot)", "sentiment_ridges.png", "Task 7: Distribution of VADER sentiment across top videos.")
    render_plot_card("9. Interactive 3D RFM Author Segmentation", "rfm_3d.html", "Task 9: Interactive 3D scatter plot of Recency, Frequency & Monetary engagement.", is_html=True)
    render_plot_card("11. Toxicity Time-of-Day Heatmap", "toxicity_heatmap.png", "Task 11: Mean toxicity scores by time of day.")
    render_plot_card("13. Lorenz Curve (Gini Inequality)", "lorenz_curve.png", "Task 13: Measures comment volume inequality across author base.")

with g_col2:
    render_plot_card("2. 2D UMAP Semantic Clusters Space", "umap_semantics.png", "Task 2: Manifold projection of comments colored by intent/topic.")
    render_plot_card("4. SHAP Feature Attribution Summary", "shap_summary.png", "Task 4: Tree SHAP feature importance predicting comment likes.")
    render_plot_card("6. Plutchik Emotion Radar Wheel", "plutchik_emotion_wheel.png", "Task 6: Polar radar chart of GoEmotions 8 primary emotional axes.")
    render_plot_card("8. Force-Directed Author Network", "author_network_force.png", "Task 8: Interaction graph colored by Louvain community detection.")
    render_plot_card("10. Sentiment Divergence (Root vs. Reply)", "sentiment_divergence.png", "Task 10: Delta in sentiment between original post and replies.")
    render_plot_card("12. Reply Depth Distribution", "reply_depth_distribution.png", "Task 12: Frequency decay of comment reply thread depth.")
    render_plot_card("14. STL Time-Series Decomposition", "stl_decomposition.png", "Task 14: STL decomposition of daily volume into Trend, Seasonality & Residuals.")