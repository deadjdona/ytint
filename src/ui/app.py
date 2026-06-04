import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from pathlib import Path

# ---------------------------------------------------------------------
# 1. UI Configuration & Aesthetic Settings
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ytint // Narrative Observatory",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
palette = config["ui"]["palette"]

# Inject custom "Bloomberg/Kibana" dark structural theme overrides via CSS
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {palette["background"]};
            color: #e2e8f0;
            font-family: 'Courier New', Courier, monospace;
        }}
        div[data-testid="stMetricValue"] {{
            color: {palette["accent"]};
            font-family: monospace;
            font-size: 1.8rem;
        }}
        .metric-card {{
            background-color: {palette["surface"]};
            border: 1px solid {palette["grid"]};
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
        }}
        .detail-panel {{
            background-color: {palette["surface"]};
            border-left: 3px solid {palette["accent"]};
            padding: 15px;
            margin-top: 15px;
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. Advanced Data Loading Layer
# ---------------------------------------------------------------------
@st.cache_data
def load_observatory_data():
    final_dir = Path(config["paths"]["final_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    
    eras = pd.read_parquet(final_dir / "compiled_eras.parquet")
    events = pd.read_parquet(final_dir / "compiled_events.parquet")
    comments = pd.read_parquet(interim_dir / "comments_clean.parquet")
    videos = pd.read_parquet(interim_dir / "videos_clean.parquet")
    
    # Advanced Metric Layers
    aging = pd.read_parquet(final_dir / "compiled_aging.parquet") if (final_dir / "compiled_aging.parquet").exists() else pd.DataFrame()
    controversy = pd.read_parquet(final_dir / "compiled_controversy.parquet") if (final_dir / "compiled_controversy.parquet").exists() else pd.DataFrame()
    
    eras['start_date'] = pd.to_datetime(eras['start_date'])
    eras['end_date'] = pd.to_datetime(eras['end_date'])
    if not events.empty:
        events['event_date'] = pd.to_datetime(events['event_date'])
    comments['published_at'] = pd.to_datetime(comments['published_at'])
    videos['published_at'] = pd.to_datetime(videos['published_at'])
    
    return eras, events, comments, videos, aging, controversy

try:
    df_eras, df_events, df_comments, df_videos, df_aging, df_controversy = load_observatory_data()
except Exception as e:
    st.error("❌ Failed to load final analytical layers. Please verify that pipeline stages s00 through s03 executed completely.")
    st.stop()

# ---------------------------------------------------------------------
# 3. Header & Master Temporal Spine (Top)
# ---------------------------------------------------------------------
st.title("📟 ytint // Narrative Observatory")
st.caption("SYSTEM STATE: ACTIVE // MODE: HISTORICAL NARRATIVE ARCHAEOLOGY")

min_date = df_comments['published_at'].min().date()
max_date = df_comments['published_at'].max().date()

selected_range = st.slider(
    "🔬 MASTER TEMPORAL FILTER",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

start_filter, end_filter = pd.to_datetime(selected_range[0]), pd.to_datetime(selected_range[1])
filtered_comments = df_comments[(df_comments['published_at'] >= start_filter) & (df_comments['published_at'] <= end_filter)]
filtered_events = df_events[(df_events['event_date'] >= start_filter) & (df_events['event_date'] <= end_filter)] if not df_events.empty else df_events

# ---------------------------------------------------------------------
# 4. Multi-Column Forensic Workspace (Left, Center, Right)
# ---------------------------------------------------------------------
col_left, col_center, col_right = st.columns([1, 2, 1.2])

# --- LEFT COLUMN: Era History & Cult Classics ---
with col_left:
    st.markdown("### 🗂️ HISTORICAL STRATA")
    
    # Tabs for clean information density separation
    tab_eras, tab_cult = st.tabs(["System Eras", "Memory Drift"])
    
    with tab_eras:
        for _, era in df_eras.iterrows():
            st.markdown(f"""
            <div class='metric-card'>
                <small>ERA {int(era['era_id'])} ({era['start_date'].strftime('%Y-%m')} to {era['end_date'].strftime('%Y-%m')})</small><br>
                <b>{era['dominant_theme']}</b><br>
                <small style='color:#ef4444'>🔴 Neg: {era['pct_negative']:.1%}</small> | 
                <small style='color:#22c55e'>🟢 Pos: {era['pct_positive']:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
            
    with tab_cult:
        if not df_aging.empty:
            st.caption("Videos that changed meaning over time")
            # Merge video titles to read names cleanly
            df_aging_named = df_aging.merge(df_videos[['video_id', 'title']], on='video_id')
            
            st.markdown("**✨ Top Cult Classics (Matured Well)**")
            for _, row in df_aging_named.sort_values(by='aging_score', ascending=False).head(2).iterrows():
                st.markdown(f"<div class='metric-card'><small>Score: +{row['aging_score']:.2f}</small><br><b>{row['title'][:45]}...</b></div>", unsafe_allow_html=True)
                
            st.markdown("**⚠️ Aged Poorly (Community Soured)**")
            for _, row in df_aging_named.sort_values(by='aging_score', ascending=True).head(2).iterrows():
                st.markdown(f"<div class='metric-card'><small style='color:#ef4444'>Score: {row['aging_score']:.2f}</small><br><b>{row['title'][:45]}...</b></div>", unsafe_allow_html=True)
        else:
            st.caption("Insufficient chronological depth to score memory drift matrices.")

# --- CENTER COLUMN: Visual Density Charts ---
with col_center:
    tab_trends, tab_friction = st.tabs(["Thematic Shifting", "Friction Index Grid"])
    
    with tab_trends:
        filtered_comments['week_bucket'] = filtered_comments['published_at'].dt.to_period('W').dt.to_timestamp()
        chart_data = filtered_comments[filtered_comments['topic_id'] != -1]
        
        if not chart_data.empty:
            topic_trends = chart_data.groupby(['week_bucket', 'topic_label']).size().reset_index(name='Volume')
            fig = px.area(
                topic_trends, x='week_bucket', y='Volume', color='topic_label',
                color_discrete_sequence=px.colors.sequential.Muted, template='plotly_dark'
            )
            fig.update_layout(
                paper_bgcolor=palette["background"], plot_bgcolor=palette["background"],
                xaxis=dict(gridcolor=palette["grid"], title="System Timeline"),
                yaxis=dict(gridcolor=palette["grid"], title="Comment Velocity"),
                legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No text records found in this slice window.")
            
    with tab_friction:
        if not df_controversy.empty:
            df_cont_named = df_controversy.merge(df_videos[['video_id', 'title', 'published_at']], on='video_id')
            
            # Scatterplot mapping maximum argument depth against raw emotional negativity ratio
            fig_scat = px.scatter(
                df_cont_named, 
                x='max_thread_depth', 
                y='negativity_ratio',
                size='controversy_score',
                hover_name='title',
                color='controversy_score',
                color_continuous_scale='OrRd',
                template='plotly_dark'
            )
            fig_scat.update_layout(
                paper_bgcolor=palette["background"], plot_bgcolor=palette["background"],
                xaxis=dict(gridcolor=palette["grid"], title="Maximum Argument Thread Depth"),
                yaxis=dict(gridcolor=palette["grid"], title="Emotional Negativity Ratio"),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("Friction mapping database tables empty.")

# --- RIGHT COLUMN: Anomaly Radar Feed ---
with col_right:
    st.markdown("### 🚨 ANOMALY RADAR FEED")
    if not filtered_events.empty:
        for _, ev in filtered_events.sort_values(by='event_date', ascending=False).iterrows():
            st.markdown(f"""
            <div class='metric-card' style='border-color: {palette["accent"]};'>
                <span style='color: {palette["accent"]}; font-weight:bold;'>⚠️ DISRUPTION SPECTRUM</span><br>
                <small>Date: {ev['event_date'].strftime('%Y-%m-%d')} // Deviation: {ev['z_score']:.2f}</small><br>
                <b>Context:</b> {ev['associated_video']}<br>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("Radar matrix silent. Zero velocity anomalies flagged.")

# ---------------------------------------------------------------------
# 6. Contextual Forensic Panel (Bottom)
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔬 FORENSIC ARTIFACT EXTRACTION FIELD")

if not filtered_events.empty:
    event_options = {f"{ev['event_date'].strftime('%Y-%m-%d')} - {ev['associated_video'][:50]}...": ev for _, ev in filtered_events.iterrows()}
    selected_event_key = st.selectbox("Select Target Event ID Coordinates", list(event_options.keys()))
    
    if selected_event_key:
        target_event = event_options[selected_event_key]
        st.markdown(f"""
        <div class='detail-panel'>
            <h4>Representative Inscription (Highest Liked Text Artifact):</h4>
            <p style='font-style: italic; font-size: 1.1rem; color: #cbd5e1;'>
                "{target_event['representative_artifact']}"
            </p>
            <small style='color: {palette["accent"]}'>
                ⚙️ SYSTEM LOG: Day Count Velocity {target_event['comment_volume']} units // Deviance Factor: {target_event['z_score']:.2f}
            </small>
        </div>
        """, unsafe_allow_html=True)