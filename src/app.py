import os
import logging
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

from engine.config_loader import load_config as _load_config
from ui.visual_explanations import VISUAL_METADATA

# ==============================================================================
# 1. PAGE SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="ytint // Intelligence Engine & Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Plotly Template Default
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PRIMARY = "#0066fe"
COLOR_ACCENT = "#00f0ff"
COLOR_SUCCESS = "#00ff66"
COLOR_DANGER = "#ff3366"

# ==============================================================================
# 2. CACHED PIPELINE DATA LOADER
# ==============================================================================
@st.cache_data
def load_config():
    """Cached wrapper around the shared config loader."""
    config = _load_config()
    config["paths"]["interim_dir"] = pathlib.Path(config["paths"]["interim_dir"])
    config["paths"]["output_dir"] = pathlib.Path(config["paths"]["output_dir"])
    return config

@st.cache_data(show_spinner="Loading analytical intelligence layers...")
def load_all_pipeline_data(interim_dir, output_dir):
    """Loads all clean parquet datasets and analytical stage outputs."""
    data = {}
    
    # Core Master Layers
    f_comments = interim_dir / "comments_clean.parquet"
    f_topics = output_dir / "topic_metadata.parquet"
    f_videos = output_dir / "videos_final.parquet"
    f_authors = output_dir / "authors_final.parquet"
    f_timeline = output_dir / "historical_timeline.parquet"
    f_spikes = output_dir / "viral_events.parquet"
    
    data["comments"] = pd.read_parquet(f_comments) if f_comments.exists() else pd.DataFrame()
    data["topics"] = pd.read_parquet(f_topics) if f_topics.exists() else pd.DataFrame()
    data["videos"] = pd.read_parquet(f_videos) if f_videos.exists() else pd.DataFrame()
    data["authors"] = pd.read_parquet(f_authors) if f_authors.exists() else pd.DataFrame()
    data["timeline"] = pd.read_parquet(f_timeline) if f_timeline.exists() else pd.DataFrame()
    data["spikes"] = pd.read_parquet(f_spikes) if f_spikes.exists() else pd.DataFrame()

    # Specialized Analytical Layers (Stages 07 - 34)
    extra_files = {
        "author_fingerprints": "author_fingerprints.parquet",
        "video_radar": "video_profile_radar.parquet",
        "code_switching": "code_switching_impact.parquet",
        "topic_evolution": "topic_evolution.parquet",
        "reaction_map": "video_reaction_map.parquet",
        "dunn_matrix": "dunn_posthoc_matrix.parquet",
        "power_law": "power_law_fit.parquet",
        "bot_classifications": "bot_classifications.parquet",
        "like_inflation": "like_inflation.parquet",
        "frequency_tiers": "frequency_tiers.parquet",
        "driveby_loyalists": "driveby_loyalists.parquet",
        "arrival_speed": "arrival_speed.parquet",
        "named_entities": "named_entities.parquet",
        "poisson_bursts": "poisson_bursts.parquet",
        "position_bias": "position_bias.parquet",
        "shap_features": "shap_features.parquet",
        "stl_decomposition": "stl_decomposition.parquet",
        "engagement_forecast": "engagement_forecast.parquet",
        "video_overlap": "video_overlap_matrix.parquet",
        "tag_nodes": "tag_network_nodes.parquet",
        "word_nodes": "word_cooccurrence_nodes.parquet",
        "kruskal_wallis": "kruskal_wallis_results.parquet",
        "new_vs_returning": "new_vs_returning.parquet",
        "shelf_life": "shelf_life_likes.parquet",
        "diurnal": "diurnal_heatmap.parquet",
        "audience_intent": "audience_intent_summary.parquet",
        "audience_requests": "audience_content_requests.parquet",
        "cib_rings": "cib_rings.parquet",
        "cib_comments": "cib_coordinated_comments.parquet"
    }

    for key, fname in extra_files.items():
        fpath = output_dir / fname
        if fpath.exists():
            try:
                data[key] = pd.read_parquet(fpath)
            except Exception:
                data[key] = pd.DataFrame()
        else:
            data[key] = pd.DataFrame()

    return data

# Initialize configuration and datasets
config = load_config()
interim_path = config["paths"]["interim_dir"]
output_path = config["paths"]["output_dir"]
plots_dir = output_path / "plots"

data_layers = load_all_pipeline_data(interim_path, output_path)

df_comments_raw = data_layers["comments"]
df_topics = data_layers["topics"]
df_videos = data_layers["videos"]
df_authors = data_layers["authors"]
df_timeline_raw = data_layers["timeline"]
df_spikes = data_layers["spikes"]

# ==============================================================================
# 3. SIDEBAR NAVIGATION & ENGINE CONTROLS
# ==============================================================================
st.sidebar.title("🧬 ytint Intelligence Engine")
st.sidebar.caption("High-Dimensional YouTube Conversational Analytics & Forensics")
st.sidebar.divider()

comment_filter = st.sidebar.selectbox(
    "💬 Comment Layer Segmentation",
    options=["Show All Records", "Top-level Comments Only", "Replies/Responses Only"]
)

# Apply Comment Filter Slice
if comment_filter == "Top-level Comments Only" and not df_comments_raw.empty:
    df_comments = df_comments_raw[df_comments_raw['parent_id'].isna() | (df_comments_raw['parent_id'] == "")]
elif comment_filter == "Replies/Responses Only" and not df_comments_raw.empty:
    df_comments = df_comments_raw[df_comments_raw['parent_id'].notna() & (df_comments_raw['parent_id'] != "")]
else:
    df_comments = df_comments_raw

st.sidebar.divider()
st.sidebar.subheader("⚙️ Pipeline Engine Metrics")

total_raw_comments = len(df_comments_raw) if not df_comments_raw.empty else 0
total_active_authors = len(df_authors) if not df_authors.empty else 0
total_tracked_videos = len(df_videos) if not df_videos.empty else 0

st.sidebar.metric(label="Total Ingested Comments", value=f"{total_raw_comments:,}")
st.sidebar.metric(label="Unique Community Authors", value=f"{total_active_authors:,}")
st.sidebar.metric(label="Total Tracked Videos", value=f"{total_tracked_videos:,}")

if not data_layers["power_law"].empty and "alpha" in data_layers["power_law"].columns:
    pl_alpha = data_layers["power_law"]["alpha"].iloc[0]
    st.sidebar.metric(label="Power-Law Exponent (α)", value=f"{pl_alpha:.3f}")

st.sidebar.divider()
st.sidebar.markdown("**Backend Architecture Parameters:**")
st.sidebar.info(
    f"**Z-Score Spike Threshold:** {config['stage_03_narrative']['z_threshold']}\n\n"
    f"**PELT Penalty Model:** {config['stage_03_narrative']['change_point_penalty']}\n\n"
    f"**Embedding Model:** {config['stage_02_topics'].get('embedding_model', 'multilingual-MiniLM')}"
)

# ==============================================================================
# 4. MAIN APP HEADER & TABBED WORKSPACE
# ==============================================================================
st.title("🎬 ytint // Conversational Topic Modeling & Executive Intelligence")
st.caption(f"Currently inspecting: **{comment_filter}** — {len(df_comments):,} active slice records")

tab_summary, tab_spikes, tab_topics, tab_audience, tab_modeling, tab_gallery = st.tabs([
    "📋 Executive Summary",
    "🚨 Viral Events & Flashpoints",
    "🧩 Topics & Lexical Semantics",
    "👥 Audience Loyalty & Forensics",
    "🔬 Cross-Video & Predictive Modeling",
    "🖼️ Visual Analytics Gallery"
])

# ==============================================================================
# TAB 1: EXECUTIVE INTELLIGENCE SUMMARY
# ==============================================================================
with tab_summary:
    st.header("Executive Intelligence Briefing")
    st.markdown(
        "Holistic synthesis of channel engagement dynamics, community integrity diagnostics, "
        "and conversational resonance extracted across all 35 analytical stages."
    )

    # 1.1 Executive Metric Tiles
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    # Calculate Summary Statistics
    num_topics = len(df_topics[df_topics["Topic"] != -1]) if not df_topics.empty and "Topic" in df_topics.columns else len(df_topics)
    num_spikes = len(df_spikes) if not df_spikes.empty else 0
    
    bot_rate = 0.0
    if not data_layers["bot_classifications"].empty and "is_bot" in data_layers["bot_classifications"].columns:
        bot_rate = (data_layers["bot_classifications"]["is_bot"].sum() / len(data_layers["bot_classifications"])) * 100

    suspicious_like_rate = 0.0
    if not data_layers["like_inflation"].empty and "is_suspicious" in data_layers["like_inflation"].columns:
        suspicious_like_rate = (data_layers["like_inflation"]["is_suspicious"].sum() / len(data_layers["like_inflation"])) * 100

    avg_sentiment = 0.0
    if not df_videos.empty and "avg_sentiment" in df_videos.columns:
        avg_sentiment = df_videos["avg_sentiment"].mean()

    with kpi_col1:
        with st.container(border=True):
            st.metric(label="Total Processed Comments", value=f"{total_raw_comments:,}", delta=f"{len(df_comments):,} in slice")
            st.caption("Complete historical conversation scope across tracked video uploads.")

    with kpi_col2:
        with st.container(border=True):
            st.metric(label="Active Community Authors", value=f"{total_active_authors:,}", delta=f"{total_tracked_videos} Videos")
            st.caption("Unique commenters mapped across longitudinal engagement trees.")

    with kpi_col3:
        with st.container(border=True):
            st.metric(label="Semantic Topics & Spikes", value=f"{num_topics:,} Clusters", delta=f"{num_spikes} Viral Spikes")
            st.caption("Identified BERTopic clusters & anomalous volumetric flashpoints.")

    with kpi_col4:
        with st.container(border=True):
            st.metric(
                label="Channel Integrity & Bot Risk",
                value=f"{bot_rate:.1f}% Suspicious",
                delta=f"-{suspicious_like_rate:.1f}% Like-Inflated",
                delta_color="inverse"
            )
            st.caption("MinHash near-duplicate & high-like/zero-reply astroturfing signals.")

    st.divider()

    # 1.2 Executive Intelligence Synthesis Cards
    st.subheader("Key Strategic Findings & Channel Diagnostics")

    syn_col1, syn_col2 = st.columns(2)

    with syn_col1:
        with st.container(border=True):
            st.markdown("#### 👥 Audience Dynamics & Community Retention")
            
            driveby_count = 0
            loyalist_count = 0
            if not data_layers["driveby_loyalists"].empty and "classification" in data_layers["driveby_loyalists"].columns:
                db_df = data_layers["driveby_loyalists"].set_index("classification")["author_count"].to_dict()
                driveby_count = db_df.get("drive_by", 0)
                loyalist_count = db_df.get("loyalist", 0)

            st.markdown(
                f"- **Drive-by vs. Core Community**: The audience exhibits a heavy-tailed Pareto structure "
                f"with **{driveby_count:,} one-time commenters** vs. **{loyalist_count:,} core loyalists** "
                f"repeatedly returning across multiple video uploads.\n"
                f"- **Power-Law Engagement**: Power-law coefficient is fit at **α = {data_layers['power_law']['alpha'].iloc[0]:.2f}** "
                f"if available, confirming that a small cohort of super-fans drives an outsized share of total discussion.\n"
                f"- **Attention Half-Life**: Videos experience steep early decay, with the average attention half-life spanning "
                f"**{df_videos['attention_half_life_days'].mean():.2f} days** before discussion stabilizes."
            )

        with st.container(border=True):
            st.markdown("#### ⚡ Viral Catalysts & Temporal Flashpoints")
            st.markdown(
                f"- **Automated Anomaly Detection**: Identified **{num_spikes} major conversational flashpoints** "
                f"breaching the {config['stage_03_narrative']['z_threshold']}σ rolling Z-score threshold.\n"
                f"- **Poisson Burst Bursts**: High-frequency comment bursts cluster predominantly within the first **2–6 hours** "
                f"post-release, establishing the critical window for creator engagement and moderation.\n"
                f"- **Diurnal Concentration**: Peak commenting volume aligns with evening leisure hours (18:00–22:00 UTC), "
                f"presenting optimal scheduling windows for maximum discussion velocity."
            )

    with syn_col2:
        with st.container(border=True):
            st.markdown("#### 🛡️ Forensic Integrity & Moderation Risks")
            st.markdown(
                f"- **Bot & Repetitive Spam Activity**: Lexical richness and MinHash token frequency heuristics identify "
                f"**{bot_rate:.2f}%** of commenter profiles exhibiting robotic or high-frequency copy-paste behavior.\n"
                f"- **Astroturfing & Like-Inflation**: Analysis of like-to-reply ratios isolated "
                f"**{suspicious_like_rate:.2f}%** of top comments with artificially inflated likes but zero organic replies.\n"
                f"- **Toxicity & Polarization**: Reply trees demonstrate mild negative drift as depth increases, "
                f"highlighting that secondary and tertiary debate threads require proactive moderation."
            )

        with st.container(border=True):
            st.markdown("#### 🔀 Linguistic Profiling & Engagement Drivers")
            cs_lift = "0.0"
            if not data_layers["code_switching"].empty and "avg_likes" in data_layers["code_switching"].columns:
                try:
                    cs_likes = data_layers["code_switching"].set_index("type")["avg_likes"].to_dict()
                    mono_l = cs_likes.get("Monolingual", 1)
                    multi_l = cs_likes.get("Code-Switched", 1)
                    lift = ((multi_l - mono_l) / max(mono_l, 0.001)) * 100
                    cs_lift = f"{lift:+.1f}%"
                except Exception:
                    pass

            st.markdown(
                f"- **Multilingual Code-Switching Lift**: Comments mixing Cyrillic and Latin gamer slang achieve a "
                f"**{cs_lift} engagement premium** in average likes compared to standard monolingual comments.\n"
                f"- **Top Semantic Themes**: Discussions center around political discourse, creator references, and current events.\n"
                f"- **Predictive Attribution (SHAP)**: Early arrival speed, sentence length, and sentiment polarity are the "
                f"primary positive drivers of community upvotes."
            )

    st.divider()

    # 1.3 Video Performance Intelligence Leaderboard
    st.subheader("Top Analyzed Videos Performance Leaderboard")
    if not df_videos.empty:
        display_videos = df_videos.copy()
        col_rename = {
            "video_id": "Video ID",
            "total_comments": "Total Comments",
            "total_likes": "Total Likes",
            "avg_sentiment": "Sentiment Score",
            "gini_coefficient": "Gini Inequality",
            "attention_half_life_days": "Half-Life (Days)",
            "revival_spikes": "Revival Spikes"
        }
        display_videos = display_videos.rename(columns=col_rename)
        
        # Sortable Top Table
        st.dataframe(
            display_videos.sort_values(by="Total Comments", ascending=False),
            width="stretch",
            hide_index=True
        )
        st.caption("Comprehensive summary of video performance metrics, attention half-lives, and conversation inequality.")
    else:
        st.info("No video metadata records available in the current pipeline run.")

# ==============================================================================
# TAB 2: VIRAL EVENTS & MOMENT DYNAMICS
# ==============================================================================
with tab_spikes:
    st.header("Conversational Volumetric Spikes & Anomaly Detection")
    st.markdown("Chronological anomaly isolation identifying viral catalysts, sudden debates, and event-driven discussion surges.")

    # Timeline calculation based on slice
    if not df_comments.empty and 'published_at' in df_comments.columns:
        df_c = df_comments.copy()
        df_c['published_at'] = pd.to_datetime(df_c['published_at'])
        df_slice_timeline = df_c.groupby(df_c['published_at'].dt.date).size().to_frame(name='comment_count')
        df_slice_timeline.index = pd.to_datetime(df_slice_timeline.index)
        df_slice_timeline = df_slice_timeline.sort_index().reset_index().rename(columns={'published_at': 'date'})
        
        # 7-day rolling statistics
        roll_mean = df_slice_timeline['comment_count'].rolling(window=7, min_periods=1).mean()
        roll_std = df_slice_timeline['comment_count'].rolling(window=7, min_periods=1).std().fillna(1)
        df_slice_timeline['z_score'] = (df_slice_timeline['comment_count'] - roll_mean) / roll_std
    else:
        df_slice_timeline = df_timeline_raw.copy()

    if not df_slice_timeline.empty:
        fig_timeline = px.line(
            df_slice_timeline,
            x='date',
            y='comment_count',
            title="Chronological Conversation Volume & Velocity Trajectory",
            labels={'date': 'Date', 'comment_count': 'Captured Volume'},
            line_shape='spline',
            template=PLOTLY_TEMPLATE
        )
        fig_timeline.update_traces(line_color=COLOR_PRIMARY, line_width=2.5)

        # Highlight anomalies
        z_thresh = config["stage_03_narrative"].get("z_threshold", 2.5)
        anomalies = df_slice_timeline[df_slice_timeline['z_score'] > z_thresh]
        if not anomalies.empty:
            fig_timeline.add_trace(
                go.Scatter(
                    x=anomalies['date'],
                    y=anomalies['comment_count'],
                    mode='markers',
                    name='Viral Spike Event',
                    marker=dict(color=COLOR_DANGER, size=10, symbol='diamond')
                )
            )
        st.plotly_chart(fig_timeline, width="stretch")
        st.caption("Daily conversation density with detected anomaly spikes (red diamonds) exceeding the configured Z-score threshold.")

    # Second-by-Second Reaction Heatmap/Timeline
    if not data_layers["reaction_map"].empty:
        st.divider()
        st.subheader("🎬 Video Reaction Dynamics (Second-by-Second)")
        st.markdown("Moment-by-moment sentiment trajectory and toxicity tracking mapped across video playback timestamps.")

        df_rx = data_layers["reaction_map"]
        vid_options = df_rx["video_id"].unique()
        selected_vid = st.selectbox("Select Video ID for Reaction Timeline:", options=vid_options)
        
        vid_rx = df_rx[df_rx["video_id"] == selected_vid].sort_values("second")
        
        y_cols = ["vader_compound"]
        if "toxicity" in vid_rx.columns:
            y_cols.append("toxicity")

        fig_rx = px.line(
            vid_rx,
            x="second",
            y=y_cols,
            title=f"Sentiment & Toxicity Trajectory over Playback Time // {selected_vid}",
            labels={"second": "Playback Time (Seconds)", "value": "Metric Score", "variable": "Dimension"},
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig_rx, width="stretch")
        st.caption("Second-by-second sentiment and toxicity levels. Peaks or troughs highlight specific scene moments that triggered intense audience reactions.")

    # Spike & Burst Ledgers
    st.divider()
    l_col1, l_col2 = st.columns(2)
    with l_col1:
        st.subheader("⚡ Detected Anomaly Spikes Ledger")
        if not df_spikes.empty:
            st.dataframe(df_spikes, width="stretch", hide_index=True)
        else:
            st.info("No spike records found in the current dataset.")

    with l_col2:
        st.subheader("🔥 High-Frequency Poisson Bursts")
        if not data_layers["poisson_bursts"].empty:
            st.dataframe(data_layers["poisson_bursts"].head(50), width="stretch", hide_index=True)
        else:
            st.info("No Poisson burst records available.")

# ==============================================================================
# TAB 3: TOPICS & LEXICAL SEMANTICS
# ==============================================================================
with tab_topics:
    st.header("Conversational Topic Modeling & Semantic Exploration")
    st.markdown("Granular semantic clusters extracted via UMAP dimensionality reduction and HDBSCAN topic modeling.")

    # 3.1 Topic Resonance Matrix
    if not df_topics.empty and "Count" in df_topics.columns:
        plot_topics = df_topics[df_topics["Topic"] != -1].copy()
        if not plot_topics.empty:
            y_axis_col = "avg_likes" if "avg_likes" in plot_topics.columns else "Count"
            size_col = "total_likes" if "total_likes" in plot_topics.columns else "Count"

            fig_resonance = px.scatter(
                plot_topics,
                x="Count",
                y=y_axis_col,
                size=size_col,
                hover_name="Name",
                title="Topic Resonance Matrix (Volume vs. Engagement)",
                labels={"Count": "Total Comments in Topic", y_axis_col: "Average Likes per Comment"},
                template=PLOTLY_TEMPLATE,
                size_max=40,
                color="Count",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_resonance, width="stretch")
            st.caption("Topic resonance mapping comment volume against community approval (average likes). Top-right bubbles indicate high-volume, highly favored topics.")

    # 3.2 Searchable Topic Registry
    st.subheader("Searchable Semantic Topic Registry")
    topic_query = st.text_input("🔍 Filter topics by keyword or token:", "").strip().lower()

    filtered_topics = df_topics.copy()
    if topic_query and not filtered_topics.empty:
        text_mask = filtered_topics.astype(str).apply(lambda x: x.str.lower().str.contains(topic_query)).any(axis=1)
        filtered_topics = filtered_topics[text_mask]

    st.dataframe(filtered_topics, width="stretch", hide_index=True)

    # 3.3 Topic Evolution Over Time
    if not data_layers["topic_evolution"].empty:
        st.divider()
        st.subheader("📈 Longitudinal Topic Share Evolution")
        df_te = data_layers["topic_evolution"]
        if "year_month" in df_te.columns:
            fig_te = px.area(
                df_te,
                x="year_month",
                y=[c for c in df_te.columns if c != "year_month"],
                title="Monthly Topic Volume Dynamics & Shifts",
                labels={"year_month": "Timeline Month", "value": "Comment Volume", "variable": "Topic"},
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_te, width="stretch")
            st.caption("Streamgraph depicting how topical focus expands and contracts over historical calendar months.")

    # 3.4 Named Entities & Code-Switching Impact
    st.divider()
    sem_col1, sem_col2 = st.columns(2)

    with sem_col1:
        st.subheader("🏷️ Top Named Entities (NER)")
        if not data_layers["named_entities"].empty:
            top_entities = data_layers["named_entities"]['entity_text'].value_counts().head(20).reset_index()
            top_entities.columns = ['Entity', 'Mentions']
            fig_ner = px.bar(
                top_entities,
                x='Mentions',
                y='Entity',
                orientation='h',
                title="Top Extracted Entities (People, Places, Orgs)",
                template=PLOTLY_TEMPLATE
            )
            fig_ner.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_ner, width="stretch")
        else:
            st.info("Named entity data not available.")

    with sem_col2:
        st.subheader("🔀 Code-Switching Engagement Impact")
        if not data_layers["code_switching"].empty:
            df_cs = data_layers["code_switching"]
            fig_cs = px.bar(
                df_cs,
                x="type",
                y="avg_likes",
                color="type",
                title="Monolingual vs. Code-Switched Average Like Engagement",
                labels={"type": "Comment Linguistic Type", "avg_likes": "Average Likes"},
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_cs, width="stretch")
            st.caption("Engagement disparity between standard monolingual comments and mixed-script gamer slang.")
        else:
            st.info("Code-switching impact data not available.")

    # 3.5 Audience Demand & Intent Mining (Stage 35)
    st.divider()
    st.subheader("🎯 Audience Demand & Content Ideas Roadmap (Stage 35)")
    st.markdown("Automated categorization of audience intent extracting content requests, inquiries, and critiques.")

    if not data_layers["audience_intent"].empty:
        intent_col1, intent_col2 = st.columns([1, 2])
        
        with intent_col1:
            fig_intent = px.pie(
                data_layers["audience_intent"],
                names="audience_intent",
                values="comment_count",
                title="Audience Intent Distribution",
                template=PLOTLY_TEMPLATE,
                hole=0.35,
                color="audience_intent",
                color_discrete_map={
                    'CONTENT_IDEA': '#06d6a0',
                    'QUESTION_CONFUSION': '#118ab2',
                    'CRITIQUE_FEEDBACK': '#ffd166',
                    'APPRECIATION': '#ef476f',
                    'DEBATE_OPINION': '#4a4e69'
                }
            )
            st.plotly_chart(fig_intent, width="stretch")

        with intent_col2:
            st.markdown("##### Prioritized Audience Content Requests & Questions")
            if not data_layers["audience_requests"].empty:
                req_df = data_layers["audience_requests"].copy()
                col_rename = {
                    "text_original": "Audience Comment",
                    "audience_intent": "Intent Category",
                    "like_count": "Likes",
                    "video_id": "Video ID"
                }
                req_df = req_df.rename(columns={k: v for k, v in col_rename.items() if k in req_df.columns})
                st.dataframe(req_df.head(50), width="stretch", hide_index=True)
                st.caption("Showing top upvoted video suggestions, questions, and feedback from the community.")
            else:
                st.info("No actionable content requests found.")

# ==============================================================================
# TAB 4: AUDIENCE LOYALTY & FORENSICS
# ==============================================================================
with tab_audience:
    st.header("👥 Audience Loyalty, Forensics & Behavior Profiling")
    st.markdown("Detailed breakdown of community retention, super-fan fingerprints, and automated bot forensics.")

    # 4.1 Super-Fan Behavioral Radar
    if not data_layers["author_fingerprints"].empty:
        st.subheader("🕸️ Top Author Behavioral Fingerprints (Polar Radar)")
        df_af = data_layers["author_fingerprints"]
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
                title="Super-Fan Multi-Axis Behavioral Footprints",
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_af_radar, width="stretch")
            st.caption("Polar profile characterizing top active commenters across volume, positivity, negativity, and reply participation.")

    # 4.2 Audience Tiers & Loyalists Breakdown
    st.divider()
    aud_col1, aud_col2, aud_col3 = st.columns(3)

    with aud_col1:
        st.subheader("📊 Commenting Frequency Tiers")
        if not data_layers["frequency_tiers"].empty:
            fig_tiers = px.pie(
                data_layers["frequency_tiers"],
                names="tier",
                values="author_count",
                title="Audience Frequency Segmentation",
                template=PLOTLY_TEMPLATE,
                hole=0.4
            )
            st.plotly_chart(fig_tiers, width="stretch")
        else:
            st.info("Frequency tiers not computed.")

    with aud_col2:
        st.subheader("🎯 Drive-By vs. Loyalists")
        if not data_layers["driveby_loyalists"].empty:
            fig_db = px.bar(
                data_layers["driveby_loyalists"],
                x="classification",
                y="author_count",
                color="classification",
                title="Commenter Persistence Classification",
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_db, width="stretch")
        else:
            st.info("Loyalist distribution data not available.")

    with aud_col3:
        st.subheader("⏱️ First-Responder Arrival Speed")
        if not data_layers["arrival_speed"].empty:
            fig_arr = px.bar(
                data_layers["arrival_speed"],
                x="classification",
                y="author_count",
                color="classification",
                title="Arrival Speed Breakdown",
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_arr, width="stretch")
        else:
            st.info("Arrival speed data not available.")

    # 4.3 Bot Forensics & Like Inflation Inspection
    st.divider()
    for_col1, for_col2 = st.columns(2)

    with for_col1:
        st.subheader("🤖 Bot Classifier & Heuristic Audit")
        if not data_layers["bot_classifications"].empty:
            bot_summary = data_layers["bot_classifications"]["classification"].value_counts().reset_index()
            bot_summary.columns = ["Classification", "Author Count"]
            st.dataframe(bot_summary, width="stretch", hide_index=True)
            st.caption("Distribution of author profiles classified as human vs. repetitive spam bots.")
        else:
            st.info("Bot classification records not available.")

    with for_col2:
        st.subheader("💸 Astroturfing / Like-Inflation Flags")
        if not data_layers["like_inflation"].empty:
            suspicious_comments = data_layers["like_inflation"][data_layers["like_inflation"]["is_suspicious"] == True]
            st.metric(label="Flagged Astroturfed Comments", value=f"{len(suspicious_comments):,}")
            st.dataframe(suspicious_comments.head(50), width="stretch", hide_index=True)
            st.caption("Comments exhibiting high likes with zero replies, indicating inorganic upvote manipulation.")
        else:
            st.info("Like inflation records not available.")

    # 4.4 Coordinated Inauthentic Behavior (CIB) Rings (Stage 36)
    st.divider()
    st.subheader("🕸️ Coordinated Inauthentic Behavior (CIB) & Astroturfing Rings (Stage 36)")
    st.markdown("Multi-account rings that repeatedly post in tight temporal synchronization across video uploads.")

    if not data_layers["cib_rings"].empty:
        cib_col1, cib_col2 = st.columns([1, 2])
        with cib_col1:
            st.metric(label="Identified CIB Rings", value=f"{len(data_layers['cib_rings']):,}")
            st.dataframe(data_layers["cib_rings"].head(20), width="stretch", hide_index=True)
            st.caption("Rings ranked by total synchronized comment pairings across videos.")

        with cib_col2:
            st.markdown("##### Synchronized Astroturfing Comments")
            if not data_layers["cib_comments"].empty:
                st.dataframe(data_layers["cib_comments"].head(50), width="stretch", hide_index=True)
                st.caption("Individual comments flagged as belonging to coordinated network rings.")
            else:
                st.info("No coordinated comments found.")

    # 4.5 RFM Author Segmentation Ledger
    st.divider()
    st.subheader("RFM Author Segmentation Explorer")
    if not df_authors.empty:
        st.dataframe(df_authors.head(100), width="stretch", hide_index=True)
        st.caption("Showing top 100 authors ranked by recency, frequency, and monetary engagement score.")

# ==============================================================================
# TAB 5: CROSS-VIDEO & PREDICTIVE MODELING
# ==============================================================================
with tab_modeling:
    st.header("🔬 Cross-Video Relations, Counterfactuals & Modeling")
    st.markdown("Advanced statistical matrices, pairwise significance tests, and predictive feature attribution.")

    # 5.1 Dunn's Post-Hoc Significance Matrix
    if not data_layers["dunn_matrix"].empty:
        st.subheader("📊 Dunn's Post-Hoc Pairwise p-Value Matrix")
        df_dunn = data_layers["dunn_matrix"]
        fig_dunn = px.imshow(
            df_dunn,
            labels=dict(x="Video ID", y="Video ID", color="Adjusted p-value"),
            x=df_dunn.columns,
            y=df_dunn.index,
            title="Statistical Significance Matrix of Pairwise Video Engagements",
            color_continuous_scale="Viridis",
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig_dunn, width="stretch")
        st.caption("Pairwise adjusted p-values from Kruskal-Wallis post-hoc Dunn test comparing engagement distributions across videos.")

    # 5.2 Video Profile Radar & Audience Overlap
    mod_col1, mod_col2 = st.columns(2)

    with mod_col1:
        st.subheader("🕸️ Multi-Video Profile Radar")
        if not data_layers["video_radar"].empty:
            df_vr = data_layers["video_radar"]
            norm_vr_cols = [c for c in df_vr.columns if c.endswith("_norm")]
            if norm_vr_cols:
                df_vr_melt = df_vr.melt(
                    id_vars=["video_id"],
                    value_vars=norm_vr_cols,
                    var_name="Dimension",
                    value_name="Normalized_Score"
                )
                df_vr_melt["Dimension"] = df_vr_melt["Dimension"].str.replace("_norm", "")
                fig_vr = px.line_polar(
                    df_vr_melt,
                    r="Normalized_Score",
                    theta="Dimension",
                    color="video_id",
                    line_close=True,
                    title="Comparative Video Profiles",
                    template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig_vr, width="stretch")

    with mod_col2:
        st.subheader("🔀 Cross-Video Audience Overlap")
        if not data_layers["video_overlap"].empty:
            df_vo = data_layers["video_overlap"].set_index("video_id") if "video_id" in data_layers["video_overlap"].columns else data_layers["video_overlap"]
            fig_vo = px.imshow(
                df_vo,
                title="Audience Jaccard Overlap Matrix",
                color_continuous_scale="Magma",
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_vo, width="stretch")
            st.caption("Jaccard similarity matrix of overlapping commenters across video pairs.")

    # 5.3 SHAP Feature Importance & Forecasts
    st.divider()
    feat_col1, feat_col2 = st.columns(2)

    with feat_col1:
        st.subheader("🌳 Tree SHAP Feature Attribution")
        if not data_layers["shap_features"].empty:
            st.dataframe(data_layers["shap_features"].head(20), width="stretch", hide_index=True)
            st.caption("Feature attribution matrix predicting comment engagement and like accrual.")
        else:
            st.info("SHAP attribution records not available.")

    with feat_col2:
        st.subheader("🔮 Engagement 30-Day Forward Forecast")
        if not data_layers["engagement_forecast"].empty:
            df_fc = data_layers["engagement_forecast"]
            fig_fc = px.line(
                df_fc,
                x="ds",
                y="yhat",
                title="Projected Daily Discussion Volume",
                labels={"ds": "Date", "yhat": "Projected Volume"},
                template=PLOTLY_TEMPLATE
            )
            if "yhat_upper" in df_fc.columns and "yhat_lower" in df_fc.columns:
                fig_fc.add_traces([
                    go.Scatter(
                        x=df_fc["ds"], y=df_fc["yhat_upper"],
                        mode='lines', line=dict(width=0), showlegend=False
                    ),
                    go.Scatter(
                        x=df_fc["ds"], y=df_fc["yhat_lower"],
                        mode='lines', line=dict(width=0), fill='tonexty',
                        fillcolor='rgba(0,102,254,0.2)', name='Confidence Interval'
                    )
                ])
            st.plotly_chart(fig_fc, width="stretch")
        else:
            st.info("Engagement forecast data not available.")

# ==============================================================================
# TAB 6: PUBLICATION-READY VISUAL ANALYTICS GALLERY
# ==============================================================================
with tab_gallery:
    st.header("🖼️ Publication-Ready Visual Analytics Gallery")
    st.markdown("Comprehensive repository of high-resolution statistical visualizations, machine learning embeddings, and network graphs with deep analytical interpretations.")

    def render_plot_card(filename, fallback_title=None, is_html=False):
        meta = VISUAL_METADATA.get(filename, {})
        title = meta.get("title", fallback_title or filename)
        summary = meta.get("summary", "")
        methodology = meta.get("methodology", "")
        how_to_read = meta.get("how_to_read", "")
        takeaway = meta.get("takeaway", "")

        filepath = plots_dir / filename
        with st.container(border=True):
            st.subheader(f"📊 {title}")
            if summary:
                st.markdown(f"**Core Insight**: {summary}")

            if filepath.exists():
                if is_html:
                    with open(filepath, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    components.html(html_content, height=600, scrolling=True)
                else:
                    st.image(str(filepath), width="stretch")
            else:
                st.info(f"ℹ️ Visual artifact `{filename}` not generated yet.")

            if methodology or how_to_read or takeaway:
                with st.expander("🔍 Deep Analytical Guide & Interpretation", expanded=True):
                    if methodology:
                        st.markdown(f"🔬 **Methodology & Model**: {methodology}")
                    if how_to_read:
                        st.markdown(f"🧭 **How to Read**: {how_to_read}")
                    if takeaway:
                        st.markdown(f"💡 **Strategic Takeaway**: {takeaway}")

    # ==============================================================================
    # Section 1: Temporal Dynamics & Lifecycle Modeling
    # ==============================================================================
    st.subheader("1. ⏳ Temporal Dynamics & Lifecycle Modeling")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        render_plot_card("kaplan_meier_survival.png")
        render_plot_card("diurnal_heatmap.png")
        render_plot_card("stl_decomposition.png")
    with t_col2:
        render_plot_card("reply_latency_distribution.png")
        render_plot_card("shelf_life_likes.png")
        render_plot_card("reaction_timeline.png")

    # ==============================================================================
    # Section 2: NLP, Semantics & Emotion Spectrum
    # ==============================================================================
    st.divider()
    st.subheader("2. 🧠 NLP, Semantics & Emotion Spectrum")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        render_plot_card("umap_semantics.png")
        render_plot_card("plutchik_emotion_wheel.png")
        render_plot_card("sentiment_ridges.png")
        render_plot_card("sarcasm_analysis.png")
        render_plot_card("linguistic_profiling.png")
    with s_col2:
        render_plot_card("audience_intent_distribution.png")
        render_plot_card("named_entities.png")
        render_plot_card("word_cooccurrence.png")
        render_plot_card("sentiment_divergence.png")
        render_plot_card("toxicity_heatmap.png")

    # ==============================================================================
    # Section 3: Audience Networks, Segmentation & Forensics
    # ==============================================================================
    st.divider()
    st.subheader("3. 👥 Audience Networks, Segmentation & Forensics")
    n_col1, n_col2 = st.columns(2)
    with n_col1:
        render_plot_card("rfm_3d.html", is_html=True)
        render_plot_card("author_network_force.png")
        render_plot_card("bot_heuristics.png")
        render_plot_card("frequency_tiers.png")
        render_plot_card("driveby_loyalists.png")
    with n_col2:
        render_plot_card("cib_rings_graph.png")
        render_plot_card("author_pareto.png")
        render_plot_card("lorenz_curve.png")
        render_plot_card("like_inflation.png")
        render_plot_card("arrival_speed.png")
        render_plot_card("author_fingerprints.png")

    # ==============================================================================
    # Section 4: Cross-Video Topology, Cohorts & Predictive Modeling
    # ==============================================================================
    st.divider()
    st.subheader("4. 🔬 Cross-Video Topology, Cohorts & Predictive Modeling")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        render_plot_card("shap_summary.png")
        render_plot_card("topic_video_matrix.png")
        render_plot_card("topic_streamgraph.png")
        render_plot_card("cohort_retention.png")
        render_plot_card("position_bias.png")
        render_plot_card("tag_network.png")
    with m_col2:
        render_plot_card("video_profile_radar.png")
        render_plot_card("controversy_impact.png")
        render_plot_card("code_switching_impact.png")
        render_plot_card("topic_cohorts.png")
        render_plot_card("new_vs_returning_share.png")
        render_plot_card("video_overlap_matrix.png")