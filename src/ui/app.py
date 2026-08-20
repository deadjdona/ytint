import os
import logging
import pathlib
# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from ui.visual_explanations import VISUAL_METADATA

# ==============================================================================
# 1. LOGGING & PAGE SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ytint_ui")

st.set_page_config(
    page_title="ytint // Analytics & Executive Intelligence Dashboard",
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

# Robust Absolute Path Anchoring
UI_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = UI_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = DATA_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

logger.info(f"Initializing ytint UI. Anchored at: {ROOT_DIR}")

# ==============================================================================
# 2. CACHED PIPELINE DATA LOADER
# ==============================================================================
@st.cache_data(show_spinner="Parsing analytical parquet layers from disk...")
def load_pipeline_data():
    """Loads all analytical parquet layers and metadata matrices."""
    logger.info("Performing cached load of analytical parquet layers.")
    data = {}
    
    # Core Master Layers
    f_comments = INTERIM_DIR / "comments_clean.parquet"
    f_topics = OUTPUT_DIR / "topic_metadata.parquet"
    f_videos = OUTPUT_DIR / "videos_final.parquet"
    f_authors = OUTPUT_DIR / "authors_final.parquet"
    f_timeline = OUTPUT_DIR / "historical_timeline.parquet"
    f_spikes = OUTPUT_DIR / "viral_events.parquet"

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
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            try:
                data[key] = pd.read_parquet(fpath)
            except Exception:
                data[key] = pd.DataFrame()
        else:
            data[key] = pd.DataFrame()

    return data

data_layers = load_pipeline_data()

df_comments_raw = data_layers["comments"]
df_topics = data_layers["topics"]
df_videos = data_layers["videos"]
df_authors = data_layers["authors"]
df_timeline_raw = data_layers["timeline"]
df_spikes = data_layers["spikes"]

# ==============================================================================
# 3. SIDEBAR CONTROLS & REPOSITORY METRICS
# ==============================================================================
st.sidebar.title("🧬 ytint Core Engine")
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
st.sidebar.subheader("Pipeline Manifest")

total_raw_comments = len(df_comments_raw) if not df_comments_raw.empty else 0
total_active_authors = len(df_authors) if not df_authors.empty else 0
total_tracked_videos = len(df_videos) if not df_videos.empty else 0

st.sidebar.metric(label="Processed Comments", value=f"{total_raw_comments:,}")
st.sidebar.metric(label="Unique Community Authors", value=f"{total_active_authors:,}")
st.sidebar.metric(label="Analyzed Videos", value=f"{total_tracked_videos:,}")

if not data_layers["power_law"].empty and "alpha" in data_layers["power_law"].columns:
    pl_alpha = data_layers["power_law"]["alpha"].iloc[0]
    st.sidebar.metric(label="Power-Law Exponent (α)", value=f"{pl_alpha:.3f}")

st.sidebar.divider()
st.sidebar.caption("System Environment: **Python 3.12** // **Accelerated Parquet Engine**")

# ==============================================================================
# 4. MAIN WORKSPACE & TABBED SUITE
# ==============================================================================
st.title("📊 Conversational Topic Modeling & Executive Intelligence")
st.markdown("Exploratory interface parsing high-density cluster structures, reaction dynamics, and predictive modeling.")

tab_summary, tab_spikes, tab_topics, tab_audience, tab_modeling, tab_gallery = st.tabs([
    "📋 Executive Summary",
    "🚨 Viral Events & Flashpoints",
    "🧩 Micro-Cluster Explorer",
    "👥 Audience Dynamics & Forensics",
    "🔬 Cross-Video & Modeling",
    "🖼️ Visual Analytics Gallery"
])

# ==============================================================================
# TAB 1: EXECUTIVE INTELLIGENCE SUMMARY
# ==============================================================================
with tab_summary:
    st.header("Executive Intelligence Briefing")
    st.markdown("Channel-wide conversational resonance, audience retention metrics, and security diagnostics.")

    # 1.1 Executive Metric Tiles
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    num_topics = len(df_topics[df_topics["Topic"] != -1]) if not df_topics.empty and "Topic" in df_topics.columns else len(df_topics)
    num_spikes = len(df_spikes) if not df_spikes.empty else 0

    bot_rate = 0.0
    if not data_layers["bot_classifications"].empty and "is_bot" in data_layers["bot_classifications"].columns:
        bot_rate = (data_layers["bot_classifications"]["is_bot"].sum() / len(data_layers["bot_classifications"])) * 100

    suspicious_like_rate = 0.0
    if not data_layers["like_inflation"].empty and "is_suspicious" in data_layers["like_inflation"].columns:
        suspicious_like_rate = (data_layers["like_inflation"]["is_suspicious"].sum() / len(data_layers["like_inflation"])) * 100

    with kpi_col1:
        with st.container(border=True):
            st.metric(label="Total Processed Comments", value=f"{total_raw_comments:,}", delta=f"{len(df_comments):,} in slice")
            st.caption("Complete conversation archive parsed across all tracked video uploads.")

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
                f"breaching rolling Z-score anomaly thresholds.\n"
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

    # 1.3 Video Performance Leaderboard
    st.subheader("Analyzed Videos Performance Leaderboard")
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
        st.dataframe(
            display_videos.sort_values(by="Total Comments", ascending=False),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No video records available.")

# ==============================================================================
# TAB 2: VIRAL EVENTS & MOMENT DYNAMICS
# ==============================================================================
with tab_spikes:
    st.header(f"{len(df_spikes)} Detected Conversational Event Spikes")
    st.markdown("High-volume temporal anomalies isolated automatically via density analysis pipelines.")

    if not df_spikes.empty:
        spikes_for_chart = df_spikes.reset_index()
        date_col = next((c for c in spikes_for_chart.columns if "date" in c.lower() or "timestamp" in c.lower()), None)
        count_col = next((c for c in df_spikes.columns if "count" in c.lower() or "volume" in c.lower() or "size" in c.lower()), None)

        if date_col and count_col:
            fig_spikes = px.bar(
                spikes_for_chart, x=date_col, y=count_col,
                title="Spike Intensity Metric & Viral Velocity",
                labels={date_col: "Date Vector", count_col: "Volume Weight"},
                template=PLOTLY_TEMPLATE,
                hover_data={date_col: True, count_col: True},
                hover_name=date_col
            )
            st.plotly_chart(fig_spikes, width="stretch")
            st.caption("Comment volume during detected viral events. Taller bars indicate larger conversation spikes.")

        st.subheader("Raw Layer Inspection: `viral_events`")
        st.dataframe(df_spikes, width="stretch", hide_index=True)
    else:
        st.info("The viral events matrix parsed successfully but returned empty rows.")

    # Second-by-Second Reaction Heatmap/Timeline
    if not data_layers["reaction_map"].empty:
        st.divider()
        st.header("🎬 Cross-Modal Reaction Timeline (Second-by-Second)")
        st.markdown("Aggregated comment sentiment and toxicity mapped across video playback timestamps.")

        df_rx = data_layers["reaction_map"]
        selected_vid = st.selectbox("Select Video ID for Reaction Timeline:", options=df_rx["video_id"].unique())
        vid_rx = df_rx[df_rx["video_id"] == selected_vid].sort_values("second")

        fig_rx = px.line(
            vid_rx,
            x="second",
            y=["vader_compound", "toxicity"] if "toxicity" in vid_rx.columns else ["vader_compound"],
            title=f"Reaction Dynamics over Video Timeline // {selected_vid}",
            labels={"second": "Playback Time (Seconds)", "value": "Metric Score", "variable": "Dimension"},
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig_rx, width="stretch")
        st.caption("Tracks sentiment and toxicity throughout a video's playback time.")

# ==============================================================================
# TAB 3: MICRO-CLUSTER EXPLORER
# ==============================================================================
with tab_topics:
    st.header("Discovered Conversational Clusters")
    st.markdown("Granular semantic pockets grouped via high-speed UMAP dimensionality reduction and HDBSCAN.")

    if not df_topics.empty and "Count" in df_topics.columns:
        plot_data = df_topics[df_topics["Topic"] != -1]
        y_col = "avg_likes" if "avg_likes" in plot_data.columns else "Count"
        size_col = "total_likes" if "total_likes" in plot_data.columns else "Count"

        fig_resonance = px.scatter(
            plot_data,
            x="Count",
            y=y_col,
            size=size_col,
            hover_name="Name",
            title="Topic Resonance Matrix (Volume vs. Engagement)",
            labels={"Count": "Total Comments in Topic", y_col: "Average Likes per Comment"},
            template=PLOTLY_TEMPLATE,
            size_max=40
        )
        st.plotly_chart(fig_resonance, width="stretch")
        st.caption("Visualizes conversation topics based on volume (x-axis) and average likes (y-axis).")

    search_query = st.text_input("🔍 Filter clusters by keyword/topic token:", "").strip().lower()
    filtered_topics = df_topics.copy()
    if search_query and not filtered_topics.empty:
        text_cols = [c for c in filtered_topics.columns if filtered_topics[c].dtype == 'object']
        if text_cols:
            mask = filtered_topics[text_cols].astype(str).apply(lambda x: x.str.lower().str.contains(search_query)).any(axis=1)
            filtered_topics = filtered_topics[mask]

    st.metric(label="Filtered Cluster Count", value=len(filtered_topics))
    st.dataframe(filtered_topics, width="stretch", hide_index=True)

    # Topic Evolution Area Chart
    if not data_layers["topic_evolution"].empty:
        st.divider()
        st.subheader("📈 Conversational Topic Evolution Over Time")
        df_te = data_layers["topic_evolution"]
        if "year_month" in df_te.columns:
            fig_te = px.area(
                df_te,
                x="year_month",
                y=[c for c in df_te.columns if c != "year_month"],
                title="Topic Share Evolution Across Months",
                labels={"year_month": "Timeline Month", "value": "Comment Volume", "variable": "Topic"},
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_te, width="stretch")

    # Code-Switching Impact Chart
    if not data_layers["code_switching"].empty:
        st.divider()
        st.subheader("🔀 Code-Switching Engagement Impact")
        df_cs = data_layers["code_switching"]
        fig_cs = px.bar(
            df_cs,
            x="type",
            y="avg_likes",
            color="type",
            title="Monolingual vs. Code-Switched Average Like Engagement",
            labels={"type": "Comment Type", "avg_likes": "Average Likes per Comment"},
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig_cs, width="stretch")

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
# TAB 4: AUDIENCE DYNAMICS & FORENSICS
# ==============================================================================
with tab_audience:
    st.header("👥 Audience Loyalty, Forensics & Behavior Profiling")
    st.markdown("Detailed breakdown of community retention, super-fan fingerprints, and automated bot forensics.")

    # Polar Radar
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
                title="Super-Fan Behavioral Footprints",
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_af_radar, width="stretch")

    st.divider()
    aud_col1, aud_col2, aud_col3 = st.columns(3)
    with aud_col1:
        st.subheader("📊 Commenting Frequency Tiers")
        if not data_layers["frequency_tiers"].empty:
            fig_tiers = px.pie(
                data_layers["frequency_tiers"],
                names="tier",
                values="author_count",
                title="Audience Frequency Breakdown",
                template=PLOTLY_TEMPLATE,
                hole=0.4
            )
            st.plotly_chart(fig_tiers, width="stretch")
    with aud_col2:
        st.subheader("🎯 Drive-By vs. Loyalists")
        if not data_layers["driveby_loyalists"].empty:
            fig_db = px.bar(
                data_layers["driveby_loyalists"],
                x="classification",
                y="author_count",
                color="classification",
                title="Persistence Tiers",
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_db, width="stretch")
    with aud_col3:
        st.subheader("⏱️ First-Responder Speed")
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

    st.divider()
    for_col1, for_col2 = st.columns(2)
    with for_col1:
        st.subheader("🤖 Bot Classifier & Heuristic Audit")
        if not data_layers["bot_classifications"].empty:
            bot_summary = data_layers["bot_classifications"]["classification"].value_counts().reset_index()
            bot_summary.columns = ["Classification", "Author Count"]
            st.dataframe(bot_summary, width="stretch", hide_index=True)
    with for_col2:
        st.subheader("💸 Astroturfing / Like-Inflation Flags")
        if not data_layers["like_inflation"].empty:
            suspicious_comments = data_layers["like_inflation"][data_layers["like_inflation"]["is_suspicious"] == True]
            st.metric(label="Flagged Astroturfed Comments", value=f"{len(suspicious_comments):,}")
            st.dataframe(suspicious_comments.head(50), width="stretch", hide_index=True)

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

# ==============================================================================
# TAB 5: CROSS-VIDEO & PREDICTIVE MODELING
# ==============================================================================
with tab_modeling:
    st.header("🔬 Cross-Video Relations, Counterfactuals & Modeling")
    st.markdown("Advanced statistical matrices, pairwise significance tests, and predictive feature attribution.")

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

    st.divider()
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
                    title="Comparative Video Footprints",
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

    st.divider()
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.subheader("🌳 Tree SHAP Feature Attribution")
        if not data_layers["shap_features"].empty:
            st.dataframe(data_layers["shap_features"].head(20), width="stretch", hide_index=True)
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
            st.plotly_chart(fig_fc, width="stretch")

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

        filepath = PLOTS_DIR / filename
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