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
        "comments": DATA_DIR / "interim" / "comments_clean.parquet",
        "author_fingerprints": DATA_DIR / "output" / "author_fingerprints.parquet",
        "video_radar": DATA_DIR / "output" / "video_profile_radar.parquet",
        "code_switching": DATA_DIR / "output" / "code_switching_impact.parquet",
        "topic_evolution": DATA_DIR / "output" / "topic_evolution.parquet",
        "reaction_map": DATA_DIR / "output" / "video_reaction_map.parquet",
        "dunn_matrix": DATA_DIR / "output" / "dunn_posthoc_matrix.parquet",
        "power_law": DATA_DIR / "output" / "power_law_fit.parquet"
    }
    
    # Check for file existence before reading
    missing_files = [str(p.relative_to(ROOT_DIR)) for name, p in paths.items() if name not in [
        "comments", "author_fingerprints", "video_radar", "code_switching", 
        "topic_evolution", "reaction_map", "dunn_matrix", "power_law"
    ] and not p.exists()]

    if missing_files:
        logger.error(f"UI Boot Blocked. Missing target paths: {missing_files}")
        st.error(f"❌ **Data Layer Mismatch:** Missing required files: {', '.join(missing_files)}")
        st.info("💡 Please verify that your test suite passes (`pytest -v`) before booting the UI.")
        st.stop()

    try:
        df_spikes = pd.read_parquet(paths["spikes"]) if paths["spikes"].exists() else pd.DataFrame()
        df_topics = pd.read_parquet(paths["topics"]) if paths["topics"].exists() else pd.DataFrame()
        df_timeline = pd.read_parquet(paths["timeline"]) if paths["timeline"].exists() else pd.DataFrame()
        df_comments = pd.read_parquet(paths["comments"]) if paths["comments"].exists() else None

        # Load optional advanced analytics layers
        extra_layers = {}
        for key in ["author_fingerprints", "video_radar", "code_switching", "topic_evolution", "reaction_map", "dunn_matrix", "power_law"]:
            if paths[key].exists():
                extra_layers[key] = pd.read_parquet(paths[key])

        return df_spikes, df_topics, df_timeline, df_comments, extra_layers

    except Exception as e:
        logger.exception("Fatal runtime exception encountered while reading parquet matrices.")
        st.error("⚠️ **Parquet Read Engine Failure**")
        st.exception(e)
        st.stop()

# Execution of data layer pull
df_spikes, df_topics, df_timeline, df_comments, extra_layers = load_pipeline_data()

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

if "power_law" in extra_layers and not extra_layers["power_law"].empty:
    pl_df = extra_layers["power_law"]
    alpha_val = pl_df['alpha'].iloc[0] if 'alpha' in pl_df.columns else None
    if alpha_val:
        st.sidebar.metric(label="Power-Law Exponent (α)", value=f"{alpha_val:.3f}")

st.sidebar.markdown("---")
st.sidebar.caption("System Environment: **Python 3.12** + **CUDA Acceleration**")

# ==============================================================================
# 4. MAIN INTERFACE LAYOUT
# ==============================================================================
st.title("📊 Conversational Topic Modeling & Analytics Dashboard")
st.markdown("Exploratory interface parsing high-density cluster structures, reaction dynamics, and predictive modeling.")

# Layout Tabs
tab_spikes, tab_topics, tab_timeline, tab_gallery, tab_advanced = st.tabs([
    "🚨 Event Spikes & Reactions", 
    "🧩 Micro-Cluster Explorer", 
    "📅 Macro Timeline Engine",
    "🎨 Visual Analytics Gallery",
    "🚀 Advanced Analytics"
])

# ------------------------------------------------------------------------------
# TAB 1: EVENT SPIKES & REACTIONS
# ------------------------------------------------------------------------------
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
                title="Spike Intensity Metric",
                labels={date_col: "Date Vector", count_col: "Volume Weight"},
                template="plotly_dark",
                hover_data={date_col: True, count_col: True},
                hover_name=date_col
            )
            st.plotly_chart(fig_spikes, width="stretch")
        
        st.subheader("Raw Layer Inspection: `viral_events`")
        st.dataframe(df_spikes, width="stretch")
    else:
        st.info("The viral events matrix parsed successfully but returned empty rows.")

    # Second-by-Second Reaction Heatmap/Timeline
    if "reaction_map" in extra_layers and not extra_layers["reaction_map"].empty:
        st.divider()
        st.header("🎬 Cross-Modal Reaction Timeline (Second-by-Second)")
        st.markdown("Aggregated comment sentiment and toxicity mapped across video playback timestamps.")
        
        df_rx = extra_layers["reaction_map"]
        selected_vid = st.selectbox("Select Video ID for Reaction Timeline:", options=df_rx["video_id"].unique())
        vid_rx = df_rx[df_rx["video_id"] == selected_vid].sort_values("second")
        
        fig_rx = px.line(
            vid_rx, 
            x="second", 
            y=["vader_compound", "toxicity"] if "toxicity" in vid_rx.columns else ["vader_compound"],
            title=f"Reaction Dynamics over Video Timeline // {selected_vid}",
            labels={"second": "Playback Time (Seconds)", "value": "Metric Score", "variable": "Dimension"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_rx, width="stretch")

# ------------------------------------------------------------------------------
# TAB 2: MICRO-CLUSTER EXPLORER
# ------------------------------------------------------------------------------
with tab_topics:
    st.header("Discovered Conversational Clusters")
    st.markdown("Granular semantic pockets grouped via high-speed UMAP dimensionality reduction and HDBSCAN.")
    
    if "avg_likes" in df_topics.columns and not df_topics.empty:
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
        st.plotly_chart(fig_resonance, width="stretch")
            
    search_query = st.text_input("🔍 Filter clusters by keyword/topic token:", "").strip().lower()
    
    filtered_topics = df_topics.copy()
    if search_query:
        text_cols = [c for c in filtered_topics.columns if filtered_topics[c].dtype == 'object']
        if text_cols:
            mask = filtered_topics[text_cols].astype(str).apply(lambda x: x.str.lower().str.contains(search_query)).any(axis=1)
            filtered_topics = filtered_topics[mask]
            logger.info(f"Applied keyword filter '{search_query}'. Matches remaining: {len(filtered_topics)}")
            
    st.metric(label="Filtered Cluster Count", value=len(filtered_topics))
    st.dataframe(filtered_topics, width="stretch")

    # Interactive Topic Evolution Area Chart
    if "topic_evolution" in extra_layers and not extra_layers["topic_evolution"].empty:
        st.divider()
        st.subheader("📈 Conversational Topic Evolution Over Time")
        df_te = extra_layers["topic_evolution"]
        if "year_month" in df_te.columns:
            fig_te = px.area(
                df_te, 
                x="year_month", 
                y=[c for c in df_te.columns if c != "year_month"],
                title="Topic Share Evolution Across Months",
                labels={"year_month": "Timeline Month", "value": "Comment Volume", "variable": "Topic"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_te, width="stretch")

    # Code-Switching Impact Chart
    if "code_switching" in extra_layers and not extra_layers["code_switching"].empty:
        st.divider()
        st.subheader("🔀 Code-Switching Engagement Impact")
        df_cs = extra_layers["code_switching"]
        fig_cs = px.bar(
            df_cs,
            x="type",
            y="avg_likes",
            color="type",
            title="Monolingual vs. Code-Switched Average Like Engagement",
            labels={"type": "Comment Type", "avg_likes": "Average Likes per Comment"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_cs, width="stretch")

# ------------------------------------------------------------------------------
# TAB 3: MACRO TIMELINE ENGINE
# ------------------------------------------------------------------------------
with tab_timeline:
    st.header("Macro Historical Timelines")
    st.markdown("Longitudinal baseline eras across the full data collection scope.")
    
    if not df_timeline.empty:
        fig_timeline = px.line(
            df_timeline, 
            x="date", 
            y="comment_count", 
            title="Total Conversational Volume Over Time",
            template="plotly_dark"
        )
        st.plotly_chart(fig_timeline, width="stretch")
    
    st.subheader("Raw Layer Inspection: `historical_timeline`")
    st.dataframe(df_timeline, width="stretch")

# ------------------------------------------------------------------------------
# TAB 4: VISUAL ANALYTICS GALLERY (All 14 Stage 06 Visualizations)
# ------------------------------------------------------------------------------
with tab_gallery:
    st.header("🖼️ Stage 06 Publication-Ready Visual Analytics Gallery")
    st.markdown("All 14 analytical plots and interactive models rendered by `s06_visualize.py`.")

    plots_dir = DATA_DIR / "output" / "plots"
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

# ------------------------------------------------------------------------------
# TAB 5: ADVANCED ANALYTICS (Interactive Models & Stages 15 - 34)
# ------------------------------------------------------------------------------
with tab_advanced:
    st.header("🚀 Advanced Interactive Models & Visualizations (Stages 15-34)")
    st.markdown("Deep-dive interactive plots spanning author behavioral radar, post-hoc statistical matrices, and gallery figures.")
    
    # Interactive Author Fingerprints Polar Radar Chart
    if "author_fingerprints" in extra_layers and not extra_layers["author_fingerprints"].empty:
        st.subheader("🕸️ Top Author Behavioral Fingerprints (Polar Radar)")
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

    # Interactive Dunn's Post-Hoc Matrix Heatmap
    if "dunn_matrix" in extra_layers and not extra_layers["dunn_matrix"].empty:
        st.subheader("📊 Dunn's Post-Hoc Pairwise p-Value Matrix")
        df_dunn = extra_layers["dunn_matrix"]
        fig_dunn = px.imshow(
            df_dunn,
            labels=dict(x="Video ID", y="Video ID", color="Adjusted p-value"),
            x=df_dunn.columns,
            y=df_dunn.index,
            title="Statistical Significance Matrix of Pairwise Video Engagements",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        st.plotly_chart(fig_dunn, width="stretch")
        
    st.divider()
    adv_col1, adv_col2 = st.columns(2)
    
    advanced_plots = [
        ("reaction_timeline.png", "Reaction Timeline", "Early vs Late reaction velocities."),
        ("integrity_scatter.png", "Integrity Scatter", "Brigading & Spam Anomaly Detection."),
        ("thread_polarization.png", "Thread Polarization", "Polarization dynamics within comment trees."),
        ("topic_streamgraph.png", "Topic Streamgraph", "Evolution of topics over time."),
        ("reply_latency_distribution.png", "Reply Latency", "Speed of replies within the community."),
        ("shelf_life_likes.png", "Shelf Life of Likes", "How long comments continue to accrue likes."),
        ("topic_video_matrix.png", "Topic-Video Matrix", "Heatmap of topics mapped across videos."),
        ("word_cooccurrence.png", "Word Co-occurrence", "Network graph of frequently associated words."),
        ("named_entities.png", "Named Entities", "Most common NER targets."),
        ("sarcasm_analysis.png", "Sarcasm Analysis", "Heuristic detection of sarcastic comments."),
        ("linguistic_profiling.png", "Linguistic Profiling", "Lexical richness and style markers."),
        ("tag_network.png", "Tag Network", "Hashtag and @mention topologies."),
        ("bipartite_network.png", "Bipartite Network", "User-to-Video engagement clustering."),
        ("cocomment_network.png", "Co-Comment Network", "Users who frequently comment on the same videos."),
        ("cohort_retention.png", "Cohort Retention", "Long-term engagement survival of user cohorts."),
        ("video_overlap_matrix.png", "Video Overlap", "Audience cross-pollination between videos."),
        ("new_vs_returning_share.png", "New vs Returning", "Audience composition over time."),
        ("topic_cohorts.png", "Topic Cohorts", "User clustering based on thematic interests."),
        ("frequency_tiers.png", "Frequency Tiers", "Casual vs Hardcore audience breakdown."),
        ("bot_heuristics.png", "Bot Heuristics", "Temporal anomaly detection for botting."),
        ("driveby_loyalists.png", "Drive-by vs Loyalists", "Engagement consistency classification."),
        ("position_bias.png", "Position Bias", "Early-bird advantage in like accrual."),
        ("arrival_speed.png", "Arrival Speed", "First responder classification."),
        ("thread_width_dist.png", "Thread Width", "Horizontal branching vs vertical depth."),
        ("resolution_patterns.png", "Resolution Patterns", "Sentiment decay at terminal nodes."),
        ("initiator_patterns.png", "Initiator Patterns", "Broadcaster vs Responder conversational roles."),
        ("code_switching_impact.png", "Code-Switching", "Multilingual flexing & its impact on engagement."),
        ("video_profile_radar.png", "Video Profile Radar", "Comparative footprint of top videos."),
        ("controversy_impact.png", "Controversy Impact", "Channel-wide sentiment shifts before/after toxic events."),
        ("like_inflation.png", "Like Inflation", "Suspicious astroturfing (high likes, zero replies)."),
        ("author_fingerprints.png", "Author Fingerprints", "Behavioral radar for the top 5 super-fans."),
        ("impersonation_detection.png", "Impersonation Spoofing", "Fraud: Scam accounts cloning creator display names.")
    ]
    
    for idx, (filename, title, desc) in enumerate(advanced_plots):
        col = adv_col1 if idx % 2 == 0 else adv_col2
        with col:
            render_plot_card(title, filename, desc)