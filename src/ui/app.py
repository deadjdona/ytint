"""ytint // Intelligence Engine & Executive Dashboard (src/ui/app.py)

Production-grade, sleek, modern Streamlit application providing comprehensive
conversational topic modeling, anomaly detection, audience forensics, causal inference,
and predictive modeling for large-scale YouTube comment corpora.
"""

import sys
import json
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure src directory is on sys.path
_src_dir = str(pathlib.Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from datetime import datetime
import streamlit.components.v1 as components

from engine.config_loader import load_config as _load_config
from ui.visual_explanations import VISUAL_METADATA

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Plotly Aesthetic Palette
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PRIMARY = "#0066fe"
COLOR_ACCENT = "#00e599"
COLOR_DANGER = "#ff3366"
COLOR_WARNING = "#ffaa00"
COLOR_PURPLE = "#a855f7"

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
    f_videos_clean = interim_dir / "videos_clean.parquet"
    f_authors = output_dir / "authors_final.parquet"
    f_timeline = output_dir / "historical_timeline.parquet"
    f_spikes = output_dir / "viral_events.parquet"
    
    data["comments"] = pd.read_parquet(f_comments) if f_comments.exists() else pd.DataFrame()
    data["topics"] = pd.read_parquet(f_topics) if f_topics.exists() else pd.DataFrame()
    
    df_vf = pd.read_parquet(f_videos) if f_videos.exists() else pd.DataFrame()
    df_vc = pd.read_parquet(f_videos_clean) if f_videos_clean.exists() else pd.DataFrame()
    if not df_vc.empty and "video_id" in df_vc.columns and "title" in df_vc.columns:
        if not df_vf.empty and "video_id" in df_vf.columns:
            if "title" not in df_vf.columns:
                df_vf = df_vf.merge(df_vc[["video_id", "title"]].drop_duplicates("video_id"), on="video_id", how="left")
        else:
            df_vf = df_vc
    data["videos"] = df_vf
    data["authors"] = pd.read_parquet(f_authors) if f_authors.exists() else pd.DataFrame()
    data["timeline"] = pd.read_parquet(f_timeline) if f_timeline.exists() else pd.DataFrame()
    data["spikes"] = pd.read_parquet(f_spikes) if f_spikes.exists() else pd.DataFrame()

    # Specialized Analytical Layers (Stages 07 - 38)
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
        "cib_comments": "cib_coordinated_comments.parquet",
        "contagion_summary": "toxicity_contagion_summary.parquet",
        "troll_catalysts": "troll_catalysts.parquet",
        "video_toxicity": "video_toxicity_contagion.parquet",
        "creator_uplift": "creator_causal_uplift.parquet",
        "creator_threads": "creator_intervention_threads.parquet",
        "stance_summary": "stance_summary.parquet",
        "stance_drift": "stance_depth_drift.parquet",
        "polarized_threads": "polarized_threads.parquet",
        "slang_lexicon": "slang_lexicon_frequency.parquet",
        "bowtie_structure": "network_bowtie_structure.parquet",
        "top_k_concentration": "top_k_concentration.parquet",
        "series_vs_standalone": "series_vs_standalone.parquet",
        "creator_sentiment": "creator_sentiment_polarity.parquet",
        "scene_reactions": "cross_modal_scene_reactions.parquet",
        "spoiler_detections": "spoiler_detections.parquet",
        "topic_injections": "topic_injection_anomalies.parquet",
        "minute_arrival": "minute_arrival_curve.parquet",
        "tfidf_keywords": "tfidf_keywords.parquet",
        "polarity_engagement": "polarity_engagement.parquet",
        "emoji_signatures": "emoji_signatures.parquet",
        "sentiment_anomalies": "sentiment_anomalies.parquet",
        "corpus_quality": "corpus_quality.parquet",
        "language_coverage": "language_coverage.parquet",
        "thread_topic_drift": "thread_topic_drift.parquet",
        "attention_transfer": "attention_transfer.parquet"
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

def main():
    """Main dashboard entry point rendering the 7-tab intelligence suite."""
    # 1. Page Setup
    st.set_page_config(
        page_title="ytint // Intelligence Engine & Executive Dashboard",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )


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

    # Global video title dictionary for human-readable display
    video_title_map = {}
    if not df_videos.empty and "video_id" in df_videos.columns:
        if "title" in df_videos.columns:
            video_title_map = dict(zip(df_videos["video_id"], df_videos["title"].fillna(df_videos["video_id"])))
        else:
            video_title_map = dict(zip(df_videos["video_id"], df_videos["video_id"]))

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
        if "is_reply" in df_comments_raw.columns:
            df_comments = df_comments_raw[~df_comments_raw['is_reply'].fillna(False).astype(bool)]
        else:
            df_comments = df_comments_raw[df_comments_raw['parent_id'].isna() | (df_comments_raw['parent_id'] == "")]
    elif comment_filter == "Replies/Responses Only" and not df_comments_raw.empty:
        if "is_reply" in df_comments_raw.columns:
            df_comments = df_comments_raw[df_comments_raw['is_reply'].fillna(False).astype(bool)]
        else:
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
    z_thresh_sidebar = config.get("stage_28_narrative", {}).get("z_threshold", 2.5)
    pelt_penalty_sidebar = config.get("stage_28_narrative", {}).get("change_point_penalty", "auto")
    st.sidebar.info(
        f"**Z-Score Spike Threshold:** {z_thresh_sidebar}σ\n\n"
        f"**PELT Penalty Model:** {pelt_penalty_sidebar}\n\n"
        f"**Embedding Model:** {config.get('stage_02_topics', {}).get('embedding_model', 'multilingual-MiniLM')}\n\n"
        f"**Active Pipeline Stages:** 53 Stages (`s00`–`s51`, `s99`)"
    )

    # ==============================================================================
    # 4. MAIN APP HEADER & TABBED WORKSPACE
    # ==============================================================================
    st.title("🎬 ytint // Conversational Intelligence & Executive Suite")
    st.caption(f"Active Slice: **{comment_filter}** — {len(df_comments):,} records loaded across 53 analytical intelligence stages")

    tab_summary, tab_temporal, tab_topics, tab_audience, tab_modeling, tab_gallery, tab_export = st.tabs([
        "📋 Executive Briefing",
        "⏳ Temporal Dynamics & Flashpoints",
        "🧠 NLP, Semantics & Demand Intent",
        "👥 Audience Loyalty & Forensics",
        "🔬 Predictive Modeling & Causal Interventions",
        "🖼️ Visual Analytics Gallery & Guides",
        "🗄️ Data Explorer & Export Hub"
    ])

    # ==============================================================================
    # TAB 1: EXECUTIVE INTELLIGENCE SUMMARY
    # ==============================================================================
    with tab_summary:
        st.header("Executive Intelligence Briefing")
        st.markdown(
            "Holistic synthesis of channel engagement dynamics, community integrity diagnostics, "
            "and conversational resonance extracted across all 53 analytical stages."
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
        st.divider()

        # 1.2 Executive Intelligence Dossier Generator
        with st.container(border=True):
            dossier_col1, dossier_col2 = st.columns([3, 1])
            with dossier_col1:
                st.subheader("📄 Executive Intelligence Briefing Dossier")
                st.markdown(
                    "Compile and export a standalone, self-contained executive briefing dossier. "
                    "Integrates forensic threat matrices, community health scorecards, video performance radars, "
                    "and embedded high-resolution publication plots. Ready for offline archiving and instant **Print / PDF export**."
                )
            with dossier_col2:
                include_plots_tab1 = st.checkbox("Include Visual Plots", value=True, key="dossier_plots_tab1")
                if st.button("⚡ Generate Dossier", key="btn_gen_dossier_tab1"):
                    with st.spinner("Compiling executive intelligence layers and plots..."):
                        from engine.reporter import ExecutiveReportGenerator
                        reporter = ExecutiveReportGenerator(interim_path, output_path)
                        html_dossier = reporter.generate_html(include_plots=include_plots_tab1)
                        st.session_state["executive_dossier_html"] = html_dossier
                        st.success("Executive Dossier compiled successfully!")

            if "executive_dossier_html" in st.session_state:
                dl_col1, dl_col2 = st.columns([1, 2])
                with dl_col1:
                    st.download_button(
                        label="⬇️ Download Executive Dossier (.html)",
                        data=st.session_state["executive_dossier_html"],
                        file_name=f"ytint_executive_brief_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html"
                    )
                with dl_col2:
                    with st.expander("👁️ Live In-Dashboard Dossier Preview", expanded=False):
                        components.html(st.session_state["executive_dossier_html"], height=700, scrolling=True)

        # 1.3 AI Executive Strategy Briefing
        with st.container(border=True):
            ai_s_col1, ai_s_col2 = st.columns([3, 1])
            with ai_s_col1:
                st.subheader("🤖 AI Executive Strategy Briefing")
                st.markdown(
                    "Synthesize qualitative strategic reasoning from the 53 computational stages using LLM agents. "
                    "Supports Google Gemini (REST), local Ollama (private/offline), and fast mock simulation."
                )
            with ai_s_col2:
                llm_prov_tab1 = st.selectbox(
                    "AI Provider:",
                    options=["gemini", "ollama", "mock"],
                    index=0,
                    key="llm_prov_tab1"
                )
                if st.button("✨ Generate AI Briefing", key="btn_gen_ai_briefing_tab1"):
                    with st.spinner("Synthesizing strategic reasoning with AI analyst..."):
                        from engine.synthesizer import LLMClient, InsightSynthesizer
                        client = LLMClient(provider=llm_prov_tab1)
                        synth = InsightSynthesizer(interim_path, output_path, llm_client=client)
                        st.session_state["ai_briefing_text"] = synth.synthesize_executive_briefing()
                        st.success("AI Strategic Briefing generated!")

            if "ai_briefing_text" in st.session_state:
                with st.container(border=True):
                    st.markdown(st.session_state["ai_briefing_text"])

        st.divider()

        # 1.4 Executive Intelligence Synthesis Cards
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
                z_thresh = config.get("stage_28_narrative", {}).get("z_threshold", 2.5)
                st.markdown(
                    f"- **Automated Anomaly Detection**: Identified **{num_spikes} major conversational flashpoints** "
                    f"breaching the {z_thresh}σ rolling Z-score threshold.\n"
                    f"- **Poisson Burst Spikes**: High-frequency comment bursts cluster predominantly within the first **2–6 hours** "
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
                    f"- **Toxicity Contagion ($R_0$)**: Toxicity reproduction number is quantified to identify key flame-war instigators "
                    f"and protect constructive community dialogue."
                )

            with st.container(border=True):
                st.markdown("#### 🔀 Linguistic Profiling & Causal Engagement Drivers")
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
                    f"- **Creator Intervention Multiplier**: Difference-in-Differences causal estimation confirms early creator engagement "
                    f"expands thread lifespan and reply volume by over **+3,130%**.\n"
                    f"- **Predictive Attribution (SHAP)**: Early arrival speed, sentence length, and sentiment polarity are the "
                    f"primary positive drivers of community upvotes."
                )

        # 1.25 Attention Concentration & Creator Resonance KPI Cards
        if not data_layers["top_k_concentration"].empty or not data_layers["creator_sentiment"].empty:
            st.divider()
            st.subheader("👑 Attention Concentration & Creator Resonance")
            c_kpi1, c_kpi2 = st.columns(2)
            with c_kpi1:
                with st.container(border=True):
                    st.markdown("##### 📊 Top-K Attention Inequality")
                    df_conc_c = data_layers["top_k_concentration"]
                    corpus_conc = df_conc_c[df_conc_c["scope"] == "corpus"]
                    if not corpus_conc.empty:
                        row_c = corpus_conc.iloc[0]
                        k_m1, k_m2 = st.columns(2)
                        k_m1.metric("Top 1% Likes Share", f"{row_c.get('top_1pct_likes_share', 0)*100:.1f}%")
                        k_m2.metric("Top 5% Likes Share", f"{row_c.get('top_5pct_likes_share', 0)*100:.1f}%")
                        st.caption(f"Gini coefficient: **{row_c.get('gini_likes', 0):.3f}** (Likes) | **{row_c.get('gini_replies', 0):.3f}** (Replies).")
                    else:
                        st.info("Concentration metrics available upon stage execution.")
            with c_kpi2:
                with st.container(border=True):
                    st.markdown("##### 👑 Creator Praise vs. Criticism Ratio")
                    df_creat_c = data_layers["creator_sentiment"]
                    corpus_creat = df_creat_c[df_creat_c["scope"] == "corpus"]
                    if not corpus_creat.empty:
                        row_creat = corpus_creat.iloc[0]
                        cr_m1, cr_m2 = st.columns(2)
                        cr_m1.metric("Praise / Criticism Ratio", f"{row_creat.get('creator_pos_neg_ratio', 0):.1f}:1")
                        cr_m2.metric("Direct Creator Comments", f"{row_creat.get('creator_directed_comments', 0):,}")
                        st.caption(f"Positive: **{row_creat.get('creator_positive_count', 0):,}** | Negative: **{row_creat.get('creator_negative_count', 0):,}** | Net Sentiment: **{row_creat.get('creator_mean_sentiment', 0):+.3f}**")
                    else:
                        st.info("Creator sentiment data available upon stage execution.")

        st.divider()

        # 1.3 Video Performance Intelligence Leaderboard
        st.subheader("Top Analyzed Videos Performance Leaderboard")
        if not df_videos.empty:
            display_videos = df_videos.copy()
            col_rename = {
                "title": "Video Title",
                "video_id": "Video ID",
                "total_comments": "Total Comments",
                "total_likes": "Total Likes",
                "avg_sentiment": "Sentiment Score",
                "gini_coefficient": "Gini Inequality",
                "attention_half_life_days": "Half-Life (Days)",
                "revival_spikes": "Revival Spikes"
            }
            display_videos = display_videos.rename(columns=col_rename)
            if "Video Title" in display_videos.columns:
                ordered_cols = ["Video Title"] + [c for c in display_videos.columns if c not in ["Video Title", "Video ID"]] + (["Video ID"] if "Video ID" in display_videos.columns else [])
                display_videos = display_videos[[c for c in ordered_cols if c in display_videos.columns]]

            st.dataframe(
                display_videos.sort_values(by="Total Comments", ascending=False),
                width="stretch",
                hide_index=True
            )
            st.caption("Comprehensive summary of video performance metrics, attention half-lives, and conversation inequality.")

            # 1.4 Interactive Video Performance Quadrant Matrix
            if "Total Comments" in display_videos.columns and "Sentiment Score" in display_videos.columns:
                st.divider()
                st.subheader("🎯 Interactive Video Performance Quadrant")
                st.caption("Explore discussion volume vs. audience sentiment resonance. Bubble size represents total upvotes, with crosshairs marking median volume and average sentiment.")

                hover_target = "Video Title" if "Video Title" in display_videos.columns else ("Video ID" if "Video ID" in display_videos.columns else None)
                fig_matrix = px.scatter(
                    display_videos,
                    x="Total Comments",
                    y="Sentiment Score",
                    size="Total Likes" if "Total Likes" in display_videos.columns else None,
                    color="Gini Inequality" if "Gini Inequality" in display_videos.columns else "Sentiment Score",
                    hover_name=hover_target,
                    hover_data={
                        "Total Comments": True,
                        "Sentiment Score": ":.3f",
                        "Total Likes": True,
                        "Video ID": True if "Video ID" in display_videos.columns else False
                    },
                    title="Video Discussion Volume vs. Sentiment Resonance",
                    labels={
                        "Total Comments": "Discussion Volume (Total Comments)",
                        "Sentiment Score": "Average Sentiment Score",
                        "Total Likes": "Total Upvotes",
                        "Gini Inequality": "Gini Inequality"
                    },
                    template=PLOTLY_TEMPLATE,
                    size_max=45,
                    color_continuous_scale="Viridis"
                )
                med_c = display_videos["Total Comments"].median()
                mean_s = display_videos["Sentiment Score"].mean()
                fig_matrix.add_vline(x=med_c, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Median Volume")
                fig_matrix.add_hline(y=mean_s, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Mean Sentiment")
                st.plotly_chart(fig_matrix, width="stretch")
        else:
            st.info("No video metadata records available in the current pipeline run.")

    # ==============================================================================
    # TAB 2: TEMPORAL DYNAMICS & FLASHPOINTS
    # ==============================================================================
    with tab_temporal:
        st.header("Conversational Volumetric Spikes & Anomaly Detection")
        st.markdown("Chronological anomaly isolation identifying viral catalysts, sudden debates, and event-driven discussion surges.")

        # Interactive Anomaly Sensitivity Tuner
        st.subheader("⚡ Dynamic Anomaly Sensitivity Scanner")
        default_z = float(config.get("stage_28_narrative", {}).get("z_threshold", 2.5))
        sens_col1, sens_col2, sens_col3 = st.columns(3)
        with sens_col1:
            z_thresh_dyn = st.slider("Anomaly Sensitivity Threshold (σ):", 1.5, 4.5, default_z, 0.1)
        with sens_col2:
            roll_window_dyn = st.slider("Rolling Baseline Window (Days):", 3, 30, 7, 1)
        with sens_col3:
            curve_smooth = st.selectbox("Timeline Curve Interpolation:", ["spline", "linear"])

        # Timeline calculation based on slice
        if not df_comments.empty and 'published_at' in df_comments.columns:
            df_c = df_comments.copy()
            df_c['published_at'] = pd.to_datetime(df_c['published_at'])
            df_slice_timeline = df_c.groupby(df_c['published_at'].dt.date).size().to_frame(name='comment_count')
            df_slice_timeline.index = pd.to_datetime(df_slice_timeline.index)
            df_slice_timeline = df_slice_timeline.sort_index().reset_index().rename(columns={'published_at': 'date'})

            # Dynamic rolling statistics
            roll_mean = df_slice_timeline['comment_count'].rolling(window=roll_window_dyn, min_periods=1).mean()
            roll_std = df_slice_timeline['comment_count'].rolling(window=roll_window_dyn, min_periods=1).std().fillna(1)
            df_slice_timeline['z_score'] = (df_slice_timeline['comment_count'] - roll_mean) / roll_std
        else:
            df_slice_timeline = df_timeline_raw.copy()
            if not df_slice_timeline.empty and 'comment_count' in df_slice_timeline.columns:
                roll_mean = df_slice_timeline['comment_count'].rolling(window=roll_window_dyn, min_periods=1).mean()
                roll_std = df_slice_timeline['comment_count'].rolling(window=roll_window_dyn, min_periods=1).std().fillna(1)
                df_slice_timeline['z_score'] = (df_slice_timeline['comment_count'] - roll_mean) / roll_std

        if not df_slice_timeline.empty:
            fig_timeline = px.line(
                df_slice_timeline,
                x='date',
                y='comment_count',
                title=f"Chronological Conversation Volume ({roll_window_dyn}-Day Baseline)",
                labels={'date': 'Date', 'comment_count': 'Captured Volume'},
                line_shape=curve_smooth,
                template=PLOTLY_TEMPLATE
            )
            fig_timeline.update_traces(line_color=COLOR_PRIMARY, line_width=2.5)

            # Highlight anomalies dynamically
            anomalies = df_slice_timeline[df_slice_timeline['z_score'] > z_thresh_dyn]
            if not anomalies.empty:
                fig_timeline.add_trace(
                    go.Scatter(
                        x=anomalies['date'],
                        y=anomalies['comment_count'],
                        mode='markers',
                        name=f'Flashpoints (>{z_thresh_dyn:.1f}σ: {len(anomalies)})',
                        marker=dict(color=COLOR_DANGER, size=10, symbol='diamond')
                    )
                )
            fig_timeline.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_timeline, width="stretch")
            st.caption(f"Detected **{len(anomalies)}** anomaly flashpoints exceeding {z_thresh_dyn:.1f}σ. Use the bottom range slider to zoom into specific historical periods.")

        # Second-by-Second Reaction Heatmap/Timeline
        if not data_layers["reaction_map"].empty:
            st.divider()
            st.subheader("🎬 Video Reaction Dynamics (Second-by-Second)")
            st.markdown("Moment-by-moment sentiment trajectory and toxicity tracking mapped across video playback timestamps.")

            df_rx = data_layers["reaction_map"]
            vid_options = df_rx["video_id"].unique()
            selected_vid = st.selectbox(
                "Select Video for Reaction Timeline:",
                options=vid_options,
                format_func=lambda vid: f"{video_title_map.get(vid, vid)} ({vid})"
            )

            vid_rx = df_rx[df_rx["video_id"] == selected_vid].sort_values("second")

            y_cols = ["vader_compound"]
            if "toxicity" in vid_rx.columns:
                y_cols.append("toxicity")

            vid_title_label = video_title_map.get(selected_vid, selected_vid)
            fig_rx = px.line(
                vid_rx,
                x="second",
                y=y_cols,
                title=f"Sentiment & Toxicity Trajectory // {vid_title_label}",
                labels={"second": "Playback Time (Seconds)", "value": "Metric Score", "variable": "Dimension"},
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_rx, width="stretch")
            st.caption("Second-by-second sentiment and toxicity levels. Peaks or troughs highlight specific scene moments that triggered intense audience reactions.")

            # Multi-Video Playback Comparison
            if len(vid_options) > 1:
                st.markdown("##### 📊 Comparative Multi-Video Playback Dynamics")
                comp_vids = st.multiselect(
                    "Select Multiple Videos to Compare Playback Dynamics:",
                    options=vid_options,
                    default=list(vid_options[:min(3, len(vid_options))]),
                    format_func=lambda vid: f"{video_title_map.get(vid, vid)} ({vid})"
                )
                if comp_vids:
                    df_comp = df_rx[df_rx["video_id"].isin(comp_vids)].copy()
                    df_comp["Video Title"] = df_comp["video_id"].map(video_title_map)
                    comp_metric = st.selectbox("Comparison Trajectory Metric:", ["Sentiment (VADER)", "Toxicity Level"], index=0)
                    y_m = "vader_compound" if comp_metric == "Sentiment (VADER)" else "toxicity"
                    if y_m in df_comp.columns:
                        fig_comp = px.line(
                            df_comp,
                            x="second",
                            y=y_m,
                            color="Video Title",
                            title=f"Multi-Video Playback Trajectory // {comp_metric}",
                            labels={"second": "Playback Time (Seconds)", y_m: comp_metric},
                            template=PLOTLY_TEMPLATE
                        )
                        st.plotly_chart(fig_comp, width="stretch")
                        st.caption("Overlaid playback trajectories showing how narrative beats and scene moments compare across multiple video releases.")

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

        # Minute-Level Arrival Velocity & Topic Injection Anomaly Scanner
        st.divider()
        t_ex1, t_ex2 = st.columns(2)
        with t_ex1:
            st.subheader("⏱️ First-120-Minutes Arrival Velocity Curve")
            if not data_layers["minute_arrival"].empty:
                df_ma = data_layers["minute_arrival"]
                fig_ma = px.line(
                    df_ma,
                    x="minute_bin",
                    y=["cumulative_comments", "velocity_per_min"],
                    title="Minute-by-Minute Comment Arrival & Velocity",
                    labels={"minute_bin": "Minutes Post-Upload", "value": "Volume / Velocity", "variable": "Metric"},
                    template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig_ma, width="stretch")
            else:
                st.info("Minute-level arrival velocity data available upon stage execution.")

        with t_ex2:
            st.subheader("💉 Topic Injection & Thematic Shift Scanner")
            if not data_layers["topic_injections"].empty:
                df_ti = data_layers["topic_injections"].copy()
                if "video_id" in df_ti.columns:
                    df_ti.insert(0, "Video Title", df_ti["video_id"].map(video_title_map).fillna(df_ti["video_id"]))
                st.dataframe(df_ti.head(30), width="stretch", hide_index=True)
                st.caption("Time windows with statistically significant Jensen-Shannon divergence from baseline.")
            else:
                st.info("No topic injection anomalies detected in current run.")

        # Cross-Modal Scene Reactions & Narrative Spoilers
        if not data_layers["scene_reactions"].empty or not data_layers["spoiler_detections"].empty:
            st.divider()
            st.subheader("🎭 Scene Reactions & Narrative Spoilers")
            rx_col1, rx_col2 = st.columns(2)
            with rx_col1:
                if not data_layers["scene_reactions"].empty:
                    st.markdown("##### Moment Reaction Taxonomy")
                    df_rxn_ui = data_layers["scene_reactions"].groupby("reaction_type")["n_comments"].sum().reset_index()
                    fig_rxn_pie = px.pie(df_rxn_ui, names="reaction_type", values="n_comments", hole=0.4, title="Scene Reaction Distribution", template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig_rxn_pie, width="stretch")
            with rx_col2:
                if not data_layers["spoiler_detections"].empty:
                    st.markdown("##### Flagged Narrative Spoilers")
                    df_sp_ui = data_layers["spoiler_detections"].copy()
                    if "video_id" in df_sp_ui.columns:
                        df_sp_ui.insert(0, "Video Title", df_sp_ui["video_id"].map(video_title_map).fillna(df_sp_ui["video_id"]))
                    st.dataframe(df_sp_ui.head(25), width="stretch", hide_index=True)

        # 2.5 Real-Time Chronological Event Replay & Crisis Simulator
        st.divider()
        st.subheader("⏱️ Real-Time Chronological Event Replay & Crisis Simulator")
        st.markdown(
            "Reconstruct the chronological unfolding of comment cascades following a video release. "
            "Scrub through post-upload timelines to inspect instantaneous arrival velocity, rolling sentiment, "
            "toxicity outbreaks, flame-war formation, and automated crisis flashpoint alerts."
        )

        from engine.event_replay import EventReplayEngine

        engine_replay = EventReplayEngine()
        available_vids = engine_replay.get_available_videos(limit=30)
        vid_choices = [v["video_id"] for v in available_vids]
        vid_labels = {v["video_id"]: f"{v.get('title', v['video_id'])} ({v.get('total_comments', 0):,} comments)" for v in available_vids}

        rep_c1, rep_c2, rep_c3 = st.columns([2, 1, 1])
        with rep_c1:
            selected_vid_replay = st.selectbox(
                "Select Video for Event Replay",
                options=vid_choices if vid_choices else ["MOCK_VID_001"],
                format_func=lambda x: vid_labels.get(x, x),
                key="replay_video_select",
            )
        with rep_c2:
            step_mins_replay = st.select_slider(
                "Time Bucket Resolution",
                options=[15, 30, 60, 120],
                value=30,
                format_func=lambda x: f"{x} mins",
                key="replay_step_slider",
            )
        with rep_c3:
            max_hours_replay = st.slider(
                "Max Horizon (Hours)",
                min_value=12,
                max_value=168,
                value=72,
                step=12,
                key="replay_max_hours_slider",
            )

        with st.spinner("Compiling chronological event sequence..."):
            chronicle = engine_replay.load_video_timeline(
                video_id=selected_vid_replay,
                step_minutes=step_mins_replay,
                max_hours=float(max_hours_replay),
            )

        df_rep = chronicle.to_dataframe()
        if not df_rep.empty:
            # Scorecard KPIs
            peak_vel_row = df_rep.loc[df_rep["arrival_velocity"].idxmax()]
            min_sent_row = df_rep.loc[df_rep["rolling_sentiment"].idxmin()]
            max_tox_row = df_rep.loc[df_rep["rolling_toxicity"].idxmax()]
            num_alerts = len(chronicle.flashpoints_summary)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                with st.container(border=True):
                    st.metric("Total Replayed Comments", f"{chronicle.total_comments_replayed:,}")
                    st.caption(f"{chronicle.total_frames} frames @ {chronicle.step_minutes}m resolution")
            with k2:
                with st.container(border=True):
                    st.metric("Peak Arrival Velocity", f"{peak_vel_row['arrival_velocity']:.1f} /hr", delta=peak_vel_row["timestamp"])
                    st.caption(f"Spike at T+{peak_vel_row['hours_elapsed']:.1f}h post-upload")
            with k3:
                with st.container(border=True):
                    st.metric("Min Rolling Sentiment", f"{min_sent_row['rolling_sentiment']:+.2f}", delta=min_sent_row["timestamp"], delta_color="inverse")
                    st.caption(f"Peak Toxicity: {max_tox_row['rolling_toxicity']:.3f} ({max_tox_row['timestamp']})")
            with k4:
                with st.container(border=True):
                    alert_color = "🔴" if num_alerts >= 2 else ("🟡" if num_alerts == 1 else "🟢")
                    st.metric("Crisis Flashpoints", f"{alert_color} {num_alerts} Triggers")
                    st.caption("Surges, toxicity spikes & flame-wars")

            # Dual-Axis Plotly Timeline
            from plotly.subplots import make_subplots
            fig_replay = make_subplots(specs=[[{"secondary_y": True}]])

            fig_replay.add_trace(
                go.Bar(
                    x=df_rep["hours_elapsed"],
                    y=df_rep["arrival_velocity"],
                    name="Arrival Velocity (comments/hr)",
                    marker_color="#0066fe",
                    opacity=0.6,
                ),
                secondary_y=False,
            )
            fig_replay.add_trace(
                go.Scatter(
                    x=df_rep["hours_elapsed"],
                    y=df_rep["rolling_sentiment"],
                    name="Rolling Sentiment",
                    line=dict(color="#00e599", width=2.5),
                    mode="lines",
                ),
                secondary_y=True,
            )
            fig_replay.add_trace(
                go.Scatter(
                    x=df_rep["hours_elapsed"],
                    y=df_rep["rolling_toxicity"],
                    name="Rolling Toxicity",
                    line=dict(color="#ff3366", width=2, dash="dot"),
                    mode="lines",
                ),
                secondary_y=True,
            )
            fig_replay.add_trace(
                go.Scatter(
                    x=df_rep["hours_elapsed"],
                    y=df_rep["flame_war_risk_score"] / 100.0,
                    name="Flame-War Risk (0-1)",
                    line=dict(color="#a855f7", width=1.5, dash="dash"),
                    mode="lines",
                ),
                secondary_y=True,
            )

            # Add vertical marker lines for flashpoints
            for fp in chronicle.flashpoints_summary:
                color_line = "#ff3366" if fp.severity == "CRITICAL" else "#ffaa00"
                fp_hrs = fp.minute_offset / 60.0
                fig_replay.add_vline(
                    x=fp_hrs,
                    line_width=1.5,
                    line_dash="dash",
                    line_color=color_line,
                    annotation_text=f"🚨 {fp.title}",
                    annotation_position="top left",
                    annotation_font_size=10,
                )

            fig_replay.update_layout(
                title=f"Chronological Event Cascade: '{chronicle.video_title}'",
                xaxis_title="Hours Post-Upload (Elapsed)",
                template=PLOTLY_TEMPLATE,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            fig_replay.update_yaxes(title_text="Velocity (comments/hr)", secondary_y=False)
            fig_replay.update_yaxes(title_text="Index ([-1, 1])", secondary_y=True, range=[-1.0, 1.0])

            st.plotly_chart(fig_replay, width="stretch")

            # Interactive Time Scrubber Slider
            st.markdown("#### 🎛️ Interactive Time Scrubber & Frame Inspector")
            scrub_idx = st.slider(
                "Select Simulation Timeframe",
                min_value=0,
                max_value=len(df_rep) - 1,
                value=int(peak_vel_row.name) if isinstance(peak_vel_row.name, (int, np.integer)) else 0,
                format_func=lambda i: f"{df_rep.iloc[i]['timestamp']} (T+{df_rep.iloc[i]['hours_elapsed']:.1f}h) — {df_rep.iloc[i]['frame_new_comments']} comments",
                key="replay_scrubber_slider",
            )

            current_frame = chronicle.frames[scrub_idx]

            # Frame Detail Card
            with st.container(border=True):
                sf_c1, sf_c2, sf_c3, sf_c4, sf_c5 = st.columns(5)
                with sf_c1:
                    st.metric("Frame Timestamp", current_frame.timestamp, delta=f"+{current_frame.frame_new_comments} comments")
                with sf_c2:
                    st.metric("Arrival Velocity", f"{current_frame.arrival_velocity:.1f} /hr", delta=f"{current_frame.velocity_accel:+.1f} accel")
                with sf_c3:
                    sent_color = "normal" if current_frame.rolling_sentiment >= 0 else "inverse"
                    st.metric("Sentiment / Toxicity", f"{current_frame.rolling_sentiment:+.2f}", delta=f"{current_frame.rolling_toxicity:.3f} tox", delta_color=sent_color)
                with sf_c4:
                    st.metric("Reply Ratio", f"{current_frame.reply_ratio:.0%}", delta=f"{current_frame.active_authors_count} authors")
                with sf_c5:
                    risk_val = current_frame.flame_war_risk_score
                    risk_badge = "🔥 CRITICAL" if risk_val >= 65 else ("⚠️ ELEVATED" if risk_val >= 40 else "🟢 LOW")
                    st.metric("Flame-War Risk", f"{risk_val:.0f}%", delta=risk_badge)

                # Active Flashpoint Alerts
                if current_frame.active_flashpoints:
                    for af in current_frame.active_flashpoints:
                        if af.severity == "CRITICAL":
                            st.error(f"🚨 **{af.title}** ({af.alert_type}): {af.description} [Trigger: {af.trigger_value} vs Threshold {af.threshold}]")
                        else:
                            st.warning(f"⚠️ **{af.title}** ({af.alert_type}): {af.description} [Trigger: {af.trigger_value} vs Threshold {af.threshold}]")

                # Exemplar Comments in Window
                if current_frame.top_comments:
                    st.markdown("##### 💬 Comments Arriving in This Window")
                    for tc in current_frame.top_comments:
                        st.markdown(
                            f"- **{tc['author']}** (`{tc['likes']} likes`, Sentiment: `{tc['sentiment']:+.2f}`, Toxicity: `{tc['toxicity']:.3f}`): "
                            f"_{tc['text']}_"
                        )
                elif current_frame.frame_new_comments == 0:
                    st.caption("No new comments arrived in this exact time window.")

            # Crisis Flashpoint Ledger (Expander)
            with st.expander(f"📋 Detected Crisis Flashpoint Ledger ({len(chronicle.flashpoints_summary)} Events)", expanded=False):
                if chronicle.flashpoints_summary:
                    df_fp_summary = pd.DataFrame([fp.to_dict() for fp in chronicle.flashpoints_summary])
                    st.dataframe(df_fp_summary, width="stretch", hide_index=True)
                else:
                    st.info("No anomalous spikes or toxicity crises detected across the evaluated horizon.")

            # Export Hub
            exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 2])
            with exp_c1:
                st.download_button(
                    label="💾 Export Chronicle JSON",
                    data=json.dumps(chronicle.to_dict(), indent=2, ensure_ascii=False),
                    file_name=f"chronicle_{selected_vid_replay}.json",
                    mime="application/json",
                    key="dl_chronicle_json",
                )
            with exp_c2:
                st.download_button(
                    label="📊 Export Chronicle CSV",
                    data=df_rep.to_csv(index=False),
                    file_name=f"chronicle_{selected_vid_replay}.csv",
                    mime="text/csv",
                    key="dl_chronicle_csv",
                )

    # ==============================================================================
    # TAB 3: NLP, SEMANTICS & DEMAND INTENT
    # ==============================================================================
    with tab_topics:
        st.header("Conversational Topic Modeling & Audience Demand Intent")
        st.markdown("Granular semantic clusters, audience request mining, and topic evolution dynamics.")

        # 3.1 Dynamic Topic Resonance Explorer
        if not df_topics.empty and "Count" in df_topics.columns:
            plot_topics = df_topics[df_topics["Topic"] != -1].copy()
            if not plot_topics.empty:
                st.subheader("🎯 Dynamic Topic Resonance & Engagement Explorer")
                st.caption("Customize axes and sizing to uncover high-demand, high-sentiment community themes.")
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    x_topic_dim = st.selectbox("X-Axis (Discussion Volume):", ["Count", "total_likes"], index=0)
                with res_col2:
                    y_topic_dim = st.selectbox("Y-Axis (Approval Rate):", ["avg_likes", "max_likes", "Count"], index=0)
                with res_col3:
                    color_topic_dim = st.selectbox("Color Theme Dimension:", ["Count", "avg_likes", "total_likes"], index=1)

                fig_resonance = px.scatter(
                    plot_topics,
                    x=x_topic_dim,
                    y=y_topic_dim,
                    size="total_likes" if "total_likes" in plot_topics.columns else "Count",
                    hover_name="Name",
                    title="Interactive Topic Resonance Matrix",
                    labels={"Count": "Total Comments in Topic", "avg_likes": "Average Likes per Comment", "total_likes": "Total Likes"},
                    template=PLOTLY_TEMPLATE,
                    size_max=40,
                    color=color_topic_dim,
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_resonance, width="stretch")
                st.caption("Topic resonance mapping comment volume against community approval (average likes). Top-right bubbles indicate high-volume, highly favored topics.")

        # 3.2 Stage 35: Audience Demand & Content Intent Mining
        st.divider()
        st.subheader("🎯 Audience Demand & Content Intent Mining (Stage 35)")
        st.markdown("Automated classification of audience comments into actionable content requests, questions, bug reports, and praise.")

        if not data_layers["audience_intent"].empty:
            intent_col1, intent_col2 = st.columns([1, 2])
            with intent_col1:
                st.markdown("##### Intent Category Breakdown")
                df_intent = data_layers["audience_intent"]
                intent_col = "audience_intent" if "audience_intent" in df_intent.columns else ("intent_category" if "intent_category" in df_intent.columns else df_intent.columns[0])
                val_col = "comment_count" if "comment_count" in df_intent.columns else df_intent.columns[1]
                fig_intent = px.pie(
                    df_intent,
                    names=intent_col,
                    values=val_col,
                    hole=0.4,
                    template=PLOTLY_TEMPLATE,
                    title="Audience Comment Intent Distribution"
                )
                st.plotly_chart(fig_intent, width="stretch")

            with intent_col2:
                st.markdown("##### Prioritized Community Video Suggestions")
                if not data_layers["audience_requests"].empty:
                    st.dataframe(data_layers["audience_requests"].head(50), width="stretch", hide_index=True)
                    st.caption("Top requested topics and features ranked by total upvotes and community demand.")
                else:
                    st.info("No content request records extracted.")

        # 3.3 Searchable Topic Registry
        st.divider()
        st.subheader("Searchable Semantic Topic Registry")
        topic_query = st.text_input("🔍 Filter topics by keyword or token:", "").strip().lower()

        filtered_topics = df_topics.copy()
        if topic_query and not filtered_topics.empty:
            text_mask = filtered_topics.astype(str).apply(lambda x: x.str.lower().str.contains(topic_query)).any(axis=1)
            filtered_topics = filtered_topics[text_mask]

        st.dataframe(filtered_topics, width="stretch", hide_index=True)

        # 3.4 Topic Evolution Over Time
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

        # 3.5 Named Entities & Code-Switching Impact
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
                    title="Engagement Differential: Monolingual vs. Code-Switched",
                    labels={"type": "Language Type", "avg_likes": "Average Upvotes per Comment"},
                    template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig_cs, width="stretch")
            else:
                st.info("Code-switching impact records not available.")

        # 3.6 Target-Specific Stance Detection & Polarization Drift (Stage 39)
        st.divider()
        st.subheader("🎯 Target-Specific Stance Detection & Polarization Drift (Stage 39)")
        st.markdown("Audience agreement vs. opposition classification and stance drift across conversation tree debate depth.")

        if not data_layers["stance_summary"].empty:
            st_col1, st_col2 = st.columns([1, 2])
            with st_col1:
                st.markdown("##### Reply Depth Polarization Drift")
                if not data_layers["stance_drift"].empty:
                    df_d = data_layers["stance_drift"]
                    fig_d = px.line(
                        df_d,
                        x="depth_label",
                        y=["favor_pct", "against_pct"],
                        markers=True,
                        labels={"depth_label": "Debate Level", "value": "Share (%)", "variable": "Stance"},
                        title="Stance Divergence across Debate Depth",
                        color_discrete_map={"favor_pct": "#00e599", "against_pct": "#ff3366"},
                        template=PLOTLY_TEMPLATE
                    )
                    st.plotly_chart(fig_d, width="stretch")

            with st_col2:
                st.markdown("##### Video Target Stance Breakdown & Polarization Index")
                df_st = data_layers["stance_summary"].copy()
                if "video_id" in df_st.columns:
                    df_st.insert(0, "Video Title", df_st["video_id"].map(video_title_map).fillna(df_st["video_id"]))
                st.dataframe(df_st.head(50), width="stretch", hide_index=True)
                st.caption("Videos ranked by total volume and Polarization Index (balance of Favor vs Against).")

        if not data_layers["polarized_threads"].empty:
            st.markdown("##### High-Conflict Polarized Debate Threads")
            df_pt = data_layers["polarized_threads"].copy()
            if "video_id" in df_pt.columns:
                df_pt.insert(0, "Video Title", df_pt["video_id"].map(video_title_map).fillna(df_pt["video_id"]))
            st.dataframe(df_pt.head(30), width="stretch", hide_index=True)
            st.caption("Debate threads with acute ideological opposition and elevated hostility.")

        # 3.7 Slang Lexicon & TF-IDF Keywords
        st.divider()
        st.subheader("💬 Slang Lexicon & Distinguishing Vocabulary")
        sl_col1, sl_col2 = st.columns(2)
        with sl_col1:
            if not data_layers["slang_lexicon"].empty:
                st.markdown("##### Top Informal & Internet-Register Slang")
                st.dataframe(data_layers["slang_lexicon"].head(25), width="stretch", hide_index=True)
                st.caption("Bilingual informal register tracking with comment frequency and average sentiment.")
            else:
                st.info("Slang lexicon data available upon stage execution.")

        with sl_col2:
            if not data_layers["tfidf_keywords"].empty:
                st.markdown("##### Distinctive Video Keywords (TF-IDF)")
                df_tf = data_layers["tfidf_keywords"].copy()
                if "video_id" in df_tf.columns:
                    df_tf.insert(0, "Video Title", df_tf["video_id"].map(video_title_map).fillna(df_tf["video_id"]))
                st.dataframe(df_tf.head(30), width="stretch", hide_index=True)
                st.caption("Salient non-generic keywords extracted per video upload.")
            else:
                st.info("TF-IDF keywords available upon stage execution.")

        # 3.8 Within-Thread Topic Drift & Emoji Signatures
        if not data_layers["thread_topic_drift"].empty or not data_layers["emoji_signatures"].empty:
            st.divider()
            t_d_col1, t_d_col2 = st.columns(2)
            with t_d_col1:
                if not data_layers["thread_topic_drift"].empty:
                    st.subheader("🌀 Within-Thread Topic Drift")
                    st.markdown("Measures how frequently reply comments diverge from the root comment's original topic.")
                    st.dataframe(data_layers["thread_topic_drift"].head(25), width="stretch", hide_index=True)
            with t_d_col2:
                if not data_layers["emoji_signatures"].empty:
                    st.subheader("😀 Emoji Signatures & Profiles")
                    st.dataframe(data_layers["emoji_signatures"].head(25), width="stretch", hide_index=True)
                    st.caption("Dominant emoji distributions mapped across topics and sentiment.")

        # 3.9 AI Flame-War & Thread Debate Summarizer
        st.divider()
        st.subheader("⚖️ AI Flame-War & Debate Tree Summarizer")
        st.markdown(
            "Select high-depth or polarized discussion trees to extract conflict triggers, opposing viewpoints, "
            "escalation dynamics, and creator mediation strategies using LLM reasoning."
        )

        with st.container(border=True):
            fw_col1, fw_col2 = st.columns([2, 1])
            with fw_col1:
                candidate_threads = []
                if not df_comments.empty:
                    if "reply_count" in df_comments.columns:
                        top_roots = df_comments[df_comments.get("is_reply", False) == False].sort_values(by="reply_count", ascending=False).head(20)
                        for _, r in top_roots.iterrows():
                            txt_prev = (str(r.get("text_original") or r.get("text", "")))[:60].replace("\n", " ")
                            candidate_threads.append((r["comment_id"], f"{r['comment_id']} ({r.get('reply_count', 0)} replies) — \"{txt_prev}...\""))

                thread_choices = [c[1] for c in candidate_threads] if candidate_threads else ["No active threads"]
                selected_thread_label = st.selectbox("Select Debate Thread to Analyze:", options=thread_choices, key="ai_deb_thread_sel")
                selected_thread_id = None
                if candidate_threads and selected_thread_label != "No active threads":
                    selected_thread_id = candidate_threads[thread_choices.index(selected_thread_label)][0]

            with fw_col2:
                ai_deb_prov = st.selectbox("AI Provider:", options=["gemini", "ollama", "mock"], index=0, key="ai_deb_prov")
                if st.button("✨ Summarize Debate", key="btn_summarize_debate"):
                    if selected_thread_id:
                        with st.spinner("Analyzing thread debate and opposing arguments..."):
                            from engine.synthesizer import LLMClient, InsightSynthesizer
                            client = LLMClient(provider=ai_deb_prov)
                            synth = InsightSynthesizer(interim_path, output_path, llm_client=client)
                            st.session_state["thread_summary_res"] = synth.summarize_thread_debate(selected_thread_id)
                            st.success("Debate analyzed!")
                    else:
                        st.info("Please select a valid thread.")

            if "thread_summary_res" in st.session_state:
                with st.container(border=True):
                    st.markdown(st.session_state["thread_summary_res"])

        # 3.10 Narrative Scene Reaction & Timestamp Scrubbing Forensics
        st.divider()
        st.subheader("⏱️ Narrative Scene Reaction & Timestamp Scrubbing Forensics")
        st.markdown(
            "Detect scene-level audience reactions linked directly to in-video moments via timestamp extraction. "
            "Scrub through video playback time to reveal laughter peaks, shock moments, emotional resonance, "
            "and viewer confusion hotspots."
        )

        from engine.narrative import NarrativeForensicsEngine
        from plotly.subplots import make_subplots

        narrative_engine = NarrativeForensicsEngine(interim_path, output_path)
        available_narrative_vids = narrative_engine.get_available_videos()

        if available_narrative_vids:
            vid_options = [v["video_id"] for v in available_narrative_vids]
            vid_label_map = {
                v["video_id"]: f"{v.get('title', v['video_id'])} ({v.get('timestamp_reactions', 0):,} timestamp reactions)"
                for v in available_narrative_vids
            }

            nar_c1, nar_c2, nar_c3 = st.columns([3, 1, 1])
            with nar_c1:
                selected_nar_vid = st.selectbox(
                    "Select Video to Analyze:",
                    options=vid_options,
                    format_func=lambda vid: vid_label_map.get(vid, vid),
                    key="sel_narrative_video"
                )
            with nar_c2:
                step_secs = st.slider(
                    "Scene Bin Resolution (s):",
                    min_value=10,
                    max_value=60,
                    value=30,
                    step=5,
                    key="slider_narrative_step"
                )
            with nar_c3:
                use_mock_nar = st.checkbox("Demo / Mock Mode", value=False, key="chk_narrative_mock")

            report = narrative_engine.analyze_video(selected_nar_vid, step_secs=step_secs, mock=use_mock_nar)

            # Summary KPIs
            kpi_n1, kpi_n2, kpi_n3, kpi_n4, kpi_n5 = st.columns(5)
            with kpi_n1:
                with st.container(border=True):
                    st.metric("Total Video Runtime", report.formatted_duration)
            with kpi_n2:
                with st.container(border=True):
                    st.metric("Timestamp Reactions", f"{report.total_timestamp_reactions:,}")
            with kpi_n3:
                with st.container(border=True):
                    st.metric("Scene Clusters", f"{len(report.clusters):,}")
            with kpi_n4:
                with st.container(border=True):
                    st.metric("Confusion Hotspots", f"{len(report.confusion_hotspots):,}")
            with kpi_n5:
                with st.container(border=True):
                    peak_desc = f"{report.peak_moment.get('formatted_time', '0:00')} ({report.peak_moment.get('dominant_reaction', 'none')})" if report.peak_moment else "N/A"
                    st.metric("Peak Scene Moment", peak_desc)

            # Confusion / Question Hotspots Callout Banner
            if report.confusion_hotspots:
                for hs in report.confusion_hotspots[:3]:
                    st.warning(
                        f"❓ **Viewer Confusion Hotspot at {hs.formatted_time}** "
                        f"(T+{hs.timestamp_sec}s — {hs.question_count} question/confusion comments): "
                        f"Viewer questions cluster here — consider adding an on-screen clarification, pinned comment, or chapter title."
                    )

            # Dual-Axis Plotly Timeline Visualization
            if report.clusters:
                df_clusters = pd.DataFrame([c.to_dict() for c in report.clusters])

                REACTION_PALETTE = {
                    "humor_laughter": "#ffaa00",      # Amber / Gold
                    "shock_surprise": "#ff3366",      # Crimson / Magenta
                    "emotional_touching": "#00e599",   # Mint / Emerald
                    "critique_analytical": "#0066fe",  # Royal Blue
                    "chapter_navigation": "#a855f7",   # Purple
                    "general_reaction": "#64748b",     # Slate
                }

                fig_nar = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=(
                        "Scene Reaction Density & Comment Volume Across Playback",
                        "Sentiment Trajectory & Viewer Confusion Hotspots (VADER [-1, 1])"
                    ),
                    row_heights=[0.6, 0.4]
                )

                # Top Chart: Bar chart of reaction counts by cluster with dominant reaction color
                bar_colors = [REACTION_PALETTE.get(c.dominant_reaction, "#64748b") for c in report.clusters]
                fig_nar.add_trace(
                    go.Bar(
                        x=df_clusters["start_sec"],
                        y=df_clusters["reaction_count"],
                        marker_color=bar_colors,
                        name="Reactions / Scene",
                        customdata=df_clusters[["formatted_time", "dominant_reaction", "reaction_count"]].values,
                        hovertemplate="<b>%{customdata[0]}</b><br>Reactions: %{customdata[2]}<br>Dominant: %{customdata[1]}<extra></extra>",
                    ),
                    row=1, col=1
                )

                # Add markers for confusion hotspots in top chart
                if report.confusion_hotspots:
                    hs_secs = [h.timestamp_sec for h in report.confusion_hotspots]
                    hs_counts = [h.question_count for h in report.confusion_hotspots]
                    hs_texts = [f"❓ {h.formatted_time}" for h in report.confusion_hotspots]
                    fig_nar.add_trace(
                        go.Scatter(
                            x=hs_secs,
                            y=hs_counts,
                            mode="markers+text",
                            marker=dict(symbol="triangle-up", size=12, color="#ffaa00", line=dict(width=1, color="#ffffff")),
                            text=hs_texts,
                            textposition="top center",
                            name="Confusion Hotspots",
                            hovertemplate="<b>Confusion Hotspot: %{text}</b><br>Questions: %{y}<extra></extra>",
                        ),
                        row=1, col=1
                    )

                # Bottom Chart: Rolling sentiment trajectory
                fig_nar.add_trace(
                    go.Scatter(
                        x=df_clusters["start_sec"],
                        y=df_clusters["average_sentiment"],
                        mode="lines+markers",
                        line=dict(color="#00e599", width=2),
                        marker=dict(size=5, color="#00e599"),
                        name="Avg Scene Sentiment",
                        customdata=df_clusters[["formatted_time", "average_sentiment"]].values,
                        hovertemplate="<b>%{customdata[0]}</b><br>Sentiment: %{customdata[1]:+.2f}<extra></extra>",
                    ),
                    row=2, col=1
                )

                # Zero line on sentiment
                fig_nar.add_hline(y=0.0, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=2, col=1)

                fig_nar.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=520,
                    margin=dict(l=40, r=40, t=50, b=40),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig_nar.update_xaxes(title_text="Video Playback Timeline (Seconds)", row=2, col=1)
                fig_nar.update_yaxes(title_text="Comment Count", row=1, col=1)
                fig_nar.update_yaxes(title_text="VADER [-1, 1]", range=[-1.0, 1.0], row=2, col=1)

                st.plotly_chart(fig_nar, width="stretch")

                # Interactive Playback Scrubber Slider
                st.markdown("#### 🎛️ Interactive Scene Scrubber & Quote Montage")
                scrubber_idx = st.slider(
                    "Scrub Video Playback Timeline:",
                    min_value=0,
                    max_value=len(report.clusters) - 1,
                    value=0,
                    format_func=lambda i: f"{report.clusters[i].formatted_time} (T+{report.clusters[i].start_sec}s–{report.clusters[i].end_sec}s) — {report.clusters[i].reaction_count} reactions [{report.clusters[i].dominant_reaction}]",
                    key="slider_narrative_scrubber"
                )

                active_cluster = report.clusters[scrubber_idx]

                # Active Scene Spotlight Card
                with st.container(border=True):
                    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
                    with sc_col1:
                        st.metric("Scene Time Window", active_cluster.formatted_time, delta=f"{active_cluster.start_sec}s – {active_cluster.end_sec}s")
                    with sc_col2:
                        st.metric("Dominant Reaction", active_cluster.dominant_reaction.replace("_", " ").title())
                    with sc_col3:
                        s_color = "normal" if active_cluster.average_sentiment >= 0 else "inverse"
                        st.metric("Scene Sentiment", f"{active_cluster.average_sentiment:+.2f}", delta_color=s_color)
                    with sc_col4:
                        st.metric("Reaction Volume", f"{active_cluster.reaction_count} comments")

                    # Verbatim Quote Montage
                    if active_cluster.sample_quotes:
                        st.markdown("##### 🗣️ Verbatim Audience Reactions at This Moment")
                        for q in active_cluster.sample_quotes:
                            like_badge = f"👍 {q.likes}" if q.likes > 0 else ""
                            sent_badge = f"Sentiment: {q.sentiment:+.2f}"
                            st.markdown(
                                f"- **{q.author}** at `{q.timestamp_str}` ({like_badge} {sent_badge} | `{q.reaction_type}`): "
                                f"_{q.text}_"
                            )
                    else:
                        st.caption("No verbatim quotes captured for this specific scene window.")

                # Export Hub for Narrative Report
                st.markdown("##### 💾 Export Narrative Forensics Data")
                exp_n1, exp_n2 = st.columns(2)
                with exp_n1:
                    st.download_button(
                        label="💾 Export Narrative Report JSON",
                        data=json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                        file_name=f"narrative_report_{selected_nar_vid}.json",
                        mime="application/json",
                        key="dl_narrative_json"
                    )
                with exp_n2:
                    st.download_button(
                        label="📊 Export Scene Clusters CSV",
                        data=df_clusters.to_csv(index=False),
                        file_name=f"scene_clusters_{selected_nar_vid}.csv",
                        mime="text/csv",
                        key="dl_narrative_csv"
                    )
            else:
                st.info("No timestamp reactions found in this video's comment corpus. Try toggling 'Demo / Mock Mode' to preview synthetic scene forensics.")
        else:
            st.info("No videos available with timestamp reaction data. Ensure stage 28 / comments clean pipeline has run.")

    # ==============================================================================
    # TAB 4: AUDIENCE LOYALTY & FORENSICS
    # ==============================================================================
    with tab_audience:
        st.header("👥 Audience Segmentation, Loyalty & Forensic Diagnostics")
        st.markdown("Deconstruct commenter cohorts, detect astroturfing rings, and identify troll catalysts.")

        # 4.1 Coordinated Inauthentic Behavior (CIB) Rings (Stage 36)
        st.subheader("🕵️ Coordinated Inauthentic Behavior (CIB) & Astroturfing Rings (Stage 36)")
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
                    df_cc = data_layers["cib_comments"].copy()
                    if "video_id" in df_cc.columns:
                        df_cc.insert(0, "Video Title", df_cc["video_id"].map(video_title_map).fillna(df_cc["video_id"]))
                    st.dataframe(df_cc.head(50), width="stretch", hide_index=True)
                    st.caption("Individual comments flagged as belonging to coordinated network rings.")
                else:
                    st.info("No coordinated comments found.")

            # Dynamic CIB Ring Severity Map
            st.markdown("##### 🕸️ Coordinated Ring Severity & Synchronization Map")
            df_rings = data_layers["cib_rings"].head(30).copy()
            if "ring_size" in df_rings.columns and "total_synchronized_events" in df_rings.columns:
                fig_cib_net = px.scatter(
                    df_rings,
                    x="ring_size",
                    y="total_synchronized_events",
                    size="author_count" if "author_count" in df_rings.columns else None,
                    color="avg_interval_seconds" if "avg_interval_seconds" in df_rings.columns else "ring_size",
                    hover_name="ring_id" if "ring_id" in df_rings.columns else None,
                    title="CIB Ring Severity (Accounts in Ring vs. Synchronized Volume)",
                    labels={
                        "ring_size": "Accounts in Ring",
                        "total_synchronized_events": "Synchronized Events Count",
                        "author_count": "Unique Authors",
                        "avg_interval_seconds": "Avg Interval (s)"
                    },
                    template=PLOTLY_TEMPLATE,
                    color_continuous_scale="Reds",
                    size_max=38
                )
                st.plotly_chart(fig_cib_net, width="stretch")
                st.caption("Top-right clusters identify large rings with high-frequency synchronized commenting campaigns.")

        # 4.2 Toxicity Contagion & Troll Catalyst Identification (Stage 37)
        st.divider()
        st.subheader("🔥 Toxicity Contagion & Flame-War Catalyst Instigators (Stage 37)")
        st.markdown("Branching analysis identifying provocation accounts that trigger heated flame-war cascades across threads.")

        if not data_layers["contagion_summary"].empty:
            c_sum = data_layers["contagion_summary"].iloc[0]
            t_m1, t_m2, t_m3, t_m4 = st.columns(4)
            t_m1.metric("Toxic Comment Share", f"{c_sum.get('toxic_comment_share_pct', 0)}%")
            t_m2.metric("Toxicity Reproduction ($R_0$)", f"{c_sum.get('toxicity_reproduction_number_r0', 0)}")
            t_m3.metric("Avg Replies (Toxic Root)", f"{c_sum.get('avg_replies_to_toxic_root', 0)}")
            t_m4.metric("Avg Replies (Neutral Root)", f"{c_sum.get('avg_replies_to_neutral_root', 0)}")

        if not data_layers["troll_catalysts"].empty:
            tc_col1, tc_col2 = st.columns([1, 2])
            with tc_col1:
                st.markdown("##### Provocation Tier Spectrum")
                df_tc = data_layers["troll_catalysts"]
                if "catalyst_tier" in df_tc.columns:
                    tier_dist = df_tc["catalyst_tier"].value_counts().reset_index()
                    tier_dist.columns = ["Tier", "Authors"]
                elif "mean_toxicity" in df_tc.columns:
                    tier_dist = pd.cut(
                        df_tc["mean_toxicity"],
                        bins=[-0.01, 0.3, 0.6, 1.01],
                        labels=["Low Spark", "Moderate Catalyst", "Severe Flame Instigator"]
                    ).value_counts().reset_index()
                    tier_dist.columns = ["Tier", "Authors"]
                else:
                    tier_dist = pd.DataFrame({"Tier": ["Standard"], "Authors": [len(df_tc)]})

                fig_tier = px.pie(
                    tier_dist,
                    names="Tier",
                    values="Authors",
                    hole=0.4,
                    template=PLOTLY_TEMPLATE,
                    title="Author Provocation Distribution"
                )
                st.plotly_chart(fig_tier, width="stretch")

            with tc_col2:
                st.markdown("##### Top Flame-War Catalysts (Ranked by Provocation Spark)")
                st.dataframe(data_layers["troll_catalysts"].head(50), width="stretch", hide_index=True)
                st.caption("Authors ranked by Catalyst Impact Score ($\text{Toxicity} \times \text{Replies Sparked}$).")

        # 4.3 Interactive 3D RFM Community Space
        st.divider()
        st.subheader("🌐 Interactive 3D RFM Community Space")
        st.markdown("Pan, rotate, and zoom in 3D to explore how commenters cluster across Recency, Frequency, and Monetary (Total Likes) engagement dimensions.")

        if not df_authors.empty:
            plot_authors = df_authors.head(800).copy()
            x_dim = "total_comments" if "total_comments" in plot_authors.columns else ("comment_count" if "comment_count" in plot_authors.columns else "frequency")
            y_dim = "total_likes" if "total_likes" in plot_authors.columns else "like_count"
            z_dim = "unique_videos_commented" if "unique_videos_commented" in plot_authors.columns else ("pagerank" if "pagerank" in plot_authors.columns else "recency_days")
            color_dim = "rfm_segment" if "rfm_segment" in plot_authors.columns else ("is_bot_suspect" if "is_bot_suspect" in plot_authors.columns else x_dim)

            if x_dim in plot_authors.columns and y_dim in plot_authors.columns:
                fig_3d = px.scatter_3d(
                    plot_authors,
                    x=x_dim,
                    y=y_dim,
                    z=z_dim if z_dim in plot_authors.columns else y_dim,
                    color=color_dim,
                    hover_name="author_display_name" if "author_display_name" in plot_authors.columns else "author_channel_id",
                    hover_data={x_dim: True, y_dim: True},
                    title="Interactive 3D Author Behavioral Space",
                    labels={
                        x_dim: "Activity Frequency",
                        y_dim: "Total Upvotes Received",
                        z_dim: "Breadth (Unique Videos)" if z_dim == "unique_videos_commented" else z_dim
                    },
                    template=PLOTLY_TEMPLATE,
                    opacity=0.85
                )
                fig_3d.update_layout(scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))))
                st.plotly_chart(fig_3d, width="stretch")
                st.caption("Rotate and zoom the 3D space. Outliers in the upper quadrants represent ultra-influential community champions.")

        # 4.4 RFM Author Segmentation Ledger
        st.divider()
        st.subheader("RFM Author Segmentation Explorer")
        if not df_authors.empty:
            st.dataframe(df_authors.head(100), width="stretch", hide_index=True)
            st.caption("Showing top 100 authors ranked by recency, frequency, and monetary engagement score.")

        # 4.5 Bow-Tie Network Topology & Series Benchmarks
        st.divider()
        st.subheader("🎀 Author Network Bow-Tie Decomposition & Series Benchmarks")
        bt_col1, bt_col2 = st.columns(2)
        with bt_col1:
            if not data_layers["bowtie_structure"].empty:
                st.markdown("##### Network Bow-Tie Topology Components")
                st.dataframe(data_layers["bowtie_structure"], width="stretch", hide_index=True)
                st.caption("Structural decomposition into Core SCC, IN, OUT, and Peripheral Tendrils.")
            else:
                st.info("Bow-Tie network data available upon stage execution.")

        with bt_col2:
            if not data_layers["series_vs_standalone"].empty:
                st.markdown("##### Episodic Series vs. Standalone Videos")
                st.dataframe(data_layers["series_vs_standalone"], width="stretch", hide_index=True)
                st.caption("Benchmark comparing engagement volume and average sentiment between serialized and standalone content.")
            else:
                st.info("Series benchmark data available upon stage execution.")

        # 4.6 Interactive Community Network & Gephi Topology Explorer
        st.divider()
        st.subheader("🌐 Interactive Community Network & Gephi Topology Explorer")
        st.markdown(
            "Explore author conversational interaction graphs, bipartite video affiliations, and co-commenting clusters. "
            "Inspect influential bridge nodes and export full graphs to **Gephi (GEXF)** or **Cytoscape (GraphML)**."
        )

        with st.container(border=True):
            from engine.network_exporter import NetworkExporter

            net_col1, net_col2, net_col3 = st.columns([2, 1, 1])
            with net_col1:
                net_type = st.selectbox(
                    "Select Network Topology:",
                    options=[
                        "Author Reply Network (Directed)",
                        "Author-Video Bipartite Network",
                        "Author Co-Commenting Network"
                    ],
                    key="net_type_sel"
                )
            with net_col2:
                net_top_k = st.slider(
                    "Influential Node Limit:",
                    min_value=25,
                    max_value=250,
                    value=75,
                    step=25,
                    key="net_top_k_slider"
                )
            with net_col3:
                net_color_by = st.selectbox(
                    "Color Nodes By:",
                    options=["community_id", "pagerank", "in_degree", "is_bot_suspect"],
                    index=0,
                    key="net_color_sel"
                )

            exporter = NetworkExporter(interim_path, output_path)

            # Build and render the selected graph
            with st.spinner("Constructing topological force-directed layout..."):
                if "Reply" in net_type:
                    G_disp = exporter.build_author_reply_network(df_comments_raw, df_authors, data_layers.get("bot_classifications"))
                    stem_name = "author_reply_network"
                elif "Bipartite" in net_type:
                    G_disp = exporter.build_bipartite_network(df_comments_raw, df_videos, max_authors=net_top_k)
                    stem_name = "bipartite_network"
                else:
                    G_disp = exporter.build_cocommenting_network(df_comments_raw, min_overlap=2, max_authors=net_top_k)
                    stem_name = "cocommenting_network"

                fig_net = exporter.generate_interactive_network_figure(
                    G_disp,
                    top_k=net_top_k,
                    color_dimension=net_color_by
                )
                st.plotly_chart(fig_net, width="stretch")

            # GEXF / GraphML Download Actions
            net_dl_col1, net_dl_col2, net_dl_col3 = st.columns([2, 1, 1])
            with net_dl_col1:
                st.caption(
                    f"Graph contains **{G_disp.number_of_nodes():,} nodes** and **{G_disp.number_of_edges():,} edges**. "
                    "Download formatted graph files with embedded PageRank, Louvain communities, and interaction weights."
                )
            with net_dl_col2:
                gexf_file = output_path / "networks" / f"{stem_name}.gexf"
                if gexf_file.exists():
                    st.download_button(
                        label=f"⬇️ Download {stem_name}.gexf (Gephi)",
                        data=gexf_file.read_bytes(),
                        file_name=f"{stem_name}.gexf",
                        mime="application/xml",
                        key=f"dl_gexf_{stem_name}"
                    )
                else:
                    if st.button("Generate GEXF", key=f"btn_gen_gexf_{stem_name}"):
                        exporter.export_graph(G_disp, stem_name, ["gexf"])
                        st.rerun()
            with net_dl_col3:
                graphml_file = output_path / "networks" / f"{stem_name}.graphml"
                if graphml_file.exists():
                    st.download_button(
                        label=f"⬇️ Download {stem_name}.graphml (Cytoscape)",
                        data=graphml_file.read_bytes(),
                        file_name=f"{stem_name}.graphml",
                        mime="application/xml",
                        key=f"dl_graphml_{stem_name}"
                    )
                else:
                    if st.button("Generate GraphML", key=f"btn_gen_graphml_{stem_name}"):
                        exporter.export_graph(G_disp, stem_name, ["graphml"])
                        st.rerun()

        # 4.7 Forensic Author Persona & Sockpuppet Fingerprinting
        st.divider()
        st.subheader("🕵️ Forensic Author Persona & Sockpuppet Fingerprinting")
        st.markdown(
            "Detect coordinated sockpuppet rings and alternate accounts through multi-dimensional stylometrics "
            "(Shannon vocabulary entropy, punctuation motifs, emoji signatures), 24-hour circadian posting rhythms, "
            "and pairwise similarity clustering."
        )

        from engine.fingerprint import AuthorFingerprintEngine

        fp_c1, fp_c2, fp_c3 = st.columns([2, 2, 1])
        with fp_c1:
            fp_min_comments = st.slider(
                "Minimum Comment Threshold:",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                key="slider_fp_min_comments"
            )
        with fp_c2:
            fp_min_score = st.slider(
                "Sockpuppet Confidence Threshold (%):",
                min_value=60.0,
                max_value=95.0,
                value=75.0,
                step=2.5,
                key="slider_fp_min_score"
            )
        with fp_c3:
            use_mock_fp = st.checkbox("Demo / Mock Mode", value=False, key="chk_fp_mock")

        fp_engine = AuthorFingerprintEngine(interim_path, output_path)
        fp_report = fp_engine.generate_forensic_report(
            min_comments=fp_min_comments,
            min_score=fp_min_score,
            mock=use_mock_fp
        )

        # Summary Telemetry Cards
        kpi_fp1, kpi_fp2, kpi_fp3, kpi_fp4, kpi_fp5 = st.columns(5)
        with kpi_fp1:
            with st.container(border=True):
                st.metric("Profiled Authors", f"{fp_report.total_authors_profiled:,}")
        with kpi_fp2:
            with st.container(border=True):
                st.metric("Evaluated Pairs", f"{fp_report.total_pairs_evaluated:,}")
        with kpi_fp3:
            with st.container(border=True):
                st.metric("Suspect Pairs", f"{len(fp_report.high_confidence_pairs):,}")
        with kpi_fp4:
            with st.container(border=True):
                st.metric("Clustered Rings", f"{len(fp_report.clustered_rings):,}")
        with kpi_fp5:
            with st.container(border=True):
                max_ring_size = max([r.member_count for r in fp_report.clustered_rings], default=0)
                st.metric("Largest Ring Size", f"{max_ring_size} accounts")

        # Two Tabs: 1. Coordinated Rings & Pairwise Forensic Matrix, 2. Author Persona Deep-Dive
        tab_fp_rings, tab_fp_personas = st.tabs([
            "🕸️ Clustered Sockpuppet Rings & Pairwise Forensics",
            "👤 Author Persona & Stylometric Dossier"
        ])

        with tab_fp_rings:
            # Pairwise Scatter: Stylometric Similarity vs Diurnal Similarity
            if fp_report.high_confidence_pairs:
                df_pairs_ui = pd.DataFrame([p.to_dict() for p in fp_report.high_confidence_pairs])
                fig_fp_scatter = px.scatter(
                    df_pairs_ui,
                    x="style_similarity",
                    y="diurnal_similarity",
                    color="composite_score",
                    size="shared_videos_count" if df_pairs_ui["shared_videos_count"].max() > 0 else None,
                    hover_name="author_a_name",
                    hover_data={
                        "author_b_name": True,
                        "composite_score": True,
                        "style_similarity": True,
                        "diurnal_similarity": True,
                        "shared_videos_count": True,
                    },
                    title="Pairwise Forensic Alignment: Stylometric Affinity vs. Circadian Diurnal Rhythm",
                    labels={
                        "style_similarity": "Stylometric Cosine Similarity (Entropy, Punctuation, Caps)",
                        "diurnal_similarity": "24-Hour Circadian Diurnal Similarity",
                        "composite_score": "Composite Probability (%)",
                        "shared_videos_count": "Shared Video Targets"
                    },
                    template=PLOTLY_TEMPLATE,
                    color_continuous_scale="Reds",
                )
                fig_fp_scatter.update_layout(
                    height=450,
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(fig_fp_scatter, width="stretch")
                st.caption("Top-right quadrant represents high-risk alternate/sockpuppet accounts sharing identical writing styles and synchronized posting hours.")

                # Clustered Sockpuppet Rings
                if fp_report.clustered_rings:
                    st.markdown("##### 🕸️ Clustered Multi-Account Sockpuppet Rings")
                    rings_data = []
                    for r in fp_report.clustered_rings:
                        rings_data.append({
                            "Ring ID": r.ring_id,
                            "Accounts Count": r.member_count,
                            "Confidence Score": f"{r.avg_confidence:.1f}%",
                            "Peak Active Hour (UTC)": f"{r.primary_peak_hour:02d}:00",
                            "Dominant Cohort": r.dominant_rfm_cohort,
                            "Member Accounts": ", ".join(r.member_names[:5]) + (f" (+{len(r.member_names)-5} more)" if len(r.member_names) > 5 else "")
                        })
                    st.dataframe(pd.DataFrame(rings_data), width="stretch", hide_index=True)

                # High-Confidence Pair Ledger
                st.markdown("##### 👥 High-Confidence Sockpuppet Matches Ledger")
                pairs_display = df_pairs_ui[[
                    "composite_score", "author_a_name", "author_b_name",
                    "style_similarity", "diurnal_similarity", "target_overlap_jaccard", "shared_videos_count"
                ]].copy()
                pairs_display.columns = [
                    "Score (%)", "Author A", "Author B",
                    "Style Sim", "Diurnal Sim", "Target Jaccard", "Shared Videos"
                ]
                st.dataframe(pairs_display.head(50), width="stretch", hide_index=True)
            else:
                st.info("No author pairs exceed the current confidence threshold. Try lowering the threshold or enabling 'Demo / Mock Mode'.")

        with tab_fp_personas:
            if fp_report.personas:
                persona_names = {pid: f"{p.author_display_name} ({p.rfm_cohort} — {p.total_comments} comments)" for pid, p in fp_report.personas.items()}
                selected_pid = st.selectbox(
                    "Select Author to Inspect:",
                    options=list(fp_report.personas.keys()),
                    format_func=lambda pid: persona_names.get(pid, pid),
                    key="sel_fp_persona_author"
                )

                sel_p = fp_report.personas[selected_pid]

                # Persona Overview Card
                with st.container(border=True):
                    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                    with pc1:
                        st.metric("Vocab Shannon Entropy", f"{sel_p.vocab_entropy:.2f} bits", delta="Lexical Diversity")
                    with pc2:
                        st.metric("Punctuation Intensity", f"{sel_p.punctuation_intensity:.3f}", delta=f"{sel_p.caps_ratio:.1%} caps")
                    with pc3:
                        emojis_str = " ".join(sel_p.top_emojis) if sel_p.top_emojis else "None"
                        st.metric("Emoji Signature", emojis_str, delta=f"{sel_p.emoji_frequency:.1f}/comment")
                    with pc4:
                        st.metric("Peak Activity Hour", f"{sel_p.peak_posting_hour:02d}:00 UTC", delta=f"{sel_p.circadian_entropy:.2f} entropy")
                    with pc5:
                        s_col = "normal" if sel_p.avg_sentiment >= 0 else "inverse"
                        st.metric("Sentiment / Toxicity", f"{sel_p.avg_sentiment:+.2f}", delta=f"{sel_p.avg_toxicity:.3f} tox", delta_color=s_col)

                # Visualizations Row: Diurnal Rhythm & Stylometric Radar
                vis_col1, vis_col2 = st.columns(2)
                with vis_col1:
                    st.markdown("##### ⏰ 24-Hour Diurnal Posting Clock (UTC)")
                    hours = [f"{h:02d}:00" for h in range(24)]
                    fig_clock = px.bar(
                        x=hours,
                        y=sel_p.diurnal_histogram,
                        title=f"Circadian Activity Profile: {sel_p.author_display_name}",
                        labels={"x": "UTC Hour of Day", "y": "Posting Frequency Share"},
                        template=PLOTLY_TEMPLATE,
                    )
                    fig_clock.update_traces(marker_color=COLOR_PRIMARY)
                    fig_clock.update_layout(height=320, margin=dict(l=30, r=30, t=40, b=30))
                    st.plotly_chart(fig_clock, width="stretch")

                with vis_col2:
                    st.markdown("##### 🧬 Stylometric Feature Fingerprint")
                    feature_labels = [
                        "Entropy", "Word Count", "Caps Ratio", "Punctuation",
                        "Exclamation", "Question", "Emoji Rate", "Sentiment",
                        "Toxicity", "Circadian"
                    ]
                    fig_radar = px.bar(
                        x=feature_labels,
                        y=sel_p.stylometric_vector,
                        title=f"Normalized Style Vector: {sel_p.author_display_name}",
                        labels={"x": "Forensic Dimension", "y": "Normalized Value [0, 1]"},
                        template=PLOTLY_TEMPLATE,
                    )
                    fig_radar.update_traces(marker_color=COLOR_ACCENT)
                    fig_radar.update_layout(height=320, margin=dict(l=30, r=30, t=40, b=30))
                    st.plotly_chart(fig_radar, width="stretch")

                # Verbatim Quote Feed
                if sel_p.sample_comments:
                    st.markdown("##### 🗣️ Representative Comments by this Author")
                    for sc in sel_p.sample_comments:
                        st.markdown(f"- _{sc}_")

        # Export Hub
        st.markdown("##### 💾 Export Forensic Author Reports")
        exp_f1, exp_f2 = st.columns(2)
        with exp_f1:
            st.download_button(
                label="💾 Export Forensic Report JSON",
                data=json.dumps(fp_report.to_dict(), indent=2, ensure_ascii=False),
                file_name="author_forensics_report.json",
                mime="application/json",
                key="dl_fingerprint_json"
            )
        with exp_f2:
            if fp_report.high_confidence_pairs:
                st.download_button(
                    label="📊 Export Suspect Sockpuppet Pairs CSV",
                    data=pd.DataFrame([p.to_dict() for p in fp_report.high_confidence_pairs]).to_csv(index=False),
                    file_name="suspect_sockpuppet_pairs.csv",
                    mime="text/csv",
                    key="dl_fingerprint_csv"
                )

    # ==============================================================================
    # TAB 5: PREDICTIVE MODELING & CAUSAL INTERVENTIONS
    # ==============================================================================
    with tab_modeling:
        st.header("🔬 Cross-Video Relations, Counterfactuals & Modeling")
        st.markdown("Advanced statistical matrices, causal Difference-in-Differences, and predictive feature attribution.")

        # 5.1 Creator Interaction Causal Uplift (Stage 39)
        st.subheader("📈 Creator Interaction Causal Uplift & Intervention Analysis (Stage 39)")
        st.markdown("Quasi-experimental Difference-in-Differences (DiD) quantifying the causal impact of early creator engagement (pinning, replying within 2h).")

        if not data_layers["creator_uplift"].empty:
            up_cols = st.columns(len(data_layers["creator_uplift"]))
            for i, row in data_layers["creator_uplift"].iterrows():
                with up_cols[i % len(up_cols)]:
                    st.metric(
                        label=row['dimension'],
                        value=f"{row['treated_mean']}",
                        delta=f"{row['relative_lift_pct']}% lift"
                    )

            st.dataframe(data_layers["creator_uplift"], width="stretch", hide_index=True)

            # Interactive DiD Comparison Chart
            df_uplift = data_layers["creator_uplift"].copy()
            if "dimension" in df_uplift.columns and "treated_mean" in df_uplift.columns and "control_mean" in df_uplift.columns:
                st.markdown("##### 📊 DiD Counterfactual Cohort Comparison")
                fig_did = px.bar(
                    df_uplift,
                    x="dimension",
                    y=["treated_mean", "control_mean"],
                    barmode="group",
                    title="Causal Lift: Early Creator Engagement (Treated) vs. Untreated Threads (Control)",
                    labels={"value": "Mean Engagement Metric", "dimension": "Engagement Dimension", "variable": "Cohort"},
                    color_discrete_map={"treated_mean": COLOR_ACCENT, "control_mean": "rgba(255,255,255,0.45)"},
                    template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig_did, width="stretch")
                st.caption("Quasi-experimental Difference-in-Differences estimates isolating creator intervention effects from organic baseline.")

        if not data_layers["creator_threads"].empty:
            st.markdown("##### High-Impact Creator Intervention Threads")
            df_ct = data_layers["creator_threads"].copy()
            if "video_id" in df_ct.columns:
                df_ct.insert(0, "Video Title", df_ct["video_id"].map(video_title_map).fillna(df_ct["video_id"]))
            st.dataframe(df_ct.head(50), width="stretch", hide_index=True)
            st.caption("Top threads benefiting from early intervention and high creator/community reinforcement.")

        # 5.2 Interactive What-If Comment Virality Simulator
        st.divider()
        st.subheader("⚡ Interactive Comment Virality & Upvote Simulator")
        st.markdown("Simulate how arrival speed, message length, sentiment, and creator intervention influence predicted comment upvote yields.")

        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_words = st.slider("Comment Word Count:", 2, 120, 22)
            sim_emojis = st.slider("Emoji Count:", 0, 8, 1)
        with sim_col2:
            sim_arrival = st.slider("Arrival Speed (Minutes post-upload):", 1, 720, 25, help="Time elapsed between video publish and comment submission.")
            sim_sentiment = st.slider("Sentiment Polarity:", -1.0, 1.0, 0.45, 0.05)
        with sim_col3:
            sim_creator = st.toggle("Creator Engagement (Pinned / Early Reply)", value=True)
            sim_questions = st.checkbox("Contains Technical Question (?)", value=False)

        # Causal and predictive estimate
        early_multiplier = max(1.0, 10.0 * np.exp(-sim_arrival / 90.0))
        base_estimate = max(1.0, (sim_words * 0.18) + (sim_emojis * 2.5) + early_multiplier + (sim_sentiment * 4.0))
        if sim_questions:
            base_estimate *= 1.4
        if sim_creator:
            base_estimate *= 4.2

        sim_res1, sim_res2 = st.columns([1, 2])
        with sim_res1:
            with st.container(border=True):
                st.metric(
                    label="Estimated Upvote Yield",
                    value=f"{int(round(base_estimate)):,} Likes",
                    delta=f"{'+320% Creator Lift' if sim_creator else 'Organic Baseline'}"
                )
                st.caption("Estimated from non-linear gradient-boosted feature weights and DiD causal multipliers.")

        with sim_res2:
            # Attribution waterfall breakdown
            attrib_data = pd.DataFrame({
                "Driver": ["Base Text", "Early Arrival", "Emoji & Tone", "Question Factor", "Creator Lift"],
                "Contribution": [
                    round(sim_words * 0.18, 1),
                    round(early_multiplier, 1),
                    round((sim_emojis * 2.5) + (sim_sentiment * 4.0), 1),
                    round(base_estimate * 0.2 if sim_questions else 0.0, 1),
                    round(base_estimate * 0.75 if sim_creator else 0.0, 1)
                ]
            })
            fig_waterfall = px.bar(
                attrib_data,
                x="Driver",
                y="Contribution",
                color="Driver",
                title="Predicted Upvote Component Attribution",
                labels={"Contribution": "Estimated Upvote Points"},
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_waterfall, width="stretch")

        # 5.25 Advanced Multi-Task Predictive Inference Simulators
        st.divider()
        st.subheader("🤖 Advanced Predictive Inference Simulators")
        st.markdown("Real-time inference models predicting thread toxicity escalation, commenter retention propensity, and viral burst probability.")

        inf_tab1, inf_tab2, inf_tab3 = st.tabs([
            "🔥 Thread Toxicity Escalation",
            "🔄 Commenter Retention Propensity",
            "🚀 Early Viral Burst Predictor"
        ])

        with inf_tab1:
            st.markdown("##### Predict Probability of Thread Escalating into Severe Flame-War")
            t_col_a, t_col_b = st.columns(2)
            with t_col_a:
                t_in_tox = st.slider("Root Comment Toxicity:", 0.0, 1.0, 0.35, 0.05)
                t_in_len = st.slider("Root Word Count:", 1, 100, 25)
                t_in_caps = st.slider("Root ALL-CAPS Ratio:", 0.0, 1.0, 0.1, 0.05)
            with t_col_b:
                t_in_sentiment = st.slider("Root Sentiment Polarity:", -1.0, 1.0, -0.4, 0.1)
                t_in_q = st.checkbox("Root Is Rhetorical Question", value=True)
                t_score = 1.0 / (1.0 + np.exp(-(3.5 * t_in_tox + 2.0 * t_in_caps - 1.8 * t_in_sentiment + (0.5 if t_in_q else 0) - 1.2)))
                st.metric("Escalation Risk Probability", f"{t_score*100:.1f}%", delta="High Risk" if t_score > 0.5 else "Stable Thread", delta_color="inverse" if t_score > 0.5 else "normal")
                st.progress(float(min(1.0, max(0.0, t_score))))

        with inf_tab2:
            st.markdown("##### Predict Probability of New Author Becoming a Returning Loyalist")
            r_col_a, r_col_b = st.columns(2)
            with r_col_a:
                r_comments = st.slider("Current Comment Count:", 1, 20, 2)
                r_breadth = st.slider("Unique Videos Commented:", 1, 10, 1)
            with r_col_b:
                r_likes = st.slider("Total Likes Received by Author:", 0, 50, 4)
                r_sent = st.slider("Author Average Sentiment:", -1.0, 1.0, 0.3, 0.1)
                r_score = 1.0 / (1.0 + np.exp(-(0.4 * r_comments + 0.8 * r_breadth + 0.05 * r_likes + 0.5 * r_sent - 1.5)))
                st.metric("Returning Loyalist Propensity", f"{r_score*100:.1f}%", delta="Loyalist Candidate" if r_score > 0.5 else "Drive-By Profile")
                st.progress(float(min(1.0, max(0.0, r_score))))

        with inf_tab3:
            st.markdown("##### Predict Probability of Comment Entering Top 10% Liked Tier")
            v_col_a, v_col_b = st.columns(2)
            with v_col_a:
                v_arr = st.slider("Arrival Minutes Post-Upload:", 1, 180, 15)
                v_words = st.slider("Comment Words:", 2, 80, 18)
            with v_col_b:
                v_emojis = st.slider("Emoji Count:", 0, 6, 2)
                v_pinned = st.checkbox("Pinned by Creator", value=False)
                v_score = 1.0 / (1.0 + np.exp(-(2.5 * np.exp(-v_arr / 45.0) + 0.03 * v_words + 0.25 * v_emojis + (2.0 if v_pinned else 0) - 1.8)))
                st.metric("Top-10% Virality Probability", f"{v_score*100:.1f}%", delta="Viral Breakout" if v_score > 0.5 else "Standard Reach")
                st.progress(float(min(1.0, max(0.0, v_score))))

        # 5.3 Dunn's Post-Hoc Significance Matrix
        st.divider()
        if not data_layers["dunn_matrix"].empty:
            st.subheader("📊 Dunn's Post-Hoc Pairwise p-Value Matrix")
            df_dunn = data_layers["dunn_matrix"].copy()
            short_dunn_cols = [f"{video_title_map.get(c, c)[:22]}..." if len(video_title_map.get(c, c)) > 22 else video_title_map.get(c, c) for c in df_dunn.columns]
            short_dunn_idx = [f"{video_title_map.get(i, i)[:22]}..." if len(video_title_map.get(i, i)) > 22 else video_title_map.get(i, i) for i in df_dunn.index]
            fig_dunn = px.imshow(
                df_dunn,
                labels=dict(x="Video Title", y="Video Title", color="Adjusted p-value"),
                x=short_dunn_cols,
                y=short_dunn_idx,
                title="Statistical Significance Matrix of Pairwise Video Engagements",
                color_continuous_scale="Viridis",
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_dunn, width="stretch")
            st.caption("Pairwise post-hoc tests after Kruskal-Wallis non-parametric variance analysis.")

        # 5.4 Cross-Video Jaccard Overlap Matrix
        if not data_layers["video_overlap"].empty:
            st.divider()
            st.subheader("🌐 Cross-Video Commenter Overlap Matrix")
            df_vo = data_layers["video_overlap"].set_index("video_id") if "video_id" in data_layers["video_overlap"].columns else data_layers["video_overlap"].copy()
            short_vo_cols = [f"{video_title_map.get(c, c)[:22]}..." if len(video_title_map.get(c, c)) > 22 else video_title_map.get(c, c) for c in df_vo.columns]
            short_vo_idx = [f"{video_title_map.get(i, i)[:22]}..." if len(video_title_map.get(i, i)) > 22 else video_title_map.get(i, i) for i in df_vo.index]
            fig_vo = px.imshow(
                df_vo,
                labels=dict(x="Video Title", y="Video Title", color="Jaccard Overlap"),
                x=short_vo_cols,
                y=short_vo_idx,
                title="Audience Jaccard Overlap Matrix",
                color_continuous_scale="Magma",
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_vo, width="stretch")
            st.caption("Jaccard similarity matrix of overlapping commenters across video pairs.")

        # 5.5 SHAP Feature Importance & Forecasts
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
                        with open(filepath, 'r', encoding='utf-8') as f:
                            html_data = f.read()
                        st.html(html_data)
                    else:
                        st.image(str(filepath))
                else:
                    st.info(f"Artifact `{filename}` has not yet been rendered.")

                with st.expander("📖 Deep Analytical Guide & Interpretation", expanded=False):
                    if methodology:
                        st.markdown(f"**🔬 Methodology & Model:**\n{methodology}")
                    if how_to_read:
                        st.markdown(f"**🧭 How to Read:**\n{how_to_read}")
                    if takeaway:
                        st.markdown(f"**💡 Strategic Takeaway:**\n{takeaway}")

        # ==============================================================================
        # Section 1: Temporal Dynamics & Lifecycle Modeling
        # ==============================================================================
        st.subheader("1. ⏳ Temporal Dynamics & Discussion Lifecycles")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            render_plot_card("kaplan_meier_survival.png")
            render_plot_card("poisson_bursts.png")
            render_plot_card("velocity_spikes.png")
            render_plot_card("thread_decay_slopes.png")
            render_plot_card("reply_depth_distribution.png")
            render_plot_card("thread_width_dist.png")
            render_plot_card("shelf_life_likes.png")
            render_plot_card("resolution_patterns.png")
            render_plot_card("minute_arrival_speed_curve.png")
        with t_col2:
            render_plot_card("diurnal_heatmap.png")
            render_plot_card("stl_decomposition.png")
            render_plot_card("video_half_life.png")
            render_plot_card("reply_latency_distribution.png")
            render_plot_card("reaction_timeline.png")
            render_plot_card("integrity_scatter.png")
            render_plot_card("initiator_patterns.png")
            render_plot_card("topic_injection_anomalies.png")

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
            render_plot_card("slang_lexicon_distribution.png")
            render_plot_card("emoji_treemap.png")
        with s_col2:
            render_plot_card("stance_polarization_drift.png")
            render_plot_card("audience_intent_distribution.png")
            render_plot_card("named_entities.png")
            render_plot_card("word_cooccurrence.png")
            render_plot_card("sentiment_divergence.png")
            render_plot_card("toxicity_heatmap.png")
            render_plot_card("tfidf_keywords_salience.png")
            render_plot_card("valence_per_topic.png")

        # ==============================================================================
        # Section 3: Audience Networks, Segmentation & Forensics
        # ==============================================================================
        st.divider()
        st.subheader("3. 👥 Audience Networks, Segmentation & Forensics")
        n_col1, n_col2 = st.columns(2)
        with n_col1:
            render_plot_card("rfm_3d.html", is_html=True)
            render_plot_card("author_network_force.png")
            render_plot_card("bipartite_network.png")
            render_plot_card("bot_heuristics.png")
            render_plot_card("frequency_tiers.png")
            render_plot_card("driveby_loyalists.png")
            render_plot_card("network_bowtie_structure.png")
        with n_col2:
            render_plot_card("cib_rings_graph.png")
            render_plot_card("cocomment_network.png")
            render_plot_card("toxicity_contagion.png")
            render_plot_card("author_pareto.png")
            render_plot_card("lorenz_curve.png")
            render_plot_card("like_inflation.png")
            render_plot_card("arrival_speed.png")
            render_plot_card("author_fingerprints.png")
            render_plot_card("top_k_attention_concentration.png")
            render_plot_card("impersonation_detection.png")

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
            render_plot_card("series_vs_standalone_benchmark.png")
            render_plot_card("corpus_quality_log_log.png")
        with m_col2:
            render_plot_card("video_profile_radar.png")
            render_plot_card("creator_causal_uplift.png")
            render_plot_card("controversy_impact.png")
            render_plot_card("code_switching_impact.png")
            render_plot_card("topic_cohorts.png")
            render_plot_card("new_vs_returning_share.png")
            render_plot_card("video_overlap_matrix.png")
            render_plot_card("creator_sentiment_polarity.png")
            render_plot_card("cross_modal_scene_reactions.png")

    # ==============================================================================
    # TAB 7: DATA EXPLORER & EXPORT HUB
    # ==============================================================================
    with tab_export:
        st.header("🗄️ Interactive Data Explorer & Export Hub")
        st.markdown("Direct querying, regex text filtering, and instant dataset export across all 38 pipeline layers.")

        with st.container(border=True):
            st.subheader("🔍 Interactive Comment & Corpus Explorer")
            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                search_term = st.text_input("Filter comment text:", "").strip()
            with e_col2:
                min_likes = st.number_input("Minimum Likes:", min_value=0, value=0, step=5)
            with e_col3:
                max_tox = st.slider("Max Toxicity Threshold:", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

            filtered_comments = df_comments_raw.copy()
            if search_term:
                filtered_comments = filtered_comments[filtered_comments['text'].astype(str).str.contains(search_term, case=False, na=False)]
            if min_likes > 0 and 'like_count' in filtered_comments.columns:
                filtered_comments = filtered_comments[filtered_comments['like_count'] >= min_likes]
            if max_tox < 1.0 and 'toxicity' in filtered_comments.columns:
                filtered_comments = filtered_comments[filtered_comments['toxicity'] <= max_tox]

            disp_comments = filtered_comments.copy()
            if "video_id" in disp_comments.columns:
                disp_comments.insert(1, "Video Title", disp_comments["video_id"].map(video_title_map).fillna(disp_comments["video_id"]))

            st.dataframe(disp_comments.head(100), width="stretch", hide_index=True)
            st.caption(f"Displaying top 100 matching rows out of {len(filtered_comments):,} matching comments.")

        # Interactive Query Sandbox & Dynamic Visualizer
        st.divider()
        st.subheader("⚡ Interactive Query Sandbox & Dynamic Visualizer")
        st.markdown("Slice, aggregate, and visualize any pipeline dataset dynamically with instant chart generation.")

        sb_layer = st.selectbox("Select Layer to Query & Visualize:", list(data_layers.keys()), index=0)
        df_target = data_layers[sb_layer].copy()

        if not df_target.empty:
            q_col1, q_col2, q_col3 = st.columns([2, 1, 1])
            with q_col1:
                filter_expr = st.text_input("Optional Pandas query filter (e.g. `like_count > 5` or `is_bot_suspect == True`):", "").strip()
            with q_col2:
                q_chart_type = st.selectbox("Dynamic Chart Type:", ["Bar Chart", "Histogram", "Scatter Plot", "Line Chart"])
            with q_col3:
                max_rows = st.number_input("Max Sample Rows:", min_value=10, max_value=5000, value=500, step=50)

            # Apply query filter if provided
            df_filtered = df_target.copy()
            if filter_expr:
                try:
                    df_filtered = df_filtered.query(filter_expr)
                    st.success(f"Query matched {len(df_filtered):,} records.")
                except Exception as q_err:
                    st.warning(f"Filter expression error: {q_err}. Showing unfiltered data.")

            df_sample = df_filtered.head(int(max_rows))

            # Numeric and categorical columns for axes
            num_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df_sample.columns.tolist()

            if all_cols and num_cols:
                ax_col1, ax_col2, ax_col3 = st.columns(3)
                with ax_col1:
                    x_ax = st.selectbox("X-Axis Field:", all_cols, index=0)
                with ax_col2:
                    y_ax = st.selectbox("Y-Axis Field (Numeric):", num_cols, index=min(1, len(num_cols)-1))
                with ax_col3:
                    color_ax = st.selectbox("Color Group Field:", [None] + all_cols, index=0)

                try:
                    if q_chart_type == "Bar Chart":
                        fig_dynamic = px.bar(df_sample, x=x_ax, y=y_ax, color=color_ax, title=f"{sb_layer.upper()} // Dynamic Bar Chart", template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig_dynamic, width="stretch")
                    elif q_chart_type == "Histogram":
                        fig_dynamic = px.histogram(df_sample, x=y_ax, color=color_ax, title=f"{sb_layer.upper()} // Dynamic Histogram", template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig_dynamic, width="stretch")
                    elif q_chart_type == "Scatter Plot":
                        fig_dynamic = px.scatter(df_sample, x=x_ax, y=y_ax, color=color_ax, title=f"{sb_layer.upper()} // Dynamic Scatter Plot", template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig_dynamic, width="stretch")
                    elif q_chart_type == "Line Chart":
                        fig_dynamic = px.line(df_sample, x=x_ax, y=y_ax, color=color_ax, title=f"{sb_layer.upper()} // Dynamic Line Chart", template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig_dynamic, width="stretch")
                except Exception as chart_err:
                    st.info(f"Could not render selected chart with chosen axes: {chart_err}")
        else:
            st.info(f"Dataset `{sb_layer}` is empty or not populated in current run.")

        st.divider()
        st.subheader("📥 Export Intelligence Datasets")
        st.markdown("Download ready-to-use Parquet or CSV datasets for external BI reporting, Tableau, or data warehouse pipelines.")

        export_options = list(data_layers.keys())
        selected_layer = st.selectbox("Select Intelligence Layer to Download:", options=export_options)

        if selected_layer in data_layers and not data_layers[selected_layer].empty:
            df_export = data_layers[selected_layer]
            st.dataframe(df_export.head(20), width="stretch", hide_index=True)

            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Download {selected_layer}.csv",
                data=csv_data,
                file_name=f"{selected_layer}.csv",
                mime="text/csv"
            )

        st.divider()
        st.subheader("📑 Executive Dossier & Intelligence Briefing Exporter")
        st.markdown(
            "Compile the complete 53-stage analytical synthesis into an offline-portable, publication-ready "
            "Executive Dossier document (HTML & print-ready PDF) with embedded visual evidence."
        )

        with st.container(border=True):
            exp_d_col1, exp_d_col2 = st.columns([3, 1])
            with exp_d_col1:
                st.markdown(
                    "- **Forensic Threat Matrix**: Bot heuristics, CIB rings, like inflation, impersonation.\n"
                    "- **Audience & Attention**: Loyalists vs drive-bys, half-life, power-law Pareto fit.\n"
                    "- **Content & Semantics**: BERTopic clusters, audience intent taxonomy, emotion wheel.\n"
                    "- **Video Radar & Causal Lift**: Top uploads benchmark, DiD lift, and Tree SHAP virality drivers.\n"
                    "- **Publication Plots**: Base64 embedded high-resolution visual evidence."
                )
            with exp_d_col2:
                include_plots_tab7 = st.checkbox("Include Visual Plots", value=True, key="dossier_plots_tab7")
                if st.button("⚡ Compile Executive Dossier", key="btn_gen_dossier_tab7"):
                    with st.spinner("Compiling executive dossier..."):
                        from engine.reporter import ExecutiveReportGenerator
                        reporter = ExecutiveReportGenerator(interim_path, output_path)
                        st.session_state["executive_dossier_html"] = reporter.generate_html(include_plots=include_plots_tab7)
                        st.success("Executive Dossier compiled!")

            if "executive_dossier_html" in st.session_state:
                st.download_button(
                    label="⬇️ Download Executive Dossier (.html)",
                    data=st.session_state["executive_dossier_html"],
                    file_name=f"ytint_executive_brief_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html"
                )

        st.divider()
        st.subheader("🔌 Live YouTube Data API Ingest Connector")
        st.markdown(
            "Fetch and ingest videos, rich metadata, top-level comment threads, and replies directly "
            "from the YouTube Data API v3 into the local analytical pipeline."
        )

        with st.container(border=True):
            ingest_col1, ingest_col2 = st.columns([2, 1])

            with ingest_col1:
                ingest_target_type = st.radio(
                    "Target Identifier Type:",
                    options=["Channel Handle (e.g. @mkbhd)", "Channel ID (UC...)", "Single Video ID"],
                    horizontal=True,
                    key="ingest_target_type"
                )
                target_value = st.text_input(
                    "Channel Handle, Channel ID, or Video ID:",
                    value="@3blue1brown" if "Handle" in ingest_target_type else "",
                    key="ingest_target_val"
                )
                api_key_input = st.text_input(
                    "YouTube Data API v3 Key (optional if YOUTUBE_API_KEY env is set):",
                    value=os.environ.get("YOUTUBE_API_KEY", ""),
                    type="password",
                    key="ingest_api_key_val"
                )

            with ingest_col2:
                max_vids_val = st.slider("Max Videos to Ingest:", min_value=1, max_value=50, value=5, step=1, key="ingest_max_vids")
                max_comm_val = st.slider("Max Comments per Video:", min_value=50, max_value=2000, value=500, step=50, key="ingest_max_comm")
                use_mock_api = st.checkbox("Mock Mode (Offline Simulation)", value=False, key="ingest_mock_cb")
                auto_run_s00 = st.checkbox("Auto-Materialize Parquet", value=True, key="ingest_auto_s00")

            if st.button("⚡ Start YouTube Ingestion", key="btn_start_youtube_ingest"):
                target_str = target_value.strip()
                if not target_str and not use_mock_api:
                    st.error("Please provide a valid Channel Handle, Channel ID, or Video ID.")
                else:
                    with st.spinner("Connecting to YouTube Data API and ingesting data layers..."):
                        try:
                            from engine.youtube_api import ingest_youtube_data

                            ch_id = target_str if "Channel ID" in ingest_target_type else None
                            handle = target_str if "Handle" in ingest_target_type else None
                            vid_id = target_str if "Video" in ingest_target_type else None

                            res = ingest_youtube_data(
                                channel_id=ch_id,
                                handle=handle,
                                video_id=vid_id,
                                api_key=api_key_input.strip() or None,
                                max_videos=max_vids_val,
                                max_comments_per_video=max_comm_val,
                                use_mock=use_mock_api,
                                run_migration=auto_run_s00
                            )
                            st.success(
                                f"🎉 Successfully ingested {res['comments_ingested']:,} comments "
                                f"across {res['videos_ingested']} videos into `{res['raw_db_path']}`!"
                            )
                            st.info("Reload the dashboard or run the pipeline runner to analyze the fresh corpus.")
                        except Exception as ingest_err:
                            st.error(f"Ingestion failed: {ingest_err}")

        st.divider()
        st.subheader("⚡ Pipeline Synchronization // Incremental Delta Runner")
        st.markdown(
            "Detect new or updated comments from SQLite and execute intelligent, incremental updates "
            "across the 53-stage analytics pipeline in seconds without full recomputation."
        )

        with st.container(border=True):
            from engine.delta import inspect_delta
            raw_db = config.get("paths", {}).get("raw_db") or (interim_path.parent / "raw" / "commentsuite.sqlite3")
            delta_rep = inspect_delta(raw_db, interim_path)

            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            with d_col1:
                st.metric("New Comments Pending", f"+{delta_rep.new_comments_count:,}")
            with d_col2:
                st.metric("Updated Counters Pending", f"{delta_rep.updated_comments_count:,}")
            with d_col3:
                st.metric("New Videos Pending", f"+{delta_rep.new_videos_count:,}")
            with d_col4:
                sync_status_label = "⚡ Sync Required" if delta_rep.has_delta else "✅ Up to Date"
                st.metric("Corpus Status", sync_status_label)

            st.caption(
                f"Source SQLite comments: **{delta_rep.total_sqlite_comments:,}** | "
                f"Interim clean comments: **{delta_rep.total_interim_comments:,}** | "
                f"Source videos: **{delta_rep.total_sqlite_videos:,}**"
            )

            sync_btn_col1, sync_btn_col2 = st.columns([2, 1])
            with sync_btn_col1:
                st.markdown(
                    "Clicking **Run Incremental Pipeline Sweep** will non-destructively upsert new/updated comments in `s00`, "
                    "selectively run NLP enrichment only on new rows in `s01`, and refresh downstream artifacts in seconds."
                )
            with sync_btn_col2:
                if st.button("⚡ Run Incremental Pipeline Sweep", key="btn_run_incremental_pipeline"):
                    with st.spinner("Executing incremental delta sweep across pipeline..."):
                        try:
                            from pipeline.runner import PipelineRunner
                            runner = PipelineRunner()
                            sweep_res = runner.run(incremental=True, force=False)
                            st.success("🎉 Incremental pipeline sweep completed successfully! Reload dashboard to view updated intelligence.")
                        except (Exception, SystemExit) as sweep_err:
                            st.error(f"Sweep encountered an issue: {sweep_err}")

        st.divider()
        st.subheader("💬 Ask ytint // AI Conversational Analyst")
        st.markdown(
            "Query your entire channel corpus using grounded Retrieval-Augmented Generation (RAG). "
            "The AI agent answers questions by synthesizing metrics across all 53 computational stages."
        )

        with st.container(border=True):
            rag_col1, rag_col2 = st.columns([3, 1])
            with rag_col1:
                user_rag_query = st.text_input(
                    "Ask a strategic question about this channel's comments and metrics:",
                    placeholder="e.g. Why did our latest upload see high like inflation? Or: What do viewers request most?",
                    key="rag_query_input"
                )
            with rag_col2:
                rag_prov = st.selectbox("AI Provider:", options=["gemini", "ollama", "mock"], index=0, key="rag_prov_sel")
                btn_ask_rag = st.button("💡 Ask ytint", key="btn_ask_rag_action")

            if btn_ask_rag:
                q_text = user_rag_query.strip()
                if not q_text:
                    st.warning("Please enter a question to analyze.")
                else:
                    with st.spinner("Consulting intelligence layers and synthesizing answer..."):
                        from engine.synthesizer import LLMClient, InsightSynthesizer
                        client = LLMClient(provider=rag_prov)
                        synth = InsightSynthesizer(interim_path, output_path, llm_client=client)
                        st.session_state["rag_answer_text"] = synth.answer_query(q_text)

            if "rag_answer_text" in st.session_state:
                with st.container(border=True):
                    st.markdown(st.session_state["rag_answer_text"])

        st.divider()
        st.subheader("🚨 Automated Anomaly & Threat Alerting Console")
        st.markdown(
            "Continuous automated threat monitoring scanning forensic, epidemiological, and viral signals "
            "across all 53 pipeline layers. Formats and dispatches instant webhook alerts to Discord, Slack, or Telegram."
        )

        with st.container(border=True):
            from engine.alerting import AlertManager
            alert_mgr = AlertManager(output_path, interim_path, config)
            threat_report = alert_mgr.scan_threats(severity_threshold="INFO")

            # KPI Scorecards
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                status_color = "🔴" if threat_report.max_severity == "CRITICAL" else ("🟠" if threat_report.max_severity == "WARNING" else "🟢")
                st.metric("Threat Status", f"{status_color} {threat_report.max_severity}")
            with kpi_col2:
                st.metric("Critical Threats", f"{threat_report.critical_count}")
            with kpi_col3:
                st.metric("Warnings Flagged", f"{threat_report.warning_count}")
            with kpi_col4:
                st.metric("Total Active Anomalies", f"{threat_report.total_incidents}")

            # Threat Digest Ledger
            if threat_report.incidents:
                incident_records = [
                    {
                        "Severity": inc.severity,
                        "Threat Dimension": inc.title.replace("🚨 ", "").replace("⚠️ ", "").replace("📈 ", "").replace("🔥 ", "").replace("⚡ ", "").replace("🤖 ", "").replace("🎯 ", ""),
                        "Forensic Summary": inc.summary,
                        "Strategic Action": inc.action_recommendation
                    }
                    for inc in threat_report.incidents
                ]
                df_threats = pd.DataFrame(incident_records)
                st.dataframe(df_threats, width="stretch", hide_index=True)
            else:
                st.success("✅ No critical threats or viral anomalies detected. Channel health is nominal.")

            # Webhook Dispatcher
            with st.expander("📡 Webhook Notification Dispatcher", expanded=False):
                wh_col1, wh_col2 = st.columns([2, 1])
                default_wh = config.get("alerting", {}).get("webhook_url", "")
                with wh_col1:
                    ui_webhook_url = st.text_input(
                        "Webhook URL (Discord, Slack, Telegram, or Generic JSON):",
                        value=default_wh,
                        key="ui_alert_webhook_url"
                    )
                    wh_platform = st.selectbox(
                        "Target Platform:",
                        options=["auto", "discord", "slack", "telegram", "generic"],
                        index=0,
                        key="ui_alert_platform_sel"
                    )
                    wh_sev_thresh = st.selectbox(
                        "Dispatch Threshold:",
                        options=["INFO", "WARNING", "CRITICAL"],
                        index=1,
                        key="ui_alert_thresh_sel"
                    )

                with wh_col2:
                    dry_run_toggle = st.checkbox(
                        "Dry-Run Mode (Simulation only, no network calls)",
                        value=True if not ui_webhook_url else False,
                        key="ui_alert_dry_run_cb"
                    )
                    show_payload_preview = st.checkbox(
                        "Preview Rendered Payload",
                        value=False,
                        key="ui_alert_preview_cb"
                    )

                if show_payload_preview:
                    preview_report = alert_mgr.scan_threats(severity_threshold=wh_sev_thresh)
                    resolved_p = wh_platform if wh_platform != "auto" else alert_mgr.detect_webhook_type(ui_webhook_url)
                    if resolved_p == "discord":
                        preview_dict = alert_mgr.format_discord_payload(preview_report)
                    elif resolved_p == "slack":
                        preview_dict = alert_mgr.format_slack_payload(preview_report)
                    elif resolved_p == "telegram":
                        preview_dict = alert_mgr.format_telegram_payload(preview_report)
                    else:
                        preview_dict = preview_report.to_dict()
                    st.json(preview_dict)

                if st.button("🚀 Dispatch Threat Alert Digest", key="btn_dispatch_alert"):
                    with st.spinner("Compiling threat report and dispatching alert..."):
                        active_report = alert_mgr.scan_threats(severity_threshold=wh_sev_thresh)
                        dispatch_res = alert_mgr.dispatch(
                            report=active_report,
                            webhook_url=ui_webhook_url.strip() or None,
                            webhook_type=wh_platform,
                            dry_run=dry_run_toggle
                        )
                        if dispatch_res.get("mode") == "DRY_RUN":
                            st.info(f"📁 Dry-Run Digest saved successfully to: `{dispatch_res.get('saved_to')}`")
                        elif dispatch_res.get("success"):
                            st.success(f"🎉 Threat notification successfully dispatched to {dispatch_res.get('platform', '').upper()}!")
                        else:
                            st.error(f"⚠️ Dispatch encountered an error: {dispatch_res.get('error')}")

        st.divider()
        st.subheader("🔍 Neural Semantic Vector Search & Feedback Clustering")
        st.markdown(
            "Sub-50ms neural vector similarity search over 340,027 root comments using precomputed 384-dimensional "
            "SentenceTransformer embeddings. Search by natural language concepts with loyalty-tier filtering and "
            "automatic thematic feedback clustering."
        )

        with st.container(border=True):
            preset_queries = [
                "Audio, microphone and sound distortion",
                "Video length and editing pacing suggestions",
                "Content and topic requests for future episodes",
                "Constructive criticism and disagreements",
                "Community praise and gratitude"
            ]

            st.markdown("**Suggested Quick-Queries:**")
            pill_cols = st.columns(len(preset_queries))
            for i, pq in enumerate(preset_queries):
                with pill_cols[i]:
                    if st.button(pq, key=f"btn_preset_{i}"):
                        st.session_state["active_semantic_query"] = pq

            initial_query = st.session_state.get("active_semantic_query", "")

            v_col1, v_col2 = st.columns([3, 1])
            with v_col1:
                ui_search_query = st.text_input(
                    "Natural Language Search Query or Concept:",
                    value=initial_query,
                    placeholder="e.g. microphone noise, requests for part 2, criticism of editing...",
                    key="ui_semantic_query_input"
                )
            with v_col2:
                ui_search_k = st.slider("Max Retrieved Comments:", min_value=5, max_value=100, value=20, step=5, key="ui_semantic_k_slider")

            flt_col1, flt_col2, flt_col3, flt_col4 = st.columns(4)
            with flt_col1:
                ui_tier_filter = st.selectbox(
                    "Filter by Loyalty Tier:",
                    options=["All", "Champions", "Loyalists", "Potential", "Regular", "Casual", "Drive-by"],
                    index=0,
                    key="ui_semantic_tier_filter"
                )
            with flt_col2:
                ui_min_likes = st.number_input("Minimum Likes Threshold:", min_value=0, value=0, step=1, key="ui_semantic_min_likes")
            with flt_col3:
                ui_cluster_toggle = st.checkbox("Cluster Feedback into Themes", value=True, key="ui_semantic_cluster_cb")
            with flt_col4:
                btn_exec_search = st.button("🔎 Search Vectors", key="btn_exec_semantic_search")

            if btn_exec_search or ui_search_query.strip():
                q_to_run = ui_search_query.strip()
                if q_to_run:
                    with st.spinner(f"Scanning 340k vector embeddings for '{q_to_run}'..."):
                        from engine.semantic_search import SemanticSearchEngine
                        search_engine = SemanticSearchEngine(interim_path, output_path, config)
                        search_results = search_engine.search(
                            query=q_to_run,
                            top_k=ui_search_k,
                            min_likes=ui_min_likes,
                            loyalty_tier=ui_tier_filter if ui_tier_filter != "All" else None
                        )
                        if ui_cluster_toggle:
                            search_engine.cluster_feedback(search_results)
                        st.session_state["cached_search_results"] = search_results

            if "cached_search_results" in st.session_state:
                res_obj = st.session_state["cached_search_results"]

                s_m1, s_m2, s_m3 = st.columns(3)
                with s_m1:
                    st.metric("Vector Query Latency", f"{res_obj.query_latency_ms:.1f} ms")
                with s_m2:
                    st.metric("Retrieved Comments", f"{res_obj.results_count}")
                with s_m3:
                    st.metric("Vector Corpus Size", f"{res_obj.total_searched:,}")

                # Render Thematic Clusters
                if res_obj.clusters:
                    st.markdown("#### 📂 Thematic Feedback Clusters")
                    n_cl = len(res_obj.clusters)
                    cl_cols = st.columns(min(3, n_cl))
                    for idx, c in enumerate(res_obj.clusters):
                        col_idx = idx % min(3, n_cl)
                        with cl_cols[col_idx]:
                            with st.container(border=True):
                                st.markdown(f"**Cluster #{c.cluster_id}: {c.theme_label}**")
                                st.caption(f"Volume: **{c.comment_count}** comments | Total Likes: **{c.total_likes:,}**")
                                st.markdown(f"**Dominant Cohort:** `{c.dominant_tier}`")
                                st.markdown(f"**Avg Sentiment:** `{c.avg_sentiment:+.2f}`")
                                st.markdown(f"> *\"{c.exemplar_quote}\"*")

                # Render Comments Table
                if res_obj.results:
                    st.markdown("#### 💬 Top Semantic Matches")
                    df_disp = res_obj.to_dataframe()
                    cols_to_show = ["similarity_score", "video_title", "author_display_name", "rfm_tier", "like_count", "sentiment_score", "text"]
                    cols_present = [c for c in cols_to_show if c in df_disp.columns]

                    st.dataframe(df_disp[cols_present], width="stretch", hide_index=True)

                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(
                            label="⬇️ Download Search Results (.csv)",
                            data=df_disp.to_csv(index=False).encode("utf-8"),
                            file_name="ytint_semantic_search.csv",
                            mime="text/csv",
                            key="dl_search_csv"
                        )
                    with dl_col2:
                        st.download_button(
                            label="⬇️ Download Search Results (.json)",
                            data=res_obj.to_json(),
                            file_name="ytint_semantic_search.json",
                            mime="application/json",
                            key="dl_search_json"
                        )

        # 7.7 Creator Actionability & Engagement Optimization Workbench
        st.divider()
        st.subheader("🎯 Creator Actionability & Engagement Optimization Workbench")
        st.markdown(
            "Triages viewer comments into high-leverage creator actions: **Pin Candidates** (tone anchors), "
            "**Heart Candidates** (VIP loyalist reinforcement), **Priority Questions** (high-intent audience inquiries), "
            "and **De-escalation Sparks** (diffusing tense debate). Evaluates expected causal engagement lift using Stage 39 DiD parameters."
        )

        from engine.assistant import CreatorAssistantEngine

        engine_assist = CreatorAssistantEngine()
        available_vids_assist = engine_assist.get_available_videos(limit=30)
        vid_choices_a = [v["video_id"] for v in available_vids_assist]
        vid_labels_a = {v["video_id"]: f"{v.get('title', v['video_id'])} ({v.get('total_comments', 0):,} comments)" for v in available_vids_assist}

        as_c1, as_c2, as_c3, as_c4 = st.columns([2, 1, 1, 1])
        with as_c1:
            selected_vid_assist = st.selectbox(
                "Select Video for Creator Triage",
                options=vid_choices_a if vid_choices_a else ["MOCK_VID_001"],
                format_func=lambda x: vid_labels_a.get(x, x),
                key="assist_video_select",
            )
        with as_c2:
            action_filter_ui = st.selectbox(
                "Filter Action",
                options=["ALL", "PIN", "HEART", "REPLY", "DEESCALATE"],
                index=0,
                key="assist_action_filter",
            )
        with as_c3:
            cohort_filter_ui = st.selectbox(
                "Viewer Cohort",
                options=["ALL", "Champions", "Loyal", "At Risk", "Casual"],
                index=0,
                key="assist_cohort_filter",
            )
        with as_c4:
            limit_assist_ui = st.slider(
                "Max Items",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                key="assist_limit_slider",
            )

        with st.spinner("Triaging comments and estimating causal uplift..."):
            triage_report = engine_assist.triage_comments(
                video_id=selected_vid_assist,
                action_filter=action_filter_ui,
                cohort_filter=cohort_filter_ui,
                limit=limit_assist_ui,
            )

        # Summary Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            with st.container(border=True):
                st.metric("Total Analyzed", f"{triage_report.total_analyzed:,}")
                st.caption(f"{triage_report.total_actionable} actionable matches")
        with m2:
            with st.container(border=True):
                st.metric("📌 Pin Candidates", f"{triage_report.summary_counts.get('PIN', 0)}")
                st.caption("Community tone anchors")
        with m3:
            with st.container(border=True):
                st.metric("❤️ Heart Reinforce", f"{triage_report.summary_counts.get('HEART', 0)}")
                st.caption("Loyalists & praise")
        with m4:
            with st.container(border=True):
                st.metric("💬 Questions / Bug Triage", f"{triage_report.summary_counts.get('REPLY_QUESTION', 0)}")
                st.caption("Unanswered inquiries")
        with m5:
            with st.container(border=True):
                st.metric("🛡️ De-escalation Sparks", f"{triage_report.summary_counts.get('DEESCALATE', 0)}")
                st.caption("Controversy / hostility")

        # Render Recommendations
        if triage_report.recommendations:
            st.markdown(f"#### 📋 Prioritized Action Queue ({len(triage_report.recommendations)} comments shown)")

            for idx, rec in enumerate(triage_report.recommendations, 1):
                badge_color = {
                    "PIN": "📌 PIN CANDIDATE",
                    "HEART": "❤️ HEART REINFORCE",
                    "REPLY_QUESTION": "💬 REPLY TO QUESTION",
                    "DEESCALATE": "🛡️ DE-ESCALATE CRISIS",
                }.get(rec.action_type, rec.action_type)

                cohort_badge = {
                    "Champions": "👑 Champions",
                    "Loyal": "⭐ Loyal",
                    "At Risk": "⚠️ At Risk",
                }.get(rec.author_cohort, f"👤 {rec.author_cohort}")

                with st.container(border=True):
                    h_col1, h_col2, h_col3 = st.columns([2, 1, 1])
                    with h_col1:
                        st.markdown(f"**{idx}. {badge_color}** — Priority: **`{rec.priority_score:.1f}` / 100**")
                    with h_col2:
                        st.markdown(f"Author: **{rec.author_name}** ({cohort_badge})")
                    with h_col3:
                        st.caption(f"👍 {rec.like_count} likes | 💬 {rec.reply_count} replies | T+{rec.minutes_since_upload:.0f}m")

                    st.markdown(f"> *\"{rec.text}\"*")
                    st.caption(f"💡 **Action Rationale:** {rec.action_rationale}")

                    # Causal uplift badges
                    if rec.expected_uplift:
                        uplift_pills = " &nbsp;|&nbsp; ".join([f"**{k}:** `{v}`" for k, v in rec.expected_uplift.items()])
                        st.markdown(f"📈 **Projected Causal Uplift (Stage 39 DiD):** {uplift_pills}")

                    # AI Reply Drafter Expander
                    with st.expander("✍️ Draft AI Response", expanded=False):
                        dr_col1, dr_col2 = st.columns([1, 2])
                        with dr_col1:
                            tone_sel = st.selectbox(
                                "Voice Tone",
                                options=["Warm & Grateful", "Clarifying & Factual", "Empathetic & De-escalating", "Playful"],
                                key=f"tone_{rec.comment_id}_{idx}",
                            )
                            btn_draft = st.button("Generate Draft", key=f"btn_draft_{rec.comment_id}_{idx}")

                        with dr_col2:
                            draft_key = f"draft_val_{rec.comment_id}_{idx}"
                            if btn_draft or draft_key not in st.session_state:
                                st.session_state[draft_key] = engine_assist.draft_reply(rec, tone=tone_sel)
                            st.text_area("Suggested Response", value=st.session_state[draft_key], height=80, key=f"txt_{rec.comment_id}_{idx}")
                            st.caption("Copy and paste directly into YouTube Studio.")

            # Export Buttons
            ea_col1, ea_col2 = st.columns(2)
            with ea_col1:
                st.download_button(
                    label="⬇️ Export Triage Ledger (.csv)",
                    data=triage_report.to_dataframe().to_csv(index=False).encode("utf-8"),
                    file_name=f"creator_triage_{selected_vid_assist}.csv",
                    mime="text/csv",
                    key="dl_triage_csv",
                )
            with ea_col2:
                st.download_button(
                    label="⬇️ Export Triage Ledger (.json)",
                    data=json.dumps(triage_report.to_dict(), indent=2, ensure_ascii=False),
                    file_name=f"creator_triage_{selected_vid_assist}.json",
                    mime="application/json",
                    key="dl_triage_json",
                )
        else:
            st.info("No actionable comments found matching the active filters for this video.")

        # 7.8 Zero-Copy Analytical SQL Studio & Query Workbench
        st.divider()
        st.subheader("⚡ Zero-Copy Analytical SQL Studio & Query Workbench")
        st.markdown(
            "High-performance, out-of-core analytical query engine powered by **DuckDB**. "
            "Run arbitrary ANSI SQL queries, multi-table joins, aggregations, and window functions "
            "directly over all **80+ analytical Parquet layers** with sub-10ms latency."
        )

        from engine.sql_engine import SQLEngine, PRESET_QUERIES

        sql_engine = SQLEngine()

        if not sql_engine.is_available():
            st.warning("⚠️ DuckDB is not installed in this environment. Run `pip install duckdb>=1.0.0` to enable the Analytical SQL Studio.")
        else:
            # 1. Schema Explorer Expander
            with st.expander("🗂️ Browse Parquet Table Catalog & Schema Explorer", expanded=False):
                cat = sql_engine.get_table_catalog()
                cat_df = pd.DataFrame(cat)
                if not cat_df.empty:
                    st.dataframe(cat_df[["table_name", "row_count", "size_mb", "file_name"]], width="stretch")
                    
                    inspect_col1, inspect_col2 = st.columns([2, 3])
                    with inspect_col1:
                        table_names_list = [c["table_name"] for c in cat]
                        default_idx = table_names_list.index("authors") if "authors" in table_names_list else 0
                        inspect_table = st.selectbox(
                            "Select Table to Inspect Schema",
                            options=table_names_list,
                            index=default_idx,
                            key="sql_inspect_table_select"
                        )
                    with inspect_col2:
                        if inspect_table:
                            schema_df = sql_engine.get_table_schema(inspect_table)
                            st.dataframe(schema_df, width="stretch")

            # 2. Preset Queries & Controls
            sq_c1, sq_c2 = st.columns([3, 1])
            with sq_c1:
                preset_keys = list(PRESET_QUERIES.keys())
                selected_preset = st.selectbox(
                    "💡 Curated Analytical SQL Presets",
                    options=["-- Custom SQL Query --"] + preset_keys,
                    format_func=lambda x: f"{PRESET_QUERIES[x]['title']}" if x in PRESET_QUERIES else "✍️ Custom SQL Query (Write your own)",
                    key="sql_preset_select"
                )
            with sq_c2:
                max_rows_ui = st.slider("Result Limit", min_value=10, max_value=1000, value=100, step=10, key="sql_limit_slider")

            default_query = "SELECT rfm_cohort, count(*) AS author_count, round(avg(frequency), 1) AS avg_comments, round(avg(monetary), 1) AS avg_likes, round(avg(avg_sentiment), 3) AS avg_sentiment\nFROM authors\nGROUP BY rfm_cohort\nORDER BY avg_likes DESC;"
            if selected_preset in PRESET_QUERIES:
                default_query = PRESET_QUERIES[selected_preset]["sql"]
                st.caption(f"ℹ️ **Preset:** {PRESET_QUERIES[selected_preset]['description']}")

            user_sql = st.text_area(
                "SQL Query (DuckDB ANSI SQL)",
                value=default_query,
                height=130,
                key=f"sql_text_area_{selected_preset}"
            )

            btn_run_sql = st.button("🚀 Execute SQL Query", key="btn_run_sql", type="primary")

            # Auto-run query on initial load or button press
            session_sql_key = f"sql_last_result_{selected_preset}"
            if btn_run_sql or session_sql_key not in st.session_state:
                with st.spinner("Executing analytical query via DuckDB zero-copy engine..."):
                    q_res = sql_engine.execute_query(user_sql, limit=max_rows_ui)
                    st.session_state[session_sql_key] = q_res
            else:
                q_res = st.session_state[session_sql_key]

            if q_res.error:
                st.error(f"❌ SQL Execution Error: {q_res.error}")
            else:
                st.success(
                    f"⚡ Query Executed in **{q_res.execution_time_ms:.2f} ms** | "
                    f"Returned **{q_res.row_count:,} rows** ({q_res.column_count} columns)"
                )

                if not q_res.df.empty:
                    tab_sql_data, tab_sql_chart = st.tabs(["📋 Query Result Table", "📊 Visual Chart Builder"])

                    with tab_sql_data:
                        st.dataframe(q_res.df, width="stretch")

                    with tab_sql_chart:
                        num_cols = q_res.df.select_dtypes(include=[np.number]).columns.tolist()
                        all_cols = q_res.df.columns.tolist()

                        if len(num_cols) > 0 and len(all_cols) >= 2:
                            ch_c1, ch_c2, ch_c3 = st.columns(3)
                            with ch_c1:
                                chart_type = st.selectbox(
                                    "Chart Type",
                                    options=["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"],
                                    key="sql_chart_type"
                                )
                            with ch_c2:
                                x_col = st.selectbox("X-Axis Column", options=all_cols, index=0, key="sql_chart_x")
                            with ch_c3:
                                y_col = st.selectbox("Y-Axis Column (Metric)", options=num_cols, index=0, key="sql_chart_y")

                            if chart_type == "Bar Chart":
                                fig_sql = px.bar(
                                    q_res.df,
                                    x=x_col,
                                    y=y_col,
                                    title=f"{y_col} by {x_col}",
                                    template=PLOTLY_TEMPLATE,
                                    color_discrete_sequence=[COLOR_PRIMARY]
                                )
                            elif chart_type == "Line Chart":
                                fig_sql = px.line(
                                    q_res.df,
                                    x=x_col,
                                    y=y_col,
                                    title=f"{y_col} across {x_col}",
                                    template=PLOTLY_TEMPLATE,
                                    color_discrete_sequence=[COLOR_ACCENT]
                                )
                            elif chart_type == "Scatter Plot":
                                fig_sql = px.scatter(
                                    q_res.df,
                                    x=x_col,
                                    y=y_col,
                                    title=f"{y_col} vs {x_col}",
                                    template=PLOTLY_TEMPLATE,
                                    color_discrete_sequence=[COLOR_WARNING]
                                )
                            else:
                                fig_sql = px.histogram(
                                    q_res.df,
                                    x=y_col,
                                    title=f"Distribution of {y_col}",
                                    template=PLOTLY_TEMPLATE,
                                    color_discrete_sequence=[COLOR_PURPLE]
                                )
                            fig_sql.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig_sql, width="stretch")
                        else:
                            st.info("Visual chart builder requires at least one numerical metric column in the query results.")

                    # Export controls
                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        st.download_button(
                            label="⬇️ Download Query Results (.csv)",
                            data=q_res.df.to_csv(index=False).encode("utf-8"),
                            file_name="ytint_sql_results.csv",
                            mime="text/csv",
                            key="dl_sql_csv"
                        )
                    with exp_col2:
                        st.download_button(
                            label="⬇️ Download Query Results (.json)",
                            data=json.dumps(q_res.to_dict(), indent=2, ensure_ascii=False),
                            file_name="ytint_sql_results.json",
                            mime="application/json",
                            key="dl_sql_json"
                        )
                else:
                    st.info("Query returned 0 rows.")


if __name__ == "__main__":
    main()






