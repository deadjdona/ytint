"""ytint Visualization Engine Package.

Modular publication-quality plot generators decomposed by analytical phase.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from engine.config_loader import load_config
from .theme import setup_theme, plt, sns

from .nlp_plots import (
    plot_umap_semantics,
    plot_cooccurrence_network,
    plot_ner_distribution,
    plot_sarcasm_distribution,
    plot_linguistic_features,
    plot_tag_network,
    plot_code_switching,
    plot_audience_intent,
    plot_stance_drift,
    plot_emoji_treemap,
    plot_length_vs_likes_hexbin,
    plot_language_distribution,
    plot_valence_per_topic,
    plot_slang_lexicon,
    plot_tfidf_keywords
)
from .thread_plots import (
    plot_reply_depth_distribution,
    plot_thread_polarization,
    plot_reply_latency_distribution,
    plot_position_bias,
    plot_arrival_speed,
    plot_thread_width,
    plot_resolution_patterns,
    plot_initiator_patterns,
    plot_toxicity_contagion,
    plot_thread_sunburst,
    plot_corpus_quality
)
from .author_plots import (
    plot_author_pareto,
    plot_force_directed_network,
    plot_rfm_3d,
    plot_lorenz_curve,
    plot_bipartite_network,
    plot_cocommenting_network,
    plot_frequency_tiers,
    plot_bot_heuristics,
    plot_driveby_loyalists,
    plot_like_inflation,
    plot_author_fingerprints,
    plot_impersonation,
    plot_cib_rings,
    plot_bowtie_structure,
    plot_top_k_concentration
)
from .temporal_plots import (
    plot_diurnal_heatmap,
    plot_plutchik_radar,
    plot_sentiment_ridges,
    plot_sentiment_divergence,
    plot_toxicity_heatmap,
    plot_stl_decomposition,
    plot_reaction_timeline,
    plot_anomaly_scatter,
    plot_topic_streamgraph,
    plot_shelf_life,
    plot_video_half_life,
    plot_velocity_spikes,
    plot_topic_video_matrix,
    plot_cohort_retention,
    plot_video_overlap,
    plot_new_vs_returning,
    plot_topic_cohorts,
    plot_cross_video,
    plot_topic_injection_anomalies,
    plot_minute_arrival_curve
)
from .model_plots import (
    plot_kaplan_meier,
    plot_shap_summary,
    plot_creator_uplift,
    plot_poisson_bursts,
    plot_series_vs_standalone,
    plot_creator_sentiment_polarity,
    plot_cross_modal_scene_reactions
)

def run_visualizations():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # File Paths
    authors_file = out_dir / "authors_final.parquet"
    semantics_file = interim_dir / "semantic_topics.parquet"
    survival_file = out_dir / "kaplan_meier_survival.parquet"
    shap_vals_file = out_dir / "shap_values.npy"
    shap_feat_file = out_dir / "shap_features.parquet"
    comments_file = interim_dir / "comments_clean.parquet"
    diurnal_file = out_dir / "diurnal_heatmap.parquet"
    stl_file = out_dir / "stl_decomposition.parquet"
    videos_file = out_dir / "videos_final.parquet"
    timeline_file = out_dir / "historical_timeline.parquet"
    viral_file = out_dir / "viral_events.parquet"
    
    # Phase 3 File Paths
    reaction_file = out_dir / "video_reaction_map.parquet"
    integrity_file = interim_dir / "integrity_flags.parquet"
    slopes_file = out_dir / "thread_decay_slopes.parquet"
    topic_evo_file = out_dir / "topic_evolution.parquet"
    latency_file = out_dir / "reply_latency.parquet"
    shelf_life_file = out_dir / "shelf_life_likes.parquet"
    matrix_file = out_dir / "topic_video_matrix.parquet"
    coocc_edges_file = out_dir / "word_cooccurrence_edges.parquet"
    coocc_nodes_file = out_dir / "word_cooccurrence_nodes.parquet"
    ner_file = out_dir / "named_entities.parquet"
    tag_edges_file = out_dir / "tag_network_edges.parquet"
    tag_nodes_file = out_dir / "tag_network_nodes.parquet"
    bipartite_edges_file = out_dir / "bipartite_edges.parquet"
    bipartite_nodes_file = out_dir / "bipartite_nodes.parquet"
    cocomment_edges_file = out_dir / "cocomment_edges.parquet"
    cocomment_nodes_file = out_dir / "cocomment_nodes.parquet"
    cohort_retention_file = out_dir / "cohort_retention.parquet"
    cohort_retention_pct_file = out_dir / "cohort_retention_pct.parquet"
    new_vs_returning_file = out_dir / "new_vs_returning.parquet"
    video_overlap_file = out_dir / "video_overlap_matrix.parquet"
    topic_cohorts_file = out_dir / "topic_cohorts.parquet"
    frequency_tiers_file = out_dir / "frequency_tiers.parquet"
    bot_file = out_dir / "bot_classifications.parquet"
    driveby_loyalists_file = out_dir / "driveby_loyalists.parquet"
    position_bias_file = out_dir / "position_bias.parquet"
    arrival_speed_file = out_dir / "arrival_speed.parquet"
    thread_width_file = out_dir / "thread_width_dist.parquet"
    resolution_file = out_dir / "resolution_patterns.parquet"
    initiator_file = out_dir / "initiator_patterns.parquet"
    code_switching_file = out_dir / "code_switching_impact.parquet"
    video_radar_file = out_dir / "video_profile_radar.parquet"
    controversy_file = out_dir / "controversy_impact.parquet"
    inflation_file = out_dir / "like_inflation.parquet"
    fingerprint_file = out_dir / "author_fingerprints.parquet"
    impersonation_file = out_dir / "impersonation_detection.parquet"
    intent_summary_file = out_dir / "audience_intent_summary.parquet"
    cib_rings_file = out_dir / "cib_rings.parquet"
    troll_catalysts_file = out_dir / "troll_catalysts.parquet"
    creator_uplift_file = out_dir / "creator_causal_uplift.parquet"
    stance_summary_file = out_dir / "stance_summary.parquet"
    stance_drift_file = out_dir / "stance_depth_drift.parquet"
    bursts_file = out_dir / "poisson_bursts.parquet"
    emoji_file = out_dir / "emoji_signatures.parquet"
    quality_file = out_dir / "corpus_quality.parquet"
    slang_file = out_dir / "slang_lexicon_frequency.parquet"
    tfidf_file = out_dir / "tfidf_keywords.parquet"
    bowtie_file = out_dir / "network_bowtie_structure.parquet"
    concentration_file = out_dir / "top_k_concentration.parquet"
    series_file = out_dir / "series_vs_standalone.parquet"
    creator_pol_file = out_dir / "creator_sentiment_polarity.parquet"
    reactions_file = out_dir / "cross_modal_scene_reactions.parquet"
    injection_file = out_dir / "topic_injection_anomalies.parquet"
    minute_curve_file = out_dir / "minute_arrival_curve.parquet"
    
    print(f"🎨 Initializing Visualization Engine (Outputting to {plots_dir})...")
    
    # === Original Charts ===
    
    # 1. Temporal Survival Plot
    if survival_file.exists():
        df_surv = pd.read_parquet(survival_file)
        plot_kaplan_meier(df_surv, plots_dir)
        
    # 1b. Poisson Arrival Bursts
    if bursts_file.exists():
        plot_poisson_bursts(plots_dir, bursts_file)
        
    # 2. Semantic Cluster Scatter/Hexbin
    if semantics_file.exists():
        df_sem = pd.read_parquet(semantics_file)
        plot_umap_semantics(df_sem, plots_dir)
        
    # 3. Author Power-Law Pareto
    if authors_file.exists():
        df_auth = pd.read_parquet(authors_file)
        plot_author_pareto(df_auth, plots_dir)
        
    # 4. XGBoost Feature Importance
    if shap_vals_file.exists() and shap_feat_file.exists():
        shap_vals = np.load(shap_vals_file)
        shap_feat = pd.read_parquet(shap_feat_file)
        plot_shap_summary(shap_vals, shap_feat, plots_dir)
    
    # === NEW Phase 2 Charts ===
    
    # Load comments once for all comment-based visualizations
    df_comments = None
    if comments_file.exists():
        df_comments = pd.read_parquet(comments_file)
    
    # 5. Diurnal Activity Heatmap (24×7)
    if diurnal_file.exists():
        plot_diurnal_heatmap(plots_dir, diurnal_file)
    
    # 6. Plutchik Emotion Radar
    if df_comments is not None:
        plot_plutchik_radar(df_comments, plots_dir)
    
    # 7. Sentiment Ridge Plot
    if df_comments is not None and 'vader_compound' in df_comments.columns:
        plot_sentiment_ridges(df_comments, plots_dir)
    
    # 8. Force-Directed Network
    plot_force_directed_network(interim_dir, plots_dir)
    
    # 9. RFM 3D Scatter
    if authors_file.exists():
        df_auth = pd.read_parquet(authors_file)
        plot_rfm_3d(df_auth, plots_dir)
    
    # 10. Sentiment Divergence
    if df_comments is not None:
        plot_sentiment_divergence(df_comments, plots_dir)
    
    # 11. Toxicity Heatmap
    if df_comments is not None:
        plot_toxicity_heatmap(df_comments, plots_dir)
    
    # 12. Reply Depth Distribution
    if df_comments is not None:
        plot_reply_depth_distribution(df_comments, plots_dir)
    
    # 13. Lorenz Curve
    if authors_file.exists():
        df_auth = pd.read_parquet(authors_file)
        plot_lorenz_curve(df_auth, plots_dir)
    
    # 14. STL Decomposition
    if stl_file.exists():
        plot_stl_decomposition(plots_dir, stl_file)
        
    # === NEW Phase 3 Charts (s07-s15) ===
    if reaction_file.exists():
        plot_reaction_timeline(plots_dir, reaction_file)
    if integrity_file.exists() and comments_file.exists():
        plot_anomaly_scatter(plots_dir, integrity_file, comments_file)
    if slopes_file.exists():
        plot_thread_polarization(plots_dir, slopes_file)
    if topic_evo_file.exists():
        plot_topic_streamgraph(plots_dir, topic_evo_file)
    if latency_file.exists():
        plot_reply_latency_distribution(plots_dir, latency_file)
    if shelf_life_file.exists():
        plot_shelf_life(plots_dir, shelf_life_file)
    if videos_file.exists():
        plot_video_half_life(plots_dir, videos_file)
    if timeline_file.exists() and viral_file.exists():
        plot_velocity_spikes(plots_dir, timeline_file, viral_file)
    if matrix_file.exists():
        plot_topic_video_matrix(plots_dir, matrix_file)
    if coocc_edges_file.exists() and coocc_nodes_file.exists():
        plot_cooccurrence_network(plots_dir, coocc_edges_file, coocc_nodes_file)
    if ner_file.exists():
        plot_ner_distribution(plots_dir, ner_file)
    if comments_file.exists():
        plot_sarcasm_distribution(plots_dir, comments_file)
        plot_linguistic_features(plots_dir, comments_file)
    if tag_edges_file.exists() and tag_nodes_file.exists():
        plot_tag_network(plots_dir, tag_edges_file, tag_nodes_file)
    if bipartite_edges_file.exists() and bipartite_nodes_file.exists():
        plot_bipartite_network(plots_dir, bipartite_edges_file, bipartite_nodes_file)
    if cocomment_edges_file.exists() and cocomment_nodes_file.exists():
        plot_cocommenting_network(plots_dir, cocomment_edges_file, cocomment_nodes_file)
    if cohort_retention_file.exists() and cohort_retention_pct_file.exists():
        plot_cohort_retention(plots_dir, cohort_retention_pct_file, cohort_retention_file)
    if new_vs_returning_file.exists():
        plot_new_vs_returning(plots_dir, new_vs_returning_file)
    if video_overlap_file.exists():
        plot_video_overlap(plots_dir, video_overlap_file)
    if topic_cohorts_file.exists():
        plot_topic_cohorts(plots_dir, topic_cohorts_file)
    if frequency_tiers_file.exists():
        plot_frequency_tiers(plots_dir, frequency_tiers_file)
    if bot_file.exists():
        plot_bot_heuristics(plots_dir, bot_file)
    if driveby_loyalists_file.exists():
        plot_driveby_loyalists(plots_dir, driveby_loyalists_file)
    if position_bias_file.exists():
        plot_position_bias(plots_dir, position_bias_file)
    if arrival_speed_file.exists():
        plot_arrival_speed(plots_dir, arrival_speed_file)
    if thread_width_file.exists():
        plot_thread_width(plots_dir, thread_width_file)
    if resolution_file.exists():
        plot_resolution_patterns(plots_dir, resolution_file)
    if initiator_file.exists():
        plot_initiator_patterns(plots_dir, initiator_file)
    if code_switching_file.exists():
        plot_code_switching(plots_dir, code_switching_file)
    if video_radar_file.exists() or controversy_file.exists():
        plot_cross_video(plots_dir, video_radar_file, controversy_file)
    if inflation_file.exists():
        plot_like_inflation(plots_dir, inflation_file)
    if fingerprint_file.exists():
        plot_author_fingerprints(plots_dir, fingerprint_file)
    if impersonation_file.exists():
        plot_impersonation(plots_dir, impersonation_file)
    if intent_summary_file.exists():
        plot_audience_intent(plots_dir, intent_summary_file)
    if cib_rings_file.exists():
        plot_cib_rings(plots_dir, cib_rings_file)
    if troll_catalysts_file.exists():
        plot_toxicity_contagion(plots_dir, troll_catalysts_file)
    if creator_uplift_file.exists():
        plot_creator_uplift(plots_dir, creator_uplift_file)
    if stance_summary_file.exists() and stance_drift_file.exists():
        plot_stance_drift(plots_dir, stance_summary_file, stance_drift_file)

    # === NEW Phase 6 Charts ===
    # Emoji treemap
    if emoji_file.exists():
        plot_emoji_treemap(plots_dir, emoji_file)
    # Length vs likes hexbin
    if df_comments is not None:
        plot_length_vs_likes_hexbin(df_comments, plots_dir)
        plot_language_distribution(df_comments, plots_dir)
        plot_valence_per_topic(df_comments, plots_dir)
    # Thread sunburst
    if comments_file.exists():
        plot_thread_sunburst(plots_dir, comments_file)
    # Corpus quality log-log
    if quality_file.exists():
        plot_corpus_quality(plots_dir, quality_file)
    # Slang & TF-IDF
    if slang_file.exists():
        df_slang = pd.read_parquet(slang_file)
        plot_slang_lexicon(df_slang, plots_dir)
    if tfidf_file.exists():
        df_tfidf = pd.read_parquet(tfidf_file)
        plot_tfidf_keywords(df_tfidf, plots_dir)
    # Bow-Tie & Concentration
    if bowtie_file.exists():
        df_bt = pd.read_parquet(bowtie_file)
        plot_bowtie_structure(df_bt, plots_dir)
    if concentration_file.exists():
        df_conc = pd.read_parquet(concentration_file)
        plot_top_k_concentration(df_conc, plots_dir)
    # Topic Injection & Arrival Velocity
    if injection_file.exists():
        df_inj = pd.read_parquet(injection_file)
        plot_topic_injection_anomalies(df_inj, plots_dir)
    if minute_curve_file.exists():
        df_curv = pd.read_parquet(minute_curve_file)
        plot_minute_arrival_curve(df_curv, plots_dir)
    # Series, Creator Sentiment, Scene Reactions
    if series_file.exists():
        df_ser = pd.read_parquet(series_file)
        plot_series_vs_standalone(df_ser, plots_dir)
    if creator_pol_file.exists():
        df_creat = pd.read_parquet(creator_pol_file)
        plot_creator_sentiment_polarity(df_creat, plots_dir)
    if reactions_file.exists():
        df_rxn = pd.read_parquet(reactions_file)
        plot_cross_modal_scene_reactions(df_rxn, plots_dir)

    print("\u2705 Stage 99 Visualizations Complete! \U0001f386")

__all__ = [
    "run_visualizations",
    "setup_theme",
    "plot_umap_semantics",
    "plot_cooccurrence_network",
    "plot_ner_distribution",
    "plot_sarcasm_distribution",
    "plot_linguistic_features",
    "plot_tag_network",
    "plot_code_switching",
    "plot_audience_intent",
    "plot_stance_drift",
    "plot_reply_depth_distribution",
    "plot_thread_polarization",
    "plot_reply_latency_distribution",
    "plot_position_bias",
    "plot_arrival_speed",
    "plot_thread_width",
    "plot_resolution_patterns",
    "plot_initiator_patterns",
    "plot_toxicity_contagion",
    "plot_author_pareto",
    "plot_force_directed_network",
    "plot_rfm_3d",
    "plot_lorenz_curve",
    "plot_bipartite_network",
    "plot_cocommenting_network",
    "plot_frequency_tiers",
    "plot_bot_heuristics",
    "plot_driveby_loyalists",
    "plot_like_inflation",
    "plot_author_fingerprints",
    "plot_impersonation",
    "plot_cib_rings",
    "plot_diurnal_heatmap",
    "plot_plutchik_radar",
    "plot_sentiment_ridges",
    "plot_sentiment_divergence",
    "plot_toxicity_heatmap",
    "plot_stl_decomposition",
    "plot_reaction_timeline",
    "plot_anomaly_scatter",
    "plot_topic_streamgraph",
    "plot_shelf_life",
    "plot_video_half_life",
    "plot_velocity_spikes",
    "plot_topic_video_matrix",
    "plot_cohort_retention",
    "plot_video_overlap",
    "plot_new_vs_returning",
    "plot_topic_cohorts",
    "plot_cross_video",
    "plot_kaplan_meier",
    "plot_shap_summary",
    "plot_creator_uplift",
    "plot_poisson_bursts",
    "plot_emoji_treemap",
    "plot_length_vs_likes_hexbin",
    "plot_language_distribution",
    "plot_valence_per_topic",
    "plot_thread_sunburst",
    "plot_corpus_quality",
    "plot_slang_lexicon",
    "plot_tfidf_keywords",
    "plot_bowtie_structure",
    "plot_top_k_concentration",
    "plot_topic_injection_anomalies",
    "plot_minute_arrival_curve",
    "plot_series_vs_standalone",
    "plot_creator_sentiment_polarity",
    "plot_cross_modal_scene_reactions"
]
