"""Stage 99: Publication Visualizations (s99_visualize.py)

Thin backwards-compatible wrapper re-exporting the modularized
visualization submodules from `pipeline.visualizations`.
"""

import sys
from pathlib import Path

# Re-export all functions from pipeline.visualizations package
from pipeline.visualizations import (
    run_visualizations,
    setup_theme,
    # Phase 1: NLP & Semantics
    plot_umap_semantics,
    plot_cooccurrence_network,
    plot_ner_distribution,
    plot_sarcasm_distribution,
    plot_linguistic_features,
    plot_tag_network,
    plot_code_switching,
    plot_audience_intent,
    plot_stance_drift,
    # Phase 2: Conversation Tree & Thread Dynamics
    plot_reply_depth_distribution,
    plot_thread_polarization,
    plot_reply_latency_distribution,
    plot_position_bias,
    plot_arrival_speed,
    plot_thread_width,
    plot_resolution_patterns,
    plot_initiator_patterns,
    plot_toxicity_contagion,
    # Phase 3: Author Profiling & Forensics
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
    # Phase 4: Video Dynamics, Cohorts & Longitudinal Topology
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
    plot_topic_video_matrix,
    plot_cohort_retention,
    plot_video_overlap,
    plot_new_vs_returning,
    plot_topic_cohorts,
    plot_cross_video,
    # Phase 5: Predictive Modeling & Causal Interventions
    plot_kaplan_meier,
    plot_shap_summary,
    plot_creator_uplift,
)

if __name__ == "__main__":
    run_visualizations()
