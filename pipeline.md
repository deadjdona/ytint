# 🛠️ ytint Processing Engine & Intelligence Pipeline

The **ytint** pipeline is a high-performance, multi-stage analytical intelligence suite designed to extract forensic, behavioral, topological, and causal insights from YouTube community conversations.

---

## 📋 Executive Summary

- **Total Analytical Layers**: 48 discrete stages structured across 7 logical phases (s00–s46 + s99).
- **1:1 Canonical File Mapping**: Every stage script in `src/pipeline/` is physically named after its canonical stage ID (`s00_ingest.py` through `s40_synthesis.py`, plus `s99_visualize.py`).
- **Strict Dependency Ordering**: Foundational data ingestion and NLP enrichment execute first, followed by conversational topology, author forensics, longitudinal video dynamics, predictive modeling, and finally publication visualizations (`s99_visualize.py`).
- **Inter-Step Upstream Data Reuse**: Downstream forensic and segmentation layers directly utilize precomputed tables (`authors_final.parquet`, `videos_final.parquet`, `topic_metadata.parquet`, `bot_classifications.parquet`) to eliminate redundant compute.
- **Runner Execution**:

  ```bash
  # Execute full end-to-end pipeline sweep (with automatic caching)
  python -m pipeline.runner

  # Force execute a specific stage
  python -m pipeline.runner --stage s07
  python -m pipeline.runner --stage s25

  # Cascade execute sequentially from a specific stage onward
  python -m pipeline.runner --from-stage s19
  ```

---

## 🗺️ Pipeline Stages Reference Table

| Phase                            | Stage ID | Script Filename                                                                                                    | Stage Name                           | Primary Outputs                                         |
| :------------------------------- | :------- | :----------------------------------------------------------------------------------------------------------------- | :----------------------------------- | :------------------------------------------------------ |
| **Phase 1: NLP & Semantics**     | `s00`    | [s00_ingest.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s00_ingest.py)                                   | Ingestion & Migration                | `comments_clean.parquet`, `videos_clean.parquet`        |
|                                  | `s01`    | [s01_enrich.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s01_enrich.py)                                   | NLP & Sentiment Enrichment           | `.s01_complete` (enriched comments)                     |
|                                  | `s02`    | [s02_topics.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s02_topics.py)                                   | Topic Modeling (BERTopic)            | `topic_metadata.parquet`                                |
|                                  | `s03`    | [s03_ner.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s03_ner.py)                                         | Named Entity Recognition (NER)       | `named_entities.parquet`                                |
|                                  | `s04`    | [s04_cooccurrence.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s04_cooccurrence.py)                       | Word Co-occurrence Network           | `word_cooccurrence_edges.parquet`, `_nodes.parquet`     |
|                                  | `s05`    | [s05_tags_mentions.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s05_tags_mentions.py)                     | Hashtags & Mention Networks          | `tag_network_edges.parquet`, `_nodes.parquet`           |
|                                  | `s06`    | [s06_audience_intent.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s06_audience_intent.py)                 | Audience Demand Intent Mining        | `audience_intent_summary.parquet`, `audience_content_requests.parquet` |
|                                  | `s07`    | [s07_stance_drift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s07_stance_drift.py)                       | Target Stance & Polarization Drift   | `stance_summary.parquet`, `stance_depth_drift.parquet`, `polarized_threads.parquet` |
|                                  | `s08`    | [s08_code_switching.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s08_code_switching.py)                   | Code-Switching Language Impact       | `code_switching_impact.parquet`                         |
| **Phase 2: Thread Dynamics**     | `s09`    | [s09_thread_width.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s09_thread_width.py)                       | Thread Width & Branching Factor      | `thread_width_dist.parquet`                             |
|                                  | `s10`    | [s10_reply_latency.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s10_reply_latency.py)                     | Reply Latency Distribution           | `reply_latency.parquet`                                 |
|                                  | `s11`    | [s11_position_bias.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s11_position_bias.py)                     | Position Bias & Early Mover Effect   | `position_bias.parquet`                                 |
|                                  | `s12`    | [s12_resolution_patterns.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s12_resolution_patterns.py)         | Conversation Resolution Patterns     | `resolution_patterns.parquet`                           |
|                                  | `s13`    | [s13_initiator_patterns.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s13_initiator_patterns.py)           | Initiator / Responder Dynamics       | `initiator_patterns.parquet`                            |
|                                  | `s14`    | [s14_polarization.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s14_polarization.py)                       | Thread Polarization Dynamics         | `thread_polarization_corpus.parquet`                    |
|                                  | `s15`    | [s15_toxicity_contagion.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s15_toxicity_contagion.py)           | Toxicity Contagion & Troll Catalysts | `toxicity_contagion_summary.parquet`, `troll_catalysts.parquet`, `video_toxicity_contagion.parquet` |
| **Phase 3: Author Forensics**    | `s16`    | [s16_network.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s16_network.py)                                 | Author Reply Network Graph           | `authors_network_metrics.parquet`                       |
|                                  | `s17`    | [s17_bipartite.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s17_bipartite.py)                             | Author-Video Bipartite Graph         | `bipartite_edges.parquet`, `bipartite_nodes.parquet`    |
|                                  | `s18`    | [s18_cocommenting.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s18_cocommenting.py)                       | Co-commenting Graph                  | `cocomment_edges.parquet`, `cocomment_nodes.parquet`    |
|                                  | `s19`    | [s19_aggregation.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s19_aggregation.py)                         | Author & Video Aggregations (RFM)    | `authors_final.parquet`, `videos_final.parquet`         |
|                                  | `s20`    | [s20_frequency_tiers.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s20_frequency_tiers.py)                 | Commenting Frequency Tiers           | `frequency_tiers.parquet`                               |
|                                  | `s21`    | [s21_driveby_loyalists.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s21_driveby_loyalists.py)             | Drive-by vs Loyalists Segmentation   | `driveby_loyalists.parquet`                             |
|                                  | `s22`    | [s22_author_fingerprints.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s22_author_fingerprints.py)         | Author Stylometric Fingerprints      | `author_fingerprints.parquet`                           |
|                                  | `s23`    | [s23_impersonation_detection.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s23_impersonation_detection.py) | Creator Impersonation Detection      | `impersonation_detection.parquet`                       |
|                                  | `s24`    | [s24_bot_heuristics.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s24_bot_heuristics.py)                   | Bot & Spammer Heuristics             | `bot_classifications.parquet`                           |
|                                  | `s25`    | [s25_cib_detection.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s25_cib_detection.py)                     | Coordinated Inauthentic Rings (CIB)  | `cib_rings.parquet`, `cib_coordinated_comments.parquet` |
|                                  | `s26`    | [s26_integrity.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s26_integrity.py)                             | Integrity & Spam Flags               | `integrity_flags.parquet`                               |
|                                  | `s27`    | [s27_anomaly_inflation.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s27_anomaly_inflation.py)             | Suspicious Like Inflation            | `like_inflation.parquet`                                |
| **Phase 4: Temporal & Cohorts**  | `s28`    | [s28_narrative.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s28_narrative.py)                             | Narrative Timeline & Change-Points   | `historical_timeline.parquet`, `viral_events.parquet`   |
|                                  | `s29`    | [s29_cross_modal.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s29_cross_modal.py)                         | Cross-Modal Video Reaction Map       | `video_reaction_map.parquet`                            |
|                                  | `s30`    | [s30_shelf_life.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s30_shelf_life.py)                           | Shelf Life of Likes & Video Decay    | `shelf_life_likes.parquet`                              |
|                                  | `s31`    | [s31_topic_evolution.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s31_topic_evolution.py)                 | Longitudinal Topic Evolution         | `topic_evolution.parquet`                               |
|                                  | `s32`    | [s32_topic_matrix.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s32_topic_matrix.py)                       | Topic × Video Heatmap Matrix         | `topic_video_matrix.parquet`                            |
|                                  | `s33`    | [s33_cohorts.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s33_cohorts.py)                                 | Acquisition Cohorts & Retention      | `cohort_retention.parquet`, `cohort_retention_pct.parquet`, `new_vs_returning.parquet` |
|                                  | `s34`    | [s34_overlap.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s34_overlap.py)                                 | Cross-Video Audience Overlap         | `video_overlap_matrix.parquet`                          |
|                                  | `s35`    | [s35_topic_cohorts.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s35_topic_cohorts.py)                     | Topic-Cohort Affinity Clustering     | `topic_cohorts.parquet`                                 |
|                                  | `s36`    | [s36_arrival_speed.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s36_arrival_speed.py)                     | Arrival Speed Dynamics               | `arrival_speed.parquet`                                 |
|                                  | `s37`    | [s37_cross_video_comparisons.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s37_cross_video_comparisons.py) | Comparative Video Radar Profiles     | `video_profile_radar.parquet`                           |
| **Phase 5: Predictive & Causal** | `s38`    | [s38_modeling.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s38_modeling.py)                               | XGBoost, Tree SHAP & Kaplan-Meier    | `xgboost_like_predictor.pkl`, `shap_values.npy`, `shap_features.parquet`, `kaplan_meier_survival.parquet`, `stl_decomposition.parquet`, `diurnal_heatmap.parquet`, `poisson_bursts.parquet`, `engagement_forecast.parquet`, `power_law_fit.parquet`, `toxicity_predictor.pkl`, `return_propensity.pkl`, `viral_comment_predictor.pkl` |
|                                  | `s39`    | [s39_creator_uplift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s39_creator_uplift.py)                   | Creator Causal Uplift (DiD)          | `creator_causal_uplift.parquet`, `creator_intervention_threads.parquet` |
|                                  | `s40`    | [s40_synthesis.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s40_synthesis.py)                             | UI Metric Synthesis                  | Updates `topic_metadata.parquet` in-place               |
| **Phase 5b: Extended Analysis**  | `s41`    | [s41_tfidf_keywords.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s41_tfidf_keywords.py)                   | TF-IDF Keyword Extraction            | `tfidf_keywords.parquet`                                |
|                                  | `s42`    | [s42_polarity_engagement.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s42_polarity_engagement.py)         | Polarity vs Engagement Analysis      | `polarity_engagement.parquet`                           |
|                                  | `s43`    | [s43_emoji_signatures.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s43_emoji_signatures.py)               | Emoji Signatures & Sentiment Mapping | `emoji_signatures.parquet`                              |
|                                  | `s44`    | [s44_sentiment_anomalies.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s44_sentiment_anomalies.py)         | Sentiment Anomaly Detection          | `sentiment_anomalies.parquet`                           |
|                                  | `s45`    | [s45_corpus_quality.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s45_corpus_quality.py)                   | Meta & Corpus Quality Analysis       | `corpus_quality.parquet`, `language_coverage.parquet`   |
|                                  | `s46`    | [s46_thread_topic_drift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s46_thread_topic_drift.py)           | Within-Thread Topic Drift            | `thread_topic_drift.parquet`, `video_topic_drift_summary.parquet` |
| **Phase 6: Visualizations**      | `s99`    | [s99_visualize.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s99_visualize.py)                             | Publication Visualizations Gallery   | Renders all 61+ plots in `data/output/plots/`           |

---

## 🔍 Detailed Stage Breakdown

### Phase 1: Ingestion & Foundational NLP / Semantics

#### `s00`: Raw SQLite Data Ingestion & Synthesis ([s00_ingest.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s00_ingest.py))

- **Purpose**: Ingests raw data from SQLite databases, cleans text, standardizes column formats, and writes high-efficiency compressed Parquet tables.
- **Inputs**: `data/raw/commentsuite.sqlite3`
- **Outputs**: `data/interim/comments_clean.parquet`, `data/interim/videos_clean.parquet`

#### `s01`: Deep Sentiment & Linguistic Enrichment ([s01_enrich.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s01_enrich.py))

- **Purpose**: Computes multi-dimensional NLP features for every comment.
- **Inputs**: `comments_clean.parquet`, `videos_clean.parquet`
- **Outputs**: Enriches `comments_clean.parquet`, writes `.s01_complete`

#### `s02`: Topic Modeling ([s02_topics.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s02_topics.py))

- **Purpose**: Extracts latent conversational themes across comments using BERTopic and sentence embeddings.
- **Inputs**: `comments_clean.parquet`
- **Outputs**: `data/output/topic_metadata.parquet` (also updates `comments_clean.parquet` in-place with `topic` column)

#### `s03`: Named Entity Extraction ([s03_ner.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s03_ner.py))

- **Purpose**: Identifies named entities (People, Organizations, Locations, Products, Events) mentioned across the corpus.
- **Inputs**: `comments_clean.parquet`
- **Outputs**: `data/output/named_entities.parquet`

#### `s04`: Word Co-occurrence Network ([s04_cooccurrence.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s04_cooccurrence.py))

- **Purpose**: Constructs a semantic graph of terms that frequently co-occur within the same comment context.
- **Inputs**: `comments_clean.parquet`
- **Outputs**: `data/output/word_cooccurrence_edges.parquet`, `word_cooccurrence_nodes.parquet`

#### `s05`: Hashtag & Mention Networks ([s05_tags_mentions.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s05_tags_mentions.py))

- **Purpose**: Analyzes social tagging behavior and user-to-user `@` mentions.
- **Inputs**: `comments_clean.parquet`
- **Outputs**: `data/output/tag_network_edges.parquet`, `tag_network_nodes.parquet`

#### `s06`: Audience Demand & Content Intent Mining ([s06_audience_intent.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s06_audience_intent.py))

- **Purpose**: Classifies comments into strategic actionable intents: Video Requests, Technical Questions, Praise/Appreciation, Bug Reports, and Debates.
- **Inputs**: `comments_clean.parquet`
- **Outputs**: `data/output/audience_intent_summary.parquet`, `audience_content_requests.parquet`

#### `s07`: Target-Specific Stance Detection & Polarization Drift ([s07_stance_drift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s07_stance_drift.py))

- **Purpose**: Classifies commenter stance (**Favor**, **Against**, **Neutral**) toward key entities and evaluates ideological drift across conversation tree depth.
- **Inputs**: `comments_clean.parquet`
- **Outputs**: `data/output/stance_summary.parquet`, `stance_depth_drift.parquet`, `polarized_threads.parquet`

#### `s08`: Code-Switching & Multilingual Detection ([s08_code_switching.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s08_code_switching.py))

- **Purpose**: Detects bilingual text mixing and quantifies its engagement impact.
- **Inputs**: `comments_clean.parquet`
- **Outputs**: `data/output/code_switching_impact.parquet`

---

### Phase 2: Conversation Tree & Thread Dynamics

#### `s09`: Thread Width & Branching Factor ([s09_thread_width.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s09_thread_width.py))

- **Purpose**: Maps the structural geometry of reply trees.
- **Outputs**: `data/output/thread_width_dist.parquet`

#### `s10`: Reply Latency Distribution ([s10_reply_latency.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s10_reply_latency.py))

- **Purpose**: Analyzes response speed and temporal decay of thread conversations.
- **Outputs**: `data/output/reply_latency.parquet`

#### `s11`: Position Bias & Early Mover Advantage ([s11_position_bias.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s11_position_bias.py))

- **Purpose**: Quantifies the advantage gained by posting immediately after video upload versus later arrival.
- **Outputs**: `data/output/position_bias.parquet`

#### `s12`: Conversation Resolution Patterns ([s12_resolution_patterns.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s12_resolution_patterns.py))

- **Purpose**: Determines whether online debates end in consensus, agreement, mutual hostility, or abandonment.
- **Outputs**: `data/output/resolution_patterns.parquet`

#### `s13`: Initiator / Responder Dynamics ([s13_initiator_patterns.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s13_initiator_patterns.py))

- **Purpose**: Contrasts the linguistic, emotional, and toxicity profiles of thread starters versus responders.
- **Outputs**: `data/output/initiator_patterns.parquet`

#### `s14`: Thread Polarization Dynamics ([s14_polarization.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s14_polarization.py))

- **Purpose**: Identifies highly contentious threads characterized by bimodal sentiment distributions.
- **Outputs**: `data/output/thread_polarization_corpus.parquet`

#### `s15`: Toxicity Contagion & Troll Catalyst Identification ([s15_toxicity_contagion.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s15_toxicity_contagion.py))

- **Purpose**: Models the epidemiological spread of toxicity across reply trees ($R_0$) and identifies catalyst comments.
- **Outputs**: `data/output/toxicity_contagion_summary.parquet`, `data/output/troll_catalysts.parquet`, `data/output/video_toxicity_contagion.parquet`

---

### Phase 3: Author Profiling, Network Graphs & Behavioral Forensics

#### `s16`: Author Network & Graph Construction ([s16_network.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s16_network.py))

- **Purpose**: Builds directed author interaction graphs from reply connections (PageRank, Louvain).
- **Outputs**: `data/interim/authors_network_metrics.parquet`

#### `s17`: Author-Video Bipartite Graph ([s17_bipartite.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s17_bipartite.py))

- **Purpose**: Maps two-mode network connections between authors and videos.
- **Outputs**: `data/output/bipartite_edges.parquet`, `bipartite_nodes.parquet`

#### `s18`: Co-commenting Graph ([s18_cocommenting.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s18_cocommenting.py))

- **Purpose**: Uncovers implicit audience communities via Jaccard similarity across author repertoires.
- **Outputs**: `data/output/cocomment_edges.parquet`, `cocomment_nodes.parquet`

#### `s19`: Author & Video Aggregations (RFM) ([s19_aggregation.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s19_aggregation.py))

- **Purpose**: Core aggregation layer producing comprehensive author-level profiles (RFM) and video-level summaries.
- **Outputs**: `data/output/authors_final.parquet`, `data/output/videos_final.parquet`

#### `s20`: Commenting Frequency Tiers ([s20_frequency_tiers.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s20_frequency_tiers.py))

- **Purpose**: Segments the audience into volume frequency tiers.
- **Outputs**: `data/output/frequency_tiers.parquet`

#### `s21`: Drive-by vs Loyalists Segmentation ([s21_driveby_loyalists.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s21_driveby_loyalists.py))

- **Purpose**: Measures channel audience breadth by classifying commenters by unique videos engaged.
- **Outputs**: `data/output/driveby_loyalists.parquet`

#### `s22`: Author Stylometric Fingerprints ([s22_author_fingerprints.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s22_author_fingerprints.py))

- **Purpose**: Extracts stylometric and behavioral radar profiles for top active community contributors.
- **Outputs**: `data/output/author_fingerprints.parquet`

#### `s23`: Creator Impersonation Detection ([s23_impersonation_detection.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s23_impersonation_detection.py))

- **Purpose**: Detects scam/impersonation accounts sharing creator display names.
- **Outputs**: `data/output/impersonation_detection.parquet`

#### `s24`: Bot & Spammer Heuristics Classifier ([s24_bot_heuristics.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s24_bot_heuristics.py))

- **Purpose**: Identifies automated accounts using behavioral, textual, and graph-topological signals.
- **Outputs**: `data/output/bot_classifications.parquet`

#### `s25`: Coordinated Inauthentic Behavior (CIB) Rings ([s25_cib_detection.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s25_cib_detection.py))

- **Purpose**: Uncovers astroturfing rings posting in tight temporal synchronization ($\Delta t \le 120\text{s}$).
- **Outputs**: `data/output/cib_rings.parquet`, `cib_coordinated_comments.parquet`

#### `s26`: Integrity & Spam Flags ([s26_integrity.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s26_integrity.py))

- **Purpose**: Scans comments for MinHash LSH spam clusters, commercial spam patterns, and temporal burst brigading.
- **Outputs**: `data/interim/integrity_flags.parquet`

#### `s27`: Suspicious Like Inflation & Astroturfing ([s27_anomaly_inflation.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s27_anomaly_inflation.py))

- **Purpose**: Detects astroturfed comments with anomalous like-to-reply inflation ratios.
- **Outputs**: `data/output/like_inflation.parquet`

---

### Phase 4: Video Dynamics, Cohorts & Longitudinal Topology

#### `s28`: Narrative Timeline & Change-Points ([s28_narrative.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s28_narrative.py))

- **Purpose**: Constructs historical timeline and detects PELT structural change-points and viral anomalies ($\ge 2.5\sigma$).
- **Outputs**: `data/output/historical_timeline.parquet`, `data/output/viral_events.parquet`

#### `s29`: Cross-Modal Video Reaction Map ([s29_cross_modal.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s29_cross_modal.py))

- **Purpose**: Maps comment timestamps to exact video runtimes (MM:SS) to identify high-reaction scenes.
- **Outputs**: `data/output/video_reaction_map.parquet`

#### `s30`: Shelf Life of Likes & Video Decay ([s30_shelf_life.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s30_shelf_life.py))

- **Purpose**: Models exponential decay of engagement and like accrual ($t_{1/2}$).
- **Outputs**: `data/output/shelf_life_likes.parquet`

#### `s31`: Longitudinal Topic Evolution ([s31_topic_evolution.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s31_topic_evolution.py))

- **Purpose**: Tracks how audience topical interests wax and wane across calendar months.
- **Outputs**: `data/output/topic_evolution.parquet`

#### `s32`: Topic × Video Heatmap Matrix ([s32_topic_matrix.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s32_topic_matrix.py))

- **Purpose**: Maps thematic concentration of topics across individual video uploads.
- **Outputs**: `data/output/topic_video_matrix.parquet`

#### `s33`: User Acquisition Cohorts & Retention ([s33_cohorts.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s33_cohorts.py))

- **Purpose**: Measures long-term audience retention across monthly acquisition cohorts.
- **Outputs**: `data/output/cohort_retention.parquet`, `data/output/cohort_retention_pct.parquet`, `data/output/new_vs_returning.parquet`

#### `s34`: Cross-Video Audience Overlap ([s34_overlap.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s34_overlap.py))

- **Purpose**: Computes pairwise Jaccard similarity across video audience commenter sets.
- **Outputs**: `data/output/video_overlap_matrix.parquet`

#### `s35`: Topic-Cohort Affinity Clustering ([s35_topic_cohorts.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s35_topic_cohorts.py))

- **Purpose**: Identifies which topics attract and retain specific seasonal commenter cohorts.
- **Outputs**: `data/output/topic_cohorts.parquet`

#### `s36`: Arrival Speed Dynamics ([s36_arrival_speed.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s36_arrival_speed.py))

- **Purpose**: Segments commenters into speed tiers (First Responders $<1\text{h}$, Early Day $\le 24\text{h}$, Long-Tail $>24\text{h}$).
- **Outputs**: `data/output/arrival_speed.parquet`

#### `s37`: Comparative Video Radar Profiles ([s37_cross_video_comparisons.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s37_cross_video_comparisons.py))

- **Purpose**: Generates multi-dimensional radar metrics across leading video uploads.
- **Outputs**: `data/output/video_profile_radar.parquet`

---

### Phase 5: Predictive, Causal Modeling & Synthesis

#### `s38`: Advanced Statistical & Predictive Modeling ([s38_modeling.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s38_modeling.py))

- **Purpose**: Machine learning like-prediction (XGBoost), Tree SHAP feature attributions, and Kaplan-Meier thread survival decay.
- **Outputs**: `data/output/xgboost_like_predictor.pkl`, `shap_values.npy`, `shap_features.parquet`, `kaplan_meier_survival.parquet`, `stl_decomposition.parquet`, `diurnal_heatmap.parquet`, `poisson_bursts.parquet`, `engagement_forecast.parquet`, `power_law_fit.parquet`, `kruskal_wallis_results.parquet`, `dunn_posthoc_matrix.parquet`

#### `s39`: Creator Interaction Causal Uplift (DiD) ([s39_creator_uplift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s39_creator_uplift.py))

- **Purpose**: Evaluates causal impact of creator interactions using Difference-in-Differences (DiD) quasi-experiments.
- **Outputs**: `data/output/creator_causal_uplift.parquet`, `data/output/creator_intervention_threads.parquet`

#### `s40`: UI Metric Synthesis ([s40_synthesis.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s40_synthesis.py))

- **Purpose**: Final pre-computation step synthesizing dashboard metadata and indexing topic titles in-place.
- **Outputs**: Updates `topic_metadata.parquet` in-place

---

### Phase 6: Final Visualizations & Reporting

#### `s99`: Publication Visualizations Gallery ([s99_visualize.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s99_visualize.py))

- **Purpose**: Generates all 55+ publication-ready visual artifacts in isolated submodules under `src/pipeline/visualizations/`.
- **Inputs**: All `.parquet` datasets produced across Phases 1 through 5.
- **Outputs**: All PNG/HTML plot figures saved to `data/output/plots/`
- **Execution Principle**: **Runs strictly last (`s99`)**, guaranteeing all upstream analytical tables are fully materialized.
