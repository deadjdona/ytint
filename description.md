# 📊 YouTube Comment Corpus Analysis

## 0. Data Dimensions
- **Content**: text, thread structure, timestamp mentions  
- **Author**: channel ID, display name, recurrence, account age  
- **Time**: publish/update times, reply latency  
- **Engagement**: likes, replies, creator hearts, pinned status  
- **Video context**: video ID, title, tags, category, duration, views/likes  
- **Derived**: language, sentiment, embeddings, topics, toxicity, bot-likelihood  

---

## 1. Temporal / Attention Dynamics
- Comment velocity curve → half-life of discussion  
- Survival curve of threads  
- Diurnal + weekly heatmaps  
- Upload-event alignment (first N hours)  
- Revival detection (spikes)  
- Reply latency distribution  
- Cohort retention (returning commenters)  
- Shelf life of likes  

**Visualisations**: streamgraph, horizon charts, heatmaps, Kaplan–Meier curves, ridge plots  

---

## 2. Topic & Semantic
- Topic modelling (LDA, BERTopic, Top2Vec)  
- Topic × video matrix  
- Topic prevalence over time  
- Semantic embeddings (UMAP/t-SNE)  
- Semantic drift across months/videos  
- Keyword frequency (TF-IDF)  
- Co-occurrence networks  
- Named entity extraction  
- Intent taxonomy (praise, complaint, spam, etc.)  

**Visualisations**: UMAP scatter, streamgraph, treemap, alluvial diagram, co-occurrence graph  

---

## 3. Sentiment & Affect
- Valence distribution per video/topic  
- Sentiment trajectory over time  
- Emotion classification (Plutchik wheel)  
- Sarcasm/irony detection  
- Toxicity prevalence  
- Polarity vs engagement  
- Creator-positive vs negative ratio  
- Sentiment divergence between replies  

**Visualisations**: emotion wheel, ridge plots, scatter plots, diverging bar charts  

---

## 4. Linguistic & Stylistic
- Language distribution  
- Lexical richness (type-token ratio, MTLD)  
- Readability scores  
- Emoji usage & sentiment mapping  
- Slang frequency  
- Comment length vs likes  
- Punctuation intensity (ALL-CAPS, repetition)  
- Code-switching detection  
- Hashtag/@mention networks  

**Visualisations**: emoji treemaps, hexbin plots, stack bars, violin plots  

---

## 5. Network / Graph
- Reply-tree forest  
- Author–video bipartite graph  
- Co-commenting graph → cliques/tribes  
- Reply networks (who replies to whom)  
- Community detection (Louvain, Leiden)  
- Centrality analysis  
- K-core decomposition  
- Reciprocity of replies  
- Clique enumeration  

**Visualisations**: force-directed layouts, hive plots, arc diagrams, sankey flows  

---

## 6. Cohorts & Segmentation
- RFM segmentation (Recency, Frequency, Monetary)  
- Behavioural cohorts: lurkers, regulars, power users  
- Acquisition cohorts by upload event  
- Cross-video overlap matrix (Jaccard)  
- Fan loyalty metrics  
- New vs returning commenter share  
- Topic-cohort clustering  

**Visualisations**: retention curves, scatter plots, sankey diagrams, heatmaps  

---

## 7. Engagement & Attention Economy
- Like-count distribution (power law)  
- Gini coefficient of attention inequality  
- Top-K concentration (Pareto principle)  
- Reply depth distribution  
- Branching factor of threads  
- Pinned/hearted effects  
- First-mover advantage  
- Position bias  
- Attention transfer  

**Visualisations**: log-log plots, Lorenz curves, Pareto charts, histograms  

---

## 8. Thread / Conversational Structure
- Thread depth & width  
- Conversation resolution patterns  
- Sentiment trajectory within threads  
- Reply chain length  
- Initiator/response patterns  
- Topic evolution within threads  

**Visualisations**: sunburst trees, dendrograms, ridge plots, sankey diagrams  

---

## 9. Comparative & Cross-Video
- Video profile radar (sentiment, toxicity, likes, entropy)  
- Channel-to-channel comparison  
- Before/after controversy analysis  
- Series vs standalone comparison  
- Category benchmarking  

**Visualisations**: radar charts, parallel coordinates, dumbbell plots  

---

## 10. Anomaly, Spam & Integrity
- Bot detection (temporal bursts, phrasing)  
- Near-duplicate clustering (MinHash)  
- Coordinated brigading  
- Spam template discovery  
- Topic injection anomalies  
- Sentiment spikes  
- Suspicious like inflation  

**Visualisations**: anomaly scatter, dendrograms, overlay charts, heatmaps  

---

## 11. Author-Level / Identity
- Power-law of commenter activity  
- Author persistence  
- Fingerprint (topics, sentiment, length)  
- Cross-channel overlap  
- Display-name reuse/impersonation  

**Visualisations**: Zipf plots, fingerprint heatmaps, stacked bars  

---

## 12. Cross-Modal (Comments ↔ Video Content)
- Timestamp mentions → reaction mapping  
- Moment-level heatmaps  
- Comment–transcript alignment  
- Topic match with tags  
- Spoiler detection  
- Scene-based reactions  

**Visualisations**: timeline ribbons, dual heatmaps, annotation peaks  

---

## 13. Predictive & Causal
- Predict likes from features  
- Predict video engagement from early comments  
- SHAP feature importance  
- Toxicity prediction from context  
- Early-burst detection  
- Propensity models  

**Visualisations**: SHAP plots, lift curves, confusion matrices, ROC/PR curves  

---

## 14. Meta & Corpus Quality
- Comment-volume vs view-count scaling  
- Engagement rate (comments per 1k views)  
- Disabled/missing comments flagging  
- Sampling bias audit  
- Language coverage gaps  

**Visualisations**: scatter plots, funnel charts, coverage matrices  

---

## 📌 Visualisation Catalog
- Streamgraph, Sankey, Alluvial, Chord diagram  
- Force-directed, hive, arc layouts  
- UMAP/t-SNE/PCA scatter  
- Heatmaps, horizon charts  
- Treemap, sunburst, dendrogram  
- Ridge/violin/hexbin distributions  
- Kaplan–Meier curves, Lorenz curves, Pareto, Gini  
- Radar, parallel-coordinates, small multiples  
- Emotion wheel, valence-arousal plane  
- Video-timeline ribbons  
- SHAP/lift/ROC charts  

---

## 🔗 Cross-Cutting Combinations
- Topic × sentiment × time → mood shifts  
- Cohort × community → tribes  
- Network centrality × toxicity → influence vs harm  
- Reply-depth × sentiment trajectory → polarization  
- Timestamped reactions × transcript → audience reaction map  

---

## 🚀 Pipeline Architecture (53 Canonical Stages)
The analysis is formally orchestrated through a high-performance, deterministic sequential pipeline located in `src/pipeline/`, spanning 53 discrete stages organized into 7 logical phases:

### Phase 1: Ingestion & Foundational NLP / Semantics
- **[s00] Ingest & Migration** ([s00_ingest.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s00_ingest.py)): Raw SQLite schema normalization, data cleansing, and Parquet table generation.
- **[s01] NLP & Sentiment Enrichment** ([s01_enrich.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s01_enrich.py)): Multi-dimensional NLP features: VADER compound valence, RoBERTa GoEmotions (28 classes), Detoxify toxicity scoring, speech acts, readability, and lexical richness (TTR, Yule's K, MTLD).
- **[s02] Topic Modeling & Semantic Drift** ([s02_topics.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s02_topics.py)): BERTopic extraction with `all-MiniLM-L6-v2` embeddings, UMAP projection, and Procrustes-aligned Word2Vec semantic drift.
- **[s03] Named Entity Extraction** ([s03_ner.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s03_ner.py)): High-throughput spaCy NER extracting People, Organizations, Locations, and Products.
- **[s04] Word Co-occurrence Networks** ([s04_cooccurrence.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s04_cooccurrence.py)): Contextual term co-occurrence graph with PMI edge weighting.
- **[s05] Hashtag & Mention Networks** ([s05_tags_mentions.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s05_tags_mentions.py)): Entity mention graphs and topical hashtag propagation topologies.
- **[s06] Audience Demand Intent Mining** ([s06_audience_intent.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s06_audience_intent.py)): 6-class intent taxonomy classification (Praise, Complaint, Question, Suggestion, Off-Topic, Spam).
- **[s07] Target Stance & Polarization Drift** ([s07_stance_drift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s07_stance_drift.py)): Creator/topic stance modeling across reply depths.
- **[s08] Code-Switching Language Impact** ([s08_code_switching.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s08_code_switching.py)): Multilingual script mixing (Cyrillic/Latin) and linguistic engagement differentials.

### Phase 2: Conversational Structure & Thread Dynamics
- **[s09] Thread Width & Branching Factor** ([s09_thread_width.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s09_thread_width.py)): Reply tree depth, branching factors, and attention transfer across sibling comments.
- **[s10] Reply Latency Distribution** ([s10_reply_latency.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s10_reply_latency.py)): Log-normal reply response latency curves and inter-arrival timing.
- **[s11] Position Bias & Early Mover Effect** ([s11_position_bias.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s11_position_bias.py)): First-mover advantage decay and comment rank position bias on engagement.
- **[s12] Conversation Resolution Patterns** ([s12_resolution_patterns.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s12_resolution_patterns.py)): Terminal leaf resolution classification (Answered, Unresolved, Abandoned).
- **[s13] Initiator / Responder Dynamics** ([s13_initiator_patterns.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s13_initiator_patterns.py)): Conversational dialogue profiling and reciprocity between thread initiators and respondents.
- **[s14] Thread Polarization Dynamics** ([s14_polarization.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s14_polarization.py)): Root-to-leaf sentiment decay slopes with dynamic self-healing DAG depth calculation.
- **[s15] Toxicity Contagion & Troll Catalysts** ([s15_toxicity_contagion.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s15_toxicity_contagion.py)): Epidemiological branching modeling ($R_0$ reproduction rate) and flame-war catalyst detection.

### Phase 3: Author Topologies, Forensics & Integrity
- **[s16] Author Reply Network Graph** ([s16_network.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s16_network.py)): Directed author interaction network with Louvain community detection, PageRank, betweenness, and K-core decomposition.
- **[s17] Author-Video Bipartite Graph** ([s17_bipartite.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s17_bipartite.py)): Bipartite author-video projection and affiliation structures.
- **[s18] Co-commenting Graph** ([s18_cocommenting.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s18_cocommenting.py)): Projected co-commenting clique network identifying audience tribes.
- **[s19] Author & Video Aggregations (RFM)** ([s19_aggregation.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s19_aggregation.py)): Recency, Frequency, Monetary (likes) scoring, K-Means behavioral clustering, and Gini inequality.
- **[s20] Commenting Frequency Tiers** ([s20_frequency_tiers.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s20_frequency_tiers.py)): Activity distribution spectrum (1-time lurkers through power engagers).
- **[s21] Drive-by vs Loyalists Segmentation** ([s21_driveby_loyalists.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s21_driveby_loyalists.py)): Channel breadth and loyalty classification.
- **[s22] Author Stylometric Fingerprints** ([s22_author_fingerprints.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s22_author_fingerprints.py)): Stylometric vectors (vocabulary, syntax, sentiment, punct) per recurring commenter.
- **[s23] Creator Impersonation Detection** ([s23_impersonation_detection.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s23_impersonation_detection.py)): Levenshtein distance matching and channel verification checks to catch scam handles.
- **[s24] Bot & Spammer Heuristics** ([s24_bot_heuristics.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s24_bot_heuristics.py)): Multi-signal bot probability scoring (burst timing, duplicate text, account age skew).
- **[s25] Coordinated Inauthentic Behavior (CIB)** ([s25_cib_detection.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s25_cib_detection.py)): Temporal synchronization graph clustering detecting coordinated brigading rings.
- **[s26] Integrity & Spam Flags** ([s26_integrity.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s26_integrity.py)): MinHash LSH near-duplicate clustering and commercial spam template discovery.
- **[s27] Suspicious Like Inflation** ([s27_anomaly_inflation.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s27_anomaly_inflation.py)): Statistical outlier detection flagging artificial vote manipulation.

### Phase 4: Longitudinal Dynamics, Cross-Modal & Cohorts
- **[s28] Narrative Timeline & Change-Points** ([s28_narrative.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s28_narrative.py)): PELT structural changepoint detection and second-derivative viral anomaly extraction.
- **[s29] Cross-Modal Video Reaction Map** ([s29_cross_modal.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s29_cross_modal.py)): Quoted timestamp mention extraction (`@MM:SS`) mapped along video durations.
- **[s30] Shelf Life of Likes & Video Decay** ([s30_shelf_life.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s30_shelf_life.py)): Empirical CDF curves of 90% like accumulation and engagement longevity.
- **[s31] Longitudinal Topic Evolution** ([s31_topic_evolution.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s31_topic_evolution.py)): Multi-period streamgraph topic shifts and semantic transition matrices.
- **[s32] Topic × Video Heatmap Matrix** ([s32_topic_matrix.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s32_topic_matrix.py)): Clustered cross-video thematic distribution matrix.
- **[s33] Acquisition Cohorts & Retention** ([s33_cohorts.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s33_cohorts.py)): Upload-event arrival cohorts, vintage retention decay, and new vs returning commenter shares.
- **[s34] Cross-Video Audience Overlap** ([s34_overlap.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s34_overlap.py)): Pairwise Jaccard similarity matrices of author sets across all videos.
- **[s35] Topic-Cohort Affinity Clustering** ([s35_topic_cohorts.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s35_topic_cohorts.py)): Audience clustering segmented by topical preferences.
- **[s36] Arrival Speed Dynamics** ([s36_arrival_speed.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s36_arrival_speed.py)): High-resolution first-N-hours comment velocity curves and early arrival kinetics.
- **[s37] Comparative Video Radar Profiles** ([s37_cross_video_comparisons.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s37_cross_video_comparisons.py)): Multi-dimensional radar profiles and 14-day controversy impact tracking.

### Phase 5: Predictive Modeling & Causal Inference
- **[s38] Machine Learning, Tree SHAP & Survival** ([s38_modeling.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s38_modeling.py)): Full predictive suite: XGBoost like prediction, exact Tree SHAP attributions, Kaplan-Meier thread survival estimation, STL time-series decomposition, Poisson burst modeling, category ANOVA/Kruskal-Wallis, toxicity escalation classifiers, and author return propensity models.
- **[s39] Creator Causal Uplift (DiD)** ([s39_creator_uplift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s39_creator_uplift.py)): Difference-in-Differences quasi-experimental causal estimation of creator hearts and pinned comment interventions.
- **[s40] UI Metric Synthesis** ([s40_synthesis.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s40_synthesis.py)): Rollup aggregation synthesizing metric tables for real-time dashboard consumption.

### Phase 5b: Extended Supplementary Analysis
- **[s41] TF-IDF Keyword Extraction** ([s41_tfidf_keywords.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s41_tfidf_keywords.py)): Per-video salient term extraction with TF-IDF and n-gram filtering.
- **[s42] Polarity vs Engagement Analysis** ([s42_polarity_engagement.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s42_polarity_engagement.py)): Non-linear relationship between emotional extremity and like/reply yields.
- **[s43] Emoji Signatures & Sentiment Mapping** ([s43_emoji_signatures.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s43_emoji_signatures.py)): Per-topic emoji diversity profiles and emoji-to-sentiment cross-mappings.
- **[s44] Sentiment Anomaly Detection** ([s44_sentiment_anomalies.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s44_sentiment_anomalies.py)): Rolling Gaussian $Z$-score negativity spike detectors.
- **[s45] Meta & Corpus Quality Analysis** ([s45_corpus_quality.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s45_corpus_quality.py)): Video view-comment scaling laws, comment disabled checks, and language coverage gaps.
- **[s46] Within-Thread Topic Drift** ([s46_thread_topic_drift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s46_thread_topic_drift.py)): Tracking semantic topic divergence from thread root to leaf replies.
- **[s47] Slang & Internet-Register Lexicon** ([s47_slang_lexicon.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s47_slang_lexicon.py)): Frequency tracking of internet slang, acronyms, and community neologisms.
- **[s48] Bow-Tie & Top-K Concentration** ([s48_bowtie_concentration.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s48_bowtie_concentration.py)): Network bow-tie partitioning (SCC, IN, OUT, Tendrils) and Pareto attention concentration.
- **[s49] Series vs Standalone & Creator Net Polarity** ([s49_series_creator_sentiment.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s49_series_creator_sentiment.py)): Performance differentials between episodic series and standalone uploads, plus creator-directed net sentiment polarity.
- **[s50] Scene Reaction Taxonomy & Spoilers** ([s50_cross_modal_reactions.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s50_cross_modal_reactions.py)): 5-class scene reaction taxonomy (Humor, Surprise, Emotional, Critical, General) and plot spoiler detection.
- **[s51] Topic Injection Anomaly Scanner** ([s51_topic_injection.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s51_topic_injection.py)): Information-theoretic Jensen-Shannon divergence monitors flagging exogenous narrative takeovers.

### Phase 6: Publication Visualizations Gallery
- **[s99] Publication Plots Gallery** ([s99_visualize.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s99_visualize.py)): Generates 68+ publication-grade vector figures in `data/output/plots/` (streamgraphs, Kaplan-Meier curves, Pareto distributions, Plutchik emotion wheels, UMAP semantic projections, and force-directed topologies).

---

## ⚡ Standalone Analytical Engines & CLI Tools
Complementing the sequential pipeline, **ytint** provides 8 production CLI engines in `src/engine/`:

1. **Live Ingest Connector** (`ytint-ingest` / [`src/engine/youtube_api.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/youtube_api.py)): Live YouTube Data API v3 ingester with token-bucket rate limiting and auto-pagination.
2. **Channel & Corpus Comparator** (`ytint-compare` / [`src/engine/comparator.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/comparator.py)): Pairwise multi-channel benchmarking, cross-channel audience overlap, and comparative radar diagnostics.
3. **Forensic Stylometry & Impersonation Engine** (`ytint-fingerprint` / [`src/engine/fingerprint.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/fingerprint.py)): Author stylometric profiling and creator impersonation detection.
4. **Cohort Survival & Churn Engine** (`ytint-cohort` / [`src/engine/cohort_survival.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/cohort_survival.py)): Longitudinal commenter retention matrices, Kaplan-Meier churn modeling, and return hazard rates.
5. **Causal Impact & Intervention Estimator** (`ytint-causal` / [`src/engine/causal_impact.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/causal_impact.py)): Quasi-experimental Difference-in-Differences and counterfactual uplift estimators.
6. **Agent-Based Conversational Simulator** (`ytint-simulate` / [`src/engine/agent_simulator.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/agent_simulator.py)): Monte Carlo agent-based model simulating thread polarization, community formation, and toxicity spread.
7. **Multi-Format Export Manager** (`ytint-export` / [`src/engine/export_manager.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/export_manager.py)): Multi-format analytical exporter generating PDF briefing reports, multi-tab Excel workbooks, and Parquet archives.
8. **Automated Intelligence Narrator** (`ytint-narrate` / [`src/engine/briefing_narrator.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/briefing_narrator.py)): Natural language executive summary generator summarizing channel health, crisis flashpoints, and strategic recommendations.

---

## 🖥️ Interactive Command Center (`ytint-dash`)
The platform includes a 9-module Streamlit intelligence command center in [`src/ui/app.py`](file:///c:/Users/deadj/Sources/ytint/src/ui/app.py):
- **Tab 1: Executive Overview & KPI Cockpit**: Channel health scores, macro metrics, and activity timelines.
- **Tab 2: Temporal Dynamics, Velocity & Crisis Monitors**: Diurnal heatmaps, comment velocity curves, STL decomposition, and real-time crisis flashpoint alerts.
- **Tab 3: Conversational Structure & Reply Trees**: Deep thread tree visualization, reply latency distributions, resolution patterns, and polarization slopes.
- **Tab 4: Audience Forensics, RFM & CIB Rings**: 3D interactive RFM scatter space, Louvain community networks, coordinated inauthentic behavior rings, and bot scoring.
- **Tab 5: Predictive Models, SHAP & What-If Simulator**: Interactive like prediction with real-time Tree SHAP waterfall attribution and virality simulator.
- **Tab 6: Cross-Modal Scene Reactions & Timeline Ribbons**: Video player aligned with moment-level reaction taxonomy, spoken transcripts, and spoiler detectors.
- **Tab 7: Cross-Video Comparative Benchmark**: Multi-video radar profiles, series vs standalone benchmarks, and controversy impact timelines.
- **Tab 8: Agentic AI Assistant & Executive Briefings**: Interactive conversational analyst for dataset querying and one-click executive report generation.
- **Tab 9: Ad-Hoc SQL Query Studio & Parquet Data Explorer**: Interactive DuckDB SQL editor with query presets and direct Parquet schema browser.

---

## 🛡️ Pipeline Lineage & Execution Integrity
- **Topological DAG Ordering**: All 53 stages execute in strict order without forward dependency inversions.
- **Self-Healing Thread Depth**: [`s14_polarization.py`](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s14_polarization.py) dynamically computes comment thread depth via NetworkX topological DAG walk if `comment_depth` has not yet been computed by `s16_network.py`, guaranteeing deterministic clean runs.
- **Incremental Cascade Protection**: [`src/pipeline/runner.py`](file:///c:/Users/deadj/Sources/ytint/src/pipeline/runner.py) automatically cascades execution forward when an upstream ingestion delta is processed (`has_delta = True`), ensuring downstream analytical layers never become stale.
- **Comprehensive Artifact Registry**: All generated artifacts (including 14 modeling artifacts in `s38` and multi-input dependencies in `s49`) are tracked in the pipeline runner registry.

---

## 🔮 Advanced Architectural Frontiers
- ✅ **Creator Persona Tone-Matching & Stylometric LoRA Engine** ([persona.py](file:///c:/Users/deadj/Sources/ytint/src/engine/persona.py) & [assistant.py](file:///c:/Users/deadj/Sources/ytint/src/engine/assistant.py)): **Operational**. Extracts authentic creator stylometric DNA (vocabulary entropy, punctuation rhythms, emoji signatures, greetings, signoffs), injects dynamic semantic few-shot examples into AI draft replies, and generates paired instruction-tuning datasets (Alpaca/ChatML JSONL + Ollama Modelfile) for PEFT/LoRA fine-tuning.
- **Live Stream WebSocket Chat Watcher** (`engine.watcher`): Continuous background daemon for real-time live stream chat polling, sentiment anomaly alerts, and flame-war monitoring.
- **Distributed Worker Scaling (Ray / Polars Backend)**: Scaling NLP tokenization and embedding workflows across distributed worker clusters for 10M+ comment corpora.


