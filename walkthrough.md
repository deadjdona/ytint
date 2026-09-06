# 🚀 Comprehensive Architecture & Implementation Walkthrough: `ytint`

The **ytint** platform is an end-to-end, high-dimensional YouTube Conversational Intelligence & Forensics system. It processes raw YouTube comment exports into multi-dimensional forensic, topological, semantic, causal, and predictive intelligence across **53 canonical analytical layers** (`s00_ingest.py` through `s51_topic_injection.py`, plus `s99_visualize.py`).

---

## 🏗️ 1. Complete Architecture Summary

```mermaid
graph TD
    subgraph Phase 0 & 1: Ingestion & Core NLP
        Raw[(commentsuite.sqlite3)] --> s00[s00: Ingest & Clean]
        s00 --> Interim[(Interim Parquet)]
        s01[s01: NLP & Sentiment] --> Interim
        Interim --> P1[Phase 1: Semantics & Intent<br>s02–s08]
    end

    subgraph Phases 2–5b: Computational Pipeline
        Interim --> P2[Phase 2: Conversation Tree Dynamics<br>s09–s15]
        Interim --> P3[Phase 3: Author Loyalty & Forensics<br>s16–s27]
        Interim --> P4[Phase 4: Longitudinal & Cohorts<br>s28–s37]
        Interim --> P5[Phase 5: ML & Causal DiD<br>s38–s40]
        Interim --> P5b[Phase 5b: Extended Analysis<br>s41–s51]

        P3 --> AuthorsFinal[(authors_final.parquet)]
        P4 --> VideosFinal[(videos_final.parquet)]
        P1 --> TopicMeta[(topic_metadata.parquet)]
    end

    subgraph Presentation Tier
        P1 & P2 & P3 & P4 & P5 & P5b --> Output[(Output Parquet Tables)]
        Output --> s99[s99: Publication Visualizations<br>68+ Statistical Charts]
        Output --> App[Streamlit Executive Dashboard<br>7 Tabs, 3D RFM, Simulators, Query Sandbox]
        s99 --> App
    end
```

---

## 🗺️ 2. Comprehensive 53-Stage Analytical Breakdown

### Phase 1: Ingestion, NLP & Demand Intent (`s00`–`s08`)
- **`s00` Ingestion & Synthesis** ([s00_ingest.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s00_ingest.py)): Migrates raw SQLite databases, standardizes schemas, and writes compressed Parquet tables (`comments_clean.parquet`, `videos_clean.parquet`).
- **`s01` NLP & Sentiment Enrichment** ([s01_enrich.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s01_enrich.py)): Computes VADER continuous sentiment, XLM-R / RubERT 3-class sentiment, GoEmotions emotion classification, Detoxify toxicity scores, text readability (Flesch), and lexical diversity (MATTR, MTLD, Yule's K).
- **`s02` Topic Modeling** ([s02_topics.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s02_topics.py)): Generates multilingual c-TF-IDF topic representations using BERTopic with UMAP 2D semantic clustering and Procrustes-aligned semantic drift.
- **`s03` Named Entity Recognition** ([s03_ner.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s03_ner.py)): Extracts proper entities (`PER`, `ORG`, `LOC`, `PRODUCT`, `EVENT`) across comments.
- **`s04` Word Co-occurrence Graph** ([s04_cooccurrence.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s04_cooccurrence.py)): Builds co-occurrence edge networks across significant terms.
- **`s05` Hashtag & Mention Networks** ([s05_tags_mentions.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s05_tags_mentions.py)): Constructs social tagging and author `@mention` interaction graphs.
- **`s06` Audience Demand Intent Mining** ([s06_audience_intent.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s06_audience_intent.py)): Categorizes audience comments into strategic intents (Feature Requests, Technical Questions, Praise, Bug Reports, Debates).
- **`s07` Target Stance & Polarization Drift** ([s07_stance_drift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s07_stance_drift.py)): Traces ideological stance divergence (`favor_pct` vs `against_pct`) across reply tree depth.
- **`s08` Code-Switching Detection** ([s08_code_switching.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s08_code_switching.py)): Identifies bilingual script-mixing and assesses upvote engagement lift.

### Phase 2: Conversational Tree & Thread Dynamics (`s09`–`s15`)
- **`s09` Thread Width & Branching Factor** ([s09_thread_width.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s09_thread_width.py)): Maps conversation tree geometry and attention transfer ratios.
- **`s10` Reply Latency Distribution** ([s10_reply_latency.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s10_reply_latency.py)): Models conversation response time decay.
- **`s11` Position Bias & Early Mover Effect** ([s11_position_bias.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s11_position_bias.py)): Quantifies the upvote advantage of early commenting.
- **`s12` Conversation Resolution Patterns** ([s12_resolution_patterns.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s12_resolution_patterns.py)): Classifies terminal thread leaves into consensus, hostility, or abandonment.
- **`s13` Initiator / Responder Dynamics** ([s13_initiator_patterns.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s13_initiator_patterns.py)): Profiles linguistic and sentiment deltas between thread starters and responders.
- **`s14` Thread Polarization Dynamics** ([s14_polarization.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s14_polarization.py)): Isolates contentious bimodal debate trees.
- **`s15` Toxicity Contagion & Troll Catalysts** ([s15_toxicity_contagion.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s15_toxicity_contagion.py)): Models epidemic toxicity reproduction ($R_0$) and ranks flame-war catalysts.

### Phase 3: Author Loyalty, Network Topology & Forensics (`s16`–`s27`)
- **`s16` Author Reply Network Graph** ([s16_network.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s16_network.py)): Computes PageRank and Louvain modularity clusters.
- **`s17` Author-Video Bipartite Graph** ([s17_bipartite.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s17_bipartite.py)): Projects two-mode network structure between commenters and video catalogs.
- **`s18` Co-commenting Network** ([s18_cocommenting.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s18_cocommenting.py)): Identifies implicit sub-communities via Jaccard similarity.
- **`s19` Author & Video Aggregations (RFM)** ([s19_aggregation.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s19_aggregation.py)): Core aggregation generating `authors_final.parquet` (RFM scores) and `videos_final.parquet`.
- **`s20` Commenting Frequency Tiers** ([s20_frequency_tiers.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s20_frequency_tiers.py)): Buckets commenters by engagement frequency.
- **`s21` Drive-by vs Loyalists** ([s21_driveby_loyalists.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s21_driveby_loyalists.py)): Segments one-off casual commenters from cross-video community loyalists.
- **`s22` Author Stylometric Fingerprints** ([s22_author_fingerprints.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s22_author_fingerprints.py)): Generates multi-axis radar profiles for top active authors.
- **`s23` Creator Impersonation Detection** ([s23_impersonation_detection.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s23_impersonation_detection.py)): Flags unauthorized accounts copying creator channel names.
- **`s24` Bot & Spammer Classifier** ([s24_bot_heuristics.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s24_bot_heuristics.py)): Flags automated spam bots using volume vs uniqueness ratios and Isolation Forests.
- **`s25` Coordinated Inauthentic Behavior (CIB)** ([s25_cib_detection.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s25_cib_detection.py)): Detects astroturfing rings synchronizing comments within $\Delta t \le 120\text{s}$.
- **`s26` Integrity & Spam Flags** ([s26_integrity.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s26_integrity.py)): Evaluates MinHash LSH text duplicates and burst brigading.
- **`s27` Suspicious Like Inflation** ([s27_anomaly_inflation.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s27_anomaly_inflation.py)): Detects astroturfed comments with skewed like-to-reply ratios.

### Phase 4: Video Dynamics, Cohorts & Longitudinal Topology (`s28`–`s37`)
- **`s28` Narrative Timeline & Change-Points** ([s28_narrative.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s28_narrative.py)): Computes timeline statistics, PELT change-points, and viral anomaly spikes ($>2.5\sigma$).
- **`s29` Cross-Modal Video Reaction Map** ([s29_cross_modal.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s29_cross_modal.py)): Maps comment timestamps to exact video playback runtimes (MM:SS).
- **`s30` Shelf Life of Likes & Video Decay** ([s30_shelf_life.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s30_shelf_life.py)): Estimates exponential decay rates and engagement half-life ($t_{1/2}$).
- **`s31` Longitudinal Topic Evolution** ([s31_topic_evolution.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s31_topic_evolution.py)): Tracks topic interest changes across months.
- **`s32` Topic × Video Matrix** ([s32_topic_matrix.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s32_topic_matrix.py)): Constructs clustered heatmap of topics across videos.
- **`s33` User Acquisition Cohorts & Retention** ([s33_cohorts.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s33_cohorts.py)): Tracks retention curves across monthly acquisition cohorts.
- **`s34` Cross-Video Audience Overlap** ([s34_overlap.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s34_overlap.py)): Computes pairwise Jaccard audience overlap matrices.
- **`s35` Topic-Cohort Affinity Clustering** ([s35_topic_cohorts.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s35_topic_cohorts.py)): Maps cohort retention to specific video topics.
- **`s36` Arrival Speed Dynamics** ([s36_arrival_speed.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s36_arrival_speed.py)): Segments arrival velocity into First Responders ($<1\text{h}$), Early Day ($\le 24\text{h}$), and Long-Tail ($>24\text{h}$).
- **`s37` Comparative Video Radar Profiles** ([s37_cross_video_comparisons.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s37_cross_video_comparisons.py)): Generates multi-dimensional radar comparison metrics across leading videos.

### Phase 5: Predictive, Causal Modeling & Synthesis (`s38`–`s40`)
- **`s38` Predictive Modeling & Survival Analysis** ([s38_modeling.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s38_modeling.py)): Trains XGBoost like predictors, calculates Tree SHAP attributions, right-censored Kaplan-Meier thread survival, Poisson burst detection, and Kruskal-Wallis / Dunn's post-hoc pairwise testing.
- **`s39` Creator Causal Uplift (DiD)** ([s39_creator_uplift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s39_creator_uplift.py)): Computes Difference-in-Differences quasi-experimental treatment effects for early creator interactions.
- **`s40` UI Metric Synthesis** ([s40_synthesis.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s40_synthesis.py)): Pre-computes and indexes topic metadata in-place for high-speed dashboard loading.

### Phase 5b: Extended Supplementary Analysis (`s41`–`s51`)
- **`s41` TF-IDF Keyword Extraction per Video** ([s41_tfidf_keywords.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s41_tfidf_keywords.py)): Extracts salient differentiating keywords per video asset.
- **`s42` Polarity vs Engagement** ([s42_polarity_engagement.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s42_polarity_engagement.py)): Measures emotional polarity vs upvote/reply yield correlation.
- **`s43` Emoji Signatures & Sentiment Mapping** ([s43_emoji_signatures.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s43_emoji_signatures.py)): Builds empirical emoji-to-sentiment profiles.
- **`s44` Sentiment Anomaly Detection** ([s44_sentiment_anomalies.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s44_sentiment_anomalies.py)): Detects sudden hourly and daily negativity spikes ($>2.5\sigma$).
- **`s45` Meta & Corpus Quality Analysis** ([s45_corpus_quality.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s45_corpus_quality.py)): Audits HTTP codes, comment disable flags, and ISO language identification coverage.
- **`s46` Within-Thread Topic Drift** ([s46_thread_topic_drift.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s46_thread_topic_drift.py)): Measures topic divergence across deep conversational trees.
- **`s47` Slang & Internet-Register Lexicon** ([s47_slang_lexicon.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s47_slang_lexicon.py)): Tracks gaming neologisms, slang frequency, and sentiment correlations.
- **`s48` Network Bow-Tie Structure & Top-K Concentration** ([s48_bowtie_concentration.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s48_bowtie_concentration.py)): Partitions author graph into Bow-Tie components and Pareto shares.
- **`s49` Series vs Standalone & Creator Sentiment Polarity** ([s49_series_creator_sentiment.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s49_series_creator_sentiment.py)): Benchmarks episodic vs standalone videos and creator net sentiment polarity.
- **`s50` Cross-Modal Scene Reactions & Spoiler Detection** ([s50_cross_modal_reactions.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s50_cross_modal_reactions.py)): Classifies scene reaction types (Humor, Shock, Emotional, Critique, Navigation) and detects narrative spoilers.
- **`s51` Topic Injection & Thematic Hijack Scanner** ([s51_topic_injection.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s51_topic_injection.py)): Detects unnatural topic injections across rolling time windows using Jensen-Shannon divergence.

### Phase 6: Publication-Ready Visualizations (`s99`)
- **`s99` Visualizations Gallery** ([s99_visualize.py](file:///c:/Users/deadj/Sources/ytint/src/pipeline/s99_visualize.py)): Modular package in `src/pipeline/visualizations/` generating 68+ publication-quality Matplotlib/Seaborn statistical plots across NLP, threads, author forensics, temporal modeling, and predictive analytics.

---

## 🤖 4. Native Intelligence Engines & CLI Extensions

### 📄 Executive Intelligence Dossier Generator ([`src/engine/reporter.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/reporter.py))
- **Command**: `ytint-report` or `ytint-runner --report`
- **Features**: Compiles a standalone, offline-portable HTML report synthesizing forensic threat levels, audience Pareto ratios, topic demand distributions, video performance rankings, and base64-embedded high-resolution publication charts. Includes `@media print` rules for instant 1-click **Save as PDF**.

### 🔌 Live YouTube Data API v3 Ingestion ([`src/engine/youtube_api.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/youtube_api.py))
- **Command**: `ytint-ingest --handle '@channel' --max-videos 10 --max-comments 1000`
- **Features**: Fetches channel uploads, metadata, view/like statistics, and complete comment trees directly via Google YouTube Data API v3. Upserts safely into `commentsuite.sqlite3` and automatically updates Parquet tables. Supports deterministic `--mock` mode for offline testing.

### 🧠 LLM & Gemini Strategic Insight Synthesizer & RAG Agent ([`src/engine/synthesizer.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/synthesizer.py))
- **Command**: `ytint-ai --summary`, `ytint-ai --thread <ID>`, `ytint-ai --ask "..."`
- **Features**: Multi-provider LLM reasoning engine (Google Gemini REST, local offline Ollama, OpenAI-compatible, or mock simulation). Synthesizes deep mathematical pipeline statistics (DiD causal lift, Tree SHAP attributions, $R_0$ toxicity contagion, BERTopic clusters, CIB astroturfing rings) into natural language executive strategy briefings, flame-war debate summaries, and conversational Q&A.

### 🌐 Interactive Network Graph & Gephi Exporter ([`src/engine/network_exporter.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/network_exporter.py))
- **Command**: `ytint-graph` or `ytint-runner --networks`
- **Features**:
  1. **Author Reply Network (Directed)**: Constructs who-replies-to-whom interaction topologies with Louvain modularity communities, PageRank centrality, in/out-degrees, interaction weights, and sentiment/toxicity edge means.
  2. **Author-Video Bipartite Network**: Multi-modal two-mode connection graph linking active multi-video authors to published video nodes with title mappings.
  3. **Author Co-Commenting Network**: Shared engagement community graph linking authors who co-participate across identical video uploads with Jaccard overlap indices.
  4. **Industry-Standard Serializers**: Exports to native `.gexf` (Gephi) and `.graphml` (Cytoscape / yEd) with strict primitive type-safety.
  5. **Interactive WebGL Topology Explorer**: Renders 2D force-directed spring layouts in Plotly with dynamic node sizing, community coloring, and rich multi-attribute hover cards.
  6. **Dashboard Integration**: Tab 4 Section 4.6 interactive network explorer with dynamic top-$K$ filter slider (25-250), color dimension picker, and direct 1-click GEXF/GraphML download buttons.

### 🚨 Automated Webhook Anomaly & Threat Alerting Daemon ([`src/engine/alerting.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/alerting.py))
- **Command**: `ytint-alert --dry-run` or `ytint-runner --alert`
- **Features**:
  1. **Continuous Threat Scanning**: Scans 7 forensic and anomaly layers across CIB astroturfing rings, creator impersonation, viral volume surges, negative sentiment shocks, toxicity outbreaks, troll catalyst instigators, like inflation, and thematic hijacking.
  2. **Multi-Platform Webhook Dispatcher**: Native, rich payload formatting for Discord embeds (color severity coding, timestamped threat fields), Slack Block Kit (section blocks, metric fields), Telegram HTML, and generic JSON endpoints.
  3. **Zero-SDK Network Leanliness**: Built using Python standard library `urllib.request` with strict 7-second timeouts and robust error handling.
  4. **Built-In Dry-Run & Simulation**: Automatically generates and saves structured alert digests to `data/output/alerts/latest_alert.json` when offline or in test mode without failing.
  5. **Pipeline & CLI Integration**: Registered `ytint-alert` console script and added `--alert` flag to `pipeline.runner`.
  6. **Interactive Dashboard Console**: Tab 7 Section 7.5 Alerting Console with live threat scorecards, active incident ledger, webhook configuration, payload preview, and 1-click dispatch.

### ⚡ High-Speed Zero-Copy Analytical SQL Engine ([`src/engine/sql_engine.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/sql_engine.py))
- **Command**: `ytint-sql` or `ytint-sql --interactive`
- **Features**:
  1. **Zero-Copy Parquet Mapping**: Registers 95+ Parquet tables across `data/interim/` and `data/output/` as clean SQL views directly over disk.
  2. **Sub-10ms ANSI SQL**: Executes complex analytical queries, joins, aggregations, and window functions in DuckDB out-of-core with 0 bytes loaded into unmanaged RAM.
  3. **Curated Presets**: 8 production-grade query templates (VIP champions, toxicity outbreaks, DiD causal lift, CIB rings, topical valence drift, video performance benchmarks, bot audits, sentiment attention economy).
  4. **Strict AST Guardrails**: Built-in AST/regex safety enforcement strictly rejecting destructive SQL operations (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`).
  5. **SQL Studio & Visual Chart Builder**: Tab 7 Section 7.8 workbench with catalog explorer, query editor, live telemetry badge, dynamic results table, Plotly chart generator (Bar, Line, Scatter, Histogram), and 1-click CSV/JSON export.

### ⏱️ Temporal Narrative Scene Reaction Forensics ([`src/engine/narrative.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/narrative.py))
- **Command**: `ytint-narrative` or `ytint-narrative --video-id <ID>`
- **Features**:
  1. **Dual-Format Timestamp Extraction**: Regex parser matching plain-text timestamps (`MM:SS`, `HH:MM:SS`, `@MM:SS`) and YouTube watch URL timestamps (`&t=450s`).
  2. **Multi-Class Reaction Taxonomy**: Granular heuristic reaction classifier categorizing comments into `humor_laughter`, `shock_surprise`, `emotional_touching`, `critique_analytical`, `chapter_navigation`, and `general_reaction`.
  3. **Temporal Scene Clustering**: Groups second-by-second audience reactions into configurable resolution bins (10s to 60s) with dominant reaction classification, average VADER sentiment, and verbatim quote montages.
  4. **Viewer Confusion Hotspots**: Proactively identifies scene moments where audience questions and perplexity cluster, recommending creator interventions (pinned clarifications, chapter titles, or video updates).
  5. **Interactive Playback Scrubber Workbench**: Tab 3 Section 3.10 interactive console featuring video selector, scene resolution slider, dual-axis Plotly timeline with reaction taxonomy coloring and confusion hotspot badges, active scene spotlight card with quote montage, and 1-click JSON/CSV export.

### 🕵️ Forensic Author Persona & Sockpuppet Fingerprinting ([`src/engine/fingerprint.py`](file:///c:/Users/deadj/Sources/ytint/src/engine/fingerprint.py))
- **Command**: `ytint-fingerprint` or `ytint-fingerprint --list-rings`
- **Features**:
  1. **Multi-Dimensional Stylometrics**: Computes Shannon vocabulary entropy ($H$), capitalization intensity, punctuation motifs (exclamation, question, ellipsis rates), average word count, and emoji signatures with Unicode symbol frequency parsing.
  2. **Circadian Posting Rhythms**: Models 24-hour diurnal posting distributions (UTC hours 0..23) with peak hour identification and circadian entropy computation (distinguishing narrow automated bot bursts from organic human dispersion).
  3. **Pairwise Stylistic & Temporal Forensics**: Computes stylometric cosine similarity, diurnal cosine similarity, and video target Jaccard overlap to generate composite sockpuppet likelihood scores ($0$–$100\%$).
  4. **Coordinated Ring Graph Clustering**: Employs graph connected-components clustering to group high-confidence pairs into cohesive sockpuppet and alternate account rings.
  5. **Interactive Dashboard Workbench**: Tab 4 Section 4.7 with confidence threshold sliders, summary KPI scorecards, pairwise scatter plot ($X$=Style, $Y$=Diurnal, Color=Score), ring ledger, suspect pair ledger, author persona dossier inspector with 24h diurnal clock and stylometric radar bar charts, and 1-click JSON/CSV exports.

---

## 🎨 5. Streamlit Executive Dashboard Architecture

The dashboard is structured into **7 comprehensive analytical perspectives** in [`src/ui/app.py`](file:///c:/Users/deadj/Sources/ytint/src/ui/app.py):

1. **Executive Briefing & Macro Intelligence**: High-level channel KPIs, 🤖 **AI Executive Strategy Briefing** live container, 📄 one-click Executive Dossier generator & PDF export, community loyalty summaries, and dynamic Plotly bubble quadrant matrix ($X$=Volume, $Y$=Sentiment, Size=Likes, Color=Gini).
2. **Temporal Dynamics & Conversational Flashpoints**: Dynamic Anomaly Sensitivity Scanner ($Z$-score $1.5\sigma$–$4.5\sigma$, rolling baseline window 3–30 days), zoom range-slider, second-by-second reaction trajectories, comparative multi-video playback dynamics, and ⏱️ **Real-Time Chronological Event Replay & Crisis Simulator** (interactive frame scrubber, dual-axis velocity/sentiment/toxicity timeline, flame-war risk gauges, active crisis trigger alerts, and JSON/CSV export).
3. **Conversational Topic Modeling & Audience Demand Intent**: Dynamic topic resonance explorer, 6-class audience demand intent donut, Plutchik emotion wheel, ⚖️ **AI Flame-War & Debate Tree Summarizer**, target stance drift across debate depth, and ⏱️ **Narrative Scene Reaction & Timestamp Scrubbing Forensics** (video selector, scene resolution slider, dual-axis reaction taxonomy and confusion hotspot timeline, interactive playback scrubber, active scene spotlight card with verbatim quote montage, and 1-click JSON/CSV exports).
4. **Audience Segmentation, Loyalty & Forensic Diagnostics**: Interactive 3D RFM community space, Coordinated Inauthentic Behavior (CIB) ring severity map, toxicity contagion ($R_0$), troll catalyst rankings, 🌐 **Interactive Community Network & Gephi Topology Explorer** (2D force-directed spring layout, top-$K$ node filter slider 25-250, Louvain community coloring, and 1-click GEXF/GraphML export buttons), and 🕵️ **Forensic Author Persona & Sockpuppet Fingerprinting** (stylometric feature radar bar charts, 24-hour diurnal posting clocks, pairwise scatter matrix, clustered ring ledger, suspect pair matches, and JSON/CSV exports).
5. **Cross-Video Relations, Counterfactuals & Modeling**: Quasi-experimental Difference-in-Differences (DiD) creator intervention lift chart, interactive "What-If" comment virality simulator with attribution waterfall chart, Tree SHAP feature attribution, and Dunn's post-hoc matrix.
6. **Publication-Ready Visual Analytics Gallery**: 68+ high-resolution statistical plots with structured analytical expanders (**Methodology & Model**, **How to Read**, **Strategic Takeaway**).
7. **Interactive Data Explorer & Export Hub**: Full corpus regex search, multi-layer query sandbox with dynamic chart generation, 🔌 Live YouTube Ingest console, 💬 **"Ask ytint" Conversational AI Analyst (RAG)**, 🔄 **Quick-Sync Incremental Pipeline Runner**, 🚨 **Automated Anomaly & Threat Alerting Console**, 🔍 **Neural Semantic Vector Search Console**, 🛠️ **Creator Actionability & Reply Drafter Workbench**, ⚡ **Zero-Copy Analytical SQL Studio & Query Workbench** (DuckDB-powered out-of-core SQL engine, table schema browser, 8 analytical presets, live telemetry, and Plotly visual chart builder), and instant CSV/Parquet export across all 53 layers.

---

## 🧪 6. Test Suite Validation

```powershell
# Run the complete test suite
.\.venv\Scripts\python.exe -m pytest -v
```

**Results**:
- **Total Tests**: **207 test cases** across 38 test modules
- **Passed**: **207 passed (100% pass rate)**
- **Failed**: **0 failed**
- **Duration**: ~75s
- **Coverage**: Engine unit tests (`fingerprint`, `narrative`, `sql_engine`, `assistant`, `event_replay`, `semantic_search`, `alerting`, `network_exporter`, `delta`, `reporter`, `youtube_api`, `synthesizer`), pipeline stage unit tests (`s00` through `s51`), AST syntax & dashboard compatibility, modular visualization rendering, and forensic configuration checks.



