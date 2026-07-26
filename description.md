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
