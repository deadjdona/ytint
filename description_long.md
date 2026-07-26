# ⚙️ YouTube Comment Corpus — Algorithms, Libraries & Pitfalls

## 1. Temporal / Attention Dynamics
- **Comment velocity curve**  
  - `pandas.resample('1H')` or 1-min bins for first hour  
  - Fit exponential decay: \(N(t) = N₀·e^{−λt}\) → half-life = ln(2)/λ  
  - Mixture of exponentials/lognormal for dual peaks  
  - ⚠️ Pitfall: timestamps in UTC → diurnal patterns smeared  

- **Decay / survival curve**  
  - Thread death = last reply; censor active threads  
  - Kaplan–Meier estimator via `lifelines` (Python) or `survival` (R)  
  - Log-rank test for comparison  

- **Diurnal + weekly heatmaps**  
  - 168 bins (hour × week) → 24×7 heatmap  
  - Normalize per-video (z-score)  
  - STL decomposition (`statsmodels`)  

- **Upload-event alignment**  
  - Offset from publish time  
  - Regression: log(final) ~ log(first-hour count), R² > 0.8  

- **Revival events**  
  - Smooth with Savitzky–Golay filter  
  - Compute second derivative, flag >3σ spikes  

- **Reply latency**  
  - Δt = reply − parent  
  - Log-normal distribution → report median + IQR  

- **Cohort retention**  
  - Track return fraction at video N+k  
  - Hazard function for churn  

- **Shelf life of likes**  
  - Time to 90% of likes  
  - CDF plots per video  

---

## 2. Topic & Semantic
- **LDA**  
  - Preprocess: spaCy tokenize, lemmatize, stopword removal  
  - Add bigrams/trigrams (PMI threshold)  
  - Params: passes=10, iterations=500, alpha/eta='auto'  
  - Sweep K=5–50, pick max coherence  
  - ⚠️ Pitfall: short comments → use BTM or aggregate  

- **BERTopic**  
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`  
  - UMAP: n_neighbors=15, n_components=5, min_dist=0.0  
  - HDBSCAN: min_cluster_size=15  
  - Dynamic topics with timestamps  

- **Top2Vec**  
  - Doc2Vec/transformer embeddings → UMAP → clustering  

- **Semantic embeddings**  
  - USE, SBERT, OpenAI embeddings  
  - Cache to disk, FAISS for large corpora  

- **UMAP/t-SNE**  
  - UMAP: n_neighbors=15, min_dist=0.1  
  - t-SNE: perplexity=30, use FIt-SNE for scale  

- **Semantic drift**  
  - Word2Vec per time slice + Procrustes alignment  

- **Keyword / TF-IDF**  
  - `TfidfVectorizer`, YAKE, RAKE  
  - PMI for bigrams  

- **Co-occurrence networks**  
  - Window ±5 tokens, weighted by PMI  
  - Visualize with NetworkX/Gephi  

- **NER**  
  - spaCy transformer models + EntityRuler  
  - Link to Wikidata  

- **Intent taxonomy**  
  - Zero-shot: `facebook/bart-large-mnli`  
  - Categories: praise, complaint, spam, etc.  

---

## 3. Sentiment & Affect
- **Valence distribution**  
  - VADER, Twitter-RoBERTa, multilingual XLM-R  
  - Output: continuous [−1, 1]  

- **Trajectory**  
  - Aggregate by time bin, rolling mean smoothing  
  - Detect peaks with `scipy.signal.find_peaks`  

- **Emotion classification**  
  - GoEmotions (27 categories)  
  - Plutchik wheel visualization  

- **Sarcasm detection**  
  - Heuristic: polarity mismatch  
  - Fine-tuned BERT on SARC dataset  

- **Toxicity detection**  
  - Perspective API, Detoxify, profanity lexicons  
  - Threshold tuning, handle class imbalance  

- **Polarity vs engagement**  
  - Spearman correlation, U-shaped curve  

- **Sentiment divergence**  
  - δ = mean(reply sentiment) − root sentiment  

---

## 4. Linguistic & Stylistic
- **Language detection**: fastText, cld3, langdetect  
- **Lexical richness**: MATTR, MTLD, Yule’s K  
- **Readability**: `textstat` (Flesch, Fog, SMOG)  
- **Emoji usage**: `emoji` library, PMI associations  
- **Slang lexicon**: track adoption curves  
- **Comment length**: log-normal distribution  
- **ALL-CAPS / punctuation intensity**: regex + ratios  
- **Code-switching**: sentence-level detection  
- **Hashtags/mentions**: regex extraction  

---

## 5. Network / Graph
- **Reply-tree forest**: `networkx.DiGraph`, igraph for scale  
- **Author–video bipartite graph**: projections with Jaccard overlap  
- **Co-commenting graph**: edge weights via Jaccard  
- **Reply networks**: directed edges A→B  
- **Community detection**: Louvain, Leiden, Infomap  
- **Centrality**: in-degree, PageRank, betweenness, closeness  
- **K-core decomposition**: prune nodes iteratively  
- **Bow-tie structure**: SCC, IN, OUT, tendrils  
- **Clique enumeration**: Bron–Kerbosch algorithm  
- **Reciprocity**: bidirectional edge ratio  
- **Author specialization**: entropy of video distribution  

---

## 6. Cohorts & Segmentation
- **RFM segmentation**: quintiles → Champions, Loyal, At Risk  
- **Behavioural cohorts**: K-means/GMM clustering  
- **Acquisition cohorts**: retention curves  
- **Cross-video overlap**: Jaccard similarity matrix  
- **Fan loyalty**: fraction of comments to one channel  
- **New vs returning commenters**  
- **Topic-cohort persistence**  

---

## 7. Engagement & Attention Economy
- **Like-count distribution**: power law, MLE fit  
- **Gini coefficient**: inequality measure  
- **Top-K concentration**: Pareto principle  
- **Reply depth distribution**: exponential decay  
- **Branching factor**: avg children per comment  
- **Pinned/hearted effect**: propensity-score matching  
- **First-mover advantage**: monotonic decline in likes  
- **Position bias**: disentangle time vs quality  
- **Attention transfer**: cross-correlation of likes  

---

## 8. Thread / Conversational Structure
- Depth & width distributions  
- Resolution patterns (answered vs unresolved)  
- Sentiment trajectory (monotonic, U-shaped, oscillating)  
- Reply chain length (linear vs branching)  
- Initiator/response patterns  
- Topic evolution within threads  

---

## 9. Comparative & Cross-Video
- **Video profile radar**: normalized metrics  
- **Channel comparison**: Kruskal–Wallis, effect size  
- **Before/after analysis**: CausalImpact, ARIMA  
- **Series vs standalone**: retention tracking  
- **Category benchmarking**: Kruskal–Wallis + Dunn’s test  

---

## 10. Anomaly, Spam & Integrity
- **Bot detection**: Isolation Forest, LOF, DBSCAN  
- **Near-duplicate clustering**: MinHash, SimHash  
- **Coordinated brigading**: Poisson process bursts  
- **Spam template discovery**: cluster n-grams  
- **Topic injection anomalies**: KL divergence  
- **Sentiment anomalies**: χ² or KL divergence  
- **Suspicious like inflation**: deviations from power law  

---

## 11. Author-Level / Identity
- **Power-law activity**: Zipf distribution  
- **Persistence**: Kaplan–Meier survival analysis  
- **Fingerprinting**: feature vectors + clustering  
- **Cross-channel overlap**: audience networks  
- **Display-name reuse**: impersonation risk  

---

## 12. Cross-Modal (Comments ↔ Video Content)
- **Timestamp mentions**: regex extraction → timeline mapping  
- **Reaction heatmap**: bin by 10s intervals  
- **Transcript alignment**: embeddings + cosine similarity  
- **Topic match**: tags vs comment topics  
- **Spoiler detection**: late-event references  
- **Scene reactions**: CLIP embeddings for clustering  

---

## 13. Predictive & Causal
- **Predict likes**: XGBoost/LightGBM, SHAP attribution  
- **Video engagement prediction**: ARIMA, Prophet, LSTM  
- **Toxicity prediction**: multi-label BERT, weighted loss  
- **Early-burst detection**: anomaly detection on like accumulation  
- **Propensity models**: logistic regression, gradient boosting  

---

## 14. Meta & Corpus Quality
- **Scaling law**: comments ~ views^b (b=0.5–0.8)  
- **Engagement rate**: comments per 1k views  
- **Disabled flagging**: API vs UI cross-check  
- **Sampling bias audit**: coverage ratio  
- **
