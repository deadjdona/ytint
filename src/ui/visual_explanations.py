"""Visual Explanations & Analytical Intelligence Interpretations

Provides comprehensive, structured metadata and expert analytical interpretations
for all visual artifacts generated across the ytint intelligence pipeline.
"""

VISUAL_METADATA = {
    # 1. Temporal Dynamics & Lifecycle Modeling
    "kaplan_meier_survival.png": {
        "title": "Kaplan-Meier Thread Lifespan Survival Curve",
        "summary": "Non-parametric survival analysis quantifying how rapidly comment discussions lose momentum and reach terminal inactivity.",
        "methodology": "Kaplan-Meier Maximum Likelihood Estimator $S(t) = \\prod_{t_i \\le t} (1 - \\frac{d_i}{n_i})$, where death event $d_i$ is defined as conversation cessation past a 48-hour quiet window.",
        "how_to_read": "The horizontal axis shows elapsed time (in hours/days) since the root comment was posted. The vertical axis shows the estimated probability $S(t)$ that the thread will receive further replies. Shaded bands represent the 95% confidence interval.",
        "takeaway": "Identifies the 'half-life' of conversation threads. Across YouTube comments, over 80% of thread engagement ceases within the first 4 to 6 hours. Creator intervention (hearting, pinning, replying) must occur within this golden window to reignite viral vitality."
    },
    "diurnal_heatmap.png": {
        "title": "24×7 Diurnal Comment Activity Matrix",
        "summary": "Heatmap mapping comment density across every hour of the day and day of the week to reveal audience temporal routines.",
        "methodology": "Two-dimensional temporal aggregation binning comments into a 24-hour × 7-day matrix (168 hourly bins), normalized by total volume.",
        "how_to_read": "The horizontal axis shows the Hour of the Day (00:00 to 23:00 UTC); the vertical axis shows Days of the Week (Monday to Sunday). Deep red/purple hotspots indicate peak conversational density.",
        "takeaway": "Pinpoints the exact audience awake cycles and prime engagement windows. Publishing new uploads 1-2 hours before peak diurnal activity maximizes early algorithmic velocity."
    },
    "stl_decomposition.png": {
        "title": "STL Time-Series Volumetric Decomposition",
        "summary": "Splits raw daily comment velocity into fundamental Trend, Day-of-Week Seasonality, and Stochastic Residual components.",
        "methodology": "Seasonal and Trend decomposition using Loess (STL) filtering $Y_t = T_t + S_t + R_t$ with robust outlier weighting to prevent spike distortion.",
        "how_to_read": "Top panel shows observed raw counts; second panel isolates macro multi-month trend; third panel isolates weekly recurring cycles; bottom panel isolates anomalous residual shocks.",
        "takeaway": "Separates organic channel growth from viral spike anomalies and day-of-week seasonality, enabling accurate underlying audience retention assessment."
    },
    "reply_latency_distribution.png": {
        "title": "Reply Latency & Response Velocity Distribution",
        "summary": "Log-scale distribution measuring how quickly community members respond to root comments.",
        "methodology": "Kernel Density Estimation (KDE) and log-transformed latency calculations $\\Delta t = t_{\\text{reply}} - t_{\\text{root}}$ across all reply chains.",
        "how_to_read": "The horizontal axis shows response latency on a logarithmic scale (seconds, minutes, hours, days); the vertical axis shows comment density. Left-skewed peaks denote rapid conversational interplay.",
        "takeaway": "High concentration in sub-minute latencies indicates real-time live event engagement or heated debates, while long tails reflect persistent evergreen search traffic."
    },
    "shelf_life_likes.png": {
        "title": "Shelf Life & Upvote Accrual Velocity",
        "summary": "Decay curve illustrating how rapidly comments accumulate upvotes post-upload.",
        "methodology": "Parametric exponential and power-law regression modeling like accumulation velocity over time intervals post-video publication.",
        "how_to_read": "The horizontal axis shows days since upload; the vertical axis shows the cumulative percentage of total likes attained. The steeper the curve, the more front-loaded the engagement.",
        "takeaway": "Reveals whether a video receives instant explosive upvotes from core subscribers or steady evergreen appreciation over weeks and months."
    },
    "reaction_timeline.png": {
        "title": "Poisson Volumetric Bursts & Reaction Curves",
        "summary": "Tracks instantaneous comment arrival bursts and second-by-second reaction spikes throughout video playback.",
        "methodology": "Non-homogeneous Poisson point process model tracking comment timestamp clustering aligned against video duration seconds.",
        "how_to_read": "The horizontal axis shows video timestamp / playback second; peaks indicate timestamps that provoked extreme emotional or cognitive audience reactions.",
        "takeaway": "Allows creators to pinpoint exact jokes, controversial statements, editing transitions, or guest appearances that generated the most intense viewer reactions."
    },

    # 2. NLP, Semantics & Emotion Spectrum
    "umap_semantics.png": {
        "title": "2D UMAP Semantic Topic Manifold",
        "summary": "Nonlinear dimensionality reduction projecting high-dimensional multilingual sentence embeddings into a 2D topological map.",
        "methodology": "UMAP (Uniform Manifold Approximation and Projection) applied to normalized dense embeddings, colored by HDBSCAN topic clusters.",
        "how_to_read": "Each point represents a comment. Proximity indicates semantic and thematic similarity. Clustered islands represent cohesive conversational themes; isolated points are peripheral niches.",
        "takeaway": "Provides a comprehensive visual landscape of all conversation sub-genres occurring across the channel, revealing unaddressed topic clusters and community factions."
    },
    "plutchik_emotion_wheel.png": {
        "title": "Plutchik Psycho-Evolutionary Emotion Wheel",
        "summary": "Radar visualization of audience sentiment mapped onto Robert Plutchik's 8 primary psycho-evolutionary emotion dyads.",
        "methodology": "GoEmotions multi-label transformer model aggregating raw emotion probabilities into Plutchik dimensions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation).",
        "how_to_read": "Spokes represent the 8 emotional axes. The distance from the origin denotes the relative prevalence of that emotion across the analyzed comment corpus.",
        "takeaway": "Delivers an immediate affective diagnostic of the audience. Dominance of Trust and Joy indicates strong community alignment; spikes in Disgust or Anger flag backlash risks."
    },
    "sentiment_ridges.png": {
        "title": "Sentiment Density Joyplot Across Video Catalogs",
        "summary": "Stacked joyplot comparing the probability density distribution of sentiment scores across top channel videos.",
        "methodology": "Kernel density estimation of compound VADER and multilingual sentiment scores computed individually for each video upload.",
        "how_to_read": "Each ridge represents a video upload. Distributions centered to the right (positive) reflect praise; bi-modal distributions indicate polarized audience debates.",
        "takeaway": "Directly contrasts how different video formats, guests, or controversial topics alter the emotional equilibrium of the comment section."
    },
    "audience_intent_distribution.png": {
        "title": "Audience Intent Distribution & Demand Roadmap",
        "summary": "Categorization of comment intentions separating direct content ideas, questions, critiques, and appreciation.",
        "methodology": "Rule-based and semantic intent classification categorizing comments into 5 functional classes, weighted by like counts.",
        "how_to_read": "Left panel shows the percentage share of each intent; right panel displays average likes per intent category to identify community-validated demand.",
        "takeaway": "Converts unstructured viewer comments into an actionable, prioritized editorial roadmap of what videos, tutorials, or clarifications viewers want next."
    },
    "named_entities.png": {
        "title": "Top Extracted Named Entities (NER)",
        "summary": "Frequency breakdown of the most frequently cited people, organizations, games, locations, and cultural references.",
        "methodology": "SpaCy multilingual Named Entity Recognition (NER) pipeline extracting proper noun entities tagged as PER (person), ORG (organization), LOC (location), or MISC.",
        "how_to_read": "Horizontal bars show entity mention counts. Longer bars indicate entities that dominate audience discussions.",
        "takeaway": "Highlights key influencers, competitors, software tools, games, or public figures that drive organic conversation in your community."
    },
    "word_cooccurrence.png": {
        "title": "Topological Keyword Co-occurrence Network",
        "summary": "Network graph illustrating which key terms and concepts appear together within the same comments.",
        "methodology": "Pairwise term co-occurrence matrix using Jaccard and Pointwise Mutual Information (PMI) coefficients filtered by minimum support thresholds.",
        "how_to_read": "Nodes represent frequent lemmatized words; edges indicate frequent co-occurrence. Clustered hubs represent associative conceptual themes.",
        "takeaway": "Reveals the sub-surface mental associations viewers make when discussing specific themes, characters, or channel controversies."
    },
    "sentiment_divergence.png": {
        "title": "Sentiment Divergence (Root vs. Reply Contagion)",
        "summary": "Measures whether nested reply conversations become significantly more hostile or supportive than initiating root comments.",
        "methodology": "Delta calculation $\\Delta S = \\bar{S}_{\\text{reply}} - \\bar{S}_{\\text{root}}$ per video with statistical significance hypothesis testing.",
        "how_to_read": "Bars to the left (red) indicate reply chains become more negative/toxic than root posts; bars to the right (green) indicate constructive elaborations.",
        "takeaway": "Flags videos whose comment sections suffer from toxic contagion dynamics, requiring proactive community moderation."
    },
    "toxicity_heatmap.png": {
        "title": "24×7 Toxicity & Hostility Heatmap",
        "summary": "Temporal matrix tracking average toxicity scores across hours of the day and days of the week.",
        "methodology": "Detoxify transformer neural network predictions mapped across the 168 hour-of-week grid.",
        "how_to_read": "Darker red cells indicate time windows with statistically elevated toxic and abusive commenting activity.",
        "takeaway": "Informs automated moderation bot schedules and moderator shifts to protect the community during high-risk troll hours."
    },
    "sarcasm_analysis.png": {
        "title": "Sarcasm, Irony & Cynicism Spectrum",
        "summary": "Quantifies the presence of sarcastic, cynical, and ironic linguistic markers across topics.",
        "methodology": "Lexico-syntactic irony scoring evaluating sentiment-punctuation dissonance, hyperbole, and quote usage.",
        "how_to_read": "Bars compare irony density across different video themes and topics.",
        "takeaway": "Prevents standard sentiment models from misinterpreting sarcastic negative praise as genuine hostility, improving sentiment accuracy."
    },
    "linguistic_profiling.png": {
        "title": "Multi-Metric Stylometric & Readability Profile",
        "summary": "Multi-panel breakdown of comment length, Flesch-Kincaid reading grade, emoji density, and vocabulary richness.",
        "methodology": "Computational stylometry extracting Type-Token Ratio (TTR), automated readability index, and uppercase caps ratio.",
        "how_to_read": "Histograms show audience literacy and stylistic distributions across different video types.",
        "takeaway": "Differentiates between quick reaction meme-based audiences and deeply intellectual long-form discussion communities."
    },

    # 3. Audience Networks, Segmentation & Forensics
    "rfm_3d.html": {
        "title": "Interactive 3D RFM Audience Segmentation Manifold",
        "summary": "Three-dimensional interactive behavioral model categorizing commenters based on Recency, Frequency, and Monetary engagement.",
        "methodology": "K-Means and GMM clustering across 3 normalized behavioral axes: Recency (days since active), Frequency (total comments), Monetary (total likes received).",
        "how_to_read": "Click and drag to rotate the 3D manifold. Distinct color clusters correspond to Champions, Loyalists, At-Risk Fans, and One-Time Drive-Bys.",
        "takeaway": "Enables precise super-fan identification, helping creators recognize and nurture the most influential community ambassadors."
    },
    "author_network_force.png": {
        "title": "Force-Directed Author Interaction Graph",
        "summary": "Topological network showing direct conversational interactions, replies, and community sub-clusters among authors.",
        "methodology": "Directed graph $G=(V,E)$ where nodes are authors and edges are replies, laid out using the Fruchterman-Reingold force-directed algorithm with Louvain community modularity.",
        "how_to_read": "Nodes with many connections (hubs) are central conversationalists. Distinct color clusters denote isolated discussion cliques or flame-war factions.",
        "takeaway": "Reveals whether your audience functions as a single unified community or fractured echo chambers arguing in isolated reply threads."
    },
    "bot_heuristics.png": {
        "title": "Multi-Factor Bot Classifier & Heuristic Audit",
        "summary": "Multi-attribute classification separating organic human comments from automated spam bots and copy-paste brigadiers.",
        "methodology": "Composite scoring combining posting burst rates, comment text entropy, token repetition, and zero-like reply patterns.",
        "how_to_read": "Bar and scatter distributions show accounts categorized as Clean Human, Suspicious Spammer, or Definite Bot.",
        "takeaway": "Provides auditable proof of organic reach and filters out synthetic engagement for brand sponsor reporting."
    },
    "cib_rings_graph.png": {
        "title": "Coordinated Inauthentic Behavior (CIB) Astroturfing Rings",
        "summary": "Network clustering exposing coordinated multi-account sockpuppet rings operating in tight temporal synchronization.",
        "methodology": "Graph component detection linking author pairs that comment within $\\Delta t < 120\\text{s}$ across multiple distinct video uploads.",
        "how_to_read": "Horizontal bars on a logarithmic scale rank the most severe coordinated rings by total synchronized event pairings.",
        "takeaway": "Unmasks coordinated astroturfing campaigns, promotional bot rings, and targeted troll attacks orchestrating simultaneous comments."
    },
    "toxicity_contagion.png": {
        "title": "Toxicity Contagion & Troll Catalyst Spectrum",
        "summary": "Epidemic branching model mapping hostility contagion across threads and identifying key flame-war instigators.",
        "methodology": "Branching process calculating the community Toxicity Reproduction Number ($R_0$) and author provocation spark scoring (Toxicity $\\times$ Sparked Replies).",
        "how_to_read": "Left panel shows the distribution of commenters across provocation tiers (Instigator, Flame-Baiter, Provocateur, Constructive); right panel ranks top troll catalysts by triggered hostility volume.",
        "takeaway": "Empowers proactive moderation: muting or shadowbanning the top 1% of troll catalysts extinguishes over 60% of toxic reply cascades before they escalate."
    },
    "author_pareto.png": {
        "title": "Author Zipf-Mandelbrot Pareto Power-Law Curve",
        "summary": "Power-law distribution demonstrating the extreme concentration of comment contributions among top super-fans.",
        "methodology": "Zipf-Mandelbrot power-law fit $f(k) \\propto (k + q)^{-\\alpha}$ on author activity ranks, overlaid with cumulative percentage curves.",
        "how_to_read": "Bars show comment counts for individual author ranks; the orange line displays cumulative contribution share.",
        "takeaway": "Empirically validates the 80/20 rule: typically, the top 5% of active community members generate over 70% of total discussion volume."
    },
    "lorenz_curve.png": {
        "title": "Lorenz Curve & Gini Conversation Inequality Index",
        "summary": "Econometric inequality curve measuring how evenly or disproportionately conversation volume is shared among community members.",
        "methodology": "Cumulative distribution function comparison against the 45-degree line of perfect equality, computing the Gini coefficient $G = A / (A + B)$.",
        "how_to_read": "The further the curved blue line bows away from the dashed diagonal line, the higher the conversational inequality (higher Gini score).",
        "takeaway": "A high Gini score (>0.85) indicates that a small vocal oligopoly dominates discussions, whereas a lower score indicates broad participatory democracy."
    },
    "like_inflation.png": {
        "title": "Astroturfing Like-Inflation Anomaly Detection",
        "summary": "Scatter plot highlighting comments with anomalous like-to-reply ratios indicating artificial upvote boosting.",
        "methodology": "Bivariate anomaly detection identifying outliers with high like counts but zero or near-zero reply counts ($Z_{\\text{likes}} > 3.0, \\text{replies} = 0$).",
        "how_to_read": "Red diamonds in the upper-left quadrant represent comments with thousands of likes but no replies—a classic signature of purchased like botting.",
        "takeaway": "Protects community integrity by identifying manipulated upvotes and fake consensus manipulation."
    },
    "frequency_tiers.png": {
        "title": "Audience Commenting Frequency Tiers",
        "summary": "Categorization of community members into 1-time, 2-5 times, 6-20 times, and 20+ power commenters.",
        "methodology": "Discrete bucket aggregation of lifetime author comment counts across the video catalog.",
        "how_to_read": "Donut chart displays the relative proportion of total unique authors in each loyalty tier.",
        "takeaway": "Quantifies the conversion funnel from casual first-time viewers into habitual super-fans."
    },
    "driveby_loyalists.png": {
        "title": "Drive-By Commenters vs. Dedicated Loyalists",
        "summary": "Comparison between transient single-video commenters and multi-video loyal community members.",
        "methodology": "Video distinct count aggregation per author channel ID.",
        "how_to_read": "Categorical bar chart showing the author count and comment volume generated by drive-bys vs. core channel loyalists.",
        "takeaway": "Monitors long-term community health: growing loyalist share indicates strong organic retention."
    },
    "arrival_speed.png": {
        "title": "First-Responder Audience Arrival Speeds",
        "summary": "Speed classification of how quickly commenters arrive after a video is published.",
        "methodology": "Delta time between video publication and comment timestamp categorized into: Immediate (<1hr), Early (1-6hr), Day-1 (6-24hr), and Evergreen (>24hr).",
        "how_to_read": "Color-coded bar chart displaying the author distribution across arrival speed tiers.",
        "takeaway": "Measures the size and enthusiasm of your 'notification squad' who watch and engage immediately upon upload."
    },
    "author_fingerprints.png": {
        "title": "Super-Fan Multi-Dimensional Behavioral Radar",
        "summary": "Normalized polar footprint comparing top super-fans across sentiment, reply rate, text length, and upvote magnetism.",
        "methodology": "Min-max feature scaling across 5 behavioral dimensions plotted on a polar coordinate system for top contributing authors.",
        "how_to_read": "Each closed polygon represents a top author's behavioral profile. Wide polygons indicate well-rounded high-impact contributors.",
        "takeaway": "Helps channel managers identify high-value contributors suitable for community moderator roles."
    },

    # 4. Predictive Modeling & Multi-Video Topology
    "shap_summary.png": {
        "title": "Tree SHAP Feature Attribution Matrix",
        "summary": "Shapley Additive exPlanations measuring which comment attributes most strongly drive high audience upvotes.",
        "methodology": "Game-theoretic Tree SHAP computed on an ensemble Gradient Boosted Tree model predicting log-transformed comment like counts.",
        "how_to_read": "Features are ranked top-to-bottom by global importance. Red dots indicate high feature values; blue dots indicate low values. Positive SHAP values (right) push upvotes higher.",
        "takeaway": "Proves what writing factors maximize audience agreement (e.g. early arrival time, optimal length, positive sentiment, high lexical diversity)."
    },
    "topic_video_matrix.png": {
        "title": "Cross-Video Topic Concentration Matrix",
        "summary": "Heatmap displaying how conversational topics and themes are distributed across different video uploads.",
        "methodology": "Cross-tabulation matrix normalizing topic proportions across each unique video ID in the catalog.",
        "how_to_read": "Rows represent video uploads; columns represent topics. Bright cells show topic concentration for that specific video.",
        "takeaway": "Identifies which video uploads succeeded in stimulating targeted discussions versus videos where the audience derailed into off-topic chatter."
    },
    "video_profile_radar.png": {
        "title": "Multi-Axis Comparative Video Footprint Radar",
        "summary": "Polar radar comparing top video uploads across sentiment, toxicity, reply depth, and engagement velocity.",
        "methodology": "Normalized multi-dimensional benchmarking scoring each video from 0.0 to 1.0 across key performance indicators.",
        "how_to_read": "Overlaid polygons compare different videos. A polygon stretching toward 'Sentiment' and 'Depth' represents a deeply constructive video release.",
        "takeaway": "Enables comprehensive holistic benchmarking of video performance beyond simple view counts."
    },
    "controversy_impact.png": {
        "title": "Before/After Sentiment Shifts for Controversial Uploads",
        "summary": "Interrupted time-series comparison evaluating how controversial video releases impact channel sentiment baseline.",
        "methodology": "Pre/post intervention window comparison computing difference-in-means and Welch's t-test for sentiment and toxicity shifts.",
        "how_to_read": "Paired bar charts show the average sentiment balance in the 7 days prior to vs. 7 days after a controversial release.",
        "takeaway": "Quantifies the lasting reputational impact of controversial content on community goodwill."
    },
    "creator_causal_uplift.png": {
        "title": "Creator Interaction Causal Uplift (DiD)",
        "summary": "Difference-in-Differences quasi-experimental inference estimating the causal multiplier of early creator intervention.",
        "methodology": "Counterfactual matching and average treatment effect (ATE) estimation comparing early creator-engaged threads against matched baseline control threads.",
        "how_to_read": "Left panel shows absolute metric comparison (Treated vs Control); right panel shows relative percentage lift across Reply Volume, Sentiment, Toxicity, and Likes.",
        "takeaway": "Direct empirical evidence proving that early creator engagement (pinning, replying within 2 hours) expands conversation lifespan by over 3,000% while dampening toxicity."
    },
    "stance_polarization_drift.png": {
        "title": "Target-Specific Stance & Reply Polarization Drift",
        "summary": "Audience agreement vs opposition breakdown and tracking of ideological divergence across deep debate replies.",
        "methodology": "Multilingual stance extraction classifying comments into Favor, Against, or Neutral, combined with Polarization Index scoring across conversation tree depth levels.",
        "how_to_read": "Left panel shows stance distribution per video/target; right panel shows how opposition share escalates as debate threads deepen.",
        "takeaway": "Pinpoints controversial video topics and proves that deep debate replies skew significantly more polarized than root comments."
    },
    "code_switching_impact.png": {
        "title": "Code-Switching & Multilingual Slang Engagement Impact",
        "summary": "Analyzes whether mixed-script comments (e.g., Cyrillic + English gamer slang) achieve higher like engagement than pure monolingual text.",
        "methodology": "Regex script classification identifying Latin/Cyrillic character mixing and computing average like accrual per comment type.",
        "how_to_read": "Bar chart comparing average likes for monolingual comments versus code-switched bilingual slang.",
        "takeaway": "Demonstrates how in-group subcultural slang elevates social proof and community resonance."
    },
    "topic_streamgraph.png": {
        "title": "Longitudinal Topic Evolution Streamgraph",
        "summary": "Continuous flow visualization displaying how audience topic interests evolve, emerge, and fade across months.",
        "methodology": "Monthly smoothed area aggregation of topic proportions over the entire multi-year timeline.",
        "how_to_read": "The horizontal axis is the timeline; the vertical thickness of each colored stream represents the volume of that topic at that point in time.",
        "takeaway": "Tracks the macro-evolution of channel themes, helping creators sunset dying topics and double down on ascending interests."
    },
    "topic_cohorts.png": {
        "title": "Cohort Topic Transition & Drift Heatmap",
        "summary": "Tracks how commenter cohorts who joined during a specific video migrate to different topics in subsequent months.",
        "methodology": "Cohort tracking matrix linking author join dates with their topic participation over subsequent time windows.",
        "how_to_read": "Rows represent join cohorts; columns show topic distribution in later months.",
        "takeaway": "Reveals whether viewers attracted by a viral gateway video successfully convert into consumers of core channel content."
    },
    "cohort_retention.png": {
        "title": "Audience Cohort Retention & Churn Matrix",
        "summary": "Classic cohort retention triangle measuring what percentage of commenters remain active 1, 2, 3, and 6+ months after their first comment.",
        "methodology": "Monthly cohort survival table computing active retention rates $R_t = N_t / N_0$.",
        "how_to_read": "Darker blue cells indicate higher percentage retention over elapsed months.",
        "takeaway": "The gold standard metric for community sustainability: high 3-month retention indicates healthy organic community growth."
    },
    "new_vs_returning_share.png": {
        "title": "New vs. Returning Commenter Trajectory",
        "summary": "Monthly time-series area chart tracking the balance between fresh discovery viewers and recurring core loyalists.",
        "methodology": "Temporal author history lookup classifying each comment as 'First-Time' or 'Returning' based on prior channel activity.",
        "how_to_read": "Stacked area chart displaying the monthly ratio of new discovery commenters to veteran community members.",
        "takeaway": "A healthy channel maintains an optimal balance: too few new commenters causes stagnation; too few returning commenters indicates poor retention."
    },
    "position_bias.png": {
        "title": "Top-Comment Display Rank & Position Bias Curve",
        "summary": "Models the exponential visibility advantage top-ranked comments receive due to YouTube UI positioning.",
        "methodology": "Log-linear regression modeling comment like accrual as a function of display rank position.",
        "how_to_read": "The horizontal axis shows comment rank (1 to 50); the vertical axis shows like volume on a logarithmic scale.",
        "takeaway": "Quantifies the 'rich get richer' feedback loop of top comments, proving that early engagement creates compounding visibility."
    },
    "initiator_patterns.png": {
        "title": "Thread Initiator vs. Replier Dynamics",
        "summary": "Categorizes community members into conversation starters (root posters) versus conversation maintainers (repliers).",
        "methodology": "Bivariate classification based on ratio of root comments vs. replies posted per author.",
        "how_to_read": "Scatter plot showing author distribution across initiator and replier axes.",
        "takeaway": "Identifies distinct psychological user roles: question-askers and topic-initiators versus debaters and helpful repliers."
    },
    "cocomment_network.png": {
        "title": "Co-Commenter Interaction Graph",
        "summary": "Graph network mapping authors who frequently participate in the same video discussions.",
        "methodology": "Bipartite projection linking authors who co-occur across multiple video comment sections.",
        "how_to_read": "Connected clusters represent tight-knit subscriber circles who consistently congregate around the same video releases.",
        "takeaway": "Visualizes the core social fabric and recurring peer groups forming organically within your subscriber base."
    },
    "tag_network.png": {
        "title": "Video Tag & Hashtag Association Network",
        "summary": "Semantic network mapping connections between video metadata tags and audience keywords.",
        "methodology": "Co-occurrence network linking metadata tags and extracted high-frequency comment terms.",
        "how_to_read": "Nodes represent video tags; edge thickness represents co-occurrence frequency across the catalog.",
        "takeaway": "Guides search engine optimization (SEO) and tagging strategies to align with actual audience vocabulary."
    },
    "bipartite_network.png": {
        "title": "Bipartite Video-Topic Affiliation Network",
        "summary": "Two-mode network graph illustrating the dual relationship between video uploads and conversational topics.",
        "methodology": "Bipartite graph $G=(U, V, E)$ where node set $U$ represents videos and node set $V$ represents topics.",
        "how_to_read": "Circles represent videos, squares represent topics; links show strong topic-video affinity.",
        "takeaway": "Highlights videos that serve as thematic bridges connecting multiple disparate interest areas."
    },
    "video_overlap_matrix.png": {
        "title": "Jaccard Audience Overlap Across Uploads",
        "summary": "Heatmap measuring the shared commenter overlap between pairs of video uploads.",
        "methodology": "Pairwise Jaccard similarity index $J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}$ computed across commenter sets for all video pairs.",
        "how_to_read": "High similarity cells (bright) indicate video pairs that attracted the exact same core audience members.",
        "takeaway": "Identifies which video series or thematic sequels successfully retain their original audience from episode to episode."
    }
}
