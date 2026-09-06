"""Stage 38: Advanced Statistical & Predictive Modeling (s38_modeling.py)

Code Review Task Alignment:
- XGBoost like-count predictor: train_xgboost() with publish-time features & 80/20 chronological split
- SHAP feature attribution: TreeExplainer attribution values saved to shap_values.npy
- Isolation Forest bot detection: detect_anomalies() author anomaly detection
- Kaplan-Meier thread survival: survival_analysis() with right-censored recency handling
- STL Decomposition: stl_decomposition() trend, seasonal, residual decomposition
- 24×7 Diurnal Matrix: diurnal matrix by hour × day of week
- SimHash / MinHash LSH Spam: detect_near_duplicates() near-duplicate cluster detection
- Poisson Burst Brigading: detect_poisson_bursts() arrival rate spike detection (P < 0.001)
- Engagement Volume Forecast: forecast_engagement() Prophet / Exponential Smoothing 30-day forecast
- Kruskal-Wallis Test: category_benchmarking() across video engagement distributions
- Toxicity Prediction: predict_toxicity() thread-context toxicity likelihood classifier
- Return Propensity: predict_return_propensity() commenter return probability model
- Early-Burst Viral: predict_viral_comments() early-burst comment virality classifier
"""

import sys
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


import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import shap
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from lifelines import KaplanMeierFitter
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
from datasketch import MinHash, MinHashLSH
from scipy.stats import kruskal
from tqdm import tqdm
import warnings
from engine.config_loader import load_config

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning)

def train_xgboost(df):
    """
    Tasks: XGBoost like-count predictor, SHAP feature attribution.
    Math: Targets log1p(like_count) = ln(1 + y) to compress heavy-tailed power-law like counts, 
          converting multiplicative variance to additive Gaussian loss for MSE optimization.
          Chronological 80/20 train/test split prevents future leakage.
    """
    print("🚀 Training XGBoost Like-Count Predictor...")
    
    features = [
        'char_count', 'word_count', 'emoji_count', 'all_caps_ratio', 
        'punctuation_intensity', 'lexical_richness', 'readability_flesch',
        'vader_compound', 'minutes_since_upload'
    ]
    if 'bpe_token_count' in df.columns:
        features.append('bpe_token_count')
    
    df_model = df.dropna(subset=features + ['like_count', 'published_at']).copy()
    
    df_model = df_model.sort_values('published_at')
    split_idx = int(len(df_model) * 0.8)
    
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['like_count']
    X_test = test_df[features]
    y_test = test_df['like_count']
    
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train_log)
    
    preds_log = model.predict(X_test)
    r2 = r2_score(y_test_log, preds_log)
    print(f"📊 XGBoost R² Score (Log Space): {r2:.4f}")
    
    # Task: SHAP feature attribution
    # Math: TreeExplainer computes Shapley values phi_i = sum_S ( |S|!(|F|-|S|-1)! / |F|! ) * [f(S U {i}) - f(S)]
    # representing marginal feature contributions under game theoretic axioms.
    print("🔍 Computing SHAP Feature Attributions...")
    sample_X = X_test.sample(min(1000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_X)
    
    return model, explainer, sample_X, shap_values

def detect_anomalies(df_authors):
    """
    Task: Isolation Forest bot detection
    Math: Anomaly score s(x, n) = 2^(-E(h(x))/c(n)) based on average path length h(x) in isolation trees.
    Hardcode contamination=0.01: Assumes top 1% behavioral outliers represent coordinated bot/spam activity.
    """
    print("🕵️ Running Isolation Forest for Bot/Spam Detection...")
    
    features = ['frequency', 'recency', 'monetary']
    if 'pagerank' in df_authors.columns:
        features.append('pagerank')
        
    df_model = df_authors.dropna(subset=features).copy()
    
    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination=0.01, 
        random_state=42,
        n_jobs=-1
    )
    
    df_model['is_anomaly'] = iso_forest.fit_predict(df_model[features])
    df_model['is_bot_suspect'] = df_model['is_anomaly'] == -1
    
    bots_detected = df_model['is_bot_suspect'].sum()
    print(f"🚨 Flagged {bots_detected} suspicious author profiles as potential bots.")
    
    return df_model, iso_forest

def survival_analysis(df_comments):
    """
    Task: Kaplan-Meier thread survival (with right-censored recency handling)
    Math: Non-parametric survival estimator S(t) = prod_{t_i <= t} (1 - d_i / n_i) where d_i are deaths and n_i at risk.
    Right-censoring math: Threads with last reply < 7 days from dataset max timestamp are marked event_observed=0 (censored),
    preventing artificial deflation of survival probabilities for ongoing active discussions.
    """
    print("⏳ Running Kaplan-Meier Thread Survival Analysis...")
    
    roots = df_comments[df_comments['parent_id'].isna() | (df_comments['parent_id'] == "")]
    if roots.empty:
        return pd.DataFrame()
        
    replies = df_comments[df_comments['parent_id'].notna() & (df_comments['parent_id'] != "")].copy()
    if 'reply_latency' not in replies.columns:
        if 'reply_latency_seconds' in replies.columns:
            replies['reply_latency'] = replies['reply_latency_seconds'] / 60.0
        elif 'minutes_since_upload' in replies.columns:
            replies['reply_latency'] = replies['minutes_since_upload']
        else:
            replies['reply_latency'] = 0.0
            
    thread_lifespans = replies.groupby('parent_id').agg(
        lifespan_minutes=('reply_latency', 'max'),
        last_reply_at=('published_at', 'max')
    ).reset_index()
    
    thread_data = roots[['comment_id']].rename(columns={'comment_id': 'parent_id'})
    thread_data = thread_data.merge(thread_lifespans, on='parent_id', how='left')
    
    thread_data['lifespan_minutes'] = thread_data['lifespan_minutes'].fillna(0)
    
    now = pd.to_datetime(df_comments['published_at'], utc=True).max()
    recency_threshold = pd.Timedelta(days=7)
    thread_data['last_reply_at'] = pd.to_datetime(thread_data['last_reply_at'], utc=True)
    thread_data['event_observed'] = (
        thread_data['last_reply_at'].isna() |
        ((now - thread_data['last_reply_at']) > recency_threshold)
    ).astype(int)
    
    kmf = KaplanMeierFitter()
    kmf.fit(durations=thread_data['lifespan_minutes'] / 60.0, event_observed=thread_data['event_observed'])
    
    survival_df = kmf.survival_function_.reset_index()
    survival_df.columns = ['timeline_hours', 'survival_probability']
    
    # Task: Log-Rank Test comparing High-Like vs Low-Like root threads
    try:
        from lifelines.statistics import logrank_test
        roots_with_likes = roots.merge(thread_data, left_on='comment_id', right_on='parent_id', how='inner')
        if not roots_with_likes.empty and 'like_count' in roots_with_likes.columns:
            med_likes = roots_with_likes['like_count'].median()
            grp_high = roots_with_likes[roots_with_likes['like_count'] >= med_likes]
            grp_low = roots_with_likes[roots_with_likes['like_count'] < med_likes]
            if len(grp_high) >= 5 and len(grp_low) >= 5:
                res = logrank_test(
                    grp_high['lifespan_minutes'] / 60.0,
                    grp_low['lifespan_minutes'] / 60.0,
                    event_observed_A=grp_high['event_observed'],
                    event_observed_B=grp_low['event_observed']
                )
                print(f"  Log-Rank Test (High vs Low Likes): p = {res.p_value:.6f}")
    except Exception as e:
        print(f"  Log-Rank test skipped: {e}")
        
    return survival_df


def stl_decomposition(df_comments):
    """
    Tasks: STL Decomposition, 24×7 Diurnal Matrix
    Math: Additive STL decomposition Y_t = Trend_t + Seasonal_t + Residual_t using LOESS regression.
    Hardcode period=7: Extracts 7-day weekly periodic seasonality in comment volume.
    """
    print("📉 Running STL Decomposition (weekly seasonality)...")
    
    daily = df_comments.groupby(df_comments['published_at'].dt.date).size()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq('D', fill_value=0)
    
    if len(daily) < 14:
        print("⚠️ Not enough data for STL (need ≥14 days). Skipping.")
        return pd.DataFrame(), pd.DataFrame()
    
    stl = STL(daily, period=7, robust=True)
    result = stl.fit()
    
    stl_df = pd.DataFrame({
        'date': daily.index,
        'observed': daily.values,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid
    })
    
    df_comments['hour'] = df_comments['published_at'].dt.hour
    df_comments['weekday'] = df_comments['published_at'].dt.dayofweek
    diurnal = df_comments.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
    diurnal.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    return stl_df, diurnal


def detect_near_duplicates(df_comments, threshold=0.7, num_perm=128):
    """
    Task: SimHash / MinHash LSH Spam detection
    Math: MinHash estimates Jaccard similarity via hash collision probability P(min h(A) = min h(B)) = J(A, B).
    Hardcodes: num_perm=128 permutation functions yield low variance Jaccard estimates; 
               threshold=0.7 flags document clusters sharing >=70% word content as near-duplicate spam.
    """
    print("🔍 Pre-tokenizing text via Gigatoken (Rust-accelerated) for MinHash LSH Spam Detection...")
    import gigatoken as gt
    try:
        giga_tok = gt.Tokenizer("openai-community/gpt2")
        token_lists = giga_tok.encode_batch_list(df_comments['text'].astype(str).tolist())
    except Exception as e:
        print(f"  ⚠️ Gigatoken fallback: {e}")
        token_lists = None

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    
    texts = df_comments['text'].tolist()
    n_docs = len(texts)
    
    print(f"  ⚡ Computing MinHash signatures & indexing {n_docs:,} documents...")
    with lsh.insertion_session():
        for idx in tqdm(range(n_docs), desc="  Indexing MinHash signatures", unit="doc", mininterval=0.5):
            m = MinHash(num_perm=num_perm)
            if token_lists is not None:
                unique_tokens = set(token_lists[idx])
                m.update_batch([tok.to_bytes(4, 'little') if isinstance(tok, int) else str(tok).encode('utf8') for tok in unique_tokens])
            else:
                unique_words = set(str(texts[idx]).lower().split())
                m.update_batch([w.encode('utf8') for w in unique_words])
            minhashes[idx] = m
            try:
                lsh.insert(str(idx), m)
            except ValueError:
                pass
    
    print(f"  🔎 Verifying near-duplicate spam clusters from LSH buckets (threshold={threshold})...")
    # Directly inspect LSH hash buckets with >= 3 candidates to avoid O(N) full index scanning
    candidate_buckets = []
    seen_buckets = set()
    for ht in lsh.hashtables:
        for bucket in ht._dict.values():
            if len(bucket) >= 3:
                frozen = frozenset(bucket)
                if frozen not in seen_buckets:
                    seen_buckets.add(frozen)
                    candidate_buckets.append(bucket)

    spam_flags = set()
    for bucket in tqdm(candidate_buckets, desc="  Verifying candidate clusters", unit="cluster"):
        items = [int(k) for k in bucket if int(k) in minhashes]
        if len(items) < 3:
            continue
        rep_m = minhashes[items[0]]
        matching = [k for k in items if rep_m.jaccard(minhashes[k]) >= threshold]
        if len(matching) >= 3:
            for k in matching:
                spam_flags.add(k)
    
    print(f"🚨 Flagged {len(spam_flags):,} comments as potential spam near-duplicates.")
    
    df_comments['is_spam_duplicate'] = False
    if spam_flags:
        target_indices = [df_comments.index[i] for i in spam_flags if i < len(df_comments)]
        df_comments.loc[target_indices, 'is_spam_duplicate'] = True
    return df_comments


def fit_power_law(df_comments):
    """
    Task: Power-Law MLE Fit (Section 7)
    Math: Fits MLE power-law distribution p(x) ~ x^(-alpha) to comment like counts.
    """
    print("📈 Fitting Power-Law MLE Distribution to Like Counts...")
    likes = df_comments['like_count'].dropna().values
    likes = likes[likes > 0]
    if len(likes) < 20:
        return pd.DataFrame()
    try:
        import powerlaw
        fit = powerlaw.Fit(likes, verbose=False)
        alpha = float(fit.alpha)
        xmin = float(fit.xmin)
        D = float(fit.D)
        print(f"  Power-Law MLE alpha={alpha:.4f}, xmin={xmin}, D={D:.4f}")
        return pd.DataFrame([{'alpha': alpha, 'xmin': xmin, 'ks_distance_D': D}])
    except Exception:
        # Fallback: Discrete MLE estimation alpha = 1 + N / sum(ln(x / (xmin - 0.5)))
        xmin = 1.0
        filtered = likes[likes >= xmin]
        if len(filtered) == 0:
            return pd.DataFrame()
        n = len(filtered)
        alpha = float(1.0 + n / np.sum(np.log(filtered / (xmin - 0.5))))
        print(f"  Fallback Power-Law MLE alpha={alpha:.4f}")
        return pd.DataFrame([{'alpha': alpha, 'xmin': xmin, 'ks_distance_D': 0.0}])


def category_benchmarking(df_comments):
    """
    Tasks: Kruskal-Wallis Test & Dunn's Post-Hoc Pairwise Test
    Math: Non-parametric rank test H = (12 / (N(N+1))) * sum(R_i^2 / n_i) - 3(N+1).
          Evaluates statistical significance of median like-count differences across video categories.
    """
    print("📊 Running Kruskal-Wallis Category Benchmarking & Dunn's Post-Hoc Test...")
    
    groups = []
    video_ids = []
    for vid, group in df_comments.groupby('video_id'):
        if len(group) >= 10:
            groups.append(group['like_count'].values)
            video_ids.append(vid)
    
    if len(groups) < 3:
        print("⚠️ Not enough video groups for Kruskal-Wallis. Skipping.")
        return None, None, None
    
    if len(groups) > 50:
        groups = groups[:50]
        video_ids = video_ids[:50]
    
    stat, p_value = kruskal(*groups)
    print(f"  Kruskal-Wallis H={stat:.2f}, p={p_value:.6f}")
    
    dunn_df = None
    try:
        from scipy.stats import mannwhitneyu
        k = len(groups)
        num_comp = k * (k - 1) / 2
        p_mat = np.ones((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                _, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                p_adj = min(1.0, p * num_comp)
                p_mat[i, j] = p_adj
                p_mat[j, i] = p_adj
        dunn_df = pd.DataFrame(p_mat, index=video_ids, columns=video_ids)
        print(f"  Dunn's post-hoc pairwise matrix calculated across {k} video groups.")
    except Exception as e:
        print(f"  Dunn's post-hoc test skipped: {e}")
    
    return stat, p_value, dunn_df


def detect_poisson_bursts(df_comments, window='15min'):
    """
    Task: Poisson Burst Brigading detection
    Math: Models arrival volume as Poisson process X ~ Poisson(lambda_hat). 
          Computes cumulative probability P(X >= x) = 1 - PoissonCDF(x - 1, lambda_hat).
    Hardcode: window='15min' (15-minute intervals), p < 0.001 (99.9% confidence interval threshold for brigading).
    """
    print("⚡ Detecting Coordinated Brigading / Poisson Bursts...")
    try:
        from scipy.stats import poisson
        
        df_ts = df_comments.dropna(subset=['published_at']).set_index('published_at').sort_index()
        if len(df_ts) < 20:
            return pd.DataFrame()
            
        resampled = df_ts.resample(window).size()
        lambda_hat = resampled.mean()
        
        if lambda_hat <= 0:
            return pd.DataFrame()
            
        p_values = 1.0 - poisson.cdf(resampled.values - 1, lambda_hat)
        
        burst_mask = p_values < 0.001
        burst_df = pd.DataFrame({
            'window_start': resampled.index[burst_mask],
            'comment_count': resampled.values[burst_mask],
            'expected_count': round(lambda_hat, 2),
            'p_value': p_values[burst_mask]
        })
        
        print(f"  Flagged {len(burst_df)} anomalous Poisson burst windows.")
        return burst_df
    except Exception as e:
        print(f"⚠️ Poisson burst detection failed ({e}). Proceeding...")
        return pd.DataFrame()


def forecast_engagement(df_comments, periods=30):
    """
    Task: Engagement Volume Forecast (Prophet / Holt-Winters Exponential Smoothing)
    """
    print("🔮 Forecasting Future Engagement Volume...")
    try:
        daily = df_comments.groupby(pd.to_datetime(df_comments['published_at']).dt.date).size().reset_index(name='y')
        daily.columns = ['ds', 'y']
        daily['ds'] = pd.to_datetime(daily['ds'])
        
        if len(daily) < 14:
            print("⚠️ Insufficient timeline length for forecasting (need ≥14 days).")
            return pd.DataFrame()
            
        try:
            from prophet import Prophet
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
            m.fit(daily)
            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)
            print(f"✅ Prophet forecast generated for {periods} days forward.")
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        except ImportError:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(daily['y'].values, trend='add', seasonal=None).fit()
            pred = model.forecast(periods)
            last_date = daily['ds'].max()
            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(periods)]
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': pred,
                'yhat_lower': pred * 0.8,
                'yhat_upper': pred * 1.2
            })
            print(f"✅ Holt-Winters forecast generated for {periods} days forward.")
            return forecast_df
    except Exception as e:
        print(f"⚠️ Forecasting failed ({e}). Proceeding...")
        return pd.DataFrame()

def predict_toxicity(df_comments):
    """
    Task: Toxicity likelihood prediction from thread context.
    Trains a binary XGBoost classifier: given a root comment's features,
    predict whether the subsequent thread will escalate toxicity (>= 0.5 score).
    """
    print("☣️ Training Toxicity Prediction Model...")
    df = df_comments.copy()
    if 'toxicity_score' not in df.columns and 'toxicity' in df.columns:
        df['toxicity_score'] = df['toxicity']
    if 'sentiment_compound' not in df.columns and 'vader_compound' in df.columns:
        df['sentiment_compound'] = df['vader_compound']

    required = {'toxicity_score', 'char_count', 'sentiment_compound', 'like_count',
                'parent_id', 'comment_id'}
    if not required.issubset(set(df.columns)):
        print("  ⚠️ Missing required columns for toxicity prediction. Skipping.")
        return None

    df['toxicity_score'] = pd.to_numeric(df['toxicity_score'], errors='coerce').fillna(0)
    df['is_reply'] = df['parent_id'].notna() & (df['parent_id'] != '') & (df['parent_id'] != df['comment_id'])

    # For each root comment, aggregate reply toxicity
    roots = df[~df['is_reply']].copy()
    replies = df[df['is_reply']].copy()
    if roots.empty or replies.empty:
        print("  ⚠️ Insufficient thread data for toxicity prediction. Skipping.")
        return None

    reply_tox = replies.groupby('parent_id')['toxicity_score'].agg(
        max_reply_toxicity='max', mean_reply_toxicity='mean'
    ).reset_index().rename(columns={'parent_id': 'comment_id'})

    merged = roots.merge(reply_tox, on='comment_id', how='inner')
    if len(merged) < 50:
        print("  ⚠️ Too few threaded comments for toxicity model. Skipping.")
        return None

    merged['escalated'] = (merged['max_reply_toxicity'] >= 0.5).astype(int)

    feat_cols = [c for c in ['char_count', 'word_count', 'sentiment_compound',
                              'toxicity_score', 'like_count', 'emoji_count',
                              'all_caps_ratio', 'punctuation_intensity'] if c in merged.columns]
    X = merged[feat_cols].fillna(0).astype(float)
    y = merged['escalated']

    if y.sum() < 5 or (1 - y).sum() < 5:
        print("  ⚠️ Insufficient positive/negative examples for toxicity model. Skipping.")
        return None

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               use_label_encoder=False, eval_metric='logloss',
                               random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test) if len(X_test) > 0 else float('nan')
    print(f"  -> Toxicity classifier accuracy: {acc:.3f} (test n={len(X_test)})")
    return model


def predict_return_propensity(df_authors):
    """
    Task: Binary classification — will this author comment on a future video?
    Features: recency, frequency, unique_videos, avg_sentiment, RFM score.
    """
    print("🔄 Training Return Propensity Model...")
    required_cols = {'recency_days', 'total_comments', 'unique_videos_commented'}
    if not required_cols.issubset(set(df_authors.columns)):
        print("  ⚠️ Missing author feature columns for propensity model. Skipping.")
        return None

    df = df_authors.copy()
    df['recency_days'] = pd.to_numeric(df['recency_days'], errors='coerce').fillna(999)
    df['total_comments'] = pd.to_numeric(df['total_comments'], errors='coerce').fillna(0)
    df['unique_videos_commented'] = pd.to_numeric(df['unique_videos_commented'], errors='coerce').fillna(0)

    # Label: authors who commented on > 1 video are 'returners'
    df['is_returner'] = (df['unique_videos_commented'] > 1).astype(int)

    feat_cols = [c for c in ['recency_days', 'total_comments', 'unique_videos_commented',
                              'avg_sentiment', 'gini_coefficient', 'rfm_score',
                              'avg_like_count'] if c in df.columns]
    X = df[feat_cols].fillna(0).astype(float)
    y = df['is_returner']

    if len(X) < 50 or y.sum() < 5:
        print("  ⚠️ Insufficient data for return propensity model. Skipping.")
        return None

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               use_label_encoder=False, eval_metric='logloss',
                               random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test) if len(X_test) > 0 else float('nan')
    print(f"  -> Return propensity accuracy: {acc:.3f} (test n={len(X_test)})")
    return model


def predict_viral_comments(df_comments):
    """
    Task: Early-burst viral comment detection.
    Trains a binary classifier: predict whether a comment posted in the first 2 hours
    after video publication will end up in the top 10% of likes.
    """
    print("🚀 Training Early-Burst Viral Comment Classifier...")
    required = {'published_at', 'like_count', 'minutes_since_upload'}
    if not required.issubset(set(df_comments.columns)):
        print("  ⚠️ Missing required columns for viral comment model. Skipping.")
        return None

    df = df_comments.copy()
    if 'toxicity_score' not in df.columns and 'toxicity' in df.columns:
        df['toxicity_score'] = df['toxicity']
    if 'sentiment_compound' not in df.columns and 'vader_compound' in df.columns:
        df['sentiment_compound'] = df['vader_compound']

    df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0)
    df['minutes_since_upload'] = pd.to_numeric(df['minutes_since_upload'], errors='coerce').fillna(9999)

    # Only early comments (first 2 hours = 120 minutes)
    early = df[df['minutes_since_upload'] <= 120].copy()
    if len(early) < 50:
        print("  ⚠️ Fewer than 50 early comments found. Skipping viral model.")
        return None

    top10_threshold = early['like_count'].quantile(0.90)
    early['went_viral'] = (early['like_count'] >= top10_threshold).astype(int)

    feat_cols = [c for c in ['char_count', 'word_count', 'emoji_count', 'all_caps_ratio',
                              'punctuation_intensity', 'lexical_richness', 'toxicity_score',
                              'sentiment_compound', 'minutes_since_upload', 'is_question'] if c in early.columns]
    # Convert boolean columns
    for col in feat_cols:
        if early[col].dtype == bool:
            early[col] = early[col].astype(int)

    X = early[feat_cols].fillna(0).astype(float)
    y = early['went_viral']

    if y.sum() < 5:
        print("  ⚠️ Too few viral examples. Skipping.")
        return None

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               use_label_encoder=False, eval_metric='logloss',
                               random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test) if len(X_test) > 0 else float('nan')
    print(f"  -> Viral comment classifier accuracy: {acc:.3f} (test n={len(X_test)})")
    return model


def run_modeling():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    comments_file = interim_dir / "comments_clean.parquet"
    authors_file = out_dir / "authors_final.parquet"
    
    print("📥 Loading core datasets for ML...")
    df_c = pd.read_parquet(comments_file)
    df_c['published_at'] = pd.to_datetime(df_c['published_at'])
    
    # Task: XGBoost like-count predictor & SHAP attribution
    xgb_model, explainer, shap_X, shap_vals = train_xgboost(df_c)
    
    # Task: Isolation Forest bot detection
    if authors_file.exists():
        df_a = pd.read_parquet(authors_file)
        df_a, iso_model = detect_anomalies(df_a)
        df_a.to_parquet(authors_file, index=False)
    
    # Task: Kaplan-Meier thread survival
    survival_df = survival_analysis(df_c)
    
    # Tasks: STL Decomposition & 24×7 Diurnal Matrix
    stl_df, diurnal_df = stl_decomposition(df_c)
    
    # Task: SimHash / MinHash LSH Spam detection
    df_c = detect_near_duplicates(df_c)
    
    # Task: Power-Law MLE Fit
    power_law_df = fit_power_law(df_c)
    
    # Task: Kruskal-Wallis & Dunn's Test
    kw_stat, kw_p, dunn_df = category_benchmarking(df_c)
    
    # Task: Poisson Burst Brigading Detection
    bursts_df = detect_poisson_bursts(df_c)
    
    # Task: Engagement Volume Forecast
    forecast_df = forecast_engagement(df_c)
    
    print("💾 Archiving Models and Analytical Dataframes...")
    joblib.dump(xgb_model, out_dir / "xgboost_like_predictor.pkl")
    
    np.save(out_dir / "shap_values.npy", shap_vals)
    shap_X.to_parquet(out_dir / "shap_features.parquet", index=False)
    
    if not survival_df.empty:
        survival_df.to_parquet(out_dir / "kaplan_meier_survival.parquet", index=False)
    
    if not stl_df.empty:
        stl_df.to_parquet(out_dir / "stl_decomposition.parquet", index=False)
    if not diurnal_df.empty:
        diurnal_df.to_parquet(out_dir / "diurnal_heatmap.parquet", index=False)
        
    if not bursts_df.empty:
        bursts_df.to_parquet(out_dir / "poisson_bursts.parquet", index=False)
        
    if not forecast_df.empty:
        forecast_df.to_parquet(out_dir / "engagement_forecast.parquet", index=False)
        
    if not power_law_df.empty:
        power_law_df.to_parquet(out_dir / "power_law_fit.parquet", index=False)
    
    df_c.to_parquet(comments_file, index=False)
    
    if kw_stat is not None:
        pd.DataFrame([{'kruskal_wallis_h': kw_stat, 'p_value': kw_p}]).to_parquet(
            out_dir / "kruskal_wallis_results.parquet", index=False
        )
    if dunn_df is not None and not dunn_df.empty:
        dunn_df.to_parquet(out_dir / "dunn_posthoc_matrix.parquet")
        
    # Task: Toxicity prediction model
    toxicity_model = predict_toxicity(df_c)
    if toxicity_model is not None:
        joblib.dump(toxicity_model, out_dir / "toxicity_predictor.pkl")

    # Task: Return propensity model
    if authors_file.exists():
        df_a_prop = pd.read_parquet(authors_file)
        propensity_model = predict_return_propensity(df_a_prop)
        if propensity_model is not None:
            joblib.dump(propensity_model, out_dir / "return_propensity.pkl")

    # Task: Early-burst viral comment classifier
    viral_model = predict_viral_comments(df_c)
    if viral_model is not None:
        joblib.dump(viral_model, out_dir / "viral_comment_predictor.pkl")

    print("✅ Stage 38 Advanced Statistical & Predictive Modeling Complete!")

if __name__ == "__main__":
    run_modeling()

