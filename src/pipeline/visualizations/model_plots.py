"""Phase 5: Predictive Modeling, Tree SHAP, Survival Analysis, and Causal DiD Uplift."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from .theme import setup_theme


def plot_kaplan_meier(df_survival, out_dir):
    """
    Task 1: Kaplan-Meier Survival Plot
    Generates the Thread Lifespan Survival Curve.
    """
    print("📈 Generating Kaplan-Meier Thread Survival Plot...")
    plt.figure()
    
    # Step plot is standard for Kaplan-Meier
    plt.step(df_survival['timeline_hours'], df_survival['survival_probability'], where="post", color="darkred", linewidth=2)
    plt.fill_between(df_survival['timeline_hours'], df_survival['survival_probability'], step="post", color="darkred", alpha=0.1)
    
    plt.title("Kaplan-Meier Survival Estimate of Thread Lifespans", pad=15)
    plt.xlabel("Time Since First Comment (Hours)")
    plt.ylabel("Probability of Receiving More Replies")
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Shows the probability of a thread continuing to receive replies over time. A steeper drop indicates shorter-lived discussions.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / "kaplan_meier_survival.png")
    plt.close() # 🛡️ Prevent Memory Leaks

def plot_shap_summary(shap_values, shap_features, out_dir):
    """
    Task 4: SHAP Feature Importance Summary Beeswarm
    Visualizes XGBoost Feature Importance via SHAP Values.
    """
    print("🧠 Generating SHAP XGBoost Attribution Summary...")
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(shap_values, shap_features, show=False)
    
    plt.title("SHAP Feature Importance (Predicting Like-Counts)", pad=15)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Highlights which linguistic or emotional features most strongly drive comment likes. Red dots indicate high feature values.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / "shap_summary.png")
    plt.close()

def plot_creator_uplift(out_dir, uplift_file):
    print("📈 Generating Creator Causal Uplift Charts...")
    df = pd.read_parquet(uplift_file)
    if df.empty: return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    df_melt = df.melt(id_vars=['dimension'], value_vars=['treated_mean', 'control_mean'], var_name='Cohort', value_name='Score')
    df_melt['Cohort'] = df_melt['Cohort'].map({'treated_mean': 'Creator Engaged (Treated)', 'control_mean': 'Baseline (Control)'})
    
    sns.barplot(data=df_melt, x='dimension', y='Score', hue='Cohort', palette=['#0066fe', '#888888'], ax=axes[0])
    axes[0].set_title('Absolute Metric Comparison: Treated vs. Baseline Threads', pad=15, fontsize=13)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Mean Metric Value')
    axes[0].tick_params(axis='x', rotation=15)
    
    colors = ['#00cc66' if v >= 0 else '#ff3366' for v in df['relative_lift_pct']]
    sns.barplot(data=df, x='dimension', y='relative_lift_pct', palette=colors, hue='dimension', legend=False, ax=axes[1])
    axes[1].set_title('Estimated Relative Causal Lift (%)', pad=15, fontsize=13)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Relative Lift (%)')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    plt.figtext(0.5, 0.015, "Explanation: Difference-in-Differences causal inference estimating the multiplier effect of creator intervention on thread engagement, sentiment, and toxicity.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / 'creator_causal_uplift.png')
    plt.close()

def plot_poisson_bursts(out_dir, bursts_file):
    """
    Task: Poisson Burst Brigading Visualization
    Plots statistically significant comment volume surges over time.
    """
    print("⚡ Generating Poisson Volumetric Bursts Plot...")
    df = pd.read_parquet(bursts_file)
    if df.empty:
        return

    df['window_start'] = pd.to_datetime(df['window_start'])
    df = df.sort_values('window_start')

    fig, ax = plt.subplots(figsize=(12, 6))
    
    expected = df['expected_count'].iloc[0] if 'expected_count' in df.columns else df['comment_count'].mean()
    intensity = df['comment_count'] / max(expected, 1.0)
    
    scatter = ax.scatter(
        df['window_start'], 
        df['comment_count'],
        c=intensity, 
        cmap='YlOrRd', 
        s=np.clip(intensity * 12, 20, 180),
        alpha=0.85,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.axhline(expected, color='gray', linestyle='--', linewidth=1.2, label=f'Expected Poisson Baseline (λ={expected:.1f})')
    
    # Highlight top 3 highest burst events
    top_bursts = df.nlargest(3, 'comment_count')
    for _, row in top_bursts.iterrows():
        ax.annotate(
            f"{int(row['comment_count'])} comments",
            xy=(row['window_start'], row['comment_count']),
            xytext=(0, 12),
            textcoords="offset points",
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#d90429',
            arrowprops=dict(arrowstyle="->", color='#d90429', lw=1.2)
        )
    
    cb = plt.colorbar(scatter, ax=ax, pad=0.02)
    cb.set_label('Burst Multiplier (Observed / Expected λ)', fontsize=10)
    
    ax.set_title("Statistically Significant Poisson Arrival Bursts (p < 0.001)", pad=15, fontsize=14)
    ax.set_xlabel("Timeline Window")
    ax.set_ylabel("15-Minute Comment Volume")
    ax.legend(loc="upper left")
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(0.5, 0.015, "Explanation: Statistically significant comment volume surges (p < 0.001 under Poisson arrival model), indicating coordinated brigading, viral shares, or external media mentions.", ha="center", fontsize=10, color="dimgray", wrap=True)
    plt.savefig(out_dir / "poisson_bursts.png")
    plt.close()

