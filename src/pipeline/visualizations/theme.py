"""Shared aesthetics, theme configuration, and Matplotlib backend setup."""

import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns

def setup_theme():
    """Configure publication-grade styling for Matplotlib and Seaborn."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Arial', 'Segoe UI Emoji', 'DejaVu Sans']

setup_theme()
