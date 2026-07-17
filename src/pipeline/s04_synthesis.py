import pandas as pd
import pathlib
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("s04_synthesis")

def get_project_root() -> pathlib.Path:
    """Dynamically traverses upwards from this file to find the project root anchor."""
    current_dir = pathlib.Path(__file__).resolve().parent
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "data").is_dir() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("Could not dynamically resolve project root. Missing 'data' or 'src' anchor.")

def compile_ui_metrics():
    """
    Stage 04: Cross-Layer Aggregations & UI Metric Synthesis
    Fuses enriched comment features (likes, sentiment) with the topic metadata
    to generate the final structured layers for the Streamlit dashboard.
    """
    # Dynamically resolve paths
    PROJECT_ROOT = get_project_root()
    data_dir = PROJECT_ROOT / "data"
    
    interim_dir = data_dir / "interim"
    output_dir = data_dir / "output"
    
    comments_path = interim_dir / "comments_clean.parquet"
    topics_path = output_dir / "topic_metadata.parquet"
       
    # --------------------------------------------------------------------------
    # Pre-Flight Checks
    # --------------------------------------------------------------------------
    if not comments_path.exists() or not topics_path.exists():
        logger.error("Missing required parquet files. Ensure Stage 01 and 02 have completed successfully.")
        return

    phases = [
        "Loading Interim Data Layers",
        "Calculating Consensus & Engagement Vectors",
        "Extracting Emotional Polarity Matrices",
        "Synthesizing & Exporting Final UI Layer"
    ]
    
    with tqdm(total=len(phases), desc="Initializing Stage 04", bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
        
        # ======================================================================
        # Phase 1: Load Data
        # ======================================================================
        pbar.set_description(f"📥 {phases[0]}")
        df_comments = pd.read_parquet(comments_path)
        df_topics = pd.read_parquet(topics_path)
        pbar.update(1)

        # ======================================================================
        # Phase 2: Calculate Consensus & Engagement (Like Metrics)
        # ======================================================================
        pbar.set_description(f"🔥 {phases[1]}")
        
        # Ensure we have the target columns
        if 'like_count' in df_comments.columns and 'topic' in df_comments.columns:
            engagement_metrics = df_comments.groupby('topic').agg(
                total_likes=('like_count', 'sum'),
                avg_likes=('like_count', 'mean'),
                max_likes=('like_count', 'max')
            ).reset_index()
        else:
            logger.warning("'like_count' or 'topic' missing. Skipping engagement metrics.")
            engagement_metrics = pd.DataFrame(columns=['topic', 'total_likes', 'avg_likes', 'max_likes'])
            
        pbar.update(1)

        # ======================================================================
        # Phase 3: Emotional Polarity Matrix (Sentiment Distributions)
        # ======================================================================
        pbar.set_description(f"🎭 {phases[2]}")
        
        if 'sentiment_label' in df_comments.columns and 'sentiment_confidence' in df_comments.columns:
            # 1. Percentage distribution of sentiments per topic
            sentiment_counts = df_comments.groupby(['topic', 'sentiment_label']).size().unstack(fill_value=0)
            
            # Prevent division by zero
            row_sums = sentiment_counts.sum(axis=1)
            sentiment_pct = sentiment_counts.div(row_sums.replace(0, 1), axis=0) * 100
            
            # Prepend 'pct_' to sentiment labels for clean column names
            sentiment_pct.columns = [f"pct_{str(col).lower()}" for col in sentiment_pct.columns]
            sentiment_pct = sentiment_pct.reset_index()
            
            # 2. Average model confidence per topic
            confidence_metrics = df_comments.groupby('topic').agg(
                avg_confidence=('sentiment_confidence', 'mean')
            ).reset_index()
        else:
            logger.warning("Sentiment columns missing. Skipping polarity matrix.")
            sentiment_pct = pd.DataFrame(columns=['topic'])
            confidence_metrics = pd.DataFrame(columns=['topic'])

        pbar.update(1)

        # ======================================================================
        # Phase 4: Merge & Export
        # ======================================================================
        pbar.set_description(f"💾 {phases[3]}")
        
        # Identify columns that might already exist from previous runs to prevent '_x' / '_y' duplication
        new_cols = ['total_likes', 'avg_likes', 'max_likes', 'avg_confidence'] + list(sentiment_pct.columns)
        new_cols = [c for c in new_cols if c != 'topic']
        df_topics = df_topics.drop(columns=[c for c in new_cols if c in df_topics.columns], errors='ignore')
        
        # Merge all dataframes on the Topic ID
        # (Topic ID in df_topics is 'Topic', but 'topic' in df_comments)
        if not engagement_metrics.empty:
            df_topics = df_topics.merge(engagement_metrics, left_on='Topic', right_on='topic', how='left').drop(columns=['topic'], errors='ignore')
            
        if not sentiment_pct.empty:
            df_topics = df_topics.merge(sentiment_pct, left_on='Topic', right_on='topic', how='left').drop(columns=['topic'], errors='ignore')
            
        if not confidence_metrics.empty:
            df_topics = df_topics.merge(confidence_metrics, left_on='Topic', right_on='topic', how='left').drop(columns=['topic'], errors='ignore')

        # Fill any NaN values introduced by the merge (e.g., if a topic has no comments with a specific sentiment)
        df_topics.fillna(0, inplace=True)

        # Save back to the output directory
        df_topics.to_parquet(topics_path)
        
        pbar.set_description("✅ Synthesis Complete")
        pbar.update(1)

if __name__ == "__main__":
    # Local manual testing block
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
    compile_ui_metrics(PROJECT_ROOT / "data")
