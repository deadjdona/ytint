# tests/test_pipeline.py
import os
import pathlib
import pytest
import pandas as pd

# Define the root path relative to this test file
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"

EXPECTED_FILES = {
    "ingest_comments": DATA_DIR / "interim" / "comments_clean.parquet",
    "ingest_videos": DATA_DIR / "interim" / "videos_clean.parquet",
    "topics": DATA_DIR / "output" / "topic_metadata.parquet",
    "historical_timeline": DATA_DIR / "output" / "historical_timeline.parquet",
    "viral_events": DATA_DIR / "output" / "viral_events.parquet",
}

def test_data_directory_exists():
    """Verify the core data pipeline directory tree exists."""
    assert DATA_DIR.exists(), f"Root data directory missing at {DATA_DIR}"
    assert (DATA_DIR / "output").exists(), "data/output/ subdirectory is missing"

@pytest.mark.parametrize("stage,file_path", EXPECTED_FILES.items())
def test_artifact_existence(stage, file_path):
    """Ensure every pipeline stage wrote its expected output target."""
    assert file_path.exists(), (
        f"Missing output for Stage [{stage}]. "
        f"Expected file at: {file_path.relative_to(ROOT_DIR)}"
    )

@pytest.mark.parametrize("stage,file_path", EXPECTED_FILES.items())
def test_artifact_not_empty(stage, file_path):
    """All persisted layers except optional anomaly rows contain records."""
    if not file_path.exists():
        pytest.skip(f"Skipping empty check: {stage} file missing.")

    df = pd.read_parquet(file_path)
    if stage != "viral_events":
        assert not df.empty, f"Artifact for Stage [{stage}] is empty (0 rows)."


# Anomaly detection can legitimately produce zero rows when no date crosses the
# configured threshold. Its artifact remains a required, schema-stable output.
def test_viral_events_schema_is_stable():
    events = pd.read_parquet(EXPECTED_FILES["viral_events"])
    assert list(events.columns) == ["date", "comment_count", "z_score"]
    assert pd.api.types.is_datetime64_any_dtype(events["date"])
    assert pd.api.types.is_numeric_dtype(events["comment_count"])
    assert pd.api.types.is_numeric_dtype(events["z_score"])