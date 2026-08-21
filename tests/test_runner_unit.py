"""Unit tests for the pipeline runner orchestrator (runner.py)."""

import pytest
from pathlib import Path
from pipeline.runner import PipelineRunner


def test_pipeline_runner_init():
    """Verify PipelineRunner initializes registry and workspace paths."""
    runner = PipelineRunner()
    assert runner.config is not None
    assert runner.registry is not None
    assert "s00" in runner.registry
    assert "s99" in runner.registry
    assert runner.ordered_stages[-1] == "s99"  # Visualizations must be strictly last


def test_stage_resolution_and_aliases():
    """Verify stage resolving handles canonical IDs and named convenience aliases."""
    runner = PipelineRunner()
    assert runner.resolve_stage_id("s00") == "s00"
    assert runner.resolve_stage_id("s06") == "s06"  # Audience Intent
    assert runner.resolve_stage_id("s16") == "s16"  # Network
    assert runner.resolve_stage_id("s99") == "s99"  # Visualizations
    assert runner.resolve_stage_id("visualize") == "s99"
    assert runner.resolve_stage_id("s_visualize") == "s99"
    assert runner.resolve_stage_id("ingest") == "s00"
    assert runner.resolve_stage_id("network") == "s16"
    assert runner.resolve_stage_id("cib") == "s25"
    assert runner.resolve_stage_id("uplift") == "s39"


def test_stage_requires_execution_missing_file(tmp_path, monkeypatch):
    """Verify stage_requires_execution logic when output artifacts are missing."""
    runner = PipelineRunner()
    # Mock output path for a stage to a non-existent temp file
    missing_file = tmp_path / "non_existent_output.parquet"
    monkeypatch.setitem(runner.registry["s00"], "outputs", [missing_file])

    assert runner.stage_requires_execution("s00") is True


def test_stage_requires_execution_valid_file(tmp_path, monkeypatch):
    """Verify stage_requires_execution logic when output artifacts exist."""
    runner = PipelineRunner()
    valid_file = tmp_path / "valid_output.parquet"
    valid_file.touch()
    monkeypatch.setitem(runner.registry["s00"], "outputs", [valid_file])
    monkeypatch.setattr(runner, "raw_db", tmp_path / "non_existent_db.sqlite3")

    assert runner.stage_requires_execution("s00") is False
