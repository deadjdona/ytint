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
    assert "s06" in runner.registry


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
