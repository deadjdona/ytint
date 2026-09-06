"""Unit tests for the Executive Intelligence Dossier Generator (src/engine/reporter.py)."""

import pytest
from pathlib import Path
from engine.reporter import ExecutiveReportGenerator, generate_executive_report


def test_reporter_instantiation():
    """Verify generator initializes cleanly with auto-resolved paths."""
    gen = ExecutiveReportGenerator()
    assert gen.interim_dir.exists()
    assert gen.output_dir.exists()
    assert isinstance(gen.plots_dir, Path)


def test_extract_metrics():
    """Verify metrics extraction schema and values from available pipeline layers."""
    gen = ExecutiveReportGenerator()
    metrics = gen.extract_metrics()

    # Core metadata
    assert "timestamp" in metrics
    assert "date_range_str" in metrics
    assert "total_comments" in metrics
    assert "total_authors" in metrics
    assert "total_videos" in metrics

    # Forensic indicators
    assert "threat_level" in metrics
    assert metrics["threat_level"] in ["LOW / NOMINAL", "MODERATE", "ELEVATED", "CRITICAL"]
    assert "threat_color" in metrics
    assert "bot_pct" in metrics
    assert "inflation_pct" in metrics
    assert "cib_clusters_count" in metrics

    # Numeric invariants
    assert metrics["total_comments"] >= 0
    assert metrics["total_authors"] >= 0
    assert metrics["total_videos"] >= 0
    assert metrics["bot_pct"] >= 0.0


def test_generate_html_with_plots():
    """Verify HTML generation contains required doctype, styling, and sections."""
    gen = ExecutiveReportGenerator()
    html = gen.generate_html(include_plots=True)

    assert "<!DOCTYPE html>" in html
    assert "ytint // Executive Intelligence Dossier" in html
    assert "@media print" in html
    assert "window.print()" in html
    assert "kpi-grid" in html
    assert "Forensic Threat & Astroturfing Assessment" in html
    assert "Audience Topology & Engagement Dynamics" in html
    assert "Visual Intelligence Evidence Gallery" in html
    assert "Executive Recommendations & Action Items" in html


def test_generate_html_without_plots():
    """Verify HTML generation succeeds and is lightweight when plots are omitted."""
    gen = ExecutiveReportGenerator()
    html = gen.generate_html(include_plots=False)

    assert "<!DOCTYPE html>" in html
    assert "ytint // Executive Intelligence Dossier" in html
    # Should not contain data:image base64 data URIs
    assert "data:image/png;base64," not in html


def test_export_report_file(tmp_path):
    """Verify export_report writes the HTML file to a target location."""
    gen = ExecutiveReportGenerator()
    target_file = tmp_path / "test_dossier.html"
    result_path = gen.export_report(output_filepath=target_file, include_plots=False)

    assert result_path.exists()
    assert result_path == target_file
    assert target_file.stat().st_size > 500


def test_reporter_resilience_empty_directory(tmp_path):
    """Verify reporter handles empty/missing directories gracefully without crashing."""
    empty_interim = tmp_path / "interim"
    empty_output = tmp_path / "output"
    empty_interim.mkdir()
    empty_output.mkdir()

    gen = ExecutiveReportGenerator(interim_dir=empty_interim, output_dir=empty_output)
    metrics = gen.extract_metrics()

    assert metrics["total_comments"] == 0
    assert metrics["total_authors"] == 0
    assert metrics["total_videos"] == 0
    assert metrics["threat_level"] == "LOW / NOMINAL"

    html = gen.generate_html(include_plots=True)
    assert "<!DOCTYPE html>" in html
    assert "ytint // Executive Intelligence Dossier" in html


def test_generate_executive_report_convenience(tmp_path):
    """Verify functional wrapper generate_executive_report."""
    target = tmp_path / "convenience_brief.html"
    exported = generate_executive_report(output_filepath=target, include_plots=False)
    assert exported.exists()
    assert exported.stat().st_size > 500
