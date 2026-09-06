"""Unit and integration tests for Audience Churn & Longitudinal Cohort Survival Engine (tests/test_cohort_survival.py)."""

import json
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pytest

from engine.cohort_survival import (
    AudienceCohortEngine,
    AudienceCohortIntelligenceReport,
    CohortRetentionMatrix,
    CommunityQuickRatioReport,
    KaplanMeierSurvivalCurve,
    MonthlyStateDynamics,
    StratifiedSurvivalComparison,
    SurvivalMilestone,
)


@pytest.fixture
def synthetic_comments_df() -> pd.DataFrame:
    """Generates a small controlled multi-month comments dataset."""
    dates = [
        # 2026-01: Author 1 and 2 arrive
        ("c1", "a1", "2026-01-10 10:00:00", 5, 1, 0.4),
        ("c2", "a2", "2026-01-15 11:00:00", 0, 0, -0.2),
        # 2026-02: Author 1 repeats, Author 3 arrives
        ("c3", "a1", "2026-02-12 12:00:00", 2, 0, 0.5),
        ("c4", "a3", "2026-02-18 14:00:00", 1, 1, 0.1),
        # 2026-03: Author 1 repeats, Author 2 resurrects, Author 4 arrives
        ("c5", "a1", "2026-03-05 09:00:00", 10, 2, 0.8),
        ("c6", "a2", "2026-03-20 16:00:00", 1, 0, -0.1),
        ("c7", "a4", "2026-03-22 18:00:00", 0, 0, 0.0),
        # 2026-04: Author 3 repeats, Author 5 arrives
        ("c8", "a3", "2026-04-10 15:00:00", 0, 0, 0.2),
        ("c9", "a5", "2026-04-15 17:00:00", 3, 1, 0.6),
    ]
    df = pd.DataFrame(dates, columns=[
        "comment_id", "author_channel_id", "published_at", "like_count", "reply_count", "vader_compound"
    ])
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def test_cohort_matrix_calculation(synthetic_comments_df):
    """Verifies monthly cohort retention calculation and percentage matrix."""
    engine = AudienceCohortEngine()
    matrix = engine.compute_cohort_matrix(synthetic_comments_df, granularity="month", min_cohort_size=1)

    assert isinstance(matrix, CohortRetentionMatrix)
    assert matrix.granularity == "month"
    assert "2026-01" in matrix.cohort_labels
    assert "2026-02" in matrix.cohort_labels

    # 2026-01 had a1 and a2 (size 2)
    assert matrix.cohort_sizes["2026-01"] == 2
    # M+0 must always be 100%
    assert matrix.retention_matrix_pct["2026-01"]["M+0"] == 100.0

    # In M+1 (Feb), only a1 was active (1 / 2 = 50%)
    assert matrix.retention_matrix_pct["2026-01"]["M+1"] == 50.0

    # In M+2 (March), both a1 and a2 were active (2 / 2 = 100%)
    assert matrix.retention_matrix_pct["2026-01"]["M+2"] == 100.0

    # DataFrame conversion
    df_mat = matrix.to_dataframe()
    assert isinstance(df_mat, pd.DataFrame)
    assert "Cohort" in df_mat.columns
    assert "M+0" in df_mat.columns


def test_quarterly_cohort_granularity(synthetic_comments_df):
    """Verifies quarterly cohort matrix generation."""
    engine = AudienceCohortEngine()
    matrix = engine.compute_cohort_matrix(synthetic_comments_df, granularity="quarter", min_cohort_size=1)

    assert matrix.granularity == "quarter"
    assert "Q+0" in matrix.offset_labels


def test_survival_curve_fitting(synthetic_comments_df):
    """Verifies Kaplan-Meier churn survival curve generation and milestones."""
    engine = AudienceCohortEngine()
    surv = engine.compute_survival_curve(synthetic_comments_df, churn_inactivity_days=30.0)

    assert isinstance(surv, KaplanMeierSurvivalCurve)
    assert surv.total_authors == 5
    assert surv.churned_authors + surv.censored_authors == 5
    assert len(surv.milestones) > 0

    # Check non-increasing survival probabilities
    for i in range(len(surv.survival_prob) - 1):
        assert surv.survival_prob[i] >= surv.survival_prob[i + 1] - 1e-5

    # Check milestones structure
    m30 = next((m for m in surv.milestones if m.days == 30), None)
    assert m30 is not None
    assert 0.0 <= m30.survival_probability <= 1.0


def test_quick_ratio_dynamics(synthetic_comments_df):
    """Verifies community state classification and Quick Ratio math."""
    engine = AudienceCohortEngine()
    qr_report = engine.compute_quick_ratio(synthetic_comments_df)

    assert isinstance(qr_report, CommunityQuickRatioReport)
    assert len(qr_report.time_series) == 4  # Jan, Feb, Mar, Apr

    # In March 2026:
    # a1 was retained (active in Feb and active in Mar)
    # a2 was resurrected (active in Jan, inactive in Feb, active in Mar)
    # a4 was new
    mar = next((item for item in qr_report.time_series if item.period == "2026-03"), None)
    assert mar is not None
    assert mar.new_authors == 1
    assert mar.retained_authors == 1
    assert mar.resurrected_authors == 1
    assert mar.active_total == 3

    # Check state conservation: Active Total == New + Retained + Resurrected
    for item in qr_report.time_series:
        assert item.active_total == item.new_authors + item.retained_authors + item.resurrected_authors

    df_qr = qr_report.to_dataframe()
    assert isinstance(df_qr, pd.DataFrame)
    assert "Quick Ratio" in df_qr.columns


def test_stratified_validation_uplift(synthetic_comments_df):
    """Verifies stratified social validation survival uplift analysis."""
    engine = AudienceCohortEngine()
    strat = engine.compute_stratified_validation_survival(synthetic_comments_df, churn_inactivity_days=30.0)

    assert isinstance(strat, StratifiedSurvivalComparison)
    assert "Validated" in strat.curves
    assert "Ignored" in strat.curves
    assert len(strat.insight_summary) > 0


def test_stratified_sentiment_survival(synthetic_comments_df):
    """Verifies initial sentiment valence survival comparison."""
    engine = AudienceCohortEngine()
    strat = engine.compute_stratified_sentiment_survival(synthetic_comments_df, churn_inactivity_days=30.0)

    assert isinstance(strat, StratifiedSurvivalComparison)
    assert "Positive" in strat.curves
    assert "Negative" in strat.curves


def test_mock_report_generation():
    """Verifies synthetic mock report completeness and schema conformity."""
    report = AudienceCohortEngine.generate_mock_report()

    assert isinstance(report, AudienceCohortIntelligenceReport)
    assert report.total_comments_analyzed > 0
    assert report.total_unique_authors > 0
    assert report.median_community_lifespan_days > 0.0
    assert report.day_30_retention_pct > 0.0
    assert report.day_90_retention_pct > 0.0
    assert len(report.cohort_matrix.cohort_labels) == 8
    assert report.quick_ratio_report.avg_quick_ratio > 0.0
    assert report.stratified_social_validation.median_survival_lift_days > 0.0


def test_json_and_csv_serialization():
    """Verifies report serialization to JSON and CSV formats."""
    report = AudienceCohortEngine.generate_mock_report()
    json_str = report.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert "cohort_matrix" in parsed
    assert "overall_survival" in parsed
    assert "quick_ratio_report" in parsed

    csv_dict = report.export_csv_matrices()
    assert "cohort_retention_matrix.csv" in csv_dict
    assert "community_quick_ratio_series.csv" in csv_dict
    assert len(csv_dict["cohort_retention_matrix.csv"]) > 20


def test_full_report_with_tempdir(synthetic_comments_df):
    """Verifies generate_full_report saves json output correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        engine = AudienceCohortEngine(interim_dir=tmp_path, output_dir=tmp_path)

        report = engine.generate_full_report(df=synthetic_comments_df, churn_inactivity_days=30.0)
        assert isinstance(report, AudienceCohortIntelligenceReport)

        out_file = tmp_path / "audience_cohort_intelligence_report.json"
        assert out_file.exists()


def test_cli_execution_mock():
    """Verifies CLI execution with various argument flags in mock mode."""
    from engine.cohort_survival import main
    import sys

    # Test running without exceptions
    orig_argv = sys.argv
    try:
        sys.argv = ["ytint-cohort", "--mock", "--matrix", "--survival", "--quick-ratio", "--stratified"]
        main()
    finally:
        sys.argv = orig_argv
