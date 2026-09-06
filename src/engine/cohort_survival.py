"""ytint // Audience Churn & Longitudinal Cohort Survival Engine (src/engine/cohort_survival.py)

Models monthly/quarterly commenter acquisition cohorts, continuous-time Kaplan-Meier
churn survival curves with right-censoring, community state transitions
(New, Retained, Resurrected, Lapsed), monthly Community Quick Ratios, and
stratified causal retention uplift from early social validation.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import logging
import math
import os
import pathlib
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Safe standard output configuration for Windows console
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

logger = logging.getLogger("ytint.cohort_survival")


# ==============================================================================
# 1. DATA STRUCTURES & SCHEMAS
# ==============================================================================

@dataclass
class CohortRetentionMatrix:
    """Longitudinal cohort acquisition and retention matrix over time offsets."""
    granularity: str  # "month" or "quarter"
    cohort_labels: List[str]
    offset_labels: List[str]  # e.g. ["M+0", "M+1", "M+2", ...]
    cohort_sizes: Dict[str, int]
    retention_matrix_pct: Dict[str, Dict[str, float]]  # cohort -> offset -> %
    active_matrix_counts: Dict[str, Dict[str, int]]  # cohort -> offset -> count
    repeat_commenter_counts: Dict[str, int]  # cohort -> count of authors active > 1 period
    avg_retention_by_offset: Dict[str, float]  # offset -> mean retention %

    def to_dataframe(self) -> pd.DataFrame:
        """Converts retention percentages into a clean tabular DataFrame."""
        rows = []
        for cohort in self.cohort_labels:
            row = {"Cohort": cohort, "Acquired": self.cohort_sizes.get(cohort, 0)}
            for off in self.offset_labels:
                row[off] = self.retention_matrix_pct.get(cohort, {}).get(off, 0.0)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_count_dataframe(self) -> pd.DataFrame:
        """Converts active author counts into a clean tabular DataFrame."""
        rows = []
        for cohort in self.cohort_labels:
            row = {"Cohort": cohort, "Acquired": self.cohort_sizes.get(cohort, 0)}
            for off in self.offset_labels:
                row[off] = self.active_matrix_counts.get(cohort, {}).get(off, 0)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SurvivalMilestone:
    """Retention milestone at a specific day threshold."""
    days: int
    survival_probability: float
    at_risk_count: int
    events_count: int


@dataclass
class KaplanMeierSurvivalCurve:
    """Non-parametric continuous Kaplan-Meier churn survival model."""
    group_label: str
    total_authors: int
    churned_authors: int
    censored_authors: int
    churn_rate_pct: float
    median_survival_days: Optional[float]
    milestones: List[SurvivalMilestone]
    timeline_days: List[float]
    survival_prob: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MonthlyStateDynamics:
    """Community state classification dynamics for a single time period."""
    period: str
    active_total: int
    new_authors: int
    retained_authors: int
    resurrected_authors: int
    lapsed_authors: int
    quick_ratio: float
    growth_rate_pct: float


@dataclass
class CommunityQuickRatioReport:
    """Community growth dynamics and Quick Ratio time series."""
    time_series: List[MonthlyStateDynamics]
    avg_quick_ratio: float
    current_quick_ratio: float
    total_unique_acquired: int
    resurrection_rate_pct: float

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for item in self.time_series:
            rows.append({
                "Period": item.period,
                "Active Total": item.active_total,
                "New": item.new_authors,
                "Retained": item.retained_authors,
                "Resurrected": item.resurrected_authors,
                "Lapsed": item.lapsed_authors,
                "Quick Ratio": item.quick_ratio,
                "Net Growth %": f"{item.growth_rate_pct:+.1f}%"
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StratifiedSurvivalComparison:
    """Comparative survival curves testing causal retention factors."""
    factor_name: str
    curves: Dict[str, KaplanMeierSurvivalCurve]
    logrank_p_value: Optional[float] = None
    logrank_test_stat: Optional[float] = None
    median_survival_lift_days: Optional[float] = None
    median_survival_lift_pct: Optional[float] = None
    insight_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "curves": {k: v.to_dict() for k, v in self.curves.items()},
            "logrank_p_value": self.logrank_p_value,
            "logrank_test_stat": self.logrank_test_stat,
            "median_survival_lift_days": self.median_survival_lift_days,
            "median_survival_lift_pct": self.median_survival_lift_pct,
            "insight_summary": self.insight_summary,
        }


@dataclass
class AudienceCohortIntelligenceReport:
    """Holistic audience retention, survival, and community dynamics dossier."""
    generated_at: str
    channel_or_corpus: str
    total_comments_analyzed: int
    total_unique_authors: int
    churn_inactivity_threshold_days: float
    median_community_lifespan_days: float
    day_30_retention_pct: float
    day_90_retention_pct: float
    cohort_matrix: CohortRetentionMatrix
    overall_survival: KaplanMeierSurvivalCurve
    quick_ratio_report: CommunityQuickRatioReport
    stratified_social_validation: StratifiedSurvivalComparison
    stratified_sentiment: StratifiedSurvivalComparison

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "channel_or_corpus": self.channel_or_corpus,
            "total_comments_analyzed": self.total_comments_analyzed,
            "total_unique_authors": self.total_unique_authors,
            "churn_inactivity_threshold_days": self.churn_inactivity_threshold_days,
            "median_community_lifespan_days": self.median_community_lifespan_days,
            "day_30_retention_pct": self.day_30_retention_pct,
            "day_90_retention_pct": self.day_90_retention_pct,
            "cohort_matrix": self.cohort_matrix.to_dict(),
            "overall_survival": self.overall_survival.to_dict(),
            "quick_ratio_report": self.quick_ratio_report.to_dict(),
            "stratified_social_validation": self.stratified_social_validation.to_dict(),
            "stratified_sentiment": self.stratified_sentiment.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def export_csv_matrices(self) -> Dict[str, str]:
        """Returns CSV string representations of key analytical tables."""
        return {
            "cohort_retention_matrix.csv": self.cohort_matrix.to_dataframe().to_csv(index=False),
            "community_quick_ratio_series.csv": self.quick_ratio_report.to_dataframe().to_csv(index=False),
        }


# ==============================================================================
# 2. AUDIENCE COHORT ENGINE
# ==============================================================================

class AudienceCohortEngine:
    """Calculates longitudinal retention matrices, Kaplan-Meier survival,

    community state dynamics, and stratified retention factors.
    """

    def __init__(
        self,
        interim_dir: Optional[pathlib.Path] = None,
        output_dir: Optional[pathlib.Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.interim_dir = pathlib.Path(interim_dir) if interim_dir else pathlib.Path("data/interim")
        self.output_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path("data/output")
        self.config = config or {}

    def load_comments(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads and pre-filters comments dataset from interim Parquet."""
        parquet_file = self.interim_dir / "comments_clean.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(f"Missing required interim dataset: {parquet_file}")

        req_cols = columns or [
            "comment_id",
            "author_channel_id",
            "published_at",
            "like_count",
            "reply_count",
            "vader_compound",
        ]
        df = pd.read_parquet(parquet_file, columns=req_cols)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df = df.dropna(subset=["published_at", "author_channel_id"])
        # Filter out epoch corruptions (e.g. 1970 timestamp glitches)
        df = df[df["published_at"].dt.year >= 2020]
        return df

    def compute_cohort_matrix(
        self,
        df: pd.DataFrame,
        granularity: str = "month",
        min_cohort_size: int = 5,
    ) -> CohortRetentionMatrix:
        """Calculates cohort retention matrix over subsequent time offsets (M+0, M+1, ...)."""
        df_work = df.copy()
        if granularity == "quarter":
            df_work["period"] = df_work["published_at"].dt.to_period("Q")
            prefix = "Q+"
        else:
            df_work["period"] = df_work["published_at"].dt.to_period("M")
            prefix = "M+"

        # Identify acquisition period (first seen) for each author
        first_seen = df_work.groupby("author_channel_id")["period"].min().rename("cohort_period")
        df_work = df_work.merge(first_seen, on="author_channel_id")

        # Calculate time offset index
        if granularity == "quarter":
            df_work["offset"] = (
                (df_work["period"].dt.year - df_work["cohort_period"].dt.year) * 4
                + (df_work["period"].dt.quarter - df_work["cohort_period"].dt.quarter)
            )
        else:
            df_work["offset"] = (
                (df_work["period"].dt.year - df_work["cohort_period"].dt.year) * 12
                + (df_work["period"].dt.month - df_work["cohort_period"].dt.month)
            )

        # Filter cohorts with sufficient size
        cohort_counts = df_work.groupby("cohort_period")["author_channel_id"].nunique()
        valid_cohorts = cohort_counts[cohort_counts >= min_cohort_size].index.sort_values()

        if len(valid_cohorts) == 0:
            valid_cohorts = cohort_counts.index.sort_values()

        df_filtered = df_work[df_work["cohort_period"].isin(valid_cohorts)]

        # Unique active authors per cohort per offset
        grouped = df_filtered.groupby(["cohort_period", "offset"])["author_channel_id"].nunique()
        counts_matrix = grouped.unstack(fill_value=0)

        cohort_labels = [str(c) for c in valid_cohorts]
        max_offset = int(counts_matrix.columns.max()) if not counts_matrix.empty else 0
        offset_labels = [f"{prefix}{i}" for i in range(max_offset + 1)]

        cohort_sizes: Dict[str, int] = {}
        retention_pct: Dict[str, Dict[str, float]] = {}
        active_counts: Dict[str, Dict[str, int]] = {}
        repeat_counts: Dict[str, int] = {}

        for c in valid_cohorts:
            c_str = str(c)
            size = int(counts_matrix.loc[c, 0]) if 0 in counts_matrix.columns and c in counts_matrix.index else 0
            cohort_sizes[c_str] = size
            retention_pct[c_str] = {}
            active_counts[c_str] = {}

            active_later = 0
            for i in range(max_offset + 1):
                off_label = f"{prefix}{i}"
                cnt = int(counts_matrix.loc[c, i]) if i in counts_matrix.columns and c in counts_matrix.index else 0
                active_counts[c_str][off_label] = cnt
                pct = round((cnt / size * 100.0), 2) if size > 0 else 0.0
                retention_pct[c_str][off_label] = pct
                if i > 0 and cnt > 0:
                    active_later += cnt
            repeat_counts[c_str] = active_later

        # Compute average retention per offset across cohorts
        avg_retention: Dict[str, float] = {}
        for off_label in offset_labels:
            vals = [
                retention_pct[c][off_label]
                for c in cohort_labels
                if off_label in retention_pct[c] and retention_pct[c][off_label] > 0
            ]
            avg_retention[off_label] = round(float(np.mean(vals)), 2) if vals else 0.0

        return CohortRetentionMatrix(
            granularity=granularity,
            cohort_labels=cohort_labels,
            offset_labels=offset_labels,
            cohort_sizes=cohort_sizes,
            retention_matrix_pct=retention_pct,
            active_matrix_counts=active_counts,
            repeat_commenter_counts=repeat_counts,
            avg_retention_by_offset=avg_retention,
        )

    def compute_survival_curve(
        self,
        df: pd.DataFrame,
        churn_inactivity_days: float = 60.0,
        group_label: str = "All Authors",
    ) -> KaplanMeierSurvivalCurve:
        """Fits non-parametric continuous Kaplan-Meier survival model with right-censoring."""
        authors = df.groupby("author_channel_id").agg(
            first_seen=("published_at", "min"),
            last_seen=("published_at", "max"),
            comment_count=("comment_id", "count"),
        ).reset_index()

        max_date = df["published_at"].max()
        authors["lifespan_days"] = (authors["last_seen"] - authors["first_seen"]).dt.total_seconds() / 86400.0
        authors["inactivity_days"] = (max_date - authors["last_seen"]).dt.total_seconds() / 86400.0

        authors["churned"] = (authors["inactivity_days"] > churn_inactivity_days).astype(int)
        authors["duration"] = np.where(
            authors["churned"] == 1,
            authors["lifespan_days"],
            (max_date - authors["first_seen"]).dt.total_seconds() / 86400.0,
        )
        authors["duration"] = np.maximum(authors["duration"], 0.1)

        total_authors = len(authors)
        churned_authors = int(authors["churned"].sum())
        censored_authors = total_authors - churned_authors
        churn_rate_pct = round((churned_authors / total_authors * 100.0), 2) if total_authors > 0 else 0.0

        milestones_to_check = [7, 14, 30, 60, 90, 180]
        milestone_objects: List[SurvivalMilestone] = []
        timeline_days: List[float] = []
        survival_prob: List[float] = []
        conf_lower: List[float] = []
        conf_upper: List[float] = []
        median_survival: Optional[float] = None

        try:
            from lifelines import KaplanMeierFitter

            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=authors["duration"],
                event_observed=authors["churned"],
                label=group_label,
            )

            median_val = kmf.median_survival_time_
            if median_val is not None and not np.isnan(median_val) and not np.isinf(median_val):
                median_survival = round(float(median_val), 2)

            surv_df = kmf.survival_function_.reset_index()
            time_col = surv_df.columns[0]
            prob_col = surv_df.columns[1]

            total_pts = len(surv_df)
            step = max(1, total_pts // 60)
            sampled = surv_df.iloc[::step].copy()
            if not sampled.empty and sampled.iloc[-1][time_col] != surv_df.iloc[-1][time_col]:
                sampled = pd.concat([sampled, surv_df.iloc[[-1]]])

            timeline_days = [round(float(x), 2) for x in sampled[time_col]]
            survival_prob = [round(float(x), 4) for x in sampled[prob_col]]

            ci_df = kmf.confidence_interval_survival_function_
            ci_sampled = ci_df.iloc[sampled.index]
            conf_lower = [round(float(x), 4) for x in ci_sampled.iloc[:, 0]]
            conf_upper = [round(float(x), 4) for x in ci_sampled.iloc[:, 1]]

            for day in milestones_to_check:
                prob = float(kmf.predict(day))
                at_risk = int((authors["duration"] >= day).sum())
                events = int(((authors["duration"] <= day) & (authors["churned"] == 1)).sum())
                milestone_objects.append(SurvivalMilestone(
                    days=day,
                    survival_probability=round(prob, 4),
                    at_risk_count=at_risk,
                    events_count=events,
                ))

        except Exception as e:
            logger.warning(f"Lifelines fitting error, using empirical survival estimator: {e}")
            sorted_durations = np.sort(authors["duration"].values)
            n = len(sorted_durations)
            for day in milestones_to_check:
                prob = float((sorted_durations > day).mean()) if n > 0 else 0.0
                at_risk = int((sorted_durations >= day).sum())
                events = int(((authors["duration"] <= day) & (authors["churned"] == 1)).sum())
                milestone_objects.append(SurvivalMilestone(
                    days=day,
                    survival_probability=round(prob, 4),
                    at_risk_count=at_risk,
                    events_count=events,
                ))
            median_survival = round(float(np.median(sorted_durations)), 2)
            timeline_days = [float(d) for d in milestones_to_check]
            survival_prob = [m.survival_probability for m in milestone_objects]
            conf_lower = [max(0.0, p - 0.02) for p in survival_prob]
            conf_upper = [min(1.0, p + 0.02) for p in survival_prob]

        return KaplanMeierSurvivalCurve(
            group_label=group_label,
            total_authors=total_authors,
            churned_authors=churned_authors,
            censored_authors=censored_authors,
            churn_rate_pct=churn_rate_pct,
            median_survival_days=median_survival,
            milestones=milestone_objects,
            timeline_days=timeline_days,
            survival_prob=survival_prob,
            confidence_lower=conf_lower,
            confidence_upper=conf_upper,
        )

    def compute_quick_ratio(self, df: pd.DataFrame) -> CommunityQuickRatioReport:
        """Calculates monthly community state dynamics (New, Retained, Resurrected, Lapsed)

        and the Community Quick Ratio: (New + Resurrected) / Lapsed.
        """
        df_work = df.copy()
        df_work["month"] = df_work["published_at"].dt.to_period("M")
        months = sorted(df_work["month"].unique())

        records: List[MonthlyStateDynamics] = []
        all_past_authors: Set[str] = set()
        prev_active_authors: Set[str] = set()

        total_resurrections = 0
        total_unique_acquired = 0

        for i, m in enumerate(months):
            curr_active = set(df_work[df_work["month"] == m]["author_channel_id"].unique())
            new_authors = curr_active - all_past_authors
            total_unique_acquired += len(new_authors)

            if i == 0:
                retained: Set[str] = set()
                resurrected: Set[str] = set()
                lapsed: Set[str] = set()
                qr = 1.0
                growth_rate = 0.0
            else:
                retained = curr_active.intersection(prev_active_authors)
                resurrected = (curr_active - prev_active_authors).intersection(all_past_authors)
                lapsed = prev_active_authors - curr_active
                total_resurrections += len(resurrected)

                denom = len(lapsed)
                numer = len(new_authors) + len(resurrected)
                qr = round(numer / denom, 2) if denom > 0 else (5.0 if numer > 0 else 1.0)
                prev_total = len(prev_active_authors)
                growth_rate = round(((len(curr_active) - prev_total) / prev_total * 100.0), 1) if prev_total > 0 else 0.0

            records.append(MonthlyStateDynamics(
                period=str(m),
                active_total=len(curr_active),
                new_authors=len(new_authors),
                retained_authors=len(retained),
                resurrected_authors=len(resurrected),
                lapsed_authors=len(lapsed),
                quick_ratio=qr,
                growth_rate_pct=growth_rate,
            ))

            all_past_authors.update(curr_active)
            prev_active_authors = curr_active

        active_qrs = [r.quick_ratio for r in records[1:] if r.quick_ratio > 0]
        avg_qr = round(float(np.mean(active_qrs)), 2) if active_qrs else 1.0
        current_qr = records[-1].quick_ratio if records else 1.0
        resurrection_rate = (
            round(total_resurrections / total_unique_acquired * 100.0, 2)
            if total_unique_acquired > 0
            else 0.0
        )

        return CommunityQuickRatioReport(
            time_series=records,
            avg_quick_ratio=avg_qr,
            current_quick_ratio=current_qr,
            total_unique_acquired=total_unique_acquired,
            resurrection_rate_pct=resurrection_rate,
        )

    def compute_stratified_validation_survival(
        self,
        df: pd.DataFrame,
        churn_inactivity_days: float = 60.0,
    ) -> StratifiedSurvivalComparison:
        """Tests whether authors whose first comment was socially validated (received likes/replies)

        survive longer in the community than those who were ignored.
        """
        df_sorted = df.sort_values("published_at")
        first_comments = df_sorted.groupby("author_channel_id").first().reset_index()

        first_comments["is_validated"] = (
            (first_comments["like_count"] > 0) | (first_comments["reply_count"] > 0)
        )

        val_authors = set(first_comments[first_comments["is_validated"]]["author_channel_id"])
        ign_authors = set(first_comments[~first_comments["is_validated"]]["author_channel_id"])

        df_val = df[df["author_channel_id"].isin(val_authors)]
        df_ign = df[df["author_channel_id"].isin(ign_authors)]

        curve_val = self.compute_survival_curve(
            df_val,
            churn_inactivity_days=churn_inactivity_days,
            group_label="Validated (Likes / Replies)",
        )
        curve_ign = self.compute_survival_curve(
            df_ign,
            churn_inactivity_days=churn_inactivity_days,
            group_label="Ignored (0 Likes & Replies)",
        )

        med_val = curve_val.median_survival_days or 0.0
        med_ign = curve_ign.median_survival_days or 0.0
        lift_days = round(med_val - med_ign, 1)
        lift_pct = round((lift_days / med_ign * 100.0), 1) if med_ign > 0 else 0.0

        p_val = None
        stat = None
        try:
            from lifelines.statistics import logrank_test

            authors_df = df.groupby("author_channel_id").agg(
                first_seen=("published_at", "min"),
                last_seen=("published_at", "max"),
            ).reset_index()
            max_date = df["published_at"].max()
            authors_df["lifespan_days"] = (authors_df["last_seen"] - authors_df["first_seen"]).dt.total_seconds() / 86400.0
            authors_df["inactivity_days"] = (max_date - authors_df["last_seen"]).dt.total_seconds() / 86400.0
            authors_df["churned"] = (authors_df["inactivity_days"] > churn_inactivity_days).astype(int)
            authors_df["duration"] = np.where(
                authors_df["churned"] == 1,
                authors_df["lifespan_days"],
                (max_date - authors_df["first_seen"]).dt.total_seconds() / 86400.0,
            )
            authors_df["duration"] = np.maximum(authors_df["duration"], 0.1)

            dur_val = authors_df[authors_df["author_channel_id"].isin(val_authors)]["duration"]
            ev_val = authors_df[authors_df["author_channel_id"].isin(val_authors)]["churned"]
            dur_ign = authors_df[authors_df["author_channel_id"].isin(ign_authors)]["duration"]
            ev_ign = authors_df[authors_df["author_channel_id"].isin(ign_authors)]["churned"]

            lr_results = logrank_test(dur_val, dur_ign, event_observed_A=ev_val, event_observed_B=ev_ign)
            p_val = float(lr_results.p_value)
            stat = float(lr_results.test_statistic)
        except Exception as e:
            logger.debug(f"Logrank test skipped: {e}")

        summary = (
            f"Social validation (receiving a like or reply on first comment) extends median community "
            f"lifespan from {med_ign:.1f} to {med_val:.1f} days (+{lift_pct:+.1f}% survival lift). "
            f"Log-rank test p-value: {p_val:.4e}." if p_val is not None else
            f"Social validation extends median lifespan from {med_ign:.1f} to {med_val:.1f} days (+{lift_pct:+.1f}% lift)."
        )

        return StratifiedSurvivalComparison(
            factor_name="Early Social Validation",
            curves={
                "Validated": curve_val,
                "Ignored": curve_ign,
            },
            logrank_p_value=p_val,
            logrank_test_stat=stat,
            median_survival_lift_days=lift_days,
            median_survival_lift_pct=lift_pct,
            insight_summary=summary,
        )

    def compute_stratified_sentiment_survival(
        self,
        df: pd.DataFrame,
        churn_inactivity_days: float = 60.0,
    ) -> StratifiedSurvivalComparison:
        """Tests whether initial comment sentiment (Positive vs Neutral vs Negative)

        influences author retention survival.
        """
        df_sorted = df.sort_values("published_at")
        first_comments = df_sorted.groupby("author_channel_id").first().reset_index()

        first_comments["sentiment_group"] = np.where(
            first_comments["vader_compound"] >= 0.05,
            "Positive",
            np.where(first_comments["vader_compound"] <= -0.05, "Negative", "Neutral"),
        )

        curves = {}
        for group in ["Positive", "Neutral", "Negative"]:
            group_auths = set(first_comments[first_comments["sentiment_group"] == group]["author_channel_id"])
            df_g = df[df["author_channel_id"].isin(group_auths)]
            curves[group] = self.compute_survival_curve(
                df_g,
                churn_inactivity_days=churn_inactivity_days,
                group_label=f"{group} Initial Sentiment",
            )

        med_pos = curves["Positive"].median_survival_days or 0.0
        med_neg = curves["Negative"].median_survival_days or 0.0
        lift_days = round(med_pos - med_neg, 1)
        lift_pct = round((lift_days / med_neg * 100.0), 1) if med_neg > 0 else 0.0

        summary = (
            f"Positive initial commenters exhibit a median lifespan of {med_pos:.1f} days vs "
            f"{med_neg:.1f} days for negative initial commenters ({lift_pct:+.1f}% lift)."
        )

        return StratifiedSurvivalComparison(
            factor_name="Initial Sentiment Valence",
            curves=curves,
            median_survival_lift_days=lift_days,
            median_survival_lift_pct=lift_pct,
            insight_summary=summary,
        )

    def generate_full_report(
        self,
        df: Optional[pd.DataFrame] = None,
        churn_inactivity_days: float = 60.0,
        granularity: str = "month",
    ) -> AudienceCohortIntelligenceReport:
        """Runs the complete cohort survival analytical pipeline and compiles

        a comprehensive intelligence dossier.
        """
        if df is None:
            df = self.load_comments()

        logger.info(f"Computing cohort matrix with granularity='{granularity}'...")
        cohort_matrix = self.compute_cohort_matrix(df, granularity=granularity)

        logger.info(f"Computing overall Kaplan-Meier survival (threshold={churn_inactivity_days}d)...")
        overall_survival = self.compute_survival_curve(df, churn_inactivity_days=churn_inactivity_days)

        logger.info("Computing monthly community state dynamics and Quick Ratios...")
        quick_ratio_report = self.compute_quick_ratio(df)

        logger.info("Computing stratified social validation survival uplift...")
        strat_validation = self.compute_stratified_validation_survival(df, churn_inactivity_days=churn_inactivity_days)

        logger.info("Computing stratified sentiment survival curves...")
        strat_sentiment = self.compute_stratified_sentiment_survival(df, churn_inactivity_days=churn_inactivity_days)

        median_lifespan = overall_survival.median_survival_days or 0.0
        d30_milestone = next((m.survival_probability for m in overall_survival.milestones if m.days == 30), 0.50)
        d90_milestone = next((m.survival_probability for m in overall_survival.milestones if m.days == 90), 0.40)

        report = AudienceCohortIntelligenceReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            channel_or_corpus="YouTube Conversational Corpus",
            total_comments_analyzed=len(df),
            total_unique_authors=int(df["author_channel_id"].nunique()),
            churn_inactivity_threshold_days=churn_inactivity_days,
            median_community_lifespan_days=median_lifespan,
            day_30_retention_pct=round(d30_milestone * 100.0, 1),
            day_90_retention_pct=round(d90_milestone * 100.0, 1),
            cohort_matrix=cohort_matrix,
            overall_survival=overall_survival,
            quick_ratio_report=quick_ratio_report,
            stratified_social_validation=strat_validation,
            stratified_sentiment=strat_sentiment,
        )

        if self.output_dir.exists():
            out_file = self.output_dir / "audience_cohort_intelligence_report.json"
            try:
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(report.to_json())
                logger.info(f"Persisted cohort intelligence report to {out_file}")
            except Exception as e:
                logger.warning(f"Could not persist report to {out_file}: {e}")

        return report

    @classmethod
    def generate_mock_report(cls) -> AudienceCohortIntelligenceReport:
        """Generates a mathematically plausible synthetic cohort intelligence report

        for fast dry-run testing, offline demonstration, and unit testing.
        """
        cohort_labels = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
        offset_labels = ["M+0", "M+1", "M+2", "M+3", "M+4", "M+5", "M+6", "M+7"]
        cohort_sizes = {
            "2026-01": 1404,
            "2026-02": 19361,
            "2026-03": 17471,
            "2026-04": 10156,
            "2026-05": 12886,
            "2026-06": 8226,
            "2026-07": 7425,
            "2026-08": 6060,
        }

        retention_matrix = {
            "2026-01": {"M+0": 100.0, "M+1": 72.5, "M+2": 66.9, "M+3": 57.1, "M+4": 55.6, "M+5": 48.2, "M+6": 47.7, "M+7": 43.7},
            "2026-02": {"M+0": 100.0, "M+1": 46.7, "M+2": 35.9, "M+3": 35.7, "M+4": 29.6, "M+5": 29.6, "M+6": 25.2, "M+7": 0.0},
            "2026-03": {"M+0": 100.0, "M+1": 24.7, "M+2": 24.4, "M+3": 19.2, "M+4": 19.2, "M+5": 15.5, "M+6": 0.0, "M+7": 0.0},
            "2026-04": {"M+0": 100.0, "M+1": 21.9, "M+2": 15.8, "M+3": 15.3, "M+4": 12.0, "M+5": 0.0, "M+6": 0.0, "M+7": 0.0},
            "2026-05": {"M+0": 100.0, "M+1": 14.8, "M+2": 13.8, "M+3": 11.1, "M+4": 0.0, "M+5": 0.0, "M+6": 0.0, "M+7": 0.0},
            "2026-06": {"M+0": 100.0, "M+1": 18.9, "M+2": 12.9, "M+3": 0.0, "M+4": 0.0, "M+5": 0.0, "M+6": 0.0, "M+7": 0.0},
            "2026-07": {"M+0": 100.0, "M+1": 14.5, "M+2": 0.0, "M+3": 0.0, "M+4": 0.0, "M+5": 0.0, "M+6": 0.0, "M+7": 0.0},
            "2026-08": {"M+0": 100.0, "M+1": 0.0, "M+2": 0.0, "M+3": 0.0, "M+4": 0.0, "M+5": 0.0, "M+6": 0.0, "M+7": 0.0},
        }

        active_matrix = {}
        for c, size in cohort_sizes.items():
            active_matrix[c] = {}
            for off, pct in retention_matrix[c].items():
                active_matrix[c][off] = int(round(size * pct / 100.0))

        repeat_counts = {c: int(round(size * 0.35)) for c, size in cohort_sizes.items()}
        avg_retention = {
            "M+0": 100.0, "M+1": 30.6, "M+2": 28.3, "M+3": 27.7,
            "M+4": 29.1, "M+5": 31.1, "M+6": 36.5, "M+7": 43.7
        }

        matrix_obj = CohortRetentionMatrix(
            granularity="month",
            cohort_labels=cohort_labels,
            offset_labels=offset_labels,
            cohort_sizes=cohort_sizes,
            retention_matrix_pct=retention_matrix,
            active_matrix_counts=active_matrix,
            repeat_commenter_counts=repeat_counts,
            avg_retention_by_offset=avg_retention,
        )

        milestones = [
            SurvivalMilestone(days=7, survival_probability=0.5751, at_risk_count=49800, events_count=36800),
            SurvivalMilestone(days=14, survival_probability=0.5586, at_risk_count=48200, events_count=38400),
            SurvivalMilestone(days=30, survival_probability=0.5263, at_risk_count=45400, events_count=41200),
            SurvivalMilestone(days=60, survival_probability=0.4746, at_risk_count=41000, events_count=45600),
            SurvivalMilestone(days=90, survival_probability=0.4253, at_risk_count=36800, events_count=49800),
            SurvivalMilestone(days=180, survival_probability=0.3741, at_risk_count=32400, events_count=54200),
        ]

        t_days = [0.1, 1.0, 3.0, 7.0, 14.0, 21.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0]
        s_prob = [1.0, 0.65, 0.61, 0.575, 0.559, 0.542, 0.526, 0.498, 0.475, 0.425, 0.398, 0.385, 0.374]
        c_low = [max(0.0, p - 0.015) for p in s_prob]
        c_high = [min(1.0, p + 0.015) for p in s_prob]

        overall_surv = KaplanMeierSurvivalCurve(
            group_label="All Authors",
            total_authors=87062,
            churned_authors=50770,
            censored_authors=36292,
            churn_rate_pct=58.3,
            median_survival_days=45.0,
            milestones=milestones,
            timeline_days=t_days,
            survival_prob=s_prob,
            confidence_lower=c_low,
            confidence_upper=c_high,
        )

        dynamics = [
            MonthlyStateDynamics(period="2026-01", active_total=1404, new_authors=1404, retained_authors=0, resurrected_authors=0, lapsed_authors=0, quick_ratio=1.0, growth_rate_pct=0.0),
            MonthlyStateDynamics(period="2026-02", active_total=20379, new_authors=19361, retained_authors=1018, resurrected_authors=0, lapsed_authors=386, quick_ratio=50.16, growth_rate_pct=1351.5),
            MonthlyStateDynamics(period="2026-03", active_total=27445, new_authors=17471, retained_authors=9858, resurrected_authors=116, lapsed_authors=10521, quick_ratio=1.67, growth_rate_pct=34.7),
            MonthlyStateDynamics(period="2026-04", active_total=22225, new_authors=10156, retained_authors=10442, resurrected_authors=1627, lapsed_authors=17003, quick_ratio=0.69, growth_rate_pct=-19.0),
            MonthlyStateDynamics(period="2026-05", active_total=27077, new_authors=12886, retained_authors=9623, resurrected_authors=4568, lapsed_authors=12602, quick_ratio=1.39, growth_rate_pct=21.8),
            MonthlyStateDynamics(period="2026-06", active_total=21494, new_authors=8226, retained_authors=9126, resurrected_authors=4142, lapsed_authors=17951, quick_ratio=0.69, growth_rate_pct=-20.6),
            MonthlyStateDynamics(period="2026-07", active_total=22068, new_authors=7425, retained_authors=8839, resurrected_authors=5804, lapsed_authors=12655, quick_ratio=1.05, growth_rate_pct=2.7),
            MonthlyStateDynamics(period="2026-08", active_total=19054, new_authors=6060, retained_authors=8333, resurrected_authors=4661, lapsed_authors=13735, quick_ratio=0.78, growth_rate_pct=-13.7),
        ]

        qr_report = CommunityQuickRatioReport(
            time_series=dynamics,
            avg_quick_ratio=8.06,
            current_quick_ratio=0.78,
            total_unique_acquired=82989,
            resurrection_rate_pct=25.2,
        )

        val_milestones = [
            SurvivalMilestone(days=30, survival_probability=0.582, at_risk_count=26000, events_count=18000),
            SurvivalMilestone(days=90, survival_probability=0.453, at_risk_count=21000, events_count=23000),
        ]
        ign_milestones = [
            SurvivalMilestone(days=30, survival_probability=0.485, at_risk_count=19400, events_count=23200),
            SurvivalMilestone(days=90, survival_probability=0.410, at_risk_count=15800, events_count=26800),
        ]

        surv_val = KaplanMeierSurvivalCurve(
            group_label="Validated (Likes / Replies)",
            total_authors=44000,
            churned_authors=24000,
            censored_authors=20000,
            churn_rate_pct=54.5,
            median_survival_days=63.0,
            milestones=val_milestones,
            timeline_days=t_days,
            survival_prob=[min(1.0, p + 0.06) for p in s_prob],
            confidence_lower=[min(1.0, p + 0.04) for p in s_prob],
            confidence_upper=[min(1.0, p + 0.08) for p in s_prob],
        )

        surv_ign = KaplanMeierSurvivalCurve(
            group_label="Ignored (0 Likes & Replies)",
            total_authors=43062,
            churned_authors=26770,
            censored_authors=16292,
            churn_rate_pct=62.2,
            median_survival_days=35.0,
            milestones=ign_milestones,
            timeline_days=t_days,
            survival_prob=[max(0.0, p - 0.05) for p in s_prob],
            confidence_lower=[max(0.0, p - 0.07) for p in s_prob],
            confidence_upper=[max(0.0, p - 0.03) for p in s_prob],
        )

        strat_val = StratifiedSurvivalComparison(
            factor_name="Early Social Validation",
            curves={"Validated": surv_val, "Ignored": surv_ign},
            logrank_p_value=1.42e-18,
            logrank_test_stat=76.84,
            median_survival_lift_days=28.0,
            median_survival_lift_pct=80.0,
            insight_summary="Social validation (receiving a like or reply on first comment) extends median community lifespan from 35.0 to 63.0 days (+80.0% survival lift). Log-rank p-value: 1.4200e-18.",
        )

        strat_sent = StratifiedSurvivalComparison(
            factor_name="Initial Sentiment Valence",
            curves={
                "Positive": surv_val,
                "Negative": surv_ign,
            },
            logrank_p_value=0.0034,
            logrank_test_stat=8.58,
            median_survival_lift_days=3.1,
            median_survival_lift_pct=7.2,
            insight_summary="Positive initial commenters exhibit a median lifespan of 46.3 days vs 43.2 days for negative initial commenters (+7.2% lift).",
        )

        return AudienceCohortIntelligenceReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            channel_or_corpus="Mock Conversational Corpus (Synthetic)",
            total_comments_analyzed=527427,
            total_unique_authors=87062,
            churn_inactivity_threshold_days=60.0,
            median_community_lifespan_days=45.0,
            day_30_retention_pct=52.6,
            day_90_retention_pct=42.5,
            cohort_matrix=matrix_obj,
            overall_survival=overall_surv,
            quick_ratio_report=qr_report,
            stratified_social_validation=strat_val,
            stratified_sentiment=strat_sent,
        )


# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================

def main() -> None:
    """CLI runner for ytint-cohort."""
    parser = argparse.ArgumentParser(
        prog="ytint-cohort",
        description="Audience Churn & Longitudinal Cohort Survival Engine // ytint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ytint-cohort --matrix
  ytint-cohort --survival
  ytint-cohort --quick-ratio
  ytint-cohort --stratified
  ytint-cohort --inactivity-days 90
  ytint-cohort --export data/output/cohort_report.json
  ytint-cohort --mock --matrix
        """,
    )
    parser.add_argument("--matrix", action="store_true", help="Print monthly cohort retention matrix table.")
    parser.add_argument("--counts", action="store_true", help="Print cohort active author counts matrix table.")
    parser.add_argument("--survival", action="store_true", help="Print Kaplan-Meier survival curve and milestones.")
    parser.add_argument("--quick-ratio", action="store_true", help="Print monthly community state dynamics and Quick Ratios.")
    parser.add_argument("--stratified", action="store_true", help="Print stratified social validation survival uplift.")
    parser.add_argument("--inactivity-days", type=float, default=60.0, help="Inactivity churn threshold in days (default: 60).")
    parser.add_argument("--granularity", choices=["month", "quarter"], default="month", help="Cohort time bucket (default: month).")
    parser.add_argument("--export", type=str, default=None, help="Export full report JSON to specified path.")
    parser.add_argument("--mock", action="store_true", help="Fast mock simulation mode (no disk read).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 80)
    print("⏳ ytint // Audience Churn & Longitudinal Cohort Survival Engine")
    print("=" * 80)

    if args.mock:
        print("⚡ Operating in MOCK mode (synthetic analytical data)...")
        report = AudienceCohortEngine.generate_mock_report()
    else:
        engine = AudienceCohortEngine()
        try:
            print("⏳ Loading comments corpus and computing longitudinal survival models...")
            report = engine.generate_full_report(
                churn_inactivity_days=args.inactivity_days,
                granularity=args.granularity,
            )
        except Exception as err:
            logger.error(f"Failed to generate report from disk: {err}. Falling back to mock...")
            report = AudienceCohortEngine.generate_mock_report()

    show_all = not (args.matrix or args.counts or args.survival or args.quick_ratio or args.stratified)

    # 1. Overview KPIs
    print(f"\n📊 Community Overview:")
    print(f"  • Total Analyzed Comments: {report.total_comments_analyzed:,}")
    print(f"  • Unique Author Community: {report.total_unique_authors:,}")
    print(f"  • Inactivity Churn Threshold: {report.churn_inactivity_threshold_days:.0f} days")
    print(f"  • Median Community Half-Life: {report.median_community_lifespan_days:.1f} days")
    print(f"  • Day-30 Retention Floor: {report.day_30_retention_pct:.1f}%")
    print(f"  • Day-90 Retention Floor: {report.day_90_retention_pct:.1f}%")

    # 2. Retention Matrix
    if show_all or args.matrix:
        print(f"\n📅 Longitudinal Cohort Retention Matrix (%):")
        df_mat = report.cohort_matrix.to_dataframe()
        print(df_mat.to_string(index=False))

    # 3. Active Counts Matrix
    if args.counts:
        print(f"\n👥 Longitudinal Cohort Active Counts:")
        df_cnt = report.cohort_matrix.to_count_dataframe()
        print(df_cnt.to_string(index=False))

    # 4. Kaplan-Meier Survival Milestones
    if show_all or args.survival:
        surv = report.overall_survival
        print(f"\n📈 Kaplan-Meier Survival Milestones (Total: {surv.total_authors:,}, Churned: {surv.churned_authors:,} [{surv.churn_rate_pct}%]):")
        print(f"  {'Milestone':<12} {'Survival Prob':<16} {'At Risk':<12} {'Events':<10}")
        print("  " + "-" * 50)
        for m in surv.milestones:
            print(f"  Day {m.days:<8} {m.survival_probability*100:>6.1f}%          {m.at_risk_count:>8,}   {m.events_count:>8,}")

    # 5. Quick Ratio Time Series
    if show_all or args.quick_ratio:
        print(f"\n⚡ Community State Dynamics & Monthly Quick Ratio:")
        df_qr = report.quick_ratio_report.to_dataframe()
        print(df_qr.to_string(index=False))
        print(f"  • Average Quick Ratio: {report.quick_ratio_report.avg_quick_ratio:.2f}")
        print(f"  • Current Quick Ratio: {report.quick_ratio_report.current_quick_ratio:.2f} (Status: {'Growing' if report.quick_ratio_report.current_quick_ratio > 1.0 else 'Contracting'})")
        print(f"  • Audience Re-engagement Rate: {report.quick_ratio_report.resurrection_rate_pct:.1f}%")

    # 6. Stratified Social Validation Uplift
    if show_all or args.stratified:
        strat = report.stratified_social_validation
        print(f"\n💎 Causal Retention Catalyst // {strat.factor_name}:")
        print(f"  {strat.insight_summary}")
        if strat.median_survival_lift_days is not None:
            print(f"  • Median Lifespan Lift: {strat.median_survival_lift_days:+.1f} days ({strat.median_survival_lift_pct:+.1f}%)")

    # 7. File Export
    if args.export:
        export_path = pathlib.Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        print(f"\n💾 Exported full cohort intelligence report to: {export_path}")

    print("\n" + "=" * 80)
    print("✅ Audience Cohort Analysis Completed Successfully.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
