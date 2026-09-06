"""Unit tests for Creator Actionability & Engagement Optimization Assistant (ytint-assist)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engine.assistant import (
    ActionRecommendation,
    CreatorAssistantEngine,
    TriageReport,
    main,
)


def test_dataclasses_serialization() -> None:
    rec = ActionRecommendation(
        comment_id="c_1",
        video_id="v_1",
        author_name="Alice",
        author_cohort="Loyal",
        text="Can you explain the methodology at 5:00?",
        like_count=10,
        reply_count=1,
        published_at="2026-09-06T12:00:00Z",
        minutes_since_upload=30.0,
        sentiment_score=0.4,
        toxicity_score=0.01,
        action_type="REPLY_QUESTION",
        priority_score=85.0,
        action_rationale="Question from Loyal viewer.",
        expected_uplift={"Replies": "+320%"},
        draft_reply="Great question, Alice!",
    )
    d = rec.to_dict()
    assert d["action_type"] == "REPLY_QUESTION"
    assert d["priority_score"] == 85.0

    report = TriageReport(
        video_id="v_1",
        video_title="Video 1",
        total_analyzed=100,
        total_actionable=1,
        summary_counts={"REPLY_QUESTION": 1},
        recommendations=[rec],
    )
    df = report.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["author_name"] == "Alice"
    assert df.iloc[0]["action_type"] == "REPLY_QUESTION"


def test_mock_triage_generation() -> None:
    engine = CreatorAssistantEngine()
    report = engine.generate_mock_triage(video_id="MOCK_TEST", limit=10)
    assert report.video_id == "MOCK_TEST"
    assert len(report.recommendations) >= 4

    action_types = {r.action_type for r in report.recommendations}
    assert "PIN" in action_types
    assert "REPLY_QUESTION" in action_types
    assert "DEESCALATE" in action_types
    assert "HEART" in action_types

    # Verify priority score ordering
    priorities = [r.priority_score for r in report.recommendations]
    assert priorities == sorted(priorities, reverse=True)


def test_action_filter_and_cohort_filter() -> None:
    engine = CreatorAssistantEngine()
    report_pin = engine.generate_mock_triage(video_id="MOCK_FILTER", action_filter="PIN")
    for r in report_pin.recommendations:
        assert r.action_type == "PIN"

    report_reply = engine.generate_mock_triage(video_id="MOCK_FILTER", action_filter="REPLY")
    for r in report_reply.recommendations:
        assert r.action_type == "REPLY_QUESTION"


def test_causal_uplift_and_scoring_logic(tmp_path: Path) -> None:
    comments = [
        # 1. Clear Pin Candidate: long, high sentiment, likes, low toxicity, root
        {
            "comment_id": "c_pin",
            "video_id": "VID_TRIAGE",
            "author_channel_id": "auth_1",
            "author_display_name": "InsightViewer",
            "text": "This video provides the most thorough and well-documented breakdown of the architecture that I have seen anywhere on the internet.",
            "like_count": 25,
            "reply_count": 3,
            "is_reply": 0,
            "published_at": "2026-09-06T10:00:00Z",
            "minutes_since_upload": 10.0,
            "vader_compound": 0.85,
            "toxicity": 0.01,
        },
        # 2. Clear Question Candidate: contains '?', unanswered
        {
            "comment_id": "c_q",
            "video_id": "VID_TRIAGE",
            "author_channel_id": "auth_2",
            "author_display_name": "CuriousUser",
            "text": "How do we configure the environment variables when running locally?",
            "like_count": 8,
            "reply_count": 0,
            "is_reply": 0,
            "published_at": "2026-09-06T10:15:00Z",
            "minutes_since_upload": 25.0,
            "vader_compound": 0.1,
            "toxicity": 0.01,
        },
        # 3. Clear De-escalation Candidate: heated, negative, toxic, active replies
        {
            "comment_id": "c_deescalate",
            "video_id": "VID_TRIAGE",
            "author_channel_id": "auth_3",
            "author_display_name": "Critic99",
            "text": "This whole premise is completely ridiculous and wrong, nobody in the industry does it this way.",
            "like_count": 12,
            "reply_count": 6,
            "is_reply": 0,
            "published_at": "2026-09-06T10:30:00Z",
            "minutes_since_upload": 40.0,
            "vader_compound": -0.65,
            "toxicity": 0.35,
        },
        # 4. Clear Heart Candidate: praise, loyal
        {
            "comment_id": "c_heart",
            "video_id": "VID_TRIAGE",
            "author_channel_id": "auth_4",
            "author_display_name": "LoyalFan",
            "text": "Another fantastic upload, absolutely loved every minute of this!",
            "like_count": 15,
            "reply_count": 0,
            "is_reply": 0,
            "published_at": "2026-09-06T10:45:00Z",
            "minutes_since_upload": 55.0,
            "vader_compound": 0.90,
            "toxicity": 0.01,
        },
    ]
    df_comments = pd.DataFrame(comments)
    comments_file = tmp_path / "comments_triage.parquet"
    df_comments.to_parquet(comments_file, index=False)

    # Authors parquet
    authors_data = [
        {"author_channel_id": "auth_1", "rfm_cohort": "Champions"},
        {"author_channel_id": "auth_2", "rfm_cohort": "Loyal"},
        {"author_channel_id": "auth_3", "rfm_cohort": "At Risk"},
        {"author_channel_id": "auth_4", "rfm_cohort": "Loyal"},
    ]
    authors_file = tmp_path / "authors_triage.parquet"
    pd.DataFrame(authors_data).to_parquet(authors_file, index=False)

    # Videos parquet
    videos_data = [{"video_id": "VID_TRIAGE", "title": "Triage Test Video", "total_comments": 4}]
    videos_file = tmp_path / "videos_triage.parquet"
    pd.DataFrame(videos_data).to_parquet(videos_file, index=False)

    engine = CreatorAssistantEngine(
        comments_path=str(comments_file),
        authors_path=str(authors_file),
        videos_path=str(videos_file),
    )

    report = engine.triage_comments("VID_TRIAGE", limit=10)
    assert report.total_actionable == 4

    rec_by_id = {r.comment_id: r for r in report.recommendations}
    assert rec_by_id["c_pin"].action_type == "PIN"
    assert rec_by_id["c_q"].action_type == "REPLY_QUESTION"
    assert rec_by_id["c_deescalate"].action_type == "DEESCALATE"
    assert rec_by_id["c_heart"].action_type == "HEART"

    # Verify expected uplift dictionary is populated
    assert len(rec_by_id["c_pin"].expected_uplift) >= 1
    assert len(rec_by_id["c_deescalate"].expected_uplift) >= 1


def test_draft_reply_tones() -> None:
    engine = CreatorAssistantEngine()
    rec = ActionRecommendation(
        comment_id="c_test",
        video_id="v_test",
        author_name="Sarah",
        author_cohort="Loyal",
        text="Why did the result change after step 3?",
        like_count=5,
        reply_count=0,
        published_at="2026-09-06T12:00:00Z",
        minutes_since_upload=10.0,
        sentiment_score=0.1,
        toxicity_score=0.01,
        action_type="REPLY_QUESTION",
        priority_score=80.0,
        action_rationale="Question from Loyal viewer.",
    )

    warm_reply = engine.draft_reply(rec, tone="warm")
    assert "Sarah" in warm_reply
    assert len(warm_reply) > 20

    clarifying_reply = engine.draft_reply(rec, tone="clarifying")
    assert "Sarah" in clarifying_reply
    assert len(clarifying_reply) > 20

    playful_reply = engine.draft_reply(rec, tone="playful")
    assert "Sarah" in playful_reply
    assert len(playful_reply) > 20


def test_export_json_and_csv(tmp_path: Path) -> None:
    engine = CreatorAssistantEngine()
    report = engine.generate_mock_triage("VID_EXP")

    # JSON export
    json_path = tmp_path / "triage.json"
    exported_json = report.export(json_path, export_format="json")
    assert Path(exported_json).exists()
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["video_id"] == "VID_EXP"
    assert len(doc["recommendations"]) >= 1

    # CSV export
    csv_path = tmp_path / "triage.csv"
    exported_csv = report.export(csv_path, export_format="csv")
    assert Path(exported_csv).exists()
    df_read = pd.read_csv(csv_path)
    assert len(df_read) >= 1
    assert "action_type" in df_read.columns
    assert "priority_score" in df_read.columns


def test_cli_main_mock_and_export(tmp_path: Path) -> None:
    export_file = tmp_path / "cli_triage_export.json"
    code = main([
        "--mock",
        "--video-id", "CLI_TEST",
        "--action", "ALL",
        "--limit", "4",
        "--draft-replies",
        "--tone", "warm",
        "--export", str(export_file),
    ])
    assert code == 0
    assert export_file.exists()
    with open(export_file, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["video_id"] == "CLI_TEST"
    assert len(doc["recommendations"]) == 4
    for r in doc["recommendations"]:
        assert r["draft_reply"] is not None
