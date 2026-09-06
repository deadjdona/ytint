"""Unit tests for Bi-Directional YouTube Moderation Action Dispatcher (tests/test_moderator.py)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from engine.moderator import (
    ActionStatus,
    ModerationAction,
    ModerationAuditRecord,
    ModerationQueueState,
    YouTubeModerationEngine,
)


@pytest.fixture
def temp_engine(tmp_path: Path) -> YouTubeModerationEngine:
    queue_file = tmp_path / "moderation_queue.json"
    audit_file = tmp_path / "moderation_audit_log.json"
    return YouTubeModerationEngine(
        queue_path=str(queue_file),
        audit_path=str(audit_file),
    )


def test_moderation_action_model() -> None:
    act = ModerationAction(
        action_id="act_test_001",
        comment_id="cmt_12345",
        video_id="vid_abc",
        author_id="auth_99",
        author_name="Test Author",
        action_type="heldForReview",
        ban_author=True,
        reason="Test violation",
        trigger_source="rule_toxicity",
    )
    d = act.to_dict()
    assert d["action_id"] == "act_test_001"
    assert d["comment_id"] == "cmt_12345"
    assert d["ban_author"] is True
    assert d["status"] == "pending"

    restored = ModerationAction.from_dict(d)
    assert restored.action_id == act.action_id
    assert restored.comment_id == act.comment_id
    assert restored.action_type == "heldForReview"


def test_moderation_audit_record_model() -> None:
    rec = ModerationAuditRecord(
        record_id="aud_test_001",
        timestamp="2026-09-06T12:00:00Z",
        action_id="act_test_001",
        comment_id="cmt_12345",
        author_name="Spam Bot",
        action_type="markAsSpam",
        ban_author=False,
        is_dry_run=True,
        status="SIMULATED",
        detail="Simulated spam marker",
    )
    d = rec.to_dict()
    assert d["record_id"] == "aud_test_001"
    assert d["is_dry_run"] is True

    restored = ModerationAuditRecord.from_dict(d)
    assert restored.status == "SIMULATED"
    assert restored.comment_id == "cmt_12345"


def test_queue_persistence(temp_engine: YouTubeModerationEngine) -> None:
    temp_engine.enqueue_action(
        comment_id="cmt_persist_1",
        video_id="vid_1",
        author_name="User1",
        action_type="heldForReview",
        reason="Reason 1",
    )
    temp_engine.enqueue_action(
        comment_id="cmt_persist_2",
        video_id="vid_2",
        author_name="User2",
        action_type="rejected",
        ban_author=True,
        reason="Reason 2",
    )

    # Re-instantiate from same files
    reloaded = YouTubeModerationEngine(
        queue_path=str(temp_engine.queue_path),
        audit_path=str(temp_engine.audit_path),
    )
    assert len(reloaded.state.actions) == 2
    assert reloaded.state.actions[0].comment_id == "cmt_persist_1"
    assert reloaded.state.actions[1].ban_author is True


def test_enqueue_idempotency(temp_engine: YouTubeModerationEngine) -> None:
    act1 = temp_engine.enqueue_action(
        comment_id="cmt_dup",
        action_type="heldForReview",
        author_name="UserDup",
    )
    act2 = temp_engine.enqueue_action(
        comment_id="cmt_dup",
        action_type="heldForReview",
        author_name="UserDup",
    )
    assert act1.action_id == act2.action_id
    assert len(temp_engine.state.actions) == 1


def test_approve_reject_delete_lifecycle(temp_engine: YouTubeModerationEngine) -> None:
    act = temp_engine.enqueue_action(
        comment_id="cmt_life",
        action_type="postReply",
        author_name="Inquirer",
        reply_text="Hello!",
    )
    assert act.status == "pending"

    # Approve
    temp_engine.approve_action(act.action_id)
    assert temp_engine.state.actions[0].status == "approved"

    # Reject
    temp_engine.reject_action(act.action_id)
    assert temp_engine.state.actions[0].status == "rejected"

    # Delete
    deleted = temp_engine.delete_action(act.action_id)
    assert deleted is True
    assert len(temp_engine.state.actions) == 0


def test_scan_and_apply_rules_mock(temp_engine: YouTubeModerationEngine) -> None:
    enqueued = temp_engine.scan_and_apply_rules(rule_categories=["all"], mock=True)
    assert len(enqueued) >= 4

    types = {a.action_type for a in enqueued}
    assert "heldForReview" in types
    assert "markAsSpam" in types
    assert "rejected" in types

    bans = [a.ban_author for a in enqueued]
    assert True in bans


def test_dispatch_dry_run_batch(temp_engine: YouTubeModerationEngine) -> None:
    temp_engine.enqueue_action(
        comment_id="cmt_disp_1",
        author_name="User1",
        action_type="heldForReview",
        ban_author=True,
    )
    temp_engine.enqueue_action(
        comment_id="cmt_disp_2",
        author_name="User2",
        action_type="markAsSpam",
        ban_author=False,
    )
    temp_engine.approve_all()

    succ, fail, results = temp_engine.dispatch_batch(dry_run=True)
    assert succ == 2
    assert fail == 0
    assert len(results) == 2

    # Check status updated to dispatched
    for a in temp_engine.state.actions:
        assert a.status == "dispatched"
        assert a.dispatched_at is not None

    # Check audit records persisted
    logs = temp_engine.load_audit_log()
    assert len(logs) == 2
    assert logs[0].status == "SIMULATED"
    assert logs[0].is_dry_run is True


def test_export_queue_and_audit(temp_engine: YouTubeModerationEngine, tmp_path: Path) -> None:
    temp_engine.generate_mock_queue()
    temp_engine.approve_all()
    temp_engine.dispatch_batch(dry_run=True)

    json_queue = tmp_path / "queue.json"
    csv_queue = tmp_path / "queue.csv"
    temp_engine.export_queue(json_queue, export_format="json")
    temp_engine.export_queue(csv_queue, export_format="csv")

    assert json_queue.exists()
    assert csv_queue.exists()
    assert json_queue.stat().st_size > 0
    assert csv_queue.stat().st_size > 0


def test_cli_execution() -> None:
    python_exe = sys.executable
    res = subprocess.run(
        [python_exe, "-m", "engine.moderator", "--status"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res.returncode == 0
    assert "YOUTUBE MODERATION ENGINE TELEMETRY" in res.stdout

    res_mock = subprocess.run(
        [python_exe, "-m", "engine.moderator", "--mock", "--queue"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res_mock.returncode == 0
    assert "YOUTUBE MODERATION ACTION QUEUE" in res_mock.stdout
