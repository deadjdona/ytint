"""Bi-Directional YouTube Moderation Action Dispatcher & Policy Hub (ytint-moderate).

Provides an auditable, rules-based moderation engine to execute creator actions:
- Set moderation status (heldForReview, published, rejected)
- Mark as spam
- Ban abusive / astroturfing authors
- Dispatch AI-drafted creator replies
- Full dry-run simulation mode with immutable audit log
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd

_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from engine.config_loader import get_paths, load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ytint_moderator")

ModerationActionType = Literal[
    "heldForReview",
    "rejected",
    "published",
    "markAsSpam",
    "postReply",
]

ActionStatus = Literal["pending", "approved", "rejected", "dispatched", "failed"]


@dataclass
class ModerationAction:
    """Represents a discrete YouTube moderation action to be executed or simulated."""

    action_id: str
    comment_id: str
    video_id: str
    author_id: str
    author_name: str
    action_type: ModerationActionType
    ban_author: bool = False
    reason: str = ""
    trigger_source: str = "manual"  # manual, rule_toxicity, rule_cib, rule_bot, rule_impersonation, triage
    status: ActionStatus = "pending"
    reply_text: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    dispatched_at: Optional[str] = None
    api_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ModerationAction:
        return cls(
            action_id=d.get("action_id", f"act_{uuid.uuid4().hex[:8]}"),
            comment_id=d.get("comment_id", ""),
            video_id=d.get("video_id", ""),
            author_id=d.get("author_id", ""),
            author_name=d.get("author_name", "Anonymous"),
            action_type=d.get("action_type", "heldForReview"),
            ban_author=bool(d.get("ban_author", False)),
            reason=d.get("reason", ""),
            trigger_source=d.get("trigger_source", "manual"),
            status=d.get("status", "pending"),
            reply_text=d.get("reply_text"),
            created_at=d.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            dispatched_at=d.get("dispatched_at"),
            api_response=d.get("api_response"),
        )


@dataclass
class ModerationAuditRecord:
    """Immutable audit record representing an executed or simulated moderation action."""

    record_id: str
    timestamp: str
    action_id: str
    comment_id: str
    author_name: str
    action_type: str
    ban_author: bool
    is_dry_run: bool
    status: str  # SUCCESS, SIMULATED, FAILED
    detail: str
    request_spec: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ModerationAuditRecord:
        return cls(
            record_id=d.get("record_id", f"aud_{uuid.uuid4().hex[:8]}"),
            timestamp=d.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            action_id=d.get("action_id", ""),
            comment_id=d.get("comment_id", ""),
            author_name=d.get("author_name", ""),
            action_type=d.get("action_type", ""),
            ban_author=bool(d.get("ban_author", False)),
            is_dry_run=bool(d.get("is_dry_run", True)),
            status=d.get("status", "SIMULATED"),
            detail=d.get("detail", ""),
            request_spec=d.get("request_spec"),
        )


@dataclass
class ModerationQueueState:
    """Represents the complete serializable state of the moderation queue."""

    queue_version: str = "1.0"
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_enqueued: int = 0
    total_approved: int = 0
    total_dispatched: int = 0
    actions: List[ModerationAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_version": self.queue_version,
            "updated_at": self.updated_at,
            "total_enqueued": len(self.actions),
            "total_approved": sum(
                1 for a in self.actions if a.status == "approved"
            ),
            "total_dispatched": sum(
                1 for a in self.actions if a.status == "dispatched"
            ),
            "actions": [a.to_dict() for a in self.actions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ModerationQueueState:
        raw_actions = d.get("actions", [])
        actions = [ModerationAction.from_dict(a) for a in raw_actions]
        return cls(
            queue_version=d.get("queue_version", "1.0"),
            updated_at=d.get(
                "updated_at", datetime.now(timezone.utc).isoformat()
            ),
            total_enqueued=len(actions),
            total_approved=sum(1 for a in actions if a.status == "approved"),
            total_dispatched=sum(
                1 for a in actions if a.status == "dispatched"
            ),
            actions=actions,
        )


class YouTubeModerationEngine:
    """Production rules engine and dispatcher for YouTube comment moderation."""

    def __init__(
        self,
        queue_path: str = "data/output/moderation_queue.json",
        audit_path: str = "data/output/moderation_audit_log.json",
        api_key: Optional[str] = None,
        oauth_token: Optional[str] = None,
    ) -> None:
        self.queue_path = Path(queue_path)
        self.audit_path = Path(audit_path)
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
        self.oauth_token = oauth_token or os.environ.get("YOUTUBE_OAUTH_TOKEN")
        self.state: ModerationQueueState = self.load_queue()

    # -------------------------------------------------------------------------
    # Queue & Audit Persistence
    # -------------------------------------------------------------------------

    def load_queue(self) -> ModerationQueueState:
        """Loads moderation queue from disk atomically."""
        if not self.queue_path.exists():
            return ModerationQueueState()
        try:
            with open(self.queue_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ModerationQueueState.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load moderation queue from {self.queue_path}: {e}")
            return ModerationQueueState()

    def save_queue(self) -> None:
        """Saves current moderation queue state atomically to disk."""
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        self.state.total_enqueued = len(self.state.actions)
        self.state.total_approved = sum(
            1 for a in self.state.actions if a.status == "approved"
        )
        self.state.total_dispatched = sum(
            1 for a in self.state.actions if a.status == "dispatched"
        )
        tmp_path = self.queue_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.queue_path)

    def load_audit_log(self, limit: int = 100) -> List[ModerationAuditRecord]:
        """Loads chronological audit records from disk."""
        if not self.audit_path.exists():
            return []
        try:
            with open(self.audit_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records = [ModerationAuditRecord.from_dict(r) for r in data]
                return records[-limit:]
            return []
        except Exception as e:
            logger.warning(f"Failed to load audit log from {self.audit_path}: {e}")
            return []

    def append_audit_record(self, record: ModerationAuditRecord) -> None:
        """Appends an immutable record to the moderation audit trail."""
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        records: List[Dict[str, Any]] = []
        if self.audit_path.exists():
            try:
                with open(self.audit_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records = data
            except Exception:
                records = []

        records.append(record.to_dict())
        tmp_path = self.audit_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.audit_path)

    # -------------------------------------------------------------------------
    # Queue Management Operations
    # -------------------------------------------------------------------------

    def enqueue_action(
        self,
        comment_id: str,
        video_id: str = "",
        author_id: str = "",
        author_name: str = "Anonymous",
        action_type: ModerationActionType = "heldForReview",
        ban_author: bool = False,
        reason: str = "",
        trigger_source: str = "manual",
        reply_text: Optional[str] = None,
        status: ActionStatus = "pending",
    ) -> ModerationAction:
        """Adds a moderation action to the queue if not already present."""
        # Check duplicate
        for existing in self.state.actions:
            if existing.comment_id == comment_id and existing.action_type == action_type:
                return existing

        action = ModerationAction(
            action_id=f"act_{uuid.uuid4().hex[:8]}",
            comment_id=comment_id,
            video_id=video_id,
            author_id=author_id,
            author_name=author_name,
            action_type=action_type,
            ban_author=ban_author,
            reason=reason,
            trigger_source=trigger_source,
            status=status,
            reply_text=reply_text,
        )
        self.state.actions.append(action)
        self.save_queue()
        logger.info(f"Enqueued moderation action {action.action_id} for comment {comment_id} [{action_type}]")
        return action

    def update_action_status(self, action_id: str, new_status: ActionStatus) -> bool:
        """Updates the lifecycle status of an action."""
        for a in self.state.actions:
            if a.action_id == action_id:
                a.status = new_status
                self.save_queue()
                return True
        return False

    def approve_action(self, action_id: str) -> bool:
        """Marks an action as approved for dispatch."""
        return self.update_action_status(action_id, "approved")

    def approve_all(self, status_filter: ActionStatus = "pending") -> int:
        """Approves all actions matching status_filter."""
        count = 0
        for a in self.state.actions:
            if a.status == status_filter:
                a.status = "approved"
                count += 1
        if count > 0:
            self.save_queue()
        return count

    def reject_action(self, action_id: str) -> bool:
        """Marks an action as rejected (not to be dispatched)."""
        return self.update_action_status(action_id, "rejected")

    def delete_action(self, action_id: str) -> bool:
        """Permanently removes an action from the queue."""
        initial_len = len(self.state.actions)
        self.state.actions = [a for a in self.state.actions if a.action_id != action_id]
        if len(self.state.actions) < initial_len:
            self.save_queue()
            return True
        return False

    def clear_queue(self, status_filter: Optional[str] = None) -> int:
        """Clears actions from queue, optionally filtering by status."""
        if status_filter is None:
            count = len(self.state.actions)
            self.state.actions = []
        else:
            initial_len = len(self.state.actions)
            self.state.actions = [a for a in self.state.actions if a.status != status_filter]
            count = initial_len - len(self.state.actions)
        self.save_queue()
        return count

    # -------------------------------------------------------------------------
    # Automated Rule Evaluation
    # -------------------------------------------------------------------------

    def scan_and_apply_rules(
        self,
        rule_categories: Optional[Sequence[str]] = None,
        mock: bool = False,
    ) -> List[ModerationAction]:
        """Scans forensic Parquet layers and enqueues violation actions according to policy rules."""
        selected_rules = set(rule_categories or ["all"])
        run_all = "all" in selected_rules
        enqueued: List[ModerationAction] = []

        if mock:
            return self._scan_mock_rules(selected_rules)

        # 1. Toxicity Catalysts Rule
        if run_all or "toxicity" in selected_rules:
            cat_path = Path("data/output/troll_catalysts.parquet")
            if cat_path.exists():
                try:
                    df_tox = pd.read_parquet(cat_path)
                    for _, row in df_tox.head(15).iterrows():
                        cid = str(row.get("comment_id", ""))
                        if not cid:
                            continue
                        auth = str(row.get("author_name", "Suspect Troll"))
                        score = float(row.get("catalyst_score", 0.0))
                        act = self.enqueue_action(
                            comment_id=cid,
                            video_id=str(row.get("video_id", "")),
                            author_name=auth,
                            action_type="heldForReview",
                            ban_author=score >= 25.0,
                            reason=f"Toxicity Contagion Catalyst (Score: {score:.1f})",
                            trigger_source="rule_toxicity",
                        )
                        enqueued.append(act)
                except Exception as e:
                    logger.warning(f"Failed scanning troll catalysts: {e}")

        # 2. Coordinated Inauthentic Behavior (CIB) Rule
        if run_all or "cib" in selected_rules:
            cib_path = Path("data/output/cib_coordinated_comments.parquet")
            if cib_path.exists():
                try:
                    df_cib = pd.read_parquet(cib_path)
                    for _, row in df_cib.head(15).iterrows():
                        for cid_col, auth_col in [("comment_id_a", "author_a"), ("comment_id_b", "author_b")]:
                            cid = str(row.get(cid_col, ""))
                            if not cid:
                                continue
                            auth = str(row.get(auth_col, "CIB Suspect"))
                            delta_t = float(row.get("time_delta_seconds", 0.0))
                            act = self.enqueue_action(
                                comment_id=cid,
                                video_id=str(row.get("video_id", "")),
                                author_name=auth,
                                action_type="heldForReview",
                                ban_author=True,
                                reason=f"Synchronized Astroturfing Ring (Δt: {delta_t:.0f}s)",
                                trigger_source="rule_cib",
                            )
                            enqueued.append(act)
                except Exception as e:
                    logger.warning(f"Failed scanning CIB rings: {e}")

        # 3. Spam Bot Duplicate Clusters Rule
        if run_all or "bot" in selected_rules:
            bot_path = Path("data/output/bot_classifications.parquet")
            if bot_path.exists():
                try:
                    df_bot = pd.read_parquet(bot_path)
                    suspects = df_bot[df_bot.get("is_bot", False) == True].head(15)  # noqa: E712
                    for _, row in suspects.iterrows():
                        cid = str(row.get("comment_id", ""))
                        if not cid:
                            continue
                        auth = str(row.get("author_name", "Bot Spammer"))
                        act = self.enqueue_action(
                            comment_id=cid,
                            video_id=str(row.get("video_id", "")),
                            author_name=auth,
                            action_type="markAsSpam",
                            ban_author=False,
                            reason="Automated BPE duplicate text pattern",
                            trigger_source="rule_bot",
                        )
                        enqueued.append(act)
                except Exception as e:
                    logger.warning(f"Failed scanning bot classifications: {e}")

        # 4. Creator Impersonation Rule
        if run_all or "impersonation" in selected_rules:
            imp_path = Path("data/output/impersonation_detection.parquet")
            if imp_path.exists():
                try:
                    df_imp = pd.read_parquet(imp_path)
                    for _, row in df_imp.head(10).iterrows():
                        cid = str(row.get("comment_id", ""))
                        if not cid:
                            continue
                        auth = str(row.get("author_name", "Impersonator"))
                        act = self.enqueue_action(
                            comment_id=cid,
                            video_id=str(row.get("video_id", "")),
                            author_name=auth,
                            action_type="rejected",
                            ban_author=True,
                            reason="Unauthorized channel name impersonation fraud",
                            trigger_source="rule_impersonation",
                        )
                        enqueued.append(act)
                except Exception as e:
                    logger.warning(f"Failed scanning impersonation: {e}")

        return enqueued

    def _scan_mock_rules(self, selected_rules: set[str]) -> List[ModerationAction]:
        """Generates synthetic rule-matched actions for offline testing."""
        enqueued: List[ModerationAction] = []
        run_all = "all" in selected_rules

        mock_violations = [
            ("tox_001", "vid_demo_1", "FlameBait_99", "heldForReview", True, "Toxicity Contagion Catalyst (Score: 28.5)", "rule_toxicity"),
            ("cib_002", "vid_demo_1", "CryptoBot_A", "heldForReview", True, "Synchronized Astroturfing Ring (Δt: 12s)", "rule_cib"),
            ("bot_003", "vid_demo_2", "SpamKing_WhatsApp", "markAsSpam", False, "Automated BPE duplicate text pattern", "rule_bot"),
            ("imp_004", "vid_demo_2", "Official Support Team", "rejected", True, "Unauthorized channel name impersonation fraud", "rule_impersonation"),
            ("tox_005", "vid_demo_3", "AggroTroll_42", "heldForReview", False, "Severe negativity surge ($R_0 = 2.1$)", "rule_toxicity"),
        ]

        for cid, vid, auth, act_type, ban, reason, trig in mock_violations:
            rule_cat = trig.replace("rule_", "")
            if run_all or rule_cat in selected_rules:
                act = self.enqueue_action(
                    comment_id=cid,
                    video_id=vid,
                    author_name=auth,
                    action_type=act_type,  # type: ignore
                    ban_author=ban,
                    reason=reason,
                    trigger_source=trig,
                )
                enqueued.append(act)

        return enqueued

    # -------------------------------------------------------------------------
    # Dispatch & Execution Engine
    # -------------------------------------------------------------------------

    def dispatch_action(
        self,
        action: ModerationAction,
        dry_run: bool = True,
        oauth_token: Optional[str] = None,
    ) -> ModerationAuditRecord:
        """Dispatches a single moderation action, either simulated in dry-run or via live API."""
        now_iso = datetime.now(timezone.utc).isoformat()
        token = oauth_token or self.oauth_token

        # Build HTTP Request Specification
        endpoint_map = {
            "heldForReview": "https://www.googleapis.com/youtube/v3/comments/setModerationStatus",
            "rejected": "https://www.googleapis.com/youtube/v3/comments/setModerationStatus",
            "published": "https://www.googleapis.com/youtube/v3/comments/setModerationStatus",
            "markAsSpam": "https://www.googleapis.com/youtube/v3/comments/markAsSpam",
            "postReply": "https://www.googleapis.com/youtube/v3/comments?part=snippet",
        }

        req_spec: Dict[str, Any] = {
            "method": "POST",
            "url": endpoint_map.get(action.action_type, "https://www.googleapis.com/youtube/v3/comments"),
            "params": {
                "id": action.comment_id,
            },
            "headers": {
                "Authorization": f"Bearer {token[:6]}..." if token else "Bearer <MOCK_OAUTH_TOKEN>",
                "Content-Type": "application/json",
            }
        }

        if action.action_type in ["heldForReview", "rejected", "published"]:
            req_spec["params"]["moderationStatus"] = action.action_type
            if action.ban_author:
                req_spec["params"]["banAuthor"] = "true"

        if action.action_type == "postReply" and action.reply_text:
            req_spec["body"] = {
                "snippet": {
                    "parentId": action.comment_id,
                    "textOriginal": action.reply_text,
                }
            }

        # Dry-run execution
        if dry_run or not token:
            detail = (
                f"[SIMULATED] Moderation {action.action_type} for comment {action.comment_id} "
                f"(author: '{action.author_name}', ban: {action.ban_author})"
            )
            action.status = "dispatched"
            action.dispatched_at = now_iso
            action.api_response = {
                "status_code": 204 if action.action_type != "postReply" else 200,
                "message": "Simulated dry-run dispatch successful (no live API mutation).",
                "request_spec": req_spec,
            }
            self.save_queue()

            audit_rec = ModerationAuditRecord(
                record_id=f"aud_{uuid.uuid4().hex[:8]}",
                timestamp=now_iso,
                action_id=action.action_id,
                comment_id=action.comment_id,
                author_name=action.author_name,
                action_type=action.action_type,
                ban_author=action.ban_author,
                is_dry_run=True,
                status="SIMULATED",
                detail=detail,
                request_spec=req_spec,
            )
            self.append_audit_record(audit_rec)
            return audit_rec

        # Live remote execution via YouTube Data API
        try:
            import requests

            resp = requests.post(
                req_spec["url"],
                params=req_spec["params"],
                headers=req_spec["headers"],
                json=req_spec.get("body"),
                timeout=10,
            )
            if resp.status_code in [200, 204]:
                action.status = "dispatched"
                action.dispatched_at = now_iso
                action.api_response = {
                    "status_code": resp.status_code,
                    "message": "Live API dispatch successful.",
                }
                status_str = "SUCCESS"
                detail = f"[LIVE] Executed {action.action_type} on comment {action.comment_id}"
            else:
                action.status = "failed"
                action.api_response = {
                    "status_code": resp.status_code,
                    "error": resp.text,
                }
                status_str = "FAILED"
                detail = f"[FAILED] HTTP {resp.status_code} executing {action.action_type}: {resp.text}"

        except Exception as e:
            action.status = "failed"
            action.api_response = {"error": str(e)}
            status_str = "FAILED"
            detail = f"[EXCEPTION] Failed live execution: {e}"

        self.save_queue()
        audit_rec = ModerationAuditRecord(
            record_id=f"aud_{uuid.uuid4().hex[:8]}",
            timestamp=now_iso,
            action_id=action.action_id,
            comment_id=action.comment_id,
            author_name=action.author_name,
            action_type=action.action_type,
            ban_author=action.ban_author,
            is_dry_run=False,
            status=status_str,
            detail=detail,
            request_spec=req_spec,
        )
        self.append_audit_record(audit_rec)
        return audit_rec

    def dispatch_batch(
        self,
        action_ids: Optional[Sequence[str]] = None,
        dry_run: bool = True,
        max_actions: int = 50,
    ) -> Tuple[int, int, List[ModerationAuditRecord]]:
        """Dispatches all approved actions (or specified IDs) up to max_actions."""
        to_dispatch: List[ModerationAction] = []
        if action_ids:
            id_set = set(action_ids)
            to_dispatch = [a for a in self.state.actions if a.action_id in id_set and a.status == "approved"]
        else:
            to_dispatch = [a for a in self.state.actions if a.status == "approved"]

        to_dispatch = to_dispatch[:max_actions]
        success_count = 0
        fail_count = 0
        results: List[ModerationAuditRecord] = []

        for act in to_dispatch:
            rec = self.dispatch_action(act, dry_run=dry_run)
            results.append(rec)
            if rec.status in ["SUCCESS", "SIMULATED"]:
                success_count += 1
            else:
                fail_count += 1
            time.sleep(0.05)  # slight rate limit pause

        return success_count, fail_count, results

    # -------------------------------------------------------------------------
    # Telemetry & Export
    # -------------------------------------------------------------------------

    def get_summary_stats(self) -> Dict[str, Any]:
        """Returns high-level statistics for dashboard KPI cards and CLI status."""
        actions = self.state.actions
        audit_logs = self.load_audit_log(limit=500)
        return {
            "total_actions": len(actions),
            "pending": sum(1 for a in actions if a.status == "pending"),
            "approved": sum(1 for a in actions if a.status == "approved"),
            "rejected": sum(1 for a in actions if a.status == "rejected"),
            "dispatched": sum(1 for a in actions if a.status == "dispatched"),
            "failed": sum(1 for a in actions if a.status == "failed"),
            "total_audited": len(audit_logs),
            "audit_simulated": sum(1 for r in audit_logs if r.is_dry_run),
            "audit_live": sum(1 for r in audit_logs if not r.is_dry_run),
            "last_updated": self.state.updated_at,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the active queue as a Pandas DataFrame for UI and export."""
        if not self.state.actions:
            return pd.DataFrame(columns=[
                "action_id", "comment_id", "video_id", "author_name",
                "action_type", "ban_author", "reason", "trigger_source", "status", "created_at"
            ])
        rows = [a.to_dict() for a in self.state.actions]
        return pd.DataFrame(rows)

    def export_queue(self, path: Union[str, Path], export_format: str = "json") -> str:
        """Exports the active queue to a file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if export_format.lower() == "csv":
            df = self.to_dataframe()
            df.to_csv(out, index=False, encoding="utf-8")
        else:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)
        return str(out)

    def generate_mock_queue(self) -> int:
        """Seeds the queue with realistic mock actions across all action types."""
        samples = [
            ("cmt_troll_881", "vid_99", "ToxicUser_X", "heldForReview", True, "Toxicity Contagion Catalyst ($R_0 = 2.4$)", "rule_toxicity"),
            ("cmt_cib_104", "vid_99", "RingMember_Alpha", "heldForReview", True, "Synchronized Astroturfing Ring (Δt: 8s)", "rule_cib"),
            ("cmt_spam_552", "vid_88", "CryptoTelegramBot", "markAsSpam", False, "Repeated BPE duplicate text pattern", "rule_bot"),
            ("cmt_imp_901", "vid_88", "Official Channel Staff", "rejected", True, "Creator display name impersonator fraud", "rule_impersonation"),
            ("cmt_q_310", "vid_77", "CuriousViewer_12", "postReply", False, "Creator triage high-priority question", "triage"),
            ("cmt_praise_411", "vid_77", "SuperFan_VIP", "published", False, "Champion cohort positive tone anchor", "triage"),
        ]
        count = 0
        for cid, vid, auth, act_t, ban, reason, trig in samples:
            self.enqueue_action(
                comment_id=cid,
                video_id=vid,
                author_name=auth,
                action_type=act_t,  # type: ignore
                ban_author=ban,
                reason=reason,
                trigger_source=trig,
                reply_text="Thank you for the thoughtful feedback! We are actively investigating this." if act_t == "postReply" else None,
            )
            count += 1
        return count


# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ytint-moderate: Bi-Directional YouTube Moderation Action Dispatcher & Policy Hub"
    )
    parser.add_argument("--status", action="store_true", help="Display queue summary stats and recent audit telemetry")
    parser.add_argument("--queue", action="store_true", help="List all actions currently in the moderation queue")
    parser.add_argument("--rules", nargs="*", metavar="RULE", help="Evaluate moderation policy rules (all, toxicity, cib, bot, impersonation)")
    parser.add_argument("--approve", metavar="ACTION_ID", help="Approve a specific action ID for execution")
    parser.add_argument("--approve-all", action="store_true", help="Approve all pending actions in queue")
    parser.add_argument("--reject", metavar="ACTION_ID", help="Reject a specific action ID")
    parser.add_argument("--dispatch", action="store_true", help="Dispatch approved moderation actions")
    parser.add_argument("--live", action="store_true", help="Execute live remote API calls (default is safe dry-run)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Simulate moderation actions safely (default: True)")
    parser.add_argument("--audit", action="store_true", help="Display recent records from the moderation audit log")
    parser.add_argument("--mock", action="store_true", help="Seed mock actions for offline testing and demonstration")
    parser.add_argument("--clear", action="store_true", help="Clear all actions from the queue")
    parser.add_argument("--export", metavar="PATH", help="Export moderation queue to JSON or CSV file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = YouTubeModerationEngine()

    if args.mock:
        added = engine.generate_mock_queue()
        print(f"✅ Generated {added} mock moderation actions into queue.")

    if args.rules is not None:
        cats = args.rules if len(args.rules) > 0 else ["all"]
        print(f"🔍 Scanning forensic layers for moderation rules: {cats}...")
        enqueued = engine.scan_and_apply_rules(cats, mock=args.mock)
        print(f"✅ Rule evaluation complete: {len(enqueued)} actions enqueued.")

    if args.approve:
        if engine.approve_action(args.approve):
            print(f"✅ Approved action: {args.approve}")
        else:
            print(f"❌ Action ID not found: {args.approve}")

    if args.approve_all:
        count = engine.approve_all()
        print(f"✅ Approved {count} pending action(s).")

    if args.reject:
        if engine.reject_action(args.reject):
            print(f"🚫 Rejected action: {args.reject}")
        else:
            print(f"❌ Action ID not found: {args.reject}")

    if args.dispatch:
        is_live = args.live
        dry_run = not is_live
        mode_str = "LIVE" if is_live else "DRY-RUN (Simulated)"
        print(f"🚀 Dispatching approved actions in [{mode_str}] mode...")
        succ, fail, results = engine.dispatch_batch(dry_run=dry_run)
        print(f"✅ Batch dispatch complete: {succ} successful, {fail} failed.")
        for r in results[:5]:
            print(f"   • [{r.status}] {r.action_type} on {r.comment_id} ({r.author_name}): {r.detail}")

    if args.clear:
        cleared = engine.clear_queue()
        print(f"🗑️ Cleared {cleared} actions from queue.")

    if args.export:
        fmt = "csv" if args.export.endswith(".csv") else "json"
        out = engine.export_queue(args.export, export_format=fmt)
        print(f"💾 Exported queue to: {out}")

    if args.queue:
        df = engine.to_dataframe()
        print("\n" + "=" * 80)
        print("               YOUTUBE MODERATION ACTION QUEUE")
        print("=" * 80)
        if df.empty:
            print("Queue is empty. Use --rules or --mock to enqueue actions.")
        else:
            cols = ["action_id", "comment_id", "author_name", "action_type", "ban_author", "status", "trigger_source"]
            display_cols = [c for c in cols if c in df.columns]
            print(df[display_cols].to_string(index=False))
        print("=" * 80 + "\n")

    if args.audit:
        records = engine.load_audit_log(limit=15)
        print("\n" + "=" * 80)
        print("             YOUTUBE MODERATION AUDIT LOG (RECENT 15)")
        print("=" * 80)
        if not records:
            print("Audit log is empty.")
        else:
            for r in records:
                mode = "DRY-RUN" if r.is_dry_run else "LIVE"
                print(f"[{r.timestamp}] [{r.status}] [{mode}] {r.action_type:<14} {r.comment_id:<14} {r.author_name:<16} ban={r.ban_author}")
                print(f"   └─ {r.detail}")
        print("=" * 80 + "\n")

    if args.status or (not any(vars(args).values())):
        stats = engine.get_summary_stats()
        print("\n" + "=" * 60)
        print("        YOUTUBE MODERATION ENGINE TELEMETRY")
        print("=" * 60)
        print(f"Total Enqueued Actions:    {stats['total_actions']}")
        print(f"Pending Approval:          {stats['pending']}")
        print(f"Approved for Dispatch:     {stats['approved']}")
        print(f"Rejected Actions:          {stats['rejected']}")
        print(f"Dispatched Actions:        {stats['dispatched']}")
        print(f"Failed Actions:            {stats['failed']}")
        print(f"Total Audited Records:     {stats['total_audited']}")
        print(f"  • Simulated (Dry-Run):   {stats['audit_simulated']}")
        print(f"  • Live Executions:       {stats['audit_live']}")
        print(f"Last Updated:              {stats['last_updated']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
