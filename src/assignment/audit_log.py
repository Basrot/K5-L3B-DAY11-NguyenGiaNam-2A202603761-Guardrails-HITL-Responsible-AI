"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._open_rows: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start an auditable request record and remember its monotonic start time."""
        key = request_id or user_id
        started = time.monotonic()
        row = {
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
        }
        self.logs.append(row)
        self._open[key] = started
        self._open_rows[key] = row
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete the matching record with output, decision and elapsed time."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        row = self._open_rows.pop(key, None)
        if row is None:
            row = {
                "request_id": request_id,
                "user_id": user_id,
                "input": "",
                "started_at": utc_now_iso(),
            }
            self.logs.append(row)
        row.update(
            {
                "output": text,
                "blocked": bool(blocked),
                "layer": layer,
                "completed_at": utc_now_iso(),
                "latency_ms": round(max(0.0, time.monotonic() - started) * 1000, 3)
                if started is not None
                else 0.0,
            }
        )
        return row

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
