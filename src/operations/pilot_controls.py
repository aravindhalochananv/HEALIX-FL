"""Operational controls for a controlled internal research pilot.

Audit events intentionally exclude raw patient values, identifiers, and model
probabilities. This module is not a replacement for institutional governance.
"""

import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "results" / "operations" / "audit.jsonl"
DEFAULT_RETENTION_DAYS = 30
APP_VERSION = "1.1-pilot"
_SESSION_ID = os.environ.get("HEALIX_SESSION_ID", secrets.token_hex(12))


def _audit_path():
    configured = os.environ.get("HEALIX_AUDIT_LOG_PATH")
    return Path(configured) if configured else DEFAULT_AUDIT_PATH


def get_session_id():
    """Return a non-identifying process session identifier."""
    return _SESSION_ID


def _retention_days():
    value = os.environ.get("HEALIX_AUDIT_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))
    try:
        days = int(value)
    except ValueError as exc:
        raise ValueError("HEALIX_AUDIT_RETENTION_DAYS must be an integer.") from exc
    if days < 1:
        raise ValueError("HEALIX_AUDIT_RETENTION_DAYS must be at least 1.")
    return days


def purge_expired_audit_events():
    """Purge operational events older than the configured retention period."""
    path = _audit_path()
    if not path.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=_retention_days())
    retained = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
            timestamp = datetime.fromisoformat(event["timestamp"])
        except (ValueError, KeyError, json.JSONDecodeError):
            removed += 1
            continue
        if timestamp >= cutoff:
            retained.append(json.dumps(event, separators=(",", ":")))
        else:
            removed += 1
    path.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    return removed


def record_audit_event(event, **metadata):
    """Append non-identifying operational metadata and enforce retention."""
    if not event:
        raise ValueError("event is required.")
    path = _audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
        "session_id": get_session_id(),
        "app_version": APP_VERSION,
    }
    payload.update({key: str(value) for key, value in metadata.items()})
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
    purge_expired_audit_events()


def get_pilot_status():
    """Return implementation status for the controlled pilot checklist.

    The evidence and governance entries represent a demonstration package
    generated from repository artifacts, not external clinical approval.
    """
    return {
        "Audit logging": True,
        "Data retention enforcement": True,
        "Operational monitoring": True,
        "Model validation evidence (demonstration)": True,
        "Privacy / governance package (demonstration)": True,
        "SSO / authentication": False,
        "HTTPS termination": False,
        "External clinical validation": False,
        "Institutional approval": False,
    }
