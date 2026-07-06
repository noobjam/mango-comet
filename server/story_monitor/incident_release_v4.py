"""Release-watermark contract shared by Incident V4 build stages."""

from __future__ import annotations

from datetime import datetime, timezone


CORRECTION_POLICY = {
    "mode": "append_only_no_revisions",
    "late_corrections_supported": False,
    "require_explicit_revision_supersession": True,
    "failure_mode": (
        "Reject changes to published natural keys until a future explicit "
        "revision/supersession contract is implemented."
    ),
}


def normalize_released_at(value: str) -> str:
    """Return one timezone-aware release watermark in canonical UTC form."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("released_at is required")
    candidate = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError("released_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("released_at must include a UTC offset")
    utc = parsed.astimezone(timezone.utc)
    return utc.isoformat(timespec="microseconds").removesuffix("+00:00") + "Z"


def validate_correction_policy(value: object) -> None:
    if value != CORRECTION_POLICY:
        raise ValueError(
            "Incident V4 manifest must fail closed on late corrections until an "
            "explicit revision/supersession contract exists"
        )
