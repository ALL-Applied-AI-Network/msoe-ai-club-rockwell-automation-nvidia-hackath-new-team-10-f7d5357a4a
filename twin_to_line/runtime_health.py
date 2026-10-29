"""Fail-closed, observational health state for the live inspection runtime.

This module does not make manufacturing decisions. It invalidates availability
while each input is processed and reports whether the latest completed input
produced a successful inspection. A recovery signal or a decodable frame is not
enough to publish availability: only a new ``scored`` event can do that.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import os
import stat
import tempfile
from typing import Any


SCHEMA_VERSION = "runtime-health-v1"
AVAILABLE = "INSPECTION_AVAILABLE"
UNAVAILABLE = "INSPECTION_UNAVAILABLE"

VALID_FRAME_STATUSES = frozenset({"arm", "settling", "unchanged", "scored"})
FAULT_STATUSES = frozenset({"source_error", "unreadable", "wrong_size", "refused", "error"})
SOURCE_MARKER_STATUSES = frozenset({"source_error", "source_recovered"})
FAIL_CLOSED_STATUS = (
    b'{"schema_version":"runtime-health-v1","state":"INSPECTION_UNAVAILABLE",'
    b'"reason_code":"status_persistence_failed","current_input_inspected":false,'
    b'"availability_scope":"processing_runtime_only","source_liveness_guaranteed":false,'
    b'"operation":{"mode":"shadow_evaluation_only","automatic_pass_authorized":false,'
    b'"actuation_enabled":false,"automatic_release_enabled":false}}\n'
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _percentile(values: list[float], q: float) -> float | None:
    """Linearly interpolated percentile, matching the usual inclusive endpoints."""
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)
    return round(value, 3)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": round(min(values), 3) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": round(max(values), 3) if values else None,
    }


def write_json_atomic(path: str, value: dict[str, Any]) -> None:
    """Replace a JSON status file without exposing a truncated intermediate file."""
    directory = os.path.dirname(os.path.abspath(path))
    try:
        existing = os.stat(path, follow_symlinks=False)
        mode = stat.S_IMODE(existing.st_mode) if stat.S_ISREG(existing.st_mode) else 0o644
    except FileNotFoundError:
        mode = 0o644
    descriptor, temporary = tempfile.mkstemp(prefix=".runtime-json-", suffix=".tmp", dir=directory)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def invalidate_runtime_status(path: str) -> bool:
    """Best-effort removal or fail-closed invalidation of an ephemeral health file.

    This is intentionally *not* used for ``session.json``: durable session
    evidence retains the previous complete version when its replacement fails.
    """
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        pass

    descriptor = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC)
        remaining = memoryview(FAIL_CLOSED_STATUS)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("could not invalidate runtime status")
            remaining = remaining[written:]
        os.fsync(descriptor)
        return True
    except OSError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


class RuntimeHealth:
    """Track inspection availability without ever granting release/PASS authority."""

    def __init__(self, *, started_utc: str | None = None) -> None:
        self.state = UNAVAILABLE
        self.reason_code = "no_successful_inspection_yet"
        self.status_counts: Counter[str] = Counter()
        self.latencies_ms: list[float] = []
        self.scored_latencies_ms: list[float] = []
        self.consecutive_faults = 0
        self.last_input: dict[str, Any] | None = None
        self.last_fault: dict[str, Any] | None = None
        self.last_successful_inspection: dict[str, Any] | None = None
        self.closing_utc: str | None = None
        self.ended_utc: str | None = None
        self.stop_reason: str | None = None
        self.status_updated_utc = started_utc

    def begin_input(self, *, timestamp_utc: str, input_file: str) -> None:
        """Invalidate prior availability before any work begins on a new input."""
        if self.ended_utc is not None:
            raise RuntimeError("cannot begin input after the runtime health record has ended")
        self.state = UNAVAILABLE
        self.reason_code = "input_processing"
        self.status_updated_utc = timestamp_utc
        self.last_input = {
            "input_started_utc": timestamp_utc,
            "input_file": input_file,
            "status": "processing",
            "inspection_completed": False,
        }

    def record(
        self,
        *,
        status: str,
        timestamp_utc: str,
        input_started_utc: str | None = None,
        latency_ms: float,
        input_file: str,
        event: int | None = None,
        event_frame: str | None = None,
        note: str = "",
    ) -> None:
        if self.ended_utc is not None:
            raise RuntimeError("cannot record input after the runtime health record has ended")
        latency = float(latency_ms)
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("latency_ms must be finite and nonnegative")

        self.status_counts[status] += 1
        self.latencies_ms.append(latency)
        if status == "scored":
            self.scored_latencies_ms.append(latency)
        self.status_updated_utc = timestamp_utc
        self.last_input = {
            "timestamp_utc": timestamp_utc,
            "input_started_utc": input_started_utc or timestamp_utc,
            "completed_utc": timestamp_utc,
            "input_file": input_file,
            "status": status,
            "inspection_completed": status == "scored",
        }
        if event is not None:
            self.last_input["event"] = int(event)
        if event_frame:
            self.last_input["event_frame"] = event_frame

        if status == "scored":
            self.state = AVAILABLE
            self.reason_code = "last_event_scored"
            self.consecutive_faults = 0
            self.last_successful_inspection = {
                "timestamp_utc": timestamp_utc,
                "input_started_utc": input_started_utc or timestamp_utc,
                "completed_utc": timestamp_utc,
                "input_file": input_file,
                "event": int(event) if event is not None else None,
                "event_frame": event_frame or input_file,
            }
            return

        if status in FAULT_STATUSES or status not in VALID_FRAME_STATUSES | {"source_recovered"}:
            self.state = UNAVAILABLE
            self.reason_code = f"input_{status}" if status else "input_unknown_status"
            self.consecutive_faults += 1
            self.last_fault = {
                "timestamp_utc": timestamp_utc,
                "input_started_utc": input_started_utc or timestamp_utc,
                "completed_utc": timestamp_utc,
                "input_file": input_file,
                "status": status or "unknown",
                "note": note,
            }
            return

        self.consecutive_faults = 0
        if status == "source_recovered":
            self.state = UNAVAILABLE
            self.reason_code = "source_recovered_awaiting_scored_event"
        else:
            self.state = UNAVAILABLE
            self.reason_code = f"input_not_scored_{status}"

    def end(self, *, timestamp_utc: str, stop_reason: str) -> None:
        self.ended_utc = timestamp_utc
        self.stop_reason = stop_reason
        self.state = UNAVAILABLE
        self.reason_code = "session_ended"
        self.status_updated_utc = timestamp_utc

    def begin_close(self, *, timestamp_utc: str) -> None:
        """Invalidate availability before any potentially blocking close work."""
        if self.ended_utc is not None:
            raise RuntimeError("cannot begin closing after the runtime health record has ended")
        self.closing_utc = timestamp_utc
        self.state = UNAVAILABLE
        self.reason_code = "session_closing"
        self.status_updated_utc = timestamp_utc

    def _metrics(self) -> dict[str, Any]:
        total = sum(self.status_counts.values())
        input_frames = total - sum(self.status_counts[s] for s in SOURCE_MARKER_STATUSES)
        valid_frames = sum(self.status_counts[s] for s in VALID_FRAME_STATUSES)
        scored = self.status_counts["scored"]
        failed_or_refused = max(0, input_frames - valid_frames)
        return {
            "records_total": total,
            "input_frames": input_frames,
            "valid_frames": valid_frames,
            "scored_events": scored,
            "failed_or_refused_inputs": failed_or_refused,
            "input_validity_coverage": _ratio(valid_frames, input_frames),
            "scored_input_rate": _ratio(scored, input_frames),
            "all_record_latency_ms": _latency_summary(self.latencies_ms),
            "scored_event_latency_ms": _latency_summary(self.scored_latencies_ms),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "state": self.state,
            "reason_code": self.reason_code,
            "status_updated_utc": self.status_updated_utc,
            "availability_scope": "processing_runtime_only",
            "source_liveness_guaranteed": False,
            "current_input_inspected": bool(
                self.last_input
                and self.last_input.get("inspection_completed")
                and self.closing_utc is None
                and self.ended_utc is None
            ),
            "operation": {
                "mode": "shadow_evaluation_only",
                "automatic_pass_authorized": False,
                "actuation_enabled": False,
                "automatic_release_enabled": False,
            },
            "consecutive_faults": self.consecutive_faults,
            "last_input": self.last_input,
            "last_fault": self.last_fault,
            "last_successful_inspection": self.last_successful_inspection,
            "closing_utc": self.closing_utc,
            "ended_utc": self.ended_utc,
            "stop_reason": self.stop_reason,
            "status_counts": dict(self.status_counts),
            "metrics": self._metrics(),
        }
