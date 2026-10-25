"""Read-only adapter for the checker's append-only output directory."""

from __future__ import annotations

import csv
import io
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import EXPECTED_POSITIONS, EventSnapshot, FinalState, HoleResult


class OutputSourceError(RuntimeError):
    """Base class for checker-output failures."""


class OutputUnavailable(OutputSourceError):
    """A required checker artifact does not exist or cannot be read."""


class OutputBusy(OutputSourceError):
    """An artifact changed while it was being read."""


class InvalidOutput(OutputSourceError):
    """A complete-looking event violates the handoff contract."""


@dataclass(frozen=True, slots=True)
class SourceHealth:
    state: str
    source_age_seconds: float | None
    last_trial_status: str | None
    message: str


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("numeric value must be finite")
    return parsed


def _optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


class CheckerOutputRepository:
    """Reads checker artifacts without modifying or locking them.

    The repository retains the last known-good values when a writer is in the
    middle of appending or atomically replacing an artifact.
    """

    REQUIRED_COLUMNS = {
        "run",
        "event",
        "frame",
        "position",
        "predicted_state",
        "final_state",
    }

    def __init__(self, output_dir: str | Path, *, stale_after_seconds: float = 30.0):
        root = Path(output_dir).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)

        self.output_dir = root
        self.predictions_path = root / "predictions.csv"
        self.session_path = root / "session.json"
        self.trial_log_path = root / "trial_log.csv"
        self.overlay_path = root / "overlay_latest.png"
        self.stale_after_seconds = stale_after_seconds

        self._event_signature: tuple[int, int] | None = None
        self._event_cache: EventSnapshot | None = None
        self._session_signature: tuple[int, int] | None = None
        self._session_cache: dict[str, Any] = {}
        self._trial_signature: tuple[int, int] | None = None
        self._trial_status_cache: str | None = None
        self._overlay_signature: tuple[int, int] | None = None
        self._overlay_cache: bytes | None = None
        self.last_warning = ""

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        try:
            stat = path.stat()
        except OSError as exc:
            raise OutputUnavailable(f"cannot stat {path.name}: {exc}") from exc
        return stat.st_mtime_ns, stat.st_size

    @classmethod
    def _stable_text(cls, path: Path) -> tuple[str, tuple[int, int]]:
        before = cls._signature(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OutputUnavailable(f"cannot read {path.name}: {exc}") from exc
        after = cls._signature(path)
        if before != after:
            raise OutputBusy(f"{path.name} changed while being read")
        return text, after

    @classmethod
    def _stable_bytes(cls, path: Path) -> tuple[bytes, tuple[int, int]]:
        before = cls._signature(path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise OutputUnavailable(f"cannot read {path.name}: {exc}") from exc
        after = cls._signature(path)
        if before != after:
            raise OutputBusy(f"{path.name} changed while being read")
        return data, after

    def latest_complete_event(self) -> EventSnapshot | None:
        try:
            signature = self._signature(self.predictions_path)
            if signature == self._event_signature:
                return self._event_cache
            text, stable_signature = self._stable_text(self.predictions_path)
        except OutputBusy as exc:
            self.last_warning = str(exc)
            return self._event_cache

        try:
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if reader.fieldnames is None:
                raise InvalidOutput("predictions.csv has no header")
            missing_columns = self.REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing_columns:
                raise InvalidOutput(
                    "predictions.csv is missing columns: " + ", ".join(sorted(missing_columns))
                )

            groups: dict[tuple[str, str], list[dict[str, str]]] = {}
            for row in reader:
                key = (row.get("run", ""), row.get("event", ""))
                groups.setdefault(key, []).append(row)
        except csv.Error as exc:
            raise InvalidOutput(f"cannot parse predictions.csv: {exc}") from exc

        warning = ""
        selected: EventSnapshot | None = None
        for rows in reversed(list(groups.values())):
            try:
                selected = self._parse_complete_event(rows)
            except InvalidOutput as exc:
                warning = str(exc)
                continue
            if selected is not None:
                break

        self._event_signature = stable_signature
        self._event_cache = selected
        self.last_warning = warning
        return selected

    def _parse_complete_event(self, rows: list[dict[str, str]]) -> EventSnapshot | None:
        by_position: dict[str, dict[str, str]] = {}
        for row in rows:
            position = row.get("position", "")
            if position not in EXPECTED_POSITIONS:
                raise InvalidOutput(f"event contains invalid position {position!r}")
            if position in by_position:
                raise InvalidOutput(f"event contains duplicate position {position}")
            by_position[position] = row

        if set(by_position) != set(EXPECTED_POSITIONS):
            return None

        ordered = [by_position[position] for position in EXPECTED_POSITIONS]
        for field in ("run", "event", "frame", "K"):
            values = {row.get(field, "") for row in ordered}
            if len(values) > 1:
                raise InvalidOutput(f"event has conflicting {field} values")

        holes: list[HoleResult] = []
        for row in ordered:
            try:
                final_state = FinalState(row["final_state"])
                score = _optional_float(row.get("defect_score"))
                cap_fraction = _optional_float(row.get("cap_frac"))
                cap_offset = _optional_float(row.get("cap_offset"))
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidOutput(
                    f"invalid result for {row.get('position', '?')}: {exc}"
                ) from exc

            holes.append(
                HoleResult(
                    position=row["position"],
                    predicted_state=row.get("predicted_state", ""),
                    final_state=final_state,
                    score=score,
                    reason=row.get("reason", ""),
                    cap_fraction=cap_fraction,
                    cap_offset=cap_offset,
                )
            )

        first = ordered[0]
        try:
            frontier = _optional_int(first.get("K"))
        except ValueError as exc:
            raise InvalidOutput(f"invalid fill frontier K: {exc}") from exc
        if frontier is not None and not 0 <= frontier <= 30:
            raise InvalidOutput("fill frontier K must be between 0 and 30")
        if not first["run"] or not first["event"] or not first["frame"]:
            raise InvalidOutput("run, event, and frame must be non-empty")

        return EventSnapshot(
            run=first["run"],
            event=first["event"],
            frame=first["frame"],
            fill_frontier=frontier,
            holes=tuple(holes),
        )

    def session_metadata(self) -> dict[str, Any]:
        try:
            signature = self._signature(self.session_path)
            if signature == self._session_signature:
                return dict(self._session_cache)
            text, stable_signature = self._stable_text(self.session_path)
            value = json.loads(text)
            if not isinstance(value, dict):
                raise InvalidOutput("session.json must contain an object")
        except (OutputBusy, json.JSONDecodeError) as exc:
            self.last_warning = f"session metadata unavailable: {exc}"
            return dict(self._session_cache)

        self._session_signature = stable_signature
        self._session_cache = value
        return dict(value)

    def _last_trial_status(self) -> str | None:
        try:
            signature = self._signature(self.trial_log_path)
            if signature == self._trial_signature:
                return self._trial_status_cache
            text, stable_signature = self._stable_text(self.trial_log_path)
            rows = deque(csv.DictReader(io.StringIO(text, newline="")), maxlen=1)
            status = rows[0].get("status") if rows else None
        except OutputSourceError:
            return self._trial_status_cache

        self._trial_signature = stable_signature
        self._trial_status_cache = status
        return status

    def source_health(self) -> SourceHealth:
        try:
            stat = self.trial_log_path.stat()
        except OSError as exc:
            return SourceHealth("UNAVAILABLE", None, None, f"trial log unavailable: {exc}")

        age = max(0.0, time.time() - stat.st_mtime)
        try:
            session = self.session_metadata()
        except OutputSourceError as exc:
            session = {}
            self.last_warning = str(exc)
        last_status = self._last_trial_status()

        if session.get("ended_utc"):
            state = "COMPLETE"
            message = session.get("stop_reason", "session complete")
        elif age > self.stale_after_seconds:
            state = "STALE"
            message = f"checker output has not changed for {age:.1f} seconds"
        else:
            state = "LIVE"
            message = ""

        if self.last_warning and not message:
            message = self.last_warning
        return SourceHealth(state, round(age, 1), last_status, message)

    def overlay_png(self) -> bytes | None:
        try:
            signature = self._signature(self.overlay_path)
            if signature == self._overlay_signature:
                return self._overlay_cache
            data, stable_signature = self._stable_bytes(self.overlay_path)
        except OutputSourceError as exc:
            self.last_warning = str(exc)
            return self._overlay_cache

        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            self.last_warning = "overlay_latest.png is not a valid PNG"
            return self._overlay_cache

        self._overlay_signature = stable_signature
        self._overlay_cache = data
        return data
