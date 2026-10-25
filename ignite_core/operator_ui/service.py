"""Application service that exposes a stable operator-view representation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from .models import EventSnapshot, FinalState
from .repository import CheckerOutputRepository, OutputSourceError


class SnapshotService:
    SCHEMA_VERSION = 1

    def __init__(self, repository: CheckerOutputRepository):
        self.repository = repository

    @staticmethod
    def _severity(
        snapshot: EventSnapshot | None,
        health_state: str,
    ) -> str:
        if snapshot is None:
            if health_state in {"UNAVAILABLE", "STALE"}:
                return "DATA_UNAVAILABLE"
            return "WAITING_FOR_EVENT"

        states = {hole.final_state for hole in snapshot.holes}

        if FinalState.DEFECT in states or FinalState.MISSED in states:
            return "ACTION_REQUIRED"

        if FinalState.UNSURE in states:
            return "REVIEW_REQUIRED"

        return "NORMAL"

    def snapshot_payload(self) -> dict[str, Any]:
        error = ""
        try:
            snapshot = self.repository.latest_complete_event()
        except OutputSourceError as exc:
            snapshot = None
            error = str(exc)

        health = self.repository.source_health()
        health_message = error or health.message
        severity = self._severity(snapshot, health.state)

        if snapshot is None:
            return {
                "schema_version": self.SCHEMA_VERSION,
                "health": {
                    "state": health.state,
                    "source_age_seconds": health.source_age_seconds,
                    "last_trial_status": health.last_trial_status,
                    "message": health_message,
                },
                "event": None,
                "counts": {state.value: 0 for state in FinalState},
                "holes": [],
            }

        counts = Counter(hole.final_state.value for hole in snapshot.holes)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "health": {
                "state": health.state,
                "source_age_seconds": health.source_age_seconds,
                "last_trial_status": health.last_trial_status,
                "message": health_message,
            },
            "event": {
                "run": snapshot.run,
                "event": snapshot.event,
                "frame": snapshot.frame,
                "fill_frontier": snapshot.fill_frontier,
                "severity": severity,
            },
            "counts": {state.value: counts.get(state.value, 0) for state in FinalState},
            "holes": [
                {
                    "position": hole.position,
                    "predicted_state": hole.predicted_state,
                    "final_state": hole.final_state.value,
                    "score": hole.score,
                    "reason": hole.reason,
                    "cap_fraction": hole.cap_fraction,
                    "cap_offset": hole.cap_offset,
                }
                for hole in snapshot.holes
            ],
        }

    def session_payload(self) -> dict[str, Any]:
        try:
            return self.repository.session_metadata()
        except OutputSourceError as exc:
            return {"error": str(exc)}

    def health_payload(self) -> dict[str, Any]:
        return asdict(self.repository.source_health())

    def overlay_png(self) -> bytes | None:
        return self.repository.overlay_png()
