"""Unit tests for operator-facing snapshot policy and serialization."""

import unittest

from operator_ui.models import (
    EXPECTED_POSITIONS,
    EventSnapshot,
    FinalState,
    HoleResult,
)
from operator_ui.repository import SourceHealth
from operator_ui.service import SnapshotService

VISUAL_BY_STATE = {
    FinalState.PASS: "pass",
    FinalState.DEFECT: "crooked_cap",
    FinalState.MISSED: "no_vial",
    FinalState.UNSURE: "unknown",
    FinalState.WAITING: "no_vial",
}


def make_snapshot(*, overrides=None):
    overrides = overrides or {}
    holes = []

    for position in EXPECTED_POSITIONS:
        state = overrides.get(position, FinalState.PASS)

        holes.append(
            HoleResult(
                position=position,
                predicted_state=VISUAL_BY_STATE[state],
                final_state=state,
                score=0.125,
                reason="test result",
                cap_fraction=0.8,
                cap_offset=1.25,
            )
        )

    return EventSnapshot(
        run="1",
        event="18",
        frame="event-018.png",
        fill_frontier=17,
        holes=tuple(holes),
    )


class FakeRepository:
    def __init__(self, snapshot, *, health=None):
        self.snapshot = snapshot
        self.health = health or SourceHealth(
            "LIVE",
            0.2,
            "scored",
            "",
        )
        self.png = b"\x89PNG\r\n\x1a\nservice-test"

    def latest_complete_event(self):
        return self.snapshot

    def source_health(self):
        return self.health

    def session_metadata(self):
        return {
            "session_id": "test-session",
            "source": "replay",
        }

    def overlay_png(self):
        return self.png


class ServiceTests(unittest.TestCase):
    def test_missed_and_unsure_are_counted_and_require_action(self):
        snapshot = make_snapshot(
            overrides={
                "G07": FinalState.MISSED,
                "G11": FinalState.UNSURE,
                "G12": FinalState.UNSURE,
            }
        )

        payload = SnapshotService(FakeRepository(snapshot)).snapshot_payload()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            payload["event"]["severity"],
            "ACTION_REQUIRED",
        )
        self.assertEqual(payload["counts"]["PASS"], 27)
        self.assertEqual(payload["counts"]["MISSED"], 1)
        self.assertEqual(payload["counts"]["UNSURE"], 2)
        self.assertEqual(len(payload["holes"]), 30)

        self.assertEqual(
            payload["holes"][6]["position"],
            "G07",
        )
        self.assertEqual(
            payload["holes"][6]["final_state"],
            "MISSED",
        )

    def test_unsure_without_defect_or_missed_requires_review(self):
        snapshot = make_snapshot(overrides={"G01": FinalState.UNSURE})

        payload = SnapshotService(FakeRepository(snapshot)).snapshot_payload()

        self.assertEqual(
            payload["event"]["severity"],
            "REVIEW_REQUIRED",
        )

    def test_all_pass_live_event_is_normal(self):
        payload = SnapshotService(FakeRepository(make_snapshot())).snapshot_payload()

        self.assertEqual(
            payload["event"]["severity"],
            "NORMAL",
        )
        self.assertEqual(payload["counts"]["PASS"], 30)

    def test_stale_source_does_not_hide_action_required_event(self):
        health = SourceHealth(
            "STALE",
            45.0,
            "scored",
            "checker output has not changed for 45.0 seconds",
        )

        payload = SnapshotService(
            FakeRepository(
                make_snapshot(
                    overrides={
                        "G07": FinalState.MISSED,
                    }
                ),
                health=health,
            )
        ).snapshot_payload()

        self.assertEqual(
            payload["event"]["severity"],
            "ACTION_REQUIRED",
        )
        self.assertEqual(
            payload["health"]["state"],
            "STALE",
        )

    def test_no_complete_event_returns_zero_counts(self):
        payload = SnapshotService(FakeRepository(None)).snapshot_payload()

        self.assertIsNone(payload["event"])
        self.assertEqual(payload["holes"], [])
        self.assertEqual(
            payload["counts"],
            {
                "PASS": 0,
                "DEFECT": 0,
                "MISSED": 0,
                "UNSURE": 0,
                "WAITING": 0,
            },
        )

    def test_session_and_overlay_are_passed_through(self):
        repository = FakeRepository(make_snapshot())
        service = SnapshotService(repository)

        self.assertEqual(
            service.session_payload()["source"],
            "replay",
        )
        self.assertEqual(
            service.overlay_png(),
            repository.png,
        )


if __name__ == "__main__":
    unittest.main()
