"""Contract tests for the read-only checker-output repository."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from operator_ui.models import EXPECTED_POSITIONS
from operator_ui.repository import CheckerOutputRepository


PREDICTION_FIELDS = [
    "run",
    "event",
    "frame",
    "position",
    "predicted_state",
    "defect_score",
    "pred_sequence",
    "final_state",
    "K",
    "cap_frac",
    "cap_offset",
    "reason",
]


def make_event_rows(event: int, *, overrides=None):
    overrides = overrides or {}
    rows = []

    for position in EXPECTED_POSITIONS:
        final_state = overrides.get(position, "PASS")
        predicted_state = {
            "PASS": "pass",
            "DEFECT": "crooked_cap",
            "MISSED": "no_vial",
            "UNSURE": "unknown",
            "WAITING": "no_vial",
        }[final_state]

        rows.append(
            {
                "run": "1",
                "event": str(event),
                "frame": f"event-{event:03d}.png",
                "position": position,
                "predicted_state": predicted_state,
                "defect_score": "0.125",
                "pred_sequence": "",
                "final_state": final_state,
                "K": "17",
                "cap_frac": "0.800",
                "cap_offset": "1.250",
                "reason": "test result",
            }
        )

    return rows


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary_directory.name)
        self.predictions_path = self.output_dir / "predictions.csv"

        (self.output_dir / "session.json").write_text(
            json.dumps({"session_id": "test-session"}),
            encoding="utf-8",
        )

        (self.output_dir / "trial_log.csv").write_text(
            "status\nscored\n",
            encoding="utf-8",
        )

        (self.output_dir / "overlay_latest.png").write_bytes(
            b"\x89PNG\r\n\x1a\nvalid-test-payload"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_rows(self, rows):
        with self.predictions_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=PREDICTION_FIELDS,
            )
            writer.writeheader()
            writer.writerows(rows)

    def append_rows(self, rows):
        with self.predictions_path.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=PREDICTION_FIELDS,
            )
            writer.writerows(rows)

    def test_complete_event_is_returned_in_twin_order(self):
        # The writer's row order must not control the fixture layout.
        self.write_rows(reversed(make_event_rows(1)))

        snapshot = CheckerOutputRepository(
            self.output_dir
        ).latest_complete_event()

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.event, "1")
        self.assertEqual(
            tuple(hole.position for hole in snapshot.holes),
            EXPECTED_POSITIONS,
        )

    def test_partial_append_keeps_previous_event_then_updates_at_row_30(self):
        self.write_rows(make_event_rows(1))
        repository = CheckerOutputRepository(self.output_dir)

        first = repository.latest_complete_event()
        self.assertEqual(first.event, "1")

        second_event = make_event_rows(
            2,
            overrides={"G07": "MISSED"},
        )

        # Simulate the checker while it is still writing event 2.
        self.append_rows(second_event[:29])

        while_incomplete = repository.latest_complete_event()

        # Operators must continue seeing the last complete event.
        self.assertEqual(while_incomplete.event, "1")

        # Complete event 2.
        self.append_rows(second_event[29:])

        after_completion = repository.latest_complete_event()

        self.assertEqual(after_completion.event, "2")
        self.assertEqual(after_completion.holes[6].position, "G07")
        self.assertEqual(
            after_completion.holes[6].final_state.value,
            "MISSED",
        )

    def test_invalid_newest_event_falls_back_to_previous_event(self):
        event_two = make_event_rows(2)
        event_two.append(dict(event_two[0]))  # Duplicate G01.

        self.write_rows(make_event_rows(1) + event_two)

        repository = CheckerOutputRepository(self.output_dir)
        snapshot = repository.latest_complete_event()

        self.assertEqual(snapshot.event, "1")
        self.assertIn(
            "duplicate position G01",
            repository.last_warning,
        )

    def test_invalid_overlay_does_not_replace_cached_good_overlay(self):
        repository = CheckerOutputRepository(self.output_dir)
        expected = repository.overlay_png()

        # Simulates reading while an unsafe/non-atomic writer is replacing it.
        (self.output_dir / "overlay_latest.png").write_bytes(
            b"not-a-png"
        )

        self.assertEqual(repository.overlay_png(), expected)
        self.assertIn(
            "not a valid PNG",
            repository.last_warning,
        )


if __name__ == "__main__":
    unittest.main()