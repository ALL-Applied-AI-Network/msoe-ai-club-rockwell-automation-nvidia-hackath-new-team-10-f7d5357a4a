"""Contract tests for the read-only CSI Excel sidecar."""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csi_excel_report as report


PLAN = Path(report.__file__).with_name("tuesday_staging_plan.csv")


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ExcelReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "ttl"
        self.visit = self.root / "visit_20261027T120000Z"
        (self.visit / "records").mkdir(parents=True)
        (self.visit / "staged").mkdir()
        (self.visit / "attempts" / "live_001").mkdir(parents=True)

        self.rows = [
            {
                "setup": "00", "status": "staged", "source_name": "camera-a.jpg",
                "copy_name": "setup_00.jpg", "time": "2026-10-27T17:00:00+00:00",
                "sha256": "a" * 64,
            },
            {
                "setup": "00", "status": "retake", "source_name": "camera-b.jpg",
                "copy_name": "setup_00_r1.jpg", "time": "2026-10-27T17:00:10+00:00",
                "sha256": "b" * 64,
            },
            {
                "setup": "01", "status": "skipped", "source_name": "",
                "copy_name": "", "time": "2026-10-27T17:00:20+00:00", "sha256": "",
            },
        ]
        write_csv(
            self.visit / "records" / "staging_log.csv",
            ["setup", "status", "source_name", "copy_name", "time", "sha256"],
            self.rows,
        )
        (self.visit / "staged" / "setup_00.jpg").write_bytes(b"old-image")
        (self.visit / "staged" / "setup_00_r1.jpg").write_bytes(b"final-image")
        write_csv(
            self.visit / "records" / "staging_notes.csv",
            ["setup", "note"],
            [{"setup": "00", "note": "=HYPERLINK(\"bad\",\"text\")\x01"}],
        )
        (self.visit / "records" / "seal_selection.json").write_text(
            json.dumps({"final": [self.rows[1]], "earlier_attempts": [self.rows[0]], "skipped": 1}),
            encoding="utf-8",
        )
        seal = self.visit / "seal_test" / "evidence.tar"
        seal.parent.mkdir()
        seal.write_bytes(b"verified-by-frozen-kit")
        (self.visit / "operator.env").write_text(
            "\n".join([
                "READY=1",
                "PKG=ttl_gb10_4c24ec2",
                f"DAY={self.visit}",
                f"STAGED={self.visit / 'staged'}",
                f"SEALED={seal}",
                f"SEALED_SHA256={'c' * 64}",
                "SEAL_STATUS='SEALED USB OK: /isolated/usb/sealed_test; reset fixture before live'",
                "",
            ]),
            encoding="utf-8",
        )

        live = self.visit / "attempts" / "live_001"
        (live / "session.json").write_text(json.dumps({
            "session": "live_001",
            "started_utc": "2026-10-27T17:01:00+00:00",
            "ended_utc": "2026-10-27T17:01:05.500+00:00",
            "source": "rtsp://operator:secret@camera/path?token=also-secret&mode=live",
            "gate": "events", "mode": "live_fill", "git": "4c24ec2",
            "stop_reason": "ctrl-c", "frames": 2, "events": 1,
            "status_counts": {"unchanged": 1, "scored": 1}, "latency_ms_median": 4.5,
        }), encoding="utf-8")
        write_csv(
            live / "trial_log.csv",
            ["time_utc", "input_file", "frame_t", "status", "arm_px", "event",
             "event_frame", "event_delay_s", "K", "n_flagged", "latency_ms", "path",
             "note", "intervention"],
            [
                {"time_utc": "2026-10-27T17:01:00+00:00", "input_file": "f0.png",
                 "frame_t": 0, "status": "unchanged", "arm_px": 0},
                {"time_utc": "2026-10-27T17:01:05+00:00", "input_file": "f1.png",
                 "frame_t": 5, "status": "scored", "arm_px": 0, "event": 1,
                 "event_frame": "f1.png", "event_delay_s": 0.25, "K": 1,
                 "n_flagged": 0, "latency_ms": 4.5},
            ],
        )
        prediction_rows = []
        for index in range(1, 31):
            prediction_rows.append({
                "run": 1, "event": 1, "frame": "f1.png", "position": f"G{index:02d}",
                "predicted_state": "pass", "defect_score": 0.1, "pred_sequence": "",
                "final_state": "PASS", "K": 1, "K_source": "occupancy", "cap_frac": 0.9,
                "cap_offset": 0.1, "reason": "test",
            })
        write_csv(live / "predictions.csv", report.PREDICTION_FIELDS, prediction_rows)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def by_name(sheets):
        return {sheet.name: sheet for sheet in sheets}

    def test_collects_plan_timing_and_never_reads_images_or_predictions_by_default(self):
        original_read_bytes = Path.read_bytes

        def guarded_read_bytes(path):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                raise AssertionError(f"image was read: {path}")
            if path.name == "predictions.csv":
                raise AssertionError("predictions were read without opt-in")
            return original_read_bytes(path)

        with mock.patch.object(Path, "read_bytes", guarded_read_bytes):
            sheets, metadata = report.collect_report(
                self.root, PLAN, generated_utc="2026-10-27T17:02:00+00:00"
            )
        tables = self.by_name(sheets)
        self.assertNotIn("Position Results", tables)
        self.assertEqual(len(tables["Setup Checklist"].rows), 22)
        self.assertEqual(len(tables["Predeclared Ground Truth"].rows), 91)
        self.assertEqual(
            [row["interval_since_previous_record_s"] for row in tables["Staging Attempts"].rows],
            [None, 10.0, 10.0],
        )
        self.assertEqual(tables["Summary"].rows[0]["total_live_duration_s"], 5.5)
        self.assertEqual(tables["Summary"].rows[0]["state"], "COLLECTED")
        self.assertTrue(tables["Summary"].rows[0]["usb_checkpoint_ok"])
        self.assertEqual(metadata["errors"], 0)
        self.assertEqual(tables["Live Runs"].rows[0]["source"],
                         "rtsp://***@camera/path?token=%2A%2A%2A&mode=live")

    def test_predictions_are_explicit_opt_in(self):
        sheets, metadata = report.collect_report(self.root, PLAN, include_predictions=True)
        tables = self.by_name(sheets)
        self.assertTrue(metadata["include_predictions"])
        self.assertEqual(len(tables["Position Results"].rows), 30)
        self.assertEqual(tables["Live Events"].rows[0]["PASS"], 30)

    def test_malformed_seal_selection_is_not_treated_as_authoritative(self):
        broken = dict(self.rows[1])
        broken["sha256"] = "d" * 64
        (self.visit / "records" / "seal_selection.json").write_text(
            json.dumps({"final": [broken], "earlier_attempts": [self.rows[0]]}),
            encoding="utf-8",
        )
        sheets, metadata = report.collect_report(self.root, PLAN)
        tables = self.by_name(sheets)
        self.assertGreater(metadata["errors"], 0)
        self.assertEqual(tables["Summary"].rows[0]["state"], "SEAL INVALID / REVIEW")
        self.assertEqual(tables["Setup Checklist"].rows[0]["capture_status"], "not recorded")

    def test_xlsx_is_deterministic_valid_and_formula_safe(self):
        generated = "2026-10-27T17:02:00+00:00"
        sheets, _ = report.collect_report(self.root, PLAN, generated_utc=generated)
        first = report.write_xlsx(self.base / "first.xlsx", sheets, generated)
        second = report.write_xlsx(self.base / "second.xlsx", sheets, generated)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        report.validate_xlsx(first, len(sheets))
        with zipfile.ZipFile(first) as archive:
            xml = b"\n".join(
                archive.read(name) for name in archive.namelist()
                if name.startswith("xl/worksheets/")
            )
        self.assertIn(b"=HYPERLINK", xml)
        self.assertNotIn(b"<f>", xml)
        self.assertNotIn(b"\x01", xml)

    def test_cli_refuses_output_inside_evidence(self):
        with self.assertRaises(SystemExit) as caught:
            report.main([
                "--ttl-root", str(self.root),
                "--output", str(self.root / "report.xlsx"),
            ])
        self.assertEqual(caught.exception.code, 2)

    def test_guide_sidecar_hashes_match_files(self):
        guide = Path(report.__file__).with_name("CSI_TUESDAY_AGENT_GUIDE.md").read_text(
            encoding="utf-8"
        )
        for path in (Path(report.__file__), PLAN):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertIn(digest, guide)


if __name__ == "__main__":
    unittest.main()
