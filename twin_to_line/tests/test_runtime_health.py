"""Data-independent safety tests for live runtime health and availability."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import stat
import sys
import threading

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import live_inspect as L  # noqa: E402
import runtime_health as H  # noqa: E402


def record(health: H.RuntimeHealth, status: str, latency: float, number: int) -> None:
    health.record(
        status=status,
        timestamp_utc=f"2026-10-28T12:00:{number:02d}.000+00:00",
        latency_ms=latency,
        input_file=f"frame_{number}.png",
        event=number if status == "scored" else None,
    )


def test_health_is_fail_closed_until_scored_and_after_every_fault():
    health = H.RuntimeHealth()
    initial = health.snapshot()
    assert initial["state"] == H.UNAVAILABLE
    assert initial["reason_code"] == "no_successful_inspection_yet"
    assert initial["current_input_inspected"] is False
    assert initial["availability_scope"] == "processing_runtime_only"
    assert initial["source_liveness_guaranteed"] is False
    assert initial["operation"] == {
        "mode": "shadow_evaluation_only",
        "automatic_pass_authorized": False,
        "actuation_enabled": False,
        "automatic_release_enabled": False,
    }

    record(health, "settling", 1, 1)
    assert health.snapshot()["state"] == H.UNAVAILABLE
    assert health.snapshot()["reason_code"] == "input_not_scored_settling"

    record(health, "scored", 2, 2)
    assert health.snapshot()["state"] == H.AVAILABLE
    assert health.snapshot()["current_input_inspected"] is True

    record(health, "arm", 3, 3)
    assert health.snapshot()["state"] == H.UNAVAILABLE
    assert health.snapshot()["reason_code"] == "input_not_scored_arm"
    assert health.snapshot()["current_input_inspected"] is False

    record(health, "unreadable", 4, 4)
    record(health, "wrong_size", 5, 5)
    faulted = health.snapshot()
    assert faulted["state"] == H.UNAVAILABLE
    assert faulted["consecutive_faults"] == 2
    assert faulted["last_fault"]["status"] == "wrong_size"

    record(health, "source_recovered", 6, 6)
    recovered = health.snapshot()
    assert recovered["state"] == H.UNAVAILABLE
    assert recovered["reason_code"] == "source_recovered_awaiting_scored_event"
    assert recovered["consecutive_faults"] == 0

    record(health, "settling", 7, 7)
    assert health.snapshot()["state"] == H.UNAVAILABLE
    assert health.snapshot()["reason_code"] == "input_not_scored_settling"

    record(health, "scored", 8, 8)
    assert health.snapshot()["state"] == H.AVAILABLE
    assert health.snapshot()["last_successful_inspection"]["event"] == 8

    # An unrecognized status is a fault, never an optimistic/available state.
    record(health, "future_unhandled_status", 9, 9)
    assert health.snapshot()["state"] == H.UNAVAILABLE
    assert health.snapshot()["reason_code"] == "input_future_unhandled_status"


def test_metrics_name_their_denominators_and_report_tail_latency():
    health = H.RuntimeHealth()
    for number, (status, latency) in enumerate(
        [
            ("scored", 10),
            ("unreadable", 20),
            ("source_error", 30),
            ("source_recovered", 40),
            ("settling", 50),
            ("scored", 60),
        ],
        start=1,
    ):
        record(health, status, latency, number)

    metrics = health.snapshot()["metrics"]
    assert metrics == {
        "records_total": 6,
        "input_frames": 4,
        "valid_frames": 3,
        "scored_events": 2,
        "failed_or_refused_inputs": 1,
        "input_validity_coverage": 0.75,
        "scored_input_rate": 0.5,
        "all_record_latency_ms": {
            "count": 6,
            "min": 10.0,
            "p50": 35.0,
            "p95": 57.5,
            "p99": 59.5,
            "max": 60.0,
        },
        "scored_event_latency_ms": {
            "count": 2,
            "min": 10.0,
            "p50": 35.0,
            "p95": 57.5,
            "p99": 59.5,
            "max": 60.0,
        },
    }


def test_non_scored_records_cannot_hide_slow_scored_inference():
    health = H.RuntimeHealth()
    for number in range(1, 101):
        record(health, "settling", 1, number)
    record(health, "scored", 500, 101)

    metrics = health.snapshot()["metrics"]
    assert metrics["all_record_latency_ms"]["p50"] == 1.0
    assert metrics["scored_event_latency_ms"] == {
        "count": 1,
        "min": 500.0,
        "p50": 500.0,
        "p95": 500.0,
        "p99": 500.0,
        "max": 500.0,
    }


def test_zero_event_session_is_explicitly_unavailable(tmp_path):
    out = tmp_path / "zero-events"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    live = json.loads((out / L.RUNTIME_STATUS).read_text(encoding="utf-8"))
    assert live["state"] == H.UNAVAILABLE
    assert live["metrics"]["scored_events"] == 0
    assert live["metrics"]["input_validity_coverage"] is None

    session.close()
    ended = json.loads((out / L.RUNTIME_STATUS).read_text(encoding="utf-8"))
    summary = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert ended["state"] == H.UNAVAILABLE
    assert ended["reason_code"] == "session_ended"
    assert ended["last_successful_inspection"] is None
    assert ended["metrics"]["scored_events"] == 0
    assert summary["events"] == 0
    assert summary["runtime_health"] == ended
    assert summary["input_validity_coverage"] is None
    assert summary["scored_event_latency_ms_p50"] is None


def test_session_fault_recovery_requires_a_new_successful_event(tmp_path):
    out = tmp_path / "runtime"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)

    first = session.process_frame("frame_1.png", image, t=1.0)
    assert first["status"] == "scored"
    assert json.loads((out / L.RUNTIME_STATUS).read_text())["state"] == H.AVAILABLE

    bad = session.process_frame("frame_2.png", None, t=2.0)
    assert bad["status"] == "unreadable"
    assert json.loads((out / L.RUNTIME_STATUS).read_text())["state"] == H.UNAVAILABLE

    recovered = session.process_frame("camera-folder", None, t=3.0, source_error="")
    assert recovered["status"] == "source_recovered"
    status = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status["state"] == H.UNAVAILABLE
    assert status["current_input_inspected"] is False

    second = session.process_frame("frame_3.png", image, t=4.0)
    assert second["status"] == "scored"
    status = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status["state"] == H.AVAILABLE
    assert status["last_successful_inspection"]["event"] == 2
    assert status["operation"]["automatic_pass_authorized"] is False

    session.close()
    final = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert final["state"] == H.UNAVAILABLE
    assert final["reason_code"] == "session_ended"
    assert final["current_input_inspected"] is False


def test_session_counts_consecutive_faults_across_begin_input_updates(tmp_path):
    out = tmp_path / "consecutive-faults"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    assert session.process_frame("bad-1.png", None, t=1.0)["status"] == "unreadable"
    assert session.process_frame("bad-2.png", None, t=2.0)["status"] == "unreadable"
    status = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status["state"] == H.UNAVAILABLE
    assert status["consecutive_faults"] == 2
    assert status["last_fault"]["input_file"] == "bad-2.png"
    session.close()


def test_atomic_json_failure_preserves_previous_status(tmp_path, monkeypatch):
    destination = tmp_path / "runtime_status.json"
    destination.write_text('{"state":"old"}\n', encoding="utf-8")

    def fail(*args, **kwargs):
        args[1].write('{"state":"partial"')
        raise OSError("simulated full disk")

    monkeypatch.setattr(H.json, "dump", fail)
    with pytest.raises(OSError, match="full disk"):
        H.write_json_atomic(str(destination), {"state": "new"})

    assert destination.read_text(encoding="utf-8") == '{"state":"old"}\n'
    assert not list(tmp_path.glob(".runtime-json-*.tmp"))


def test_atomic_json_is_readable_and_preserves_existing_mode(tmp_path):
    destination = tmp_path / "status.json"
    H.write_json_atomic(str(destination), {"state": "new"})
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644

    destination.chmod(0o640)
    H.write_json_atomic(str(destination), {"state": "updated"})
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_runtime_status_write_failure_removes_stale_available_state(tmp_path, monkeypatch):
    out = tmp_path / "status-failure"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    assert session.process_frame("good.png", image, t=1.0)["status"] == "scored"
    assert json.loads((out / L.RUNTIME_STATUS).read_text())["state"] == H.AVAILABLE

    original = L.write_json_atomic

    def fail_status(path, value):
        if os.fspath(path) == session.health_path:
            raise OSError("simulated status persistence failure")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(L, "write_json_atomic", fail_status)
        with pytest.raises(OSError, match="status persistence failure"):
            session.process_frame("bad.png", None, t=2.0)
        assert not (out / L.RUNTIME_STATUS).exists()
        assert session.restart_required is True
        assert "runtime_status_persistence_failure" in session.fatal_error

    # Once persistence is restored, close publishes an explicit ended/unavailable state.
    session.close()
    final = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert final["state"] == H.UNAVAILABLE
    assert final["reason_code"] == "session_ended"
    with (out / "trial_log.csv").open(newline="") as stream:
        assert [row["status"] for row in csv.DictReader(stream)] == ["scored"]


def test_blocked_inference_is_unavailable_until_the_new_event_completes(tmp_path):
    started, release = threading.Event(), threading.Event()

    class BlockingInspector(L.RuleInspector):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def inspect_batch(self, crops):
            self.calls += 1
            if self.calls == 2:
                started.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release inference")
            return super().inspect_batch(crops)

    out = tmp_path / "blocked"
    session = L.Session(str(out), gate="each", ix=None, quiet=True, inspector=BlockingInspector())
    image = np.full((480, 640, 3), 90, np.uint8)
    assert session.process_frame("first.png", image, t=1.0)["status"] == "scored"

    result, failure = [], []

    def process():
        try:
            result.append(session.process_frame("second.png", image, t=2.0))
        except BaseException as error:  # surfaced in the main test thread
            failure.append(error)

    worker = threading.Thread(target=process)
    worker.start()
    assert started.wait(5)
    while_processing = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert while_processing["state"] == H.UNAVAILABLE
    assert while_processing["reason_code"] == "input_processing"
    assert while_processing["current_input_inspected"] is False
    assert while_processing["metrics"]["records_total"] == 1
    started_utc = while_processing["last_input"]["input_started_utc"]
    assert while_processing["status_updated_utc"] == started_utc

    release.set(); worker.join(5)
    assert not worker.is_alive()
    assert not failure
    assert result[0]["status"] == "scored"
    completed = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert completed["state"] == H.AVAILABLE
    assert completed["reason_code"] == "last_event_scored"
    assert completed["metrics"]["records_total"] == 2
    assert completed["last_input"]["input_started_utc"] == started_utc
    assert completed["last_input"]["completed_utc"] == completed["status_updated_utc"]
    session.close()


@pytest.mark.parametrize("failed_write", [1, 2])
def test_overlay_write_failure_never_scores_or_becomes_available(tmp_path, monkeypatch, failed_write):
    out = tmp_path / f"overlay-failure-{failed_write}"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    original = L.cv2.imwrite
    calls = 0

    def fail_one(path, value):
        nonlocal calls
        calls += 1
        return False if calls == failed_write else original(path, value)

    monkeypatch.setattr(L.cv2, "imwrite", fail_one)
    with pytest.raises(L.SessionRestartRequired, match="restart required"):
        session.process_frame("failure.png", image, t=1.0)
    status = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status["state"] == H.UNAVAILABLE
    assert "restart_required_after_score_failure" in status["last_fault"]["note"]
    assert not (out / "overlay_latest.png").exists()
    assert not list(out.glob(".runtime-image-*"))
    with (out / "predictions.csv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    if failed_write == 1:
        assert session.events == 0
        assert not list((out / "overlays").iterdir())
        assert predictions == []
    else:
        # Predictions and archival evidence committed before publication of
        # the convenience latest overlay. The ID is reserved and the partial
        # state is explicit/fatal rather than silently reused.
        assert session.events == 1
        assert len(list((out / "overlays").iterdir())) == 1
        assert len(predictions) == 30
        assert "predictions and archival overlay committed" in status["last_fault"]["note"]
    session.close()


def test_legacy_stale_overlay_temp_is_never_promoted_on_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "stale-overlay"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    assert session.process_frame("first.png", image, t=1.0)["status"] == "scored"
    latest = out / "overlay_latest.png"
    previous = latest.read_bytes()
    legacy_stale = out / ".overlay_latest.png"
    legacy_stale.write_bytes(b"stale fixed-temp bytes")

    original = L.cv2.imwrite
    calls = 0

    def fail_latest(path, value):
        nonlocal calls
        calls += 1
        return False if calls == 2 else original(path, value)

    monkeypatch.setattr(L.cv2, "imwrite", fail_latest)
    with pytest.raises(L.SessionRestartRequired, match="restart required"):
        session.process_frame("second.png", image, t=2.0)
    assert session.events == 2
    assert latest.read_bytes() == previous
    assert legacy_stale.read_bytes() == b"stale fixed-temp bytes"
    assert not list(out.glob(".runtime-image-*"))
    status = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status["state"] == H.UNAVAILABLE
    assert "predictions and archival overlay committed" in status["last_fault"]["note"]
    session.close()


def test_prediction_append_failure_preserves_prior_latest_and_removes_archive(tmp_path, monkeypatch):
    out = tmp_path / "prediction-failure"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    assert session.process_frame("first.png", image, t=1.0)["status"] == "scored"
    previous = (out / "overlay_latest.png").read_bytes()

    monkeypatch.setattr(session.preds, "write", lambda rows: (_ for _ in ()).throw(OSError("predictions full")))
    with pytest.raises(L.SessionRestartRequired, match="restart required"):
        session.process_frame("second.png", image, t=2.0)

    assert session.events == 1
    assert (out / "overlay_latest.png").read_bytes() == previous
    assert not list((out / "overlays").glob("e02_*"))
    status = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status["state"] == H.UNAVAILABLE
    assert "predictions full" in status["last_fault"]["note"]
    session.close()


def test_stateful_event_score_failure_latches_restart_and_refuses_future_inputs(tmp_path, monkeypatch):
    out = tmp_path / "fatal-event"
    session = L.Session(str(out), gate="events", track=True, ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    assert session.process_frame("a.png", image, t=0.0)["status"] == "settling"
    assert session.process_frame("b.png", image, t=2.0)["status"] == "settling"
    monkeypatch.setattr(L, "write_image_atomic", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(L.SessionRestartRequired, match="restart required"):
        session.process_frame("c.png", image, t=4.0)
    status_before = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status_before["state"] == H.UNAVAILABLE
    assert "restart_required_after_score_failure" in status_before["last_fault"]["note"]

    with pytest.raises(L.SessionRestartRequired, match="prior state/evidence failure"):
        session.process_frame("d.png", image, t=6.0)
    status_after = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status_after == status_before
    with (out / "trial_log.csv").open(newline="") as stream:
        assert [row["status"] for row in csv.DictReader(stream)] == ["settling", "settling", "error"]
    session.close()


def test_run_records_fatal_state_evidence_stop_reason(tmp_path, monkeypatch):
    out = tmp_path / "fatal-run"
    session = L.Session(str(out), gate="events", track=True, ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    frames = iter([("a.png", image, 0.0), ("b.png", image, 2.0), ("c.png", image, 4.0)])
    monkeypatch.setattr(L, "write_image_atomic", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(L.SessionRestartRequired, match="restart required"):
        L.run(session, frames)
    info = json.loads((out / "session.json").read_text())
    assert info["restart_required"] is True
    assert info["stop_reason"].startswith("fatal state/evidence failure; restart required:")
    assert info["runtime_health"]["state"] == H.UNAVAILABLE


def test_run_does_not_mislabel_status_persistence_failure_as_source_error(tmp_path, monkeypatch):
    out = tmp_path / "fatal-status-run"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    real_write = L.write_json_atomic

    def fail_status_only(path, value):
        if Path(path).name == L.RUNTIME_STATUS:
            raise OSError("runtime status disk failure")
        return real_write(path, value)

    monkeypatch.setattr(L, "write_json_atomic", fail_status_only)
    with pytest.raises(L.SessionRestartRequired, match="close evidence failed"):
        L.run(session, iter([("frame.png", image, 1.0)]))

    summary = json.loads((out / "session.json").read_text())
    assert summary["stop_reason"].startswith(
        "fatal state/evidence failure; restart required:"
    )
    assert "source error" not in summary["stop_reason"]
    assert summary["restart_required"] is True


def test_gate_changed_exception_after_decided_mutation_latches_restart(tmp_path, monkeypatch):
    out = tmp_path / "fatal-gate"
    session = L.Session(str(out), gate="events", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    assert session.process_frame("a.png", image, t=0.0)["status"] == "settling"
    assert session.process_frame("b.png", image, t=2.0)["status"] == "settling"
    monkeypatch.setattr(
        session.gate,
        "_changed",
        lambda image: (_ for _ in ()).throw(RuntimeError("changed calculation failed")),
    )
    with pytest.raises(L.SessionRestartRequired, match="restart required"):
        session.process_frame("c.png", image, t=4.0)
    status = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status["state"] == H.UNAVAILABLE
    assert "restart_required_after_gate_failure" in status["last_fault"]["note"]
    assert session.gate.decided is True
    with pytest.raises(L.SessionRestartRequired, match="prior state/evidence failure"):
        session.process_frame("later.png", image, t=2.0)
    session.close()


def test_trial_log_failure_after_score_commit_latches_restart(tmp_path, monkeypatch):
    out = tmp_path / "fatal-trial-log"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    assert session.process_frame("first.png", image, t=1.0)["status"] == "scored"

    monkeypatch.setattr(session.trial, "write", lambda rows: (_ for _ in ()).throw(OSError("trial disk full")))
    with pytest.raises(L.SessionRestartRequired, match="trial-log evidence failed"):
        session.process_frame("second.png", image, t=2.0)

    assert session.events == 2
    status = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status["state"] == H.UNAVAILABLE
    assert "restart_required_after_trial_log_failure" in status["last_fault"]["note"]
    with (out / "predictions.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 60
    with pytest.raises(L.SessionRestartRequired, match="prior state/evidence failure"):
        session.process_frame("later.png", image, t=3.0)
    session.close()


def test_partial_prediction_append_is_rolled_back_before_fatal_stop(tmp_path, monkeypatch):
    out = tmp_path / "partial-predictions"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    writerow = session.preds.w.writerow

    def write_one_then_fail(rows):
        rows = list(rows)
        writerow(rows[0])
        raise OSError("partial prediction append")

    monkeypatch.setattr(session.preds.w, "writerows", write_one_then_fail)
    with pytest.raises(L.SessionRestartRequired, match="restart required"):
        session.process_frame("partial.png", image, t=1.0)

    with (out / "predictions.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []
    assert not list((out / "overlays").iterdir())
    status = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status["state"] == H.UNAVAILABLE
    assert "partial prediction append" in status["last_fault"]["note"]
    session.close()


def test_prediction_rollback_failure_is_explicit_and_fatal(tmp_path, monkeypatch):
    out = tmp_path / "rollback-failure"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    real_file = session.preds.f
    writerow = session.preds.w.writerow

    class FailingTruncateProxy:
        def __getattr__(self, name):
            return getattr(real_file, name)

        def truncate(self, *args, **kwargs):
            raise OSError("rollback truncate failed")

    def write_one_then_fail(rows):
        rows = list(rows)
        writerow(rows[0])
        raise OSError("append failed")

    session.preds.f = FailingTruncateProxy()
    monkeypatch.setattr(session.preds.w, "writerows", write_one_then_fail)
    with pytest.raises(L.SessionRestartRequired, match="restart required"):
        session.process_frame("rollback.png", image, t=1.0)

    status = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert status["state"] == H.UNAVAILABLE
    assert "CSV append failed" in status["last_fault"]["note"]
    assert "rollback failed" in status["last_fault"]["note"]
    assert session.restart_required is True
    session.close()


def test_close_failure_still_ends_health_closes_other_log_and_attempts_summary(tmp_path, monkeypatch):
    out = tmp_path / "close-failure"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    assert session.process_frame("scored.png", image, t=1.0)["status"] == "scored"
    assert json.loads((out / L.RUNTIME_STATUS).read_text())["state"] == H.AVAILABLE

    predictions_close = session.preds.close
    trial_close = session.trial.close
    trial_attempted = []

    def fail_after_predictions_close():
        predictions_close()
        raise OSError("predictions close failed")

    def record_trial_close():
        trial_attempted.append(True)
        trial_close()

    monkeypatch.setattr(session.preds, "close", fail_after_predictions_close)
    monkeypatch.setattr(session.trial, "close", record_trial_close)
    with pytest.raises(L.SessionRestartRequired, match="close evidence failed"):
        session.close()

    assert trial_attempted == [True]
    ended = json.loads((out / L.RUNTIME_STATUS).read_text())
    summary = json.loads((out / "session.json").read_text())
    assert ended["state"] == H.UNAVAILABLE
    assert ended["reason_code"] == "session_ended"
    assert "fatal evidence close failure" in ended["stop_reason"]
    assert summary["runtime_health"] == ended
    assert summary["restart_required"] is True
    assert any("predictions_close_failure" in item for item in summary["close_errors"])


def test_blocked_close_publishes_unavailable_before_waiting(tmp_path, monkeypatch):
    out = tmp_path / "blocked-close"
    session = L.Session(str(out), gate="each", ix=None, quiet=True)
    image = np.full((480, 640, 3), 90, np.uint8)
    assert session.process_frame("scored.png", image, t=1.0)["status"] == "scored"
    assert json.loads((out / L.RUNTIME_STATUS).read_text())["state"] == H.AVAILABLE

    real_close = session.preds.close
    close_started = threading.Event()
    allow_close = threading.Event()
    failures = []

    def blocking_close():
        close_started.set()
        assert allow_close.wait(timeout=5)
        real_close()

    def close_session():
        try:
            session.close()
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)

    monkeypatch.setattr(session.preds, "close", blocking_close)
    worker = threading.Thread(target=close_session)
    worker.start()
    assert close_started.wait(timeout=5)
    closing = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert closing["state"] == H.UNAVAILABLE
    assert closing["reason_code"] == "session_closing"
    assert closing["current_input_inspected"] is False

    allow_close.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert failures == []
    ended = json.loads((out / L.RUNTIME_STATUS).read_text())
    assert ended["state"] == H.UNAVAILABLE
    assert ended["reason_code"] == "session_ended"


def test_browser_copy_never_presents_the_last_overlay_as_a_current_pass():
    assert "last successful event overlay only" in L.VIEW_PAGE
    assert "not a current-input PASS or release authorization" in L.VIEW_PAGE
    assert "runtime_status.json" in L.VIEW_PAGE


def test_offline_staging5_builder_includes_the_runtime_health_dependency():
    from gb10 import build_staging5_kit

    assert "runtime_health.py" in build_staging5_kit.APP_FILES


def test_runtime_provenance_includes_health_and_decision_policy_sources():
    assert "runtime_health.py" in L.RUNTIME_FILES
    assert "staging5_decision_policy.py" in L.RUNTIME_FILES
