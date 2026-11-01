"""Metadata-only gates for a future policy-v2 YAAD training candidate."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


TTL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TTL))

import prepare_yaad_candidate_v2 as C  # noqa: E402
import staging_dataset as D  # noqa: E402


GATE_TIME = datetime(2026, 11, 1, 13, 0, tzinfo=timezone.utc)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_rows(
    *, split: str, session: str, event_prefix: str
) -> list[dict[str, str]]:
    classes = tuple(D.CANONICAL_CLASSES)
    rows: list[dict[str, str]] = []
    # Five events provide independent event support for all classes.  Each
    # event is still exactly one complete 30-position fixture.
    for event_index in range(5):
        event_uid = f"{event_prefix}-event-{event_index}"
        source_id = f"{event_prefix}-source-{event_index}"
        digest = hashlib.sha256(source_id.encode()).hexdigest()
        for position in range(1, 31):
            visual = classes[(position - 1) % len(classes)]
            occupancy = "empty" if visual == "no_vial" and position % 2 else "occupied"
            rows.append(
                {
                    "sample_id": f"{event_uid}-G{position:02d}",
                    "event_uid": event_uid,
                    "session_id": session,
                    "event_id": f"event-{event_index}",
                    "source_image_id": source_id,
                    "source_image_relpath": f"development/{source_id}.jpg",
                    "source_image_sha256": digest,
                    "capture_time": "2026-11-01T10:10:00Z",
                    "fixture_id": "G",
                    "position_id": f"G{position:02d}",
                    "expected_occupancy": occupancy,
                    "visual_class": visual,
                    "inspection_mode": "fixture",
                    "gt_sequence": (
                        "waiting" if visual == "no_vial" and occupancy == "empty"
                        else "missed" if visual == "no_vial"
                        else ""
                    ),
                    "designation": "development",
                    "usable": "true",
                    "split": split,
                    "split_reason": "separate development visit",
                    "label_source": "adjudicated_visible_content",
                    "annotator_id": "reviewer",
                    "annotation_confidence": "1.0",
                    "review_state": "adjudicated",
                    "notes": "synthetic test metadata",
                }
            )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=D.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _counts(path: Path) -> dict[str, object]:
    rows = D.read_manifest(path, verify_images=False)
    return C._counts(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _refresh_manifest_lock(lock: Path, manifest: Path, split: str) -> None:
    value = json.loads(lock.read_text(encoding="utf-8"))
    value["development_manifests"][split] = {
        "sha256": _sha(manifest),
        "counts": _counts(manifest),
    }
    lock.write_text(json.dumps(value), encoding="utf-8")


def _refresh_manifest_hash_only(lock: Path, manifest: Path, split: str) -> None:
    value = json.loads(lock.read_text(encoding="utf-8"))
    value["development_manifests"][split]["sha256"] = _sha(manifest)
    lock.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    train = tmp_path / "train_manifest.csv"
    validation = tmp_path / "validation_manifest.csv"
    _write_manifest(
        train,
        _manifest_rows(split="train", session="train-visit", event_prefix="train"),
    )
    _write_manifest(
        validation,
        _manifest_rows(
            split="validation",
            session="validation-visit",
            event_prefix="validation",
        ),
    )
    start = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    digest = "a" * 64
    lock = {
        "schema_version": 1,
        "candidate_id": "staging5-policy-v2-candidate-r1",
        "status": "development_inputs_locked",
        "canonical_classes": list(D.CANONICAL_CLASSES),
        "development_start_not_before_utc": start.isoformat().replace("+00:00", "Z"),
        "authorization": {
            "record_sha256": "b" * 64,
            "authorized_at_utc": (start - timedelta(hours=3)).isoformat().replace("+00:00", "Z"),
            "staging_training_authorized": True,
            "allowed_uses": ["baseline_scoring", "train", "validation"],
        },
        "synthetic_baseline": {
            "baseline_id": "synthetic-baseline-r1",
            "freeze_sha256": digest,
            "frozen_at_utc": (start - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
            "score_receipt_sha256": "c" * 64,
            "predictions_sha256": "d" * 64,
            "prediction_rows": 300,
            "scoring_completed_at_utc": (start - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "scored_before_real_adaptation": True,
            "decision_policy_version": "staging5_score_to_visual_v2",
            "decision_policy_sha256": "e" * 64,
        },
        "development_manifests": {
            "train": {"sha256": _sha(train), "counts": _counts(train)},
            "validation": {
                "sha256": _sha(validation),
                "counts": _counts(validation),
            },
        },
        "fixture_map_sha256": {"G": "f" * 64},
        "protected_reservation": {
            "lock_sha256": "1" * 64,
            "sealed_before_development": True,
            "different_session_visit_verified": True,
            "event_session_source_and_path_identifiers_included": False,
            "labels_included": False,
        },
        "policy": C.EXPECTED_POLICY,
    }
    lock_path = tmp_path / "development_lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8")
    return lock_path, train, validation


def _authenticate(lock: Path, train: Path, validation: Path):
    return C.authenticate_candidate_inputs(
        development_lock_path=lock,
        train_manifest_path=train,
        validation_manifest_path=validation,
        candidate_id="staging5-policy-v2-candidate-r1",
        seed=7,
        now_utc=lambda: GATE_TIME,
    )


def test_valid_lock_emits_metadata_only_event_balanced_receipt(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    receipt = _authenticate(lock, train, validation)

    assert receipt["metadata_gate_passed"] is True
    assert receipt["ready_for_development_byte_verification"] is True
    assert "ready_for_future_bundle_and_trainer" not in receipt
    assert "development image bytes" in receipt["next_required_gate"]
    assert "fixture-map" in receipt["next_required_gate"]
    assert receipt["contract_version"] == C.CONTRACT_VERSION
    assert receipt["canonical_classes"] == list(D.CANONICAL_CLASSES)
    assert receipt["protected_identifiers_loaded"] is False
    assert receipt["protected_labels_loaded"] is False
    assert receipt["protected_pixels_loaded"] is False
    assert receipt["development_image_pixels_loaded"] is False
    assert receipt["fixture_map_sha256"] == {"G": "f" * 64}
    assert receipt["manifests"]["train_class_counts"] == {
        name: 30 for name in D.CANONICAL_CLASSES
    }
    assert receipt["manifests"]["validation_class_counts"] == {
        name: 30 for name in D.CANONICAL_CLASSES
    }
    assert receipt["schedule"]["version"] == C.SCHEDULE_VERSION
    assert receipt["schedule"]["class_counts"] == {
        name: 5 for name in D.CANONICAL_CLASSES
    }
    assert receipt["augmentation"]["identity_only"] is True
    assert receipt["unsupported_classes"] == []


def test_legacy_split_lock_is_rejected_before_manifests_are_opened(
    tmp_path, monkeypatch
):
    legacy = tmp_path / "split_lock.json"
    legacy.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "canonical_classes": list(D.CANONICAL_CLASSES),
                "events": {"test": ["private-event"]},
            }
        ),
        encoding="utf-8",
    )
    original_read = C._read_regular_bytes

    def guarded_read(path, field, *, max_bytes):
        if field != "development_lock":
            pytest.fail("manifest opened after legacy lock")
        return original_read(path, field, max_bytes=max_bytes)

    monkeypatch.setattr(C, "_read_regular_bytes", guarded_read)
    with pytest.raises(D.ManifestError, match="unsafe or unsupported shape"):
        C.authenticate_candidate_inputs(
            development_lock_path=legacy,
            train_manifest_path=tmp_path / "missing-train.csv",
            validation_manifest_path=tmp_path / "missing-validation.csv",
        )


def test_manifest_hash_mismatch_is_rejected_before_manifest_parse(
    tmp_path, monkeypatch
):
    lock, train, validation = _fixture(tmp_path)
    with train.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    monkeypatch.setattr(
        C,
        "_parse_manifest_bytes",
        lambda *args, **kwargs: pytest.fail("manifest parsed after hash mismatch"),
    )
    with pytest.raises(D.ManifestError, match="SHA-256 differs"):
        _authenticate(lock, train, validation)


def test_manifest_row_with_surplus_private_cell_is_rejected(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    lines = train.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1] = lines[1].rstrip("\r\n") + ",PROTECTED-EVENT-SECRET\n"
    train.write_text("".join(lines), encoding="utf-8")
    _refresh_manifest_hash_only(lock, train, "train")

    with pytest.raises(D.ManifestError, match="wrong number of CSV fields"):
        _authenticate(lock, train, validation)


def test_manifest_hash_parse_and_receipt_use_one_immutable_snapshot(
    tmp_path, monkeypatch
):
    lock, train, validation = _fixture(tmp_path)
    original_train = train.read_bytes()
    original_read = C._read_regular_bytes
    swapped = False

    def swap_after_snapshot(path, field, *, max_bytes):
        nonlocal swapped
        snapshot = original_read(path, field, max_bytes=max_bytes)
        if Path(path) == train and not swapped:
            swapped = True
            train.write_bytes(b"replacement bytes that are not a manifest\n")
        return snapshot

    monkeypatch.setattr(C, "_read_regular_bytes", swap_after_snapshot)
    receipt = _authenticate(lock, train, validation)

    assert swapped is True
    assert receipt["metadata_gate_passed"] is True
    assert receipt["manifests"]["train_sha256"] == hashlib.sha256(
        original_train
    ).hexdigest()
    assert receipt["manifests"]["train_sha256"] != _sha(train)


def test_lock_parse_and_receipt_use_one_immutable_snapshot(tmp_path, monkeypatch):
    lock, train, validation = _fixture(tmp_path)
    original_lock = lock.read_bytes()
    original_read = C._read_regular_bytes
    swapped = False

    def swap_lock_after_snapshot(path, field, *, max_bytes):
        nonlocal swapped
        snapshot = original_read(path, field, max_bytes=max_bytes)
        if field == "development_lock" and not swapped:
            swapped = True
            lock.write_text('{"replacement": true}\n', encoding="utf-8")
        return snapshot

    monkeypatch.setattr(C, "_read_regular_bytes", swap_lock_after_snapshot)
    receipt = _authenticate(lock, train, validation)

    assert swapped is True
    assert receipt["metadata_gate_passed"] is True
    assert receipt["development_lock_sha256"] == hashlib.sha256(
        original_lock
    ).hexdigest()
    assert receipt["development_lock_sha256"] != _sha(lock)


def test_symlinked_metadata_is_rejected(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    real_train = tmp_path / "real_train.csv"
    train.rename(real_train)
    train.symlink_to(real_train.name)
    with pytest.raises(D.ManifestError, match="regular non-symlink"):
        _authenticate(lock, train, validation)


def test_development_start_time_is_enforced_with_injected_clock(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    before_start = datetime(2026, 11, 1, 11, 59, 59, tzinfo=timezone.utc)
    with pytest.raises(D.ManifestError, match="cannot start before"):
        C.authenticate_candidate_inputs(
            development_lock_path=lock,
            train_manifest_path=train,
            validation_manifest_path=validation,
            now_utc=lambda: before_start,
        )

    at_start = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    receipt = C.authenticate_candidate_inputs(
        development_lock_path=lock,
        train_manifest_path=train,
        validation_manifest_path=validation,
        now_utc=lambda: at_start,
    )
    assert receipt["metadata_gate_passed"] is True


def test_injected_clock_must_be_timezone_aware(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    with pytest.raises(D.ManifestError, match="timezone-aware"):
        C.authenticate_candidate_inputs(
            development_lock_path=lock,
            train_manifest_path=train,
            validation_manifest_path=validation,
            now_utc=lambda: datetime(2026, 11, 1, 13, 0),
        )


def test_lock_cannot_smuggle_protected_identifiers_in_extra_fields(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    value = json.loads(lock.read_text(encoding="utf-8"))
    value["protected_event_ids"] = ["must-not-enter-training"]
    lock.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(D.ManifestError, match="extra=.*protected_event_ids"):
        _authenticate(lock, train, validation)


def test_lock_rejects_duplicate_json_keys_before_privacy_claim(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    original = lock.read_text(encoding="utf-8")
    assert original.startswith("{")
    lock.write_text(
        '{"policy":"PROTECTED-EVENT-SECRET",' + original[1:],
        encoding="utf-8",
    )
    with pytest.raises(D.ManifestError, match="duplicate JSON object key"):
        _authenticate(lock, train, validation)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_lock_rejects_nonfinite_json_constants(tmp_path, constant):
    lock, train, validation = _fixture(tmp_path)
    original = lock.read_text(encoding="utf-8")
    assert original.startswith("{")
    lock.write_text(
        f'{{"unexpected_numeric":{constant},' + original[1:],
        encoding="utf-8",
    )
    with pytest.raises(D.ManifestError, match="non-finite JSON numeric"):
        _authenticate(lock, train, validation)


def test_fifo_metadata_path_is_rejected_without_blocking(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is unavailable")
    fifo = tmp_path / "development_lock.json"
    os.mkfifo(fifo)
    with pytest.raises(D.ManifestError, match="regular file"):
        C.authenticate_candidate_inputs(
            development_lock_path=fifo,
            train_manifest_path=tmp_path / "train.csv",
            validation_manifest_path=tmp_path / "validation.csv",
            now_utc=lambda: GATE_TIME,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sealed_before_development", False, "sealed_before_development"),
        ("different_session_visit_verified", False, "different_session_visit"),
        ("event_session_source_and_path_identifiers_included", True, "identifiers_included"),
        ("labels_included", True, "labels_included"),
    ],
)
def test_private_reservation_boundary_must_be_attested(
    tmp_path, field, value, message
):
    lock, train, validation = _fixture(tmp_path)
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["protected_reservation"][field] = value
    lock.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(D.ManifestError, match=message):
        _authenticate(lock, train, validation)


def test_train_and_validation_must_be_different_visits(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    rows = _read_rows(validation)
    for row in rows:
        row["session_id"] = "train-visit"
        row["event_id"] = f"validation-{row['event_id']}"
    _write_manifest(validation, rows)
    _refresh_manifest_lock(lock, validation, "validation")
    with pytest.raises(D.ManifestError, match="session/visit leakage"):
        _authenticate(lock, train, validation)


def test_duplicate_sample_id_across_manifests_is_rejected(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    train_rows = _read_rows(train)
    validation_rows = _read_rows(validation)
    validation_rows[0]["sample_id"] = train_rows[0]["sample_id"]
    _write_manifest(validation, validation_rows)
    _refresh_manifest_hash_only(lock, validation, "validation")
    with pytest.raises(D.ManifestError, match="duplicate sample_id"):
        _authenticate(lock, train, validation)


def test_duplicate_digest_cannot_alias_a_different_event(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    train_rows = _read_rows(train)
    validation_rows = _read_rows(validation)
    duplicated_digest = train_rows[0]["source_image_sha256"]
    event_uid = validation_rows[0]["event_uid"]
    for row in validation_rows:
        if row["event_uid"] == event_uid:
            row["source_image_sha256"] = duplicated_digest
    _write_manifest(validation, validation_rows)
    _refresh_manifest_hash_only(lock, validation, "validation")
    with pytest.raises(D.ManifestError, match="duplicate source-image digest"):
        _authenticate(lock, train, validation)


def test_physical_session_event_cannot_alias_multiple_event_uids(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    train_rows = _read_rows(train)
    validation_rows = _read_rows(validation)
    target_uid = validation_rows[0]["event_uid"]
    for row in validation_rows:
        if row["event_uid"] == target_uid:
            row["session_id"] = train_rows[0]["session_id"]
            row["event_id"] = train_rows[0]["event_id"]
    _write_manifest(validation, validation_rows)
    _refresh_manifest_hash_only(lock, validation, "validation")
    with pytest.raises(D.ManifestError, match="aliases multiple event_uids"):
        _authenticate(lock, train, validation)


def test_event_uid_must_have_consistent_session_and_event_metadata(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    validation_rows = _read_rows(validation)
    validation_rows[0]["event_id"] = "inconsistent-event-id"
    _write_manifest(validation, validation_rows)
    _refresh_manifest_hash_only(lock, validation, "validation")
    with pytest.raises(D.ManifestError, match="aliases inconsistent session/event"):
        _authenticate(lock, train, validation)


def test_capture_time_must_be_timezone_aware_iso8601(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    validation_rows = _read_rows(validation)
    validation_rows[0]["capture_time"] = "not-a-timestamp"
    _write_manifest(validation, validation_rows)
    _refresh_manifest_hash_only(lock, validation, "validation")
    with pytest.raises(D.ManifestError, match="capture_time must be an ISO-8601"):
        _authenticate(lock, train, validation)


@pytest.mark.parametrize(
    "bad_time",
    ["2026-11-01T09:59:59Z", "2026-11-01T11:00:00Z"],
)
def test_capture_time_must_be_between_baseline_freeze_and_scoring(
    tmp_path, bad_time
):
    lock, train, validation = _fixture(tmp_path)
    validation_rows = _read_rows(validation)
    first_event = validation_rows[0]["event_uid"]
    for row in validation_rows:
        if row["event_uid"] == first_event:
            row["capture_time"] = bad_time
    _write_manifest(validation, validation_rows)
    _refresh_manifest_hash_only(lock, validation, "validation")
    with pytest.raises(D.ManifestError, match="freeze < capture <"):
        _authenticate(lock, train, validation)


def test_capture_time_is_consistent_within_source_and_event(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    validation_rows = _read_rows(validation)
    validation_rows[0]["capture_time"] = "2026-11-01T10:11:00Z"
    _write_manifest(validation, validation_rows)
    _refresh_manifest_hash_only(lock, validation, "validation")
    with pytest.raises(D.ManifestError, match="inconsistent capture_time"):
        _authenticate(lock, train, validation)


def test_authorization_uses_are_exactly_bounded(tmp_path):
    lock, train, validation = _fixture(tmp_path)
    value = json.loads(lock.read_text(encoding="utf-8"))
    value["authorization"]["allowed_uses"].append("protected_test_tuning")
    lock.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(D.ManifestError, match="limited to exactly"):
        _authenticate(lock, train, validation)


def test_cli_refuses_current_legacy_lock_without_touching_images(tmp_path, capsys):
    legacy = tmp_path / "split_lock.json"
    legacy.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    code = C.main(
        [
            "--development-lock",
            str(legacy),
            "--train-manifest",
            str(tmp_path / "train.csv"),
            "--validation-manifest",
            str(tmp_path / "validation.csv"),
        ]
    )
    assert code == 2
    assert "POLICY-V2 CANDIDATE REFUSED" in capsys.readouterr().err
