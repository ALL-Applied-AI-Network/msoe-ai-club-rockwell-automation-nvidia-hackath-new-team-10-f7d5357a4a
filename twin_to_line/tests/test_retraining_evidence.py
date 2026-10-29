import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prepare_retraining_evidence as R
import staging_dataset as D


H = hashlib.sha256(b"artifact").hexdigest()


def _event_rows(
    session,
    event,
    split,
    designation,
    *,
    fixture="G",
    visual="pass",
    capture_time="2026-11-01T05:05:00-05:00",
):
    source = f"source-{session}-{event}-{fixture}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    rows = []
    for index in range(1, 31):
        target = visual if index == 1 else "pass"
        rows.append(
            D.ManifestRow(
                sample_id=f"{session}:{event}:{fixture}{index:02d}",
                event_uid=f"{session}:{event}",
                session_id=session,
                event_id=event,
                source_image_id=source,
                source_image_relpath=f"raw/{source}.png",
                source_image_sha256=digest,
                capture_time=capture_time,
                fixture_id=fixture,
                position_id=f"{fixture}{index:02d}",
                expected_occupancy="occupied",
                visual_class=target,
                inspection_mode="final_inspection",
                gt_sequence="",
                designation=designation,
                usable="true",
                split=split,
                split_reason="predeclared future-candidate partition",
                label_source="adjudicated staging review",
                annotator_id="reviewer-1",
                annotation_confidence="1.0",
                review_state="adjudicated",
                notes="",
            )
        )
    return rows


def _write_manifest(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=D.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(row.as_csv_dict() for row in rows)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_predictions(path, rows, *, timestamp="2026-11-01T10:15:00Z"):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=R.BASELINE_PREDICTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_image_id": row.source_image_id,
                    "fixture_id": row.fixture_id,
                    "position_id": row.position_id,
                    "expected_occupancy": row.expected_occupancy,
                    "predicted_visual_class": "pass",
                    "model_score": "0.10",
                    "applicable_threshold": "0.50",
                    "inspection_decision": "PASS",
                    "reason_code": "synthetic_baseline_below_threshold",
                    "inspection_timestamp_utc": timestamp,
                    "event_uid": row.event_uid,
                }
            )


def _refresh_prediction_receipt(args):
    receipt = json.loads(
        args["baseline_score_receipt_path"].read_text(encoding="utf-8")
    )
    receipt["predictions_sha256"] = D.sha256_file(args["baseline_predictions_path"])
    _write_json(args["baseline_score_receipt_path"], receipt)


def _change_first_prediction(args, **updates):
    path = args["baseline_predictions_path"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(updates)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=R.BASELINE_PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_prediction_receipt(args)


def _fixture(tmp_path, *, validation_fixture="G", maps=None, prediction_time=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    train = _event_rows("train-visit", "train-event", "train", "development")
    validation = _event_rows(
        "validation-visit",
        "validation-event",
        "validation",
        "development",
        fixture=validation_fixture,
        visual="crooked_cap",
    )
    protected = _event_rows(
        "protected-visit",
        "protected-event",
        "test",
        "protected_test",
        visual="missing_cap",
    )
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    protected_path = tmp_path / "protected.csv"
    _write_manifest(train_path, train)
    _write_manifest(validation_path, validation)
    _write_manifest(protected_path, protected)

    freeze_path = tmp_path / "synthetic_freeze.json"
    freeze = {
        "schema_version": 1,
        "baseline_id": "synthetic-v1",
        "training_data_scope": "synthetic_only",
        "frozen_at_utc": "2026-11-01T10:00:00Z",
        "canonical_classes": list(D.CANONICAL_CLASSES),
        "model_sha256": H,
        "thresholds_sha256": H,
        "decision_policy_sha256": H,
        "decision_policy_version": "synthetic-score-policy-v1",
        "preprocessing_sha256": {"to_tensor_array": H, "frame_warp": H},
        "code_sha256": {"inference": H},
        "fixture_map_sha256": maps or {"G": H},
    }
    _write_json(freeze_path, freeze)

    authorization_path = tmp_path / "authorization.json"
    _write_json(
        authorization_path,
        {
            "schema_version": 1,
            "staging_training_authorized": True,
            "authorized_by": "project owner",
            "authorized_at_utc": "2026-11-01T08:00:00Z",
            "dataset_scope": "new CSI staging collection",
            "allowed_uses": ["baseline_scoring", "train", "validation"],
            "authorization_statement": (
                "Frozen synthetic baseline scores first; staging development data may "
                "then train and validate a separately versioned real-adapted candidate."
            ),
        },
    )

    predictions_path = tmp_path / "baseline_predictions.csv"
    _write_predictions(
        predictions_path,
        train + validation,
        timestamp=prediction_time or "2026-11-01T10:15:00Z",
    )
    receipt_path = tmp_path / "baseline_score_receipt.json"
    _write_json(
        receipt_path,
        {
            "schema_version": 1,
            "baseline_id": "synthetic-v1",
            "baseline_freeze_sha256": D.sha256_file(freeze_path),
            "predictions_sha256": D.sha256_file(predictions_path),
            "decision_policy_sha256": H,
            "decision_policy_version": "synthetic-score-policy-v1",
            "completed_at_utc": "2026-11-01T10:30:00Z",
            "scored_before_real_adaptation": True,
        },
    )
    return {
        "candidate_id": "real-adapted-v2",
        "baseline_freeze_path": freeze_path,
        "baseline_predictions_path": predictions_path,
        "baseline_score_receipt_path": receipt_path,
        "authorization_path": authorization_path,
        "train_manifest_path": train_path,
        "validation_manifest_path": validation_path,
        "protected_manifest_path": protected_path,
        "development_start_utc": "2026-11-01T11:00:00Z",
        "protected_sealed_at_utc": "2026-11-01T10:20:00Z",
        "development_lock_out": tmp_path / "training_side/development_lock.json",
        "protected_lock_out": tmp_path / "audit_side/protected_lock.json",
    }


def test_future_candidate_locks_pre_adaptation_baseline_and_fresh_visit(tmp_path):
    args = _fixture(tmp_path)
    development_path, protected_path = R.prepare_evidence(**args)

    development = json.loads(development_path.read_text(encoding="utf-8"))
    protected = json.loads(protected_path.read_text(encoding="utf-8"))
    assert tuple(development["canonical_classes"]) == D.CANONICAL_CLASSES
    assert development["synthetic_baseline"]["prediction_rows"] == 60
    assert development["synthetic_baseline"]["scored_before_real_adaptation"] is True
    assert development["synthetic_baseline"]["decision_policy_version"] == (
        "synthetic-score-policy-v1"
    )
    assert development["authorization"]["allowed_uses"] == [
        "baseline_scoring",
        "train",
        "validation",
    ]
    assert development["protected_reservation"]["lock_sha256"] == D.sha256_file(
        protected_path
    )
    assert development["protected_reservation"]["labels_included"] is False
    assert protected["manifest"]["counts_without_labels"] == {
        "rows": 30,
        "events": 1,
        "sessions": 1,
        "images": 1,
    }
    assert "classes" not in protected["manifest"]["counts_without_labels"]
    development_text = development_path.read_text(encoding="utf-8")
    for protected_identifier in (
        "protected-visit",
        "protected-event",
        "source-protected-visit",
        "raw/source-protected",
    ):
        assert protected_identifier not in development_text
    assert development_path.stat().st_mode & 0o777 == 0o444
    assert protected_path.stat().st_mode & 0o777 == 0o444


def test_protected_test_must_be_a_different_session_visit(tmp_path):
    args = _fixture(tmp_path)
    protected = _event_rows(
        "train-visit", "fresh-event", "test", "protected_test", visual="missing_cap"
    )
    _write_manifest(args["protected_manifest_path"], protected)
    with pytest.raises(D.ManifestError, match="different sealed session/visit"):
        R.prepare_evidence(**args)


def test_baseline_predictions_must_cover_development_only(tmp_path):
    args = _fixture(tmp_path)
    protected = D.read_manifest(args["protected_manifest_path"], verify_images=False)
    with args["baseline_predictions_path"].open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=R.BASELINE_PREDICTION_COLUMNS)
        row = protected[0]
        writer.writerow(
            {
                "source_image_id": row.source_image_id,
                "fixture_id": row.fixture_id,
                "position_id": row.position_id,
                "expected_occupancy": row.expected_occupancy,
                "predicted_visual_class": "pass",
                "model_score": "0.1",
                "applicable_threshold": "0.5",
                "inspection_decision": "PASS",
                "reason_code": "should_not_be_here",
                "inspection_timestamp_utc": "2026-11-01T10:15:00Z",
                "event_uid": row.event_uid,
            }
        )
    _refresh_prediction_receipt(args)

    with pytest.raises(D.ManifestError, match="train/validation targets"):
        R.prepare_evidence(**args)


def test_fixture_s_requires_a_distinct_calibrated_map(tmp_path):
    args = _fixture(tmp_path, validation_fixture="S")
    with pytest.raises(D.ManifestError, match="no calibrated position-map hash"):
        R.prepare_evidence(**args)

    args = _fixture(tmp_path / "same-map", validation_fixture="S", maps={"G": H, "S": H})
    with pytest.raises(D.ManifestError, match="cannot reuse the G map hash"):
        R.prepare_evidence(**args)


def test_baseline_scoring_must_finish_before_real_adaptation(tmp_path):
    args = _fixture(tmp_path, prediction_time="2026-11-01T11:15:00Z")
    receipt = json.loads(args["baseline_score_receipt_path"].read_text(encoding="utf-8"))
    receipt["completed_at_utc"] = "2026-11-01T11:30:00Z"
    _write_json(args["baseline_score_receipt_path"], receipt)
    with pytest.raises(D.ManifestError, match="completed after real-data development started"):
        R.prepare_evidence(**args)


def test_authorization_cannot_allow_protected_test_tuning(tmp_path):
    args = _fixture(tmp_path)
    authorization = json.loads(args["authorization_path"].read_text(encoding="utf-8"))
    authorization["allowed_uses"].append("protected_test_tuning")
    _write_json(args["authorization_path"], authorization)
    with pytest.raises(D.ManifestError, match="unrecognized or unsafe uses"):
        R.prepare_evidence(**args)


def test_evidence_locks_are_never_overwritten(tmp_path):
    args = _fixture(tmp_path)
    args["development_lock_out"].parent.mkdir(parents=True)
    args["development_lock_out"].write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        R.prepare_evidence(**args)


def test_baseline_must_be_frozen_before_every_staging_capture(tmp_path):
    args = _fixture(tmp_path)
    freeze = json.loads(args["baseline_freeze_path"].read_text(encoding="utf-8"))
    freeze["frozen_at_utc"] = "2026-11-01T10:05:00Z"
    _write_json(args["baseline_freeze_path"], freeze)
    with pytest.raises(
        D.ManifestError, match="frozen before every development and protected capture"
    ):
        R.prepare_evidence(**args)


def test_protected_capture_before_baseline_freeze_is_rejected(tmp_path):
    args = _fixture(tmp_path)
    protected = D.read_manifest(args["protected_manifest_path"], verify_images=False)
    _write_manifest(
        args["protected_manifest_path"],
        [
            replace(row, capture_time="2026-11-01T04:59:00-05:00")
            for row in protected
        ],
    )
    with pytest.raises(
        D.ManifestError, match="frozen before every development and protected capture"
    ):
        R.prepare_evidence(**args)


def test_each_baseline_prediction_must_follow_its_capture(tmp_path):
    args = _fixture(tmp_path, prediction_time="2026-11-01T10:05:00Z")
    with pytest.raises(D.ManifestError, match="before or at its corresponding capture"):
        R.prepare_evidence(**args)


def test_score_receipt_must_complete_at_or_after_every_prediction(tmp_path):
    args = _fixture(tmp_path)
    receipt = json.loads(args["baseline_score_receipt_path"].read_text(encoding="utf-8"))
    receipt["completed_at_utc"] = "2026-11-01T10:14:59Z"
    _write_json(args["baseline_score_receipt_path"], receipt)
    with pytest.raises(D.ManifestError, match="after score receipt completion"):
        R.prepare_evidence(**args)


def test_protected_capture_cannot_occur_after_seal(tmp_path):
    args = _fixture(tmp_path)
    args["protected_sealed_at_utc"] = "2026-11-01T10:04:59Z"
    with pytest.raises(D.ManifestError, match="protected capture occurred after"):
        R.prepare_evidence(**args)


def test_protected_seal_must_strictly_precede_development(tmp_path):
    args = _fixture(tmp_path)
    args["protected_sealed_at_utc"] = args["development_start_utc"]
    with pytest.raises(D.ManifestError, match="sealed before development starts"):
        R.prepare_evidence(**args)


def test_score_completion_must_strictly_precede_development(tmp_path):
    args = _fixture(tmp_path)
    receipt = json.loads(args["baseline_score_receipt_path"].read_text(encoding="utf-8"))
    receipt["completed_at_utc"] = args["development_start_utc"]
    _write_json(args["baseline_score_receipt_path"], receipt)
    with pytest.raises(D.ManifestError, match="completed after real-data development started"):
        R.prepare_evidence(**args)


@pytest.mark.parametrize(
    ("visual", "decision"),
    [
        ("missing_cap", "PASS"),
        ("unknown", "PASS"),
        ("no_vial", "WAITING"),
    ],
)
def test_unsafe_visual_decision_pairs_are_rejected(tmp_path, visual, decision):
    args = _fixture(tmp_path)
    _change_first_prediction(
        args,
        predicted_visual_class=visual,
        inspection_decision=decision,
        model_score="0.90",
    )
    with pytest.raises(D.ManifestError, match="unsafe visual/decision pair"):
        R.prepare_evidence(**args)


def test_supported_defect_score_must_reach_threshold(tmp_path):
    args = _fixture(tmp_path)
    _change_first_prediction(
        args,
        predicted_visual_class="missing_cap",
        inspection_decision="DEFECT",
        model_score="0.49",
        applicable_threshold="0.50",
    )
    with pytest.raises(D.ManifestError, match="defect prediction is below"):
        R.prepare_evidence(**args)


def test_score_receipt_attests_frozen_decision_policy(tmp_path):
    args = _fixture(tmp_path)
    receipt = json.loads(args["baseline_score_receipt_path"].read_text(encoding="utf-8"))
    receipt["decision_policy_version"] = "different-policy"
    _write_json(args["baseline_score_receipt_path"], receipt)
    with pytest.raises(D.ManifestError, match="different decision-policy version"):
        R.prepare_evidence(**args)


def test_duplicate_raw_digest_cannot_represent_distinct_events(tmp_path):
    args = _fixture(tmp_path)
    train = D.read_manifest(args["train_manifest_path"], verify_images=False)
    validation = D.read_manifest(args["validation_manifest_path"], verify_images=False)
    duplicated = [
        replace(row, source_image_sha256=train[0].source_image_sha256)
        for row in validation
    ]
    _write_manifest(args["validation_manifest_path"], duplicated)
    with pytest.raises(D.ManifestError, match="duplicate raw image digest"):
        R.prepare_evidence(**args)


def test_short_prediction_csv_is_a_manifest_error(tmp_path):
    args = _fixture(tmp_path)
    args["baseline_predictions_path"].write_text(
        ",".join(R.BASELINE_PREDICTION_COLUMNS) + "\nonly-a-source-id\n",
        encoding="utf-8",
    )
    _refresh_prediction_receipt(args)
    with pytest.raises(D.ManifestError, match="malformed/short CSV field"):
        R.prepare_evidence(**args)


def test_dual_lock_publication_rolls_back_first_lock_on_second_failure(
    tmp_path, monkeypatch
):
    args = _fixture(tmp_path)
    real_link = R.os.link
    calls = 0

    def fail_second_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second-lock publication failure")
        return real_link(source, destination)

    monkeypatch.setattr(R.os, "link", fail_second_link)
    with pytest.raises(OSError, match="second-lock publication failure"):
        R.prepare_evidence(**args)
    assert not args["protected_lock_out"].exists()
    assert not args["development_lock_out"].exists()
    assert not list(args["protected_lock_out"].parent.glob(".*.tmp"))
    assert not list(args["development_lock_out"].parent.glob(".*.tmp"))


def test_authorization_free_text_cannot_leak_protected_identifiers(tmp_path):
    args = _fixture(tmp_path)
    authorization = json.loads(args["authorization_path"].read_text(encoding="utf-8"))
    protected_strings = (
        "protected-visit",
        "protected-event",
        "raw/source-protected-visit-protected-event-G.png",
    )
    authorization["authorized_by"] = protected_strings[0]
    authorization["dataset_scope"] = protected_strings[1]
    authorization["authorization_statement"] = protected_strings[2]
    _write_json(args["authorization_path"], authorization)

    development_path, _ = R.prepare_evidence(**args)
    development_text = development_path.read_text(encoding="utf-8")
    assert all(value not in development_text for value in protected_strings)
