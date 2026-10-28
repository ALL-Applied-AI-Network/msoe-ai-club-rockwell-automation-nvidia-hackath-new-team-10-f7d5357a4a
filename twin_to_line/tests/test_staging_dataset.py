import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import positions as P
import build_staging_manifest as B
import evaluate_resnet5_test as E
import make_yaad_training_bundle as Y
import staging_dataset as D
import train_resnet5 as T


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event_rows(
    root,
    event,
    target="pass",
    designation="development",
    fixture="G",
    color=20,
    session=None,
):
    session = session or f"session-{event}"
    image = np.full((480, 640, 3), color, np.uint8)
    path = root / f"event_{event}.png"
    assert cv2.imwrite(str(path), image)
    image_sha = digest(path)
    rows = []
    for index in range(1, 31):
        visual = target if index == 1 else "pass"
        expected = "occupied"
        sequence = ""
        if visual == "no_vial":
            sequence = "missed"
        rows.append(
            D.ManifestRow(
                sample_id=f"{session}:{event}:{fixture}{index:02d}",
                event_uid=f"{session}:{event}",
                session_id=session,
                event_id=str(event),
                source_image_id=f"image-{event}",
                source_image_relpath=path.name,
                source_image_sha256=image_sha,
                capture_time=f"2026-10-27T12:{int(event):02d}:00-05:00",
                fixture_id=fixture,
                position_id=f"{fixture}{index:02d}",
                expected_occupancy=expected,
                visual_class=visual,
                inspection_mode="final_inspection",
                gt_sequence=sequence,
                designation=designation,
                usable="true",
                split="",
                split_reason="",
                label_source="human-confirmed staging manifest",
                annotator_id="reviewer-1",
                annotation_confidence="1.0",
                review_state="adjudicated",
                notes="",
            )
        )
    return rows


def write_manifest(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=D.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(row.as_csv_dict() for row in rows)


def development_rows(root):
    classes = [
        "pass",
        "missing_cap",
        "crooked_cap",
        "no_vial",
        "missing_cap",
        "crooked_cap",
        "no_vial",
        "pass",
        "missing_cap",
        "crooked_cap",
    ]
    rows = []
    for index, visual in enumerate(classes, 1):
        designation = "protected_test" if index == len(classes) else "development"
        rows.extend(event_rows(root, str(index), visual, designation, color=10 + index))
    return rows


def test_manifest_split_is_event_atomic_and_protected(tmp_path):
    rows = development_rows(tmp_path)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    loaded = D.read_manifest(manifest, image_root=tmp_path)
    assigned, assignment = D.assign_event_splits(
        loaded, seed=17, validation_fraction=0.2, test_fraction=0.2
    )
    assert assignment["session-10:10"] == "test"
    assert set(assignment.values()) == {"train", "validation", "test"}
    for event in assignment:
        assert {row.split for row in assigned if row.event_uid == event} == {assignment[event]}
    D.validate_split_integrity(assigned)


def test_split_bundle_is_locked_and_training_hashes_are_verified(tmp_path):
    rows = development_rows(tmp_path)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    loaded = D.read_manifest(manifest, image_root=tmp_path)
    assigned, assignment = D.assign_event_splits(
        loaded, seed=3, validation_fraction=0.2, test_fraction=0.2
    )
    out = tmp_path / "splits"
    lock_path = D.write_split_bundle(
        assigned,
        assignment,
        out,
        source_manifest=manifest,
        image_root=tmp_path,
        seed=3,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    lock = D.verify_training_manifests(
        out / "train_manifest.csv", out / "validation_manifest.csv", lock_path
    )
    assert lock["events"]["test"]
    assert oct((out / "protected_test" / "test_manifest.csv").stat().st_mode & 0o777) == "0o444"
    with pytest.raises(FileExistsError):
        D.write_split_bundle(
            assigned,
            assignment,
            out,
            source_manifest=manifest,
            image_root=tmp_path,
            seed=3,
            validation_fraction=0.2,
            test_fraction=0.2,
        )


def test_split_files_are_deterministic_when_source_rows_are_reordered(tmp_path):
    rows = development_rows(tmp_path)
    first_manifest = tmp_path / "first.csv"
    second_manifest = tmp_path / "second.csv"
    write_manifest(first_manifest, rows)
    write_manifest(second_manifest, list(reversed(rows)))
    outputs = []
    for number, manifest in enumerate((first_manifest, second_manifest), 1):
        loaded = D.read_manifest(manifest, image_root=tmp_path)
        assigned, assignment = D.assign_event_splits(
            loaded, seed=9, validation_fraction=0.2, test_fraction=0.2
        )
        out = tmp_path / f"split-{number}"
        D.write_split_bundle(
            assigned,
            assignment,
            out,
            source_manifest=manifest,
            image_root=tmp_path,
            seed=9,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
        outputs.append(out)
    for name in ("train_manifest.csv", "validation_manifest.csv", "protected_test/test_manifest.csv"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()


def test_collection_session_is_atomic_across_events(tmp_path):
    rows = []
    rows.extend(event_rows(tmp_path, "1", session="visit-a", color=31))
    rows.extend(event_rows(tmp_path, "2", session="visit-a", color=32))
    rows.extend(event_rows(tmp_path, "3", session="visit-b", color=33))
    rows.extend(event_rows(tmp_path, "4", session="visit-c", color=34))
    rows.extend(event_rows(tmp_path, "5", session="visit-d", color=35))
    assigned, assignment = D.assign_event_splits(
        rows, seed=7, validation_fraction=0.25, test_fraction=0.25
    )
    assert assignment["visit-a:1"] == assignment["visit-a:2"]
    D.validate_split_integrity(assigned)


def test_one_visit_cannot_be_split_into_train_validation_test(tmp_path):
    rows = []
    for event in ("1", "2", "3"):
        rows.extend(event_rows(tmp_path, event, session="one-visit", color=40 + int(event)))
    with pytest.raises(D.ManifestError, match="three independent collection sessions"):
        D.assign_event_splits(rows, seed=7)


def test_explicit_within_visit_event_split_is_locked_and_disclosed(tmp_path):
    rows = []
    for event in ("1", "2", "3", "4", "5", "6"):
        designation = "protected_test" if event == "6" else "development"
        rows.extend(
            event_rows(
                tmp_path,
                event,
                target="missing_cap" if event in {"2", "4", "6"} else "pass",
                designation=designation,
                session="one-visit",
                color=50 + int(event),
            )
        )
    assigned, assignment = D.assign_event_splits(
        rows,
        seed=7,
        validation_fraction=0.2,
        test_fraction=0.2,
        explicit_validation_events=["one-visit:5"],
        split_unit="event",
    )
    assert assignment["one-visit:6"] == "test"
    assert assignment["one-visit:5"] == "validation"
    assert set(assignment.values()) == {"train", "validation", "test"}
    manifest = tmp_path / "one-visit.csv"
    write_manifest(manifest, rows)
    with pytest.raises(D.ManifestError, match="explicitly acknowledge"):
        D.write_split_bundle(
            assigned,
            assignment,
            tmp_path / "refused",
            source_manifest=manifest,
            image_root=tmp_path,
            seed=7,
            validation_fraction=0.2,
            test_fraction=0.2,
            split_unit="event",
        )
    lock_path = D.write_split_bundle(
        assigned,
        assignment,
        tmp_path / "accepted",
        source_manifest=manifest,
        image_root=tmp_path,
        seed=7,
        validation_fraction=0.2,
        test_fraction=0.2,
        split_unit="event",
        acknowledge_within_visit_evaluation=True,
        leakage_justification="project owner requested an exploratory setup-event holdout",
    )
    lock = json.loads(lock_path.read_text())
    assert lock["split_unit"] == "event"
    assert lock["independent_visit_test"] is False
    assert lock["sessions_crossing_splits"] == ["one-visit"]
    assert "not independent" in lock["metric_warning"]


def test_protected_test_cannot_be_relabelled_train_before_pixel_read(tmp_path):
    rows = event_rows(tmp_path, "1", designation="protected_test")
    rows = [replace(row, split="train") for row in rows]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    for image in tmp_path.glob("*.png"):
        image.unlink()
    with pytest.raises(D.ProtectedTestError):
        D.read_manifest(manifest, verify_images=False)


def test_byte_identical_images_cannot_cross_events(tmp_path):
    rows_a = event_rows(tmp_path, "1", color=44)
    rows_b = event_rows(tmp_path, "2", color=55)
    source = tmp_path / "event_1.png"
    duplicate = tmp_path / "event_2.png"
    shutil.copyfile(source, duplicate)
    duplicate_sha = digest(duplicate)
    rows_b = [replace(row, source_image_sha256=duplicate_sha) for row in rows_b]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows_a + rows_b)
    with pytest.raises(D.ManifestError, match="byte-identical image"):
        D.read_manifest(manifest, image_root=tmp_path)


def test_training_adapter_refuses_test_and_unknown_fixture_map(tmp_path):
    test_rows = [replace(row, split="test") for row in event_rows(tmp_path, "1")]
    with pytest.raises(D.ProtectedTestError):
        D.StagingCropSet(test_rows, tmp_path, allowed_split="test")
    s_rows = [replace(row, split="train") for row in event_rows(tmp_path, "2", fixture="S")]
    with pytest.raises(D.ManifestError, match="no calibrated position map"):
        D.StagingCropSet(s_rows, tmp_path, allowed_split="train", fixture_maps={"G": P.PositionMap()})


def test_zero_damaged_keeps_five_indices_and_finite_zero_weight(tmp_path):
    train = []
    validation = []
    supported = ["pass", "missing_cap", "crooked_cap", "no_vial"]
    for index, visual in enumerate(supported, 1):
        train.extend(
            replace(row, split="train")
            for row in event_rows(tmp_path, f"1{index}", visual, color=80 + index)
        )
        validation.extend(
            replace(row, split="validation")
            for row in event_rows(tmp_path, f"2{index}", visual, color=90 + index)
        )
    assert T.validate_class_support(train, validation) == ["damaged_vial"]
    weights = T.class_weights(T.class_counts(train))
    assert len(weights) == 5
    assert np.isfinite(weights).all()
    assert weights[D.CANONICAL_CLASSES.index("damaged_vial")] == 0


def test_threshold_is_validation_only_and_unsupported_damage_escalates():
    labels = np.array([0, 1, 2, 4])
    probabilities = np.array(
        [
            [0.90, 0.02, 0.02, 0.01, 0.05],
            [0.01, 0.85, 0.05, 0.04, 0.05],
            [0.01, 0.04, 0.86, 0.04, 0.05],
            [0.05, 0.02, 0.02, 0.01, 0.90],
        ],
        dtype=np.float64,
    )
    threshold = T.choose_threshold(probabilities, labels, ["damaged_vial"])
    assert threshold["threshold_source"] == "validation_only"
    assert threshold["per_supported_defect_threshold"]["damaged_vial"] is None
    predicted, score, reason = T.decide_one(
        np.array([0.00, 0.05, 0.04, 0.90, 0.01]),
        float(threshold["t_defect"]),
        ["damaged_vial"],
    )
    assert predicted == "unknown" and score >= threshold["t_defect"]
    assert "unsupported" in reason


def test_protected_test_metrics_report_visual_and_process_accuracy(tmp_path):
    missing = replace(
        event_rows(tmp_path, "31", "missing_cap", color=131)[0], split="test"
    )
    passed = replace(event_rows(tmp_path, "32", "pass", color=132)[0], split="test")
    missed = replace(event_rows(tmp_path, "33", "no_vial", color=133)[0], split="test")
    rows = [missing, passed, missed]
    probabilities = np.array(
        [
            [0.03, 0.86, 0.04, 0.02, 0.05],
            [0.90, 0.02, 0.02, 0.01, 0.05],
            [0.03, 0.02, 0.02, 0.01, 0.92],
        ],
        dtype=np.float64,
    )
    metrics, predictions = E.compute_metrics(
        rows, probabilities, threshold=0.5, unsupported_classes=["damaged_vial"]
    )
    assert metrics["visual_exact_accuracy"] == 1.0
    assert metrics["process_decision_accuracy"] == 1.0
    assert metrics["supported_visual_defect_recall"] == 1.0
    assert metrics["missed_placement_recall"] == 1.0
    assert metrics["per_class"]["damaged_vial"]["supported"] is False
    assert predictions[2]["inspection_decision"] == "MISSED"


def test_preprocessing_inventory_records_registration_hash_or_missing(
    tmp_path, monkeypatch
):
    registration = tmp_path / "registration_dev.json"
    warp = tmp_path / "frame_warp_1080.json"
    position_map = tmp_path / "position_map_G.json"
    warp.write_text("warp-v1", encoding="utf-8")
    position_map.write_text("map-v1", encoding="utf-8")
    monkeypatch.setattr(T.P, "REG_PATH", str(registration))

    missing = T.preprocessing_sha256(warp, {"G": position_map})
    assert missing["registration_dev.json"] is None
    assert missing["position_map_G"] == digest(position_map)
    assert missing["train_resnet.py"] == digest(Path(T.__file__).with_name("train_resnet.py"))

    registration.write_text("registration-v1", encoding="utf-8")
    present = T.preprocessing_sha256(warp, {"G": position_map})
    assert present["registration_dev.json"] == digest(registration)
    with pytest.raises(D.ManifestError, match="registration_dev.json"):
        T.verify_preprocessing_sha256(missing, warp, {"G": position_map})


def test_protected_test_refuses_preprocessing_change_before_pixel_read(
    tmp_path, monkeypatch
):
    registration = tmp_path / "registration_dev.json"
    warp = tmp_path / "frame_warp_1080.json"
    position_map = tmp_path / "position_map_G.json"
    split_lock = tmp_path / "split_lock.json"
    test_manifest = tmp_path / "test_manifest.csv"
    registration.write_text("registration-v1", encoding="utf-8")
    warp.write_text("warp-v1", encoding="utf-8")
    position_map.write_text("map-v1", encoding="utf-8")
    split_lock.write_text("locked split", encoding="utf-8")
    test_manifest.write_text("sealed test", encoding="utf-8")
    monkeypatch.setattr(T.P, "REG_PATH", str(registration))

    recorded = T.preprocessing_sha256(warp, {"G": position_map})
    position_map.write_text("map-v2", encoding="utf-8")

    class DummyInspector:
        def __init__(self, model_dir, device):
            del model_dir, device
            self.run = {
                "split_lock": {"sha256": digest(split_lock)},
                "threshold_source": "validation_only",
                "preprocessing_sha256": recorded,
            }

    pixel_read_attempted = False

    def forbidden_read(*args, **kwargs):
        nonlocal pixel_read_attempted
        pixel_read_attempted = True
        raise AssertionError("protected pixels were reached before provenance checks")

    monkeypatch.setattr(E, "verify_test_manifest", lambda *args: {})
    monkeypatch.setattr(E, "ResNet5Inspector", DummyInspector)
    monkeypatch.setattr(E, "_parse_maps", lambda values: ({"G": object()}, {"G": position_map}))
    monkeypatch.setattr(E.FW, "FrameWarp", lambda path: object())
    monkeypatch.setattr(E, "read_manifest", forbidden_read)

    with pytest.raises(D.ManifestError, match="position_map_G"):
        E.main(
            [
                "--model-dir",
                str(tmp_path / "model"),
                "--test-manifest",
                str(test_manifest),
                "--split-lock",
                str(split_lock),
                "--image-root",
                str(tmp_path),
                "--out",
                str(tmp_path / "out"),
                "--frame-warp",
                str(warp),
                "--device",
                "cpu",
            ]
        )
    assert pixel_read_attempted is False


def test_manifest_requires_all_thirty_positions(tmp_path):
    rows = event_rows(tmp_path, "1")[:-1]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    with pytest.raises(D.ManifestError, match="expected all 30 positions"):
        D.read_manifest(manifest, image_root=tmp_path)


def test_exact_taxonomy_rejects_aliases(tmp_path):
    rows = event_rows(tmp_path, "1")
    rows[0] = replace(rows[0], visual_class="damaged")
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    with pytest.raises(D.ManifestError, match="exact five-class taxonomy"):
        D.read_manifest(manifest, verify_images=False)


def test_plan_expansion_maps_partial_fill_and_lean_without_pixel_inference():
    plan_path = Path(__file__).parents[2] / "ignite_core" / "tuesday_staging_plan.csv"
    plan = B.load_plan(plan_path)
    partial = B.expand_plan_setup(plan["19"])
    assert partial["G21"] == ("pass", "occupied", "")
    assert partial["G22"] == ("no_vial", "occupied", "missed")
    assert partial["G26"] == ("no_vial", "empty", "waiting")
    lean = B.expand_plan_setup(plan["14"])
    assert lean["G02"] == ("pass", "occupied", "")


def test_manifest_builder_requires_human_plan_confirmation(tmp_path):
    image = np.zeros((480, 640, 3), np.uint8)
    assert cv2.imwrite(str(tmp_path / "capture.png"), image)
    review = tmp_path / "review.csv"
    with review.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=B.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "session_id": "visit-1",
                "event_id": "setup-01",
                "setup": "01",
                "source_image_id": "capture-1",
                "source_image_relpath": "capture.png",
                "capture_time": "2026-10-27T12:00:00-05:00",
                "fixture_id": "G",
                "designation": "development",
                "usable": "true",
                "confirmed_matches_plan": "false",
                "label_source": "operator review",
                "annotator_id": "operator-1",
                "annotation_confidence": "0.8",
                "review_state": "adjudicated",
                "notes": "physical deviation",
            }
        )
    plan_path = Path(__file__).parents[2] / "ignite_core" / "tuesday_staging_plan.csv"
    with pytest.raises(D.ManifestError, match="intended-plan reference labels"):
        B.build_rows(review, plan_path, tmp_path)

    protocol_review = list(csv.DictReader(review.open(newline="", encoding="utf-8")))[0]
    protocol_review["review_state"] = "protocol_reference"
    with review.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=B.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(protocol_review)
    protocol_rows = B.build_rows(
        review,
        plan_path,
        tmp_path,
        allow_intended_plan_reference=True,
    )
    assert len(protocol_rows) == 30
    assert {row.review_state for row in protocol_rows} == {"protocol_reference"}


def test_manifest_requires_adjudicator_confidence_and_review_state(tmp_path):
    rows = event_rows(tmp_path, "1")
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [replace(row, review_state="draft") for row in rows])
    with pytest.raises(D.ManifestError, match="review_state must be"):
        D.read_manifest(manifest, verify_images=False)
    write_manifest(
        tmp_path / "confidence.csv",
        [replace(row, annotation_confidence="1.2") for row in rows],
    )
    with pytest.raises(D.ManifestError, match="annotation_confidence"):
        D.read_manifest(tmp_path / "confidence.csv", verify_images=False)


def test_yaad_bundle_excludes_protected_test_manifest_and_pixels(tmp_path):
    rows = development_rows(tmp_path)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    loaded = D.read_manifest(manifest, image_root=tmp_path)
    assigned, assignment = D.assign_event_splits(
        loaded, seed=5, validation_fraction=0.2, test_fraction=0.2
    )
    splits = tmp_path / "splits"
    D.write_split_bundle(
        assigned,
        assignment,
        splits,
        source_manifest=manifest,
        image_root=tmp_path,
        seed=5,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    bundle = tmp_path / "bundle"
    bundle_manifest = Y.build_bundle(splits, tmp_path, bundle)
    info = json.loads(bundle_manifest.read_text())
    assert info["protected_test_images_copied"] == 0
    assert not (bundle / "splits" / "protected_test").exists()
    test_images = {
        row.source_image_relpath
        for row in assigned
        if row.split == "test"
    }
    copied = {str(path.relative_to(bundle / "images")) for path in (bundle / "images").rglob("*.png")}
    assert copied.isdisjoint(test_images)


def _build_yaad_job_fixture(tmp_path):
    rows = development_rows(tmp_path)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    loaded = D.read_manifest(manifest, image_root=tmp_path)
    assigned, assignment = D.assign_event_splits(
        loaded, seed=5, validation_fraction=0.2, test_fraction=0.2
    )
    splits = tmp_path / "splits"
    D.write_split_bundle(
        assigned,
        assignment,
        splits,
        source_manifest=manifest,
        image_root=tmp_path,
        seed=5,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    repo_root = Path(Y.__file__).resolve().parents[1]
    job = tmp_path / "job"
    Y.build_job_bundle(splits, tmp_path, repo_root, job)
    return job, splits, repo_root


def test_yaad_job_bundle_allowlists_code_and_omits_protected_metadata(tmp_path):
    job, splits, repo_root = _build_yaad_job_fixture(tmp_path)
    code_root = job / "team_repo" / "twin_to_line"
    copied_code = {
        str(path.relative_to(code_root))
        for path in code_root.rglob("*")
        if path.is_file()
    }

    assert copied_code == set(Y.TRAINING_CODE_ALLOWLIST)
    assert (repo_root / "twin_to_line/staging/test_events_2026-10-27.txt").is_file()
    assert (repo_root / "twin_to_line/staging/capture_review_2026-10-27_protocol_reference.csv").is_file()
    assert not (code_root / "staging").exists()
    assert not any(path.name.startswith("test_events") for path in job.rglob("*"))
    assert not any(path.name.startswith("capture_review") for path in job.rglob("*"))
    assert not any(path.name == "test_manifest.csv" for path in job.rglob("*"))
    assert not any("protected_test" in path.parts for path in job.rglob("*"))
    assert (job / "dataset/splits/split_lock.json").read_bytes() == (
        splits / "split_lock.json"
    ).read_bytes()
    Y.audit_job_bundle(job)


def test_yaad_job_bundle_audit_rejects_code_outside_allowlist(tmp_path):
    job, _, _ = _build_yaad_job_fixture(tmp_path)
    leaked = job / "team_repo/twin_to_line/staging/test_events_private.txt"
    leaked.parent.mkdir()
    leaked.write_text("protected-event\n", encoding="utf-8")

    with pytest.raises(D.ManifestError, match="allowlist"):
        Y.audit_job_bundle(job)
