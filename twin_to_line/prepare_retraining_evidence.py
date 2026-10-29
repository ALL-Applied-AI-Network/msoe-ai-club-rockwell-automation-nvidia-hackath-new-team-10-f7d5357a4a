#!/usr/bin/env python3
"""Lock evidence boundaries before adapting a synthetic baseline to real staging data.

This command is metadata-only: it validates manifests and recorded SHA-256
digests without hashing or decoding any source image.  It authenticates that a
frozen synthetic-only baseline scored every development target before real-data
adaptation starts, then emits separate development and protected-reservation
locks.  The development lock contains no protected event, session, source, or
path identifiers and is the only lock that may accompany a training job.

This is for a future candidate.  It does not alter the closed staging5 split,
model, protected result, or USB kit.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from staging_dataset import (
    CANONICAL_CLASSES,
    DEFECT_CLASSES,
    FIXTURES,
    ManifestError,
    ManifestRow,
    read_manifest,
    sha256_file,
    validate_split_integrity,
)


SCHEMA_VERSION = 1
BASELINE_PREDICTION_COLUMNS = (
    "source_image_id",
    "fixture_id",
    "position_id",
    "expected_occupancy",
    "predicted_visual_class",
    "model_score",
    "applicable_threshold",
    "inspection_decision",
    "reason_code",
    "inspection_timestamp_utc",
    "event_uid",
)
VISUAL_OUTPUTS = frozenset(CANONICAL_CLASSES) | {"unknown"}
INSPECTION_DECISIONS = {"PASS", "DEFECT", "MISSED", "WAITING", "UNSURE"}
SHA256_FIELDS = (
    "model_sha256",
    "thresholds_sha256",
    "decision_policy_sha256",
)
AUTHORIZED_USES = frozenset({"baseline_scoring", "train", "validation"})


def _load_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a nonempty string")
    return value.strip()


def _sha256(value: object, field: str) -> str:
    text = _required_text(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ManifestError(f"{field} must be 64 lowercase hexadecimal digits")
    return text


def _timestamp(value: object, field: str) -> datetime:
    text = _required_text(value, field)
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ManifestError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ManifestError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_mapping(value: object, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ManifestError(f"{field} must be a nonempty object of SHA-256 values")
    result: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        name = _required_text(raw_name, f"{field} key")
        result[name] = _sha256(raw_digest, f"{field}.{name}")
    return result


def validate_baseline_freeze(
    value: Mapping[str, object], fixtures: set[str]
) -> tuple[str, datetime, dict[str, str], str, str]:
    """Validate the immutable synthetic-baseline inventory."""
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("baseline freeze has an unsupported schema_version")
    baseline_id = _required_text(value.get("baseline_id"), "baseline_freeze.baseline_id")
    if value.get("training_data_scope") != "synthetic_only":
        raise ManifestError("baseline freeze must declare training_data_scope=synthetic_only")
    if tuple(value.get("canonical_classes", ())) != CANONICAL_CLASSES:
        raise ManifestError("baseline freeze does not preserve the exact five-class taxonomy")
    frozen_at = _timestamp(value.get("frozen_at_utc"), "baseline_freeze.frozen_at_utc")
    hashes = {
        field: _sha256(value.get(field), f"baseline_freeze.{field}")
        for field in SHA256_FIELDS
    }
    decision_policy_version = _required_text(
        value.get("decision_policy_version"),
        "baseline_freeze.decision_policy_version",
    )
    _hash_mapping(value.get("preprocessing_sha256"), "baseline_freeze.preprocessing_sha256")
    _hash_mapping(value.get("code_sha256"), "baseline_freeze.code_sha256")
    maps = _hash_mapping(value.get("fixture_map_sha256"), "baseline_freeze.fixture_map_sha256")
    unknown = set(maps) - set(FIXTURES)
    if unknown:
        raise ManifestError(f"baseline freeze has unknown fixture maps {sorted(unknown)}")
    missing = fixtures - set(maps)
    if missing:
        raise ManifestError(
            f"baseline freeze has no calibrated position-map hash for fixture(s) {sorted(missing)}"
        )
    if {"G", "S"} <= fixtures and maps["G"] == maps["S"]:
        raise ManifestError("fixture S must have its own calibrated map; it cannot reuse the G map hash")
    return (
        baseline_id,
        frozen_at,
        maps,
        decision_policy_version,
        hashes["decision_policy_sha256"],
    )


def validate_authorization(
    value: Mapping[str, object], development_start: datetime
) -> tuple[datetime, list[str]]:
    """Validate the project owner's explicit staging-data use authorization."""
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("authorization record has an unsupported schema_version")
    if value.get("staging_training_authorized") is not True:
        raise ManifestError("authorization record must explicitly allow staging-data training")
    _required_text(value.get("authorized_by"), "authorization.authorized_by")
    authorized_at = _timestamp(value.get("authorized_at_utc"), "authorization.authorized_at_utc")
    _required_text(value.get("dataset_scope"), "authorization.dataset_scope")
    _required_text(
        value.get("authorization_statement"), "authorization.authorization_statement"
    )
    uses = value.get("allowed_uses")
    if not isinstance(uses, list) or not all(isinstance(item, str) for item in uses):
        raise ManifestError("authorization.allowed_uses must be a string list")
    if any(item != item.strip() or not item for item in uses):
        raise ManifestError("authorization.allowed_uses entries must be nonempty normalized strings")
    if len(uses) != len(set(uses)):
        raise ManifestError("authorization.allowed_uses must not contain duplicates")
    unrecognized = set(uses) - AUTHORIZED_USES
    if unrecognized:
        raise ManifestError(
            f"authorization.allowed_uses contains unrecognized or unsafe uses: {sorted(unrecognized)}"
        )
    if not {"train", "validation"} <= set(uses):
        raise ManifestError("authorization must allow both train and validation use")
    if authorized_at > development_start:
        raise ManifestError("staging-data authorization was recorded after development started")
    return authorized_at, list(uses)


def _read_partition(path: Path, split: str, designation: str) -> list[ManifestRow]:
    # Intentionally preserve recorded raw hashes without opening protected pixels.
    rows = read_manifest(path, verify_images=False)
    actual_splits = {row.split for row in rows}
    if actual_splits != {split}:
        raise ManifestError(f"{path}: expected only split={split}, got {sorted(actual_splits)}")
    actual_designations = {row.designation for row in rows}
    if actual_designations != {designation}:
        raise ManifestError(
            f"{path}: expected only designation={designation}, got {sorted(actual_designations)}"
        )
    return rows


def _partition_sets(rows: Sequence[ManifestRow]) -> dict[str, set[str]]:
    return {
        "events": {row.event_uid for row in rows},
        "sessions": {row.session_id for row in rows},
        "sources": {row.source_image_id for row in rows},
        "digests": {row.source_image_sha256 for row in rows},
    }


def _capture_times(
    rows: Sequence[ManifestRow], partition: str
) -> dict[tuple[str, str, str, str], datetime]:
    """Parse every recorded capture time and normalize it to UTC."""
    result: dict[tuple[str, str, str, str], datetime] = {}
    event_times: dict[tuple[str, str], datetime] = {}
    for row in rows:
        captured_at = _timestamp(
            row.capture_time,
            f"{partition} manifest {row.sample_id}.capture_time",
        )
        key = (row.event_uid, row.fixture_id, row.position_id, row.source_image_id)
        result[key] = captured_at
        event_key = (row.event_uid, row.source_image_id)
        prior = event_times.setdefault(event_key, captured_at)
        if prior != captured_at:
            raise ManifestError(
                f"{partition} event/source {event_key} has inconsistent capture_time values"
            )
    return result


def validate_partitions(
    train_rows: Sequence[ManifestRow],
    validation_rows: Sequence[ManifestRow],
    protected_rows: Sequence[ManifestRow],
) -> None:
    """Require event-atomic, visit-isolated, byte-disjoint partitions."""
    rows = list(train_rows) + list(validation_rows) + list(protected_rows)
    digest_events: dict[str, set[str]] = {}
    for row in rows:
        digest_events.setdefault(row.source_image_sha256, set()).add(row.event_uid)
    duplicate_digests = {
        digest: sorted(events)
        for digest, events in digest_events.items()
        if len(events) > 1
    }
    if duplicate_digests:
        raise ManifestError(
            f"duplicate raw image digest appears in distinct events: {duplicate_digests}"
        )
    partitions = {
        "train": _partition_sets(train_rows),
        "validation": _partition_sets(validation_rows),
        "protected_test": _partition_sets(protected_rows),
    }
    names = tuple(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for kind in ("events", "sessions", "sources", "digests"):
                overlap = partitions[left][kind] & partitions[right][kind]
                if overlap:
                    if "protected_test" in {left, right} and kind == "sessions":
                        raise ManifestError(
                            "the candidate protected test must be a different sealed session/visit "
                            "from all development data"
                        )
                    raise ManifestError(
                        f"{left} and {right} overlap in {kind}: {sorted(overlap)}"
                    )
    sample_ids = [row.sample_id for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ManifestError("sample_id is not unique across lifecycle partitions")
    position_keys = [
        (row.event_uid, row.fixture_id, row.position_id) for row in rows
    ]
    if len(position_keys) != len(set(position_keys)):
        raise ManifestError("event/fixture/position is not unique across lifecycle partitions")
    validate_split_integrity(rows)


def _prediction_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    return (
        row["event_uid"].strip(),
        row["fixture_id"].strip(),
        row["position_id"].strip(),
        row["source_image_id"].strip(),
    )


def validate_baseline_predictions(
    path: Path,
    development_rows: Sequence[ManifestRow],
    *,
    frozen_at: datetime,
    completed_at: datetime,
    development_start: datetime,
    capture_times: Mapping[tuple[str, str, str, str], datetime],
) -> int:
    """Require exactly one pre-adaptation baseline output for every development target."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing_columns = [
            column for column in BASELINE_PREDICTION_COLUMNS
            if column not in (reader.fieldnames or ())
        ]
        if missing_columns:
            raise ManifestError(f"baseline predictions are missing columns {missing_columns}")
        try:
            predictions = list(reader)
        except csv.Error as exc:
            raise ManifestError(f"baseline predictions contain malformed CSV: {exc}") from exc
    expected = {
        (row.event_uid, row.fixture_id, row.position_id, row.source_image_id): row
        for row in development_rows
    }
    actual: dict[tuple[str, str, str, str], Mapping[str, str]] = {}
    inspected_times: dict[tuple[str, str, str, str], datetime] = {}
    for number, raw_prediction in enumerate(predictions, 2):
        prediction: dict[str, str] = {}
        for field in BASELINE_PREDICTION_COLUMNS:
            raw_value = raw_prediction.get(field)
            if not isinstance(raw_value, str):
                raise ManifestError(
                    f"baseline predictions row {number}: malformed/short CSV field {field}"
                )
            prediction[field] = raw_value.strip()
        key = _prediction_key(prediction)
        if key in actual:
            raise ManifestError(f"baseline predictions row {number}: duplicate target {key}")
        actual[key] = prediction
        visual = prediction["predicted_visual_class"]
        if visual not in VISUAL_OUTPUTS:
            raise ManifestError(
                f"baseline predictions row {number}: invalid predicted_visual_class {visual!r}"
            )
        decision = prediction["inspection_decision"]
        if decision not in INSPECTION_DECISIONS:
            raise ManifestError(
                f"baseline predictions row {number}: invalid inspection_decision {decision!r}"
            )
        if not prediction["reason_code"]:
            raise ManifestError(f"baseline predictions row {number}: blank reason_code")
        for field in ("model_score", "applicable_threshold"):
            try:
                numeric = float(prediction[field])
            except ValueError as exc:
                raise ManifestError(
                    f"baseline predictions row {number}: {field} must be numeric"
                ) from exc
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ManifestError(
                    f"baseline predictions row {number}: {field} must be finite and in [0, 1]"
                )
        inspected_at = _timestamp(
            prediction["inspection_timestamp_utc"],
            f"baseline predictions row {number}.inspection_timestamp_utc",
        )
        inspected_times[key] = inspected_at
        if inspected_at < frozen_at:
            raise ManifestError(
                f"baseline predictions row {number}: baseline was not frozen before scoring"
            )
        if inspected_at > completed_at:
            raise ManifestError(
                f"baseline predictions row {number}: timestamp is after score receipt completion"
            )
        if inspected_at > development_start:
            raise ManifestError(
                f"baseline predictions row {number}: scoring occurred after development started"
            )
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    if missing or extra:
        raise ManifestError(
            "baseline predictions must cover exactly the train/validation targets; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    for key, prediction in actual.items():
        manifest_row = expected[key]
        if inspected_times[key] <= capture_times[key]:
            raise ManifestError(
                f"baseline prediction occurred before or at its corresponding capture for {key}"
            )
        occupancy = prediction["expected_occupancy"]
        if occupancy != manifest_row.expected_occupancy:
            raise ManifestError(f"baseline prediction expected_occupancy disagrees for {key}")
        visual = prediction["predicted_visual_class"]
        decision = prediction["inspection_decision"]
        if visual == "unknown":
            expected_decision = "UNSURE"
        elif visual in DEFECT_CLASSES:
            expected_decision = "DEFECT"
        elif visual == "pass":
            expected_decision = "PASS"
        else:
            expected_decision = "MISSED" if occupancy == "occupied" else "WAITING"
        if decision != expected_decision:
            raise ManifestError(
                f"baseline prediction has unsafe visual/decision pair for {key}: "
                f"{visual}+{decision}, expected {expected_decision}"
            )
        score = float(prediction["model_score"])
        threshold = float(prediction["applicable_threshold"])
        if visual in DEFECT_CLASSES and score < threshold:
            raise ManifestError(
                f"baseline defect prediction is below its applicable threshold for {key}"
            )
        if visual in {"pass", "no_vial"} and score >= threshold:
            raise ManifestError(
                f"baseline non-defect prediction is at/above its applicable threshold for {key}"
            )
    return len(predictions)


def validate_score_receipt(
    value: Mapping[str, object],
    *,
    baseline_id: str,
    baseline_freeze_sha256: str,
    predictions_sha256: str,
    decision_policy_version: str,
    decision_policy_sha256: str,
    frozen_at: datetime,
    development_start: datetime,
) -> datetime:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("baseline score receipt has an unsupported schema_version")
    if value.get("baseline_id") != baseline_id:
        raise ManifestError("baseline score receipt names a different baseline")
    if value.get("baseline_freeze_sha256") != baseline_freeze_sha256:
        raise ManifestError("baseline score receipt does not authenticate the freeze inventory")
    if value.get("predictions_sha256") != predictions_sha256:
        raise ManifestError("baseline score receipt does not authenticate predictions.csv")
    if value.get("decision_policy_version") != decision_policy_version:
        raise ManifestError("baseline score receipt names a different decision-policy version")
    if value.get("decision_policy_sha256") != decision_policy_sha256:
        raise ManifestError("baseline score receipt names a different decision-policy hash")
    if value.get("scored_before_real_adaptation") is not True:
        raise ManifestError("baseline score receipt must confirm pre-adaptation scoring")
    completed_at = _timestamp(
        value.get("completed_at_utc"), "baseline_score_receipt.completed_at_utc"
    )
    if completed_at < frozen_at:
        raise ManifestError("baseline score receipt predates the frozen baseline")
    if completed_at >= development_start:
        raise ManifestError("baseline scoring completed after real-data development started")
    return completed_at


def _development_counts(rows: Sequence[ManifestRow]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "usable_rows": sum(row.is_usable for row in rows),
        "events": len({row.event_uid for row in rows}),
        "sessions": len({row.session_id for row in rows}),
        "images": len({row.source_image_id for row in rows}),
        "classes": {
            name: sum(row.is_usable and row.visual_class == name for row in rows)
            for name in CANONICAL_CLASSES
        },
    }


def _protected_counts(rows: Sequence[ManifestRow]) -> dict[str, int]:
    # Deliberately no class/label counts or identifiers in this lock.
    return {
        "rows": len(rows),
        "events": len({row.event_uid for row in rows}),
        "sessions": len({row.session_id for row in rows}),
        "images": len({row.source_image_id for row in rows}),
    }


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_lock_pair_once(
    protected_path: Path,
    protected_bytes: bytes,
    development_path: Path,
    development_bytes: bytes,
) -> None:
    """Stage both locks, publish without overwrite, and roll back either on failure."""
    pairs = (
        (protected_path, protected_bytes),
        (development_path, development_bytes),
    )
    for path, _ in pairs:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to replace evidence lock {path}")

    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for target, data in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o444)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            staged.append((temporary, target))
        for temporary, target in staged:
            if os.path.lexists(target):
                raise FileExistsError(f"refusing to replace evidence lock {target}")
            os.link(temporary, target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def prepare_evidence(
    *,
    candidate_id: str,
    baseline_freeze_path: Path,
    baseline_predictions_path: Path,
    baseline_score_receipt_path: Path,
    authorization_path: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    protected_manifest_path: Path,
    development_start_utc: str,
    protected_sealed_at_utc: str,
    development_lock_out: Path,
    protected_lock_out: Path,
) -> tuple[Path, Path]:
    """Validate lifecycle evidence and write non-overlapping immutable locks."""
    candidate = _required_text(candidate_id, "candidate_id")
    if development_lock_out.resolve() == protected_lock_out.resolve():
        raise ManifestError("development and protected locks must have different paths")
    for output in (development_lock_out, protected_lock_out):
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to replace evidence lock {output}")

    development_start = _timestamp(development_start_utc, "development_start_utc")
    protected_sealed = _timestamp(protected_sealed_at_utc, "protected_sealed_at_utc")
    if protected_sealed >= development_start:
        raise ManifestError("the fresh protected test must be sealed before development starts")

    train_rows = _read_partition(train_manifest_path, "train", "development")
    validation_rows = _read_partition(
        validation_manifest_path, "validation", "development"
    )
    protected_rows = _read_partition(
        protected_manifest_path, "test", "protected_test"
    )
    validate_partitions(train_rows, validation_rows, protected_rows)
    train_capture_times = _capture_times(train_rows, "train")
    validation_capture_times = _capture_times(validation_rows, "validation")
    protected_capture_times = _capture_times(protected_rows, "protected_test")
    development_capture_times = {
        **train_capture_times,
        **validation_capture_times,
    }
    if any(captured_at > protected_sealed for captured_at in protected_capture_times.values()):
        raise ManifestError("a protected capture occurred after the protected test was sealed")
    all_rows = list(train_rows) + list(validation_rows) + list(protected_rows)
    fixtures = {row.fixture_id for row in all_rows}

    freeze = _load_object(baseline_freeze_path, "baseline freeze")
    (
        baseline_id,
        frozen_at,
        map_hashes,
        decision_policy_version,
        decision_policy_sha,
    ) = validate_baseline_freeze(freeze, fixtures)
    all_capture_times = {
        **development_capture_times,
        **protected_capture_times,
    }
    if any(frozen_at >= captured_at for captured_at in all_capture_times.values()):
        raise ManifestError(
            "the synthetic baseline must be frozen before every development and protected capture"
        )
    if candidate == baseline_id:
        raise ManifestError("candidate_id must differ from the frozen synthetic baseline_id")
    authorization = _load_object(authorization_path, "authorization record")
    authorized_at, authorized_uses = validate_authorization(
        authorization, development_start
    )

    freeze_sha = sha256_file(baseline_freeze_path)
    predictions_sha = sha256_file(baseline_predictions_path)
    receipt = _load_object(baseline_score_receipt_path, "baseline score receipt")
    score_completed = validate_score_receipt(
        receipt,
        baseline_id=baseline_id,
        baseline_freeze_sha256=freeze_sha,
        predictions_sha256=predictions_sha,
        decision_policy_version=decision_policy_version,
        decision_policy_sha256=decision_policy_sha,
        frozen_at=frozen_at,
        development_start=development_start,
    )
    prediction_count = validate_baseline_predictions(
        baseline_predictions_path,
        list(train_rows) + list(validation_rows),
        frozen_at=frozen_at,
        completed_at=score_completed,
        development_start=development_start,
        capture_times=development_capture_times,
    )

    protected_lock: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate,
        "designation": "protected_test",
        "sealed_at_utc": _utc_text(protected_sealed),
        "canonical_classes": list(CANONICAL_CLASSES),
        "manifest": {
            "sha256": sha256_file(protected_manifest_path),
            "counts_without_labels": _protected_counts(protected_rows),
        },
        "fixture_ids": sorted({row.fixture_id for row in protected_rows}),
        "disjointness_verified": {
            "events": True,
            "sessions_visits": True,
            "source_image_ids": True,
            "source_image_sha256": True,
        },
        "handling": (
            "audit/evaluation side only; never copy this lock, its manifest, labels, "
            "identifiers, or pixels into training or model-selection inputs"
        ),
    }
    protected_lock_bytes = _json_bytes(protected_lock)
    protected_lock_sha = hashlib.sha256(protected_lock_bytes).hexdigest()

    development_fixtures = {row.fixture_id for row in train_rows + validation_rows}
    development_lock: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate,
        "status": "development_inputs_locked",
        "canonical_classes": list(CANONICAL_CLASSES),
        "development_start_not_before_utc": _utc_text(development_start),
        "authorization": {
            "record_sha256": sha256_file(authorization_path),
            "authorized_at_utc": _utc_text(authorized_at),
            "staging_training_authorized": True,
            "allowed_uses": authorized_uses,
        },
        "synthetic_baseline": {
            "baseline_id": baseline_id,
            "freeze_sha256": freeze_sha,
            "frozen_at_utc": _utc_text(frozen_at),
            "score_receipt_sha256": sha256_file(baseline_score_receipt_path),
            "predictions_sha256": predictions_sha,
            "prediction_rows": prediction_count,
            "scoring_completed_at_utc": _utc_text(score_completed),
            "scored_before_real_adaptation": True,
            "decision_policy_version": decision_policy_version,
            "decision_policy_sha256": decision_policy_sha,
        },
        "development_manifests": {
            "train": {
                "sha256": sha256_file(train_manifest_path),
                "counts": _development_counts(train_rows),
            },
            "validation": {
                "sha256": sha256_file(validation_manifest_path),
                "counts": _development_counts(validation_rows),
            },
        },
        "fixture_map_sha256": {
            fixture: map_hashes[fixture] for fixture in sorted(development_fixtures)
        },
        "protected_reservation": {
            "lock_sha256": protected_lock_sha,
            "sealed_before_development": True,
            "different_session_visit_verified": True,
            "event_session_source_and_path_identifiers_included": False,
            "labels_included": False,
        },
        "policy": (
            "synthetic baseline scored staging development targets first; these real "
            "targets are now development evidence and cannot test this candidate; only "
            "the separately sealed fresh visit may provide final candidate evidence"
        ),
    }
    _write_lock_pair_once(
        protected_lock_out,
        protected_lock_bytes,
        development_lock_out,
        _json_bytes(development_lock),
    )
    return development_lock_out, protected_lock_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--baseline-freeze", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--baseline-score-receipt", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--protected-manifest", type=Path, required=True)
    parser.add_argument("--development-start-utc", required=True)
    parser.add_argument("--protected-sealed-at-utc", required=True)
    parser.add_argument("--development-lock-out", type=Path, required=True)
    parser.add_argument("--protected-lock-out", type=Path, required=True)
    args = parser.parse_args(argv)
    development, protected = prepare_evidence(
        candidate_id=args.candidate_id,
        baseline_freeze_path=args.baseline_freeze,
        baseline_predictions_path=args.baseline_predictions,
        baseline_score_receipt_path=args.baseline_score_receipt,
        authorization_path=args.authorization,
        train_manifest_path=args.train_manifest,
        validation_manifest_path=args.validation_manifest,
        protected_manifest_path=args.protected_manifest,
        development_start_utc=args.development_start_utc,
        protected_sealed_at_utc=args.protected_sealed_at_utc,
        development_lock_out=args.development_lock_out,
        protected_lock_out=args.protected_lock_out,
    )
    print(f"wrote development-only lock -> {development}")
    print(f"wrote separately handled protected reservation -> {protected}")
    print("protected pixels were not opened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
