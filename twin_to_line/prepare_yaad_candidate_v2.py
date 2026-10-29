#!/usr/bin/env python3
"""Authenticate policy-v2 candidate inputs without opening source images.

This is the fail-closed front door for a future, separately versioned YAAD
trainer and bundle builder.  It accepts only the development-side lock emitted
by ``prepare_retraining_evidence.py``.  The private protected-reservation lock,
protected manifest, protected identifiers, and all image bytes remain outside
this process.

The command deliberately stops at a metadata/schedule receipt.  It does not
copy images, train a model, submit a YAAD job, or alter the frozen staging5
trainer.  Once genuine lifecycle evidence exists, both the future bundle
builder and trainer must call :func:`authenticate_candidate_inputs` before
opening any development image.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import io
import os
from pathlib import Path
import re
import stat
import sys
from typing import Callable, Mapping, Sequence

from staging5_training_foundation import (
    AUGMENTATION_VERSION,
    SCHEDULE_VERSION,
    CandidateTrainAugmentationPolicy,
    build_balanced_epoch_schedule,
)
from staging_dataset import (
    CANONICAL_CLASSES,
    MANIFEST_COLUMNS,
    ManifestError,
    ManifestRow,
)


CONTRACT_VERSION = "staging5_policy_v2_candidate_training_contract_r1"
DEVELOPMENT_LOCK_SCHEMA = 1
AUTHORIZED_USES = frozenset({"baseline_scoring", "train", "validation"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
EXPECTED_POLICY = (
    "synthetic baseline scored staging development targets first; these real "
    "targets are now development evidence and cannot test this candidate; only "
    "the separately sealed fresh visit may provide final candidate evidence"
)

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "status",
        "canonical_classes",
        "development_start_not_before_utc",
        "authorization",
        "synthetic_baseline",
        "development_manifests",
        "fixture_map_sha256",
        "protected_reservation",
        "policy",
    }
)
AUTHORIZATION_KEYS = frozenset(
    {
        "record_sha256",
        "authorized_at_utc",
        "staging_training_authorized",
        "allowed_uses",
    }
)
BASELINE_KEYS = frozenset(
    {
        "baseline_id",
        "freeze_sha256",
        "frozen_at_utc",
        "score_receipt_sha256",
        "predictions_sha256",
        "prediction_rows",
        "scoring_completed_at_utc",
        "scored_before_real_adaptation",
        "decision_policy_version",
        "decision_policy_sha256",
    }
)
MANIFEST_KEYS = frozenset({"sha256", "counts"})
PROTECTED_RESERVATION_KEYS = frozenset(
    {
        "lock_sha256",
        "sealed_before_development",
        "different_session_visit_verified",
        "event_session_source_and_path_identifiers_included",
        "labels_included",
    }
)
MAX_LOCK_BYTES = 1 << 20
MAX_MANIFEST_BYTES = 64 << 20


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ManifestError(
            f"{field} has an unsafe or unsupported shape; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ManifestError(f"{field} must be a normalized nonempty string")
    return value


def _sha256(value: object, field: str) -> str:
    text = _text(value, field)
    if not SHA256_RE.fullmatch(text):
        raise ManifestError(f"{field} must be 64 lowercase hexadecimal digits")
    return text


def _timestamp(value: object, field: str) -> datetime:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    except ValueError as exc:
        raise ManifestError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ManifestError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build one JSON object while rejecting duplicate keys at every depth."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(
                "development_lock contains a duplicate JSON object key"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ManifestError(
        "development_lock contains a non-finite JSON numeric constant"
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_bytes(path: Path, field: str, *, max_bytes: int) -> bytes:
    """Snapshot one regular non-symlink file through a single descriptor.

    ``O_NOFOLLOW`` closes the normal symlink race on platforms that provide it.
    The fallback compares ``lstat`` with ``fstat``.  The descriptor is read
    once, and its identity/size/timestamps must remain stable for the complete
    read.  Every subsequent hash and parse uses only the returned immutable
    bytes, never the path again.
    """

    path = Path(path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    before_path: os.stat_result | None = None
    if not nofollow:
        try:
            before_path = os.lstat(path)
        except OSError as exc:
            raise ManifestError(f"{field} cannot be inspected: {path}") from exc
        if stat.S_ISLNK(before_path.st_mode):
            raise ManifestError(f"{field} must not be a symlink: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | nofollow
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(
            f"{field} must be an accessible regular non-symlink file: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ManifestError(f"{field} must be a regular file: {path}")
        if before_path is not None and (
            before_path.st_dev != opened.st_dev or before_path.st_ino != opened.st_ino
        ):
            raise ManifestError(f"{field} changed while it was opened: {path}")
        if opened.st_size > max_bytes:
            raise ManifestError(
                f"{field} exceeds the {max_bytes}-byte metadata limit: {path}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ManifestError(
                    f"{field} exceeds the {max_bytes}-byte metadata limit: {path}"
                )
        closed_snapshot = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(closed_snapshot):
            raise ManifestError(f"{field} changed while it was read: {path}")
        data = b"".join(chunks)
        if len(data) != opened.st_size:
            raise ManifestError(f"{field} size changed while it was read: {path}")
        return data
    finally:
        os.close(descriptor)


def _read_lock(path: Path) -> tuple[dict[str, object], bytes]:
    data = _read_regular_bytes(
        path, "development_lock", max_bytes=MAX_LOCK_BYTES
    )
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"development_lock is not readable JSON: {path}") from exc
    lock = _object(value, "development_lock")
    # Shape validation is deliberately first.  A legacy split lock includes
    # private test identifiers and must be rejected before either manifest is
    # opened or any caller can proceed to image access.
    _exact_keys(lock, TOP_LEVEL_KEYS, "development_lock")
    return lock, data


def _parse_manifest_bytes(data: bytes, field: str) -> list[ManifestRow]:
    """Parse and validate exactly one in-memory manifest snapshot."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{field} is not UTF-8") from exc
    if "\x00" in text:
        raise ManifestError(f"{field} contains a NUL byte")
    stream = io.StringIO(text, newline="")
    try:
        reader = csv.DictReader(stream, strict=True)
        fields = tuple(reader.fieldnames or ())
        if fields != MANIFEST_COLUMNS:
            raise ManifestError(
                f"{field} columns differ from the exact manifest contract; "
                f"expected={list(MANIFEST_COLUMNS)}, actual={list(fields)}"
            )
        rows: list[ManifestRow] = []
        for number, raw in enumerate(reader, start=2):
            # DictReader stores surplus cells under a None key and represents
            # missing trailing cells with None values.  Reject both before
            # ManifestRow selects its allowlisted columns; otherwise an extra
            # private identifier could be silently discarded while the gate
            # claims none was loaded.
            if None in raw or any(value is None for value in raw.values()):
                raise ManifestError(
                    f"{field} row {number} has the wrong number of CSV fields"
                )
            rows.append(ManifestRow.from_dict(raw, number))
    except csv.Error as exc:
        raise ManifestError(f"{field} contains malformed CSV") from exc
    if not rows:
        raise ManifestError(f"{field} is empty")
    return rows


def _validate_combined_manifest_integrity(
    train_rows: Sequence[ManifestRow],
    validation_rows: Sequence[ManifestRow],
    *,
    baseline_frozen_at: datetime,
    baseline_scored_at: datetime,
) -> None:
    """Enforce physical identity and leakage invariants across both snapshots."""

    rows = list(train_rows) + list(validation_rows)
    sample_ids: set[str] = set()
    event_positions: set[tuple[str, str, str]] = set()
    event_metadata: dict[str, tuple[str, str]] = {}
    physical_events: dict[tuple[str, str], str] = {}
    event_splits: dict[str, str] = {}
    session_splits: dict[str, str] = {}
    source_metadata: dict[str, tuple[str, str, str]] = {}
    path_metadata: dict[str, tuple[str, str, str]] = {}
    digest_events: dict[str, str] = {}
    source_capture_times: dict[str, datetime] = {}
    event_capture_times: dict[str, datetime] = {}
    event_fixture_sources: dict[tuple[str, str], str] = {}
    positions_by_event_fixture: dict[tuple[str, str], set[str]] = {}

    for row in rows:
        captured_at = _timestamp(row.capture_time, f"{row.sample_id}.capture_time")
        if not baseline_frozen_at < captured_at < baseline_scored_at:
            raise ManifestError(
                f"{row.sample_id}: capture_time must satisfy synthetic baseline "
                "freeze < capture < baseline scoring completion"
            )
        if row.sample_id in sample_ids:
            raise ManifestError(f"duplicate sample_id across candidate manifests: {row.sample_id!r}")
        sample_ids.add(row.sample_id)

        position_key = (row.event_uid, row.fixture_id, row.position_id)
        if position_key in event_positions:
            raise ManifestError(
                f"duplicate event/fixture/position across candidate manifests: {position_key!r}"
            )
        event_positions.add(position_key)

        event_meta = (row.session_id, row.event_id)
        prior_event_meta = event_metadata.setdefault(row.event_uid, event_meta)
        if prior_event_meta != event_meta:
            raise ManifestError(
                f"event_uid {row.event_uid!r} aliases inconsistent session/event "
                f"metadata: {prior_event_meta!r} and {event_meta!r}"
            )
        prior_uid = physical_events.setdefault(event_meta, row.event_uid)
        if prior_uid != row.event_uid:
            raise ManifestError(
                f"physical session/event {event_meta!r} aliases multiple event_uids: "
                f"{prior_uid!r} and {row.event_uid!r}"
            )

        prior_split = event_splits.setdefault(row.event_uid, row.split)
        if prior_split != row.split:
            raise ManifestError(
                f"event_uid {row.event_uid!r} crosses candidate splits"
            )
        prior_session_split = session_splits.setdefault(row.session_id, row.split)
        if prior_session_split != row.split:
            raise ManifestError(
                f"collection session/visit leakage across splits: {row.session_id!r}"
            )

        source_meta = (
            row.event_uid,
            row.source_image_relpath,
            row.source_image_sha256,
        )
        prior_source_meta = source_metadata.setdefault(row.source_image_id, source_meta)
        if prior_source_meta != source_meta:
            raise ManifestError(
                f"source_image_id {row.source_image_id!r} aliases inconsistent event/path/hash metadata"
            )
        path_meta = (row.source_image_id, row.event_uid, row.source_image_sha256)
        prior_path_meta = path_metadata.setdefault(row.source_image_relpath, path_meta)
        if prior_path_meta != path_meta:
            raise ManifestError(
                f"source_image_relpath {row.source_image_relpath!r} aliases inconsistent source/event/hash metadata"
            )
        prior_digest_event = digest_events.setdefault(
            row.source_image_sha256, row.event_uid
        )
        if prior_digest_event != row.event_uid:
            raise ManifestError(
                "duplicate source-image digest represents multiple physical events: "
                f"{prior_digest_event!r} and {row.event_uid!r}"
            )
        prior_source_time = source_capture_times.setdefault(
            row.source_image_id, captured_at
        )
        if prior_source_time != captured_at:
            raise ManifestError(
                f"source_image_id {row.source_image_id!r} has inconsistent capture_time values"
            )
        prior_event_time = event_capture_times.setdefault(row.event_uid, captured_at)
        if prior_event_time != captured_at:
            raise ManifestError(
                f"event_uid {row.event_uid!r} has inconsistent capture_time values"
            )

        fixture_key = (row.event_uid, row.fixture_id)
        prior_source = event_fixture_sources.setdefault(
            fixture_key, row.source_image_id
        )
        if prior_source != row.source_image_id:
            raise ManifestError(
                f"event/fixture {fixture_key!r} references multiple source images"
            )
        positions_by_event_fixture.setdefault(fixture_key, set()).add(row.position_id)

    for (event_uid, fixture), positions in positions_by_event_fixture.items():
        expected = {f"{fixture}{index:02d}" for index in range(1, 31)}
        if positions != expected:
            raise ManifestError(
                f"event {event_uid!r} fixture {fixture!r} is incomplete; "
                f"missing={sorted(expected - positions)}, extra={sorted(positions - expected)}"
            )


def _counts(rows: Sequence[ManifestRow]) -> dict[str, object]:
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


def _validate_manifest_entry(
    value: object, *, field: str, data: bytes
) -> dict[str, object]:
    entry = _object(value, field)
    _exact_keys(entry, MANIFEST_KEYS, field)
    expected = _sha256(entry.get("sha256"), f"{field}.sha256")
    if _sha256_bytes(data) != expected:
        raise ManifestError(f"{field} SHA-256 differs from development_lock")
    return entry


def _validate_class_support(
    train_rows: Sequence[ManifestRow], validation_rows: Sequence[ManifestRow]
) -> list[str]:
    train = _counts(train_rows)["classes"]
    validation = _counts(validation_rows)["classes"]
    assert isinstance(train, dict) and isinstance(validation, dict)
    unsupported = [
        name
        for name in CANONICAL_CLASSES
        if int(train[name]) + int(validation[name]) == 0
    ]
    illegal = set(unsupported) - {"damaged_vial"}
    if illegal:
        raise ManifestError(
            f"development inputs have no usable examples for required classes {sorted(illegal)}"
        )
    for name in CANONICAL_CLASSES:
        if name in unsupported:
            continue
        if int(train[name]) == 0 or int(validation[name]) == 0:
            raise ManifestError(
                f"class {name!r} is not independently represented in train and "
                f"validation (train={train[name]}, validation={validation[name]})"
            )
    return unsupported


def authenticate_candidate_inputs(
    *,
    development_lock_path: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    candidate_id: str | None = None,
    seed: int = 0,
    now_utc: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Return a metadata-only receipt for authorized candidate inputs.

    The development lock is fully authenticated before either manifest is
    opened. Manifest hashes and roles are then checked by parsing the exact
    immutable byte snapshots read through their original file descriptors.
    The event-balanced epoch-zero schedule is built from metadata only,
    proving that the future trainer can use the policy-v2 scheduler without
    touching the frozen trainer.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ManifestError("seed must be a non-negative integer")
    lock, lock_bytes = _read_lock(Path(development_lock_path))

    if lock.get("schema_version") != DEVELOPMENT_LOCK_SCHEMA:
        raise ManifestError("development_lock has an unsupported schema_version")
    locked_candidate = _text(lock.get("candidate_id"), "development_lock.candidate_id")
    if not CANDIDATE_ID_RE.fullmatch(locked_candidate):
        raise ManifestError(
            "development_lock.candidate_id must use lowercase letters, digits, '.', '_' or '-'"
        )
    if candidate_id is not None and candidate_id != locked_candidate:
        raise ManifestError("requested candidate_id differs from development_lock")
    if lock.get("status") != "development_inputs_locked":
        raise ManifestError("development_lock status is not development_inputs_locked")
    if tuple(lock.get("canonical_classes", ())) != CANONICAL_CLASSES:
        raise ManifestError("development_lock does not preserve the exact five-class taxonomy")
    if lock.get("policy") != EXPECTED_POLICY:
        raise ManifestError("development_lock policy is not the generated lifecycle policy")
    development_start = _timestamp(
        lock.get("development_start_not_before_utc"),
        "development_lock.development_start_not_before_utc",
    )
    clock = now_utc if now_utc is not None else lambda: datetime.now(timezone.utc)
    current_time = clock()
    if not isinstance(current_time, datetime):
        raise ManifestError("now_utc must return a datetime")
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ManifestError("now_utc must return a timezone-aware datetime")
    current_time = current_time.astimezone(timezone.utc)
    if current_time < development_start:
        raise ManifestError(
            "candidate development cannot start before "
            "development_start_not_before_utc"
        )

    authorization = _object(lock.get("authorization"), "development_lock.authorization")
    _exact_keys(authorization, AUTHORIZATION_KEYS, "development_lock.authorization")
    authorization_sha = _sha256(
        authorization.get("record_sha256"),
        "development_lock.authorization.record_sha256",
    )
    if authorization.get("staging_training_authorized") is not True:
        raise ManifestError("development_lock does not authorize staging-data training")
    uses = authorization.get("allowed_uses")
    if (
        not isinstance(uses, list)
        or not all(isinstance(item, str) for item in uses)
        or len(uses) != len(set(uses))
        or set(uses) != AUTHORIZED_USES
    ):
        raise ManifestError(
            "development_lock authorization must be limited to exactly "
            "baseline_scoring, train, and validation"
        )
    authorized_at = _timestamp(
        authorization.get("authorized_at_utc"),
        "development_lock.authorization.authorized_at_utc",
    )
    if authorized_at > development_start:
        raise ManifestError("staging-data authorization was recorded after development start")

    baseline = _object(
        lock.get("synthetic_baseline"), "development_lock.synthetic_baseline"
    )
    _exact_keys(baseline, BASELINE_KEYS, "development_lock.synthetic_baseline")
    baseline_id = _text(
        baseline.get("baseline_id"),
        "development_lock.synthetic_baseline.baseline_id",
    )
    if baseline_id == locked_candidate:
        raise ManifestError("candidate_id must differ from the frozen synthetic baseline")
    frozen_at = _timestamp(
        baseline.get("frozen_at_utc"),
        "development_lock.synthetic_baseline.frozen_at_utc",
    )
    scored_at = _timestamp(
        baseline.get("scoring_completed_at_utc"),
        "development_lock.synthetic_baseline.scoring_completed_at_utc",
    )
    if not frozen_at < scored_at < development_start:
        raise ManifestError(
            "development_lock chronology must satisfy baseline freeze < scoring completion < development start"
        )
    if baseline.get("scored_before_real_adaptation") is not True:
        raise ManifestError("baseline was not attested as scored before real adaptation")
    prediction_rows = baseline.get("prediction_rows")
    if isinstance(prediction_rows, bool) or not isinstance(prediction_rows, int) or prediction_rows < 1:
        raise ManifestError("development_lock.synthetic_baseline.prediction_rows must be positive")
    for name in (
        "freeze_sha256",
        "score_receipt_sha256",
        "predictions_sha256",
        "decision_policy_sha256",
    ):
        _sha256(
            baseline.get(name),
            f"development_lock.synthetic_baseline.{name}",
        )
    decision_policy_version = _text(
        baseline.get("decision_policy_version"),
        "development_lock.synthetic_baseline.decision_policy_version",
    )

    reservation = _object(
        lock.get("protected_reservation"),
        "development_lock.protected_reservation",
    )
    _exact_keys(
        reservation,
        PROTECTED_RESERVATION_KEYS,
        "development_lock.protected_reservation",
    )
    reservation_sha = _sha256(
        reservation.get("lock_sha256"),
        "development_lock.protected_reservation.lock_sha256",
    )
    required_reservation_values = {
        "sealed_before_development": True,
        "different_session_visit_verified": True,
        "event_session_source_and_path_identifiers_included": False,
        "labels_included": False,
    }
    for name, expected in required_reservation_values.items():
        if reservation.get(name) is not expected:
            raise ManifestError(
                f"development_lock.protected_reservation.{name} must be {expected}"
            )

    maps = _object(
        lock.get("fixture_map_sha256"), "development_lock.fixture_map_sha256"
    )
    if not maps or set(maps) - {"G", "S"}:
        raise ManifestError("development_lock has an invalid fixture-map inventory")
    map_hashes = {
        fixture: _sha256(value, f"development_lock.fixture_map_sha256.{fixture}")
        for fixture, value in maps.items()
    }
    if set(map_hashes) == {"G", "S"} and map_hashes["G"] == map_hashes["S"]:
        raise ManifestError("fixture S cannot reuse fixture G's calibrated map hash")

    manifests = _object(
        lock.get("development_manifests"),
        "development_lock.development_manifests",
    )
    _exact_keys(
        manifests,
        frozenset({"train", "validation"}),
        "development_lock.development_manifests",
    )
    train_bytes = _read_regular_bytes(
        Path(train_manifest_path),
        "development_lock.development_manifests.train",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    validation_bytes = _read_regular_bytes(
        Path(validation_manifest_path),
        "development_lock.development_manifests.validation",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    train_entry = _validate_manifest_entry(
        manifests.get("train"),
        field="development_lock.development_manifests.train",
        data=train_bytes,
    )
    validation_entry = _validate_manifest_entry(
        manifests.get("validation"),
        field="development_lock.development_manifests.validation",
        data=validation_bytes,
    )

    # Metadata only: image_root is intentionally absent and image verification
    # is explicitly disabled.  The future bundler may verify development image
    # bytes only after this function returns successfully.
    train_rows = _parse_manifest_bytes(train_bytes, "train_manifest")
    validation_rows = _parse_manifest_bytes(
        validation_bytes, "validation_manifest"
    )
    if {row.split for row in train_rows} != {"train"}:
        raise ManifestError("train manifest contains a non-train row")
    if {row.split for row in validation_rows} != {"validation"}:
        raise ManifestError("validation manifest contains a non-validation row")
    if {row.designation for row in train_rows + validation_rows} != {"development"}:
        raise ManifestError("candidate manifests must contain development rows only")
    _validate_combined_manifest_integrity(
        train_rows,
        validation_rows,
        baseline_frozen_at=frozen_at,
        baseline_scored_at=scored_at,
    )
    if _counts(train_rows) != train_entry.get("counts"):
        raise ManifestError("train manifest counts differ from development_lock")
    if _counts(validation_rows) != validation_entry.get("counts"):
        raise ManifestError("validation manifest counts differ from development_lock")
    if int(prediction_rows) != len(train_rows) + len(validation_rows):
        raise ManifestError("baseline prediction coverage differs from development manifests")
    fixtures = {row.fixture_id for row in train_rows + validation_rows}
    if set(map_hashes) != fixtures:
        raise ManifestError(
            "fixture-map inventory differs from development manifests "
            f"(maps={sorted(map_hashes)}, fixtures={sorted(fixtures)})"
        )

    unsupported = _validate_class_support(train_rows, validation_rows)
    train_counts = _counts(train_rows)
    validation_counts = _counts(validation_rows)
    schedule = build_balanced_epoch_schedule(train_rows, seed=seed, epoch=0)
    augmentation = CandidateTrainAugmentationPolicy()
    receipt: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "candidate_id": locked_candidate,
        "canonical_classes": list(CANONICAL_CLASSES),
        "development_lock_sha256": _sha256_bytes(lock_bytes),
        "authorization_record_sha256": authorization_sha,
        "protected_reservation_lock_sha256": reservation_sha,
        "protected_identifiers_loaded": False,
        "protected_labels_loaded": False,
        "protected_pixels_loaded": False,
        "development_image_pixels_loaded": False,
        "fixture_map_sha256": dict(sorted(map_hashes.items())),
        "manifests": {
            "train_sha256": _sha256_bytes(train_bytes),
            "validation_sha256": _sha256_bytes(validation_bytes),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "train_events": len({row.event_uid for row in train_rows}),
            "validation_events": len({row.event_uid for row in validation_rows}),
            "train_sessions": len({row.session_id for row in train_rows}),
            "validation_sessions": len({row.session_id for row in validation_rows}),
            "train_class_counts": dict(train_counts["classes"]),
            "validation_class_counts": dict(validation_counts["classes"]),
        },
        "synthetic_baseline": {
            "baseline_id": baseline_id,
            "decision_policy_version": decision_policy_version,
            "scored_before_real_adaptation": True,
            "prediction_rows": prediction_rows,
        },
        "unsupported_classes": unsupported,
        "schedule": {
            "version": SCHEDULE_VERSION,
            "epoch": 0,
            "seed": seed,
            "sha256": schedule.sha256,
            "samples_per_class": schedule.samples_per_class,
            "class_counts": dict(schedule.class_counts),
            "distinct_event_support": dict(schedule.distinct_event_support),
        },
        "augmentation": {
            "version": AUGMENTATION_VERSION,
            "identity_only": True,
            "sha256": augmentation.sha256,
            "development_attestation_sha256": None,
        },
        "metadata_gate_passed": True,
        "ready_for_development_byte_verification": True,
        "next_required_gate": (
            "verify authorized development image bytes and calibrated fixture-map "
            "bytes against the authenticated manifest/map hashes before bundle "
            "construction or training"
        ),
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--development-lock", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        receipt = authenticate_candidate_inputs(
            development_lock_path=args.development_lock,
            train_manifest_path=args.train_manifest,
            validation_manifest_path=args.validation_manifest,
            candidate_id=args.candidate_id,
            seed=args.seed,
        )
    except (ManifestError, OSError, ValueError) as exc:
        print(f"POLICY-V2 CANDIDATE REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
