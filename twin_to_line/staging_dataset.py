"""Validated five-class staging data and protected event-level splits.

This module is deliberately separate from ``train_resnet.py``.  That file and
its inspector are frozen four-class, synthetic-only artifacts.  Real staging
data must enter through the manifest contract below so a protected test event
can never be decoded by training code by accident.

Manifest rows represent one fixture position in one physical placement event.
All positions belonging to an event are assigned to the same split.  Events
from the same collection session/visit are also kept together so adjacent
captures, retakes and shared acquisition conditions cannot leak across splits.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

import frame_warp as FW
import positions as P
from train_resnet import to_tensor_array


CANONICAL_CLASSES = (
    "pass",
    "missing_cap",
    "crooked_cap",
    "damaged_vial",
    "no_vial",
)
DEFECT_CLASSES = ("missing_cap", "crooked_cap", "damaged_vial")
SPLITS = ("train", "validation", "test")
DESIGNATIONS = ("development", "protected_test")
EXPECTED_OCCUPANCY = ("occupied", "empty")
FIXTURES = ("G", "S")
POSITION_RE = re.compile(r"^([GS])(0[1-9]|[12][0-9]|30)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


MANIFEST_COLUMNS = (
    "sample_id",
    "event_uid",
    "session_id",
    "event_id",
    "source_image_id",
    "source_image_relpath",
    "source_image_sha256",
    "capture_time",
    "fixture_id",
    "position_id",
    "expected_occupancy",
    "visual_class",
    "inspection_mode",
    "gt_sequence",
    "designation",
    "usable",
    "split",
    "split_reason",
    "label_source",
    "annotator_id",
    "annotation_confidence",
    "review_state",
    "notes",
)


class ManifestError(ValueError):
    """The staging manifest violates a data-boundary or taxonomy rule."""


class ProtectedTestError(PermissionError):
    """Training code attempted to read protected test data."""


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _truth(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ManifestError(f"usable must be true or false, got {value!r}")


def _within(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ManifestError(f"image path escapes image root: {relative!r}") from exc
    return candidate


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    event_uid: str
    session_id: str
    event_id: str
    source_image_id: str
    source_image_relpath: str
    source_image_sha256: str
    capture_time: str
    fixture_id: str
    position_id: str
    expected_occupancy: str
    visual_class: str
    inspection_mode: str
    gt_sequence: str
    designation: str
    usable: str
    split: str
    split_reason: str
    label_source: str
    annotator_id: str
    annotation_confidence: str
    review_state: str
    notes: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, str], row_number: int) -> "ManifestRow":
        missing = [name for name in MANIFEST_COLUMNS if name not in raw]
        if missing:
            raise ManifestError(f"manifest row {row_number}: missing columns {missing}")
        values = {name: str(raw.get(name, "")).strip() for name in MANIFEST_COLUMNS}
        row = cls(**values)
        row.validate(row_number)
        return row

    def validate(self, row_number: int | None = None) -> None:
        where = f"manifest row {row_number}" if row_number is not None else self.sample_id
        required = {
            "sample_id": self.sample_id,
            "event_uid": self.event_uid,
            "session_id": self.session_id,
            "event_id": self.event_id,
            "source_image_id": self.source_image_id,
            "source_image_relpath": self.source_image_relpath,
            "source_image_sha256": self.source_image_sha256,
            "capture_time": self.capture_time,
            "fixture_id": self.fixture_id,
            "position_id": self.position_id,
            "expected_occupancy": self.expected_occupancy,
            "visual_class": self.visual_class,
            "designation": self.designation,
            "label_source": self.label_source,
            "annotator_id": self.annotator_id,
            "annotation_confidence": self.annotation_confidence,
            "review_state": self.review_state,
        }
        blank = [name for name, value in required.items() if not value]
        if blank:
            raise ManifestError(f"{where}: blank required fields {blank}")
        if self.visual_class not in CANONICAL_CLASSES:
            raise ManifestError(
                f"{where}: visual_class {self.visual_class!r} is not in the exact five-class taxonomy"
            )
        if self.fixture_id not in FIXTURES:
            raise ManifestError(f"{where}: fixture_id must be G or S")
        match = POSITION_RE.fullmatch(self.position_id)
        if not match or match.group(1) != self.fixture_id:
            raise ManifestError(
                f"{where}: position_id {self.position_id!r} does not belong to fixture {self.fixture_id}"
            )
        if self.expected_occupancy not in EXPECTED_OCCUPANCY:
            raise ManifestError(
                f"{where}: expected_occupancy must be one of {EXPECTED_OCCUPANCY}"
            )
        if self.visual_class != "no_vial" and self.expected_occupancy != "occupied":
            raise ManifestError(
                f"{where}: {self.visual_class} requires expected_occupancy=occupied"
            )
        if self.gt_sequence and self.visual_class != "no_vial":
            raise ManifestError(f"{where}: gt_sequence is only valid for no_vial")
        if self.gt_sequence not in {"", "missed", "waiting"}:
            raise ManifestError(f"{where}: invalid gt_sequence {self.gt_sequence!r}")
        if self.visual_class == "no_vial":
            expected = "missed" if self.expected_occupancy == "occupied" else "waiting"
            if self.gt_sequence != expected:
                raise ManifestError(
                    f"{where}: no_vial with expected_occupancy={self.expected_occupancy} "
                    f"requires gt_sequence={expected}"
                )
        if self.designation not in DESIGNATIONS:
            raise ManifestError(f"{where}: designation must be one of {DESIGNATIONS}")
        _truth(self.usable)
        try:
            confidence = float(self.annotation_confidence)
        except ValueError as exc:
            raise ManifestError(
                f"{where}: annotation_confidence must be a number in [0, 1]"
            ) from exc
        if not 0.0 <= confidence <= 1.0:
            raise ManifestError(
                f"{where}: annotation_confidence must be a number in [0, 1]"
            )
        if self.review_state not in {"adjudicated", "protocol_reference"}:
            raise ManifestError(
                f"{where}: review_state must be 'adjudicated' or 'protocol_reference'"
            )
        if self.split and self.split not in SPLITS:
            raise ManifestError(f"{where}: split must be blank or one of {SPLITS}")
        if self.designation == "protected_test" and self.split not in {"", "test"}:
            raise ProtectedTestError(
                f"{where}: protected_test data cannot be assigned to {self.split}"
            )
        if not SHA256_RE.fullmatch(self.source_image_sha256):
            raise ManifestError(f"{where}: source_image_sha256 must be 64 lowercase hex digits")
        if Path(self.source_image_relpath).is_absolute():
            raise ManifestError(f"{where}: source_image_relpath must be relative")

    @property
    def is_usable(self) -> bool:
        return _truth(self.usable)

    def as_csv_dict(self) -> dict[str, str]:
        return {name: str(asdict(self)[name]) for name in MANIFEST_COLUMNS}


def read_manifest(
    path: os.PathLike[str] | str,
    *,
    image_root: os.PathLike[str] | str | None = None,
    verify_images: bool = True,
    require_complete_events: bool = True,
) -> list[ManifestRow]:
    """Read and validate a manifest without weakening protected-test roles."""
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in MANIFEST_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"{path}: missing columns {missing}")
        rows = [ManifestRow.from_dict(raw, number) for number, raw in enumerate(reader, 2)]
    if not rows:
        raise ManifestError(f"{path}: manifest is empty")

    sample_ids: set[str] = set()
    positions: set[tuple[str, str, str]] = set()
    event_roles: dict[str, set[str]] = {}
    event_sessions: dict[str, set[str]] = {}
    event_splits: dict[str, set[str]] = {}
    event_sources: dict[tuple[str, str], set[str]] = {}
    source_event: dict[str, str] = {}
    source_meta: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row.sample_id in sample_ids:
            raise ManifestError(f"duplicate sample_id {row.sample_id}")
        sample_ids.add(row.sample_id)
        position_key = (row.event_uid, row.fixture_id, row.position_id)
        if position_key in positions:
            raise ManifestError(f"duplicate event/fixture/position {position_key}")
        positions.add(position_key)
        event_roles.setdefault(row.event_uid, set()).add(row.designation)
        event_sessions.setdefault(row.event_uid, set()).add(row.session_id)
        if row.split:
            event_splits.setdefault(row.event_uid, set()).add(row.split)
        event_sources.setdefault((row.event_uid, row.fixture_id), set()).add(row.source_image_id)
        prior_event = source_event.setdefault(row.source_image_id, row.event_uid)
        if prior_event != row.event_uid:
            raise ManifestError(
                f"source_image_id {row.source_image_id!r} appears in multiple events: "
                f"{prior_event!r}, {row.event_uid!r}"
            )
        prior_meta = source_meta.setdefault(
            row.source_image_id, (row.source_image_relpath, row.source_image_sha256)
        )
        if prior_meta != (row.source_image_relpath, row.source_image_sha256):
            raise ManifestError(f"inconsistent path/hash for source_image_id {row.source_image_id}")

    for event_uid, roles in event_roles.items():
        if len(roles) != 1:
            raise ManifestError(f"event {event_uid}: mixed designations {sorted(roles)}")
    for event_uid, sessions in event_sessions.items():
        if len(sessions) != 1:
            raise ManifestError(f"event {event_uid}: mixed session IDs {sorted(sessions)}")
    for event_uid, splits in event_splits.items():
        if len(splits) != 1:
            raise ManifestError(f"event {event_uid}: positions cross splits {sorted(splits)}")
    for key, sources in event_sources.items():
        if len(sources) != 1:
            raise ManifestError(f"event/fixture {key}: positions reference multiple images {sorted(sources)}")

    if require_complete_events:
        grouped: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            grouped.setdefault((row.event_uid, row.fixture_id), set()).add(row.position_id)
        for (event_uid, fixture), actual in grouped.items():
            expected = {f"{fixture}{index:02d}" for index in range(1, 31)}
            if actual != expected:
                raise ManifestError(
                    f"event {event_uid} fixture {fixture}: expected all 30 positions; "
                    f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
                )

    if verify_images:
        if image_root is None:
            raise ManifestError("image_root is required when verify_images=True")
        root = Path(image_root).resolve()
        digest_event: dict[str, str] = {}
        for source_id, (relative, expected_digest) in sorted(source_meta.items()):
            image_path = _within(root, relative)
            if not image_path.is_file() or image_path.is_symlink():
                raise ManifestError(f"{source_id}: missing or symlinked source image {image_path}")
            actual_digest = sha256_file(image_path)
            if actual_digest != expected_digest:
                raise ManifestError(
                    f"{source_id}: image SHA-256 mismatch ({actual_digest} != {expected_digest})"
                )
            event_uid = source_event[source_id]
            prior_event = digest_event.setdefault(actual_digest, event_uid)
            if prior_event != event_uid:
                raise ManifestError(
                    f"byte-identical image is assigned to multiple events: {prior_event}, {event_uid}"
                )
    return rows


def _event_rows(rows: Iterable[ManifestRow]) -> dict[str, list[ManifestRow]]:
    grouped: dict[str, list[ManifestRow]] = {}
    for row in rows:
        grouped.setdefault(row.event_uid, []).append(row)
    return grouped


def _stable_key(seed: int, salt: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{salt}:{value}".encode("utf-8")).hexdigest()


def _balanced_subset(
    candidates: Sequence[str],
    grouped: Mapping[str, Sequence[ManifestRow]],
    count: int,
    seed: int,
    salt: str,
) -> set[str]:
    """Deterministic greedy multilabel balance over indivisible split groups."""
    if count <= 0:
        return set()
    if count > len(candidates):
        raise ManifestError(
            f"cannot choose {count} split groups from {len(candidates)} candidates"
        )
    presence = {
        event: {row.visual_class for row in grouped[event] if row.is_usable}
        for event in candidates
    }
    totals = {name: sum(name in presence[event] for event in candidates) for name in CANONICAL_CLASSES}
    targets = {name: totals[name] * count / len(candidates) for name in CANONICAL_CLASSES}
    selected: set[str] = set()
    current = {name: 0 for name in CANONICAL_CLASSES}
    while len(selected) < count:
        scored: list[tuple[float, str, str]] = []
        for event in candidates:
            if event in selected:
                continue
            after = {
                name: current[name] + int(name in presence[event])
                for name in CANONICAL_CLASSES
            }
            score = sum(
                ((after[name] - targets[name]) ** 2) / (targets[name] + 0.25)
                for name in CANONICAL_CLASSES
            )
            scored.append((score, _stable_key(seed, salt, event), event))
        _, _, chosen = min(scored)
        selected.add(chosen)
        for name in presence[chosen]:
            current[name] += 1
    return selected


def assign_event_splits(
    rows: Sequence[ManifestRow],
    *,
    seed: int,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    explicit_test_events: Iterable[str] = (),
    explicit_validation_events: Iterable[str] = (),
    split_unit: str = "session",
) -> tuple[list[ManifestRow], dict[str, str]]:
    """Lock each physical event and collection session into exactly one split.

    ``protected_test`` events are always TEST.  They are never eligible for the
    balancing algorithm, even if that leaves no trainable data.  The safe
    default is ``split_unit=session``: a protected event protects its entire
    visit because captures share acquisition conditions.  ``split_unit=event``
    is an explicit within-visit development protocol; its evaluation is not an
    independent estimate of performance on a new visit.
    """
    if not (0 <= validation_fraction < 1 and 0 <= test_fraction < 1):
        raise ManifestError("validation_fraction and test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ManifestError("validation_fraction + test_fraction must be less than 1")
    if split_unit not in {"session", "event"}:
        raise ManifestError("split_unit must be 'session' or 'event'")
    grouped = _event_rows(rows)
    event_session: dict[str, str] = {}
    session_rows: dict[str, list[ManifestRow]] = {}
    for event, event_rows in grouped.items():
        sessions = {row.session_id for row in event_rows}
        if len(sessions) != 1:
            raise ManifestError(f"event {event}: mixed session IDs {sorted(sessions)}")
        session = next(iter(sessions))
        event_session[event] = session
        session_rows.setdefault(session, []).extend(event_rows)
    split_rows = session_rows if split_unit == "session" else grouped
    event_group = event_session if split_unit == "session" else {event: event for event in grouped}
    if len(split_rows) < 3:
        group_name = (
            "collection sessions/visits" if split_unit == "session" else "physical events"
        )
        raise ManifestError(
            f"at least three independent {group_name} are required "
            "for train/validation/test"
        )

    all_events = set(grouped)
    protected_events = {
        event
        for event, event_rows in grouped.items()
        if {row.designation for row in event_rows} == {"protected_test"}
    }
    requested_test = set(explicit_test_events)
    requested_validation = set(explicit_validation_events)
    unknown = (requested_test | requested_validation) - all_events
    if unknown:
        raise ManifestError(f"explicit split references unknown events {sorted(unknown)}")
    protected_groups = {event_group[event] for event in protected_events}
    requested_test_groups = {event_group[event] for event in requested_test}
    requested_validation_groups = {event_group[event] for event in requested_validation}
    if requested_validation_groups & (protected_groups | requested_test_groups):
        raise ProtectedTestError("a protected/explicit test event was also requested for validation")

    group_assignment: dict[str, str] = {
        group: "test" for group in protected_groups | requested_test_groups
    }
    group_assignment.update(
        {group: "validation" for group in requested_validation_groups}
    )
    development_groups = sorted(
        group
        for group, group_data in split_rows.items()
        if group not in group_assignment
        and {row.designation for row in group_data} == {"development"}
    )
    non_development = set(split_rows) - set(development_groups) - set(group_assignment)
    if non_development:
        raise ManifestError(
            f"{split_unit} groups have no uniform development/test role: "
            f"{sorted(non_development)}"
        )

    target_test = max(1, int(round(len(split_rows) * test_fraction))) if test_fraction else 0
    extra_test_count = max(
        0, target_test - sum(value == "test" for value in group_assignment.values())
    )
    available_for_test = [
        group
        for group in development_groups
        if group not in requested_validation_groups
    ]
    auto_test = _balanced_subset(
        available_for_test, split_rows, extra_test_count, seed, "test"
    )
    for group in auto_test:
        group_assignment[group] = "test"

    remaining = [
        group for group in development_groups if group not in group_assignment
    ]
    target_validation = (
        max(1, int(round(len(split_rows) * validation_fraction)))
        if validation_fraction
        else 0
    )
    current_validation = sum(
        value == "validation" for value in group_assignment.values()
    )
    auto_validation_count = max(0, target_validation - current_validation)
    if auto_validation_count >= len(remaining):
        raise ManifestError("split would leave no training event")
    auto_validation = _balanced_subset(
        remaining, split_rows, auto_validation_count, seed, "validation"
    )
    for group in auto_validation:
        group_assignment[group] = "validation"
    for group in split_rows:
        group_assignment.setdefault(group, "train")

    assignment = {
        event: group_assignment[event_group[event]] for event in all_events
    }

    if set(assignment.values()) != set(SPLITS):
        raise ManifestError(
            f"all three splits are required; produced {sorted(set(assignment.values()))}"
        )
    assigned_rows = [
        replace(
            row,
            split=assignment[row.event_uid],
            split_reason=(
                f"protected source {split_unit} designation"
                if event_group[row.event_uid] in protected_groups
                else f"explicit assignment expanded to {split_unit} split group"
                if event_group[row.event_uid]
                in requested_test_groups | requested_validation_groups
                else f"deterministic {split_unit}-level balance seed={seed}"
            ),
        )
        for row in rows
    ]
    validate_split_integrity(
        assigned_rows, allow_session_overlap=split_unit == "event"
    )
    return assigned_rows, assignment


def validate_split_integrity(
    rows: Sequence[ManifestRow], *, allow_session_overlap: bool = False
) -> None:
    grouped = _event_rows(rows)
    session_splits: dict[str, set[str]] = {}
    source_splits: dict[str, set[str]] = {}
    digest_splits: dict[str, set[str]] = {}
    for event, event_rows in grouped.items():
        splits = {row.split for row in event_rows}
        if len(splits) != 1 or "" in splits:
            raise ManifestError(f"event {event} crosses splits or is unassigned: {sorted(splits)}")
        split = next(iter(splits))
        for row in event_rows:
            session_splits.setdefault(row.session_id, set()).add(split)
        if any(row.designation == "protected_test" and split != "test" for row in event_rows):
            raise ProtectedTestError(f"protected event {event} was assigned to {split}")
        for row in event_rows:
            source_splits.setdefault(row.source_image_id, set()).add(split)
            digest_splits.setdefault(row.source_image_sha256, set()).add(split)
    crossed_sources = {key: value for key, value in source_splits.items() if len(value) > 1}
    crossed_digests = {key: value for key, value in digest_splits.items() if len(value) > 1}
    if crossed_sources or crossed_digests:
        raise ManifestError(
            f"image leakage across splits: source_ids={crossed_sources}, sha256={crossed_digests}"
        )
    crossed_sessions = {
        session: splits for session, splits in session_splits.items() if len(splits) > 1
    }
    if crossed_sessions and not allow_session_overlap:
        raise ManifestError(
            f"collection session/visit leakage across splits: {crossed_sessions}"
        )


def _write_manifest(path: Path, rows: Sequence[ManifestRow]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(row.as_csv_dict() for row in rows)


def _counts(rows: Sequence[ManifestRow]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "usable_rows": sum(row.is_usable for row in rows),
        "events": len({row.event_uid for row in rows}),
        "images": len({row.source_image_id for row in rows}),
        "classes": {
            name: sum(row.is_usable and row.visual_class == name for row in rows)
            for name in CANONICAL_CLASSES
        },
    }


def write_split_bundle(
    rows: Sequence[ManifestRow],
    assignment: Mapping[str, str],
    out_dir: os.PathLike[str] | str,
    *,
    source_manifest: os.PathLike[str] | str,
    image_root: os.PathLike[str] | str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    source_repo_url: str = "",
    source_git_commit: str = "",
    split_unit: str = "session",
    acknowledge_within_visit_evaluation: bool = False,
    leakage_justification: str = "",
) -> Path:
    """Create immutable-by-default split manifests and a provenance lock."""
    if split_unit not in {"session", "event"}:
        raise ManifestError("split_unit must be 'session' or 'event'")
    session_splits: dict[str, set[str]] = {}
    for row in rows:
        session_splits.setdefault(row.session_id, set()).add(row.split)
    crossed_sessions = sorted(
        session for session, splits in session_splits.items() if len(splits) > 1
    )
    if crossed_sessions and split_unit != "event":
        raise ManifestError("session overlap requires split_unit='event'")
    if crossed_sessions and not acknowledge_within_visit_evaluation:
        raise ManifestError(
            "event-level splitting crosses a collection visit; explicitly acknowledge "
            "that resulting test metrics are within-visit and not independent"
        )
    if crossed_sessions and not leakage_justification.strip():
        raise ManifestError(
            "within-visit event splitting requires a nonempty leakage justification"
        )
    validate_split_integrity(rows, allow_session_overlap=split_unit == "event")
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"refusing to replace existing split bundle: {out}")
    out.mkdir(parents=True)
    protected = out / "protected_test"
    protected.mkdir()
    paths = {
        "train": out / "train_manifest.csv",
        "validation": out / "validation_manifest.csv",
        "test": protected / "test_manifest.csv",
    }
    split_rows = {
        name: sorted(
            (row for row in rows if row.split == name),
            key=lambda row: (
                row.event_uid,
                row.fixture_id,
                row.position_id,
                row.source_image_id,
                row.sample_id,
            ),
        )
        for name in SPLITS
    }
    for name in SPLITS:
        _write_manifest(paths[name], split_rows[name])
    lock = {
        "schema_version": 2,
        "policy": (
            "physical event is indivisible; protected_test is never train/validation; "
            + (
                "collection session/visit is indivisible"
                if split_unit == "session"
                else "event split explicitly permits within-visit acquisition overlap"
            )
        ),
        "split_unit": split_unit,
        "evaluation_scope": (
            "within_visit_event_holdout_not_independent"
            if crossed_sessions
            else "cross_visit_holdout"
        ),
        "independent_visit_test": not bool(crossed_sessions),
        "within_visit_evaluation_acknowledged": bool(
            acknowledge_within_visit_evaluation
        ),
        "leakage_justification": leakage_justification.strip(),
        "metric_warning": (
            "held-out setup-event metrics are not independent deployment accuracy: "
            "they share visit, camera, lighting and fixture conditions with training"
            if crossed_sessions
            else "none"
        ),
        "sessions_crossing_splits": crossed_sessions,
        "canonical_classes": list(CANONICAL_CLASSES),
        "defect_classes": list(DEFECT_CLASSES),
        "seed": seed,
        "fractions": {"validation": validation_fraction, "test": test_fraction},
        "source": {
            "repo_url": source_repo_url,
            "git_commit": source_git_commit,
            "manifest": str(Path(source_manifest).resolve()),
            "manifest_sha256": sha256_file(source_manifest),
            "image_root": str(Path(image_root).resolve()),
        },
        "events": {
            name: sorted(event for event, split in assignment.items() if split == name)
            for name in SPLITS
        },
        "sessions": {
            name: sorted(
                {
                    row.session_id
                    for row in rows
                    if assignment[row.event_uid] == name
                }
            )
            for name in SPLITS
        },
        "counts": {name: _counts(split_rows[name]) for name in SPLITS},
        "manifests": {
            name: {
                "path": str(paths[name].relative_to(out)),
                "sha256": sha256_file(paths[name]),
            }
            for name in SPLITS
        },
        "test_access": "test manifest is for final evaluation only; training must report test_rows_loaded=0",
    }
    lock_path = out / "split_lock.json"
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.chmod(paths["test"], 0o444)
    os.chmod(lock_path, 0o444)
    return lock_path


def _validated_split_lock(split_lock: os.PathLike[str] | str) -> dict[str, object]:
    lock = json.load(open(split_lock, encoding="utf-8"))
    if tuple(lock.get("canonical_classes", ())) != CANONICAL_CLASSES:
        raise ManifestError("split lock does not preserve the exact five-class taxonomy")
    if lock.get("split_unit") not in {"session", "event"}:
        raise ManifestError("split lock has an invalid or missing split_unit")
    if lock.get("sessions_crossing_splits"):
        if lock.get("split_unit") != "event":
            raise ManifestError("split lock crosses sessions without event split mode")
        if not lock.get("within_visit_evaluation_acknowledged"):
            raise ManifestError("split lock lacks within-visit acknowledgment")
        if not str(lock.get("leakage_justification", "")).strip():
            raise ManifestError("split lock lacks within-visit leakage justification")
    return lock


def verify_training_manifests(
    train_manifest: os.PathLike[str] | str,
    validation_manifest: os.PathLike[str] | str,
    split_lock: os.PathLike[str] | str,
) -> dict[str, object]:
    lock = _validated_split_lock(split_lock)
    base = Path(split_lock).resolve().parent
    supplied = {"train": Path(train_manifest).resolve(), "validation": Path(validation_manifest).resolve()}
    for name, path in supplied.items():
        entry = lock.get("manifests", {}).get(name, {})
        expected_path = (base / str(entry.get("path", ""))).resolve()
        if path != expected_path:
            raise ManifestError(f"{name} manifest is not the file recorded by split_lock.json")
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            raise ManifestError(f"{name} manifest SHA-256 differs from split_lock.json")
    return lock


def verify_test_manifest(
    test_manifest: os.PathLike[str] | str,
    split_lock: os.PathLike[str] | str,
) -> dict[str, object]:
    """Verify that a protected-test manifest is exactly the locked file."""
    lock = _validated_split_lock(split_lock)
    base = Path(split_lock).resolve().parent
    path = Path(test_manifest).resolve()
    entry = lock.get("manifests", {}).get("test", {})
    expected_path = (base / str(entry.get("path", ""))).resolve()
    if path != expected_path:
        raise ManifestError("test manifest is not the file recorded by split_lock.json")
    if sha256_file(path) != entry.get("sha256"):
        raise ManifestError("test manifest SHA-256 differs from split_lock.json")
    return lock


class _BaseCropSet:
    """On-demand runtime-identical crops from exactly one locked split."""

    def __init__(
        self,
        rows: Sequence[ManifestRow],
        image_root: os.PathLike[str] | str,
        *,
        allowed_split: str,
        fixture_maps: Mapping[str, P.PositionMap] | None = None,
        warp: FW.FrameWarp | None = None,
        seed: int = 0,
    ):
        if allowed_split not in SPLITS:
            raise ManifestError(f"invalid crop-set split {allowed_split!r}")
        wrong = {row.split for row in rows if row.split != allowed_split}
        if wrong:
            raise ProtectedTestError(
                f"{allowed_split} dataset received rows from other splits: {sorted(wrong)}"
            )
        self.rows = [row for row in rows if row.is_usable]
        if not self.rows:
            raise ManifestError(f"{allowed_split} has no usable rows")
        self.image_root = Path(image_root).resolve()
        self.fixture_maps = dict(fixture_maps or {"G": P.PositionMap()})
        unsupported = {row.fixture_id for row in self.rows} - set(self.fixture_maps)
        if unsupported:
            raise ManifestError(
                f"no calibrated position map for fixture(s) {sorted(unsupported)}; "
                "never apply G coordinates to S"
            )
        self.warp = warp if warp is not None else FW.load()
        self.seed = seed
        self._crop_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_crops(self, row: ManifestRow) -> dict[str, np.ndarray]:
        key = (row.source_image_id, row.fixture_id)
        if key in self._crop_cache:
            return self._crop_cache[key]
        path = _within(self.image_root, row.source_image_relpath)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        size = image.shape[1], image.shape[0]
        if size == (1920, 1080):
            if self.warp is None:
                raise ManifestError("1920x1080 source requires the frozen frame_warp_1080.json")
            image = self.warp.apply(image)
        elif size != (640, 480):
            raise ManifestError(
                f"{path}: expected 640x480 or 1920x1080, got {size[0]}x{size[1]}"
            )
        pmap = self.fixture_maps[row.fixture_id]
        crops = {
            crop.position: crop.image
            for crop in P.extract_crops(image, pmap, row.source_image_id)
        }
        self._crop_cache[key] = crops
        return crops

    def get(self, index: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, int]:
        del rng  # reserved for a predeclared training-only augmentation policy
        row = self.rows[index]
        crop = self._load_crops(row)[row.position_id]
        return to_tensor_array(crop), CANONICAL_CLASSES.index(row.visual_class)


class StagingCropSet(_BaseCropSet):
    """Training adapter that can never open protected TEST rows."""

    def __init__(self, rows, image_root, *, allowed_split, **kwargs):
        if allowed_split not in {"train", "validation"}:
            raise ProtectedTestError("training adapter may only open train or validation")
        if any(row.designation == "protected_test" for row in rows):
            raise ProtectedTestError("protected test rows reached the training adapter")
        super().__init__(rows, image_root, allowed_split=allowed_split, **kwargs)


class ProtectedTestCropSet(_BaseCropSet):
    """Evaluation-only adapter that opens exactly the locked TEST rows."""

    def __init__(self, rows, image_root, **kwargs):
        super().__init__(rows, image_root, allowed_split="test", **kwargs)

