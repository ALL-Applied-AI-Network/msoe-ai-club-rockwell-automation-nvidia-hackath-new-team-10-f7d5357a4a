#!/usr/bin/env python3
"""Inference-free, evidence-preserving vial staging image collector.

The collector watches a folder populated by the approved camera application.  It
never opens a camera device, runs a model, controls manufacturing equipment, or
turns intended scenarios into adjudicated labels.  It guides one immutable
train, validation, protected-test, or rehearsal visit and preserves every raw
capture attempt with event-level provenance.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import stat
import struct
import sys
import tarfile
import tempfile
import time
from typing import Iterable, Mapping, Sequence
import zlib


COLLECTOR_VERSION = "staging-collector-v2.1.0"
SCHEMA_VERSION = 1
CANONICAL_CLASSES = (
    "pass",
    "missing_cap",
    "crooked_cap",
    "damaged_vial",
    "no_vial",
)
OFFICIAL_ROLES = ("train", "validation", "protected_test")
ROLES = OFFICIAL_ROLES + ("rehearsal",)
FIXTURES = ("G", "S")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
RESOLUTION_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
BODY_ANGLES = (90, 88, 86, 84, 82, 80)
CAP_ANGLES = (80, 75, 70, 65, 60)
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_PNG_DECOMPRESSED_BYTES = 32 * 1024 * 1024
EVENT_CARD_MANIFEST_NAME = "MANIFEST.json"

FAMILY_COUNTS = {
    "train": {
        "pass_lean": 12,
        "crooked_cap": 15,
        "missing_cap": 8,
        "damaged_vial": 10,
        "no_vial": 8,
        "mixed": 12,
    },
    "validation": {
        "pass_lean": 6,
        "crooked_cap": 5,
        "missing_cap": 4,
        "damaged_vial": 5,
        "no_vial": 4,
        "mixed": 6,
    },
    "protected_test": {
        "pass_lean": 6,
        "crooked_cap": 5,
        "missing_cap": 4,
        "damaged_vial": 5,
        "no_vial": 4,
        "mixed": 6,
    },
    "rehearsal": {
        "pass_lean": 1,
        "crooked_cap": 1,
        "missing_cap": 1,
        "damaged_vial": 1,
        "no_vial": 1,
        "mixed": 1,
    },
}

PLAN_COLUMNS = (
    "role",
    "event_id",
    "event_uid",
    "family",
    "title",
    "fixture_id",
    "inspection_mode",
    "primary_targets",
    "instructions",
    "fresh_placement_required",
)
INTENT_COLUMNS = (
    "session_id",
    "event_id",
    "event_uid",
    "fixture_id",
    "position_id",
    "expected_occupancy",
    "intended_visual_class",
    "inspection_mode",
    "intended_gt_sequence",
    "planned_vial_axis_angle_deg",
    "planned_cap_axis_angle_deg",
    "planned_cap_to_vial_misalignment_deg",
    "planned_cap_seating_state",
    "scenario_family",
    "intent_only",
    "review_state",
    "usable",
    "notes",
)
MEASUREMENT_COLUMNS = (
    "event_uid",
    "fixture_id",
    "position_id",
    "planned_vial_axis_angle_deg",
    "planned_cap_axis_angle_deg",
    "actual_vial_axis_angle_deg",
    "actual_cap_axis_angle_deg",
    "actual_cap_to_vial_misalignment_deg",
    "actual_cap_seating_state",
    "measurement_method",
    "measurement_uncertainty_deg",
    "measured_by",
    "measured_at_utc",
    "notes",
)
ATTEMPT_COLUMNS = (
    "schema_version",
    "session_id",
    "role",
    "event_id",
    "event_uid",
    "attempt",
    "status",
    "source_name",
    "source_mtime_utc",
    "capture_name",
    "ready_confirmed_at_utc",
    "capture_time_utc",
    "sha256",
    "bytes",
    "width",
    "height",
    "fixture_id",
    "family",
    "operator_id",
    "confirmation_note",
    "label_source",
    "review_state",
    "usable",
)
ACTION_COLUMNS = (
    "time_utc",
    "session_id",
    "role",
    "event_id",
    "event_uid",
    "action",
    "reason",
)
REFUSAL_COLUMNS = (
    "time_utc",
    "event_id",
    "source_name",
    "reason",
)
CAPTURE_DIGEST_COLUMNS = (
    "reserved_at_utc",
    "sha256",
    "session_id",
    "role",
    "event_id",
    "event_uid",
    "capture_name",
)
HANDOFF_CAMPAIGN_NAME = "development_campaign.json"
HANDOFF_DIGESTS_NAME = "development_capture_sha256_reservations.csv"
HANDOFF_SEALED_SESSIONS_NAME = "development_sealed_sessions.json"
HANDOFF_RECEIPT_NAME = "protected_handoff_receipt.json"
HANDOFF_CHECKSUMS_NAME = "SHA256SUMS"
HANDOFF_FILES = {
    HANDOFF_CAMPAIGN_NAME,
    HANDOFF_DIGESTS_NAME,
    HANDOFF_SEALED_SESSIONS_NAME,
    HANDOFF_RECEIPT_NAME,
    HANDOFF_CHECKSUMS_NAME,
}
SEALED_SESSION_HASH_PATHS = (
    ("evidence_tar_sha256", "sealed/evidence.tar"),
    ("evidence_tar_checksum_file_sha256", "sealed/evidence.tar.sha256"),
    ("seal_manifest_sha256", "records/seal_manifest.json"),
    ("seal_selection_sha256", "records/seal_selection.json"),
    ("capture_attempts_sha256", "records/capture_attempts.csv"),
    ("event_actions_sha256", "records/event_actions.csv"),
)
REVIEW_COLUMNS = (
    "sample_id",
    "event_uid",
    "session_id",
    "event_id",
    "source_image_id",
    "source_image_relpath",
    "source_image_sha256",
    "capture_time_utc",
    "fixture_id",
    "position_id",
    "expected_occupancy",
    "intended_visual_class",
    "inspection_mode",
    "intended_gt_sequence",
    "actual_visual_class",
    "actual_gt_sequence",
    "reviewer_a",
    "reviewer_a_confidence",
    "reviewer_b",
    "reviewer_b_confidence",
    "adjudicator",
    "review_state",
    "usable",
    "notes",
)
BLINDED_REVIEW_COLUMNS = (
    "review_sample_id",
    "review_image_id",
    "review_image_relpath",
    "review_image_sha256",
    "fixture_id",
    "position_id",
    "reviewer_id",
    "actual_visual_class",
    "reviewer_confidence",
    "visible_evidence_notes",
    "review_state",
)
REVIEW_JOIN_COLUMNS = (
    "reviewer_packet",
    "review_sample_id",
    "review_image_id",
    "sample_id",
    "event_uid",
    "event_id",
    "source_image_id",
    "source_image_relpath",
    "source_image_sha256",
    "fixture_id",
    "position_id",
)


class CollectionError(RuntimeError):
    """A collection invariant or evidence boundary was violated."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise CollectionError(reason)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except (TypeError, ValueError) as exc:
        raise CollectionError(f"{field} must be an ISO-8601 timestamp with UTC offset") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_exclusive(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def write_bytes_atomic_exclusive(path: Path, data: bytes, mode: int = 0o644) -> None:
    """Commit complete bytes to a never-overwritten final name on one filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".capture-building-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path)
        _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_text_exclusive(path: Path, value: str, mode: int = 0o644) -> None:
    write_bytes_exclusive(path, value.encode("utf-8"), mode=mode)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"unreadable JSON: {path}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_csv_exclusive(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8", closefd=False) as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def append_csv(path: Path, columns: Sequence[str], row: Mapping[str, object]) -> None:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe log: {path}")
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=columns, extrasaction="raise").writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe CSV: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == tuple(columns),
                f"{path}: columns differ from the frozen schema")
        return list(reader)


def parse_resolution(value: str) -> tuple[int, int]:
    match = RESOLUTION_RE.fullmatch(value.strip())
    require(match is not None, "resolution must use WIDTHxHEIGHT, for example 1920x1080")
    width, height = (int(match.group(1)), int(match.group(2)))
    require((width, height) in {(640, 480), (1920, 1080)},
            "supported staging resolutions are 640x480 and 1920x1080")
    return width, height


def paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def resolve_without_symlinks(path: Path, *, must_exist: bool) -> Path:
    """Resolve a path only after refusing symlinks in every existing component."""
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() or current.is_symlink():
            require(not current.is_symlink(), f"symlink path component refused: {current}")
    if must_exist:
        require(absolute.exists(), f"path does not exist: {absolute}")
    return absolute.resolve(strict=must_exist)


def _hash_value(value: object, field: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{field} must be 64 lowercase hexadecimal digits")
    return value


def validate_baseline_freeze(path: Path, fixture: str, now: datetime | None = None) -> dict[str, str]:
    value = read_json(path)
    require(value.get("schema_version") == 1, "baseline freeze schema_version must be 1")
    require(value.get("training_data_scope") == "synthetic_only",
            "baseline freeze must declare training_data_scope=synthetic_only")
    require(tuple(value.get("canonical_classes", ())) == CANONICAL_CLASSES,
            "baseline freeze must preserve the exact five-class taxonomy and order")
    baseline_id = value.get("baseline_id")
    require(isinstance(baseline_id, str) and baseline_id.strip() and "REPLACE_" not in baseline_id,
            "baseline freeze needs a real baseline_id")
    frozen = parse_utc(str(value.get("frozen_at_utc", "")), "frozen_at_utc")
    require(frozen < (now or datetime.now(timezone.utc)),
            "synthetic baseline must be frozen before collection starts")
    for field in ("model_sha256", "thresholds_sha256", "decision_policy_sha256"):
        _hash_value(value.get(field), field)
    for field in ("code_sha256", "preprocessing_sha256"):
        mapping = value.get(field)
        require(isinstance(mapping, dict) and mapping, f"{field} must be a nonempty hash mapping")
        for name, digest in mapping.items():
            require(isinstance(name, str) and name.strip(), f"{field} has an empty key")
            _hash_value(digest, f"{field}.{name}")
    maps = value.get("fixture_map_sha256")
    require(isinstance(maps, dict), "fixture_map_sha256 must be an object")
    require(fixture in maps, f"baseline freeze has no calibrated {fixture} fixture map")
    fixture_map = _hash_value(maps[fixture], f"fixture_map_sha256.{fixture}")
    if fixture == "S":
        require("G" in maps, "fixture S collection also requires the recorded G map for separation proof")
        require(_hash_value(maps["G"], "fixture_map_sha256.G") != fixture_map,
                "fixture S must use its own calibrated map, not the G map")
    policy_version = value.get("decision_policy_version")
    require(isinstance(policy_version, str) and policy_version.strip(),
            "decision_policy_version must be recorded")
    return {
        "baseline_id": baseline_id.strip(),
        "frozen_at_utc": frozen.isoformat().replace("+00:00", "Z"),
        "freeze_file_sha256": sha256_file(path),
        "fixture_map_sha256": fixture_map,
        "decision_policy_version": policy_version.strip(),
    }


class PositionAllocator:
    def __init__(self, fixture: str, rng: random.Random):
        self._rng = rng
        self._positions = [f"{fixture}{index:02d}" for index in range(1, 31)]
        self._pool: list[str] = []

    def take(self, count: int, *, allowed: Sequence[str] | None = None) -> list[str]:
        universe = list(allowed or self._positions)
        require(count <= len(universe), "cannot allocate more unique targets than available positions")
        if allowed is not None:
            return self._rng.sample(universe, count)
        selected: list[str] = []
        while len(selected) < count:
            if not self._pool:
                self._pool = list(self._positions)
                self._rng.shuffle(self._pool)
            candidate = self._pool.pop()
            if candidate not in selected:
                selected.append(candidate)
        return selected


def _base_intent(session_id: str, event_id: str, event_uid: str, fixture: str,
                 family: str, inspection_mode: str) -> list[dict[str, object]]:
    return [
        {
            "session_id": session_id,
            "event_id": event_id,
            "event_uid": event_uid,
            "fixture_id": fixture,
            "position_id": f"{fixture}{index:02d}",
            "expected_occupancy": "occupied",
            "intended_visual_class": "pass",
            "inspection_mode": inspection_mode,
            "intended_gt_sequence": "",
            "planned_vial_axis_angle_deg": "90",
            "planned_cap_axis_angle_deg": "90",
            "planned_cap_to_vial_misalignment_deg": "0",
            "planned_cap_seating_state": "fully_seated",
            "scenario_family": family,
            "intent_only": "true",
            "review_state": "pending_independent_review",
            "usable": "false",
            "notes": "Default upright control; reviewer must label actual visible content.",
        }
        for index in range(1, 31)
    ]


def _row_by_position(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["position_id"]): row for row in rows}


def _target_text(rows: Sequence[Mapping[str, object]], positions: Sequence[str]) -> str:
    by_position = {str(row["position_id"]): row for row in rows}
    parts = []
    for position in positions:
        row = by_position[position]
        detail = str(row["intended_visual_class"])
        if row["planned_vial_axis_angle_deg"]:
            detail += f" vial={row['planned_vial_axis_angle_deg']}deg"
        if row["planned_cap_axis_angle_deg"]:
            detail += f" cap={row['planned_cap_axis_angle_deg']}deg"
        if row["intended_gt_sequence"]:
            detail += f" {row['intended_gt_sequence']}"
        parts.append(f"{position}={detail}")
    return "; ".join(parts)


def build_plan(role: str, fixture: str, session_id: str, seed_hex: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    require(role in ROLES, f"unsupported role {role!r}")
    require(fixture in FIXTURES, f"unsupported fixture {fixture!r}")
    rng = random.Random(int(hashlib.sha256(seed_hex.encode("utf-8")).hexdigest(), 16))
    allocator = PositionAllocator(fixture, rng)
    plan: list[dict[str, object]] = []
    intents: list[dict[str, object]] = []
    sequence = 0

    for family, count in FAMILY_COUNTS[role].items():
        for family_index in range(count):
            sequence += 1
            event_id = f"E{sequence:03d}"
            event_uid = f"{session_id}:{event_id}"
            inspection_mode = "final_inspection"
            rows = _base_intent(session_id, event_id, event_uid, fixture, family, inspection_mode)
            by_position = _row_by_position(rows)
            primary: list[str] = []

            if family == "pass_lean":
                angle = BODY_ANGLES[family_index % len(BODY_ANGLES)]
                primary = allocator.take(3)
                for position in primary:
                    row = by_position[position]
                    row.update(
                        planned_vial_axis_angle_deg=str(angle),
                        planned_cap_axis_angle_deg=str(angle),
                        planned_cap_to_vial_misalignment_deg="0",
                        planned_cap_seating_state="fully_seated",
                        notes="Acceptable whole-vial lean: intact vial and cap fully seated/aligned.",
                    )
                title = f"Acceptable PASS lean at {angle} degrees"
                instructions = (
                    f"Place intact, correctly capped vials at {', '.join(primary)} with body and cap axes both "
                    f"about {angle} degrees. All other targets are upright PASS controls. Whole-vial lean is not "
                    "crooked_cap."
                )
            elif family == "crooked_cap":
                cap_angle = CAP_ANGLES[family_index % len(CAP_ANGLES)]
                primary = allocator.take(3)
                for position in primary:
                    row = by_position[position]
                    row.update(
                        intended_visual_class="crooked_cap",
                        planned_vial_axis_angle_deg="90",
                        planned_cap_axis_angle_deg=str(cap_angle),
                        planned_cap_to_vial_misalignment_deg=str(90 - cap_angle),
                        planned_cap_seating_state="visibly_unseated_or_misaligned",
                        notes="Crooked only if visible seating/alignment supports it; angle alone is not ground truth.",
                    )
                title = f"Crooked-cap ladder at about {cap_angle} degrees"
                instructions = (
                    f"At {', '.join(primary)}, keep vial bodies upright and stage caps about {cap_angle} degrees "
                    "with visible improper seating or cap-to-vial misalignment. All other targets are PASS controls."
                )
            elif family == "missing_cap":
                primary = allocator.take(3)
                lean_angles = (80, 85, 90)
                for position, angle in zip(primary, lean_angles):
                    row = by_position[position]
                    row.update(
                        intended_visual_class="missing_cap",
                        planned_vial_axis_angle_deg=str(angle),
                        planned_cap_axis_angle_deg="",
                        planned_cap_to_vial_misalignment_deg="",
                        planned_cap_seating_state="cap_absent",
                        notes="Vial present and cap absent; body lean is a controlled confounder.",
                    )
                title = "Missing caps across acceptable vial-body lean"
                instructions = (
                    f"Remove caps at {', '.join(primary)}; keep the vials present at approximately 80, 85, and "
                    "90 degrees respectively. All other targets are PASS controls."
                )
            elif family == "damaged_vial":
                primary = allocator.take(3)
                damage_types = ("approved crack", "approved chip", "approved deformation")
                for position, damage_type in zip(primary, damage_types):
                    row = by_position[position]
                    row.update(
                        intended_visual_class="damaged_vial",
                        notes=f"{damage_type}; use only a safe, approved, visibly representative specimen.",
                    )
                title = "Approved visible vial damage"
                instructions = (
                    f"At {', '.join(primary)}, use safe approved specimens representing a visible crack, chip, and "
                    "deformation. If genuine approved damage is unavailable, skip this event with a reason; never "
                    "substitute glare, vial lean, or invented damage. Keep every non-primary target as an upright, "
                    "intact, correctly capped PASS control."
                )
            elif family == "no_vial":
                if family_index % 2 == 0:
                    primary = allocator.take(3)
                    for position in primary:
                        row = by_position[position]
                        row.update(
                            expected_occupancy="occupied",
                            intended_visual_class="no_vial",
                            intended_gt_sequence="missed",
                            planned_vial_axis_angle_deg="",
                            planned_cap_axis_angle_deg="",
                            planned_cap_to_vial_misalignment_deg="",
                            planned_cap_seating_state="no_vial",
                            notes="Order declared complete; expected occupied but visibly empty.",
                        )
                    title = "Missing placements in a completed fixture"
                    instructions = (
                        f"Declare the fixture complete and leave {', '.join(primary)} empty as expected-occupied "
                        "MISSED positions. Fill all remaining targets with PASS controls."
                    )
                else:
                    inspection_mode = "live_fill"
                    fill_count = (25, 18, 10)[(family_index // 2) % 3]
                    rows = _base_intent(session_id, event_id, event_uid, fixture, family, inspection_mode)
                    by_position = _row_by_position(rows)
                    filled = [f"{fixture}{index:02d}" for index in range(1, fill_count + 1)]
                    missed = allocator.take(1, allowed=filled)
                    waiting = [f"{fixture}{index:02d}" for index in range(fill_count + 1, 31)]
                    primary = missed + waiting
                    for position in missed:
                        by_position[position].update(
                            expected_occupancy="occupied", intended_visual_class="no_vial",
                            intended_gt_sequence="missed", planned_vial_axis_angle_deg="",
                            planned_cap_axis_angle_deg="", planned_cap_to_vial_misalignment_deg="",
                            planned_cap_seating_state="no_vial",
                            notes="Internal expected placement missing during live fill.",
                        )
                    for position in waiting:
                        by_position[position].update(
                            expected_occupancy="empty", intended_visual_class="no_vial",
                            intended_gt_sequence="waiting", planned_vial_axis_angle_deg="",
                            planned_cap_axis_angle_deg="", planned_cap_to_vial_misalignment_deg="",
                            planned_cap_seating_state="no_vial",
                            notes="Legitimate trailing unfilled target; deterministic decision must be WAITING.",
                        )
                    title = f"Partial fill through {fixture}{fill_count:02d}: MISSED versus WAITING"
                    instructions = (
                        f"Stage a partial order through {fixture}{fill_count:02d}; leave {missed[0]} empty inside the "
                        f"filled sequence (MISSED) and leave {fixture}{fill_count + 1:02d}-{fixture}30 legitimately "
                        "unfilled (WAITING)."
                    )
            else:  # mixed
                primary = allocator.take(5)
                lean, crooked, missing, damaged, empty = primary
                lean_angle = BODY_ANGLES[family_index % len(BODY_ANGLES)]
                cap_angle = CAP_ANGLES[family_index % len(CAP_ANGLES)]
                by_position[lean].update(
                    planned_vial_axis_angle_deg=str(lean_angle), planned_cap_axis_angle_deg=str(lean_angle),
                    notes="Acceptable aligned PASS lean adjacent to actual defects.",
                )
                by_position[crooked].update(
                    intended_visual_class="crooked_cap", planned_vial_axis_angle_deg="90",
                    planned_cap_axis_angle_deg=str(cap_angle),
                    planned_cap_to_vial_misalignment_deg=str(90 - cap_angle),
                    planned_cap_seating_state="visibly_unseated_or_misaligned",
                    notes="Visible improper seating/misalignment required.",
                )
                by_position[missing].update(
                    intended_visual_class="missing_cap", planned_cap_axis_angle_deg="",
                    planned_cap_to_vial_misalignment_deg="", planned_cap_seating_state="cap_absent",
                    notes="Vial present and cap absent.",
                )
                by_position[damaged].update(
                    intended_visual_class="damaged_vial",
                    notes="Safe approved visible damage only; otherwise skip with reason.",
                )
                by_position[empty].update(
                    expected_occupancy="occupied", intended_visual_class="no_vial",
                    intended_gt_sequence="missed", planned_vial_axis_angle_deg="",
                    planned_cap_axis_angle_deg="", planned_cap_to_vial_misalignment_deg="",
                    planned_cap_seating_state="no_vial",
                    notes="Order complete; expected occupied but visibly empty.",
                )
                title = "Mixed confounder event"
                instructions = (
                    f"Stage PASS lean at {lean}, visibly crooked cap at {crooked}, missing cap at {missing}, approved "
                    f"damage at {damaged}, and an expected-occupied empty at {empty}. Do not place multiple defects "
                    "on one target. Keep every non-primary target as an upright, intact, correctly capped PASS "
                    "control."
                )

            primary_targets = _target_text(rows, primary)
            plan.append(
                {
                    "role": role,
                    "event_id": event_id,
                    "event_uid": event_uid,
                    "family": family,
                    "title": title,
                    "fixture_id": fixture,
                    "inspection_mode": inspection_mode,
                    "primary_targets": primary_targets,
                    "instructions": instructions,
                    "fresh_placement_required": "true",
                }
            )
            intents.extend(rows)

    validate_plan(plan, intents, role, fixture, session_id)
    return plan, intents


def validate_plan(plan: Sequence[Mapping[str, object]], intents: Sequence[Mapping[str, object]],
                  role: str, fixture: str, session_id: str) -> None:
    expected_total = sum(FAMILY_COUNTS[role].values())
    require(len(plan) == expected_total, f"{role} plan must contain {expected_total} events")
    require(len(intents) == expected_total * 30, "every event must have exactly 30 intent rows")
    actual_counts = {family: 0 for family in FAMILY_COUNTS[role]}
    event_ids: set[str] = set()
    intent_by_event: dict[str, list[Mapping[str, object]]] = {}
    for event in plan:
        require(event.get("role") == role, "plan row crosses session role")
        require(event.get("fixture_id") == fixture, "plan row crosses fixture")
        event_id = str(event.get("event_id", ""))
        require(event_id not in event_ids, "duplicate event_id in plan")
        event_ids.add(event_id)
        family = str(event.get("family", ""))
        require(family in actual_counts, f"unknown scenario family {family!r}")
        actual_counts[family] += 1
    require(actual_counts == FAMILY_COUNTS[role], "scenario-family counts differ from the preregistered plan")
    for row in intents:
        require(row.get("session_id") == session_id, "intent row crosses session")
        require(row.get("fixture_id") == fixture, "intent row crosses fixture")
        event_id = str(row.get("event_id", ""))
        require(event_id in event_ids, "intent row references an unknown event")
        intent_by_event.setdefault(event_id, []).append(row)
        visual_class = str(row.get("intended_visual_class", ""))
        require(visual_class in CANONICAL_CLASSES, f"noncanonical intended class {visual_class!r}")
        require(row.get("intent_only") == "true" and row.get("review_state") == "pending_independent_review",
                "intended scenarios must remain pending independent review")
        require(row.get("usable") == "false", "collector may not mark intended labels usable")
        occupancy = row.get("expected_occupancy")
        sequence = row.get("intended_gt_sequence")
        if visual_class == "no_vial":
            require((occupancy, sequence) in {("occupied", "missed"), ("empty", "waiting")},
                    "no_vial intent must distinguish expected MISSED from legitimate WAITING")
        else:
            require(occupancy == "occupied" and sequence == "",
                    "only no_vial may carry empty/waiting sequence context")
        if visual_class == "pass":
            vial = int(str(row.get("planned_vial_axis_angle_deg", "0")))
            cap = int(str(row.get("planned_cap_axis_angle_deg", "0")))
            require(80 <= vial <= 90 and cap == vial,
                    "supervised PASS intent requires aligned cap/vial axes within 80-90 degrees")
        if visual_class == "crooked_cap":
            require(row.get("planned_cap_seating_state") == "visibly_unseated_or_misaligned",
                    "crooked_cap requires visible seating/alignment support")
    expected_positions = {f"{fixture}{index:02d}" for index in range(1, 31)}
    for event_id, rows in intent_by_event.items():
        require(len(rows) == 30, f"{event_id}: expected 30 position rows")
        require({str(row["position_id"]) for row in rows} == expected_positions,
                f"{event_id}: position coverage differs from {fixture}01-{fixture}30")
    if role != "rehearsal":
        lean_angles = {
            int(str(row["planned_vial_axis_angle_deg"]))
            for row in intents
            if row["scenario_family"] == "pass_lean"
        }
        crooked_angles = {
            int(str(row["planned_cap_axis_angle_deg"]))
            for row in intents
            if row["intended_visual_class"] == "crooked_cap"
        }
        no_vial_states = {
            str(row["intended_gt_sequence"])
            for row in intents
            if row["intended_visual_class"] == "no_vial"
        }
        require(set(BODY_ANGLES) <= lean_angles, "plan lacks the complete 80-90 degree PASS lean ladder")
        require(set(CAP_ANGLES) <= crooked_angles, "plan lacks the complete feasible crooked-cap ladder")
        require({"missed", "waiting"} <= no_vial_states, "plan lacks MISSED/WAITING no_vial coverage")


def _png_dimensions(data: bytes) -> tuple[int, int]:
    require(data.startswith(b"\x89PNG\r\n\x1a\n"), "invalid PNG signature")
    offset = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    saw_ihdr = saw_plte = saw_idat = False
    palette_entries = 0
    idat_ended = False
    saw_iend = False
    chunk_index = 0
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + length
        require(end <= len(data), "truncated PNG chunk")
        payload = data[offset + 8:offset + 8 + length]
        recorded_crc = struct.unpack(">I", data[offset + 8 + length:end])[0]
        require(zlib.crc32(kind + payload) & 0xFFFFFFFF == recorded_crc, "PNG chunk CRC mismatch")
        if kind == b"IHDR":
            require(chunk_index == 0 and length == 13 and not saw_ihdr, "invalid PNG IHDR")
            width, height = struct.unpack(">II", payload[:8])
            require(0 < width <= 1920 and 0 < height <= 1080 and width * height <= 1920 * 1080,
                    "PNG dimensions exceed the supported staging safety bound")
            bit_depth, color_type, compression, filtering, interlace = payload[8:13]
            require(compression == 0 and filtering == 0, "unsupported PNG compression or filter method")
            require(interlace == 0, "interlaced PNG is unsupported; configure the camera for non-interlaced PNG")
            legal_depths = {
                0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8},
                4: {8, 16}, 6: {8, 16},
            }
            require(color_type in legal_depths and bit_depth in legal_depths[color_type],
                    "invalid PNG color type or bit depth")
            saw_ihdr = True
        elif kind == b"PLTE":
            require(saw_ihdr and not saw_idat and not saw_plte and length % 3 == 0 and 0 < length <= 768,
                    "invalid PNG palette")
            require(color_type not in {0, 4}, "PNG palette is forbidden for grayscale color types")
            saw_plte = True
            palette_entries = length // 3
        elif kind == b"IDAT":
            require(saw_ihdr and not idat_ended, "PNG IDAT chunks must be contiguous after IHDR")
            idat.extend(payload)
            saw_idat = True
        elif kind == b"IEND":
            require(saw_ihdr and saw_idat and length == 0, "invalid PNG IEND")
            saw_iend = True
            require(end == len(data), "PNG has bytes after IEND")
            break
        else:
            require(saw_ihdr, "PNG chunk appears before IHDR")
            if saw_idat:
                idat_ended = True
            require(not (65 <= kind[0] <= 90), f"unsupported critical PNG chunk {kind!r}")
        offset = end
        chunk_index += 1
    require(width is not None and height is not None and width > 0 and height > 0,
            "PNG has no valid dimensions")
    require(idat and saw_iend, "PNG lacks image data or IEND")
    require(color_type != 3 or saw_plte, "indexed PNG lacks its required palette")
    if color_type == 3:
        require(palette_entries <= (1 << int(bit_depth)),
                "indexed PNG palette has more entries than its bit depth permits")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[int(color_type)]
    row_bytes = (int(width) * channels * int(bit_depth) + 7) // 8
    expected_decoded = (row_bytes + 1) * int(height)
    require(expected_decoded <= MAX_PNG_DECOMPRESSED_BYTES,
            "PNG raster exceeds the decompression safety limit")
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(bytes(idat), expected_decoded + 1)
        require(len(decoded) <= expected_decoded and not decompressor.unconsumed_tail,
                "PNG decompressed data exceeds the safety limit")
        decoded += decompressor.flush(expected_decoded + 1 - len(decoded))
        require(decompressor.eof and not decompressor.unused_data and len(decoded) == expected_decoded,
                "PNG image stream is incomplete or exceeds the safety limit")
        require(all(decoded[row * (row_bytes + 1)] <= 4 for row in range(int(height))),
                "PNG scanline uses an invalid filter type")
    except zlib.error as exc:
        raise CollectionError("PNG image data cannot be decompressed") from exc
    return int(width), int(height)


class _JpegBitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.position = 0

    def read(self, count: int) -> int:
        require(self.position + count <= len(self.data) * 8, "truncated JPEG coefficient stream")
        value = 0
        for _ in range(count):
            byte = self.data[self.position // 8]
            value = (value << 1) | ((byte >> (7 - self.position % 8)) & 1)
            self.position += 1
        return value

    def symbol(self, table: Mapping[tuple[int, int], int]) -> int:
        code = 0
        for length in range(1, 17):
            code = (code << 1) | self.read(1)
            symbol = table.get((length, code))
            if symbol is not None:
                return symbol
        raise CollectionError("JPEG coefficient uses no declared Huffman code")

    def require_padding(self) -> None:
        remaining = len(self.data) * 8 - self.position
        require(remaining <= 7, "JPEG entropy segment contains excess coefficient bytes")
        if remaining:
            require(self.read(remaining) == (1 << remaining) - 1,
                    "JPEG entropy padding bits must be all ones")


def _jpeg_huffman_codes(counts: bytes, symbols: bytes) -> dict[tuple[int, int], int]:
    table: dict[tuple[int, int], int] = {}
    code = 0
    symbol_index = 0
    for length, count in enumerate(counts, 1):
        require(code + count <= (1 << length), "oversubscribed JPEG Huffman table")
        for _ in range(count):
            table[(length, code)] = symbols[symbol_index]
            symbol_index += 1
            code += 1
        code <<= 1
    require(symbol_index == len(symbols), "JPEG Huffman symbol count differs")
    return table


def _validate_jpeg_coefficients(
        *, width: int, height: int,
        frame_components: Mapping[int, tuple[int, int, int]],
        scan_components: Sequence[tuple[int, int, int]],
        huffman_tables: Mapping[tuple[int, int], Mapping[tuple[int, int], int]],
        restart_interval: int, entropy_segments: Sequence[bytes],
        restart_markers: Sequence[int]) -> None:
    max_horizontal = max(value[0] for value in frame_components.values())
    max_vertical = max(value[1] for value in frame_components.values())
    mcu_columns = (width + 8 * max_horizontal - 1) // (8 * max_horizontal)
    mcu_rows = (height + 8 * max_vertical - 1) // (8 * max_vertical)
    total_mcus = mcu_columns * mcu_rows
    require(total_mcus > 0, "JPEG has no minimum coded units")
    if restart_interval:
        expected_segments = (total_mcus + restart_interval - 1) // restart_interval
        require(len(entropy_segments) == expected_segments and
                len(restart_markers) == expected_segments - 1,
                "JPEG restart marker count differs from the declared interval")
        for index, marker in enumerate(restart_markers):
            require(marker == 0xD0 + index % 8, "JPEG restart markers are out of sequence")
    else:
        require(len(entropy_segments) == 1 and not restart_markers,
                "JPEG restart marker appears without a restart interval")

    decoded_mcus = 0
    for segment_index, segment in enumerate(entropy_segments):
        require(segment, "JPEG entropy segment is empty")
        segment_mcus = (
            min(restart_interval, total_mcus - decoded_mcus)
            if restart_interval else total_mcus
        )
        reader = _JpegBitReader(segment)
        for _ in range(segment_mcus):
            for component_id, dc_table_id, ac_table_id in scan_components:
                horizontal, vertical, _ = frame_components[component_id]
                dc_table = huffman_tables[(0, dc_table_id)]
                ac_table = huffman_tables[(1, ac_table_id)]
                for _block in range(horizontal * vertical):
                    dc_category = reader.symbol(dc_table)
                    require(dc_category <= 11, "JPEG DC coefficient category is invalid")
                    reader.read(dc_category)
                    coefficient = 1
                    while coefficient < 64:
                        symbol = reader.symbol(ac_table)
                        if symbol == 0:
                            break
                        run, size = symbol >> 4, symbol & 0x0F
                        if symbol == 0xF0:
                            coefficient += 16
                            require(coefficient <= 64, "JPEG AC zero run exceeds the block")
                            continue
                        require(1 <= size <= 10, "JPEG AC coefficient size is invalid")
                        coefficient += run
                        require(coefficient < 64, "JPEG AC run exceeds the block")
                        reader.read(size)
                        coefficient += 1
            decoded_mcus += 1
        reader.require_padding()
    require(decoded_mcus == total_mcus, "JPEG coefficient stream has the wrong MCU count")


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    require(len(data) >= 8 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9",
            "invalid or truncated JPEG")
    offset = 2
    dimensions: tuple[int, int] | None = None
    frame_components: dict[int, tuple[int, int, int]] = {}
    quant_tables: set[int] = set()
    huffman_tables: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    restart_interval = 0
    scans = 0
    entropy_bytes = 0
    saw_eoi = False
    while offset < len(data):
        require(data[offset] == 0xFF, "malformed JPEG marker stream")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        require(offset < len(data), "truncated JPEG marker")
        marker = data[offset]
        offset += 1
        if marker == 0xD9:
            require(offset == len(data), "JPEG has bytes after EOI")
            saw_eoi = True
            break
        require(marker not in {0x00, 0xD8, 0x01} and marker not in range(0xD0, 0xD8),
                "unexpected standalone JPEG marker outside entropy data")
        require(offset + 2 <= len(data), "truncated JPEG segment length")
        length = struct.unpack(">H", data[offset:offset + 2])[0]
        require(length >= 2 and offset + length <= len(data), "invalid JPEG segment length")
        payload = data[offset + 2:offset + length]
        offset += length
        if marker == 0xDB:  # DQT
            cursor = 0
            while cursor < len(payload):
                precision_table = payload[cursor]
                cursor += 1
                precision, table_id = precision_table >> 4, precision_table & 0x0F
                require(precision == 0 and table_id <= 3,
                        "baseline JPEG requires an 8-bit quantization table")
                table_bytes = 64 * (precision + 1)
                require(cursor + table_bytes <= len(payload), "truncated JPEG quantization table")
                require(all(payload[cursor:cursor + table_bytes]),
                        "JPEG quantization table contains a forbidden zero value")
                quant_tables.add(table_id)
                cursor += table_bytes
            require(cursor == len(payload), "invalid JPEG DQT payload")
        elif marker == 0xC4:  # DHT
            cursor = 0
            while cursor < len(payload):
                table_class_id = payload[cursor]
                cursor += 1
                table_class, table_id = table_class_id >> 4, table_class_id & 0x0F
                require(table_class in {0, 1} and table_id <= 3,
                        "invalid JPEG Huffman table selector")
                require(cursor + 16 <= len(payload), "truncated JPEG Huffman counts")
                counts = payload[cursor:cursor + 16]
                symbol_count = sum(counts)
                cursor += 16
                require(0 < symbol_count <= 256 and cursor + symbol_count <= len(payload),
                        "invalid JPEG Huffman symbols")
                symbols = payload[cursor:cursor + symbol_count]
                cursor += symbol_count
                if table_class == 0:
                    require(all(symbol <= 11 for symbol in symbols),
                            "JPEG DC Huffman table declares an invalid category")
                else:
                    require(all(
                        symbol in {0x00, 0xF0} or 1 <= (symbol & 0x0F) <= 10
                        for symbol in symbols
                    ), "JPEG AC Huffman table declares an invalid run/size symbol")
                huffman_tables[(table_class, table_id)] = _jpeg_huffman_codes(counts, symbols)
            require(cursor == len(payload), "invalid JPEG DHT payload")
        elif marker == 0xC0:  # baseline Huffman JPEG
            require(dimensions is None and len(payload) >= 6, "duplicate or truncated JPEG SOF")
            require(payload[0] == 8, "only 8-bit Huffman JPEG frames are supported")
            height, width = struct.unpack(">HH", payload[1:5])
            require(0 < width <= 1920 and 0 < height <= 1080 and width * height <= 1920 * 1080,
                    "JPEG dimensions exceed the supported staging safety bound")
            component_count = payload[5]
            require(1 <= component_count <= 4 and len(payload) == 6 + 3 * component_count,
                    "invalid JPEG frame component list")
            for index in range(component_count):
                component_id, sampling, quant_id = payload[6 + 3 * index:9 + 3 * index]
                horizontal, vertical = sampling >> 4, sampling & 0x0F
                require(component_id not in frame_components and 1 <= horizontal <= 4 and
                        1 <= vertical <= 4 and quant_id <= 3,
                        "invalid JPEG frame component")
                frame_components[component_id] = (horizontal, vertical, quant_id)
            dimensions = (int(width), int(height))
        if marker == 0xDA:
            require(scans == 0, "multiple JPEG scans are unsupported; configure baseline interleaved JPEG")
            require(dimensions is not None and quant_tables and huffman_tables,
                    "JPEG scan appears before frame/quantization/Huffman tables")
            require(len(payload) >= 6, "truncated JPEG SOS")
            scan_components = payload[0]
            require(1 <= scan_components <= len(frame_components) and
                    len(payload) == 4 + 2 * scan_components,
                    "invalid JPEG scan component list")
            require(scan_components == len(frame_components),
                    "non-interleaved JPEG scans are unsupported; configure baseline interleaved JPEG")
            scan_ids: set[int] = set()
            scan_tables: list[tuple[int, int, int]] = []
            for index in range(scan_components):
                component_id, tables = payload[1 + 2 * index:3 + 2 * index]
                dc_table, ac_table = tables >> 4, tables & 0x0F
                require(component_id in frame_components and component_id not in scan_ids,
                        "JPEG scan references an invalid component")
                require(frame_components[component_id][2] in quant_tables and
                        (0, dc_table) in huffman_tables and (1, ac_table) in huffman_tables,
                        "JPEG scan references a missing coding table")
                scan_ids.add(component_id)
                scan_tables.append((component_id, dc_table, ac_table))
            spectral_start, spectral_end, approximation = payload[-3:]
            require((spectral_start, spectral_end, approximation) == (0, 63, 0),
                    "invalid baseline JPEG scan parameters")
            entropy_segments = [bytearray()]
            restart_markers: list[int] = []
            while offset < len(data):
                if data[offset] != 0xFF:
                    entropy_segments[-1].append(data[offset])
                    offset += 1
                    continue
                marker_start = offset
                offset += 1
                require(offset < len(data), "truncated JPEG entropy marker")
                entropy_marker = data[offset]
                if entropy_marker == 0x00:
                    entropy_segments[-1].append(0xFF)
                    offset += 1
                    continue
                if entropy_marker in range(0xD0, 0xD8):
                    restart_markers.append(entropy_marker)
                    entropy_segments.append(bytearray())
                    offset += 1
                    continue
                require(entropy_marker != 0xFF, "JPEG entropy contains ambiguous marker fill bytes")
                offset = marker_start
                break
            require(offset == len(data) - 2 and data[offset:offset + 2] == b"\xff\xd9",
                    "baseline JPEG must end immediately after its single entropy-coded scan")
            _validate_jpeg_coefficients(
                width=dimensions[0], height=dimensions[1],
                frame_components=frame_components, scan_components=scan_tables,
                huffman_tables=huffman_tables, restart_interval=restart_interval,
                entropy_segments=[bytes(segment) for segment in entropy_segments],
                restart_markers=restart_markers,
            )
            entropy_bytes += sum(len(segment) for segment in entropy_segments)
            scans += 1
        elif marker == 0xDD:
            require(len(payload) == 2, "invalid JPEG restart interval")
            restart_interval = struct.unpack(">H", payload)[0]
            require(restart_interval > 0, "JPEG restart interval must be nonzero")
        elif marker in {0xCC, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            raise CollectionError("unsupported JPEG coding mode; configure baseline Huffman JPEG (SOF0)")
    require(dimensions is not None and scans > 0 and entropy_bytes > 0 and saw_eoi,
            "JPEG lacks a complete frame, entropy-coded scan, or EOI")
    return dimensions


def image_dimensions(data: bytes) -> tuple[str, int, int]:
    if data.startswith(b"\x89PNG"):
        width, height = _png_dimensions(data)
        return ".png", width, height
    if data.startswith(b"\xff\xd8"):
        width, height = _jpeg_dimensions(data)
        return ".jpg", width, height
    raise CollectionError("file is not a structurally valid JPEG or PNG")


def _fingerprint(stat_result: os.stat_result) -> list[int]:
    return [stat_result.st_dev, stat_result.st_ino, stat_result.st_size, stat_result.st_mtime_ns]


def camera_snapshot(folder: Path) -> dict[str, list[int]]:
    snapshot: dict[str, list[int]] = {}
    for path in sorted(folder.iterdir()):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if path.name.startswith(".") or path.suffix.lower() not in EXTENSIONS:
            continue
        if stat.S_ISREG(info.st_mode) and not path.is_symlink():
            snapshot[path.name] = _fingerprint(info)
    return snapshot


@dataclass(frozen=True)
class StableImage:
    source_name: str
    source_mtime_utc: str
    data: bytes
    suffix: str
    width: int
    height: int
    fingerprint: list[int]


def stable_read(path: Path, stability_seconds: float) -> StableImage:
    require(path.parent.is_dir(), "camera folder disappeared")
    first = path.lstat()
    require(stat.S_ISREG(first.st_mode) and not path.is_symlink(), "camera candidate is not a regular file")
    require(first.st_size > 0, "camera candidate is empty")
    require(first.st_size <= MAX_IMAGE_BYTES,
            f"camera candidate exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB safety limit")
    time.sleep(stability_seconds)
    second = path.lstat()
    require(_fingerprint(first) == _fingerprint(second), "camera image is still changing")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "camera candidate changed type")
        chunks = []
        total = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            total += len(block)
            require(total <= MAX_IMAGE_BYTES,
                    f"camera candidate exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB safety limit")
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(_fingerprint(before) == _fingerprint(after) == _fingerprint(second),
            "camera image changed while it was read")
    data = b"".join(chunks)
    suffix, width, height = image_dimensions(data)
    mtime = datetime.fromtimestamp(second.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")
    return StableImage(path.name, mtime, data, suffix, width, height, _fingerprint(second))


def wait_for_new_image(folder: Path, seen: Mapping[str, list[int]], stability_seconds: float,
                       wait_seconds: float) -> tuple[StableImage, dict[str, list[int]]]:
    deadline = time.monotonic() + wait_seconds
    last_reason = "no new image"
    while time.monotonic() < deadline:
        current = camera_snapshot(folder)
        candidates = [
            folder / name for name, fingerprint in current.items()
            if seen.get(name) != fingerprint
        ]
        candidates.sort(key=lambda path: (path.lstat().st_mtime_ns, path.name), reverse=True)
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates[:5])
            extra = "..." if len(candidates) > 5 else ""
            raise CollectionError(
                f"multiple new camera images are ambiguous ({names}{extra}); leave exactly one new still"
            )
        for path in candidates:
            try:
                image = stable_read(path, stability_seconds)
                refreshed = camera_snapshot(folder)
                require(refreshed.get(path.name) == image.fingerprint,
                        "camera image changed after validation")
                refreshed_new = [
                    name for name, fingerprint in refreshed.items()
                    if seen.get(name) != fingerprint
                ]
                require(refreshed_new == [path.name],
                        "multiple new camera images appeared during validation; leave exactly one new still")
                time.sleep(stability_seconds)
                quiescent = camera_snapshot(folder)
                require(quiescent == refreshed,
                        "camera folder changed during the final quiescence check; leave exactly one stable still")
                return image, quiescent
            except (CollectionError, FileNotFoundError, OSError) as exc:
                last_reason = str(exc)
                if "multiple new camera images" in last_reason or "camera folder changed" in last_reason:
                    raise CollectionError(last_reason) from exc
        time.sleep(min(0.25, max(0.05, stability_seconds / 4)))
    raise CollectionError(f"no new stable image within {wait_seconds:g}s ({last_reason})")


def _root_kind(role: str) -> str:
    if role in {"train", "validation"}:
        return "development"
    return role


def _designation(role: str) -> str:
    if role == "protected_test":
        return "protected_test"
    if role in {"train", "validation"}:
        return "development"
    return "rehearsal"


def _root_paths(root: Path) -> dict[str, Path]:
    return {
        "marker": root / ".vial_guard_collection_root.json",
        "campaign": root / ".vial_guard_campaign.json",
        "campaign_lock": root / ".vial_guard_campaign.lock",
        "commitments": root / ".session_commitments",
        "capture_digests": root / ".capture_sha256_reservations.csv",
        "development_closed": root / ".development_frozen_for_protected.json",
        "development_sealed_sessions": root / ".development_sealed_sessions.json",
        "development_handoff": root / ".development_handoff",
    }


@contextmanager
def _lock_collection_roots(roots: Iterable[Path]):
    """Serialize root-wide commitments without ever following a lock symlink."""
    streams = []
    unique = sorted({str(root): root for root in roots}.values(), key=lambda item: str(item))
    try:
        for root in unique:
            lock = _root_paths(root)["campaign_lock"]
            require(lock.is_file() and not lock.is_symlink(), f"missing or unsafe campaign lock: {lock}")
            stream = lock.open("r+")
            fcntl.flock(stream, fcntl.LOCK_EX)
            streams.append(stream)
        yield
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()


def _validate_campaign_manifest(value: Mapping[str, object], *, root_kind: str,
                                baseline_sha256: str | None = None) -> None:
    require(value.get("schema_version") == 1, "campaign manifest schema_version must be 1")
    require(value.get("collector_version") == COLLECTOR_VERSION,
            "campaign manifest collector version differs")
    require(value.get("root_kind") == root_kind,
            f"campaign manifest is for {value.get('root_kind')}, not {root_kind}")
    campaign_id = value.get("campaign_id")
    require(isinstance(campaign_id, str) and SAFE_ID_RE.fullmatch(campaign_id) is not None,
            "campaign manifest has an invalid campaign_id")
    manifest_baseline = value.get("baseline_freeze_sha256")
    _hash_value(manifest_baseline, "campaign baseline_freeze_sha256")
    if baseline_sha256 is not None:
        require(manifest_baseline == baseline_sha256,
                "official collection root is locked to a different baseline freeze")
    require(tuple(value.get("canonical_classes", ())) == CANONICAL_CLASSES,
            "campaign manifest taxonomy differs from the exact five-class order")
    parse_utc(str(value.get("created_at_utc", "")), "campaign created_at_utc")


def _read_campaign_manifest(root: Path, *, root_kind: str,
                            baseline_sha256: str | None = None) -> dict:
    value = read_json(_root_paths(root)["campaign"])
    _validate_campaign_manifest(value, root_kind=root_kind, baseline_sha256=baseline_sha256)
    return value


def _ensure_official_campaign(root: Path, role: str, baseline: Mapping[str, str],
                              protected_handoff: Path | None) -> dict[str, str]:
    """Create once, then validate the immutable baseline-bound campaign record."""
    baseline_sha256 = _hash_value(
        baseline.get("freeze_file_sha256"), "baseline freeze_file_sha256"
    )
    handoff: dict[str, object] | None = None
    if role == "protected_test":
        require(protected_handoff is not None,
                "protected collection requires a frozen development handoff")
        handoff = _validate_protected_handoff(protected_handoff, baseline_sha256=baseline_sha256)
    with _lock_collection_roots([root]):
        root_paths = _root_paths(root)
        if role in {"train", "validation"}:
            require(not root_paths["development_closed"].exists(),
                    "development collection is frozen because protected collection has begun")
            if root_paths["campaign"].exists():
                campaign = _read_campaign_manifest(
                    root, root_kind="development", baseline_sha256=baseline_sha256
                )
            else:
                campaign = {
                    "schema_version": 1,
                    "collector_version": COLLECTOR_VERSION,
                    "campaign_id": f"campaign-{secrets.token_hex(12)}",
                    "root_kind": "development",
                    "baseline_id": baseline["baseline_id"],
                    "baseline_freeze_sha256": baseline_sha256,
                    "canonical_classes": list(CANONICAL_CLASSES),
                    "created_at_utc": utc_now(),
                }
                write_text_exclusive(
                    root_paths["campaign"],
                    json.dumps(campaign, indent=2, sort_keys=True) + "\n",
                    mode=0o440,
                )
        else:
            require(handoff is not None, "protected campaign has no validated handoff")
            installed = _install_protected_handoff(root, Path(str(handoff["path"])))
            development_campaign = installed["campaign"]
            require(isinstance(development_campaign, dict), "handoff campaign is invalid")
            if root_paths["campaign"].exists():
                campaign = _read_campaign_manifest(
                    root, root_kind="protected_test", baseline_sha256=baseline_sha256
                )
                require(campaign.get("campaign_id") == development_campaign["campaign_id"],
                        "protected root belongs to a different collection campaign")
                require(campaign.get("development_campaign_manifest_sha256") ==
                        installed["campaign_sha256"],
                        "protected campaign no longer matches the development campaign manifest")
                require(campaign.get("development_capture_reservations_sha256") ==
                        installed["capture_reservations_sha256"] and
                        campaign.get("development_sealed_sessions_manifest_sha256") ==
                        installed["sealed_sessions_manifest_sha256"] and
                        campaign.get("development_handoff_receipt_sha256") ==
                        installed["receipt_sha256"],
                        "protected campaign differs from the frozen development handoff")
            else:
                campaign = {
                    "schema_version": 1,
                    "collector_version": COLLECTOR_VERSION,
                    "campaign_id": development_campaign["campaign_id"],
                    "root_kind": "protected_test",
                    "baseline_id": baseline["baseline_id"],
                    "baseline_freeze_sha256": baseline_sha256,
                    "canonical_classes": list(CANONICAL_CLASSES),
                    "development_campaign_manifest_sha256": installed["campaign_sha256"],
                    "development_capture_reservations_sha256":
                        installed["capture_reservations_sha256"],
                    "development_sealed_sessions_manifest_sha256":
                        installed["sealed_sessions_manifest_sha256"],
                    "development_handoff_receipt_sha256": installed["receipt_sha256"],
                    "created_at_utc": utc_now(),
                }
                write_text_exclusive(
                    root_paths["campaign"],
                    json.dumps(campaign, indent=2, sort_keys=True) + "\n",
                    mode=0o400,
                )
    result = {
        "campaign_id": str(campaign["campaign_id"]),
        "baseline_freeze_sha256": baseline_sha256,
        "campaign_manifest_sha256": sha256_file(_root_paths(root)["campaign"]),
    }
    if role == "protected_test":
        result["development_handoff_receipt_sha256"] = str(
            campaign["development_handoff_receipt_sha256"]
        )
        result["development_capture_reservations_sha256"] = str(
            campaign["development_capture_reservations_sha256"]
        )
        result["development_sealed_sessions_manifest_sha256"] = str(
            campaign["development_sealed_sessions_manifest_sha256"]
        )
    return result


def prepare_output_root(root: Path, role: str) -> Path:
    root = resolve_without_symlinks(root, must_exist=False)
    marker = _root_paths(root)["marker"]
    kind = _root_kind(role)
    if root.exists():
        require(root.is_dir() and not root.is_symlink(), "output root must be a real directory")
        entries = list(root.iterdir())
        if marker.exists():
            value = read_json(marker)
            require(value.get("root_kind") == kind,
                    f"output root is already reserved for {value.get('root_kind')}, not {kind}")
        else:
            require(not entries, "refusing a nonempty output root without a Vial Guard role marker")
    else:
        root.mkdir(parents=True, exist_ok=False, mode=0o700 if role == "protected_test" else 0o750)
    if role == "protected_test":
        require(stat.S_IMODE(root.stat().st_mode) & 0o077 == 0,
                "protected output root must be private (mode 0700; no group/other permissions)")
    if not marker.exists():
        write_text_exclusive(
            marker,
            json.dumps({"schema_version": 1, "root_kind": kind, "created_at_utc": utc_now()},
                       indent=2, sort_keys=True) + "\n",
            mode=0o600 if role == "protected_test" else 0o640,
        )
    root_paths = _root_paths(root)
    campaign_lock = root_paths["campaign_lock"]
    if campaign_lock.exists():
        require(campaign_lock.is_file() and not campaign_lock.is_symlink(),
                "campaign lock is unsafe")
    else:
        write_text_exclusive(campaign_lock, "", mode=0o600)
    capture_digests = root_paths["capture_digests"]
    if capture_digests.exists():
        read_csv(capture_digests, CAPTURE_DIGEST_COLUMNS)
    else:
        write_csv_exclusive(capture_digests, CAPTURE_DIGEST_COLUMNS, [])
    commitments = root_paths["commitments"]
    if commitments.exists():
        require(commitments.is_dir() and not commitments.is_symlink(),
                "session commitment directory is unsafe")
    else:
        commitments.mkdir(mode=0o700, exist_ok=False)
    return root


def _validate_capture_digest_records(rows: Sequence[Mapping[str, str]],
                                     allowed_roles: set[str]) -> None:
    seen: set[str] = set()
    for row in rows:
        require(SHA256_RE.fullmatch(row["sha256"]) is not None,
                "capture reservation has an invalid SHA-256")
        require(row["sha256"] not in seen,
                "capture reservation ledger contains a duplicate SHA-256")
        seen.add(row["sha256"])
        require(row["role"] in allowed_roles,
                "capture reservation role crosses the collection-root boundary")
        require(SAFE_ID_RE.fullmatch(row["session_id"]) is not None,
                "capture reservation has an unsafe session ID")
        require(SAFE_ID_RE.fullmatch(row["event_id"]) is not None,
                "capture reservation has an unsafe event ID")
        parse_utc(row["reserved_at_utc"], "capture reserved_at_utc")


def _handoff_checksum_map(path: Path) -> dict[str, str]:
    checksum_path = path / HANDOFF_CHECKSUMS_NAME
    require(checksum_path.is_file() and not checksum_path.is_symlink(),
            "protected handoff checksum file is missing or unsafe")
    values: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        require(len(words) == 2 and SHA256_RE.fullmatch(words[0]) is not None,
                "protected handoff has an invalid SHA256SUMS line")
        name = words[1].lstrip("*")
        require(Path(name).name == name and name not in values,
                "protected handoff checksum has an unsafe or duplicate filename")
        values[name] = words[0]
    expected = HANDOFF_FILES - {HANDOFF_CHECKSUMS_NAME}
    require(set(values) == expected, "protected handoff SHA256SUMS membership differs")
    for name, digest in values.items():
        require(sha256_file(path / name) == digest,
                f"protected handoff checksum failed: {name}")
    return values


def _sealed_session_entry(session: Path, session_id: str, role: str,
                          commitment_sha256: str) -> dict[str, str]:
    require(session.name == session_id and SAFE_ID_RE.fullmatch(session_id) is not None,
            "sealed-session evidence has an invalid session identity")
    require(role in {"train", "validation"},
            "sealed-session evidence has an invalid development role")
    entry = {
        "session_id": session_id,
        "role": role,
        "seal_status": "COMPLETE",
        "session_commitment_sha256": _hash_value(
            commitment_sha256, "session commitment SHA-256"
        ),
    }
    for field, relative in SEALED_SESSION_HASH_PATHS:
        artifact = session / relative
        require(artifact.is_file() and not artifact.is_symlink(),
                f"{session_id}: missing or unsafe sealed artifact {relative}")
        entry[field] = sha256_file(artifact)
    return entry


def _validate_sealed_sessions_manifest(value: Mapping[str, object], *, campaign_id: str,
                                       baseline_sha256: str) -> list[dict[str, str]]:
    require(value.get("schema_version") == 1 and
            value.get("collector_version") == COLLECTOR_VERSION,
            "development sealed-session manifest schema is invalid")
    require(value.get("campaign_id") == campaign_id and
            value.get("baseline_freeze_sha256") == baseline_sha256,
            "development sealed-session manifest belongs to a different campaign")
    parse_utc(str(value.get("created_at_utc", "")), "sealed-session manifest created_at_utc")
    raw_sessions = value.get("sessions")
    require(isinstance(raw_sessions, list) and raw_sessions,
            "development sealed-session manifest has no sessions")
    required = {
        "session_id", "role", "seal_status", "session_commitment_sha256",
        *(field for field, _ in SEALED_SESSION_HASH_PATHS),
    }
    sessions: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_sessions:
        require(isinstance(raw, dict) and set(raw) == required,
                "development sealed-session manifest entry schema differs")
        entry = {key: str(raw[key]) for key in required}
        session_id = entry["session_id"]
        require(SAFE_ID_RE.fullmatch(session_id) is not None and session_id not in seen,
                "development sealed-session manifest has an unsafe or duplicate session ID")
        seen.add(session_id)
        require(entry["role"] in {"train", "validation"} and
                entry["seal_status"] == "COMPLETE",
                "development sealed-session manifest includes non-COMPLETE evidence")
        for field in required - {"session_id", "role", "seal_status"}:
            _hash_value(entry[field], f"sealed-session manifest {session_id}.{field}")
        sessions.append(entry)
    require({entry["role"] for entry in sessions} == {"train", "validation"},
            "development sealed-session manifest must contain train and validation evidence")
    require(sessions == sorted(sessions, key=lambda entry: entry["session_id"]),
            "development sealed-session manifest sessions are not in canonical order")
    return sessions


def _validate_protected_handoff(path: Path, *, baseline_sha256: str | None = None) -> dict[str, object]:
    path = resolve_without_symlinks(path, must_exist=True)
    require(path.is_dir() and not path.is_symlink(),
            "protected handoff must be a real directory")
    entries = {item.name for item in path.iterdir()}
    require(entries == HANDOFF_FILES,
            "protected handoff contains missing or unexpected files")
    for item in path.iterdir():
        require(item.is_file() and not item.is_symlink(),
                f"protected handoff entry is unsafe: {item.name}")
    _handoff_checksum_map(path)
    campaign_path = path / HANDOFF_CAMPAIGN_NAME
    digests_path = path / HANDOFF_DIGESTS_NAME
    sealed_sessions_path = path / HANDOFF_SEALED_SESSIONS_NAME
    receipt_path = path / HANDOFF_RECEIPT_NAME
    campaign = read_json(campaign_path)
    _validate_campaign_manifest(
        campaign, root_kind="development", baseline_sha256=baseline_sha256
    )
    rows = read_csv(digests_path, CAPTURE_DIGEST_COLUMNS)
    _validate_capture_digest_records(rows, {"train", "validation"})
    sealed_sessions_value = read_json(sealed_sessions_path)
    sealed_sessions = _validate_sealed_sessions_manifest(
        sealed_sessions_value,
        campaign_id=str(campaign["campaign_id"]),
        baseline_sha256=str(campaign["baseline_freeze_sha256"]),
    )
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == 1 and
            receipt.get("collector_version") == COLLECTOR_VERSION and
            receipt.get("handoff_type") == "development-to-protected-metadata-only",
            "protected handoff receipt schema is invalid")
    require(receipt.get("campaign_id") == campaign.get("campaign_id") and
            receipt.get("baseline_freeze_sha256") == campaign.get("baseline_freeze_sha256"),
            "protected handoff receipt belongs to a different campaign")
    require(receipt.get("development_campaign_manifest_sha256") == sha256_file(campaign_path) and
            receipt.get("capture_reservations_sha256") == sha256_file(digests_path) and
            receipt.get("development_sealed_sessions_manifest_sha256") ==
            sha256_file(sealed_sessions_path),
            "protected handoff receipt hashes differ from its frozen metadata")
    require(set(receipt.get("development_roles", ())) == {"train", "validation"},
            "protected handoff does not attest both train and validation roles")
    require(receipt.get("development_session_count") == len(sealed_sessions) and
            receipt.get("capture_reservation_count") == len(rows),
            "protected handoff receipt counts differ from its frozen manifests")
    _hash_value(receipt.get("development_session_commitments_sha256"),
                "development_session_commitments_sha256")
    parse_utc(str(receipt.get("development_frozen_at_utc", "")),
              "development_frozen_at_utc")
    return {
        "path": str(path),
        "campaign": campaign,
        "campaign_sha256": sha256_file(campaign_path),
        "capture_reservations_sha256": sha256_file(digests_path),
        "sealed_sessions_manifest_sha256": sha256_file(sealed_sessions_path),
        "receipt_sha256": sha256_file(receipt_path),
        "capture_rows": rows,
        "sealed_sessions": sealed_sessions,
    }


def _install_protected_handoff(root: Path, source: Path) -> dict[str, object]:
    source_value = _validate_protected_handoff(source)
    destination = _root_paths(root)["development_handoff"]
    if destination.exists():
        require(destination.is_dir() and not destination.is_symlink(),
                "installed development handoff is unsafe")
        installed = _validate_protected_handoff(destination)
        for field in (
            "campaign_sha256", "capture_reservations_sha256",
            "sealed_sessions_manifest_sha256", "receipt_sha256",
        ):
            require(installed[field] == source_value[field],
                    "protected root is already tied to a different development handoff")
        return installed
    destination.mkdir(mode=0o700, exist_ok=False)
    for name in sorted(HANDOFF_FILES):
        write_bytes_exclusive(destination / name, (source / name).read_bytes(), mode=0o400)
    _fsync_directory(destination)
    return _validate_protected_handoff(destination)


def _session_paths(session: Path) -> dict[str, Path]:
    return {
        "state": session / "session.json",
        "plan": session / "plan.csv",
        "intents": session / "intended_positions.csv",
        "event_cards": session / "records" / "event_cards",
        "event_cards_manifest": (
            session / "records" / "event_cards" / EVENT_CARD_MANIFEST_NAME
        ),
        "measurements": session / "records" / "event_measurements_template.csv",
        "baseline": session / "records" / "baseline_freeze.json",
        "attempts": session / "records" / "capture_attempts.csv",
        "actions": session / "records" / "event_actions.csv",
        "refusals": session / "records" / "capture_refusals.csv",
        "seen": session / "records" / "camera_seen.json",
        "pending": session / "records" / "pending_capture.json",
        "captures": session / "captures",
        "sealed": session / "sealed",
        "lock": session / "records" / "writer.lock",
    }


def _event_card_filename(event: Mapping[str, object]) -> str:
    event_id = str(event.get("event_id", ""))
    family = str(event.get("family", ""))
    require(re.fullmatch(r"E[0-9]{3}", event_id) is not None,
            "event card requires a safe event ID")
    require(family in {name for counts in FAMILY_COUNTS.values() for name in counts},
            "event card requires a known scenario family")
    return f"{event_id}_{family}.md"


def _markdown_cell(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "—"
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _render_event_card(*, role: str, fixture: str, session_id: str,
                       event: Mapping[str, object],
                       intents: Sequence[Mapping[str, object]]) -> str:
    event_id = str(event["event_id"])
    event_uid = str(event["event_uid"])
    require(len(intents) == 30, f"{event_id}: event card requires 30 position rows")
    require({str(row["event_id"]) for row in intents} == {event_id} and
            {str(row["event_uid"]) for row in intents} == {event_uid},
            f"{event_id}: event card intent identity differs from the plan")
    role_label = {
        "train": "Training",
        "validation": "Validation",
        "protected_test": "Protected test",
        "rehearsal": "Rehearsal",
    }[role]
    lines = [
        f"# {role_label} {event_id} — {event['title']}",
        "",
        "> **READ-ONLY SETUP REFERENCE. DO NOT EDIT THIS FILE.** It is checksum-bound to the session.",
        "",
        (
            "> **INTENDED PHYSICAL SETUP ONLY — NOT VISIBLE GROUND TRUTH.** "
            "Actual visible content requires two independent blinded reviews and adjudication."
        ),
        "",
        "## Immutable event reference",
        "",
        f"- Session: `{session_id}`",
        f"- Event UID: `{event_uid}`",
        f"- Role: `{role}`",
        f"- Fixture: `{fixture}`",
        f"- Scenario family: `{event['family']}`",
        f"- Inspection mode: `{event['inspection_mode']}`",
        f"- Fresh physical placement required: `{event['fresh_placement_required']}`",
        "",
        "## Fixture-specific numbering prerequisite",
        "",
        (
            f"Use only the approved physical `{fixture}01`–`{fixture}30` numbering guide that "
            f"matches the authorized fixture-`{fixture}` map. STOP if the guide or map is "
            "missing or ambiguous. Never substitute another fixture's coordinates."
        ),
        "",
        "## Exact planned tray condition",
        "",
        str(event["instructions"]),
        "",
        f"Primary-target summary from the frozen plan: `{event['primary_targets']}`",
        "",
        (
            "The immutable session `plan.csv` and `intended_positions.csv` are authoritative. "
            "If this card ever differs from either file, STOP; do not improvise a setup."
        ),
        "",
        (
            "| Position | Expected occupancy | Intended visual setup | Vial axis | Cap axis | "
            "Cap seating | Sequence context | Setup note |"
        ),
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in sorted(intents, key=lambda item: str(item["position_id"])):
        vial_angle = (
            f"{row['planned_vial_axis_angle_deg']}°"
            if str(row["planned_vial_axis_angle_deg"]).strip() else ""
        )
        cap_angle = (
            f"{row['planned_cap_axis_angle_deg']}°"
            if str(row["planned_cap_axis_angle_deg"]).strip() else ""
        )
        lines.append("| " + " | ".join(_markdown_cell(value) for value in (
            row["position_id"],
            row["expected_occupancy"],
            row["intended_visual_class"],
            vial_angle,
            cap_angle,
            row["planned_cap_seating_state"],
            row["intended_gt_sequence"],
            row["notes"],
        )) + " |")
    lines.extend([
        "",
        "## Required gate before typing `READY`",
        "",
        "The operator must verify every item below; this reference is not a writable checklist:",
        "",
        "- This is a fresh physical placement for this event.",
        "- The collector-displayed targets, the frozen plan, and the physical tray agree.",
        (
            "- Applicable primary-target angles, method, uncertainty, measurer, and UTC time "
            "were recorded contemporaneously in the authorized measurement record."
        ),
        "- Motion has stopped and hands, robot arm, and tools are clear.",
        "- Camera, mount, lighting, and approved settings are unchanged.",
        "- Exactly one new baseline-Huffman JPEG or non-interlaced PNG was saved to the camera directory.",
        "",
        (
            "Type uppercase `READY` only after all applicable requirements are true. Use `q` "
            "to pause or `s` to skip with a reason. A retake may not move or restage any vial "
            "or cap; a changed physical state requires a new event/session."
        ),
        "",
        (
            "The collector records capture times, filenames, hashes, selections, retakes, and "
            "skips. Do not write runtime results into this integrity-bound setup card."
        ),
    ])
    return "\n".join(lines) + "\n"


def _write_event_cards(*, session: Path, role: str, fixture: str,
                       plan: Sequence[Mapping[str, object]],
                       intents: Sequence[Mapping[str, object]],
                       plan_sha256: str, intents_sha256: str) -> tuple[Path, str]:
    paths = _session_paths(session)
    cards_dir = paths["event_cards"]
    directory_mode = 0o700 if role == "protected_test" else 0o750
    file_mode = 0o400 if role == "protected_test" else 0o440
    cards_dir.mkdir(mode=directory_mode, exist_ok=False)
    os.chmod(cards_dir, directory_mode)
    by_event: dict[str, list[Mapping[str, object]]] = {}
    for row in intents:
        by_event.setdefault(str(row["event_id"]), []).append(row)
    entries: list[dict[str, object]] = []
    for event in plan:
        event_id = str(event["event_id"])
        filename = _event_card_filename(event)
        content = _render_event_card(
            role=role,
            fixture=fixture,
            session_id=session.name,
            event=event,
            intents=by_event.get(event_id, []),
        ).encode("utf-8")
        card = cards_dir / filename
        write_bytes_exclusive(card, content, mode=file_mode)
        os.chmod(card, file_mode)
        entries.append({
            "path": filename,
            "event_id": event_id,
            "event_uid": str(event["event_uid"]),
            "family": str(event["family"]),
            "bytes": len(content),
            "sha256": sha256_bytes(content),
        })
    manifest = {
        "schema_version": 1,
        "collector_version": COLLECTOR_VERSION,
        "session_id": session.name,
        "role": role,
        "fixture_id": fixture,
        "source_plan_sha256": plan_sha256,
        "source_intended_positions_sha256": intents_sha256,
        "card_count": len(entries),
        "files": entries,
        "warning": (
            "Read-only intended-setup references; intent is not visible ground truth and cards must not enter "
            "blinded review packets."
        ),
    }
    manifest_path = paths["event_cards_manifest"]
    write_text_exclusive(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        mode=file_mode,
    )
    os.chmod(manifest_path, file_mode)
    _fsync_directory(cards_dir)
    return manifest_path, sha256_file(manifest_path)


def _validate_event_cards(*, state: Mapping[str, object], paths: Mapping[str, Path],
                          plan: Sequence[Mapping[str, object]],
                          intents: Sequence[Mapping[str, object]]) -> None:
    role = str(state.get("role", ""))
    fixture = str(state.get("fixture_id", ""))
    session_id = str(state.get("session_id", ""))
    cards_dir = paths["event_cards"]
    manifest_path = paths["event_cards_manifest"]
    require(cards_dir.is_dir() and not cards_dir.is_symlink(),
            "missing or unsafe event-card directory")
    require(manifest_path.is_file() and not manifest_path.is_symlink(),
            "missing or unsafe event-card manifest")
    expected_directory_mode = 0o700 if role == "protected_test" else 0o750
    expected_file_mode = 0o400 if role == "protected_test" else 0o440
    require(stat.S_IMODE(cards_dir.stat().st_mode) == expected_directory_mode,
            "event-card directory permissions differ from the session role")
    require(stat.S_IMODE(manifest_path.stat().st_mode) == expected_file_mode,
            "event-card manifest permissions differ from the session role")
    expected_manifest_relpath = f"records/event_cards/{EVENT_CARD_MANIFEST_NAME}"
    require(state.get("event_cards_manifest_relpath") == expected_manifest_relpath and
            state.get("event_cards_manifest_sha256") == sha256_file(manifest_path),
            "event-card manifest differs from the immutable session state")
    require(state.get("event_cards_count") == len(plan),
            "event-card count differs from the immutable session state")
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == 1 and
            manifest.get("collector_version") == COLLECTOR_VERSION and
            manifest.get("session_id") == session_id and
            manifest.get("role") == role and
            manifest.get("fixture_id") == fixture,
            "event-card manifest crosses immutable session identity")
    require(manifest.get("source_plan_sha256") == state.get("plan_sha256") and
            manifest.get("source_intended_positions_sha256") ==
            state.get("intended_positions_sha256"),
            "event-card manifest differs from its frozen plan or intent source")
    entries = manifest.get("files")
    require(isinstance(entries, list) and manifest.get("card_count") == len(plan) == len(entries),
            "event-card manifest has the wrong number of entries")
    by_event: dict[str, list[Mapping[str, object]]] = {}
    for row in intents:
        by_event.setdefault(str(row["event_id"]), []).append(row)
    expected_names = {_event_card_filename(event) for event in plan}
    actual_names = {item.name for item in cards_dir.iterdir()}
    require(actual_names == expected_names | {EVENT_CARD_MANIFEST_NAME},
            "event-card directory membership differs from the frozen plan")
    entries_by_path: dict[str, Mapping[str, object]] = {}
    for value in entries:
        require(isinstance(value, dict), "event-card manifest entry must be an object")
        relative = str(value.get("path", ""))
        require(relative in expected_names and relative not in entries_by_path,
                "event-card manifest has an unknown or duplicate path")
        entries_by_path[relative] = value
    require(set(entries_by_path) == expected_names,
            "event-card manifest membership differs from the frozen plan")
    for event in plan:
        filename = _event_card_filename(event)
        card = cards_dir / filename
        require(card.is_file() and not card.is_symlink(),
                f"missing or unsafe event card: {filename}")
        require(stat.S_IMODE(card.stat().st_mode) == expected_file_mode,
                f"event-card permissions differ from the session role: {filename}")
        expected_content = _render_event_card(
            role=role,
            fixture=fixture,
            session_id=session_id,
            event=event,
            intents=by_event.get(str(event["event_id"]), []),
        ).encode("utf-8")
        entry = entries_by_path[filename]
        require(entry.get("event_id") == event["event_id"] and
                entry.get("event_uid") == event["event_uid"] and
                entry.get("family") == event["family"],
                f"event-card manifest identity mismatch: {filename}")
        require(entry.get("bytes") == card.stat().st_size == len(expected_content) and
                entry.get("sha256") == sha256_file(card) == sha256_bytes(expected_content),
                f"event card changed after session initialization: {filename}")


def initialize_session(*, role: str, camera_dir: Path, output_root: Path, fixture: str,
                       expected_resolution: tuple[int, int], operator_id: str, camera_id: str,
                       baseline_freeze: Path | None, protected_handoff: Path | None,
                       session_id: str | None = None) -> Path:
    require(role in ROLES, f"role must be one of {', '.join(ROLES)}")
    require(fixture in FIXTURES, f"fixture must be one of {', '.join(FIXTURES)}")
    require(operator_id.strip(), "operator ID is required")
    require(camera_id.strip(), "camera ID/profile is required")
    created_at = datetime.now(timezone.utc)
    if session_id is None:
        stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{role}-{stamp}-{secrets.token_hex(3)}"
    require(SAFE_ID_RE.fullmatch(session_id) is not None,
            "session ID may contain only letters, numbers, dot, underscore, and hyphen")
    camera_dir = resolve_without_symlinks(camera_dir, must_exist=True)
    require(camera_dir.is_dir() and not camera_dir.is_symlink(), "camera path must be a real directory")
    camera_identity = camera_dir.stat()
    output_candidate = resolve_without_symlinks(output_root, must_exist=False)
    require(not paths_overlap(camera_dir, output_candidate),
            "camera source and evidence output must not overlap")
    planned_session = output_candidate / session_id
    require(not planned_session.exists() and not planned_session.is_symlink(),
            f"session already exists: {planned_session}")
    resolved_handoff: Path | None = None
    if role == "protected_test":
        require(protected_handoff is not None,
                "protected collection requires a frozen metadata handoff")
        resolved_handoff = resolve_without_symlinks(protected_handoff, must_exist=True)
        require(not paths_overlap(camera_dir, resolved_handoff),
                "protected camera source must not overlap the development handoff")
        require(not paths_overlap(output_candidate, resolved_handoff),
                "protected evidence output must not overlap the development handoff")
    else:
        require(protected_handoff is None,
                "--protected-handoff is valid only for protected_test collection")
    baseline: dict[str, str] | None = None
    baseline_path: Path | None = None
    campaign: dict[str, str] | None = None
    if role in OFFICIAL_ROLES:
        require(baseline_freeze is not None,
                "official train/validation/protected collection requires --baseline-freeze")
        baseline_path = resolve_without_symlinks(baseline_freeze, must_exist=True)
        baseline = validate_baseline_freeze(baseline_path, fixture, created_at)
        if role == "protected_test":
            require(resolved_handoff is not None, "protected collection has no validated handoff")
            _validate_protected_handoff(
                resolved_handoff, baseline_sha256=baseline["freeze_file_sha256"]
            )
    else:
        require(fixture == "G", "rehearsal is limited to the currently calibrated G fixture")
    root = prepare_output_root(output_candidate, role)
    if baseline is not None:
        campaign = _ensure_official_campaign(root, role, baseline, resolved_handoff)
    session = root / session_id
    session.mkdir(mode=0o750, exist_ok=False)
    paths = _session_paths(session)
    paths["captures"].mkdir(mode=0o750)
    (session / "records").mkdir(mode=0o750)
    paths["sealed"].mkdir(mode=0o750)
    paths["lock"].touch(mode=0o600, exist_ok=False)

    seed_hex = secrets.token_hex(32) if role == "protected_test" else hashlib.sha256(
        f"{COLLECTOR_VERSION}|{role}|{fixture}|{session_id}".encode("utf-8")
    ).hexdigest()
    plan, intents = build_plan(role, fixture, session_id, seed_hex)
    write_csv_exclusive(paths["plan"], PLAN_COLUMNS, plan)
    write_csv_exclusive(paths["intents"], INTENT_COLUMNS, intents)
    plan_sha256 = sha256_file(paths["plan"])
    intended_positions_sha256 = sha256_file(paths["intents"])
    event_cards_manifest, event_cards_manifest_sha256 = _write_event_cards(
        session=session,
        role=role,
        fixture=fixture,
        plan=plan,
        intents=intents,
        plan_sha256=plan_sha256,
        intents_sha256=intended_positions_sha256,
    )
    measurements = [
        {
            "event_uid": row["event_uid"],
            "fixture_id": row["fixture_id"],
            "position_id": row["position_id"],
            "planned_vial_axis_angle_deg": row["planned_vial_axis_angle_deg"],
            "planned_cap_axis_angle_deg": row["planned_cap_axis_angle_deg"],
            "actual_vial_axis_angle_deg": "",
            "actual_cap_axis_angle_deg": "",
            "actual_cap_to_vial_misalignment_deg": "",
            "actual_cap_seating_state": "",
            "measurement_method": "",
            "measurement_uncertainty_deg": "",
            "measured_by": "",
            "measured_at_utc": "",
            "notes": "Complete outside the immutable raw-capture workflow.",
        }
        for row in intents
    ]
    write_csv_exclusive(paths["measurements"], MEASUREMENT_COLUMNS, measurements)
    write_csv_exclusive(paths["attempts"], ATTEMPT_COLUMNS, [])
    write_csv_exclusive(paths["actions"], ACTION_COLUMNS, [])
    write_csv_exclusive(paths["refusals"], REFUSAL_COLUMNS, [])
    if baseline_path is not None and baseline is not None:
        write_bytes_exclusive(
            paths["baseline"], baseline_path.read_bytes(),
            mode=0o400 if role == "protected_test" else 0o440,
        )
        require(sha256_file(paths["baseline"]) == baseline["freeze_file_sha256"],
                "retained baseline freeze differs from the validated source")
    atomic_json(paths["seen"], camera_snapshot(camera_dir))
    state = {
        "schema_version": SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "session_id": session_id,
        "role": role,
        "root_kind": _root_kind(role),
        "designation": _designation(role),
        "fixture_id": fixture,
        "created_at_utc": created_at.isoformat().replace("+00:00", "Z"),
        "operator_id": operator_id.strip(),
        "camera_id": camera_id.strip(),
        "camera_dir": str(camera_dir),
        "camera_dir_identity": {
            "st_dev": camera_identity.st_dev,
            "st_ino": camera_identity.st_ino,
        },
        "expected_resolution": list(expected_resolution),
        "plan_seed_sha256": sha256_bytes(seed_hex.encode("utf-8")),
        "plan_sha256": plan_sha256,
        "intended_positions_sha256": intended_positions_sha256,
        "event_cards_manifest_relpath": event_cards_manifest.relative_to(session).as_posix(),
        "event_cards_manifest_sha256": event_cards_manifest_sha256,
        "event_cards_count": len(plan),
        "measurement_template_sha256": sha256_file(paths["measurements"]),
        "baseline_freeze": baseline,
        "baseline_freeze_relpath": (
            "records/baseline_freeze.json" if baseline is not None else None
        ),
        "campaign": campaign,
        "development_handoff": ({
            "receipt_sha256": campaign["development_handoff_receipt_sha256"],
            "capture_reservations_sha256": campaign["development_capture_reservations_sha256"],
            "sealed_sessions_manifest_sha256":
                campaign["development_sealed_sessions_manifest_sha256"],
        } if campaign is not None and role == "protected_test" else None),
        "capture_time_semantics": (
            "collector ingestion UTC after a newly observed stable camera file; source filesystem mtime is also logged"
        ),
        "label_semantics": (
            "scenario rows are intent only; actual visible labels require two-reviewer independent adjudication"
        ),
        "inference_performed": False,
        "manufacturing_control_performed": False,
    }
    write_text_exclusive(paths["state"], json.dumps(state, indent=2, sort_keys=True) + "\n")
    commitment = {
        "schema_version": 1,
        "collector_version": COLLECTOR_VERSION,
        "session_id": session_id,
        "role": role,
        "root_kind": _root_kind(role),
        "designation": _designation(role),
        "session_state_sha256": sha256_file(paths["state"]),
        "root_marker_sha256": sha256_file(_root_paths(root)["marker"]),
        "campaign_id": campaign["campaign_id"] if campaign is not None else None,
        "campaign_manifest_sha256": (
            campaign["campaign_manifest_sha256"] if campaign is not None else None
        ),
        "baseline_freeze_sha256": (
            baseline["freeze_file_sha256"] if baseline is not None else None
        ),
        "development_handoff_receipt_sha256": (
            campaign.get("development_handoff_receipt_sha256")
            if campaign is not None else None
        ),
        "development_sealed_sessions_manifest_sha256": (
            campaign.get("development_sealed_sessions_manifest_sha256")
            if campaign is not None else None
        ),
        "plan_sha256": state["plan_sha256"],
        "intended_positions_sha256": state["intended_positions_sha256"],
        "event_cards_manifest_sha256": state["event_cards_manifest_sha256"],
        "event_cards_count": state["event_cards_count"],
        "committed_at_utc": utc_now(),
    }
    commitment_path = _root_paths(root)["commitments"] / f"{session_id}.json"
    write_text_exclusive(
        commitment_path,
        json.dumps(commitment, indent=2, sort_keys=True) + "\n",
        mode=0o400 if role == "protected_test" else 0o440,
    )
    return session


def _require_safe_session_tree(session: Path) -> None:
    require(session.is_dir() and not session.is_symlink(), "session path must be a real directory")
    for required_directory in (session / "records", session / "captures", session / "sealed"):
        require(required_directory.is_dir() and not required_directory.is_symlink(),
                f"missing or unsafe session directory: {required_directory.name}")
    for item in session.rglob("*"):
        require(not item.is_symlink(), f"symlink inside session evidence refused: {item}")


def load_session(session: Path) -> tuple[dict, dict[str, Path]]:
    session = resolve_without_symlinks(session, must_exist=True)
    _require_safe_session_tree(session)
    paths = _session_paths(session)
    for required_file in (
        paths["state"], paths["plan"], paths["intents"], paths["measurements"],
        paths["attempts"], paths["actions"], paths["refusals"], paths["seen"], paths["lock"],
        paths["event_cards_manifest"],
    ):
        require(required_file.is_file() and not required_file.is_symlink(),
                f"missing or unsafe session file: {required_file}")
    root = resolve_without_symlinks(session.parent, must_exist=True)
    root_paths = _root_paths(root)
    require(root_paths["commitments"].is_dir() and not root_paths["commitments"].is_symlink(),
            "missing or unsafe session commitment directory")
    require(root_paths["campaign_lock"].is_file() and not root_paths["campaign_lock"].is_symlink(),
            "missing or unsafe campaign lock")
    read_csv(root_paths["capture_digests"], CAPTURE_DIGEST_COLUMNS)
    marker = read_json(root_paths["marker"])
    state = read_json(paths["state"])
    require(state.get("schema_version") == SCHEMA_VERSION, "unsupported session schema")
    require(state.get("collector_version") == COLLECTOR_VERSION, "collector version differs from session")
    role = state.get("role")
    require(role in ROLES, "session has an invalid role")
    require(session.name == state.get("session_id"), "session directory and immutable ID differ")
    expected_root_kind = _root_kind(str(role))
    require(state.get("root_kind") == expected_root_kind and marker.get("root_kind") == expected_root_kind,
            "session role and collection-root designation differ")
    require(state.get("designation") == _designation(str(role)),
            "session role and immutable designation differ")
    commitment_path = root_paths["commitments"] / f"{session.name}.json"
    commitment = read_json(commitment_path)
    require(commitment.get("schema_version") == 1 and
            commitment.get("collector_version") == COLLECTOR_VERSION,
            "unsupported external session commitment")
    require(commitment.get("session_id") == state.get("session_id") and
            commitment.get("role") == role and
            commitment.get("root_kind") == expected_root_kind and
            commitment.get("designation") == state.get("designation"),
            "session identity or role differs from its external commitment")
    require(commitment.get("session_state_sha256") == sha256_file(paths["state"]),
            "session.json differs from its external session commitment")
    require(commitment.get("root_marker_sha256") == sha256_file(root_paths["marker"]),
            "collection-root marker differs from the session commitment")
    require(sha256_file(paths["plan"]) == state.get("plan_sha256"), "plan changed after session initialization")
    require(sha256_file(paths["intents"]) == state.get("intended_positions_sha256"),
            "intended-position records changed after initialization")
    require(commitment.get("plan_sha256") == state.get("plan_sha256") and
            commitment.get("intended_positions_sha256") == state.get("intended_positions_sha256"),
            "session plan hashes differ from the external commitment")
    require(commitment.get("event_cards_manifest_sha256") ==
            state.get("event_cards_manifest_sha256") and
            commitment.get("event_cards_count") == state.get("event_cards_count"),
            "event-card evidence differs from the external commitment")
    plan = read_csv(paths["plan"], PLAN_COLUMNS)
    intents = read_csv(paths["intents"], INTENT_COLUMNS)
    _validate_event_cards(state=state, paths=paths, plan=plan, intents=intents)
    baseline = state.get("baseline_freeze")
    campaign = state.get("campaign")
    if role in OFFICIAL_ROLES:
        require(isinstance(baseline, dict) and isinstance(campaign, dict),
                "official session is missing baseline or campaign evidence")
        baseline_sha256 = _hash_value(
            baseline.get("freeze_file_sha256"), "session baseline freeze_file_sha256"
        )
        require(state.get("baseline_freeze_relpath") == "records/baseline_freeze.json" and
                paths["baseline"].is_file() and not paths["baseline"].is_symlink() and
                sha256_file(paths["baseline"]) == baseline_sha256,
                "retained baseline freeze is missing or differs from the session commitment")
        campaign_manifest = _read_campaign_manifest(
            root, root_kind=expected_root_kind, baseline_sha256=baseline_sha256
        )
        campaign_sha256 = sha256_file(root_paths["campaign"])
        require(campaign.get("campaign_id") == campaign_manifest.get("campaign_id") and
                campaign.get("baseline_freeze_sha256") == baseline_sha256 and
                campaign.get("campaign_manifest_sha256") == campaign_sha256,
                "session campaign evidence differs from the root campaign manifest")
        require(commitment.get("campaign_id") == campaign_manifest.get("campaign_id") and
                commitment.get("campaign_manifest_sha256") == campaign_sha256 and
                commitment.get("baseline_freeze_sha256") == baseline_sha256,
                "external session commitment differs from the root campaign")
        if role == "protected_test":
            installed = _validate_protected_handoff(
                root_paths["development_handoff"], baseline_sha256=baseline_sha256
            )
            development_campaign = installed["campaign"]
            require(isinstance(development_campaign, dict), "installed handoff campaign is invalid")
            handoff_state = state.get("development_handoff")
            require(isinstance(handoff_state, dict) and
                    handoff_state.get("receipt_sha256") == installed["receipt_sha256"] and
                    handoff_state.get("capture_reservations_sha256") ==
                    installed["capture_reservations_sha256"] and
                    handoff_state.get("sealed_sessions_manifest_sha256") ==
                    installed["sealed_sessions_manifest_sha256"],
                    "protected session differs from its installed development handoff")
            require(development_campaign.get("campaign_id") == campaign_manifest.get("campaign_id") and
                    campaign_manifest.get("development_campaign_manifest_sha256") ==
                    installed["campaign_sha256"] and
                    campaign_manifest.get("development_capture_reservations_sha256") ==
                    installed["capture_reservations_sha256"] and
                    campaign_manifest.get("development_sealed_sessions_manifest_sha256") ==
                    installed["sealed_sessions_manifest_sha256"] and
                    campaign_manifest.get("development_handoff_receipt_sha256") ==
                    installed["receipt_sha256"],
                    "protected session is no longer tied to its frozen development handoff")
            require(commitment.get("development_handoff_receipt_sha256") ==
                    installed["receipt_sha256"] and
                    commitment.get("development_sealed_sessions_manifest_sha256") ==
                    installed["sealed_sessions_manifest_sha256"],
                    "external session commitment differs from the development handoff")
        else:
            require(state.get("development_handoff") is None,
                    "development session unexpectedly claims a protected handoff")
            if root_paths["development_closed"].exists():
                _verify_loaded_development_session_freeze(
                    root=root,
                    session=session,
                    session_id=str(state["session_id"]),
                    role=str(role),
                    baseline_sha256=baseline_sha256,
                    campaign_id=str(campaign_manifest["campaign_id"]),
                    commitment_path=commitment_path,
                )
    else:
        require(baseline is None and campaign is None and state.get("development_handoff") is None and
                state.get("baseline_freeze_relpath") is None and not paths["baseline"].exists(),
                "rehearsal must not claim official baseline or campaign evidence")
        require(commitment.get("campaign_id") is None and
                commitment.get("campaign_manifest_sha256") is None and
                commitment.get("baseline_freeze_sha256") is None and
                commitment.get("development_handoff_receipt_sha256") is None and
                commitment.get("development_sealed_sessions_manifest_sha256") is None,
                "rehearsal commitment must not claim an official campaign")
    return state, paths


def _verify_loaded_development_session_freeze(*, root: Path, session: Path,
                                              session_id: str, role: str,
                                              baseline_sha256: str, campaign_id: str,
                                              commitment_path: Path) -> None:
    root_paths = _root_paths(root)
    receipt = read_json(root_paths["development_closed"])
    manifest_path = root_paths["development_sealed_sessions"]
    require(manifest_path.is_file() and not manifest_path.is_symlink() and
            receipt.get("development_sealed_sessions_manifest_sha256") ==
            sha256_file(manifest_path),
            "development sealed-session manifest differs from the frozen receipt")
    manifest = read_json(manifest_path)
    entries = _validate_sealed_sessions_manifest(
        manifest, campaign_id=campaign_id, baseline_sha256=baseline_sha256
    )
    frozen = next((entry for entry in entries if entry["session_id"] == session_id), None)
    require(frozen is not None and frozen["role"] == role,
            "development session is absent from the frozen sealed-evidence manifest")
    current = _sealed_session_entry(
        session, session_id, role, sha256_file(commitment_path)
    )
    require(current == frozen,
            "development sealed artifacts differ from the frozen handoff manifest")


def _capture_digest_rows(root: Path) -> list[dict[str, str]]:
    root_paths = _root_paths(root)
    rows = read_csv(root_paths["capture_digests"], CAPTURE_DIGEST_COLUMNS)
    marker = read_json(root_paths["marker"])
    kind = marker.get("root_kind")
    if kind == "development" and root_paths["development_closed"].exists():
        closed = read_json(root_paths["development_closed"])
        require(closed.get("capture_reservations_sha256") ==
                sha256_file(root_paths["capture_digests"]),
                "development capture ledger changed after the protected boundary was frozen")
        require(root_paths["development_sealed_sessions"].is_file() and
                not root_paths["development_sealed_sessions"].is_symlink() and
                closed.get("development_sealed_sessions_manifest_sha256") ==
                sha256_file(root_paths["development_sealed_sessions"]),
                "development sealed-session manifest changed after the protected boundary was frozen")
    allowed_roles = {
        "development": {"train", "validation"},
        "protected_test": {"protected_test"},
        "rehearsal": {"rehearsal"},
    }.get(str(kind), set())
    require(bool(allowed_roles), "collection root has an invalid role marker")
    _validate_capture_digest_records(rows, allowed_roles)
    return rows


def _verify_development_sessions_for_freeze(root: Path) -> dict[str, object]:
    commitments = _root_paths(root)["commitments"]
    entries = sorted(commitments.iterdir(), key=lambda item: item.name)
    require(entries, "development campaign has no committed sessions")
    roles: list[str] = []
    commitment_hashes: list[str] = []
    sealed_entries: list[dict[str, str]] = []
    for entry in entries:
        require(entry.is_file() and not entry.is_symlink() and entry.suffix == ".json",
                f"unsafe entry in session commitment directory: {entry.name}")
        commitment = read_json(entry)
        session_id = str(commitment.get("session_id", ""))
        role = str(commitment.get("role", ""))
        require(entry.name == f"{session_id}.json" and SAFE_ID_RE.fullmatch(session_id) is not None,
                "session commitment filename or ID is invalid")
        require(role in {"train", "validation"},
                "development campaign contains a non-development session commitment")
        session = root / session_id
        report = verify_session(session, verify_archive=True)
        require(report.get("role") == role and report.get("sealed") is True and
                report.get("collection_complete") is True,
                f"{session_id}: development session must be complete and sealed before handoff")
        seal = read_json(session / "records" / "seal_manifest.json")
        require(seal.get("seal_status") == "COMPLETE",
                f"{session_id}: PARTIAL evidence cannot enter the protected handoff boundary")
        roles.append(role)
        commitment_hashes.append(sha256_file(entry))
        sealed_entries.append(
            _sealed_session_entry(session, session_id, role, sha256_file(entry))
        )
    require(set(roles) == {"train", "validation"},
            "development handoff requires complete sealed train and validation sessions")
    aggregate = sha256_bytes(("\n".join(sorted(commitment_hashes)) + "\n").encode("ascii"))
    return {
        "development_roles": sorted(set(roles)),
        "development_session_count": len(roles),
        "development_session_commitments_sha256": aggregate,
        "sealed_session_entries": sorted(
            sealed_entries, key=lambda item: item["session_id"]
        ),
    }


def freeze_development_for_protected(development_root: Path, out_dir: Path) -> dict[str, object]:
    """Close development capture and emit a pixel-free, transferable custody handoff."""
    root = resolve_without_symlinks(development_root, must_exist=True)
    require(root.is_dir() and not root.is_symlink(), "development root must be a real directory")
    marker = read_json(_root_paths(root)["marker"])
    require(marker.get("root_kind") == "development",
            "freeze-for-protected requires a marked development collection root")
    output = resolve_without_symlinks(out_dir, must_exist=False)
    require(not output.exists(), "protected handoff output already exists")
    require(not paths_overlap(root, output),
            "protected handoff output must be outside the development evidence root")
    with _lock_collection_roots([root]):
        campaign = _read_campaign_manifest(root, root_kind="development")
        rows = _capture_digest_rows(root)
        verified = _verify_development_sessions_for_freeze(root)
        sealed_entries = verified.get("sealed_session_entries")
        require(isinstance(sealed_entries, list),
                "development freeze verification did not produce sealed-session evidence")
        verified_summary = {
            key: value for key, value in verified.items()
            if key != "sealed_session_entries"
        }
        root_paths = _root_paths(root)
        sealed_manifest_path = root_paths["development_sealed_sessions"]
        if sealed_manifest_path.exists():
            sealed_manifest = read_json(sealed_manifest_path)
            frozen_entries = _validate_sealed_sessions_manifest(
                sealed_manifest,
                campaign_id=str(campaign["campaign_id"]),
                baseline_sha256=str(campaign["baseline_freeze_sha256"]),
            )
            require(frozen_entries == sealed_entries,
                    "existing sealed-session manifest differs from current sealed evidence")
        else:
            sealed_manifest = {
                "schema_version": 1,
                "collector_version": COLLECTOR_VERSION,
                "campaign_id": campaign["campaign_id"],
                "baseline_freeze_sha256": campaign["baseline_freeze_sha256"],
                "created_at_utc": utc_now(),
                "sessions": sealed_entries,
            }
            write_text_exclusive(
                sealed_manifest_path,
                json.dumps(sealed_manifest, indent=2, sort_keys=True) + "\n",
                mode=0o440,
            )
        receipt_path = root_paths["development_closed"]
        expected_static = {
            "schema_version": 1,
            "collector_version": COLLECTOR_VERSION,
            "handoff_type": "development-to-protected-metadata-only",
            "campaign_id": campaign["campaign_id"],
            "baseline_freeze_sha256": campaign["baseline_freeze_sha256"],
            "development_campaign_manifest_sha256": sha256_file(root_paths["campaign"]),
            "capture_reservations_sha256": sha256_file(root_paths["capture_digests"]),
            "capture_reservation_count": len(rows),
            "development_sealed_sessions_manifest_sha256":
                sha256_file(sealed_manifest_path),
            **verified_summary,
        }
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            require(all(receipt.get(key) == value for key, value in expected_static.items()),
                    "existing development freeze receipt differs from current campaign evidence")
        else:
            receipt = {
                **expected_static,
                "development_frozen_at_utc": utc_now(),
                "reason": "development campaign finalized for separate-custodian protected collection",
            }
            write_text_exclusive(
                receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n", mode=0o440
            )

        output.mkdir(parents=True, mode=0o755, exist_ok=False)
        sources = {
            HANDOFF_CAMPAIGN_NAME: root_paths["campaign"],
            HANDOFF_DIGESTS_NAME: root_paths["capture_digests"],
            HANDOFF_SEALED_SESSIONS_NAME: sealed_manifest_path,
            HANDOFF_RECEIPT_NAME: receipt_path,
        }
        for name, source in sources.items():
            write_bytes_exclusive(output / name, source.read_bytes(), mode=0o444)
        checksums = "".join(
            f"{sha256_file(output / name)}  {name}\n" for name in sorted(sources)
        )
        write_text_exclusive(output / HANDOFF_CHECKSUMS_NAME, checksums, mode=0o444)
        _fsync_directory(output)
    handoff = _validate_protected_handoff(output)
    return {
        "campaign_id": campaign["campaign_id"],
        "baseline_freeze_sha256": campaign["baseline_freeze_sha256"],
        "capture_reservation_count": len(rows),
        "handoff_dir": str(output),
        "handoff_receipt_sha256": handoff["receipt_sha256"],
    }


def _reserve_capture_digest(session: Path, state: Mapping[str, object], event: Mapping[str, str],
                            digest: str, capture_name: str) -> None:
    """Atomically refuse image reuse across sessions and frozen split metadata."""
    root = session.parent
    with _lock_collection_roots([root]):
        if state.get("role") in {"train", "validation"}:
            require(not _root_paths(root)["development_closed"].exists(),
                    "development collection is frozen because protected collection has begun")
        existing = _capture_digest_rows(root)
        if state.get("role") == "protected_test":
            handoff = _validate_protected_handoff(_root_paths(root)["development_handoff"])
            handoff_rows = handoff["capture_rows"]
            require(isinstance(handoff_rows, list), "installed handoff digest ledger is invalid")
            existing = [*existing, *handoff_rows]
        for row in existing:
            require(row["sha256"] != digest,
                    "duplicate image SHA-256 already retained in this collection campaign")
        append_csv(_root_paths(root)["capture_digests"], CAPTURE_DIGEST_COLUMNS, {
            "reserved_at_utc": utc_now(),
            "sha256": digest,
            "session_id": state["session_id"],
            "role": state["role"],
            "event_id": event["event_id"],
            "event_uid": event["event_uid"],
            "capture_name": capture_name,
        })


def _attempts(paths: Mapping[str, Path]) -> list[dict[str, str]]:
    return read_csv(paths["attempts"], ATTEMPT_COLUMNS)


def _actions(paths: Mapping[str, Path]) -> list[dict[str, str]]:
    return read_csv(paths["actions"], ACTION_COLUMNS)


@contextmanager
def _session_writer(paths: Mapping[str, Path], *, already_held: bool = False):
    if already_held:
        yield
        return
    with paths["lock"].open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CollectionError("another collector process is already writing this session") from exc
        yield


def selected_attempts(attempts: Sequence[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    selected: dict[str, Mapping[str, str]] = {}
    for row in attempts:
        if row["status"] == "accepted_by_operator":
            selected[row["event_id"]] = row
    return selected


def skipped_events(actions: Sequence[Mapping[str, str]]) -> set[str]:
    return {row["event_id"] for row in actions if row["action"] == "skipped"}


def _quarantine_interrupted_copies(paths: Mapping[str, Path]) -> None:
    interrupted = [
        path for path in paths["captures"].glob(".capture-building-*")
        if path.is_file() and not path.is_symlink()
    ]
    if not interrupted:
        return
    quarantine = paths["captures"].parent / "records" / "interrupted_capture_copies"
    quarantine.mkdir(mode=0o700, exist_ok=True)
    require(quarantine.is_dir() and not quarantine.is_symlink(),
            "interrupted-copy quarantine is unsafe")
    for source in interrupted:
        destination = quarantine / source.name
        require(not destination.exists(), "interrupted-copy quarantine name collision")
        os.replace(source, destination)
        _log_refusal(
            paths, "", source.name,
            "interrupted atomic copy quarantined; it was never committed as a capture attempt",
        )
    _fsync_directory(paths["captures"])
    _fsync_directory(quarantine)


def recover_pending(session: Path, *, _lock_held: bool = False) -> None:
    state, paths = load_session(session)
    with _session_writer(paths, already_held=_lock_held):
        require(not (paths["sealed"] / "evidence.tar").exists(),
                "session is sealed; recovery cannot mutate it")
        _quarantine_interrupted_copies(paths)
        if not paths["pending"].exists():
            return
        pending = read_json(paths["pending"])
        capture_name = str(pending.get("capture_name", ""))
        require(Path(capture_name).name == capture_name, "unsafe pending capture name")
        capture = paths["captures"] / capture_name
        if capture.exists():
            require(capture.is_file() and not capture.is_symlink(), "pending capture is unsafe")
            require(sha256_file(capture) == pending.get("sha256"), "pending capture digest differs")
            existing = _attempts(paths)
            key = (str(pending["event_id"]), str(pending["attempt"]))
            if not any((row["event_id"], row["attempt"]) == key for row in existing):
                pending["status"] = "unconfirmed_recovered"
                pending["confirmation_note"] = "Collector interrupted before human setup confirmation."
                append_csv(paths["attempts"], ATTEMPT_COLUMNS, pending)
        else:
            _log_refusal(
                paths, str(pending.get("event_id", "")), str(pending.get("source_name", "")),
                "collector interrupted before the retained capture was atomically committed",
            )
        paths["pending"].unlink()
        _fsync_directory(paths["pending"].parent)
        require(state.get("role") in ROLES, "session role changed during recovery")


def _log_refusal(paths: Mapping[str, Path], event_id: str, source_name: str, reason: str) -> None:
    append_csv(paths["refusals"], REFUSAL_COLUMNS, {
        "time_utc": utc_now(), "event_id": event_id, "source_name": source_name, "reason": reason,
    })


def ingest_capture(session: Path, event: Mapping[str, str], *, stability_seconds: float,
                   wait_seconds: float, ready_confirmed_at_utc: str | None = None,
                   _lock_held: bool = False) -> dict[str, object]:
    state, paths = load_session(session)
    ready_at = ready_confirmed_at_utc or utc_now()
    parse_utc(ready_at, "ready_confirmed_at_utc")
    if not _lock_held:
        with _session_writer(paths):
            return ingest_capture(
                session, event, stability_seconds=stability_seconds, wait_seconds=wait_seconds,
                ready_confirmed_at_utc=ready_at, _lock_held=True,
            )
    require(not (paths["sealed"] / "evidence.tar").exists(), "session is sealed; capture is closed")
    require(not paths["pending"].exists(), "pending capture must be recovered before another ingest")
    camera = resolve_without_symlinks(Path(str(state["camera_dir"])), must_exist=True)
    require(camera.is_dir() and not camera.is_symlink(), "camera directory is unavailable or unsafe")
    camera_identity = state.get("camera_dir_identity")
    current_camera_identity = camera.stat()
    require(isinstance(camera_identity, dict) and
            camera_identity.get("st_dev") == current_camera_identity.st_dev and
            camera_identity.get("st_ino") == current_camera_identity.st_ino,
            "camera directory identity changed; start a new session after camera/mount replacement")
    seen = read_json(paths["seen"])
    try:
        image, current = wait_for_new_image(camera, seen, stability_seconds, wait_seconds)
    except CollectionError as exc:
        _log_refusal(paths, event["event_id"], "", str(exc))
        raise
    for source_name in sorted(
        name for name, fingerprint in current.items()
        if seen.get(name) != fingerprint and name != image.source_name
    ):
        _log_refusal(
            paths,
            event["event_id"],
            source_name,
            "not selected: an older concurrent camera file; source left untouched",
        )
    expected = tuple(state["expected_resolution"])
    if (image.width, image.height) != expected:
        atomic_json(paths["seen"], current)
        reason = f"resolution {image.width}x{image.height} differs from locked {expected[0]}x{expected[1]}"
        _log_refusal(paths, event["event_id"], image.source_name, reason)
        raise CollectionError(reason)
    digest = sha256_bytes(image.data)
    attempts = _attempts(paths)
    if any(row["sha256"] == digest for row in attempts):
        atomic_json(paths["seen"], current)
        reason = "duplicate image SHA-256 already retained; a new physical event needs a new capture"
        _log_refusal(paths, event["event_id"], image.source_name, reason)
        raise CollectionError(reason)
    attempt_number = 1 + sum(row["event_id"] == event["event_id"] for row in attempts)
    capture_name = f"{event['event_id'].lower()}_attempt_{attempt_number:02d}{image.suffix}"
    try:
        _reserve_capture_digest(session, state, event, digest, capture_name)
    except CollectionError as exc:
        atomic_json(paths["seen"], current)
        _log_refusal(paths, event["event_id"], image.source_name, str(exc))
        raise
    capture = paths["captures"] / capture_name
    capture_time = utc_now()
    pending: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "session_id": state["session_id"],
        "role": state["role"],
        "event_id": event["event_id"],
        "event_uid": event["event_uid"],
        "attempt": str(attempt_number),
        "status": "pending_confirmation",
        "source_name": image.source_name,
        "source_mtime_utc": image.source_mtime_utc,
        "capture_name": capture_name,
        "ready_confirmed_at_utc": ready_at,
        "capture_time_utc": capture_time,
        "sha256": digest,
        "bytes": str(len(image.data)),
        "width": str(image.width),
        "height": str(image.height),
        "fixture_id": state["fixture_id"],
        "family": event["family"],
        "operator_id": state["operator_id"],
        "confirmation_note": "",
        "label_source": "intended_scenario_only",
        "review_state": "pending_independent_review",
        "usable": "false",
    }
    write_text_exclusive(paths["pending"], json.dumps(pending, indent=2, sort_keys=True) + "\n", mode=0o600)
    write_bytes_atomic_exclusive(capture, image.data, mode=0o444)
    require(sha256_file(capture) == digest, "retained capture checksum verification failed")
    atomic_json(paths["seen"], current)
    return pending


def finalize_capture(session: Path, *, accepted: bool, note: str,
                     _lock_held: bool = False) -> dict[str, str]:
    _, paths = load_session(session)
    if not _lock_held:
        with _session_writer(paths):
            return finalize_capture(session, accepted=accepted, note=note, _lock_held=True)
    require(not (paths["sealed"] / "evidence.tar").exists(),
            "session is sealed; capture confirmation is closed")
    require(paths["pending"].is_file(), "there is no pending capture to confirm")
    pending = {key: str(value) for key, value in read_json(paths["pending"]).items()}
    pending["status"] = "accepted_by_operator" if accepted else "rejected_by_operator"
    pending["confirmation_note"] = note.strip()
    append_csv(paths["attempts"], ATTEMPT_COLUMNS, pending)
    paths["pending"].unlink()
    _fsync_directory(paths["pending"].parent)
    return pending


def skip_event(session: Path, event: Mapping[str, str], reason: str,
               *, _lock_held: bool = False) -> None:
    state, paths = load_session(session)
    if not _lock_held:
        with _session_writer(paths):
            skip_event(session, event, reason, _lock_held=True)
            return
    require(not (paths["sealed"] / "evidence.tar").exists(), "session is sealed; event actions are closed")
    require(reason.strip(), "skip reason is required")
    attempts = _attempts(paths)
    require(event["event_id"] not in selected_attempts(attempts), "cannot skip an event with an accepted capture")
    require(event["event_id"] not in skipped_events(_actions(paths)), "event is already skipped")
    append_csv(paths["actions"], ACTION_COLUMNS, {
        "time_utc": utc_now(), "session_id": state["session_id"], "role": state["role"],
        "event_id": event["event_id"], "event_uid": event["event_uid"],
        "action": "skipped", "reason": reason.strip(),
    })


def _validate_attempts(state: Mapping[str, object], paths: Mapping[str, Path],
                       plan: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    attempts = _attempts(paths)
    reservations = _capture_digest_rows(paths["state"].parent.parent)
    reserved = {
        (row["session_id"], row["role"], row["event_id"], row["event_uid"],
         row["capture_name"], row["sha256"])
        for row in reservations
    }
    plan_by_id = {row["event_id"]: row for row in plan}
    names: set[str] = set()
    digests: set[str] = set()
    counts: dict[str, int] = {}
    baseline = state.get("baseline_freeze")
    frozen_at = parse_utc(str(baseline["frozen_at_utc"]), "baseline frozen_at") if isinstance(baseline, dict) else None
    for row in attempts:
        require((row["session_id"], row["role"], row["event_id"], row["event_uid"],
                 row["capture_name"], row["sha256"]) in reserved,
                "capture attempt has no matching root-level SHA-256 reservation")
        require(row["session_id"] == state["session_id"] and row["role"] == state["role"],
                "capture attempt crosses immutable session role")
        require(row["event_id"] in plan_by_id and row["event_uid"] == plan_by_id[row["event_id"]]["event_uid"],
                "capture attempt references an unknown event")
        counts[row["event_id"]] = counts.get(row["event_id"], 0) + 1
        require(row["attempt"] == str(counts[row["event_id"]]), "attempt numbers are not contiguous")
        require(row["status"] in {"accepted_by_operator", "rejected_by_operator", "unconfirmed_recovered"},
                "capture attempt has an invalid status")
        name = row["capture_name"]
        require(Path(name).name == name and name not in names, "unsafe or duplicate retained capture name")
        names.add(name)
        require(SHA256_RE.fullmatch(row["sha256"]) is not None and row["sha256"] not in digests,
                "duplicate or invalid retained capture digest")
        digests.add(row["sha256"])
        capture = paths["captures"] / name
        require(capture.is_file() and not capture.is_symlink(), f"missing or unsafe capture {name}")
        require(str(capture.stat().st_size) == row["bytes"], f"{name}: byte count changed")
        require(sha256_file(capture) == row["sha256"], f"{name}: SHA-256 changed")
        suffix, width, height = image_dimensions(capture.read_bytes())
        require(suffix == capture.suffix and (str(width), str(height)) == (row["width"], row["height"]),
                f"{name}: image type or dimensions changed")
        require((width, height) == tuple(state["expected_resolution"]), f"{name}: session resolution changed")
        ready_at = parse_utc(row["ready_confirmed_at_utc"], f"{name}.ready_confirmed_at_utc")
        captured_at = parse_utc(row["capture_time_utc"], f"{name}.capture_time_utc")
        require(ready_at <= captured_at, f"{name}: capture predates the operator READY confirmation")
        if frozen_at is not None:
            require(frozen_at < captured_at, f"{name}: official capture is not after the synthetic freeze")
    actual_names = {
        path.name for path in paths["captures"].iterdir()
        if path.is_file() and not path.is_symlink()
    }
    require(actual_names == names, "capture directory membership differs from the append-only log")
    return attempts


def verify_session(session: Path, *, verify_archive: bool = True) -> dict[str, object]:
    state, paths = load_session(session)
    require(not paths["pending"].exists(), "session has a pending capture; resume recovery before verification")
    plan = read_csv(paths["plan"], PLAN_COLUMNS)
    intents = read_csv(paths["intents"], INTENT_COLUMNS)
    validate_plan(plan, intents, str(state["role"]), str(state["fixture_id"]), str(state["session_id"]))
    require(sha256_file(paths["measurements"]) == state["measurement_template_sha256"],
            "measurement template changed inside raw collection evidence")
    attempts = _validate_attempts(state, paths, plan)
    actions = _actions(paths)
    valid_events = {row["event_id"]: row["event_uid"] for row in plan}
    skipped: set[str] = set()
    for row in actions:
        require(row["session_id"] == state["session_id"] and row["role"] == state["role"],
                "event action crosses immutable role")
        require(row["event_id"] in valid_events and row["event_uid"] == valid_events[row["event_id"]],
                "event action references an unknown event")
        require(row["action"] == "skipped" and row["reason"].strip(), "invalid event action")
        require(row["event_id"] not in skipped, "event was skipped more than once")
        skipped.add(row["event_id"])
    selected = selected_attempts(attempts)
    require(not (set(selected) & skipped), "an event cannot be both selected and skipped")
    pending_events = [event_id for event_id in valid_events if event_id not in selected and event_id not in skipped]
    result: dict[str, object] = {
        "ok": True,
        "session_id": state["session_id"],
        "role": state["role"],
        "events_planned": len(plan),
        "events_selected": len(selected),
        "events_skipped": len(skipped),
        "events_pending": len(pending_events),
        "capture_attempts": len(attempts),
        "collection_complete": not skipped and not pending_events and len(selected) == len(plan),
        "sealed": (paths["sealed"] / "evidence.tar").exists(),
    }
    if result["sealed"] and verify_archive:
        archive = paths["sealed"] / "evidence.tar"
        checksum = paths["sealed"] / "evidence.tar.sha256"
        seal_path = session / "records" / "seal_manifest.json"
        seal_value = read_json(seal_path)
        require(seal_value.get("session_id") == state["session_id"] and
                seal_value.get("role") == state["role"],
                "seal manifest crosses immutable session identity or role")
        inventory = seal_value.get("inventory")
        require(isinstance(inventory, list) and inventory, "seal manifest has no file inventory")
        inventory_paths: set[str] = set()
        for item in inventory:
            require(isinstance(item, dict), "seal inventory entry must be an object")
            relative = str(item.get("path", ""))
            candidate = (session / relative).resolve()
            try:
                candidate.relative_to(session.resolve())
            except ValueError as exc:
                raise CollectionError("seal inventory path escapes the session") from exc
            require(relative not in inventory_paths, "duplicate path in seal inventory")
            inventory_paths.add(relative)
            require(candidate.is_file() and not candidate.is_symlink(),
                    f"seal inventory file is missing or unsafe: {relative}")
            require(candidate.stat().st_size == item.get("bytes") and
                    sha256_file(candidate) == item.get("sha256"),
                    f"seal inventory verification failed: {relative}")
        require(checksum.is_file() and not checksum.is_symlink(), "sealed archive checksum is missing")
        words = checksum.read_text(encoding="utf-8").strip().split()
        require(len(words) == 2 and words[1] == "evidence.tar" and words[0] == sha256_file(archive),
                "sealed archive SHA-256 verification failed")
        with tarfile.open(archive, "r") as bundle:
            members = [member for member in bundle.getmembers() if member.isfile()]
            archive_names = {member.name for member in members}
            expected_names = {
                path.relative_to(session).as_posix()
                for base in (paths["captures"], session / "records")
                for path in base.rglob("*")
                if path.is_file() and not path.is_symlink() and path != paths["lock"]
            } | {"session.json", "plan.csv", "intended_positions.csv"}
            require(archive_names == expected_names, "sealed archive membership differs from the session seal")
            for member in members:
                stream = bundle.extractfile(member)
                require(stream is not None, f"cannot read archive member {member.name}")
                current = session / member.name
                require(sha256_bytes(stream.read()) == sha256_file(current),
                        f"archive member differs from retained evidence: {member.name}")
    return result


def _write_review_template(session: Path, state: Mapping[str, object], paths: Mapping[str, Path],
                           selected: Mapping[str, Mapping[str, str]],
                           intents: Sequence[Mapping[str, str]]) -> Path:
    review = session / "records" / "independent_review_template.csv"
    rows = []
    for intent in intents:
        attempt = selected.get(intent["event_id"])
        if attempt is None:
            continue
        rows.append({
            "sample_id": f"{intent['event_uid']}:{intent['position_id']}",
            "event_uid": intent["event_uid"],
            "session_id": state["session_id"],
            "event_id": intent["event_id"],
            "source_image_id": attempt["capture_name"],
            "source_image_relpath": f"captures/{attempt['capture_name']}",
            "source_image_sha256": attempt["sha256"],
            "capture_time_utc": attempt["capture_time_utc"],
            "fixture_id": intent["fixture_id"],
            "position_id": intent["position_id"],
            "expected_occupancy": intent["expected_occupancy"],
            "intended_visual_class": intent["intended_visual_class"],
            "inspection_mode": intent["inspection_mode"],
            "intended_gt_sequence": intent["intended_gt_sequence"],
            "actual_visual_class": "",
            "actual_gt_sequence": "",
            "reviewer_a": "",
            "reviewer_a_confidence": "",
            "reviewer_b": "",
            "reviewer_b_confidence": "",
            "adjudicator": "",
            "review_state": "pending_independent_review",
            "usable": "false",
            "notes": "Intent is not ground truth; both reviewers must inspect the full-resolution pixels.",
        })
    if review.exists():
        require(read_csv(review, REVIEW_COLUMNS) == [
            {key: str(value) for key, value in row.items()} for row in rows
        ], "existing independent-review template differs from the selected evidence")
    else:
        write_csv_exclusive(review, REVIEW_COLUMNS, rows)
    return review


def seal_session(session: Path, *, allow_partial: bool = False,
                 _lock_held: bool = False) -> dict[str, object]:
    session = session.expanduser().resolve(strict=True)
    state, paths = load_session(session)
    if not _lock_held:
        with paths["lock"].open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CollectionError("another collector process is writing; seal refused") from exc
            return seal_session(session, allow_partial=allow_partial, _lock_held=True)
    archive = paths["sealed"] / "evidence.tar"
    checksum_path = paths["sealed"] / "evidence.tar.sha256"
    if archive.exists():
        if not checksum_path.exists():
            write_text_exclusive(checksum_path, f"{sha256_file(archive)}  evidence.tar\n")
        verified_existing = verify_session(session, verify_archive=True)
        existing_seal = read_json(session / "records" / "seal_manifest.json")
        return {
            "seal_status": existing_seal["seal_status"],
            "archive": str(archive),
            "archive_sha256": sha256_file(archive),
            "events_selected": verified_existing["events_selected"],
            "events_skipped": verified_existing["events_skipped"],
            "events_pending": verified_existing["events_pending"],
        }
    report = verify_session(session, verify_archive=False)
    require(report["events_selected"] > 0, "refusing to seal a collection with zero selected events")
    if not report["collection_complete"]:
        require(allow_partial, "collection is incomplete; use --allow-partial only to create an explicit PARTIAL seal")
    plan = read_csv(paths["plan"], PLAN_COLUMNS)
    intents = read_csv(paths["intents"], INTENT_COLUMNS)
    attempts = _attempts(paths)
    actions = _actions(paths)
    selected = selected_attempts(attempts)
    review = _write_review_template(session, state, paths, selected, intents)
    selection_path = session / "records" / "seal_selection.json"
    selection = {
        "schema_version": 1,
        "session_id": state["session_id"],
        "role": state["role"],
        "selected": [dict(selected[event["event_id"]]) for event in plan if event["event_id"] in selected],
        "earlier_or_rejected_attempts": [
            row for row in attempts if selected.get(row["event_id"], {}).get("capture_name") != row["capture_name"]
        ],
        "skipped": [row for row in actions if row["action"] == "skipped"],
    }
    selection_text = json.dumps(selection, indent=2, sort_keys=True) + "\n"
    if selection_path.exists():
        require(selection_path.read_text(encoding="utf-8") == selection_text,
                "existing seal selection differs from current evidence")
    else:
        write_text_exclusive(selection_path, selection_text)
    inventory = []
    for base in (paths["captures"], session / "records"):
        for item in sorted(base.rglob("*")):
            if item in {paths["lock"], session / "records" / "seal_manifest.json"}:
                continue
            if item.is_file() and not item.is_symlink():
                inventory.append({
                    "path": item.relative_to(session).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                })
    for item in (paths["state"], paths["plan"], paths["intents"]):
        inventory.append({
            "path": item.relative_to(session).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        })
    seal_manifest = session / "records" / "seal_manifest.json"
    if seal_manifest.exists():
        seal_value = read_json(seal_manifest)
        require(seal_value.get("session_id") == state["session_id"] and
                seal_value.get("role") == state["role"] and
                seal_value.get("seal_status") == (
                    "COMPLETE" if report["collection_complete"] else "PARTIAL"
                ) and seal_value.get("inventory") == inventory,
                "existing interrupted seal manifest differs from current evidence")
    else:
        seal_value = {
            "schema_version": 1,
            "collector_version": COLLECTOR_VERSION,
            "session_id": state["session_id"],
            "role": state["role"],
            "sealed_at_utc": utc_now(),
            "seal_status": "COMPLETE" if report["collection_complete"] else "PARTIAL",
            "collection_summary": report,
            "review_template": review.relative_to(session).as_posix(),
            "inventory": inventory,
            "warning": "Intended scenarios remain unadjudicated and unusable until independent visible-content review.",
        }
        write_text_exclusive(seal_manifest, json.dumps(seal_value, indent=2, sort_keys=True) + "\n")

    include = [paths["state"], paths["plan"], paths["intents"]]
    include += [
        item for base in (paths["captures"], session / "records")
        for item in sorted(base.rglob("*"))
        if item.is_file() and not item.is_symlink() and item != paths["lock"]
    ]
    descriptor, temporary_archive_name = tempfile.mkstemp(prefix=".evidence.tar.building-", dir=paths["sealed"])
    os.close(descriptor)
    temporary_archive = Path(temporary_archive_name)
    try:
        with tarfile.open(temporary_archive, "w") as bundle:
            for item in include:
                bundle.add(item, arcname=item.relative_to(session).as_posix(), recursive=False)
        with temporary_archive.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_archive, archive)
        _fsync_directory(paths["sealed"])
    finally:
        if temporary_archive.exists():
            temporary_archive.unlink()
    archive_sha = sha256_file(archive)
    write_text_exclusive(checksum_path, f"{archive_sha}  evidence.tar\n")
    verified = verify_session(session, verify_archive=True)
    return {
        "seal_status": seal_value["seal_status"],
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "events_selected": verified["events_selected"],
        "events_skipped": verified["events_skipped"],
        "events_pending": verified["events_pending"],
    }


def _event_display(event: Mapping[str, str], index: int, total: int) -> None:
    print("\n" + "=" * 78)
    print(f"EVENT {event['event_id']} ({index}/{total}) — {event['title']}")
    print(f"Event UID for measurement records: {event['event_uid']}")
    print(f"Fixture {event['fixture_id']} | mode {event['inspection_mode']} | family {event['family']}")
    print(event["instructions"])
    print(f"Primary targets: {event['primary_targets']}")
    print(
        "Before READY, record actual primary-target angles (where applicable), method, uncertainty, "
        "measurer, and UTC time against the Event UID."
    )
    print("Intent is not ground truth. Actual pixels must be independently reviewed later.")
    print("=" * 78, flush=True)


def _yes_no(prompt: str, *, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        answer = input(prompt + suffix).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Type y or n.")


def interactive_collect(session: Path, *, stability_seconds: float, wait_seconds: float) -> int:
    state, paths = load_session(session)
    require(not (paths["sealed"] / "evidence.tar").exists(),
            "session is already sealed; use verify or export-review, not resume")
    plan = read_csv(paths["plan"], PLAN_COLUMNS)
    with paths["lock"].open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CollectionError("another collector process is already writing this session") from exc
        recover_pending(session, _lock_held=True)
        while True:
            attempts = _attempts(paths)
            actions = _actions(paths)
            selected = selected_attempts(attempts)
            skipped = skipped_events(actions)
            remaining = [row for row in plan if row["event_id"] not in selected and row["event_id"] not in skipped]
            if not remaining:
                print("\nAll planned events are either selected or explicitly skipped.")
                report = verify_session(session, verify_archive=False)
                if report["collection_complete"]:
                    if _yes_no("Seal this COMPLETE collection now?", default=True):
                        result = seal_session(session, _lock_held=True)
                        print(f"SEALED {result['seal_status']}: {result['archive']}")
                        print(f"SHA-256: {result['archive_sha256']}")
                else:
                    print("The plan contains skips; it cannot receive a COMPLETE seal.")
                    if _yes_no("Create an explicitly PARTIAL seal now?", default=False):
                        result = seal_session(session, allow_partial=True, _lock_held=True)
                        print(f"SEALED {result['seal_status']}: {result['archive']}")
                        print(f"SHA-256: {result['archive_sha256']}")
                return 0
            event = remaining[0]
            index = next(i for i, row in enumerate(plan, 1) if row["event_id"] == event["event_id"])
            _event_display(event, index, len(plan))
            answer = input(
                "Type READY only after a fresh placement, stopped motion, hands/arm clear, and a new camera "
                "image is saved; s=skip with reason; q=pause: "
            ).strip()
            if answer.lower() == "q":
                report = verify_session(session, verify_archive=False)
                print(f"PAUSED safely: {session}")
                print(json.dumps(report, indent=2, sort_keys=True))
                return 0
            if answer.lower() == "s":
                reason = input("Required skip reason: ").strip()
                skip_event(session, event, reason, _lock_held=True)
                print(f"SKIPPED {event['event_id']}: {reason}")
                continue
            if answer != "READY":
                print("No capture taken. Type the uppercase word READY, s, or q.")
                continue
            ready_at = utc_now()
            try:
                pending = ingest_capture(
                    session, event, stability_seconds=stability_seconds, wait_seconds=wait_seconds,
                    ready_confirmed_at_utc=ready_at, _lock_held=True,
                )
            except CollectionError as exc:
                print(f"CAPTURE REFUSED: {exc}. Source was not modified; event did not advance.")
                continue
            print(
                f"Retained {pending['capture_name']} ({pending['width']}x{pending['height']}, "
                f"{pending['bytes']} bytes, SHA-256 {str(pending['sha256'])[:16]}...)."
            )
            accepted = _yes_no(
                "Human operator: does this retained frame correspond to the stated physical setup with hands/arm clear?"
            )
            note = input("Confirmation/rejection note (optional): ").strip()
            finalized = finalize_capture(session, accepted=accepted, note=note, _lock_held=True)
            if not accepted:
                print(f"Attempt retained as rejected evidence: {finalized['capture_name']}. Restage and try again.")
                continue
            print(f"SELECTED {finalized['capture_name']} for {event['event_id']}; no inference was run.")
            while _yes_no("Retake this same physical event before advancing?", default=False):
                retake_ready = input(
                    "Without moving/restaging any vial or cap, save exactly one replacement image; after motion "
                    "stops and hands/arm/tools are clear, type uppercase READY (anything else cancels): "
                ).strip()
                if retake_ready != "READY":
                    print("Retake cancelled. A moved/restaged setup requires a new physical event/session.")
                    break
                ready_at = utc_now()
                try:
                    pending = ingest_capture(
                        session, event, stability_seconds=stability_seconds, wait_seconds=wait_seconds,
                        ready_confirmed_at_utc=ready_at, _lock_held=True,
                    )
                except CollectionError as exc:
                    print(f"RETAKE REFUSED: {exc}; prior selection remains authoritative.")
                    continue
                print(
                    f"Retained {pending['capture_name']} ({pending['width']}x{pending['height']}, "
                    f"SHA-256 {str(pending['sha256'])[:16]}...)."
                )
                accepted = _yes_no("Human operator: select this retake for the event?", default=False)
                note = input("Retake confirmation/rejection note (optional): ").strip()
                finalized = finalize_capture(
                    session, accepted=accepted, note=note, _lock_held=True,
                )
                print(
                    f"Retake {finalized['status']}: {finalized['capture_name']}; every earlier attempt remains retained."
                )


def _prompt(value: str | None, label: str, default: str | None = None) -> str:
    if value is not None and str(value).strip():
        return str(value).strip()
    require(sys.stdin.isatty(), f"{label} is required in noninteractive mode")
    suffix = f" [{default}]" if default else ""
    answer = input(f"{label}{suffix}: ").strip()
    return answer or str(default or "")


def export_review_working_copy(session: Path, out_dir: Path) -> dict[str, str]:
    session = resolve_without_symlinks(session, must_exist=True)
    state, paths = load_session(session)
    require((paths["sealed"] / "evidence.tar").is_file(),
            "seal the collection before exporting review working copies")
    verify_session(session, verify_archive=True)
    out_dir = resolve_without_symlinks(out_dir, must_exist=False)
    require(not paths_overlap(session, out_dir),
            "review working copies must be outside the immutable session")
    if out_dir.exists():
        require(out_dir.is_dir() and not out_dir.is_symlink() and not any(out_dir.iterdir()),
                "review output must be a new or empty real directory")
    else:
        out_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(out_dir, 0o700)

    coordinator = out_dir / "coordinator_only"
    coordinator.mkdir(mode=0o700)
    sources = {
        "event_measurements_working.csv": paths["measurements"],
        "adjudication_working.csv": session / "records" / "independent_review_template.csv",
    }
    for name, source in sources.items():
        require(source.is_file() and not source.is_symlink(), f"sealed review source is missing: {source}")
        destination = coordinator / name
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.chmod(destination, 0o600)
        require(sha256_file(destination) == sha256_file(source), f"working-copy verification failed: {name}")

    source_rows = read_csv(sources["adjudication_working.csv"], REVIEW_COLUMNS)
    join_rows: list[dict[str, str]] = []
    packet_paths: dict[str, Path] = {}
    for packet_name in ("reviewer_a_packet", "reviewer_b_packet"):
        packet = out_dir / packet_name
        images = packet / "images"
        packet.mkdir(mode=0o700)
        images.mkdir(mode=0o700)
        packet_paths[packet_name] = packet
        opaque_images: dict[str, tuple[str, str, str]] = {}
        reviewer_rows: list[dict[str, str]] = []
        for row in source_rows:
            source_name = row["source_image_id"]
            if source_name not in opaque_images:
                source_image = session / row["source_image_relpath"]
                require(source_image.parent == paths["captures"] and source_image.is_file() and
                        not source_image.is_symlink(), "review source image is missing or unsafe")
                require(sha256_file(source_image) == row["source_image_sha256"],
                        "review source image digest differs from sealed template")
                opaque_id = f"img-{secrets.token_hex(12)}"
                relative = f"images/{opaque_id}{source_image.suffix.lower()}"
                destination = packet / relative
                with source_image.open("rb") as incoming, destination.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                os.chmod(destination, 0o400)
                require(sha256_file(destination) == row["source_image_sha256"],
                        "review image copy verification failed")
                opaque_images[source_name] = (opaque_id, relative, row["source_image_sha256"])
            opaque_id, relative, digest = opaque_images[source_name]
            review_sample_id = f"sample-{secrets.token_hex(12)}"
            reviewer_rows.append({
                "review_sample_id": review_sample_id,
                "review_image_id": opaque_id,
                "review_image_relpath": relative,
                "review_image_sha256": digest,
                "fixture_id": row["fixture_id"],
                "position_id": row["position_id"],
                "reviewer_id": "",
                "actual_visual_class": "",
                "reviewer_confidence": "",
                "visible_evidence_notes": "",
                "review_state": "pending_blinded_review",
            })
            join_rows.append({
                "reviewer_packet": packet_name,
                "review_sample_id": review_sample_id,
                "review_image_id": opaque_id,
                "sample_id": row["sample_id"],
                "event_uid": row["event_uid"],
                "event_id": row["event_id"],
                "source_image_id": row["source_image_id"],
                "source_image_relpath": row["source_image_relpath"],
                "source_image_sha256": row["source_image_sha256"],
                "fixture_id": row["fixture_id"],
                "position_id": row["position_id"],
            })
        secrets.SystemRandom().shuffle(reviewer_rows)
        sheet = packet / "review.csv"
        write_csv_exclusive(sheet, BLINDED_REVIEW_COLUMNS, reviewer_rows)
        os.chmod(sheet, 0o600)
        packet_readme = (
            "INDEPENDENT INTENT-BLINDED VISUAL REVIEW PACKET\n\n"
            "Do not seek staging instructions, event IDs, original filenames, expected occupancy, sequence context, "
            "the other review packet, or coordinator files before submitting and locking this review. Inspect only "
            "the full-resolution images in images/. Fill reviewer_id, actual_visual_class, reviewer_confidence, "
            "visible_evidence_notes, and review_state in review.csv. Use exactly one visual class: pass, missing_cap, "
            "crooked_cap, damaged_vial, or no_vial. Record uncertainty; never convert uncertainty into PASS.\n"
        )
        write_text_exclusive(packet / "README_REVIEWER.txt", packet_readme, mode=0o600)
        _fsync_directory(images)
        _fsync_directory(packet)

    join_path = coordinator / "reviewer_join_map.csv"
    write_csv_exclusive(join_path, REVIEW_JOIN_COLUMNS, join_rows)
    os.chmod(join_path, 0o600)
    coordinator_readme = (
        f"Coordinator files for {state['session_id']} ({state['role']}).\n"
        "Never edit the sealed session. Distribute each reviewer packet separately; reviewers must not receive this "
        "directory or each other's packet before both reviews are locked. Use reviewer_join_map.csv only after both "
        "submissions to merge into adjudication_working.csv and adjudicate disagreements. Expected occupancy and "
        "sequence are deterministic context, not visual-review predictions. Intent is never ground truth. Complete "
        "event_measurements_working.csv only from contemporaneous records or a documented validated pixel method with "
        "uncertainty; otherwise leave actual angles blank and make no angle-stratified claim.\n"
    )
    write_text_exclusive(coordinator / "README_COORDINATOR.txt", coordinator_readme, mode=0o600)
    _fsync_directory(coordinator)

    manifest = {
        "schema_version": 1,
        "session_id": state["session_id"],
        "role": state["role"],
        "created_at_utc": utc_now(),
        "derived_files": {
            path.relative_to(out_dir).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(out_dir.rglob("*"))
            if path.is_file() and not path.is_symlink()
        },
        "warning": "Derived working files are not immutable raw evidence; reviewer packets must stay separated.",
    }
    manifest_path = out_dir / "EXPORT_MANIFEST.json"
    write_text_exclusive(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n", mode=0o600)
    _fsync_directory(out_dir)
    return {
        "reviewer_a_packet": str(packet_paths["reviewer_a_packet"]),
        "reviewer_b_packet": str(packet_paths["reviewer_b_packet"]),
        "coordinator_only": str(coordinator),
        "manifest": str(manifest_path),
    }


def _collect_command(args: argparse.Namespace) -> int:
    if args.resume:
        session = Path(args.resume)
    else:
        role = _prompt(args.role, "Session role (train/validation/protected_test/rehearsal)")
        require(role in ROLES, f"role must be one of {', '.join(ROLES)}")
        camera = Path(_prompt(args.camera_dir, "Camera image folder"))
        output = Path(_prompt(args.output_root, "Evidence output root"))
        fixture = _prompt(args.fixture, "Fixture", "G")
        resolution = parse_resolution(_prompt(args.expected_resolution, "Locked camera resolution", "1920x1080"))
        operator_id = _prompt(args.operator_id, "Operator ID")
        camera_id = _prompt(args.camera_id, "Camera ID/profile")
        baseline = args.baseline_freeze
        if role in OFFICIAL_ROLES:
            baseline = _prompt(baseline, "Frozen synthetic baseline JSON")
        handoff = args.protected_handoff
        if role == "protected_test":
            handoff = _prompt(handoff, "Frozen development-to-protected handoff directory")
            acknowledgement = _prompt(args.protected_custodian_ack,
                                      "Type PROTECTED-CUSTODIAN to confirm separate custody")
            require(acknowledgement == "PROTECTED-CUSTODIAN",
                    "protected collection requires separate-custodian acknowledgement")
        session = initialize_session(
            role=role,
            camera_dir=camera,
            output_root=output,
            fixture=fixture,
            expected_resolution=resolution,
            operator_id=operator_id,
            camera_id=camera_id,
            baseline_freeze=Path(baseline) if baseline else None,
            protected_handoff=Path(handoff) if handoff else None,
            session_id=args.session_id,
        )
        print(f"Initialized immutable {role} session: {session}")
        print("Existing camera files were snapshotted and will be ignored.")
    return interactive_collect(
        session, stability_seconds=args.stability_seconds, wait_seconds=args.wait_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the inference-free Vial Guard staging image collection workflow."
    )
    subparsers = parser.add_subparsers(dest="command")
    collect = subparsers.add_parser("collect", help="start or resume an interactive collection")
    collect.add_argument("--resume", help="existing session directory to resume")
    collect.add_argument("--role", choices=ROLES)
    collect.add_argument("--camera-dir")
    collect.add_argument("--output-root")
    collect.add_argument("--fixture", choices=FIXTURES)
    collect.add_argument("--expected-resolution")
    collect.add_argument("--operator-id")
    collect.add_argument("--camera-id")
    collect.add_argument("--baseline-freeze")
    collect.add_argument("--protected-handoff")
    collect.add_argument("--protected-custodian-ack")
    collect.add_argument("--session-id")
    collect.add_argument("--stability-seconds", type=float, default=1.0)
    collect.add_argument("--wait-seconds", type=float, default=30.0)

    status = subparsers.add_parser("status", help="verify and summarize a session")
    status.add_argument("--session-dir", required=True)
    verify = subparsers.add_parser("verify", help="tamper-check a session and any seal")
    verify.add_argument("--session-dir", required=True)
    seal = subparsers.add_parser("seal", help="seal a completed or explicitly partial session")
    seal.add_argument("--session-dir", required=True)
    seal.add_argument("--allow-partial", action="store_true")
    export_review = subparsers.add_parser(
        "export-review", help="copy sealed review/measurement templates outside immutable evidence"
    )
    export_review.add_argument("--session-dir", required=True)
    export_review.add_argument("--out-dir", required=True)
    freeze = subparsers.add_parser(
        "freeze-development",
        help="finalize development and emit a metadata-only protected-custodian handoff",
    )
    freeze.add_argument("--development-root", required=True)
    freeze.add_argument("--out-dir", required=True)
    plan = subparsers.add_parser("plan-summary", help="show preregistered event counts without creating data")
    plan.add_argument("--role", choices=ROLES, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(argv if argv is not None else sys.argv[1:])
    if not arguments:
        arguments = ["collect"]
    args = parser.parse_args(arguments)
    try:
        if args.command == "collect":
            require(args.stability_seconds >= 0.25, "stability interval must be at least 0.25 seconds")
            require(args.wait_seconds >= args.stability_seconds, "wait timeout must cover the stability interval")
            return _collect_command(args)
        if args.command in {"status", "verify"}:
            report = verify_session(Path(args.session_dir))
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "seal":
            result = seal_session(Path(args.session_dir), allow_partial=args.allow_partial)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "export-review":
            result = export_review_working_copy(Path(args.session_dir), Path(args.out_dir))
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "freeze-development":
            result = freeze_development_for_protected(
                Path(args.development_root), Path(args.out_dir)
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "plan-summary":
            counts = FAMILY_COUNTS[args.role]
            print(json.dumps({"role": args.role, "families": counts, "events": sum(counts.values())},
                             indent=2, sort_keys=True))
            return 0
        parser.print_help()
        return 2
    except (CollectionError, FileExistsError, FileNotFoundError, PermissionError, OSError, KeyboardInterrupt) as exc:
        reason = str(exc) or "interrupted; retained evidence is resumable"
        print(f"STOP: {reason}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
