#!/usr/bin/env python3
"""Create an Excel workbook from the frozen CSI collection logs.

The reporter is deliberately separate from ``ttl_kit.tar``. It reads metadata
only, never opens staged images, imports an inspector, changes the frozen kit,
or infers ground truth from predictions. The Tuesday plan supplies predeclared
ground truth; operators only need to confirm the physical setup and record
anything that differed.

Only the Python standard library is used so this can run on an offline GB10.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape, quoteattr


REPORT_VERSION = "1.0"
MAX_ROWS = 1_048_576
MAX_COLUMNS = 16_384
MAX_TEXT = 32_767
MAX_METADATA_BYTES = 256 * 1024 * 1024
NORMAL_STATUSES = {"arm", "settling", "unchanged", "scored", "source_recovered"}
VALID_SETUPS = {f"{index:02d}" for index in range(22)}
PREDICTION_FIELDS = [
    "run", "event", "frame", "position", "predicted_state", "defect_score",
    "pred_sequence", "final_state", "K", "K_source", "cap_frac",
    "cap_offset", "reason",
]
INVALID_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
INVALID_SHEET = re.compile(r"[\[\]:*?/\\]")
TARGET = re.compile(r"^(all|G(?:0[1-9]|[12][0-9]|30))(?:-(G(?:0[1-9]|[12][0-9]|30)))?$")


@dataclass
class Sheet:
    name: str
    headers: list[str]
    rows: list[dict[str, object]]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_time(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def as_iso(value: datetime | None) -> str:
    return value.isoformat(timespec="milliseconds") if value else ""


def seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 3)


def numeric(value: object) -> int | float | None:
    if value in (None, ""):
        return None
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return int(result) if result.is_integer() else result


def under(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def display(path: Path, root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root.resolve(strict=False)))
    except ValueError:
        return str(path.resolve(strict=False))


def alert(rows: list[dict[str, object]], severity: str, visit: str, area: str,
          message: str, source: object = "") -> None:
    rows.append({
        "severity": severity,
        "visit_id": visit,
        "area": area,
        "message": message,
        "source": str(source),
    })


def ensure_scoped_metadata(path: Path, scope: Path) -> None:
    """Reject traversal and every symlink component beneath a trusted visit."""

    lexical_scope = Path(os.path.abspath(scope))
    lexical_path = Path(os.path.abspath(path))
    try:
        relative = lexical_path.relative_to(lexical_scope)
    except ValueError as exc:
        raise RuntimeError(f"metadata path is outside visit: {path}") from exc
    cursor = lexical_scope
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError(f"metadata path contains a symlink: {cursor}")
    try:
        lexical_path.resolve(strict=False).relative_to(lexical_scope.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"metadata path escapes visit: {path}") from exc


def stable_bytes(path: Path, retries: int = 4, *, scope: Path | None = None) -> bytes:
    """Read a regular metadata file without accepting an in-progress rewrite."""

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            if scope is not None:
                ensure_scoped_metadata(path, scope)
            before = path.stat()
            if not path.is_file() or path.is_symlink():
                raise OSError("not a regular file")
            if before.st_size > MAX_METADATA_BYTES:
                raise OSError(
                    f"metadata file is {before.st_size} bytes; limit is {MAX_METADATA_BYTES}"
                )
            data = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino
            ):
                raise RuntimeError("file changed while being read")
            return data
        except (OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.05 * (attempt + 1))
    raise RuntimeError(f"unstable or unreadable metadata file: {path}: {last_error}")


def read_csv(path: Path, alerts: list[dict[str, object]], visit: str,
             *, required: bool = False,
             scope: Path | None = None) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        if required:
            alert(alerts, "ERROR", visit, "CSV", "required CSV is missing", path)
        return [], []
    try:
        data = stable_bytes(path, scope=scope)
        text = data.decode("utf-8-sig")
        if text and not text.endswith(("\n", "\r")):
            raise ValueError("incomplete trailing CSV row")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        return list(reader.fieldnames), [dict(row) for row in reader]
    except (RuntimeError, UnicodeError, ValueError, csv.Error) as exc:
        alert(alerts, "ERROR" if required else "WARN", visit, "CSV", str(exc), path)
        return [], []


def read_json(path: Path, alerts: list[dict[str, object]], visit: str,
              *, scope: Path | None = None) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(stable_bytes(path, scope=scope).decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("JSON root is not an object")
        return value
    except (RuntimeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        alert(alerts, "WARN", visit, "JSON", str(exc), path)
        return {}


def read_env(path: Path, alerts: list[dict[str, object]], visit: str,
             *, scope: Path | None = None) -> dict[str, str]:
    if not path.exists():
        alert(alerts, "ERROR", visit, "visit context", "operator.env is missing", path)
        return {}
    try:
        text = stable_bytes(path, scope=scope).decode("utf-8-sig")
        values: dict[str, str] = {}
        for line_number, raw in enumerate(text.splitlines(), 1):
            if not raw.strip():
                continue
            if "=" not in raw:
                raise ValueError(f"line {line_number} has no equals sign")
            key, encoded = raw.split("=", 1)
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
                raise ValueError(f"line {line_number} has an invalid key")
            words = shlex.split(encoded)
            if len(words) > 1:
                raise ValueError(f"line {line_number} has multiple values")
            values[key] = words[0] if words else ""
        return values
    except (RuntimeError, UnicodeError, ValueError) as exc:
        alert(alerts, "ERROR", visit, "visit context", str(exc), path)
        return {}


def redact_source(value: object) -> str:
    text = str(value or "")
    prefix, raw = ("stream:", text[7:]) if text.startswith("stream:") else ("", text)
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return text
    if parsed.hostname is None:
        return text
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        host += f":{port}"
    if parsed.username is not None or parsed.password is not None:
        host = f"***@{host}"
    sensitive = re.compile(r"(?:token|key|password|passwd|secret|auth|signature|credential)", re.I)
    query = urlencode([
        (key, "***" if sensitive.search(key) else item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
    ])
    return prefix + urlunsplit((parsed.scheme, host, parsed.path, query, parsed.fragment))


def normalize_setup(value: object) -> str | None:
    text = str(value or "").strip()
    if re.fullmatch(r"\d", text):
        text = f"0{text}"
    return text if text in VALID_SETUPS else None


def expand_holes(spec: str) -> list[str]:
    match = TARGET.fullmatch(spec)
    if not match:
        raise ValueError(f"invalid plan target {spec!r}")
    first, last = match.group(1), match.group(2)
    if first == "all":
        return ["all"]
    if last is None:
        return [first]
    start, end = int(first[1:]), int(last[1:])
    if end < start:
        raise ValueError(f"descending plan range {spec!r}")
    return [f"G{index:02d}" for index in range(start, end + 1)]


def load_plan(path: Path, alerts: list[dict[str, object]]) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    fields, setups = read_csv(path, alerts, "", required=True)
    required = {
        "setup", "setup_type", "what_to_stage", "targets", "last_filled",
        "inspection_mode", "priority", "notes",
    }
    if fields and not required.issubset(fields):
        raise ValueError(f"plan is missing columns: {', '.join(sorted(required - set(fields)))}")
    by_setup: set[str] = set()
    observations: list[dict[str, object]] = []
    for setup_row in setups:
        setup = normalize_setup(setup_row["setup"])
        if setup is None:
            raise ValueError(f"invalid plan setup {setup_row['setup']!r}")
        if setup in by_setup:
            raise ValueError(f"duplicate plan setup {setup}")
        by_setup.add(setup)
        observation = 0
        for token in str(setup_row["targets"]).split(";"):
            pieces = token.split("|")
            if len(pieces) != 4:
                raise ValueError(f"setup {setup}: invalid target token {token!r}")
            target_spec, state, sequence, tilt = pieces
            allowed_states = {
                "orientation_check", "all_pass", "crooked_cap", "missing_cap",
                "no_vial", "lean_seated",
            }
            if state not in allowed_states:
                raise ValueError(f"setup {setup}: invalid state {state!r}")
            if state == "no_vial":
                if sequence not in {"missed", "waiting"}:
                    raise ValueError(f"setup {setup}: no_vial needs missed or waiting")
            elif sequence:
                raise ValueError(f"setup {setup}: unexpected sequence {sequence!r}")
            angle = numeric(tilt)
            if state == "crooked_cap":
                if angle not in {10, 15, 20, 30}:
                    raise ValueError(f"setup {setup}: invalid crooked angle {tilt!r}")
            elif tilt:
                raise ValueError(f"setup {setup}: unexpected target angle {tilt!r}")
            for hole in expand_holes(target_spec):
                observation += 1
                if state == "crooked_cap":
                    claim_group = "crooked_20_plus" if angle is not None and angle >= 20 else "crooked_below_20"
                elif state == "no_vial":
                    claim_group = sequence
                else:
                    claim_group = state
                observations.append({
                    "setup": setup,
                    "observation": observation,
                    "setup_type": setup_row["setup_type"],
                    "what_to_stage": setup_row["what_to_stage"],
                    "hole": hole,
                    "planned_state": state,
                    "sequence_state": sequence,
                    "target_tilt_deg": angle,
                    "claim_group": claim_group,
                    "last_filled": setup_row["last_filled"],
                    "inspection_mode": setup_row["inspection_mode"],
                    "priority": setup_row["priority"],
                    "plan_notes": setup_row["notes"],
                })
    if by_setup != VALID_SETUPS:
        raise ValueError("Tuesday plan must contain exactly setups 00 through 21")
    totals = Counter(str(row["claim_group"]) for row in observations)
    expected_totals = {
        "crooked_20_plus": 14,
        "crooked_below_20": 3,
        "missing_cap": 14,
        "missed": 14,
        "lean_seated": 6,
        "waiting": 37,
        "all_pass": 2,
        "orientation_check": 1,
    }
    if totals != Counter(expected_totals):
        raise ValueError(f"Tuesday plan totals are {dict(totals)}, expected {expected_totals}")
    return setups, observations


def source_record(path: Path, role: str, root: Path,
                  *, scope: Path | None = None) -> dict[str, object]:
    try:
        data = stable_bytes(path, scope=scope)
        stat = path.stat()
        return {
            "source": display(path, root),
            "role": role,
            "bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="milliseconds"),
            "sha256": hashlib.sha256(data).hexdigest(),
            "state": "read",
        }
    except (RuntimeError, OSError) as exc:
        return {
            "source": display(path, root), "role": role, "bytes": None,
            "modified_utc": "", "sha256": "", "state": str(exc),
        }


def collect_report(root: Path, plan_path: Path, *, include_predictions: bool = False,
                   generated_utc: str | None = None) -> tuple[list[Sheet], dict[str, object]]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"TTL root is not a directory: {root}")
    plan_path = plan_path.expanduser().resolve(strict=True)
    generated = generated_utc or now_utc()
    alerts: list[dict[str, object]] = []
    plan_setups, plan_observations = load_plan(plan_path, alerts)

    summaries: list[dict[str, object]] = []
    setup_checklist: list[dict[str, object]] = []
    ground_truth: list[dict[str, object]] = []
    staging_attempts: list[dict[str, object]] = []
    process_times: list[dict[str, object]] = []
    live_runs: list[dict[str, object]] = []
    live_events: list[dict[str, object]] = []
    frame_log: list[dict[str, object]] = []
    interventions: list[dict[str, object]] = []
    position_results: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = [source_record(plan_path, "Tuesday staging plan", root)]

    visits = sorted(path for path in root.glob("visit_*") if path.is_dir() and not path.is_symlink())
    if not visits:
        alert(alerts, "WARN", "", "discovery", "No visit_* directories were found", root)

    for visit in visits:
        visit_id = visit.name
        env_path = visit / "operator.env"
        context = read_env(env_path, alerts, visit_id, scope=visit)
        provenance.append(source_record(env_path, "visit context", root, scope=visit))

        stage_path = visit / "records" / "staging_log.csv"
        stage_fields, stage_rows = read_csv(
            stage_path, alerts, visit_id, required=True, scope=visit
        )
        provenance.append(source_record(stage_path, "staging log", root, scope=visit))
        expected_stage_fields = {"setup", "status", "source_name", "copy_name", "time", "sha256"}
        if stage_fields and not expected_stage_fields.issubset(stage_fields):
            alert(alerts, "ERROR", visit_id, "staging",
                  f"Missing staging columns: {', '.join(sorted(expected_stage_fields - set(stage_fields)))}",
                  stage_path)

        selection_path = visit / "records" / "seal_selection.json"
        selection = read_json(selection_path, alerts, visit_id, scope=visit)
        selection_present = selection_path.exists()
        if selection_present:
            provenance.append(source_record(
                selection_path, "seal selection", root, scope=visit
            ))
        selected: dict[str, dict[str, object]] = {}
        earlier_names: set[str] = set()
        selection_valid = selection_present
        raw_final = selection.get("final", [])
        raw_earlier = selection.get("earlier_attempts", [])
        if selection_present and not isinstance(raw_final, list):
            alert(alerts, "ERROR", visit_id, "seal selection", "final is not a list", selection_path)
            selection_valid = False
            raw_final = []
        if selection_present and not isinstance(raw_earlier, list):
            alert(alerts, "ERROR", visit_id, "seal selection", "earlier_attempts is not a list", selection_path)
            selection_valid = False
            raw_earlier = []
        for row in raw_final:
            if not isinstance(row, dict):
                alert(alerts, "ERROR", visit_id, "seal selection", "final contains a non-object", selection_path)
                selection_valid = False
                continue
            setup = normalize_setup(row.get("setup"))
            if setup is None or setup in selected:
                alert(alerts, "ERROR", visit_id, "seal selection",
                      f"invalid or duplicate setup {row.get('setup')!r}", selection_path)
                selection_valid = False
                continue
            selected[setup] = dict(row)
        for row in raw_earlier:
            if not isinstance(row, dict):
                alert(alerts, "ERROR", visit_id, "seal selection",
                      "earlier_attempts contains a non-object", selection_path)
                selection_valid = False
                continue
            earlier_names.add(str(row.get("copy_name", "")))

        notes_path = visit / "records" / "staging_notes.csv"
        note_fields, note_rows = read_csv(notes_path, alerts, visit_id, scope=visit)
        notes_by_setup: dict[str, list[str]] = defaultdict(list)
        if note_rows:
            provenance.append(source_record(
                notes_path, "optional staging notes", root, scope=visit
            ))
            if "setup" not in note_fields or "note" not in note_fields:
                alert(alerts, "WARN", visit_id, "staging notes",
                      "staging_notes.csv needs setup and note columns", notes_path)
            else:
                for note in note_rows:
                    setup = normalize_setup(note.get("setup"))
                    if setup is None:
                        alert(alerts, "WARN", visit_id, "staging notes",
                              f"Ignored invalid setup {note.get('setup')!r}", notes_path)
                        continue
                    notes_by_setup[setup].append(str(note.get("note", "")))

        actual_by_setup: dict[str, dict[str, str]] = {}
        attempts_by_setup: Counter[str] = Counter()
        prior_time: datetime | None = None
        parsed_stage_times: list[datetime] = []
        seen_hashes: dict[str, str] = {}
        for row_number, row in enumerate(stage_rows, 1):
            setup = normalize_setup(row.get("setup"))
            if setup is None:
                alert(alerts, "ERROR", visit_id, "staging",
                      f"Row {row_number} has invalid setup {row.get('setup')!r}", stage_path)
                continue
            status = str(row.get("status", ""))
            if status not in {"staged", "retake", "skipped"}:
                alert(alerts, "ERROR", visit_id, "staging",
                      f"Row {row_number} has unknown status {status!r}", stage_path)
                continue
            attempts_by_setup[setup] += 1
            actual_by_setup[setup] = row
            recorded = parse_time(row.get("time"))
            if row.get("time") and recorded is None:
                alert(alerts, "WARN", visit_id, "staging time",
                      f"Row {row_number} has an invalid or timezone-free timestamp", stage_path)
            interval = seconds(prior_time, recorded)
            if interval is not None and interval < 0:
                alert(alerts, "WARN", visit_id, "staging time",
                      f"Row {row_number} is earlier than the previous record", stage_path)
                interval = None
            if recorded is not None:
                parsed_stage_times.append(recorded)

            copy_name = str(row.get("copy_name", ""))
            copy_exists: bool | None = None
            copy_bytes: int | None = None
            if copy_name:
                if Path(copy_name).name != copy_name:
                    alert(alerts, "ERROR", visit_id, "staging", "Unsafe staged copy name", copy_name)
                else:
                    copy_path = visit / "staged" / copy_name
                    try:
                        stat = copy_path.lstat()
                        copy_exists = copy_path.is_file() and not copy_path.is_symlink()
                        copy_bytes = stat.st_size if copy_exists else None
                    except OSError:
                        copy_exists = False
                    if not copy_exists:
                        alert(alerts, "ERROR", visit_id, "staging", "Logged staged copy is missing", copy_path)
            sha = str(row.get("sha256", ""))
            if sha:
                if sha in seen_hashes:
                    alert(alerts, "ERROR", visit_id, "staging",
                          f"Duplicate image hash in {copy_name} and {seen_hashes[sha]}", stage_path)
                seen_hashes[sha] = copy_name
            if status == "skipped":
                if any(row.get(field) for field in ("source_name", "copy_name", "sha256")):
                    alert(alerts, "ERROR", visit_id, "staging",
                          f"Skipped row {row_number} unexpectedly has a capture", stage_path)
                disposition = "skipped"
            elif copy_name in earlier_names:
                disposition = "earlier retained attempt"
            elif selected.get(setup, {}).get("copy_name") == copy_name:
                disposition = "final selected"
            elif not selection:
                disposition = "latest before seal"
            else:
                disposition = "not selected"
            staging_attempts.append({
                "visit_id": visit_id,
                "row": row_number,
                "setup": setup,
                "attempt": attempts_by_setup[setup],
                "status": status,
                "disposition": disposition,
                "recorded_utc": as_iso(recorded) or row.get("time", ""),
                "interval_since_previous_record_s": interval,
                "timing_quality": "derived inter-record interval" if interval is not None else "completion only",
                "source_name": row.get("source_name", ""),
                "copy_name": copy_name,
                "copy_exists": copy_exists,
                "copy_bytes": copy_bytes,
                "sha256": sha,
                "operator_note": " | ".join(notes_by_setup.get(setup, [])),
            })
            process_times.append({
                "visit_id": visit_id,
                "phase": "staging inter-record interval",
                "run_id": f"setup {setup} attempt {attempts_by_setup[setup]}",
                "start_utc": as_iso(prior_time),
                "end_utc": as_iso(recorded),
                "duration_s": interval,
                "quality": "derived" if interval is not None else "completion only",
                "source": "records/staging_log.csv",
                "limitation": "Includes physical staging, camera save, and operator response; not a pure setup duration.",
            })
            if recorded is not None:
                prior_time = recorded

        if selection_present:
            expected_selected = {
                setup for setup, row in actual_by_setup.items()
                if row.get("status") != "skipped"
            }
            if set(selected) != expected_selected:
                alert(alerts, "ERROR", visit_id, "seal selection",
                      "Selected setups do not equal the final non-skipped staging rows", selection_path)
                selection_valid = False
            for setup, chosen in selected.items():
                logged = actual_by_setup.get(setup, {})
                comparable = ("status", "source_name", "copy_name", "time", "sha256")
                if any(str(chosen.get(key, "")) != str(logged.get(key, "")) for key in comparable):
                    alert(alerts, "ERROR", visit_id, "seal selection",
                          f"Setup {setup} does not match its final staging row", selection_path)
                    selection_valid = False

        # Once sealed, the selection is authoritative; before sealing, the last
        # attempt is displayed as provisional.
        final_by_setup = (
            actual_by_setup
            if selection_valid or (not selection_present and not context.get("SEALED"))
            else {}
        )
        for planned_setup in plan_setups:
            setup = str(planned_setup["setup"]).zfill(2)
            capture = final_by_setup.get(setup, {})
            setup_checklist.append({
                "visit_id": visit_id,
                "setup": setup,
                "setup_type": planned_setup["setup_type"],
                "what_to_stage": planned_setup["what_to_stage"],
                "target_details": planned_setup["targets"],
                "last_filled": planned_setup["last_filled"],
                "inspection_mode": planned_setup["inspection_mode"],
                "priority": planned_setup["priority"],
                "capture_status": capture.get("status", "not recorded"),
                "capture_file": capture.get("copy_name", ""),
                "capture_time_utc": capture.get("time", ""),
                "attempts": attempts_by_setup.get(setup, 0),
                "operator_note": " | ".join(notes_by_setup.get(setup, [])),
                "plan_notes": planned_setup["notes"],
            })
        for planned in plan_observations:
            setup = str(planned["setup"])
            capture = final_by_setup.get(setup, {})
            capture_status = str(capture.get("status", "not recorded"))
            captured = capture_status in {"staged", "retake"}
            out = {
                "visit_id": visit_id,
                **planned,
                "capture_status": capture_status,
                "capture_time_utc": capture.get("time", ""),
                "capture_file": capture.get("copy_name", ""),
                "source_file": capture.get("source_name", ""),
                "capture_sha256": capture.get("sha256", ""),
                "attempts": attempts_by_setup.get(setup, 0),
                "operator_note": " | ".join(notes_by_setup.get(setup, [])),
                "confirmation": (
                    "operator issued ready/retake; report deviations separately"
                    if captured else "not confirmed"
                ),
            }
            ground_truth.append(out)

        seal_path = Path(context["SEALED"]).expanduser() if context.get("SEALED") else None
        if seal_path is not None and not seal_path.is_absolute():
            seal_path = visit / seal_path
        seal_completed = ""
        seal_record_valid = False
        if seal_path is not None:
            if not under(seal_path, visit):
                alert(alerts, "ERROR", visit_id, "seal", "SEALED path is outside the visit", seal_path)
            elif seal_path.is_file() and not seal_path.is_symlink() and seal_path.stat().st_size > 0:
                seal_completed = datetime.fromtimestamp(
                    seal_path.stat().st_mtime, timezone.utc
                ).isoformat(timespec="milliseconds")
                seal_record_valid = True
            else:
                alert(alerts, "ERROR", visit_id, "seal", "Sealed archive is missing", seal_path)
            expected_seal_sha = context.get("SEALED_SHA256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", expected_seal_sha):
                alert(alerts, "ERROR", visit_id, "seal",
                      "SEALED_SHA256 is missing or malformed", env_path)
                seal_record_valid = False
            if not selection_present or not selection_valid:
                alert(alerts, "ERROR", visit_id, "seal",
                      "A valid seal_selection.json is required for sealed evidence", selection_path)
                seal_record_valid = False
        if seal_completed:
            last_staging = max(parsed_stage_times) if parsed_stage_times else None
            process_times.append({
                "visit_id": visit_id,
                "phase": "staging-to-seal interval",
                "run_id": seal_path.name if seal_path else "",
                "start_utc": as_iso(last_staging),
                "end_utc": seal_completed,
                "duration_s": seconds(last_staging, parse_time(seal_completed)),
                "quality": "derived filesystem completion",
                "source": display(seal_path, root) if seal_path else "",
                "limitation": "The frozen kit does not log the seal command start time.",
            })

        visit_live: list[dict[str, object]] = []
        attempts_dir = visit / "attempts"
        candidates = sorted(attempts_dir.glob("live_*")) if attempts_dir.is_dir() else []
        for live_dir in [path for path in candidates if path.is_dir() and not path.is_symlink()]:
            live_id = live_dir.name
            session_path = live_dir / "session.json"
            trial_path = live_dir / "trial_log.csv"
            session = read_json(session_path, alerts, visit_id, scope=visit)
            trial_fields, trials = read_csv(trial_path, alerts, visit_id, scope=visit)
            if session_path.exists():
                provenance.append(source_record(
                    session_path, "live session", root, scope=visit
                ))
            if trial_path.exists():
                provenance.append(source_record(
                    trial_path, "live trial log", root, scope=visit
                ))

            started = parse_time(session.get("started_utc"))
            ended = parse_time(session.get("ended_utc"))
            duration = seconds(started, ended)
            if duration is not None and duration < 0:
                alert(alerts, "ERROR", visit_id, "live time", "Live run ended before it started", session_path)
                duration = None
            status_counts = session.get("status_counts") if isinstance(session.get("status_counts"), dict) else {}
            clean_stop = session.get("stop_reason") == "ctrl-c"
            if ended is None:
                alert(alerts, "INFO", visit_id, "live", f"{live_id} is incomplete or still running", session_path)
            elif not clean_stop:
                alert(alerts, "WARN", visit_id, "live", f"{live_id} did not record a clean Ctrl-C stop", session_path)
            frames = numeric(session.get("frames"))
            events = numeric(session.get("events"))
            scored_trials = [row for row in trials if row.get("status") == "scored"]
            if ended is not None and frames is not None and int(frames) != len(trials):
                alert(alerts, "WARN", visit_id, "live integrity",
                      f"session frames={int(frames)}, trial rows={len(trials)}", live_id)
            if ended is not None and events is not None and int(events) != len(scored_trials):
                alert(alerts, "WARN", visit_id, "live integrity",
                      f"session events={int(events)}, scored trial rows={len(scored_trials)}", live_id)

            live_row: dict[str, object] = {
                "visit_id": visit_id,
                "live_run": live_id,
                "session": session.get("session", live_id),
                "source": redact_source(session.get("source", "")),
                "gate": session.get("gate", ""),
                "mode": session.get("mode", ""),
                "git": session.get("git", ""),
                "started_utc": as_iso(started) or session.get("started_utc", ""),
                "ended_utc": as_iso(ended) or session.get("ended_utc", ""),
                "duration_s": duration,
                "timing_quality": "exact" if duration is not None else "incomplete",
                "stop_reason": session.get("stop_reason", ""),
                "clean_stop": clean_stop,
                "frames": frames,
                "events": events,
                "trial_rows": len(trials),
                "latency_ms_median": numeric(session.get("latency_ms_median")),
                "status_counts": json.dumps(status_counts, sort_keys=True, separators=(",", ":")),
            }
            for status in ("arm", "settling", "unchanged", "scored", "wrong_size", "unreadable",
                           "refused", "source_error", "source_recovered", "error"):
                live_row[f"status_{status}"] = numeric(status_counts.get(status)) or 0
            live_runs.append(live_row)
            visit_live.append(live_row)
            process_times.append({
                "visit_id": visit_id,
                "phase": "live run",
                "run_id": live_id,
                "start_utc": as_iso(started),
                "end_utc": as_iso(ended),
                "duration_s": duration,
                "quality": "exact" if duration is not None else "incomplete",
                "source": display(session_path, root),
                "limitation": "Exact software runtime; fixture reset before launch is separate.",
            })

            prediction_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
            if include_predictions:
                predictions_path = live_dir / "predictions.csv"
                _, predictions = read_csv(
                    predictions_path, alerts, visit_id, scope=visit
                )
                if predictions_path.exists():
                    provenance.append(source_record(
                        predictions_path, "live predictions", root, scope=visit
                    ))
                for prediction in predictions:
                    prediction_by_event[str(prediction.get("event", ""))].append(prediction)
                    position_results.append({
                        "visit_id": visit_id,
                        "live_run": live_id,
                        "session": session.get("session", live_id),
                        **{field: prediction.get(field, "") for field in PREDICTION_FIELDS},
                    })
                if ended is not None and events is not None and len(predictions) != int(events) * 30:
                    alert(alerts, "WARN", visit_id, "prediction integrity",
                          f"Expected {int(events) * 30} position rows, found {len(predictions)}",
                          predictions_path)

            previous_frame_time: datetime | None = None
            previous_frame_t: float | None = None
            previous_event_time: datetime | None = None
            for row_number, trial in enumerate(trials, 1):
                recorded = parse_time(trial.get("time_utc"))
                frame_t_value = numeric(trial.get("frame_t"))
                frame_t = float(frame_t_value) if frame_t_value is not None else None
                frame_log.append({
                    "visit_id": visit_id,
                    "live_run": live_id,
                    "row": row_number,
                    "recorded_utc": as_iso(recorded) or trial.get("time_utc", ""),
                    "wall_interval_s": seconds(previous_frame_time, recorded),
                    "input_file": trial.get("input_file", ""),
                    "frame_t": frame_t_value,
                    "frame_interval_s": (
                        round(frame_t - previous_frame_t, 3)
                        if frame_t is not None and previous_frame_t is not None else None
                    ),
                    "status": trial.get("status", ""),
                    "arm_px": numeric(trial.get("arm_px")),
                    "event": numeric(trial.get("event")),
                    "event_delay_s": numeric(trial.get("event_delay_s")),
                    "latency_ms": numeric(trial.get("latency_ms")),
                    "path": trial.get("path", ""),
                    "note": trial.get("note", ""),
                    "intervention": trial.get("intervention", ""),
                })
                status = str(trial.get("status", ""))
                if status and status not in NORMAL_STATUSES:
                    alert(alerts, "WARN", visit_id, "live frame",
                          f"{live_id} row {row_number}: {status}: {trial.get('note', '')}", trial_path)
                if trial.get("intervention"):
                    interventions.append({
                        "visit_id": visit_id,
                        "source": f"{live_id}/trial_log.csv",
                        "time": as_iso(recorded) or trial.get("time_utc", ""),
                        "action": "live intervention",
                        "operator": "",
                        "notes": trial.get("intervention", ""),
                    })
                if status == "scored":
                    event_id = str(trial.get("event", ""))
                    event_row: dict[str, object] = {
                        "visit_id": visit_id,
                        "live_run": live_id,
                        "event": numeric(event_id),
                        "recorded_utc": as_iso(recorded) or trial.get("time_utc", ""),
                        "interval_since_previous_event_s": seconds(previous_event_time, recorded),
                        "event_frame": trial.get("event_frame", ""),
                        "event_delay_s": numeric(trial.get("event_delay_s")),
                        "latency_ms": numeric(trial.get("latency_ms")),
                        "input_file": trial.get("input_file", ""),
                        "path": trial.get("path", ""),
                        "note": trial.get("note", ""),
                        "intervention": trial.get("intervention", ""),
                    }
                    if include_predictions:
                        states = Counter(
                            str(item.get("final_state", ""))
                            for item in prediction_by_event.get(event_id, [])
                        )
                        event_row.update({
                            "K": numeric(trial.get("K")),
                            "n_flagged": numeric(trial.get("n_flagged")),
                            "PASS": states["PASS"],
                            "DEFECT": states["DEFECT"],
                            "MISSED": states["MISSED"],
                            "UNSURE": states["UNSURE"],
                            "WAITING": states["WAITING"],
                        })
                    live_events.append(event_row)
                    if recorded is not None:
                        previous_event_time = recorded
                if recorded is not None:
                    previous_frame_time = recorded
                if frame_t is not None:
                    previous_frame_t = frame_t

        interventions_path = visit / "records" / "interventions.csv"
        _, intervention_rows = read_csv(
            interventions_path, alerts, visit_id, scope=visit
        )
        if interventions_path.exists():
            provenance.append(source_record(
                interventions_path, "interventions", root, scope=visit
            ))
        for row in intervention_rows:
            interventions.append({
                "visit_id": visit_id,
                "source": "records/interventions.csv",
                "time": row.get("time", ""),
                "action": row.get("action", ""),
                "operator": row.get("operator", ""),
                "notes": row.get("notes", ""),
            })

        setup_count = len(actual_by_setup)
        selected_count = len(selected)
        if context.get("READY") != "1":
            state = "SETUP INCOMPLETE"
        elif not seal_path:
            state = "STAGING / NOT SEALED"
        elif not seal_record_valid:
            state = "SEAL INVALID / REVIEW"
        elif any(not row.get("ended_utc") for row in visit_live):
            state = "LIVE IN PROGRESS / INCOMPLETE"
        elif visit_live and all(row.get("clean_stop") for row in visit_live):
            state = "COLLECTED"
        elif visit_live:
            state = "REVIEW LIVE STOP"
        else:
            state = "SEALED / NO LIVE RUN"
        summaries.append({
            "visit_id": visit_id,
            "state": state,
            "ready": context.get("READY") == "1",
            "package": context.get("PKG", ""),
            "package_sha256": context.get("PACKAGE_SHA256", ""),
            "setups_recorded": setup_count,
            "setups_expected": 22,
            "staging_attempts": len(stage_rows),
            "selected_after_seal": selected_count,
            "seal_record_valid": seal_record_valid,
            "usb_checkpoint_ok": str(context.get("SEAL_STATUS", "")).startswith("SEALED USB OK:"),
            "seal_status": context.get("SEAL_STATUS", ""),
            "seal_completed_utc": seal_completed,
            "live_runs": len(visit_live),
            "total_live_duration_s": round(sum(
                float(row["duration_s"]) for row in visit_live if row.get("duration_s") is not None
            ), 3),
            "frames": sum(int(row["frames"]) for row in visit_live if row.get("frames") is not None),
            "events": sum(int(row["events"]) for row in visit_live if row.get("events") is not None),
            "predictions_included": include_predictions,
            "generated_utc": generated,
            "report_version": REPORT_VERSION,
            "alerts": 0,
        })

    for summary in summaries:
        summary["alerts"] = sum(
            1 for item in alerts if item.get("visit_id") == summary["visit_id"]
        )

    timing_definitions = [
        {
            "metric": "staging inter-record interval",
            "quality": "derived",
            "start": "previous valid staging_log.csv time",
            "end": "current staging_log.csv time",
            "includes": "physical staging, camera save, and operator response",
            "does_not_mean": "pure hands-on setup time",
        },
        {
            "metric": "staging-to-seal interval",
            "quality": "derived filesystem completion",
            "start": "last valid staging record time",
            "end": "sealed archive modification time",
            "includes": "reconciliation, archive creation, verification, and USB copy",
            "does_not_mean": "seal command runtime (the kit does not log its start)",
        },
        {
            "metric": "live run duration",
            "quality": "exact software runtime when both timestamps exist",
            "start": "session.json started_utc",
            "end": "session.json ended_utc",
            "includes": "live software session",
            "does_not_mean": "fixture reset or preparation before launch",
        },
        {
            "metric": "frame/event interval",
            "quality": "derived",
            "start": "previous valid recorded timestamp",
            "end": "current recorded timestamp",
            "includes": "capture/replay and processing cadence",
            "does_not_mean": "a PLC or robot cycle time",
        },
    ]

    summary_headers = [
        "visit_id", "state", "ready", "package", "package_sha256",
        "setups_recorded", "setups_expected", "staging_attempts",
        "selected_after_seal", "seal_record_valid", "usb_checkpoint_ok",
        "seal_status", "seal_completed_utc", "live_runs",
        "total_live_duration_s", "frames", "events", "predictions_included",
        "generated_utc", "report_version", "alerts",
    ]
    checklist_headers = [
        "visit_id", "setup", "setup_type", "what_to_stage", "target_details",
        "last_filled", "inspection_mode", "priority", "capture_status",
        "capture_file", "capture_time_utc", "attempts", "operator_note", "plan_notes",
    ]
    ground_truth_headers = [
        "visit_id", "setup", "observation", "setup_type", "what_to_stage", "hole",
        "planned_state", "sequence_state", "target_tilt_deg", "claim_group",
        "last_filled", "inspection_mode", "priority", "plan_notes", "capture_status",
        "capture_time_utc", "capture_file", "source_file", "capture_sha256", "attempts",
        "operator_note", "confirmation",
    ]
    staging_headers = [
        "visit_id", "row", "setup", "attempt", "status", "disposition",
        "recorded_utc", "interval_since_previous_record_s", "timing_quality",
        "source_name", "copy_name", "copy_exists", "copy_bytes", "sha256",
        "operator_note",
    ]
    process_headers = [
        "visit_id", "phase", "run_id", "start_utc", "end_utc", "duration_s",
        "quality", "source", "limitation",
    ]
    live_headers = [
        "visit_id", "live_run", "session", "source", "gate", "mode", "git",
        "started_utc", "ended_utc", "duration_s", "timing_quality", "stop_reason",
        "clean_stop", "frames", "events", "trial_rows", "latency_ms_median",
        "status_counts", "status_arm", "status_settling", "status_unchanged",
        "status_scored", "status_wrong_size", "status_unreadable", "status_refused",
        "status_source_error", "status_source_recovered", "status_error",
    ]
    event_headers = [
        "visit_id", "live_run", "event", "recorded_utc",
        "interval_since_previous_event_s", "event_frame", "event_delay_s", "latency_ms",
        "input_file", "path", "note", "intervention",
    ]
    if include_predictions:
        event_headers.extend(["K", "n_flagged", "PASS", "DEFECT", "MISSED", "UNSURE", "WAITING"])
    frame_headers = [
        "visit_id", "live_run", "row", "recorded_utc", "wall_interval_s",
        "input_file", "frame_t", "frame_interval_s", "status", "arm_px", "event",
        "event_delay_s", "latency_ms", "path", "note", "intervention",
    ]
    sheets = [
        Sheet("Summary", summary_headers, summaries),
        Sheet("Setup Checklist", checklist_headers, setup_checklist),
        Sheet("Predeclared Ground Truth", ground_truth_headers, ground_truth),
        Sheet("Staging Attempts", staging_headers, staging_attempts),
        Sheet("Process Times", process_headers, process_times),
        Sheet("Live Runs", live_headers, live_runs),
        Sheet("Live Events", event_headers, live_events),
        Sheet("Frame Log", frame_headers, frame_log),
        Sheet("Interventions", ["visit_id", "source", "time", "action", "operator", "notes"], interventions),
        Sheet("Alerts", ["severity", "visit_id", "area", "message", "source"], alerts),
        Sheet(
            "Timing Definitions",
            ["metric", "quality", "start", "end", "includes", "does_not_mean"],
            timing_definitions,
        ),
        Sheet("Provenance", ["source", "role", "bytes", "modified_utc", "sha256", "state"], provenance),
    ]
    if include_predictions:
        sheets.append(Sheet(
            "Position Results",
            ["visit_id", "live_run", "session", *PREDICTION_FIELDS],
            position_results,
        ))
    metadata = {
        "report_version": REPORT_VERSION,
        "generated_utc": generated,
        "ttl_root": str(root),
        "plan": str(plan_path),
        "visits": len(visits),
        "warnings": sum(item["severity"] == "WARN" for item in alerts),
        "errors": sum(item["severity"] == "ERROR" for item in alerts),
        "include_predictions": include_predictions,
        "read_only": True,
    }
    return sheets, metadata


def clean_text(value: object) -> str:
    text = INVALID_XML.sub("�", str(value))
    if len(text) > MAX_TEXT:
        text = text[: MAX_TEXT - 14] + "…[truncated]"
    return text


def column_name(index: int) -> str:
    if index < 1 or index > MAX_COLUMNS:
        raise ValueError(f"column index out of range: {index}")
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell_xml(reference: str, value: object, *, header: bool = False) -> str:
    style = ' s="1"' if header else ""
    if value is None or value == "":
        return f'<c r="{reference}"{style}/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            value = clean_text(value)
        else:
            return f'<c r="{reference}"{style}><v>{value}</v></c>'
    text = escape(clean_text(value))
    return (
        f'<c r="{reference}" t="inlineStr"{style}>'
        f'<is><t xml:space="preserve">{text}</t></is></c>'
    )


def normalized_sheets(sheets: list[Sheet]) -> list[Sheet]:
    result: list[Sheet] = []
    used: set[str] = set()
    for sheet in sheets:
        name = INVALID_SHEET.sub("_", sheet.name).strip("'")[:31]
        if not name:
            raise ValueError("worksheet name cannot be empty")
        base = name
        suffix = 1
        while name.casefold() in used:
            suffix += 1
            tail = f"_{suffix}"
            name = base[: 31 - len(tail)] + tail
        used.add(name.casefold())
        if not sheet.headers or len(sheet.headers) > MAX_COLUMNS:
            raise ValueError(f"{name}: invalid number of columns")
        if len(sheet.rows) + 1 > MAX_ROWS:
            raise ValueError(f"{name}: too many rows for Excel")
        if len(set(sheet.headers)) != len(sheet.headers):
            raise ValueError(f"{name}: duplicate headers")
        result.append(Sheet(name, sheet.headers, sheet.rows))
    return result


def worksheet_xml(sheet: Sheet) -> str:
    last_column = column_name(len(sheet.headers))
    last_row = len(sheet.rows) + 1
    widths: list[float] = []
    for header in sheet.headers:
        sample = [clean_text(row.get(header, "")) for row in sheet.rows[:250]]
        widths.append(float(min(60, max(10, len(header) + 2, *(len(item) + 2 for item in sample)))))
    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, 1)
    )
    rows = [
        '<row r="1">' + "".join(
            cell_xml(f"{column_name(index)}1", header, header=True)
            for index, header in enumerate(sheet.headers, 1)
        ) + "</row>"
    ]
    for row_number, row in enumerate(sheet.rows, 2):
        rows.append(
            f'<row r="{row_number}">' + "".join(
                cell_xml(f"{column_name(index)}{row_number}", row.get(header, ""))
                for index, header in enumerate(sheet.headers, 1)
            ) + "</row>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_column}{last_row}"/>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<cols>{columns}</cols><sheetData>{"".join(rows)}</sheetData>'
        f'<autoFilter ref="A1:{last_column}{last_row}"/>'
        '</worksheet>'
    )


def package_parts(sheets: list[Sheet], generated_utc: str) -> list[tuple[str, str]]:
    timestamp = parse_time(generated_utc) or datetime.now(timezone.utc)
    w3c_time = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f'{sheet_overrides}</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        '</Relationships>'
    )
    workbook_sheets = "".join(
        f'<sheet name={quoteattr(sheet.name)} sheetId="{index}" r:id="rId{index}"/>'
        for index, sheet in enumerate(sheets, 1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{workbook_sheets}</sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, len(sheets) + 1)
        )
        + f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>CSI deterministic reporter</dc:creator><cp:lastModifiedBy>CSI deterministic reporter</cp:lastModifiedBy>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{w3c_time}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{w3c_time}</dcterms:modified>'
        '</cp:coreProperties>'
    )
    app = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>CSI deterministic reporter</Application></Properties>'
    )
    parts = [
        ("[Content_Types].xml", content_types),
        ("_rels/.rels", root_rels),
        ("docProps/core.xml", core),
        ("docProps/app.xml", app),
        ("xl/workbook.xml", workbook),
        ("xl/_rels/workbook.xml.rels", workbook_rels),
        ("xl/styles.xml", styles),
    ]
    parts.extend(
        (f"xl/worksheets/sheet{index}.xml", worksheet_xml(sheet))
        for index, sheet in enumerate(sheets, 1)
    )
    return parts


def validate_xlsx(path: Path, expected_sheets: int | None = None) -> None:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("XLSX ZIP checksum validation failed")
        names = set(archive.namelist())
        required = {"[Content_Types].xml", "xl/workbook.xml", "xl/styles.xml"}
        if not required.issubset(names):
            raise ValueError("XLSX package is missing required files")
        if expected_sheets is not None:
            worksheets = [name for name in names if name.startswith("xl/worksheets/sheet")]
            if len(worksheets) != expected_sheets:
                raise ValueError("XLSX worksheet count differs from the report")
        for name in sorted(names):
            if name.endswith((".xml", ".rels")):
                with archive.open(name) as stream:
                    ET.parse(stream)


def write_xlsx(output: Path, sheets: list[Sheet], generated_utc: str,
               *, overwrite: bool = False) -> Path:
    output = output.expanduser()
    if output.suffix.lower() != ".xlsx":
        raise ValueError("output filename must end in .xlsx")
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=False)
    prepared = normalized_sheets(sheets)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in package_parts(prepared, generated_utc):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload.encode("utf-8"))
        validate_xlsx(temporary, len(prepared))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
            temporary.unlink()
        try:
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        return output
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a read-only Excel report from frozen CSI collection metadata."
    )
    parser.add_argument("--ttl-root", type=Path, default=Path.home() / "ttl",
                        help="TTL evidence root (default: ~/ttl)")
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("tuesday_staging_plan.csv"),
                        help="Tuesday staging plan CSV")
    parser.add_argument("--output", type=Path, required=True,
                        help="New .xlsx path outside the evidence root")
    parser.add_argument("--include-predictions", action="store_true",
                        help="Include software predictions only after labels are locked")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing report; evidence inputs are still read-only")
    parser.add_argument("--generated-utc",
                        help="Override the workbook timestamp for reproducible builds")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = args.ttl_root.expanduser().resolve(strict=True)
        output = args.output.expanduser().resolve(strict=False)
        if under(output, root):
            raise ValueError("refusing to write the report inside the TTL evidence root")
        if args.generated_utc and parse_time(args.generated_utc) is None:
            raise ValueError("--generated-utc must be a timezone-aware ISO-8601 timestamp")
        sheets, metadata = collect_report(
            root,
            args.plan,
            include_predictions=args.include_predictions,
            generated_utc=args.generated_utc,
        )
        destination = write_xlsx(
            output, sheets, str(metadata["generated_utc"]), overwrite=args.overwrite
        )
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(
        f"Wrote {destination} | visits={metadata['visits']} "
        f"warnings={metadata['warnings']} errors={metadata['errors']} "
        f"predictions={metadata['include_predictions']}"
    )
    return 2 if metadata["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
