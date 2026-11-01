#!/usr/bin/env python3
"""Build the checksum-verifying, standard-library staging collection .run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
from pathlib import Path
import re
import runpy
import shutil
import tarfile
import tempfile
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
TTL = HERE.parent
REPO = TTL.parent
KIT_NAME = "ttl_kit_staging_collection"
RUN_NAME = "vial_guard_staging_collector_2026-10-29.run"
ALL_EVENTS_NAME = "ALL_STAGING_EVENTS.md"
FILES = {
    "staging_collector.py": TTL / "staging_collector.py",
    "README_USB.txt": HERE / "staging_collection_README_USB.txt",
    "CODEX_TERMINAL_PROMPT.txt": HERE / "staging_collection_CODEX_PROMPT.txt",
    "STAGING_COLLECTION_V2.md": TTL / "staging" / "STAGING_COLLECTION_V2.md",
}
ROLE_CARD_COUNTS = {
    "rehearsal": 6,
    "train": 65,
    "validation": 30,
    "protected_test": 30,
}
SHARED_ROLE_CARD_COUNTS = {
    role: count for role, count in ROLE_CARD_COUNTS.items() if role != "protected_test"
}
ROLE_CARD_LABELS = {
    "rehearsal": "Rehearsal",
    "train": "Training",
    "validation": "Validation",
    "protected_test": "Protected test",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_files(root: Path, *, exclude: Sequence[str] = ()) -> list[Path]:
    excluded = set(exclude)
    return [
        path
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


def _collector_schedule() -> Mapping[str, object]:
    """Load the collector schedule without importing it as a package."""

    return runpy.run_path(str(TTL / "staging_collector.py"))


def _angle_from_title(title: str) -> int:
    match = re.search(r"about (\d+) degrees|at (\d+) degrees", title)
    if match is None:
        raise ValueError(f"event title has no planned angle: {title}")
    return int(next(value for value in match.groups() if value is not None))


def _mixed_angles(rows: Sequence[Mapping[str, object]]) -> tuple[int, int]:
    lean = next(
        row for row in rows
        if str(row["intended_visual_class"]) == "pass"
        and "Acceptable aligned PASS lean" in str(row["notes"])
    )
    crooked = next(
        row for row in rows if str(row["intended_visual_class"]) == "crooked_cap"
    )
    return int(str(lean["planned_vial_axis_angle_deg"])), int(
        str(crooked["planned_cap_axis_angle_deg"])
    )


def _tray_condition(event: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> list[str]:
    family = str(event["family"])
    if family == "pass_lean":
        angle = _angle_from_title(str(event["title"]))
        if angle == 90:
            return [
                "Fill all 30 positions with intact, correctly capped, upright intended-PASS vials.",
                "The three runtime-assigned primary targets are 90-degree controls; all remaining positions are also upright controls.",
            ]
        return [
            f"Fill all 30 positions. At three runtime-assigned targets, align the vial and fully seated cap at approximately {angle} degrees.",
            "Keep the other 27 positions upright, intact, correctly capped intended-PASS controls.",
            "Whole-vial lean with the cap aligned to the vial is intended PASS, not crooked_cap.",
        ]
    if family == "crooked_cap":
        angle = _angle_from_title(str(event["title"]))
        return [
            f"Fill all 30 positions. At three runtime-assigned targets, keep each vial body upright and stage its cap at approximately {angle} degrees.",
            "Each target cap must be visibly unseated or misaligned relative to its vial; angle alone is insufficient.",
            "Keep the other 27 positions upright, intact, and correctly capped intended-PASS controls.",
        ]
    if family == "missing_cap":
        return [
            "Fill all 30 positions. Remove the caps from three runtime-assigned vials while leaving the vials present.",
            "Stage those three vial bodies at approximately 80, 85, and 90 degrees, in the order shown by the collector.",
            "Keep the other 27 positions upright, intact, and correctly capped intended-PASS controls.",
        ]
    if family == "damaged_vial":
        return [
            "Fill all 30 positions. At three runtime-assigned targets, use safe approved specimens with a visible crack, chip, and deformation.",
            "Never substitute glare, whole-vial lean, or invented damage; skip with a reason if approved specimens are unavailable.",
            "Keep the other 27 positions upright, intact, and correctly capped intended-PASS controls.",
        ]
    if family == "no_vial":
        waiting = [row for row in rows if str(row["intended_gt_sequence"]) == "waiting"]
        if not waiting:
            return [
                "Declare the 30-position order complete.",
                "Leave three runtime-assigned expected-occupied positions empty; their deterministic sequence state is MISSED.",
                "Fill the other 27 positions with upright, intact, correctly capped intended-PASS controls.",
            ]
        first_waiting = min(int(str(row["position_id"])[1:]) for row in waiting)
        fill_through = first_waiting - 1
        return [
            f"Stage an in-progress live fill through position {fill_through:02d}; do not declare the order complete.",
            f"Leave one runtime-assigned expected position within 01-{fill_through:02d} empty as MISSED.",
            f"Leave positions {first_waiting:02d}-30 legitimately unfilled as WAITING, and fill every other position through {fill_through:02d} with an upright intended-PASS control.",
            "MISSED versus WAITING comes from sequence context, not image appearance.",
        ]
    if family == "mixed":
        lean_angle, cap_angle = _mixed_angles(rows)
        return [
            "Fill the fixture except for one runtime-assigned expected-occupied empty position, and declare the order complete.",
            f"Use five distinct runtime-assigned targets: aligned intended-PASS whole-vial lean at {lean_angle} degrees; visibly unseated/misaligned cap at {cap_angle} degrees with upright vial; missing cap; approved visible vial damage; and empty/MISSED.",
            "Do not stack multiple intended conditions on one vial. Keep the other 25 positions upright, intact, and correctly capped intended-PASS controls.",
        ]
    raise ValueError(f"unsupported staging family: {family}")


def _event_card(role: str, event: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    label = ROLE_CARD_LABELS[role]
    event_id = str(event["event_id"])
    condition = "\n".join(f"- {line}" for line in _tray_condition(event, rows))
    return f"""# {label} {event_id} — {event['title']}

> **INTENDED PHYSICAL SETUP ONLY — NOT VISIBLE GROUND TRUTH.** Actual visible content requires two independent blinded reviews and adjudication.
> **READ-ONLY FIXTURE-G REFERENCE.** Do not write runtime identifiers, hashes, measurements, or notes onto this checksummed package copy. The collector records those values in the session evidence.

## Event metadata

- Role: `{role}`
- Template event: `{event_id}`
- Scenario family: `{event['family']}`
- Inspection mode: `{event['inspection_mode']}`
- Fixture: `G` only; stop if the collector session is for fixture S or the approved G01–G30 numbering guide is unavailable
- Fresh physical placement required: yes

## Tray condition

{condition}

## Runtime target assignment

- The collector screen and the immutable session `plan.csv` are authoritative for exact target slots.
- Do not prefill target positions from this static card; assignments vary by session.
- Confirm the approved physical G01–G30 numbering guide before staging fixture G.

## Before typing `READY`

- [ ] This is a fresh physical placement for this event.
- [ ] The displayed target slots and tray condition match the physical tray.
- [ ] Applicable angles, measurement method, uncertainty, measurer, and UTC time were recorded contemporaneously.
- [ ] Motion has stopped and hands, robot arm, and tools are clear.
- [ ] Camera, mount, lighting, and approved settings are unchanged.
- [ ] Exactly one new baseline JPEG or non-interlaced PNG was saved to the camera directory.

Type uppercase `READY` only after every applicable box is satisfied. Use `q` to pause or `s` to skip with a reason. A retake may not move or restage any vial or cap.
"""


def _proposal_document() -> str:
    return """# Proposed partial-fill/lean extension — NOT RUNNABLE

> **STOP:** These events are not implemented in the current collector plan. Do not collect them under an existing event ID, because doing so would make the intent metadata and seal inaccurate.

The following production-like events were requested for a future versioned collector update:

| Proposed event | Tray condition |
| --- | --- |
| PL15-A | Positions 01–15 occupied; 16–30 WAITING; seven correctly capped, aligned intended-PASS vials between 80 and 90 degrees. |
| PL15-B | Same occupancy with the seven leaned intended-PASS vials rotated to different filled positions. |
| PL15-M | Fill through 15; six aligned PASS leans, one separate internal MISSED position, and 16–30 WAITING. |
| PL10 | Fill through 10 with several aligned PASS leans; 11–30 WAITING. |
| PL18 | Fill through 18 with several aligned PASS leans; 19–30 WAITING. |
| PL25 | Fill through 25 with several aligned PASS leans; 26–30 WAITING. |
| PC15 | Same visible partial tray as PL15-A, but the order is declared complete so 16–30 are MISSED rather than WAITING. |

Before these become collectable, update and test the versioned collector plan, family counts, intent rows, split totals, protocol, and package together.
"""


def write_event_cards(root: Path) -> list[Path]:
    """Generate condition-only cards for every event in the current collector."""

    root.mkdir(parents=True, exist_ok=False)
    namespace = _collector_schedule()
    build_plan = namespace["build_plan"]
    family_counts = namespace["FAMILY_COUNTS"]
    generated: list[Path] = []
    root_readme = root / "README.md"
    root_readme.write_text(
        """# Vial Guard staging-process cards

This shared directory contains one condition card for every development-side event in the current collector: 6 rehearsal, 65 training, and 30 validation events. Each event represents one complete 30-position tray state and one selected full-fixture image.

The cards describe intended setup only. Exact target positions are assigned when a session is initialized and are frozen in that session's `plan.csv`; the terminal display and `plan.csv` override these static cards. Intended setup is never automatically promoted to visible ground truth.

Protected-test event cards are deliberately absent from the shared USB/development package. The protected collector creates its 30 exact cards only inside the custodian-controlled protected session after its secret plan is initialized.

These static cards are read-only fixture-G references. Never write runtime identifiers, hashes, measurements, or notes onto the checksummed package copies. Fixture S requires its own approved guide, calibration map, and session-generated instructions.

The separately marked partial-fill/lean proposal is **not runnable** with this collector version.
""",
        encoding="utf-8",
    )
    generated.append(root_readme)
    for role, expected_count in SHARED_ROLE_CARD_COUNTS.items():
        plan, intents = build_plan(role, "G", f"documentation-{role}", f"documentation-seed-{role}")
        if len(plan) != expected_count or sum(family_counts[role].values()) != expected_count:
            raise ValueError(f"{role} event-card count differs from collector schedule")
        by_event: dict[str, list[Mapping[str, object]]] = {}
        for row in intents:
            by_event.setdefault(str(row["event_id"]), []).append(row)
        role_dir = root / role
        role_dir.mkdir()
        index_lines = [
            f"# {ROLE_CARD_LABELS[role]} staging-process index",
            "",
            f"Count: **{expected_count} independent tray events**.",
            "",
            "| Event | Family | Tray condition card |",
            "| --- | --- | --- |",
        ]
        for event in plan:
            event_id = str(event["event_id"])
            family = str(event["family"])
            filename = f"{event_id}_{family}.md"
            card = role_dir / filename
            card.write_text(_event_card(role, event, by_event[event_id]), encoding="utf-8")
            generated.append(card)
            index_lines.append(f"| {event_id} | `{family}` | [{event['title']}]({filename}) |")
        index_path = role_dir / "INDEX.md"
        index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
        generated.append(index_path)
    protected_dir = root / "protected_test"
    protected_dir.mkdir()
    protected_readme = protected_dir / "README_CUSTODIAN_ONLY.md"
    protected_readme.write_text(
        """# Protected-test event cards — custodian only

The shared USB/development package intentionally contains no per-event protected-test cards, target positions, angles, session identifiers, pixels, or labels.

The public preregistration commits only to the aggregate balance: 6 PASS-lean, 5 crooked-cap, 4 missing-cap, 5 approved damaged-vial, 4 MISSED/WAITING, and 6 mixed events. After development is frozen and the protected custodian initializes the separate protected session, the collector creates 30 exact Markdown cards inside that protected session's `records/event_cards/` directory.

Never copy those runtime cards, the protected `plan.csv`, identifiers, hashes, images, or review material back to the development USB or development account.
""",
        encoding="utf-8",
    )
    generated.append(protected_readme)
    proposal = root / "PROPOSED_PARTIAL_LEAN_EXTENSION_NOT_RUNNABLE.md"
    proposal.write_text(_proposal_document(), encoding="utf-8")
    generated.append(proposal)
    return generated


def write_all_staging_events(path: Path) -> Path:
    """Write one protected-safe operator reference covering the full schedule."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite staging-event guide: {path}")
    namespace = _collector_schedule()
    build_plan = namespace["build_plan"]
    family_counts = namespace["FAMILY_COUNTS"]
    lines = [
        "# Vial Guard — all staging events",
        "",
        "> **READ-ONLY FIXTURE-G REFERENCE.** Do not write runtime identifiers, hashes, measurements, labels, or notes into this checksummed package file.",
        "",
        "> **INTENDED PHYSICAL SETUPS ONLY — NOT VISIBLE GROUND TRUTH.** Actual visible content requires two independent blinded reviews and adjudication.",
        "",
        "This document consolidates the 101 shared event templates and the public aggregate for 30 protected events in collector version `staging-collector-v2.1.0`: 6 rehearsal, 65 training, 30 validation, and 30 protected-test events. One event is one fresh complete 30-position tray state and one selected full-fixture image.",
        "",
        "The exact five-class taxonomy and order is `pass`, `missing_cap`, `crooked_cap`, `damaged_vial`, `no_vial`. Staging intent never becomes a visible label automatically.",
        "",
        "## Collection gate for every event",
        "",
        "1. Read the condition below, then use the collector display and immutable session `plan.csv` for the exact runtime-assigned target slots.",
        "2. Stage a fresh physical tray state and contemporaneously record applicable angle measurements, method, uncertainty, measurer, and UTC time.",
        "3. Save exactly one new baseline-Huffman JPEG or non-interlaced PNG after motion stops and hands, arm, and tools are clear.",
        "4. Type uppercase `READY`. The collector will not capture or advance an event automatically.",
        "5. Confirm the retained frame. A retake may not move or restage any vial or cap; a changed state requires a new event/session.",
        "",
        "Use `q` to pause safely or `s` to skip with a required reason. A skipped event prevents a COMPLETE seal. Whole-vial lean with a fully seated aligned cap is intended PASS; cap angle alone never establishes `crooked_cap`. Empty trailing positions are WAITING during live fill and MISSED only when sequence/order context supports it. Safe approved visible damage specimens are required; never substitute glare, lean, or invented damage.",
        "",
        "These shared conditions are fixture-G references. STOP if the approved G01–G30 guide is unavailable or ambiguous. Fixture S requires its own approved guide, calibration map, and session-generated instructions.",
        "",
        "Official train, validation, and protected-test collection remains blocked without an authentic frozen synthetic-only baseline and applicable fixture map. While blocked, run rehearsal only. The collector performs no inference and controls no camera, robot, PLC, conveyor, actuator, or product-release decision.",
        "",
        "Two independent blinded reviewers plus adjudication establish actual visible labels. Protected pixels, labels, runtime cards, and results never enter training or tune model parameters or thresholds.",
        "",
    ]
    section_labels = {
        "rehearsal": "Rehearsal",
        "train": "Training",
        "validation": "Validation",
        "protected_test": "Protected test",
    }
    for role, expected_count in ROLE_CARD_COUNTS.items():
        plan, intents = build_plan(role, "G", f"documentation-{role}", f"documentation-seed-{role}")
        if len(plan) != expected_count or sum(family_counts[role].values()) != expected_count:
            raise ValueError(f"{role} consolidated event count differs from collector schedule")
        by_event: dict[str, list[Mapping[str, object]]] = {}
        for row in intents:
            by_event.setdefault(str(row["event_id"]), []).append(row)
        label = section_labels[role]
        if role == "protected_test":
            lines.extend([
                "## Protected test — public aggregate only (30 custodian-generated events)",
                "",
                "> **This section is not a protected event plan.** It discloses only the preregistered aggregate balance and condition envelopes below. It intentionally contains no protected runtime event ID or order, session ID, event UID, random seed, target position, per-event angle assignment, intended-position row, capture filename/hash, pixel, review result, or actual/adjudicated label. Intended staging families are not visible ground truth.",
                "",
                "After development is complete, sealed, verified, independently adjudicated, baseline-scored, and irreversibly frozen, a separate protected custodian verifies the five-file handoff and initializes `protected_test`. The collector then creates the exact protected setup cards only inside that custodian-controlled session at `records/event_cards/`, bound to its secret `plan.csv` and `intended_positions.csv`. The custodian must use those runtime cards—not this shared document. Never copy protected runtime cards or details back to the USB, development account, training bundle, or this file. If the protected runtime card for the current event is unavailable or differs from the collector display or plan, STOP.",
                "",
                "| Public condition envelope | Quantity | Public setup semantics |",
                "| --- | ---: | --- |",
                "| Correctly capped/aligned whole-vial lean | 6 | Intact vial; cap fully seated and aligned; approved 80–90° envelope. Exact event angle and targets remain custodian-only. |",
                "| Visibly crooked/unseated cap | 5 | Upright vial body with visible cap seating/alignment defect; angle alone is insufficient. Exact event angle and targets remain custodian-only. |",
                "| Missing cap | 4 | Vial present and cap absent. Exact body angle and targets remain custodian-only. |",
                "| Approved visible vial damage | 5 | Safe approved crack/chip/deformation specimens only; skip if unavailable. Exact specimen and targets remain custodian-only. |",
                "| `no_vial` MISSED/WAITING | 4 | Includes completed-order MISSED and live-fill MISSED-versus-WAITING context. Exact fill boundary, internal miss, positions, and order remain custodian-only. |",
                "| Mixed/confounder | 6 | Distinct intended conditions without stacking defects on one target. Exact angles, target assignment, and order remain custodian-only. |",
                "| **Total** | **30** | **Thirty independent protected tray events.** |",
                "",
            ])
            continue
        lines.extend([f"## {label} events ({expected_count})", ""])
        for ordinal, event in enumerate(plan, 1):
            heading = f"### {label} {event['event_id']} — {event['title']}"
            reference = f"- Template event: `{event['event_id']}`"
            lines.extend([
                heading,
                "",
                reference,
                f"- Scenario family: `{event['family']}`",
                f"- Inspection mode: `{event['inspection_mode']}`",
                "- Fresh physical placement required: yes",
                "- Exact primary target slots: assigned and frozen when the session starts; use the collector display and `plan.csv`",
                "",
                "Tray condition:",
                "",
            ])
            lines.extend(f"- {condition}" for condition in _tray_condition(event, by_event[str(event["event_id"])]))
            lines.append("")
    lines.extend([
        "## Proposed partial-fill/lean extension — NOT RUNNABLE",
        "",
        "> **STOP:** The following requested conditions are not implemented in the current collector plan. Do not substitute them for an existing event or collect them under an existing event ID.",
        "",
        _proposal_document().split("\n", 4)[4].strip(),
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _payload(source_root: Path) -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(source_root.rglob("*"), key=lambda item: item.relative_to(source_root).as_posix()):
            if path.is_symlink():
                raise ValueError(f"refusing symlink in payload: {path}")
            info = archive.gettarinfo(str(path), arcname=path.relative_to(source_root).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if path.is_dir():
                archive.addfile(info)
            else:
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="payload.tar", mode="wb", fileobj=compressed, mtime=0) as stream:
        stream.write(tar_buffer.getvalue())
    return compressed.getvalue()


def _header(payload_sha256: str) -> bytes:
    value = f'''#!/usr/bin/bash
set -Eeuo pipefail
IFS=$' \\t\\n'
PATH=/usr/bin:/bin
export PATH
umask 077
unset TAR_OPTIONS GZIP BZIP2 BZIP XZ_OPT CDPATH PYTHONPATH PYTHONHOME ENV BASH_ENV

EXTRACT_ONLY=""
if [[ "${{1:-}}" == "--extract-only" ]]; then
    [[ "$#" -eq 2 ]] || {{ echo "Usage: bash $0 --extract-only NEW_DIRECTORY" >&2; exit 2; }}
    EXTRACT_ONLY="$2"
fi

for REQUIRED in /usr/bin/awk /usr/bin/tail /usr/bin/tar /usr/bin/sha256sum /usr/bin/mktemp \
                /usr/bin/rm /usr/bin/cp /usr/bin/mkdir /usr/bin/python3; do
    [[ -x "$REQUIRED" ]] || {{ echo "required system tool is unavailable: $REQUIRED" >&2; exit 1; }}
done
/usr/bin/python3 -I -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {{
    echo "Python 3.10 or newer is required" >&2
    exit 1
}}

PAYLOAD_LINE="$(/usr/bin/awk '/^__VIAL_GUARD_STAGING_PAYLOAD_BELOW__$/ {{ print NR + 1; exit }}' "$0")"
[[ -n "$PAYLOAD_LINE" ]] || {{ echo "embedded payload marker not found" >&2; exit 1; }}
TEMP_ROOT="$(/usr/bin/mktemp -d /tmp/vial-guard-staging.XXXXXX)"
[[ -d "$TEMP_ROOT" && ! -L "$TEMP_ROOT" && "$TEMP_ROOT" == /tmp/vial-guard-staging.* ]] || {{
    echo "unsafe temporary directory refused" >&2
    exit 1
}}
cleanup() {{
    if [[ -n "${{TEMP_ROOT:-}}" && -d "$TEMP_ROOT" && ! -L "$TEMP_ROOT" &&
          "$TEMP_ROOT" == /tmp/vial-guard-staging.* ]]; then
        /usr/bin/rm -rf -- "$TEMP_ROOT"
    fi
}}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM
/usr/bin/tail -n "+$PAYLOAD_LINE" "$0" > "$TEMP_ROOT/payload.tar.gz"
printf '%s  %s\n' '{payload_sha256}' "$TEMP_ROOT/payload.tar.gz" | /usr/bin/sha256sum --quiet -c - || {{
    echo "launch refused: embedded payload checksum mismatch" >&2
    exit 1
}}
/usr/bin/mkdir "$TEMP_ROOT/runtime"
/usr/bin/tar --no-same-owner --no-same-permissions -xzf "$TEMP_ROOT/payload.tar.gz" -C "$TEMP_ROOT/runtime"
(cd "$TEMP_ROOT/runtime" && /usr/bin/sha256sum --quiet -c SHA256SUMS) || {{
    echo "launch refused: extracted file checksum mismatch" >&2
    exit 1
}}

if [[ -n "$EXTRACT_ONLY" ]]; then
    [[ ! -e "$EXTRACT_ONLY" ]] || {{ echo "extract destination already exists" >&2; exit 1; }}
    /usr/bin/mkdir -p "$EXTRACT_ONLY"
    /usr/bin/cp -a "$TEMP_ROOT/runtime/." "$EXTRACT_ONLY/"
    echo "Checksum-verified collector extracted to $EXTRACT_ONLY"
    exit 0
fi

/usr/bin/python3 -I "$TEMP_ROOT/runtime/staging_collector.py" "$@"
exit $?
__VIAL_GUARD_STAGING_PAYLOAD_BELOW__
'''
    return value.encode("utf-8")


def build(out_dir: Path) -> Path:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {out_dir}")
    else:
        out_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="staging-collector-build-") as temporary:
        payload_root = Path(temporary) / "payload"
        payload_root.mkdir()
        for name, source in FILES.items():
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"required source is missing or unsafe: {source}")
            shutil.copyfile(source, payload_root / name)
        write_all_staging_events(payload_root / ALL_EVENTS_NAME)
        sums = "".join(
            f"{sha256(path)}  {path.relative_to(payload_root).as_posix()}\n"
            for path in _relative_files(payload_root)
        )
        (payload_root / "SHA256SUMS").write_text(sums, encoding="utf-8")
        payload = _payload(payload_root)
    payload_sha = hashlib.sha256(payload).hexdigest()
    run_path = out_dir / RUN_NAME
    with run_path.open("xb") as stream:
        stream.write(_header(payload_sha))
        stream.write(payload)
    run_path.chmod(0o755)
    outer_sha = sha256(run_path)
    (out_dir / f"{RUN_NAME}.sha256").write_text(f"{outer_sha}  {RUN_NAME}\n", encoding="utf-8")
    for name in ("README_USB.txt", "CODEX_TERMINAL_PROMPT.txt", "STAGING_COLLECTION_V2.md"):
        shutil.copyfile(FILES[name], out_dir / name)
    write_all_staging_events(out_dir / ALL_EVENTS_NAME)
    content_lines = [
        f"{RUN_NAME}\tchecksum-verifying offline collector launcher",
        f"{RUN_NAME}.sha256\tlauncher SHA-256",
        "README_USB.txt\toperator quick start and stop conditions",
        "CODEX_TERMINAL_PROMPT.txt\tbounded instructions for compute-module Codex",
        "STAGING_COLLECTION_V2.md\tfull collection and review protocol",
    ]
    content_lines.append(f"{ALL_EVENTS_NAME}\tall staging-event conditions in one operator reference")
    content_lines.extend([
        "PACKAGE_CONTENTS.txt\tthis allowlist",
        "PACKAGE_SHA256SUMS\tchecksums for every other package file",
    ])
    contents = "\n".join(content_lines) + "\n"
    (out_dir / "PACKAGE_CONTENTS.txt").write_text(contents, encoding="utf-8")
    package_sums = "".join(
        f"{sha256(path)}  {path.relative_to(out_dir).as_posix()}\n"
        for path in _relative_files(out_dir, exclude=("PACKAGE_SHA256SUMS",))
    )
    (out_dir / "PACKAGE_SHA256SUMS").write_text(package_sums, encoding="utf-8")
    return run_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / KIT_NAME)
    args = parser.parse_args()
    run_path = build(args.out_dir)
    print(run_path)
    print(f"sha256 {sha256(run_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
