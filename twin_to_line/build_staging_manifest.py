#!/usr/bin/env python3
"""Expand human-confirmed staging-plan captures into the strict 30-position manifest.

This helper never infers labels from pixels.  It may only expand a setup when a
human review row explicitly confirms that the selected final capture matches
the predeclared plan.  A deviated setup must be labeled directly in the full
manifest instead of using this helper.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

from staging_dataset import MANIFEST_COLUMNS, ManifestError, ManifestRow, sha256_file


REVIEW_COLUMNS = (
    "session_id",
    "event_id",
    "setup",
    "source_image_id",
    "source_image_relpath",
    "capture_time",
    "fixture_id",
    "designation",
    "usable",
    "confirmed_matches_plan",
    "label_source",
    "annotator_id",
    "annotation_confidence",
    "review_state",
    "notes",
)
TARGET_RE = re.compile(r"^(all|G(?:0[1-9]|[12][0-9]|30)(?:-G(?:0[1-9]|[12][0-9]|30))?)\|([^|]+)\|([^|]*)\|([^|]*)$")


def truth(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ManifestError(f"{field} must be true or false, got {value!r}")


def target_positions(value: str) -> list[str]:
    if value == "all":
        return [f"G{index:02d}" for index in range(1, 31)]
    if "-" not in value:
        return [value]
    start, end = value.split("-", 1)
    first, last = int(start[1:]), int(end[1:])
    if last < first:
        raise ManifestError(f"backwards target range {value!r}")
    return [f"G{index:02d}" for index in range(first, last + 1)]


def expand_plan_setup(plan_row: dict[str, str]) -> dict[str, tuple[str, str, str]]:
    """Return position -> (visual class, expected occupancy, sequence)."""
    last_filled = plan_row["last_filled"].strip()
    if not re.fullmatch(r"G(?:0[1-9]|[12][0-9]|30)", last_filled):
        raise ManifestError(f"setup {plan_row['setup']}: invalid last_filled {last_filled!r}")
    last = int(last_filled[1:])
    states: dict[str, tuple[str, str, str]] = {}
    for index in range(1, 31):
        position = f"G{index:02d}"
        states[position] = (
            ("pass", "occupied", "")
            if index <= last
            else ("no_vial", "empty", "waiting")
        )
    for token in plan_row["targets"].split(";"):
        match = TARGET_RE.fullmatch(token.strip())
        if not match:
            raise ManifestError(f"setup {plan_row['setup']}: invalid target token {token!r}")
        positions, staged_state, sequence, _tilt = match.groups()
        if staged_state in {"all_pass", "orientation_check", "lean_seated"}:
            visual = "pass"
        elif staged_state in {"missing_cap", "crooked_cap", "damaged_vial", "no_vial"}:
            visual = staged_state
        else:
            raise ManifestError(
                f"setup {plan_row['setup']}: staged state {staged_state!r} has no canonical mapping"
            )
        for position in target_positions(positions):
            if visual == "no_vial":
                if sequence not in {"missed", "waiting"}:
                    raise ManifestError(
                        f"setup {plan_row['setup']} {position}: no_vial requires missed/waiting"
                    )
                occupancy = "occupied" if sequence == "missed" else "empty"
                states[position] = (visual, occupancy, sequence)
            else:
                if sequence:
                    raise ManifestError(
                        f"setup {plan_row['setup']} {position}: sequence is only valid for no_vial"
                    )
                states[position] = (visual, "occupied", "")
    return states


def load_plan(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        setup = str(row.get("setup", "")).zfill(2)
        if setup in output:
            raise ManifestError(f"duplicate setup {setup} in plan")
        output[setup] = row
    if set(output) != {f"{index:02d}" for index in range(22)}:
        raise ManifestError("plan must contain exactly setups 00 through 21")
    return output


def build_rows(
    review_path: Path,
    plan_path: Path,
    image_root: Path,
    *,
    allow_intended_plan_reference: bool = False,
) -> list[ManifestRow]:
    plan = load_plan(plan_path)
    with review_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REVIEW_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"{review_path}: missing review columns {missing}")
        reviews = list(reader)
    if not reviews:
        raise ManifestError("capture review is empty")
    output: list[ManifestRow] = []
    seen_events: set[str] = set()
    root = image_root.resolve()
    for number, review in enumerate(reviews, 2):
        setup = review["setup"].strip().zfill(2)
        if setup not in plan:
            raise ManifestError(f"review row {number}: unknown setup {setup}")
        if review["fixture_id"].strip() != "G":
            raise ManifestError(
                f"review row {number}: the Tuesday plan only defines fixture G; fixture S needs its own map and labels"
            )
        usable = truth(review["usable"], "usable")
        confirmed = truth(review["confirmed_matches_plan"], "confirmed_matches_plan")
        if not usable:
            continue
        protocol_reference = review["review_state"].strip() == "protocol_reference"
        if not confirmed and not (
            allow_intended_plan_reference and protocol_reference
        ):
            raise ManifestError(
                f"review row {number}: usable capture is not confirmed to match the plan; "
                "create a direct 30-position manifest or explicitly allow intended-plan "
                "reference labels"
            )
        event_uid = f"{review['session_id'].strip()}:{review['event_id'].strip()}"
        if event_uid in seen_events:
            raise ManifestError(
                f"review row {number}: duplicate physical event {event_uid}; select one final capture/retake"
            )
        seen_events.add(event_uid)
        relative = review["source_image_relpath"].strip()
        image_path = (root / relative).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as exc:
            raise ManifestError(f"review row {number}: image path escapes image root") from exc
        if not image_path.is_file() or image_path.is_symlink():
            raise ManifestError(f"review row {number}: missing or symlinked image {image_path}")
        image_sha = sha256_file(image_path)
        states = expand_plan_setup(plan[setup])
        for position, (visual, occupancy, sequence) in sorted(states.items()):
            output.append(
                ManifestRow(
                    sample_id=f"{event_uid}:{position}",
                    event_uid=event_uid,
                    session_id=review["session_id"].strip(),
                    event_id=review["event_id"].strip(),
                    source_image_id=review["source_image_id"].strip(),
                    source_image_relpath=relative,
                    source_image_sha256=image_sha,
                    capture_time=review["capture_time"].strip(),
                    fixture_id="G",
                    position_id=position,
                    expected_occupancy=occupancy,
                    visual_class=visual,
                    inspection_mode=plan[setup]["inspection_mode"].strip(),
                    gt_sequence=sequence,
                    designation=review["designation"].strip(),
                    usable="true",
                    split="",
                    split_reason="",
                    label_source=review["label_source"].strip(),
                    annotator_id=review["annotator_id"].strip(),
                    annotation_confidence=review["annotation_confidence"].strip(),
                    review_state=review["review_state"].strip(),
                    notes=review["notes"].strip(),
                )
            )
    if not output:
        raise ManifestError("capture review contains no usable confirmed event")
    for number, row in enumerate(output, 2):
        row.validate(number)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(__file__).parents[1] / "ignite_core" / "tuesday_staging_plan.csv")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-intended-plan-reference",
        action="store_true",
        help=(
            "use protocol_reference rows as intended-state labels without claiming "
            "that image contents were visually adjudicated"
        ),
    )
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to replace existing manifest {args.out}")
    rows = build_rows(
        args.review,
        args.plan,
        args.image_root,
        allow_intended_plan_reference=args.allow_intended_plan_reference,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(row.as_csv_dict() for row in rows)
    print(
        f"wrote {len(rows)} position labels across "
        f"{len({row.event_uid for row in rows})} reference-labeled physical events -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
