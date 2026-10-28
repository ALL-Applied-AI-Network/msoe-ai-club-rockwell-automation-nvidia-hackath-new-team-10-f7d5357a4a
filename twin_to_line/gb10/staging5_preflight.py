#!/usr/bin/env python3
"""End-to-end golden preflight for the staging5 ONNX live runtime.

Only development frames selected by ``build_staging5_kit.py`` belong in the
preflight folder.  The builder records the golden rows once.  A deployed kit
can only verify them; it never rewrites its expected output.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys


# The script lives in app/twin_to_line/gb10 inside the kit.  Absolute script
# invocation otherwise puts only gb10/ on sys.path, not the runtime modules in
# its parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


GOLDEN_FIELDS = (
    "source_image_id",
    "fixture_id",
    "position_id",
    "expected_occupancy",
    "expected_occupancy_source",
    "predicted_visual_class",
    "model_argmax_class",
    "model_score",
    "applicable_threshold",
    "inspection_decision",
    "reason_code",
)
REQUIRED_LIVE_FIELDS = GOLDEN_FIELDS + (
    "inspection_timestamp_utc",
    "event_uid",
)
EXACT_FIELDS = tuple(
    name for name in GOLDEN_FIELDS if name not in {"model_score", "applicable_threshold"}
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        missing = [name for name in REQUIRED_LIVE_FIELDS if name not in fields]
        if missing:
            raise ValueError(f"{path}: missing required live fields {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no prediction rows")
    return rows


def _key(row: dict[str, str]) -> tuple[str, str]:
    return Path(row["source_image_id"]).name, row["position_id"]


def _normalized(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        item = {name: row[name] for name in GOLDEN_FIELDS}
        item["source_image_id"] = Path(item["source_image_id"]).name
        output.append(item)
    return sorted(output, key=_key)


def _score(frames: Path, model_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace preflight output: {output_dir}")
    if not frames.is_dir():
        raise FileNotFoundError(f"preflight frame directory is missing: {frames}")
    images = sorted(
        path for path in frames.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"no preflight images in {frames}")

    # Imports are delayed so --help and the package builder's schema tests do
    # not require OpenCV or NumPy.
    from live_inspect import Session, make_inspector, replay, run

    inspector = make_inspector("staging5", model_dir=str(model_dir), quiet=True)
    session = Session(
        str(output_dir),
        source=f"replay:{frames}",
        gate="each",
        ix=None,
        quiet=True,
        inspector=inspector,
        fixture_id="G",
    )
    run(session, replay(str(frames), ix=None))
    rows = _read_csv(output_dir / "predictions.csv")
    expected_count = len(images) * 30
    if len(rows) != expected_count:
        raise ValueError(
            f"preflight produced {len(rows)} rows, expected {expected_count} "
            f"({len(images)} images x 30 positions)"
        )
    # A video-relative/reproducible replay clock may not be an epoch and then
    # the timestamp is intentionally blank.  The required alternative event
    # identifier must always be present, along with every other audit value.
    value_required = tuple(
        name for name in REQUIRED_LIVE_FIELDS if name != "inspection_timestamp_utc"
    )
    missing_values = [
        (Path(row["source_image_id"]).name, row.get("position_id", ""), name)
        for row in rows
        for name in value_required
        if not str(row.get(name, "")).strip()
    ]
    if missing_values:
        raise ValueError(f"preflight live rows have blank required values: {missing_values[:10]}")
    return rows


def _write_expected(path: Path, rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace golden output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=GOLDEN_FIELDS)
        writer.writeheader()
        writer.writerows(_normalized(rows))


def _compare(
    expected_path: Path,
    rows: list[dict[str, str]],
    score_tolerance: float,
) -> dict[str, object]:
    with expected_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in GOLDEN_FIELDS if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{expected_path}: missing golden fields {missing}")
        expected_rows = list(reader)

    expected = {_key(row): row for row in expected_rows}
    actual = {_key(row): row for row in _normalized(rows)}
    if len(expected) != len(expected_rows) or len(actual) != len(rows):
        raise ValueError("duplicate (source_image_id, position_id) in preflight rows")

    missing_keys = sorted(set(expected) - set(actual))
    extra_keys = sorted(set(actual) - set(expected))
    exact_differences: list[dict[str, str]] = []
    max_score_difference = 0.0
    max_threshold_difference = 0.0
    for key in sorted(set(expected) & set(actual)):
        want, got = expected[key], actual[key]
        for field in EXACT_FIELDS:
            if got[field] != want[field]:
                exact_differences.append(
                    {"source_image_id": key[0], "position_id": key[1], "field": field,
                     "expected": want[field], "actual": got[field]}
                )
        score_difference = abs(float(got["model_score"]) - float(want["model_score"]))
        threshold_difference = abs(
            float(got["applicable_threshold"]) - float(want["applicable_threshold"])
        )
        if not math.isfinite(score_difference + threshold_difference):
            raise ValueError(f"non-finite preflight score for {key}")
        max_score_difference = max(max_score_difference, score_difference)
        max_threshold_difference = max(max_threshold_difference, threshold_difference)

    ok = (
        not missing_keys
        and not extra_keys
        and not exact_differences
        and max_score_difference <= score_tolerance
        and max_threshold_difference <= 1e-9
    )
    return {
        "ok": ok,
        "rows": len(actual),
        "missing_keys": [list(key) for key in missing_keys],
        "extra_keys": [list(key) for key in extra_keys],
        "exact_differences": exact_differences[:100],
        "max_score_difference": max_score_difference,
        "score_tolerance": score_tolerance,
        "max_threshold_difference": max_threshold_difference,
        "decisions": dict(Counter(row["inspection_decision"] for row in actual.values())),
        "visual_classes": dict(
            Counter(row["predicted_visual_class"] for row in actual.values())
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-tolerance", type=float, default=5e-4)
    parser.add_argument(
        "--record-development-golden",
        action="store_true",
        help="builder-only: create a new golden CSV from authenticated development frames",
    )
    args = parser.parse_args(argv)
    if args.score_tolerance < 0:
        parser.error("--score-tolerance must be non-negative")

    rows = _score(args.frames.resolve(), args.model_dir.resolve(), args.out.resolve())
    if args.record_development_golden:
        _write_expected(args.expected.resolve(), rows)
        print(f"RECORDED DEVELOPMENT GOLDEN: {args.expected} ({len(rows)} rows)")
        return 0

    result = _compare(args.expected.resolve(), rows, args.score_tolerance)
    result.update(
        {
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "frames": str(args.frames.resolve()),
            "model_dir": str(args.model_dir.resolve()),
            "expected": str(args.expected.resolve()),
        }
    )
    report = args.out / "staging5_preflight.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("STAGING5 PREFLIGHT OK" if result["ok"] else "STAGING5 PREFLIGHT FAILED")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
