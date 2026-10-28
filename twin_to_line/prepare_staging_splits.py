#!/usr/bin/env python3
"""Validate labels and freeze event-atomic, visit-isolated data splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from staging_dataset import (
    assign_event_splits,
    read_manifest,
    write_split_bundle,
)


def _event_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--test-events", type=Path, help="optional event_uid file; one per line")
    parser.add_argument("--validation-events", type=Path, help="optional event_uid file; one per line")
    parser.add_argument("--source-repo-url", default="")
    parser.add_argument("--source-git-commit", default="")
    parser.add_argument(
        "--split-unit",
        choices=("session", "event"),
        default="session",
        help="session is leakage-safe; event is a within-visit development evaluation",
    )
    parser.add_argument(
        "--acknowledge-within-visit-evaluation",
        action="store_true",
        help="required when event splits put one visit in multiple partitions",
    )
    parser.add_argument(
        "--leakage-justification",
        default="",
        help="required text explaining why a within-visit event split is being used",
    )
    args = parser.parse_args(argv)
    if args.split_unit == "event":
        if not args.acknowledge_within_visit_evaluation:
            parser.error(
                "--split-unit event requires --acknowledge-within-visit-evaluation"
            )
        if not args.leakage_justification.strip():
            parser.error("--split-unit event requires --leakage-justification")

    rows = read_manifest(args.manifest, image_root=args.image_root, verify_images=True)
    assigned, assignment = assign_event_splits(
        rows,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        explicit_test_events=_event_file(args.test_events),
        explicit_validation_events=_event_file(args.validation_events),
        split_unit=args.split_unit,
    )
    lock = write_split_bundle(
        assigned,
        assignment,
        args.out,
        source_manifest=args.manifest,
        image_root=args.image_root,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        source_repo_url=args.source_repo_url,
        source_git_commit=args.source_git_commit,
        split_unit=args.split_unit,
        acknowledge_within_visit_evaluation=args.acknowledge_within_visit_evaluation,
        leakage_justification=args.leakage_justification,
    )
    event_counts = {
        name: sum(split == name for split in assignment.values())
        for name in ("train", "validation", "test")
    }
    scope = "visit-isolated" if args.split_unit == "session" else "within-visit event holdout"
    print(f"locked {scope}, event-atomic splits {event_counts} -> {lock}")
    if args.split_unit == "event":
        print(
            "WARNING: within-visit held-out event metrics are exploratory and are not "
            "independent deployment/generalization accuracy"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
