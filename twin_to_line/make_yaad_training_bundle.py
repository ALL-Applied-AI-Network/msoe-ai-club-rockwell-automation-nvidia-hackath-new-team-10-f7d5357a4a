#!/usr/bin/env python3
"""Build an allowlisted YAAD job input containing TRAIN/VALIDATION only.

The job bundle contains the small, explicit code/config allowlist needed by
``yaad_train_resnet5.sh``.  Copying the repository wholesale is forbidden: it
would expose staging review material and the protected event list to the
training container.  Protected test manifests and pixels are also excluded.
``split_lock.json`` remains present because the trainer uses it to authenticate
the train and validation manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from staging_dataset import (
    ManifestError,
    read_manifest,
    sha256_file,
    verify_training_manifests,
)


# Runtime imports, immutable geometry/config, the entry point, and its focused
# in-container preflight.  Keep this list narrow and make additions deliberate.
TRAINING_CODE_ALLOWLIST = (
    "frame_warp.py",
    "frame_warp_1080.json",
    "position_map_G.json",
    "positions.py",
    "registration_dev.json",
    "requirements_train_resnet5.txt",
    "staging5_decision_policy.py",
    "staging_dataset.py",
    "tests/test_yaad_training_preflight.py",
    "train_resnet.py",
    "train_resnet5.py",
    "yaad_train_resnet5.sh",
)

FORBIDDEN_JOB_FILENAMES = frozenset({"test_manifest.csv"})
FORBIDDEN_JOB_PREFIXES = ("capture_review", "test_events")
FORBIDDEN_JOB_DIRNAMES = frozenset({"protected_test", "staging"})


def build_bundle(split_root: Path, image_root: Path, out: Path) -> Path:
    """Build the data-only portion of a YAAD input.

    Prefer :func:`build_job_bundle` for staging a runnable job.  This lower-level
    function remains useful to tests and callers that manage code separately.
    """
    if out.exists():
        raise FileExistsError(f"refusing to replace existing YAAD data bundle: {out}")
    train_path = split_root / "train_manifest.csv"
    validation_path = split_root / "validation_manifest.csv"
    lock_path = split_root / "split_lock.json"
    lock = verify_training_manifests(train_path, validation_path, lock_path)
    train = read_manifest(train_path, image_root=image_root, verify_images=True)
    validation = read_manifest(validation_path, image_root=image_root, verify_images=True)
    if {row.split for row in train} != {"train"}:
        raise ManifestError("train_manifest.csv has a non-train row")
    if {row.split for row in validation} != {"validation"}:
        raise ManifestError("validation_manifest.csv has a non-validation row")
    protected_events = set(lock.get("events", {}).get("test", []))
    copied_rows = train + validation
    leaked = protected_events & {row.event_uid for row in copied_rows}
    if leaked:
        raise ManifestError(f"protected test events reached YAAD training bundle: {sorted(leaked)}")

    split_out = out / "splits"
    image_out = out / "images"
    split_out.mkdir(parents=True)
    image_out.mkdir()
    for source in (train_path, validation_path, lock_path):
        shutil.copy2(source, split_out / source.name)

    sources: dict[str, dict[str, str]] = {}
    root = image_root.resolve()
    for row in copied_rows:
        prior = sources.setdefault(
            row.source_image_id,
            {
                "relpath": row.source_image_relpath,
                "sha256": row.source_image_sha256,
                "split": row.split,
                "event_uid": row.event_uid,
            },
        )
        expected = {
            "relpath": row.source_image_relpath,
            "sha256": row.source_image_sha256,
            "split": row.split,
            "event_uid": row.event_uid,
        }
        if prior != expected:
            raise ManifestError(f"inconsistent source metadata for {row.source_image_id}")
    for source_id, metadata in sorted(sources.items()):
        source = (root / metadata["relpath"]).resolve()
        try:
            relative = source.relative_to(root)
        except ValueError as exc:
            raise ManifestError(f"{source_id}: path escapes image root") from exc
        destination = image_out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != metadata["sha256"]:
            raise ManifestError(f"{source_id}: copied image hash changed")

    bundle_manifest = {
        "schema_version": 1,
        "purpose": "YAAD training input; TRAIN and VALIDATION only",
        "split_lock_sha256": sha256_file(lock_path),
        "protected_test_events_recorded": len(protected_events),
        "protected_test_manifests_copied": 0,
        "protected_test_images_copied": 0,
        "evaluation_scope": lock.get("evaluation_scope"),
        "independent_visit_test": lock.get("independent_visit_test"),
        "metric_warning": lock.get("metric_warning"),
        "sessions_crossing_splits": lock.get("sessions_crossing_splits", []),
        "events": {
            "train": sorted({row.event_uid for row in train}),
            "validation": sorted({row.event_uid for row in validation}),
        },
        "images": sources,
    }
    manifest_path = out / "bundle_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(bundle_manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest_path


def _copy_training_code(repo_root: Path, destination: Path) -> None:
    source_root = repo_root.resolve() / "twin_to_line"
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing twin_to_line source directory: {source_root}")
    destination.mkdir(parents=True)
    for relative_text in TRAINING_CODE_ALLOWLIST:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ManifestError(f"unsafe training-code allowlist entry: {relative_text!r}")
        source = source_root / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(
                f"allowlisted training source is missing, not a file, or a symlink: {source}"
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def audit_job_bundle(job_root: Path) -> None:
    """Fail closed if the completed job input exceeds the explicit boundary."""
    job_root = job_root.resolve()
    code_root = job_root / "team_repo" / "twin_to_line"
    dataset_root = job_root / "dataset"
    if not code_root.is_dir() or not dataset_root.is_dir():
        raise ManifestError("YAAD job bundle must contain team_repo/twin_to_line and dataset")

    actual_code = {
        str(path.relative_to(code_root))
        for path in code_root.rglob("*")
        if path.is_file()
    }
    expected_code = set(TRAINING_CODE_ALLOWLIST)
    if actual_code != expected_code:
        raise ManifestError(
            "YAAD training code differs from its allowlist; "
            f"missing={sorted(expected_code - actual_code)}, "
            f"extra={sorted(actual_code - expected_code)}"
        )

    required_dataset = {
        "bundle_manifest.json",
        "splits/split_lock.json",
        "splits/train_manifest.csv",
        "splits/validation_manifest.csv",
    }
    actual_dataset = {
        str(path.relative_to(dataset_root))
        for path in dataset_root.rglob("*")
        if path.is_file()
    }
    missing_dataset = required_dataset - actual_dataset
    unexpected_dataset = {
        relative
        for relative in actual_dataset
        if relative not in required_dataset and not relative.startswith("images/")
    }
    if missing_dataset or unexpected_dataset:
        raise ManifestError(
            "YAAD dataset exceeds its train/validation boundary; "
            f"missing={sorted(missing_dataset)}, extra={sorted(unexpected_dataset)}"
        )

    for path in job_root.rglob("*"):
        if path.is_symlink():
            raise ManifestError(f"YAAD job bundle contains a symlink: {path}")
        relative_parts = path.relative_to(job_root).parts
        if FORBIDDEN_JOB_DIRNAMES & set(relative_parts):
            raise ManifestError(f"YAAD job bundle contains protected metadata directory: {path}")
        name = path.name
        if name in FORBIDDEN_JOB_FILENAMES or name.startswith(FORBIDDEN_JOB_PREFIXES):
            raise ManifestError(f"YAAD job bundle contains protected metadata file: {path}")

    info = json.loads((dataset_root / "bundle_manifest.json").read_text(encoding="utf-8"))
    if info.get("protected_test_manifests_copied") != 0:
        raise ManifestError("bundle manifest does not prove protected-test manifest exclusion")
    if info.get("protected_test_images_copied") != 0:
        raise ManifestError("bundle manifest does not prove protected-test pixel exclusion")


def build_job_bundle(
    split_root: Path,
    image_root: Path,
    repo_root: Path,
    out: Path,
) -> Path:
    """Atomically build a complete, runnable, allowlisted YAAD job input."""
    out = out.resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to replace existing YAAD job bundle: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
    try:
        manifest = build_bundle(split_root, image_root, staging / "dataset")
        _copy_training_code(repo_root, staging / "team_repo" / "twin_to_line")
        audit_job_bundle(staging)
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out / manifest.relative_to(staging)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing twin_to_line (default: this checkout)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_job_bundle(
        args.split_root,
        args.image_root,
        args.repo_root,
        args.out,
    )
    print(f"allowlisted YAAD job bundle excludes protected test labels/pixels -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
