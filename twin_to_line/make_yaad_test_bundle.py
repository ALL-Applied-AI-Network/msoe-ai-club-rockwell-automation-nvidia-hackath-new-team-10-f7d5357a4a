#!/usr/bin/env python3
"""Build one sealed YAAD input for protected-test evaluation.

The bundle contains only the frozen model, the locked TEST manifest/pixels,
and an explicit evaluation-code allowlist.  It never contains TRAIN or
VALIDATION manifests/pixels and refuses to overwrite an existing bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import frame_warp as FW
from staging_dataset import (
    CANONICAL_CLASSES,
    ManifestError,
    read_manifest,
    sha256_file,
    verify_test_manifest,
)
from train_resnet5 import _parse_maps, verify_preprocessing_sha256


EVALUATION_CODE_ALLOWLIST = (
    "evaluate_resnet5_test.py",
    "frame_warp.py",
    "frame_warp_1080.json",
    "position_map_G.json",
    "positions.py",
    "registration_dev.json",
    "requirements_train_resnet5.txt",
    "resnet5_inspector.py",
    "rules.py",
    "staging_dataset.py",
    "tests/test_yaad_test_preflight.py",
    "train_resnet.py",
    "train_resnet5.py",
    "yaad_evaluate_resnet5.sh",
)
MODEL_FILES = ("model.pt", "run.json", "thresholds.json")


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"required regular file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _verify_frozen_model(model_dir: Path, split_lock: Path) -> dict[str, object]:
    run_path = model_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if tuple(run.get("classes", ())) != CANONICAL_CLASSES:
        raise ManifestError("frozen model does not preserve the exact five-class taxonomy")
    if run.get("test_rows_loaded") != 0 or run.get("test_images_used") != 0:
        raise ManifestError("frozen model provenance does not prove test isolation")
    if run.get("final_test_used") is not False:
        raise ManifestError("frozen model already records final-test use")
    if run.get("threshold_source") != "validation_only":
        raise ManifestError("frozen threshold is not validation-only")
    if run.get("split_lock", {}).get("sha256") != sha256_file(split_lock):
        raise ManifestError("frozen model was not trained from this split lock")
    if run.get("model_pt_sha256") != sha256_file(model_dir / "model.pt"):
        raise ManifestError("model.pt differs from frozen run.json")
    if run.get("thresholds_sha256") != sha256_file(model_dir / "thresholds.json"):
        raise ManifestError("thresholds.json differs from frozen run.json")
    _, fixture_paths = _parse_maps([])
    verify_preprocessing_sha256(
        run.get("preprocessing_sha256"), FW.WARP_PATH, fixture_paths
    )
    return run


def audit_test_bundle(root: Path) -> None:
    root = root.resolve()
    code_root = root / "team_repo" / "twin_to_line"
    dataset_root = root / "dataset"
    model_root = root / "model" / "resnet18_staging5"
    actual_code = {
        str(path.relative_to(code_root))
        for path in code_root.rglob("*")
        if path.is_file()
    }
    if actual_code != set(EVALUATION_CODE_ALLOWLIST):
        raise ManifestError("evaluation code differs from its exact allowlist")
    actual_model = {
        str(path.relative_to(model_root))
        for path in model_root.rglob("*")
        if path.is_file()
    }
    if actual_model != set(MODEL_FILES):
        raise ManifestError("evaluation model directory has missing or extra files")
    required = {
        "splits/split_lock.json",
        "splits/protected_test/test_manifest.csv",
    }
    actual_dataset = {
        str(path.relative_to(dataset_root))
        for path in dataset_root.rglob("*")
        if path.is_file()
    }
    extra = {
        name for name in actual_dataset
        if name not in required and not name.startswith("images/")
    }
    if not required <= actual_dataset or extra:
        raise ManifestError(
            f"protected-test dataset boundary failed; missing={sorted(required-actual_dataset)}, "
            f"extra={sorted(extra)}"
        )
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ManifestError("protected-test bundle contains a symlink")
    forbidden = {"train_manifest.csv", "validation_manifest.csv"}
    if any(path.name in forbidden for path in root.rglob("*")):
        raise ManifestError("development manifest reached protected-test bundle")


def build_test_bundle(
    *,
    split_root: Path,
    image_root: Path,
    model_dir: Path,
    repo_root: Path,
    out: Path,
) -> Path:
    out = out.resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to replace protected-test bundle: {out}")
    test_manifest = split_root / "protected_test" / "test_manifest.csv"
    split_lock = split_root / "split_lock.json"
    verify_test_manifest(test_manifest, split_lock)
    rows = read_manifest(test_manifest, image_root=image_root, verify_images=True)
    if {row.split for row in rows} != {"test"}:
        raise ManifestError("protected manifest contains a non-test row")
    if len({row.event_uid for row in rows}) != 5:
        raise ManifestError("expected exactly five protected events")
    run = _verify_frozen_model(model_dir, split_lock)

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
    try:
        code_source = repo_root.resolve() / "twin_to_line"
        code_out = staging / "team_repo" / "twin_to_line"
        for relative_text in EVALUATION_CODE_ALLOWLIST:
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                raise ManifestError(f"unsafe allowlist entry: {relative_text}")
            _copy_file(code_source / relative, code_out / relative)

        split_out = staging / "dataset" / "splits"
        _copy_file(split_lock, split_out / "split_lock.json")
        _copy_file(
            test_manifest,
            split_out / "protected_test" / "test_manifest.csv",
        )
        source_root = image_root.resolve()
        image_out = staging / "dataset" / "images"
        seen: dict[str, str] = {}
        for row in rows:
            prior = seen.setdefault(row.source_image_relpath, row.source_image_sha256)
            if prior != row.source_image_sha256:
                raise ManifestError(f"inconsistent image hash for {row.source_image_relpath}")
        for relative_text, expected_sha in sorted(seen.items()):
            relative = Path(relative_text)
            source = (source_root / relative).resolve()
            try:
                source.relative_to(source_root)
            except ValueError as exc:
                raise ManifestError(f"test image escapes image root: {relative}") from exc
            destination = image_out / relative
            _copy_file(source, destination)
            if sha256_file(destination) != expected_sha:
                raise ManifestError(f"copied test image hash changed: {relative}")
        if len(seen) != 5:
            raise ManifestError(f"expected exactly five protected images, got {len(seen)}")

        model_out = staging / "model" / "resnet18_staging5"
        for name in MODEL_FILES:
            _copy_file(model_dir / name, model_out / name)
        if sha256_file(model_out / "model.pt") != run["model_pt_sha256"]:
            raise ManifestError("copied model hash changed")

        audit_test_bundle(staging)
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    path = build_test_bundle(
        split_root=args.split_root,
        image_root=args.image_root,
        model_dir=args.model_dir,
        repo_root=args.repo_root,
        out=args.out,
    )
    print(f"sealed protected-test YAAD bundle -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
