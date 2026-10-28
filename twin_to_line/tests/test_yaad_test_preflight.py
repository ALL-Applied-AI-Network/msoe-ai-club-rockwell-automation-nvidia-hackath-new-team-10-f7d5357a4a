"""Focused boundary checks inside the protected-test YAAD input."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import frame_warp as FW
import staging_dataset as D
from train_resnet5 import _parse_maps, verify_preprocessing_sha256


EXPECTED_CLASSES = (
    "pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"
)
EXPECTED_CODE = {
    "evaluate_resnet5_test.py", "frame_warp.py", "frame_warp_1080.json",
    "position_map_G.json", "positions.py", "registration_dev.json",
    "requirements_train_resnet5.txt", "resnet5_inspector.py", "rules.py",
    "staging_dataset.py", "tests/test_yaad_test_preflight.py",
    "train_resnet.py", "train_resnet5.py", "yaad_evaluate_resnet5.sh",
}


def _roots() -> tuple[Path, Path, Path]:
    values = [os.environ.get(name) for name in ("PROJECT_ROOT", "DATASET_ROOT", "MODEL_DIR")]
    if not all(values):
        pytest.skip("YAAD protected-test roots are not set")
    return tuple(Path(value).resolve() for value in values)  # type: ignore[return-value]


def test_exact_evaluation_boundary_and_taxonomy():
    project, dataset, model = _roots()
    code = project / "twin_to_line"
    ignored = {"__pycache__", ".pytest_cache"}
    actual = {
        str(path.relative_to(code))
        for path in code.rglob("*")
        if path.is_file() and not ignored.intersection(path.relative_to(code).parts)
    }
    assert actual == EXPECTED_CODE
    assert D.CANONICAL_CLASSES == EXPECTED_CLASSES
    assert {path.name for path in model.iterdir() if path.is_file()} == {
        "model.pt", "run.json", "thresholds.json"
    }
    assert not (dataset / "splits" / "train_manifest.csv").exists()
    assert not (dataset / "splits" / "validation_manifest.csv").exists()


def test_locked_test_pixels_and_frozen_model_are_authenticated():
    _, dataset, model = _roots()
    split_root = dataset / "splits"
    lock_path = split_root / "split_lock.json"
    manifest_path = split_root / "protected_test" / "test_manifest.csv"
    lock = D.verify_test_manifest(manifest_path, lock_path)
    rows = D.read_manifest(manifest_path, image_root=dataset / "images", verify_images=True)
    assert {row.split for row in rows} == {"test"}
    assert len(rows) == 150
    assert len({row.event_uid for row in rows}) == 5
    assert len({row.source_image_sha256 for row in rows}) == 5

    run = json.loads((model / "run.json").read_text(encoding="utf-8"))
    assert tuple(run["classes"]) == EXPECTED_CLASSES
    assert run["final_test_used"] is False
    assert run["test_rows_loaded"] == 0
    assert run["threshold_source"] == "validation_only"
    assert run["split_lock"]["sha256"] == D.sha256_file(lock_path)
    assert run["model_pt_sha256"] == D.sha256_file(model / "model.pt")
    assert run["thresholds_sha256"] == D.sha256_file(model / "thresholds.json")
    _, map_paths = _parse_maps([])
    verify_preprocessing_sha256(run["preprocessing_sha256"], FW.WARP_PATH, map_paths)
    assert lock["independent_visit_test"] is False
