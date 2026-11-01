"""Focused checks run inside the allowlisted YAAD training input."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import staging_dataset as D
import train_resnet5 as T


EXPECTED_CLASSES = (
    "pass",
    "missing_cap",
    "crooked_cap",
    "damaged_vial",
    "no_vial",
)
EXPECTED_CODE_FILES = {
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
}


def _job_roots() -> tuple[Path, Path]:
    project_text = os.environ.get("PROJECT_ROOT")
    dataset_text = os.environ.get("DATASET_ROOT")
    if not project_text or not dataset_text:
        pytest.skip("YAAD PROJECT_ROOT and DATASET_ROOT are not set")
    return Path(project_text).resolve(), Path(dataset_text).resolve()


def _source_files(root: Path) -> set[str]:
    ignored_parts = {"__pycache__", ".pytest_cache"}
    return {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not (ignored_parts & set(path.relative_to(root).parts))
    }


def test_exact_five_output_taxonomy_is_preserved():
    assert D.CANONICAL_CLASSES == EXPECTED_CLASSES
    assert T.DEFECT_INDICES == (1, 2, 3)


def test_job_code_is_exactly_allowlisted():
    project_root, _ = _job_roots()
    code_root = project_root / "twin_to_line"
    assert _source_files(code_root) == EXPECTED_CODE_FILES
    assert not (code_root / "staging").exists()


def test_job_has_only_authenticated_train_and_validation_data():
    _, dataset_root = _job_roots()
    split_root = dataset_root / "splits"
    train_path = split_root / "train_manifest.csv"
    validation_path = split_root / "validation_manifest.csv"
    lock_path = split_root / "split_lock.json"
    lock = D.verify_training_manifests(train_path, validation_path, lock_path)
    train = D.read_manifest(train_path, image_root=dataset_root / "images")
    validation = D.read_manifest(validation_path, image_root=dataset_root / "images")

    assert {row.split for row in train} == {"train"}
    assert {row.split for row in validation} == {"validation"}
    loaded_events = {row.event_uid for row in train + validation}
    assert loaded_events.isdisjoint(set(lock["events"]["test"]))
    assert not (split_root / "protected_test").exists()
    assert not any(path.name == "test_manifest.csv" for path in dataset_root.rglob("*"))

