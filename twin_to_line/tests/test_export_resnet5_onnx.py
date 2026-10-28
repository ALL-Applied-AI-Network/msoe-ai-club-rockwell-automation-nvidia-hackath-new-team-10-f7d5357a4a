"""Focused safety and equivalence tests for the staging ONNX exporter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import export_resnet5_onnx as E
from staging_dataset import CANONICAL_CLASSES, ManifestError


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_bundle(root: Path, *, unsupported=None) -> Path:
    unsupported = ["damaged_vial"] if unsupported is None else unsupported
    root.mkdir()
    (root / "model.pt").write_bytes(b"frozen model")
    thresholds = {
        "t_defect": 0.91,
        "unsupported_classes": unsupported,
    }
    (root / "thresholds.json").write_text(
        json.dumps(thresholds), encoding="utf-8"
    )
    run = {
        "classes": list(CANONICAL_CLASSES),
        "defect_classes": ["missing_cap", "crooked_cap", "damaged_vial"],
        "model_pt_sha256": _digest(root / "model.pt"),
        "thresholds_sha256": _digest(root / "thresholds.json"),
        "test_rows_loaded": 0,
        "test_images_used": 0,
        "final_test_used": False,
        "unsupported_classes": unsupported,
        "five_class_model_ready": not unsupported,
    }
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return root


def test_model_bundle_authentication_preserves_taxonomy_and_support(tmp_path):
    model_dir = _model_bundle(tmp_path / "model")
    run, thresholds = E.authenticate_model_dir(model_dir)
    assert run["classes"] == list(CANONICAL_CLASSES)
    assert run["five_class_model_ready"] is False
    assert run["unsupported_classes"] == ["damaged_vial"]
    assert thresholds["t_defect"] == 0.91

    (model_dir / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="model.pt SHA-256"):
        E.authenticate_model_dir(model_dir)


def test_authentication_rejects_a_training_run_that_accessed_test(tmp_path):
    model_dir = _model_bundle(tmp_path / "model")
    run_path = model_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["test_rows_loaded"] = 1
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ManifestError, match="zero protected-test access"):
        E.authenticate_model_dir(model_dir)


def test_development_loader_rejects_protected_rows_before_comparison(
    tmp_path, monkeypatch
):
    train_manifest = tmp_path / "train.csv"
    validation_manifest = tmp_path / "validation.csv"
    split_lock = tmp_path / "split_lock.json"
    for path in (train_manifest, validation_manifest, split_lock):
        path.write_text(path.name, encoding="utf-8")

    monkeypatch.setattr(
        E,
        "verify_training_manifests",
        lambda *args: {"events": {"test": ["protected-event"]}},
    )
    train = [
        SimpleNamespace(
            split="train", designation="protected_test", event_uid="protected-event"
        )
    ]
    validation = [
        SimpleNamespace(
            split="validation", designation="development", event_uid="validation-event"
        )
    ]
    monkeypatch.setattr(E, "read_manifest", lambda path, **kwargs: train if path == train_manifest else validation)
    run = {
        "manifests": {
            "train": _digest(train_manifest),
            "validation": _digest(validation_manifest),
        },
        "split_lock": {"sha256": _digest(split_lock)},
    }
    with pytest.raises(ManifestError, match="protected TEST rows"):
        E.load_development_rows(
            train_manifest,
            validation_manifest,
            split_lock,
            tmp_path,
            run,
        )


def test_export_cli_has_no_protected_test_input():
    destinations = {action.dest for action in E._parser()._actions}
    assert "train_manifest" in destinations
    assert "validation_manifest" in destinations
    assert "test_manifest" not in destinations


def test_numerical_and_threshold_decisions_are_compared():
    logits = np.array(
        [
            [8.0, -2.0, -3.0, -4.0, 0.0],
            [-2.0, 8.0, 0.0, -4.0, -3.0],
            [-3.0, 0.0, 7.0, -4.0, -2.0],
            [-2.0, -3.0, -4.0, -5.0, 8.0],
        ],
        dtype=np.float32,
    )
    candidate = logits + np.float32(1e-5)
    result = E.compare_outputs(
        logits,
        candidate,
        threshold=0.9,
        unsupported_classes=["damaged_vial"],
        logit_atol=1e-3,
        probability_atol=1e-4,
    )
    assert result["status"] == "passed"
    assert result["global_argmax_mismatches"] == 0
    assert result["decision_mismatches"] == 0

    changed = candidate.copy()
    changed[0] = [-2.0, 8.0, 0.0, -4.0, -3.0]
    failed = E.compare_outputs(
        logits,
        changed,
        threshold=0.9,
        unsupported_classes=["damaged_vial"],
        logit_atol=20,
        probability_atol=2,
    )
    assert failed["status"] == "failed"
    assert failed["global_argmax_mismatches"] == 1
    assert failed["decision_mismatches"] == 1


def _onnx_value(name, fixed_dimensions):
    dimensions = [SimpleNamespace(dim_param="batch", dim_value=0)] + [
        SimpleNamespace(dim_param="", dim_value=value) for value in fixed_dimensions
    ]
    tensor = SimpleNamespace(
        elem_type=1, shape=SimpleNamespace(dim=dimensions)
    )
    return SimpleNamespace(name=name, type=SimpleNamespace(tensor_type=tensor))


def test_serialized_onnx_contract_is_checked_not_assumed():
    graph = SimpleNamespace(
        input=[_onnx_value("input", [3, 128, 128])],
        output=[_onnx_value("logits", [5])],
    )
    model = SimpleNamespace(
        graph=graph, opset_import=[SimpleNamespace(domain="", version=17)]
    )
    fake_onnx = SimpleNamespace(
        TensorProto=SimpleNamespace(FLOAT=1),
        checker=SimpleNamespace(check_model=lambda value: None),
    )
    E.validate_onnx_contract(model, fake_onnx)

    graph.output = [_onnx_value("probabilities", [5])]
    with pytest.raises(RuntimeError, match="unexpected ONNX output contract"):
        E.validate_onnx_contract(model, fake_onnx)


class _FakeNet:
    def __init__(self):
        self.features = None

    def setInput(self, features, name):
        assert name == E.INPUT_NAME
        self.features = features

    def forward(self, name):
        assert name == E.OUTPUT_NAME
        return np.zeros((len(self.features), 5), dtype=np.float32)


def test_benchmark_is_the_deployed_batch_of_thirty():
    result = E.benchmark_batch30(
        _FakeNet(),
        np.zeros((30, 3, 128, 128), dtype=np.float32),
        warmup_runs=1,
        timed_runs=2,
    )
    assert result["backend"] == "opencv_dnn_cpu"
    assert result["batch_size"] == 30
    assert result["timed_runs"] == 2
    assert result["median_crops_per_second"] > 0
    with pytest.raises(ValueError, match="benchmark requires"):
        E.benchmark_batch30(
            _FakeNet(),
            np.zeros((29, 3, 128, 128), dtype=np.float32),
            warmup_runs=0,
            timed_runs=1,
        )


def test_report_records_runtime_contract_and_incomplete_five_class_readiness(tmp_path):
    model_dir = _model_bundle(tmp_path / "model")
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx bytes")
    train_manifest = tmp_path / "train.csv"
    validation_manifest = tmp_path / "validation.csv"
    split_lock = tmp_path / "split.json"
    for path in (train_manifest, validation_manifest, split_lock):
        path.write_text(path.name, encoding="utf-8")
    run = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))
    run.update(
        {
            "safety_note": "unsupported evidence is unsure",
            "evaluation_scope": "within_visit_event_holdout_not_independent",
            "independent_visit_test": False,
        }
    )
    module = SimpleNamespace(__version__="test")
    report = E._build_report(
        model_dir=model_dir,
        onnx_path=onnx_path,
        run=run,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        split_lock=split_lock,
        verification={
            "status": "passed",
            "scope": "train_and_validation_only",
            "rows_train": 360,
            "rows_validation": 150,
            "rows_total": 510,
            "test_rows_loaded": 0,
            "protected_test_used": False,
        },
        benchmark={"batch_size": 30},
        torch=module,
        torchvision=module,
        onnx=module,
    )
    assert report["input"]["shape"] == ["batch", 3, 128, 128]
    assert report["output"]["semantics"] == "raw_logits"
    assert report["onnx"]["opset"] == 17
    assert report["model_onnx_sha256"] == _digest(onnx_path)
    assert report["source_model_pt_sha256"] == _digest(model_dir / "model.pt")
    assert report["readiness"]["five_class_model_ready"] is False
    assert report["readiness"]["unsupported_classes"] == ["damaged_vial"]
    assert report["verification"]["protected_test_used"] is False
    assert report["verification"]["rows_total"] == 510
    assert any("UNKNOWN/UNSURE" in value for value in report["limitations"])
