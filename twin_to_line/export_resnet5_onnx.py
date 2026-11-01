#!/usr/bin/env python3
"""Export the frozen staging ResNet-18 to ONNX and prove runtime equivalence.

This deployment step deliberately accepts only the locked TRAIN and VALIDATION
manifests.  It never accepts, opens, or evaluates a protected TEST manifest.
Every usable development crop is compared between PyTorch and OpenCV DNN before
``onnx_export.json`` is written with ``verification.status == "passed"``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Mapping, Sequence

import cv2
import numpy as np

import frame_warp as FW
from staging5_decision_policy import POLICY_VERSION, decide_one
from staging_dataset import (
    CANONICAL_CLASSES,
    DEFECT_CLASSES,
    ManifestError,
    ManifestRow,
    StagingCropSet,
    read_manifest,
    sha256_file,
    verify_training_manifests,
)
from train_resnet5 import (
    _parse_maps,
    verify_preprocessing_sha256,
)


SCHEMA_VERSION = 1
ONNX_OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAME = "logits"
INPUT_SHAPE = (1, 3, 128, 128)
DEFAULT_LOGIT_ATOL = 1e-3
DEFAULT_PROBABILITY_ATOL = 1e-4


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read authenticated JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: expected a JSON object")
    return value


def _require_hash(path: Path, expected: object, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ManifestError(f"missing or unsafe {label}: {path}")
    actual = sha256_file(path)
    if not isinstance(expected, str) or actual != expected:
        raise ManifestError(f"{label} SHA-256 differs from its frozen provenance")
    return actual


def authenticate_model_dir(model_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Authenticate the frozen training artifacts before loading model bytes."""
    model_dir = model_dir.resolve()
    model_path = model_dir / "model.pt"
    threshold_path = model_dir / "thresholds.json"
    run_path = model_dir / "run.json"
    if not run_path.is_file() or run_path.is_symlink():
        raise ManifestError(f"missing or unsafe run.json: {run_path}")
    run = _read_json(run_path)
    thresholds = _read_json(threshold_path)

    if tuple(run.get("classes", ())) != CANONICAL_CLASSES:
        raise ManifestError("run.json does not preserve the exact five-class taxonomy")
    if tuple(run.get("defect_classes", ())) != DEFECT_CLASSES:
        raise ManifestError("run.json does not preserve the exact defect taxonomy")
    if run.get("decision_policy_version") != POLICY_VERSION:
        raise ManifestError("run.json does not identify the canonical decision policy")
    _require_hash(model_path, run.get("model_pt_sha256"), "model.pt")
    _require_hash(threshold_path, run.get("thresholds_sha256"), "thresholds.json")
    if run.get("test_rows_loaded") != 0 or run.get("test_images_used") != 0:
        raise ManifestError("training provenance does not prove zero protected-test access")
    if run.get("final_test_used") is not False:
        raise ManifestError("training provenance does not state final_test_used=false")

    unsupported = run.get("unsupported_classes")
    if not isinstance(unsupported, list) or any(
        name not in CANONICAL_CLASSES for name in unsupported
    ):
        raise ManifestError("run.json has invalid unsupported_classes")
    if thresholds.get("unsupported_classes") != unsupported:
        raise ManifestError("threshold/model unsupported-class provenance differs")
    if thresholds.get("decision_policy_version") != POLICY_VERSION:
        raise ManifestError("threshold/model decision-policy provenance differs")
    threshold = thresholds.get("t_defect")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ManifestError("thresholds.json has no finite t_defect")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ManifestError("thresholds.json t_defect is outside [0, 1]")
    if run.get("five_class_model_ready") != (not unsupported):
        raise ManifestError("run.json five_class_model_ready contradicts class support")
    return run, thresholds


def load_development_rows(
    train_manifest: Path,
    validation_manifest: Path,
    split_lock: Path,
    image_root: Path,
    run: Mapping[str, object],
) -> tuple[dict[str, object], list[ManifestRow], list[ManifestRow]]:
    """Load exactly the two development partitions and prove TEST exclusion."""
    lock = verify_training_manifests(train_manifest, validation_manifest, split_lock)
    train = read_manifest(train_manifest, image_root=image_root, verify_images=True)
    validation = read_manifest(
        validation_manifest, image_root=image_root, verify_images=True
    )
    for name, rows in (("train", train), ("validation", validation)):
        if {row.split for row in rows} != {name}:
            raise ManifestError(f"{name} manifest contains rows from another split")
        if any(row.designation == "protected_test" for row in rows):
            raise ManifestError(f"protected TEST rows reached the {name} export input")

    protected_events = set(lock.get("events", {}).get("test", []))
    development_events = {row.event_uid for row in train + validation}
    leaked = protected_events & development_events
    if leaked:
        raise ManifestError(
            f"protected TEST events reached ONNX verification: {sorted(leaked)}"
        )
    run_manifests = run.get("manifests")
    if not isinstance(run_manifests, dict):
        raise ManifestError("run.json lacks training manifest hashes")
    expected = {
        "train": sha256_file(train_manifest),
        "validation": sha256_file(validation_manifest),
    }
    if run_manifests != expected:
        raise ManifestError("export manifests differ from the manifests used for training")
    frozen_lock = run.get("split_lock")
    if not isinstance(frozen_lock, dict) or frozen_lock.get("sha256") != sha256_file(
        split_lock
    ):
        raise ManifestError("export split lock differs from the lock used for training")
    return lock, train, validation


def _make_model(torch, torchvision, model_path: Path):
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CANONICAL_CLASSES))
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch only
        state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export_model(model, torch, destination: Path) -> None:
    """Write raw five-logit ONNX with a dynamic batch dimension."""
    example = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            destination,
            export_params=True,
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAME: {0: "batch"},
                OUTPUT_NAME: {0: "batch"},
            },
            dynamo=False,
        )


def validate_onnx_contract(model, onnx) -> None:
    """Fail closed if the serialized graph does not match the runtime ABI."""
    onnx.checker.check_model(model)
    standard_opsets = [item.version for item in model.opset_import if not item.domain]
    if standard_opsets != [ONNX_OPSET]:
        raise RuntimeError(f"expected ONNX opset {ONNX_OPSET}, got {standard_opsets}")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError("ONNX graph must have exactly one input and one output")

    def contract(value):
        tensor = value.type.tensor_type
        dimensions = tensor.shape.dim
        shape = [dimensions[0].dim_param] + [item.dim_value for item in dimensions[1:]]
        return value.name, tensor.elem_type, shape

    input_contract = contract(model.graph.input[0])
    output_contract = contract(model.graph.output[0])
    float_type = onnx.TensorProto.FLOAT
    if input_contract != (INPUT_NAME, float_type, ["batch", 3, 128, 128]):
        raise RuntimeError(f"unexpected ONNX input contract: {input_contract}")
    if output_contract != (
        OUTPUT_NAME,
        float_type,
        ["batch", len(CANONICAL_CLASSES)],
    ):
        raise RuntimeError(f"unexpected ONNX output contract: {output_contract}")


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable float32 softmax matching the deployed OpenCV path."""
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    numerator = np.exp(shifted).astype(np.float32, copy=False)
    return numerator / numerator.sum(axis=1, keepdims=True)


def compare_outputs(
    torch_logits: np.ndarray,
    onnx_logits: np.ndarray,
    *,
    threshold: float,
    unsupported_classes: Sequence[str],
    logit_atol: float,
    probability_atol: float,
) -> dict[str, object]:
    """Return numerical/class/threshold-decision equivalence measurements."""
    reference = np.asarray(torch_logits, dtype=np.float32)
    candidate = np.asarray(onnx_logits, dtype=np.float32)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise RuntimeError(
            f"ONNX output shape {candidate.shape} differs from PyTorch {reference.shape}"
        )
    if reference.shape[1] != len(CANONICAL_CLASSES):
        raise RuntimeError(f"expected five raw logits, got shape {reference.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise RuntimeError("non-finite logits encountered during ONNX verification")

    reference_probabilities = softmax(reference)
    candidate_probabilities = softmax(candidate)
    max_logit_error = float(np.max(np.abs(reference - candidate)))
    max_probability_error = float(
        np.max(np.abs(reference_probabilities - candidate_probabilities))
    )
    argmax_mismatches = int(
        np.sum(reference_probabilities.argmax(1) != candidate_probabilities.argmax(1))
    )
    reference_decisions = [
        decide_one(row, threshold, unsupported_classes)[0]
        for row in reference_probabilities
    ]
    candidate_decisions = [
        decide_one(row, threshold, unsupported_classes)[0]
        for row in candidate_probabilities
    ]
    decision_mismatches = sum(
        left != right for left, right in zip(reference_decisions, candidate_decisions)
    )
    passed = (
        max_logit_error <= logit_atol
        and max_probability_error <= probability_atol
        and argmax_mismatches == 0
        and decision_mismatches == 0
    )
    return {
        "status": "passed" if passed else "failed",
        "max_abs_logit_error": max_logit_error,
        "max_abs_probability_error": max_probability_error,
        "logit_atol": float(logit_atol),
        "probability_atol": float(probability_atol),
        "global_argmax_mismatches": argmax_mismatches,
        "decision_mismatches": int(decision_mismatches),
    }


def _forward_pytorch(model, torch, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        output = model(torch.from_numpy(np.ascontiguousarray(features)))
    return output.detach().cpu().numpy().astype(np.float32, copy=False)


def _forward_opencv(net, features: np.ndarray) -> np.ndarray:
    net.setInput(np.ascontiguousarray(features, dtype=np.float32), INPUT_NAME)
    output = np.asarray(net.forward(OUTPUT_NAME), dtype=np.float32)
    return output.reshape(len(features), len(CANONICAL_CLASSES))


def compare_development_crops(
    model,
    torch,
    net,
    datasets: Mapping[str, StagingCropSet],
    *,
    batch_size: int,
    threshold: float,
    unsupported_classes: Sequence[str],
    logit_atol: float,
    probability_atol: float,
) -> dict[str, object]:
    reference_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    rows_by_split: dict[str, int] = {}
    for split in ("train", "validation"):
        dataset = datasets[split]
        rows_by_split[split] = len(dataset)
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            features = np.stack([dataset.get(index)[0] for index in range(start, stop)])
            reference_parts.append(_forward_pytorch(model, torch, features))
            candidate_parts.append(_forward_opencv(net, features))

    result = compare_outputs(
        np.concatenate(reference_parts),
        np.concatenate(candidate_parts),
        threshold=threshold,
        unsupported_classes=unsupported_classes,
        logit_atol=logit_atol,
        probability_atol=probability_atol,
    )
    result.update(
        {
            "scope": "train_and_validation_only",
            "splits": ["train", "validation"],
            "rows_by_split": rows_by_split,
            "rows_train": rows_by_split["train"],
            "rows_validation": rows_by_split["validation"],
            "rows_total": int(sum(rows_by_split.values())),
            "test_rows_loaded": 0,
            "protected_test_used": False,
        }
    )
    return result


def benchmark_batch30(
    net,
    representative: np.ndarray,
    *,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, object]:
    """Benchmark the actual OpenCV DNN call on a fixed 30-crop batch."""
    features = np.asarray(representative, dtype=np.float32)
    if features.shape != (30, 3, 128, 128):
        raise ValueError(f"benchmark requires (30,3,128,128), got {features.shape}")
    for _ in range(warmup_runs):
        _forward_opencv(net, features)
    seconds: list[float] = []
    for _ in range(timed_runs):
        started = time.perf_counter()
        _forward_opencv(net, features)
        seconds.append(time.perf_counter() - started)
    ordered = sorted(seconds)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    median = statistics.median(seconds)
    return {
        "backend": "opencv_dnn_cpu",
        "batch_size": 30,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "mean_seconds": float(statistics.fmean(seconds)),
        "median_seconds": float(median),
        "p95_seconds": float(ordered[p95_index]),
        "min_seconds": float(min(seconds)),
        "max_seconds": float(max(seconds)),
        "median_crops_per_second": float(30 / median),
        "opencv_threads": int(cv2.getNumThreads()),
    }


def _representative_batch(dataset: StagingCropSet) -> np.ndarray:
    """Return 30 deterministic development inputs without reading another split."""
    if len(dataset) < 1:
        raise ManifestError("cannot benchmark an empty development dataset")
    return np.stack([dataset.get(index % len(dataset))[0] for index in range(30)])


def _build_report(
    *,
    model_dir: Path,
    onnx_path: Path,
    run: Mapping[str, object],
    train_manifest: Path,
    validation_manifest: Path,
    split_lock: Path,
    verification: Mapping[str, object],
    benchmark: Mapping[str, object],
    torch,
    torchvision,
    onnx,
) -> dict[str, object]:
    model_sha = sha256_file(model_dir / "model.pt")
    onnx_sha = sha256_file(onnx_path)
    unsupported = list(run["unsupported_classes"])
    ready = not unsupported
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "resnet18_staging5_onnx",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classes": list(CANONICAL_CLASSES),
        "defect_classes": list(DEFECT_CLASSES),
        "input": {
            "name": INPUT_NAME,
            "shape": ["batch", 3, 128, 128],
            "dtype": "float32",
            "preprocessing": "train_resnet.to_tensor_array",
        },
        "output": {
            "name": OUTPUT_NAME,
            "shape": ["batch", len(CANONICAL_CLASSES)],
            "dtype": "float32",
            "semantics": "raw_logits",
        },
        "onnx": {
            "format": "ONNX",
            "opset": ONNX_OPSET,
            "dynamic_batch": True,
            "model_onnx_sha256": onnx_sha,
            "checker_status": "passed",
        },
        # Stable top-level aliases are part of the offline runtime contract.
        "model_onnx_sha256": onnx_sha,
        "source_model_pt_sha256": model_sha,
        "source": {
            "model_pt_sha256": model_sha,
            "run_json_sha256": sha256_file(model_dir / "run.json"),
            "thresholds_sha256": sha256_file(model_dir / "thresholds.json"),
            "split_lock_sha256": sha256_file(split_lock),
            "train_manifest_sha256": sha256_file(train_manifest),
            "validation_manifest_sha256": sha256_file(validation_manifest),
        },
        "verification": dict(verification),
        "benchmark": dict(benchmark),
        "tools": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnx": onnx.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "readiness": {
            "decision_policy_version": POLICY_VERSION,
            "five_class_model_ready": ready,
            "unsupported_classes": unsupported,
            "safety_note": run.get("safety_note"),
            "evaluation_scope": run.get("evaluation_scope"),
            "independent_visit_test": run.get("independent_visit_test"),
        },
        "limitations": [
            "This export reproduces the frozen model; it does not create new class evidence.",
            (
                "damaged_vial is unsupported and must resolve to UNKNOWN/UNSURE; "
                "the runtime must not make an evidence-free damage claim."
                if "damaged_vial" in unsupported
                else "All five visual classes have development support."
            ),
            "Export verification used TRAIN and VALIDATION only; protected TEST was not accessed.",
            "This model is for shadow/evaluation use and must not directly actuate release decisions.",
        ],
        "export_code_sha256": sha256_file(__file__),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--position-map", action="append", default=[], metavar="FIXTURE=PATH"
    )
    parser.add_argument("--frame-warp", type=Path, default=Path(FW.WARP_PATH))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--logit-atol", type=float, default=DEFAULT_LOGIT_ATOL)
    parser.add_argument(
        "--probability-atol", type=float, default=DEFAULT_PROBABILITY_ATOL
    )
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.logit_atol < 0 or args.probability_atol < 0:
        parser.error("comparison tolerances must be nonnegative")
    if args.benchmark_warmup < 0 or args.benchmark_runs < 1:
        parser.error("benchmark warmups must be nonnegative and runs must be positive")

    model_dir = args.model_dir.resolve()
    out = args.out.resolve()
    onnx_path = out / "model.onnx"
    report_path = out / "onnx_export.json"
    temporary_onnx = out / ".model.onnx.exporting"
    temporary_report = out / ".onnx_export.json.writing"
    if any(
        path.exists()
        for path in (onnx_path, report_path, temporary_onnx, temporary_report)
    ):
        raise FileExistsError(f"refusing to replace an existing ONNX export in {out}")

    run, thresholds = authenticate_model_dir(model_dir)
    lock, train_rows, validation_rows = load_development_rows(
        args.train_manifest.resolve(),
        args.validation_manifest.resolve(),
        args.split_lock.resolve(),
        args.image_root.resolve(),
        run,
    )
    del lock  # its authenticated metadata is represented in the final report
    fixture_maps, fixture_map_paths = _parse_maps(args.position_map)
    verify_preprocessing_sha256(
        run.get("preprocessing_sha256"), args.frame_warp, fixture_map_paths
    )
    warp = FW.FrameWarp(args.frame_warp)
    datasets = {
        "train": StagingCropSet(
            train_rows,
            args.image_root,
            allowed_split="train",
            fixture_maps=fixture_maps,
            warp=warp,
            seed=int(run.get("seed", 0)),
        ),
        "validation": StagingCropSet(
            validation_rows,
            args.image_root,
            allowed_split="validation",
            fixture_maps=fixture_maps,
            warp=warp,
            seed=int(run.get("seed", 0)) + 1,
        ),
    }

    import onnx
    import torch
    import torchvision

    model = _make_model(torch, torchvision, model_dir / "model.pt")
    out.mkdir(parents=True, exist_ok=True)
    try:
        export_model(model, torch, temporary_onnx)
        validate_onnx_contract(onnx.load(str(temporary_onnx)), onnx)
        net = cv2.dnn.readNetFromONNX(str(temporary_onnx))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        verification = compare_development_crops(
            model,
            torch,
            net,
            datasets,
            batch_size=args.batch,
            threshold=float(thresholds["t_defect"]),
            unsupported_classes=list(run["unsupported_classes"]),
            logit_atol=args.logit_atol,
            probability_atol=args.probability_atol,
        )
        if verification["status"] != "passed":
            raise RuntimeError(
                "ONNX/OpenCV output verification failed; refusing deployment report: "
                + json.dumps(verification, sort_keys=True)
            )
        benchmark = benchmark_batch30(
            net,
            _representative_batch(datasets["validation"]),
            warmup_runs=args.benchmark_warmup,
            timed_runs=args.benchmark_runs,
        )
        report = _build_report(
            model_dir=model_dir,
            onnx_path=temporary_onnx,
            run=run,
            train_manifest=args.train_manifest.resolve(),
            validation_manifest=args.validation_manifest.resolve(),
            split_lock=args.split_lock.resolve(),
            verification=verification,
            benchmark=benchmark,
            torch=torch,
            torchvision=torchvision,
            onnx=onnx,
        )
        with temporary_report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_onnx.replace(onnx_path)
        temporary_report.replace(report_path)
    finally:
        temporary_onnx.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
