"""Offline OpenCV-DNN runtime for the frozen five-output staging model.

The ONNX artifact is an inference copy of the frozen PyTorch model.  This
module deliberately depends on OpenCV and NumPy, not PyTorch, so the runtime
can be carried to a GB10 compute module in an offline kit.  Construction is
fail closed: the ONNX export report, training run, threshold, model and every
recorded preprocessing input must form one authenticated provenance chain.

This inspector is shadow-only.  In particular, ``damaged_vial`` had no
development support in the current run.  If an unsupported class is the
global probability maximum, the result is ``unknown`` even when the summed
defect score is below the validation threshold.  Deterministic manufacturing
rules must turn that result into UNSURE/manual review; it can never pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import cv2
import numpy as np

import positions as P
from rules import Inspector, Observation
from staging5_decision_policy import POLICY_VERSION, decide_one
from staging_dataset import CANONICAL_CLASSES, DEFECT_CLASSES
from train_resnet import to_tensor_array


DEFAULT_MODEL_DIR = os.path.join(P.HERE, "results", "train", "resnet18_staging5")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_FILENAME = "model.onnx"
EXPORT_FILENAME = "onnx_export.json"
RUN_FILENAME = "run.json"
THRESHOLD_FILENAME = "thresholds.json"


def sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read authenticated {label} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: authenticated {label} must be a JSON object")
    return value


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{path}: required {label} is missing, not regular, or a symlink")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label}: expected a lowercase SHA-256 digest")
    return value


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected a JSON object")
    return value


def _runtime_preprocessing_paths() -> dict[str, Path]:
    """Map the frozen training inventory names to their runtime counterparts."""
    here = Path(P.HERE)
    return {
        "frame_warp.py": here / "frame_warp.py",
        "frame_warp_1080.json": here / "frame_warp_1080.json",
        "position_map_G": Path(P.MAP_PATH),
        "positions.py": Path(P.__file__),
        "registration_dev.json": Path(P.REG_PATH),
        "staging5_decision_policy.py": here / "staging5_decision_policy.py",
        "staging_dataset.py": here / "staging_dataset.py",
        "train_resnet.py": here / "train_resnet.py",
        "train_resnet5.py": here / "train_resnet5.py",
    }


def _relative(path: Path) -> str:
    return os.path.relpath(path, P.HERE)


class ResNet5OnnxInspector(Inspector):
    """Five-output ResNet-18 executed by OpenCV's CPU DNN backend."""

    name = "resnet18_staging5_onnx"

    def __init__(self, model_dir: os.PathLike[str] | str = DEFAULT_MODEL_DIR, net=None):
        self.dir = os.path.abspath(os.fspath(model_dir))
        root = Path(self.dir)
        self.model_path = root / MODEL_FILENAME
        self.export_path = root / EXPORT_FILENAME
        self.run_path = root / RUN_FILENAME
        self.threshold_path = root / THRESHOLD_FILENAME
        for path, label in (
            (self.model_path, "ONNX model"),
            (self.export_path, "ONNX export report"),
            (self.run_path, "training run metadata"),
            (self.threshold_path, "validation threshold"),
        ):
            _require_file(path, label)

        self.export = _load_json(self.export_path, "ONNX export report")
        self.run = _load_json(self.run_path, "training run metadata")
        self.thresholds = _load_json(self.threshold_path, "validation threshold")
        self._files = self._authenticate()

        # Explicit CPU settings keep behavior independent of optional CUDA
        # support in the OpenCV build installed on the target.
        self.device = "cpu"
        try:
            self.net = net if net is not None else cv2.dnn.readNetFromONNX(str(self.model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except (cv2.error, OSError, RuntimeError) as exc:
            raise ValueError(f"{self.model_path}: OpenCV could not load authenticated ONNX model") from exc

    def _authenticate(self) -> dict[str, str]:
        """Verify artifact lineage and return the complete runtime file inventory."""
        export = self.export
        run = self.run
        threshold = self.thresholds

        if export.get("schema_version") != 1:
            raise ValueError(f"{self.export_path}: unsupported ONNX export schema")
        if export.get("artifact_kind") != "resnet18_staging5_onnx":
            raise ValueError(f"{self.export_path}: wrong artifact kind")
        if tuple(export.get("classes", ())) != CANONICAL_CLASSES:
            raise ValueError(f"{self.export_path}: class order is not the exact five-class taxonomy")
        if tuple(run.get("classes", ())) != CANONICAL_CLASSES:
            raise ValueError(f"{self.run_path}: class order is not the exact five-class taxonomy")
        if run.get("decision_policy_version") != POLICY_VERSION:
            raise ValueError(f"{self.run_path}: decision policy is not {POLICY_VERSION}")

        input_meta = _require_mapping(export.get("input"), f"{self.export_path}: input")
        if input_meta.get("name") != "input":
            raise ValueError(f"{self.export_path}: ONNX input must be named input")
        if input_meta.get("shape") != ["batch", 3, 128, 128]:
            raise ValueError(f"{self.export_path}: unexpected ONNX input shape")
        if input_meta.get("dtype") != "float32":
            raise ValueError(f"{self.export_path}: ONNX input must be float32")
        if input_meta.get("preprocessing") != "train_resnet.to_tensor_array":
            raise ValueError(f"{self.export_path}: preprocessing contract differs from training")

        output_meta = _require_mapping(export.get("output"), f"{self.export_path}: output")
        if output_meta.get("name") != "logits":
            raise ValueError(f"{self.export_path}: ONNX output must be named logits")
        if output_meta.get("shape") != ["batch", len(CANONICAL_CLASSES)]:
            raise ValueError(f"{self.export_path}: unexpected ONNX output shape")
        if output_meta.get("dtype") != "float32" or output_meta.get("semantics") != "raw_logits":
            raise ValueError(f"{self.export_path}: ONNX output must be float32 raw logits")

        onnx_meta = _require_mapping(export.get("onnx"), f"{self.export_path}: onnx")
        if onnx_meta.get("opset") != 17 or onnx_meta.get("dynamic_batch") is not True:
            raise ValueError(f"{self.export_path}: ONNX export must use opset 17 and dynamic batch")
        expected_model = _require_sha256(
            onnx_meta.get("model_onnx_sha256"), f"{self.export_path}: model_onnx_sha256"
        )
        if export.get("model_onnx_sha256") != expected_model:
            raise ValueError(f"{self.export_path}: top-level ONNX digest alias disagrees")
        actual_model = sha256(self.model_path)
        if actual_model != expected_model:
            raise ValueError(f"{self.model_path}: SHA-256 differs from ONNX export report")

        source = _require_mapping(export.get("source"), f"{self.export_path}: source")
        expected_run = _require_sha256(
            source.get("run_json_sha256"), f"{self.export_path}: source.run_json_sha256"
        )
        if sha256(self.run_path) != expected_run:
            raise ValueError(f"{self.run_path}: SHA-256 differs from ONNX export report")
        expected_threshold = _require_sha256(
            source.get("thresholds_sha256"),
            f"{self.export_path}: source.thresholds_sha256",
        )
        if sha256(self.threshold_path) != expected_threshold:
            raise ValueError(f"{self.threshold_path}: SHA-256 differs from ONNX export report")
        if run.get("thresholds_sha256") != expected_threshold:
            raise ValueError(f"{self.run_path}: threshold digest disagrees with ONNX export report")

        source_model = _require_sha256(
            source.get("model_pt_sha256"), f"{self.export_path}: source.model_pt_sha256"
        )
        if export.get("source_model_pt_sha256") != source_model:
            raise ValueError(f"{self.export_path}: top-level source model digest alias disagrees")
        if run.get("model_pt_sha256") != source_model:
            raise ValueError(f"{self.run_path}: source model digest disagrees with ONNX export report")
        model_pt_path = Path(self.dir) / "model.pt"
        if model_pt_path.exists():
            _require_file(model_pt_path, "optional source PyTorch model")
            if sha256(model_pt_path) != source_model:
                raise ValueError(f"{model_pt_path}: SHA-256 differs from training run")

        manifests = _require_mapping(run.get("manifests"), f"{self.run_path}: manifests")
        split_lock = _require_mapping(run.get("split_lock"), f"{self.run_path}: split_lock")
        lineage = (
            ("train_manifest_sha256", manifests.get("train")),
            ("validation_manifest_sha256", manifests.get("validation")),
            ("split_lock_sha256", split_lock.get("sha256")),
        )
        for export_name, run_value in lineage:
            source_value = _require_sha256(
                source.get(export_name), f"{self.export_path}: source.{export_name}"
            )
            if source_value != run_value:
                raise ValueError(f"{self.export_path}: {export_name} disagrees with training run")

        if run.get("final_test_used") is not False or run.get("test_rows_loaded") != 0:
            raise ValueError(f"{self.run_path}: training provenance does not prove protected-test isolation")
        if run.get("threshold_source") != "validation_only":
            raise ValueError(f"{self.run_path}: threshold source is not validation-only")
        if threshold.get("threshold_source") != "validation_only":
            raise ValueError(f"{self.threshold_path}: threshold is not validation-only")
        if threshold.get("decision_policy_version") != POLICY_VERSION:
            raise ValueError(f"{self.threshold_path}: decision policy is not {POLICY_VERSION}")
        try:
            t_defect = float(threshold["t_defect"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{self.threshold_path}: invalid t_defect") from exc
        if not math.isfinite(t_defect) or not 0.0 <= t_defect <= 1.0:
            raise ValueError(f"{self.threshold_path}: t_defect must be finite and in [0, 1]")
        self.t_defect = t_defect
        self.threshold = t_defect

        unsupported = tuple(run.get("unsupported_classes", ()))
        if len(set(unsupported)) != len(unsupported):
            raise ValueError(f"{self.run_path}: duplicate unsupported classes")
        unknown = set(unsupported) - set(CANONICAL_CLASSES)
        if unknown:
            raise ValueError(f"{self.run_path}: unknown unsupported classes {sorted(unknown)}")
        if set(unsupported) - {"damaged_vial"}:
            raise ValueError(f"{self.run_path}: only damaged_vial may lack development support")
        if tuple(threshold.get("unsupported_classes", ())) != unsupported:
            raise ValueError(f"{self.threshold_path}: unsupported classes disagree with training run")
        if run.get("five_class_model_ready") is not (not unsupported):
            raise ValueError(f"{self.run_path}: five-class readiness disagrees with class support")
        self.unsupported = unsupported

        verification = _require_mapping(
            export.get("verification"), f"{self.export_path}: verification"
        )
        if verification.get("status") != "passed":
            raise ValueError(f"{self.export_path}: ONNX equivalence verification did not pass")
        if verification.get("scope") != "train_and_validation_only":
            raise ValueError(f"{self.export_path}: ONNX verification scope is not development-only")
        if verification.get("protected_test_used") is not False or verification.get("test_rows_loaded") != 0:
            raise ValueError(f"{self.export_path}: ONNX verification does not prove protected-test isolation")
        if verification.get("decision_mismatches") != 0:
            raise ValueError(f"{self.export_path}: ONNX decisions differ from the frozen PyTorch model")
        try:
            n_train = int(verification["rows_train"])
            n_validation = int(verification["rows_validation"])
            n_total = int(verification["rows_total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{self.export_path}: invalid ONNX verification row counts") from exc
        if n_train <= 0 or n_validation <= 0 or n_total != n_train + n_validation:
            raise ValueError(f"{self.export_path}: ONNX verification row counts are inconsistent")

        readiness = _require_mapping(export.get("readiness"), f"{self.export_path}: readiness")
        readiness_pairs = {
            "decision_policy_version": POLICY_VERSION,
            "five_class_model_ready": run.get("five_class_model_ready"),
            "unsupported_classes": list(unsupported),
            "evaluation_scope": run.get("evaluation_scope"),
            "independent_visit_test": run.get("independent_visit_test"),
        }
        for key, expected in readiness_pairs.items():
            if readiness.get(key) != expected:
                raise ValueError(f"{self.export_path}: readiness.{key} disagrees with training run")
        safety_note = readiness.get("safety_note")
        if not isinstance(safety_note, str) or not safety_note.strip():
            raise ValueError(f"{self.export_path}: missing readiness safety note")
        _require_sha256(export.get("export_code_sha256"), f"{self.export_path}: export_code_sha256")

        tools = _require_mapping(export.get("tools"), f"{self.export_path}: tools")
        if tools.get("opencv") != cv2.__version__:
            raise ValueError(
                f"{self.export_path}: OpenCV {tools.get('opencv')!r} verified the export, "
                f"but runtime has {cv2.__version__!r}"
            )
        if tools.get("numpy") != np.__version__:
            raise ValueError(
                f"{self.export_path}: NumPy {tools.get('numpy')!r} verified the export, "
                f"but runtime has {np.__version__!r}"
            )

        recorded_preprocessing = _require_mapping(
            run.get("preprocessing_sha256"), f"{self.run_path}: preprocessing_sha256"
        )
        runtime_paths = _runtime_preprocessing_paths()
        if set(recorded_preprocessing) != set(runtime_paths):
            missing = sorted(set(runtime_paths) - set(recorded_preprocessing))
            unexpected = sorted(set(recorded_preprocessing) - set(runtime_paths))
            raise ValueError(
                f"{self.run_path}: preprocessing inventory differs from runtime "
                f"(missing={missing}, unexpected={unexpected})"
            )

        # Keep model.onnx first: the surrounding live logger treats the first
        # file digest as the model identity.
        files = {
            _relative(self.model_path): actual_model,
            _relative(self.export_path): sha256(self.export_path),
            _relative(self.run_path): expected_run,
            _relative(self.threshold_path): expected_threshold,
        }
        if model_pt_path.exists():
            files[_relative(model_pt_path)] = source_model
        for name, path in runtime_paths.items():
            expected = recorded_preprocessing[name]
            if expected is None:
                if path.exists():
                    raise ValueError(f"{path}: training recorded this preprocessing file as absent")
                continue
            expected = _require_sha256(expected, f"{self.run_path}: preprocessing_sha256.{name}")
            _require_file(path, f"preprocessing input {name}")
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"{path}: preprocessing SHA-256 differs from training run for {name}")
            files[_relative(path)] = actual
        return files

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / np.sum(exponent, axis=1, keepdims=True)

    def _forward_logits(self, features: np.ndarray) -> np.ndarray:
        batch = np.ascontiguousarray(features, dtype=np.float32)
        self.net.setInput(batch, "input")
        try:
            logits = np.asarray(self.net.forward("logits"), dtype=np.float32)
        except (cv2.error, RuntimeError) as exc:
            raise RuntimeError("OpenCV DNN inference failed for authenticated staging model") from exc
        expected = (len(batch), len(CANONICAL_CLASSES))
        if logits.shape != expected:
            raise RuntimeError(f"ONNX logits have shape {logits.shape}, expected {expected}")
        if not np.isfinite(logits).all():
            raise RuntimeError("ONNX logits contain a non-finite value")
        return logits

    def probabilities(self, crops: Sequence[object]) -> np.ndarray:
        if not crops:
            return np.empty((0, len(CANONICAL_CLASSES)), dtype=np.float32)
        features = np.stack([to_tensor_array(crop.image) for crop in crops]).astype(
            np.float32, copy=False
        )
        return self._softmax(self._forward_logits(features)).astype(np.float32, copy=False)

    def observations(self, probabilities: np.ndarray) -> list[Observation]:
        values = np.asarray(probabilities, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(CANONICAL_CLASSES):
            raise ValueError(
                f"probabilities must have shape (batch, {len(CANONICAL_CLASSES)})"
            )
        output = []
        for probability in values:
            predicted, score, reason_code = decide_one(
                probability, self.t_defect, self.unsupported
            )
            global_class = (
                CANONICAL_CLASSES[int(np.argmax(probability))]
                if np.isfinite(probability).all()
                else ""
            )
            features = {
                f"p_{name}": (
                    round(float(value), 8) if np.isfinite(float(value)) else None
                )
                for name, value in zip(CANONICAL_CLASSES, probability)
            }
            features.update(
                {
                    "t_defect": self.t_defect,
                    "reason_code": reason_code,
                    "unsupported_classes": list(self.unsupported),
                    # Raw five-class model output for audit only.  It is deliberately separate from the
                    # safety-gated Observation.visual_state used by deterministic process rules.
                    "model_argmax_class": global_class,
                    "deployment_mode": "shadow_only",
                    "decision_policy_version": POLICY_VERSION,
                }
            )
            if predicted == "unknown":
                reason = (
                    "raw model output is unsupported, ambiguous, or invalid; this is not a "
                    "defect detection; manual review required; automatic pass prohibited"
                )
            else:
                reason = f"defect score {float(score):.6f}; threshold {self.t_defect:.6f}"
            output.append(
                Observation(
                    predicted,
                    None if score is None else round(score, 8),
                    features,
                    reason,
                )
            )
        return output

    def inspect_batch(self, crops):
        return self.observations(self.probabilities(crops))

    def inspect(self, crop):
        return self.inspect_batch([crop])[0]

    def warmup(self, n: int = 30) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("warmup batch size must be a positive integer")
        self._forward_logits(np.zeros((n, 3, 128, 128), dtype=np.float32))

    def info(self) -> dict:
        limitations = []
        if self.unsupported:
            limitations.append(
                "unsupported classes become UNKNOWN/UNSURE: " + ", ".join(self.unsupported)
            )
        if self.run.get("independent_visit_test") is not True:
            limitations.append("no independent-visit test")
        return {
            "name": self.name,
            "files": dict(self._files),
            "model_sha256": sha256(self.model_path),
            "source_model_pt_sha256": self.run["model_pt_sha256"],
            "classes": list(CANONICAL_CLASSES),
            "defect_classes": list(DEFECT_CLASSES),
            "t_defect": self.t_defect,
            "threshold": self.t_defect,
            "threshold_source": "validation_only",
            "decision_policy_version": POLICY_VERSION,
            "unsupported_classes": list(self.unsupported),
            "five_class_model_ready": self.run.get("five_class_model_ready"),
            "evaluation_scope": self.run.get("evaluation_scope"),
            "independent_visit_test": self.run.get("independent_visit_test"),
            "device": self.device,
            "runtime": "opencv_dnn_cpu",
            "deployment_mode": "shadow_only",
            "automatic_release_authorized": False,
            "limitations": limitations,
            "status": (
                "SHADOW-ONLY evaluation; no automatic release authority; "
                "unresolved or unsupported evidence requires manual review"
            ),
        }
