"""Safety and provenance tests for the offline five-output ONNX inspector."""

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import cv2
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import resnet5_onnx_inspector as O
from staging5_decision_policy import POLICY_VERSION
from staging_dataset import CANONICAL_CLASSES
from train_resnet import to_tensor_array


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class FakeNet:
    def __init__(self, logits=(1.0, 0.0, 0.0, 0.0, 0.0)):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.inputs = []
        self.input_names = []
        self.output_names = []
        self.backend = None
        self.target = None

    def setPreferableBackend(self, value):
        self.backend = value

    def setPreferableTarget(self, value):
        self.target = value

    def setInput(self, value, name=""):
        self.inputs.append(np.array(value, copy=True))
        self.input_names.append(name)

    def forward(self, name=""):
        self.output_names.append(name)
        n = len(self.inputs[-1])
        if self.logits.ndim == 1:
            return np.tile(self.logits, (n, 1))
        return self.logits


def make_artifacts(root, *, unsupported=("damaged_vial",), threshold=0.9):
    root = Path(root)
    model = root / O.MODEL_FILENAME
    model.write_bytes(b"synthetic ONNX bytes for unit test")

    thresholds = {
        "decision_policy_version": POLICY_VERSION,
        "threshold_source": "validation_only",
        "t_defect": threshold,
        "unsupported_classes": list(unsupported),
    }
    threshold_path = root / O.THRESHOLD_FILENAME
    write_json(threshold_path, thresholds)

    source_model_sha = "a" * 64
    preprocessing = {
        name: digest(path) if path.exists() else None
        for name, path in O._runtime_preprocessing_paths().items()
    }
    run = {
        "decision_policy_version": POLICY_VERSION,
        "classes": list(CANONICAL_CLASSES),
        "defect_classes": ["missing_cap", "crooked_cap", "damaged_vial"],
        "final_test_used": False,
        "test_rows_loaded": 0,
        "threshold_source": "validation_only",
        "thresholds_sha256": digest(threshold_path),
        "model_pt_sha256": source_model_sha,
        "manifests": {"train": "b" * 64, "validation": "c" * 64},
        "split_lock": {"sha256": "d" * 64},
        "preprocessing_sha256": preprocessing,
        "unsupported_classes": list(unsupported),
        "five_class_model_ready": not unsupported,
        "evaluation_scope": "within_visit_event_holdout_not_independent",
        "independent_visit_test": False,
    }
    run_path = root / O.RUN_FILENAME
    write_json(run_path, run)

    export = {
        "schema_version": 1,
        "artifact_kind": "resnet18_staging5_onnx",
        "classes": list(CANONICAL_CLASSES),
        "input": {
            "name": "input",
            "shape": ["batch", 3, 128, 128],
            "dtype": "float32",
            "preprocessing": "train_resnet.to_tensor_array",
        },
        "output": {
            "name": "logits",
            "shape": ["batch", 5],
            "dtype": "float32",
            "semantics": "raw_logits",
        },
        "onnx": {
            "opset": 17,
            "dynamic_batch": True,
            "model_onnx_sha256": digest(model),
        },
        "source": {
            "model_pt_sha256": source_model_sha,
            "run_json_sha256": digest(run_path),
            "thresholds_sha256": digest(threshold_path),
            "split_lock_sha256": "d" * 64,
            "train_manifest_sha256": "b" * 64,
            "validation_manifest_sha256": "c" * 64,
        },
        "verification": {
            "status": "passed",
            "scope": "train_and_validation_only",
            "test_rows_loaded": 0,
            "protected_test_used": False,
            "rows_total": 510,
            "rows_train": 360,
            "rows_validation": 150,
            "max_abs_logit_error": 1e-6,
            "max_abs_probability_error": 1e-7,
            "logit_atol": 1e-4,
            "probability_atol": 1e-5,
            "decision_mismatches": 0,
        },
        "benchmark": {"batch_size": 30},
        "tools": {
            "python": sys.version.split()[0],
            "torch": "unit-test-only",
            "torchvision": "unit-test-only",
            "onnx": "unit-test-only",
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "platform": sys.platform,
        },
        "readiness": {
            "decision_policy_version": POLICY_VERSION,
            "five_class_model_ready": not unsupported,
            "unsupported_classes": list(unsupported),
            "safety_note": "unsupported visual evidence requires manual review",
            "evaluation_scope": run["evaluation_scope"],
            "independent_visit_test": False,
        },
        "export_code_sha256": "e" * 64,
        "model_onnx_sha256": digest(model),
        "source_model_pt_sha256": source_model_sha,
    }
    export_path = root / O.EXPORT_FILENAME
    write_json(export_path, export)
    return run, thresholds, export


def rewrite_run_and_export(root, run, export):
    run_path = Path(root) / O.RUN_FILENAME
    write_json(run_path, run)
    export["source"]["run_json_sha256"] = digest(run_path)
    write_json(Path(root) / O.EXPORT_FILENAME, export)


class AuthenticatedRuntime(unittest.TestCase):
    def test_info_is_complete_and_explicitly_shadow_only(self):
        with tempfile.TemporaryDirectory() as directory:
            make_artifacts(directory)
            net = FakeNet()
            inspector = O.ResNet5OnnxInspector(directory, net=net)
            info = inspector.info()

            self.assertEqual(inspector.device, "cpu")
            self.assertEqual(inspector.t_defect, 0.9)
            self.assertEqual(info["t_defect"], 0.9)
            self.assertEqual(info["device"], "cpu")
            self.assertEqual(info["deployment_mode"], "shadow_only")
            self.assertFalse(info["automatic_release_authorized"])
            self.assertIn("SHADOW-ONLY", info["status"])
            self.assertFalse(info["five_class_model_ready"])
            self.assertEqual(info["unsupported_classes"], ["damaged_vial"])
            first_path, first_hash = next(iter(info["files"].items()))
            self.assertTrue(first_path.endswith("model.onnx"))
            self.assertEqual(first_hash, digest(Path(directory) / "model.onnx"))
            self.assertTrue(any(path.endswith("train_resnet.py") for path in info["files"]))
            self.assertEqual(net.backend, cv2.dnn.DNN_BACKEND_OPENCV)
            self.assertEqual(net.target, cv2.dnn.DNN_TARGET_CPU)

    def test_probabilities_use_exact_training_preprocessing_and_named_io(self):
        with tempfile.TemporaryDirectory() as directory:
            make_artifacts(directory)
            logits = np.array(
                [[2.0, 1.0, 0.0, -1.0, -2.0], [-2.0, -1.0, 0.0, 1.0, 2.0]],
                dtype=np.float32,
            )
            net = FakeNet(logits)
            inspector = O.ResNet5OnnxInspector(directory, net=net)
            crops = [
                SimpleNamespace(image=np.full((33, 21, 3), 40, dtype=np.uint8)),
                SimpleNamespace(image=np.full((20, 45, 3), 180, dtype=np.uint8)),
            ]

            probabilities = inspector.probabilities(crops)
            expected_inputs = np.stack([to_tensor_array(crop.image) for crop in crops])
            expected_probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            expected_probabilities /= expected_probabilities.sum(axis=1, keepdims=True)

            np.testing.assert_array_equal(net.inputs[-1], expected_inputs)
            np.testing.assert_allclose(probabilities, expected_probabilities, rtol=1e-6)
            self.assertEqual(net.input_names[-1], "input")
            self.assertEqual(net.output_names[-1], "logits")
            self.assertEqual(probabilities.dtype, np.float32)

    def test_warmup_is_a_real_thirty_crop_forward_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            make_artifacts(directory)
            net = FakeNet()
            inspector = O.ResNet5OnnxInspector(directory, net=net)
            inspector.warmup()
            self.assertEqual(net.inputs[-1].shape, (30, 3, 128, 128))
            self.assertEqual(net.inputs[-1].dtype, np.float32)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                inspector.warmup(0)

    def test_wrong_output_shape_and_nonfinite_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            make_artifacts(directory)
            crop = SimpleNamespace(image=np.zeros((10, 10, 3), dtype=np.uint8))
            wrong = O.ResNet5OnnxInspector(directory, net=FakeNet(np.zeros((1, 4))))
            with self.assertRaisesRegex(RuntimeError, "shape"):
                wrong.inspect(crop)
            nonfinite = O.ResNet5OnnxInspector(
                directory, net=FakeNet(np.array([[0, 0, np.nan, 0, 0]], np.float32))
            )
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                nonfinite.inspect(crop)


class ConservativeDecision(unittest.TestCase):
    def test_unsupported_global_argmax_is_unknown_even_below_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            make_artifacts(directory, threshold=0.9)
            inspector = O.ResNet5OnnxInspector(directory, net=FakeNet())
            probabilities = np.array(
                [
                    [0.29, 0.08, 0.07, 0.30, 0.26],  # damage max; defect sum 0.45 < 0.9
                    [0.80, 0.02, 0.01, 0.01, 0.16],  # pass
                    [0.20, 0.02, 0.01, 0.01, 0.76],  # no vial
                    [0.01, 0.70, 0.25, 0.01, 0.03],  # supported defect
                ],
                dtype=np.float32,
            )

            observations = inspector.observations(probabilities)
            self.assertEqual([item.visual_state for item in observations], [
                "unknown", "pass", "no_vial", "missing_cap"
            ])
            unsupported = observations[0]
            self.assertAlmostEqual(unsupported.score, 0.45, places=6)
            self.assertEqual(
                unsupported.features["reason_code"], "unsupported_class_is_global_argmax"
            )
            self.assertEqual(unsupported.features["model_argmax_class"], "damaged_vial")
            self.assertEqual(unsupported.features["deployment_mode"], "shadow_only")
            self.assertIn("automatic pass prohibited", unsupported.reason)
            self.assertIn("not a defect detection", unsupported.reason)

    def test_invalid_probability_array_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            make_artifacts(directory)
            inspector = O.ResNet5OnnxInspector(directory, net=FakeNet())
            with self.assertRaisesRegex(ValueError, "shape"):
                inspector.observations(np.zeros((2, 4), np.float32))
            bad = np.zeros((1, 5), np.float32)
            bad[0, 2] = np.inf
            observation = inspector.observations(bad)[0]
            self.assertEqual(observation.visual_state, "unknown")
            self.assertIsNone(observation.score)
            self.assertEqual(
                observation.features["reason_code"], "non_finite_probability"
            )
            self.assertIn("automatic pass prohibited", observation.reason)


class ProvenanceRefusals(unittest.TestCase):
    def test_tampered_onnx_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            make_artifacts(directory)
            Path(directory, O.MODEL_FILENAME).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                O.ResNet5OnnxInspector(directory, net=FakeNet())

    def test_protected_test_verification_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, export = make_artifacts(directory)
            export["verification"]["protected_test_used"] = True
            write_json(Path(directory) / O.EXPORT_FILENAME, export)
            with self.assertRaisesRegex(ValueError, "protected-test isolation"):
                O.ResNet5OnnxInspector(directory, net=FakeNet())

    def test_decision_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, export = make_artifacts(directory)
            export["verification"]["decision_mismatches"] = 1
            write_json(Path(directory) / O.EXPORT_FILENAME, export)
            with self.assertRaisesRegex(ValueError, "decisions differ"):
                O.ResNet5OnnxInspector(directory, net=FakeNet())

    def test_preprocessing_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, export = make_artifacts(directory)
            run = copy.deepcopy(run)
            run["preprocessing_sha256"]["train_resnet.py"] = "0" * 64
            rewrite_run_and_export(directory, run, export)
            with self.assertRaisesRegex(ValueError, "preprocessing SHA-256 differs"):
                O.ResNet5OnnxInspector(directory, net=FakeNet())

    def test_class_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, export = make_artifacts(directory)
            export["classes"] = list(reversed(CANONICAL_CLASSES))
            write_json(Path(directory) / O.EXPORT_FILENAME, export)
            with self.assertRaisesRegex(ValueError, "exact five-class taxonomy"):
                O.ResNet5OnnxInspector(directory, net=FakeNet())

    def test_runtime_library_version_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, export = make_artifacts(directory)
            export["tools"]["opencv"] = "0.0.0"
            write_json(Path(directory) / O.EXPORT_FILENAME, export)
            with self.assertRaisesRegex(ValueError, "runtime has"):
                O.ResNet5OnnxInspector(directory, net=FakeNet())


if __name__ == "__main__":
    unittest.main()
