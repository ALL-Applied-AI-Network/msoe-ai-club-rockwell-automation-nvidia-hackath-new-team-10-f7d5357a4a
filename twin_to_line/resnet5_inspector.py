"""Five-output staging ResNet behind the existing visual Inspector interface.

This does not replace the frozen synthetic four-class inspector.  A class with
no supporting development examples (currently allowed only for damaged_vial)
is never asserted: a high logit for it returns ``unknown`` so deterministic
manufacturing rules escalate it as UNSURE.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

import positions as P
from rules import Inspector, Observation
from staging_dataset import CANONICAL_CLASSES, DEFECT_CLASSES
from train_resnet import to_tensor_array
from train_resnet5 import decide_one


DEFAULT_MODEL_DIR = os.path.join(P.HERE, "results", "train", "resnet18_staging5")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ResNet5Inspector(Inspector):
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device=None, model=None):
        import torch
        import torchvision

        self.torch = torch
        self.dir = os.path.abspath(model_dir)
        run_path = os.path.join(self.dir, "run.json")
        threshold_path = os.path.join(self.dir, "thresholds.json")
        model_path = os.path.join(self.dir, "model.pt")
        self.run = json.load(open(run_path, encoding="utf-8"))
        if tuple(self.run.get("classes", ())) != CANONICAL_CLASSES:
            raise ValueError(
                f"{model_dir}: classes {self.run.get('classes')} != {CANONICAL_CLASSES}"
            )
        if self.run.get("final_test_used") is not False or self.run.get("test_rows_loaded") != 0:
            raise ValueError(f"{model_dir}: training provenance does not prove test isolation")
        if sha256(model_path) != self.run.get("model_pt_sha256"):
            raise ValueError(f"{model_path}: SHA-256 differs from run.json")
        if sha256(threshold_path) != self.run.get("thresholds_sha256"):
            raise ValueError(f"{threshold_path}: SHA-256 differs from run.json")
        threshold = json.load(open(threshold_path, encoding="utf-8"))
        if threshold.get("threshold_source") != "validation_only":
            raise ValueError(f"{threshold_path}: threshold is not validation-only")
        self.threshold = float(threshold["t_defect"])
        self.unsupported = tuple(self.run.get("unsupported_classes", ()))
        unknown = set(self.unsupported) - set(CANONICAL_CLASSES)
        if unknown:
            raise ValueError(f"{run_path}: unknown unsupported classes {sorted(unknown)}")
        if set(self.unsupported) - {"damaged_vial"}:
            raise ValueError(f"{run_path}: only damaged_vial may have zero development support")

        device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        if model is None:
            model = torchvision.models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(CANONICAL_CLASSES))
            model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            model.eval().to(self.device)
        self.model = model
        self.name = "resnet18_staging5"

    def probabilities(self, crops):
        features = np.stack([to_tensor_array(crop.image) for crop in crops])
        tensor = self.torch.from_numpy(features).to(self.device)
        with self.torch.no_grad():
            return self.torch.softmax(self.model(tensor), dim=1).cpu().numpy()

    def observations(self, probabilities):
        output = []
        for probability in probabilities:
            predicted, score, reason_code = decide_one(
                probability, self.threshold, self.unsupported
            )
            features = {
                f"p_{name}": round(float(value), 8)
                for name, value in zip(CANONICAL_CLASSES, probability)
            }
            features.update(
                {
                    "t_defect": self.threshold,
                    "reason_code": reason_code,
                    "unsupported_classes": list(self.unsupported),
                }
            )
            reason = (
                "unsupported damaged-vial evidence; manual review required"
                if predicted == "unknown"
                else f"defect score {score:.6f}; threshold {self.threshold:.6f}"
            )
            output.append(Observation(predicted, round(score, 8), features, reason))
        return output

    def inspect_batch(self, crops):
        return self.observations(self.probabilities(crops))

    def inspect(self, crop):
        return self.inspect_batch([crop])[0]

    def warmup(self, n=30):
        tensor = self.torch.zeros((n, 3, 128, 128), device=self.device)
        with self.torch.no_grad():
            self.model(tensor).cpu()

    def info(self):
        return {
            "name": self.name,
            "classes": list(CANONICAL_CLASSES),
            "defect_classes": list(DEFECT_CLASSES),
            "threshold": self.threshold,
            "threshold_source": "validation_only",
            "unsupported_classes": list(self.unsupported),
            "five_class_model_ready": self.run.get("five_class_model_ready"),
            "evaluation_scope": self.run.get("evaluation_scope"),
            "independent_visit_test": self.run.get("independent_visit_test"),
            "model_pt_sha256": self.run["model_pt_sha256"],
        }
