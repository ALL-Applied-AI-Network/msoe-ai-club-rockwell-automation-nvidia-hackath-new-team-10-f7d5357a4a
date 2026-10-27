"""Synthetic-only ResNet-18 inspector (train_resnet.py) behind the same Inspector interface as the rule baseline.

Per crop: train_resnet.to_tensor_array (the training input path: square pad -> 64 px -> 128 px, ImageNet mean/std, no
camera augmentation at runtime) -> softmax over pass / missing_cap / crooked_cap / no_vial.
Defect score s = P(missing_cap) + P(crooked_cap). Visual state (train_resnet.decide): s >= t -> the more likely defect
class, else the more likely of pass / no_vial. It never outputs unknown; fill_state.py turns no_vial into MISSED /
WAITING and the arm check in inspect_frame.inspect still overrides every position while the arm is over the fixture.
Threshold t: `d038` (default, thresholds_d038.json: midpoint of the empty synthetic-validation gap) or `preregistered`
(thresholds.json: the 99.5 % rule). Both values are carried in every observation's features so results can report
both (D038).

    python resnet_inspector.py FRAME [--model-dir DIR] [--threshold d038|preregistered] [--device cpu|mps]   (DEV frames only)
"""
import argparse, hashlib, json, os
import numpy as np
import positions as P
from rules import Inspector, Observation
from train_resnet import CLASSES, DEFECTS, INPUT_PX, decide, to_tensor_array

MODEL_DIR = os.path.join(P.HERE, "results", "train", "resnet18_v1")
THRESHOLD_FILES = {"d038": "thresholds_d038.json", "preregistered": "thresholds.json"}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class ResNetInspector(Inspector):
    def __init__(self, model_dir=MODEL_DIR, threshold="d038", device=None, model=None):
        import torch, torchvision
        self.torch, self.dir, self.which = torch, os.path.abspath(model_dir), threshold
        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        run = json.load(open(os.path.join(self.dir, "run.json")))
        if tuple(run["classes"]) != CLASSES:
            raise ValueError(f"{model_dir}: classes {run['classes']} != {CLASSES}")
        self.files = [os.path.join(self.dir, f) for f in ("model.pt", "thresholds.json", "thresholds_d038.json")]
        if sha256(self.files[0]) != run["model_pt_sha256"]:
            raise ValueError(f"{self.files[0]} does not match the sha256 recorded in run.json")
        self.t_d038 = json.load(open(self.files[2]))["t_defect"]
        self.t_pre = json.load(open(self.files[1]))["t_defect"]
        if self.t_d038 is None:
            raise ValueError(f"{self.files[2]}: no empty gap on synthetic validation, D038 gives no threshold")
        self.t = {"d038": self.t_d038, "preregistered": self.t_pre}[threshold]
        self.device = torch.device(device)
        if model is None:                                     # share one loaded model between threshold variants
            model = torchvision.models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
            model.load_state_dict(torch.load(self.files[0], map_location="cpu", weights_only=True))
            model.eval().to(self.device)
        self.model = model
        self.name = f"resnet18_synth_v1[{threshold}]"

    def probabilities(self, crops):
        x = self.torch.from_numpy(np.stack([to_tensor_array(c.image) for c in crops])).to(self.device)
        with self.torch.no_grad():
            return self.torch.softmax(self.model(x), 1).cpu().numpy().astype(np.float64)

    def observations(self, probs):
        out = []
        for p, k in zip(probs, decide(probs, self.t)):
            s = float(p[list(DEFECTS)].sum()); state = CLASSES[k]
            f = {f"p_{c}": round(float(v), 6) for c, v in zip(CLASSES, p)}
            f.update({"t_defect": self.t, "t_defect_preregistered": self.t_pre,
                      "state_at_preregistered": CLASSES[int(decide(p[None], self.t_pre)[0])]})
            why = "" if state == "pass" else (f"defect score {s:.4f} >= t {self.t:.4f}" if k in DEFECTS
                                               else f"defect score {s:.4f} < t {self.t:.4f}; P(no_vial) {p[3]:.3f}")
            out.append(Observation(state, round(s, 6), f, why))
        return out

    def inspect_batch(self, crops):
        return self.observations(self.probabilities(crops))

    def inspect(self, crop):
        return self.inspect_batch([crop])[0]

    def warmup(self, n=30):
        """One throw-away batch of grey crops, so the first scored event does not pay the device start-up (~0.8 s, MPS)."""
        x = self.torch.zeros((n, 3, INPUT_PX, INPUT_PX), device=self.device)
        with self.torch.no_grad():
            self.model(x).cpu()

    def info(self):
        return {"name": self.name, "threshold": self.which, "t_defect": self.t, "t_defect_preregistered": self.t_pre,
                "files": {os.path.relpath(f, P.HERE): sha256(f) for f in self.files},
                "status": "synthetic-only ResNet-18 v1, thresholds from synthetic validation (D033, D038)"}


if __name__ == "__main__":
    import real_data as R
    from inspect_frame import load
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("frame"); ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--threshold", choices=THRESHOLD_FILES, default="d038"); ap.add_argument("--device", help="default: mps if available, else cpu")
    a = ap.parse_args()
    insp = ResNetInspector(a.model_dir, a.threshold, a.device)
    img = load(R.Index(), a.frame); crops = P.extract_crops(img, P.PositionMap(), a.frame)
    for c, o in zip(crops, insp.inspect_batch(crops)):
        print(c.position, f"{o.visual_state:11s} s={o.score:.4f}", " ".join(f"{k}={v}" for k, v in o.features.items() if k.startswith("p_")))
