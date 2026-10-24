"""ResNet-18 inspector, the D038 threshold and the G3 run guard. Synthetic data only: no real frame is read here.
Model tests need results/train/resnet18_v1 and the train_v1 renders (gitignored) and are skipped without them."""
import csv, json, os, sys, tempfile, unittest
from unittest import mock
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import positions as P
import train_resnet as T

MODEL_DIR = os.path.join(P.HERE, "results", "train", "resnet18_v1")
DATA = os.path.join(P.HERE, "results", "train_v1", "data_train")
HAVE_MODEL = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in ("model.pt", "run.json", "thresholds.json", "thresholds_d038.json"))
HAVE_RENDERS = os.path.exists(os.path.join(DATA, "crops.csv"))
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def probs(rows):
    return np.array(rows, float)


class D038Threshold(unittest.TestCase):
    def test_midpoint_of_empty_gap(self):
        p = probs([[0.99, 0.0, 0.01, 0.0], [0.0, 0.0, 0.03, 0.97],      # non-defects: s 0.01, 0.03
                   [0.02, 0.9, 0.08, 0.0], [0.05, 0.0, 0.95, 0.0]])     # defects: s 0.98, 0.95
        th = T.gap_threshold(p, np.array([0, 3, 1, 2]), 0.97, {})
        self.assertAlmostEqual(th["t_defect"], (0.03 + 0.95) / 2)
        self.assertAlmostEqual(th["gap"]["width"], 0.92)
        self.assertEqual(th["val_defect_recall"], 1.0)
        self.assertEqual(th["val_false_alarm_rate"], {"pass": 0.0, "no_vial": 0.0})
        self.assertEqual(th["t_defect_preregistered"], 0.97)

    def test_overlap_gives_no_threshold(self):
        p = probs([[0.4, 0.6, 0.0, 0.0], [0.6, 0.0, 0.4, 0.0]])          # non-defect s 0.6 > defect s 0.4
        th = T.gap_threshold(p, np.array([0, 2]), 0.4, {})
        self.assertIsNone(th["t_defect"]); self.assertNotIn("val_defect_recall", th)

    def test_decide_rule(self):
        p = probs([[0.1, 0.5, 0.4, 0.0], [0.1, 0.2, 0.7, 0.0], [0.6, 0.1, 0.0, 0.3], [0.3, 0.1, 0.0, 0.6]])
        self.assertEqual(T.decide(p, 0.5).tolist(), [1, 2, 0, 3])
        self.assertEqual(T.decide(p, 0.95).tolist(), [0, 0, 0, 3])   # below t: the more likely of pass / no_vial

    def test_write_d038_from_val_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "val_predictions.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["scene_id", "position", "label", "variant", "cap_group"] + [f"p_{c}" for c in T.CLASSES])
                w.writerow(["s", "G01", "pass", "seated", "", 0.98, 0.01, 0.01, 0.0])
                w.writerow(["s", "G02", "crooked_cap", "tilt", "", 0.1, 0.0, 0.9, 0.0])
            json.dump({"t_defect": 0.9}, open(os.path.join(d, "thresholds.json"), "w"))
            th = T.write_d038(d)
            self.assertAlmostEqual(th["t_defect"], (0.02 + 0.9) / 2)
            self.assertEqual(json.load(open(os.path.join(d, "thresholds_d038.json")))["t_defect"], th["t_defect"])
            self.assertIn("val_predictions_csv_sha256", th["source"])


@unittest.skipUnless(HAVE_TORCH and HAVE_MODEL, "torch or results/train/resnet18_v1 missing")
class Inspector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from resnet_inspector import ResNetInspector
        cls.d038 = ResNetInspector(MODEL_DIR, "d038", device="cpu")
        cls.pre = ResNetInspector(MODEL_DIR, "preregistered", device="cpu", model=cls.d038.model)

    def test_thresholds_and_provenance(self):
        self.assertAlmostEqual(self.d038.t, json.load(open(os.path.join(MODEL_DIR, "thresholds_d038.json")))["t_defect"])
        self.assertAlmostEqual(self.pre.t, json.load(open(os.path.join(MODEL_DIR, "thresholds.json")))["t_defect"])
        info = self.d038.info()
        self.assertEqual(set(info["files"]), {os.path.relpath(os.path.join(MODEL_DIR, f), P.HERE)
                                              for f in ("model.pt", "thresholds.json", "thresholds_d038.json")})
        self.assertEqual(info["t_defect_preregistered"], self.pre.t)

    def test_rejects_model_that_does_not_match_run_json(self):
        from resnet_inspector import ResNetInspector
        with tempfile.TemporaryDirectory() as d:
            for f in ("run.json", "thresholds.json", "thresholds_d038.json"):
                open(os.path.join(d, f), "w").write(open(os.path.join(MODEL_DIR, f)).read())
            open(os.path.join(d, "model.pt"), "wb").write(b"not the model")
            with self.assertRaises(ValueError):
                ResNetInspector(d, device="cpu")

    def test_observation_fields(self):
        crop = P.PositionCrop("G01", 1, "x", (0, 0, 40, 60), False, (20.0, 20.0), (20.0, 30.0), (20.0, 20.0),
                              np.zeros((0, 2)), np.full((60, 40, 3), 90, np.uint8))
        o = self.d038.inspect(crop)
        self.assertIn(o.visual_state, T.CLASSES)
        self.assertAlmostEqual(sum(o.features[f"p_{c}"] for c in T.CLASSES), 1.0, places=4)
        self.assertAlmostEqual(o.score, o.features["p_missing_cap"] + o.features["p_crooked_cap"], places=5)
        self.assertEqual(o.features["t_defect_preregistered"], self.pre.t)

    @unittest.skipUnless(HAVE_RENDERS, "train_v1 renders missing")
    def test_runtime_path_on_synthetic_validation_renders(self):
        """positions.extract_crops (the runtime crop path) on un-augmented validation renders: labels reproduced, and
        inspect_frame.inspect + prediction_rows accept the inspector (eval_metrics schema, blank rule features)."""
        from inspect_frame import inspect, prediction_rows
        rows = [r for r in csv.DictReader(open(os.path.join(DATA, "crops.csv"))) if r["split"] == "val"]
        lab = {(r["scene_id"], r["position"]): r["label"] for r in rows}
        pm = P.PositionMap(offset=(0, 0)); wrong = n = 0
        for sid in sorted({r["scene_id"] for r in rows})[:4]:
            img = cv2.imread(os.path.join(DATA, "renders", sid + ".png"))
            res = inspect(img, sid, pm, self.d038, plate=pm.plate_mask())
            if res["arm_px"]:
                continue
            for p in res["positions"]:
                n += 1; wrong += p["visual_state"] != lab[(sid, p["position"])]
            pr = prediction_rows("synthetic", 1, res)
            self.assertEqual(len(pr), 30); self.assertEqual(pr[0]["cap_frac"], ""); self.assertEqual(pr[0]["cap_offset"], "")
        self.assertGreater(n, 0); self.assertLessEqual(wrong, 1)


class G3Guard(unittest.TestCase):
    def test_refuses_second_run(self):
        import g3_check
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "g3_report.json"), "w").write("{}")
            with mock.patch.object(sys, "argv", ["g3_check.py", "--out", d]), mock.patch.object(g3_check, "event_jobs") as ev:
                with self.assertRaises(SystemExit):
                    g3_check.main()
                ev.assert_not_called()

    def test_only_dev_runs(self):
        import g3_check
        self.assertEqual(g3_check.RUNS, ("1", "2", "3"))


if __name__ == "__main__":
    unittest.main()
