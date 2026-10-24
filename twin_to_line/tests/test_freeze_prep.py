"""Freeze-prep boundaries: validation-only fitting and explicit, once-only rule G3."""
import csv
import json
from pathlib import Path
import sys
from unittest import mock

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import fit_rule_synth as F
import g3_check as G
import positions as P
import rules


def data(tmp_path):
    root = tmp_path / "data"; (root / "renders").mkdir(parents=True)
    rows = []
    pm = P.PositionMap(offset=(0, 0))
    img = np.full((480, 640, 3), 90, np.uint8)
    for i, center in enumerate(pm.cap_centers):
        label = ("pass", "crooked_cap", "missing_cap", "no_vial")[i % 4]
        if label in ("pass", "crooked_cap"):
            x, y = center
            cv2.circle(img, (int(round(x + (9 if label == "crooked_cap" else 0))), int(round(y))), 10, (0, 220, 230), -1)
        rows.append(dict(scene_id="validation", split="val", position=pm.ids[i], label=label))
    # Training pixels deliberately do not exist. The adapter must never try to open them.
    rows += [dict(r, scene_id="training", split="train") for r in rows]
    with (root / "crops.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cv2.imwrite(str(root / "renders/validation.png"), img)
    return root, img, pm


def test_validation_only_uses_exact_original_features_and_fitter(tmp_path):
    root, img, pm = data(tmp_path)
    original_features = {c.position: rules.features(c) for c in P.extract_crops(img, pm)}
    rows, source = F.validation_features([root], expected_scenes=1)
    assert len(rows) == 30 and source["position_offset_px"] == [0, 0]
    assert set(source["render_sha256"]) == {"validation.png"}
    for row in rows:
        assert {k: row[k] for k in ("cap_frac", "cap_offset")} == original_features[row["position"]]
    output = tmp_path / "thresholds.json"
    artifact = F.fit([root], output, expected_scenes=1)
    assert artifact["thresholds"] == rules.fit(rows)
    assert artifact["fit_on"]["real_images_used"] == 0
    before = output.read_bytes()
    with pytest.raises(FileExistsError): F.fit([root], output, expected_scenes=1)
    assert output.read_bytes() == before


@pytest.mark.parametrize("problem", ["mixed", "duplicate", "missing", "count"])
def test_invalid_validation_split_stops_before_reading_pixels(tmp_path, problem):
    root, _, _ = data(tmp_path)
    path = root / "crops.csv"
    rows = list(csv.DictReader(path.open()))
    if problem == "mixed": rows[0]["split"] = "train"
    if problem == "duplicate": rows.append(dict(rows[0]))
    if problem == "missing": rows.pop(0)
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with mock.patch.object(F.cv2, "imread", side_effect=AssertionError("pixels read before validation")):
        with pytest.raises(ValueError): F.validation_features([root], expected_scenes=113 if problem == "count" else 1)


def test_custom_rule_g3_requires_explicit_output_and_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["g3_check.py", "--rule-thresholds", "synthetic.json"])
    with pytest.raises(SystemExit) as error: G.main()
    assert error.value.code == 2


def test_custom_rule_g3_refuses_existing_report_before_loading_frames(tmp_path, monkeypatch):
    (tmp_path / "g3_report.json").write_text("preserved")
    monkeypatch.setattr(sys, "argv", ["g3_check.py", "--rule-only", "--rule-thresholds", "synthetic.json", "--out", str(tmp_path)])
    with mock.patch.object(G, "event_jobs", side_effect=AssertionError("second G3 attempted")):
        with pytest.raises(SystemExit, match="already holds a G3 report"): G.main()
    assert (tmp_path / "g3_report.json").read_text() == "preserved"


def test_custom_rule_g3_uses_selected_file_and_correct_provenance(tmp_path, monkeypatch):
    threshold = tmp_path / "synthetic.json"
    threshold.write_text(json.dumps({"thresholds": {"t_absent": .08, "t_seated": .22, "t_offset": 4.4},
                                    "fit_on": {"split": "val"}, "note": "synthetic validation only"}))
    out = tmp_path / "g3"
    monkeypatch.setattr(sys, "argv", ["g3_check.py", "--rule-only", "--rule-thresholds", str(threshold), "--out", str(out)])
    monkeypatch.setattr(G, "event_jobs", lambda: ([], {}))
    monkeypatch.setattr(G.R, "Index", lambda: None)
    monkeypatch.setattr(G, "load_csv", lambda p: [])
    metric = dict(defect_recall=1., false_positive_rate=0., meets_g3_criteria=True,
                  counts=dict(defects=1, false_negatives=0, false_positives=0, non_defects=1), escalated_share=0.)
    def predict(jobs, frames, inspector, pm, plate):
        assert inspector.path == str(threshold)
        assert pm.offset == (.9, -1.34)  # G3 uses the real DEV map; only synthetic fitting uses zero offset.
        return [{"synthetic_test": 1}], []
    monkeypatch.setattr(G, "predict", predict)
    monkeypatch.setattr(G, "score", lambda rows, labels: metric)
    G.main()
    report = json.loads((out / "g3_report.json").read_text())
    assert report["g3"]["decided_by"] == "rule_synthetic"
    assert report["arms"]["rule_synthetic"]["inspector"]["status"] == "synthetic validation only"
