"""D033 arm B: run the original rules.fit procedure on train_v1 validation renders only.

    python fit_rule_synth.py --data results/train_v1/data_train results/train_v1/data_tilt

The fitter and features are unchanged from 2e56ab0 (the DEV-fit rule). The input adapter uses
the recorded scene split and the twin camera with zero DEV offset, as cut_train_crops.py does.
No model-input resize, random augmentation, training renders, real images, or threshold retuning.
Existing outputs are never overwritten. G3 is a separate, once-only step after this file is fixed.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import positions as P
import rules

CLASSES = {"pass", "missing_cap", "crooked_cap", "no_vial"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation_features(directories, expected_scenes=113):
    pm = P.PositionMap(offset=(0, 0))
    scenes, manifests = {}, {}
    for directory in map(Path, directories):
        manifest = directory / "crops.csv"
        manifests[directory.name + "/crops.csv"] = sha256(manifest)
        grouped = defaultdict(list)
        with manifest.open(newline="") as f:
            for row in csv.DictReader(f):
                grouped[row["scene_id"]].append(row)
        for sid, rows in grouped.items():
            splits = {r["split"] for r in rows}
            if len(splits) != 1:
                raise ValueError(f"{sid}: mixed scene splits")
            if splits != {"val"}:
                continue
            if sid in scenes or len(rows) != 30 or {r["position"] for r in rows} != set(pm.ids):
                raise ValueError(f"{sid}: expected one validation scene with 30 unique positions")
            if any(r["label"] not in CLASSES for r in rows):
                raise ValueError(f"{sid}: unsupported ground-truth class")
            scenes[sid] = (directory / "renders" / (sid + ".png"), rows)
    if len(scenes) != expected_scenes:
        raise ValueError(f"expected {expected_scenes} validation scenes, found {len(scenes)}")
    features, hashes = [], {}
    for sid, (path, rows) in sorted(scenes.items()):
        hashes[sid + ".png"] = sha256(path)
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"unreadable validation render: {path}")
        crops = {c.position: c for c in P.extract_crops(img, pm, sid)}
        for row in rows:
            features.append({"frame": sid, "position": row["position"], "ground_truth": row["label"],
                             **rules.features(crops[row["position"]])})
    return features, {"split": "val", "n_scenes": len(scenes), "n_crops": len(features),
                      "position_offset_px": list(pm.offset), "crops_csv_sha256": manifests,
                      "render_sha256": hashes, "real_images_used": 0,
                      "augmentation": "none; original feature-fitting procedure on lens-applied renders"}


def fit(directories, out, expected_scenes=113):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to replace {out}")
    rows, source = validation_features(directories, expected_scenes)
    thresholds = rules.fit(rows)                    # original procedure, including its exact grid and tie-breaks
    ranges = {}
    for label in sorted(CLASSES):
        group = [r for r in rows if r["ground_truth"] == label]
        fractions = [r["cap_frac"] for r in group]
        offsets = [r["cap_offset"] for r in group if r["cap_offset"] is not None]
        ranges[label] = {"n_observations": len(group),
                         "n_arrangement_positions": len({(r["frame"], r["position"]) for r in group}),
                         "cap_frac": [min(fractions), max(fractions)] if fractions else None,
                         "cap_offset": [min(offsets), max(offsets)] if offsets else None}
    artifact = {"inspector": rules.RuleInspector.name, "thresholds": thresholds,
                "features": {"cap_frac_disk_px": rules.CAP_R, "cap_offset_disk_px": rules.OFF_R},
                "fit_on": source, "per_class_ranges": ranges,
                "procedure": {"original_commit": "2e56ab0", "call": "rules.fit(validation_features)",
                              "rules_py_sha256": sha256(Path(P.HERE) / "rules.py"),
                              "positions_py_sha256": sha256(Path(P.HERE) / "positions.py"),
                              "position_map_sha256": sha256(P.MAP_PATH),
                              "fit_rule_synth_py_sha256": sha256(__file__)},
                "note": "D033 arm B: synthetic validation only; twin camera, zero DEV offset; no DEV retuning"}
    with out.open("x") as f:
        json.dump(artifact, f, indent=1)
        f.write("\n")
    print(json.dumps({"thresholds": thresholds, "scenes": source["n_scenes"],
                      "classes": dict(Counter(r["ground_truth"] for r in rows))}, indent=1))
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default=str(Path(P.HERE) / "rule_thresholds_synth.json"))
    args = ap.parse_args()
    fit(args.data, args.out)
