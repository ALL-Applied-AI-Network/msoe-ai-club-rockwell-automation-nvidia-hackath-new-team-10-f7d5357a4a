"""Training crops from rendered gen_scenes.py scenes, with the runtime's crop geometry (positions.extract_crops), so a
model sees the same framing on renders and on live frames. Renders carry no DEV registration offset.

    python cut_train_crops.py --renders RENDERS --scenes OUT/scenes --out CROPS [--val-every 5]

Writes CROPS/<scene_id>/<position>.png (the raw crop; the model resizes it with positions.crop_for_model) and
CROPS/crops.csv: one row per crop with its label and scene metadata. The split is by scene, never by crop: scene index
% val_every == val_every - 1 goes to validation (thresholds are chosen there, D033); every crop of a scene shares it.
"""
import argparse, csv, json, os
import cv2
import positions as P

COLS = ["file", "scene_id", "set", "split", "position", "label", "variant", "sequence", "tilt_deg", "tilt_dir_deg",
        "cap_group", "cap_ior", "lean_deg", "offset_mm", "bbox_in_crop", "exposure_ev", "color_temperature_k"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--renders", required=True); ap.add_argument("--scenes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--val-every", type=int, default=5)
    a = ap.parse_args()
    pm = P.PositionMap(offset=(0, 0)); rows, missing = [], []
    os.makedirs(a.out, exist_ok=True)
    for sid in sorted(os.listdir(a.scenes)):
        lab_path = os.path.join(a.scenes, sid, "labels.json")
        if not os.path.isfile(lab_path):
            continue
        render = os.path.join(a.renders, sid + ".png")
        img = cv2.imread(render) if os.path.isfile(render) else None
        if img is None:
            missing.append(sid); continue
        lab = json.load(open(lab_path)); light = (lab.get("lighting") or {}).get("key_overhead") or {}
        split = "val" if lab["index"] % a.val_every == a.val_every - 1 else "train"
        os.makedirs(os.path.join(a.out, sid), exist_ok=True)
        by_pos = {o["position"]: o for o in lab["positions"]}
        for c in P.extract_crops(img, pm, sid):
            o = by_pos[c.position]; f = os.path.join(sid, c.position + ".png")
            cv2.imwrite(os.path.join(a.out, f), c.image)
            rows.append({"file": f, "scene_id": sid, "set": sid.rsplit("_", 1)[0], "split": split, "position": c.position,
                         "label": o["state"], "variant": o["variant"], "sequence": o.get("sequence", ""),
                         "tilt_deg": o.get("cap_tilt_deg", ""), "tilt_dir_deg": o.get("tilt_dir_deg", ""),
                         "cap_group": o.get("cap_group", ""), "cap_ior": o.get("cap_ior", ""), "lean_deg": o.get("lean_deg", ""),
                         "offset_mm": o.get("offset_mm", ""), "bbox_in_crop": o.get("bbox_in_crop", ""),
                         "exposure_ev": light.get("exposure_ev", ""), "color_temperature_k": light.get("color_temperature_k", "")})
    with open(os.path.join(a.out, "crops.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
    n = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    print(f"{len(rows)} crops from {len({r['scene_id'] for r in rows})} renders ({n}); renders missing for {len(missing)} scenes"
          + (f": {missing[:5]}" if missing else ""))


if __name__ == "__main__":
    main()
