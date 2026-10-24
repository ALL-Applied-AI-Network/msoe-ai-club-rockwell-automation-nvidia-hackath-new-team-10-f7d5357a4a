"""Pre-flight check of the training recipe on the first rendered scenes, before the full batch: does the renderer
respond to exposure, colour temperature and per-cap diffuse colour + ior the way the render model behind
configs/recipe_dev.json predicts (calibrate_recipe.py, validated on gate run 2)?

    python check_render_recipe.py --renders RENDERS --scenes OUT/scenes [--ids id,id,...] [--model-dir results/isaac_gate/run2]

Compared per scene: the white table panel (L*, b*), and every seated, upright cap at its seat (7 px disk). Crooked,
offset or leaning caps and scene-to-scene light rotation are not modelled, so they are left out.
Pass rule (fixed Oct 23, before any training render exists):
  wall   median |L* measured - predicted| <= 6 and median |b* measured - predicted| <= 6
  caps   median dE76 measured vs predicted <= 8
  groups median b* of saturated caps - median b* of pale caps >= 10 (the pale group renders visibly paler)
Exit code 0 = pass, 1 = fail (do not start the full batch).
"""
import argparse, json, os, sys
import cv2, numpy as np
import calibrate_recipe as C


def predict_scene(m, lab):
    """Predicted wall Lab and per-position cap Lab for one scene's recipe values."""
    L = lab["lighting"]; ev = L["key_overhead"]["exposure_ev"]; T = L["key_overhead"]["color_temperature_k"]
    gain = np.sqrt(L["key_overhead"]["jitter"] * L["fill_side"]["jitter"])          # light rotation not modelled
    S = m.Sb.copy(); ys, xs = np.nonzero(m.cap_px)
    centers = m.pm.cap_centers
    owner = np.argmin(((xs[:, None] - centers[None, :, 0]) ** 2 + (ys[:, None] - centers[None, :, 1]) ** 2), 1)
    by_pos = {o["position"]: o for o in lab["positions"]}
    for i, pid in enumerate(m.pm.ids):
        o = by_pos[pid]; sel = owner == i
        if "cap_rgb" not in o:
            continue
        k = ((o["cap_ior"] - 1) / (o["cap_ior"] + 1)) ** 2 / 0.04
        S[ys[sel], xs[sel]] = m.D[sel, None] * (np.array(o["cap_rgb"]) * C.LIGHT_RGB) + k * m.Sp[sel, None] * C.LIGHT_RGB
    img = m.image(S, ev + np.log2(gain), T)
    return C.lab_of(img, m.wall), {pid: C.lab_of(img, d) for pid, d in zip(m.pm.ids, m.cap_disks)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--renders", required=True); ap.add_argument("--scenes", required=True); ap.add_argument("--ids")
    ap.add_argument("--model-dir", default=os.path.join(C.P.HERE, "results", "isaac_gate", "run2"))
    a = ap.parse_args()
    m = C.RenderModel(a.model_dir)
    ids = a.ids.split(",") if a.ids else sorted(f[:-4] for f in os.listdir(a.renders) if f.endswith(".png") and "_" in f
                                                and not f.endswith("_pinhole768.png"))
    wall_dL, wall_db, cap_dE, b_sat, b_pale = [], [], [], [], []
    for sid in ids:
        img = cv2.imread(os.path.join(a.renders, sid + ".png")); lab = json.load(open(os.path.join(a.scenes, sid, "labels.json")))
        pw, pcaps = predict_scene(m, lab); mw = C.lab_of(img, m.wall)
        wall_dL.append(abs(mw[0] - pw[0])); wall_db.append(abs(mw[2] - pw[2]))
        n = 0
        for o in lab["positions"]:
            if o["state"] != "pass" or o["variant"] != "seated" or "lean_deg" in o or "cap_rgb" not in o:
                continue
            mc = C.lab_of(img, m.cap_disks[m.pm.ids.index(o["position"])]); n += 1
            cap_dE.append(float(np.linalg.norm(mc - pcaps[o["position"]])))
            (b_pale if o["cap_group"] == "pale" else b_sat).append(mc[2])
        print(f"{sid}: wall measured {np.round(mw, 1)} predicted {np.round(pw, 1)}; {n} caps compared")
    res = {"scenes": len(ids), "wall_median_abs_dL": round(float(np.median(wall_dL)), 2),
           "wall_median_abs_db": round(float(np.median(wall_db)), 2), "caps_compared": len(cap_dE),
           "cap_median_dE": round(float(np.median(cap_dE)), 2) if cap_dE else None,
           "cap_b_saturated_median": round(float(np.median(b_sat)), 1) if b_sat else None,
           "cap_b_pale_median": round(float(np.median(b_pale)), 1) if b_pale else None}
    ok = {"wall": res["wall_median_abs_dL"] <= 6 and res["wall_median_abs_db"] <= 6,
          "caps": res["cap_median_dE"] is not None and res["cap_median_dE"] <= 8,
          "groups": bool(b_sat and b_pale) and res["cap_b_saturated_median"] - res["cap_b_pale_median"] >= 10}
    res["checks"] = ok; res["pass"] = all(ok.values())
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
