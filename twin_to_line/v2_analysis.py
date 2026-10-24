"""v2 preparation: the two G3 failure modes of ResNet-18 v1 on DEV runs 1-3, real crops next to train_v1 synthetic crops
of the same class and position, and what differs. Analysis only: no render, no training, no threshold.

  (a) real empty holes read as uncapped vials (ground truth no_vial, model missing_cap)
  (b) real seated caps read as crooked (ground truth pass, high defect score)

Real: the 20 G3 event frames (results/prototype/events; DEV runs 1-3 only through real_data), cut as G3 and the runtime
cut them (positions.PositionMap with the DEV offset), labels/dev_labels.csv (usable rows), G3 probabilities (results/g3).
Synthetic: results/train_v1 raw crops (data_train + data_tilt) of the same positions and classes, and the same crops
after train_resnet.CameraAug (what the model trained on; fixed seed, no window shift).

Per crop, in the runtime crop geometry:
  hole_*   CIELAB L*, a*, b* in a disk of HOLE_R px at the expected empty-hole opening; hole_Lrel = hole L* / plate L*
           (plate = 90th-percentile L* of the crop)
  mouth_*  the same at the expected open-vial mouth (where an uncapped vial shows its liquid)
  hole_grad, hole_Lstd, hole_Lp90  structure inside the hole (disk HOLE_IN px): mean |gradient| of L*, spread of L*,
           and the lit part of the floor (90th percentile L*)
  cap_*    the yellow blob (registration HSV range) nearest the expected seated-cap centre: L*, a*, b*, area (px),
           centroid offset from the expected centre (px), elongation (major / minor axis), specular share (near-white
           pixels inside the blob's hull); cap_grad / cap_Lstd = structure inside the hull eroded by 2 px (rim, ring,
           highlight)
  sharp    high-frequency ratio std(g - G1(g)) / std(g) on grey; sharp64 the same on the 64-px model input
  noise    std(g - G1.5(g)) on flat plate pixels (8-bit grey)
  chroma   high-frequency ratio of b* divided by that of L* (low = colour edges blurred, e.g. 4:2:0 subsampling)
  block    (full frames only) 8x8 blockiness around the fixture: mean |step| across the 8-px grid / elsewhere
Diagnostic probes (development on DEV, D033; never used for a threshold): v1 scores on real crops after one controlled
change each (brightness, structure, position, sharpness), to rank which measured difference drives each failure.
--twin DIR also prints the twin mesh and material under the G01 hole (read-only, needs usd-core).

    python v2_analysis.py [--out results/v2_analysis] [--no-probes] [--twin DIR]
"""
import argparse, csv, json, os
from collections import defaultdict
import cv2, numpy as np
import positions as P
import real_data as R
import train_resnet as T
from g3_check import event_jobs
from inspect_frame import load as load_event

HERE = P.HERE
TRAIN = [os.path.join(HERE, "results", "train_v1", d) for d in ("data_train", "data_tilt")]
G3 = os.path.join(HERE, "results", "g3", "resnet18_v1")
LABELS = os.path.join(HERE, "labels", "dev_labels.csv")
OUT = os.path.join(HERE, "results", "v2_analysis")
HOLE_R, HOLE_IN, CAP_R, BLOB_R = 6, 8, 11, 16
TOP_ROW = {f"G{i:02d}" for i in range(1, 31, 5)}            # G01 G06 ... G26: the row nearest the robot
SIGMAS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def lab(img):
    x = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    x[..., 0] *= 100 / 255; x[..., 1:] -= 128
    return x


def disk(shape, c, r):
    m = np.zeros(shape[:2], np.uint8); cv2.circle(m, (int(round(c[0])), int(round(c[1]))), r, 1, -1)
    return m.astype(bool)


def hf_ratio(ch, sigma=1.0):
    ch = ch.astype(np.float32)
    return float(np.std(ch - cv2.GaussianBlur(ch, (0, 0), sigma)) / (np.std(ch) + 1e-6))


def cap_blob(img, c):
    """The yellow component overlapping a BLOB_R disk at c with the most pixels there: (centroid, area, elongation, hull mask)."""
    y = (P.yellow_mask(img) > 0).astype(np.uint8)
    n, lab_, st, cents = cv2.connectedComponentsWithStats(y)
    near = disk(y.shape, c, BLOB_R)
    best = max(range(1, n), key=lambda k: int((near & (lab_ == k)).sum()), default=None)
    if best is None or not (near & (lab_ == best)).any() or st[best, cv2.CC_STAT_AREA] < 20:
        return None
    ys, xs = np.nonzero(lab_ == best)
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]))))
    hull = np.zeros(y.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.c_[xs, ys].astype(np.int32)), 1)
    return np.array([xs.mean(), ys.mean()]), int(len(xs)), float(np.sqrt(ev[1] / max(ev[0], 1e-6))), hull.astype(bool)


def grad(ch):
    ch = ch.astype(np.float32)
    return np.hypot(cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)) / 8


def features(img, geo):
    L = lab(img); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); gL = grad(L[..., 0])
    plate = float(np.percentile(L[..., 0], 90)); f = {"plate_L": round(plate, 2)}
    m = disk(g.shape, geo["hole"], HOLE_IN)
    f.update({"hole_grad": float(gL[m].mean()), "hole_Lstd": float(L[..., 0][m].std()), "hole_Lp90": float(np.percentile(L[..., 0][m], 90))})
    for k in ("hole", "mouth"):
        v = L[disk(g.shape, geo[k], HOLE_R)]
        f.update({f"{k}_L": float(v[:, 0].mean()), f"{k}_a": float(v[:, 1].mean()), f"{k}_b": float(v[:, 2].mean()),
                  f"{k}_Lrel": float(v[:, 0].mean() / plate)})
    b = cap_blob(img, geo["cap"])
    if b is not None:
        cen, area, elong, hull = b; inside = L[hull]; yel = L[hull & (P.yellow_mask(img) > 0)]
        f.update({"cap_L": float(yel[:, 0].mean()), "cap_a": float(yel[:, 1].mean()), "cap_b": float(yel[:, 2].mean()),
                  "cap_area": area, "cap_off": float(np.hypot(*(cen - geo["cap"]))), "cap_dx": float(cen[0] - geo["cap"][0]),
                  "cap_dy": float(cen[1] - geo["cap"][1]), "cap_elong": elong,
                  "cap_spec": float(((inside[:, 0] > 92) & (inside[:, 2] < 35)).mean())})
        core = cv2.erode(hull.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if core.sum() > 20:
            f.update({"cap_grad": float(gL[core].mean()), "cap_Lstd": float(L[..., 0][core].std())})
    flat = (L[..., 0] > plate - 8) & (cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2 < 400)
    flat = cv2.erode(flat.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    f["noise"] = float(np.std((g - cv2.GaussianBlur(g, (0, 0), 1.5))[flat])) if flat.sum() > 30 else None
    f["sharp"] = hf_ratio(g); f["sharp64"] = hf_ratio(cv2.cvtColor(T.square_resize(img), cv2.COLOR_BGR2GRAY))
    f["chroma"] = hf_ratio(L[..., 2]) / (hf_ratio(L[..., 0]) + 1e-6)
    return f


def blockiness(g):
    g = g.astype(np.float32); out = []
    for d in (np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0)).T):
        k = np.arange(d.shape[1]) % 8 == 7
        out.append(d[:, k].mean() / d[:, ~k].mean())
    return float(np.mean(out))


def geometry(crop):
    return {"cap": np.array(crop.cap_center), "hole": np.array(crop.hole_center), "mouth": np.array(crop.tube_top_center)}


def real_crops():
    """(key, crop image, geometry, row) for every usable labelled DEV observation, plus full-frame blockiness per run."""
    ix, pm = R.Index(), P.PositionMap()
    lab_ = {(r["frame"], r["position"]): r for r in csv.DictReader(open(LABELS)) if r["usable"] == "1"}
    prob = {(r["frame"], r["position"]): r for r in csv.DictReader(open(os.path.join(G3, "probabilities.csv")))}
    pred = {(r["frame"], r["position"]): r for r in csv.DictReader(open(os.path.join(G3, "predictions_resnet_d038.csv")))}
    out, block, (x0, y0, x1, y1) = [], defaultdict(list), P.fixture_window(pm)
    for run, ev, src in event_jobs()[0]:
        R.check_allowed(run)
        img = load_event(ix, src)
        block[run].append(blockiness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]))
        for c in P.extract_crops(img, pm, src):
            k = (src, c.position)
            if k in lab_:
                out.append((k, c.image, geometry(c), {**lab_[k], **{p: float(prob[k][p]) for p in ("p_pass", "p_missing_cap", "p_crooked_cap", "p_no_vial", "defect_score")},
                                                       "predicted_state": pred[k]["predicted_state"]}))
    return out, {r: float(np.median(v)) for r, v in block.items()}


def synthetic_crops(labels, positions):
    pm0 = P.PositionMap(offset=(0, 0)); geo = {}
    for i, pid in enumerate(pm0.ids):
        (x0, y0, _, _), _ = pm0.crop_box(i); o = np.array([x0, y0], float)
        geo[pid] = {"cap": pm0.cap_centers[i] - o, "hole": pm0.plate_hole_centers[i] - o, "mouth": pm0.tube_top_centers[i] - o}
    out = []
    for d in TRAIN:
        for r in csv.DictReader(open(os.path.join(d, "crops.csv"))):
            if r["label"] in labels and r["position"] in positions:
                out.append((r, os.path.join(d, "crops", r["file"]), geo[r["position"]]))
    return out


def summarize(rows, keys):
    s = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        if len(v):
            s[k] = {"n": int(len(v)), "p10": round(float(np.percentile(v, 10)), 3), "median": round(float(np.median(v)), 3),
                    "p90": round(float(np.percentile(v, 90)), 3)}
    return s


def blur_match(real_rows, syn_imgs, key="sharp"):
    """The Gaussian sigma (px) that brings the synthetic median high-frequency ratio to the real median."""
    target = float(np.median([r[key] for r in real_rows])); res = {}
    for s in SIGMAS:
        v = [hf_ratio(cv2.cvtColor(cv2.GaussianBlur(im, (0, 0), s) if s else im, cv2.COLOR_BGR2GRAY)) for im in syn_imgs]
        res[s] = float(np.median(v))
    best = min(res, key=lambda s: abs(res[s] - target))
    return {"real_median": round(target, 4), "synthetic_median_by_sigma": {str(k): round(v, 4) for k, v in res.items()}, "best_sigma": best}


def sheet(rows, path, title, scale=4):
    """rows: [(label, [(image, caption)])] -> one PNG, one row per position."""
    ch = max(im.shape[0] for _, cells in rows for im, _ in cells) * scale + 16
    cw = max(im.shape[1] for _, cells in rows for im, _ in cells) * scale + 6
    ncol = max(len(c) for _, c in rows); lw = 52
    canvas = np.full((28 + len(rows) * ch, lw + ncol * cw, 3), 30, np.uint8)
    cv2.putText(canvas, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (label, cells) in enumerate(rows):
        y = 28 + i * ch
        cv2.putText(canvas, label, (6, y + ch // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        for j, (im, cap) in enumerate(cells):
            big = cv2.resize(im, (im.shape[1] * scale, im.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
            x = lw + j * cw; canvas[y + 14:y + 14 + big.shape[0], x:x + big.shape[1]] = big
            cv2.putText(canvas, cap, (x, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def lab_stats(imgs):
    """Median over crops of each crop's whole-crop L*a*b* mean and std."""
    v = np.array([[*lab(im).reshape(-1, 3).mean(0), *lab(im).reshape(-1, 3).std(0)] for im in imgs])
    return np.median(v[:, :3], 0), np.median(v[:, 3:], 0)


def transfer(im, target):
    """Whole-crop colour statistics moved to target (mean, std per L*a*b* channel); local structure kept."""
    L = lab(im); m, sd = L.reshape(-1, 3).mean(0), L.reshape(-1, 3).std(0) + 1e-6
    L = (L - m) / sd * target[1] + target[0]; L[..., 0] *= 255 / 100; L[..., 1:] += 128
    return cv2.cvtColor(np.clip(L, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def probes(real_a, real_b, syn_hole_L, out, target_a=None, target_b=None):
    """v1 scores on real crops after one controlled change each (diagnostic, DEV development only)."""
    from resnet_inspector import ResNetInspector
    insp = ResNetInspector(); torch = insp.torch

    def score(imgs):
        x = torch.from_numpy(np.stack([T.to_tensor_array(im) for im in imgs])).to(insp.device)
        with torch.no_grad():
            p = torch.softmax(insp.model(x), 1).cpu().numpy()
        return p
    res = {}
    if real_a:
        imgs = [im for im, _ in real_a]; base = score(imgs)
        def darken(im, geo):                                 # hole interior L* scaled to the synthetic median
            L = lab(im); m = disk(im.shape, geo["hole"], HOLE_R + 2); cur = L[..., 0][disk(im.shape, geo["hole"], HOLE_R)].mean()
            L[..., 0][m] *= syn_hole_L / max(cur, 1e-3); L[..., 1:] += 128; L[..., 0] *= 255 / 100
            return cv2.cvtColor(np.clip(L, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        def smooth_in(im, m, sigma=2.5):                     # structure removed inside m, mean colour kept
            b = cv2.GaussianBlur(im, (0, 0), sigma); out = im.copy(); out[m] = b[m]
            return out
        hole_mask = lambda im, g: disk(im.shape, g["hole"], HOLE_IN)
        for name, fn in (("hole interior darkened to the synthetic median L*", lambda im, g: darken(im, g)),
                         ("hole interior smoothed (structure removed, brightness kept)", lambda im, g: smooth_in(im, hole_mask(im, g))),
                         ("hole interior smoothed and darkened", lambda im, g: darken(smooth_in(im, hole_mask(im, g)), g)),
                         ("whole hole incl. rim and inner wall (r 10) smoothed and darkened",
                          lambda im, g: darken(smooth_in(im, disk(im.shape, g["hole"], 10), 4.0), g)),
                         ("sharpened (unsharp mask, sigma 1, amount 1)", lambda im, g: cv2.addWeighted(im, 2.0, cv2.GaussianBlur(im, (0, 0), 1), -1.0, 0)),
                         ("whole-crop colour statistics moved to synthetic+aug empty", lambda im, g: transfer(im, target_a))):
            p = score([fn(im, g) for im, g in real_a])
            res[f"(a) {name}"] = {"n": len(imgs), "read_missing_cap_before": int((base.argmax(1) == 1).sum()),
                                  "read_missing_cap_after": int((p.argmax(1) == 1).sum()),
                                  "median_p_missing_before": round(float(np.median(base[:, 1])), 3),
                                  "median_p_missing_after": round(float(np.median(p[:, 1])), 3)}
    if real_b:
        imgs = [im for im, _, _ in real_b]; base = score(imgs)
        def recentre(im, geo, off):                           # move the cap blob onto the expected centre
            M = np.float32([[1, 0, -off[0]], [0, 1, -off[1]]])
            return cv2.warpAffine(im, M, im.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
        def smooth_cap(im, g, o):                            # rim / ring / highlight structure removed inside the cap
            b = cap_blob(im, g["cap"])
            if b is None:
                return im
            core = cv2.erode(b[3].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            out = im.copy(); out[core] = cv2.GaussianBlur(im, (0, 0), 2.5)[core]
            return out
        for name, fn in (("cap re-centred on the expected seat", lambda im, g, o: recentre(im, g, o)),
                         ("cap interior smoothed (rim, ring, highlight removed)", lambda im, g, o: smooth_cap(im, g, o)),
                         ("sharpened (unsharp mask, sigma 1, amount 1)", lambda im, g, o: cv2.addWeighted(im, 2.0, cv2.GaussianBlur(im, (0, 0), 1), -1.0, 0)),
                         ("whole-crop colour statistics moved to synthetic+aug seated", lambda im, g, o: transfer(im, target_b))):
            p = score([fn(im, g, o) for im, g, o in real_b])
            res[f"(b) {name}"] = {"n": len(imgs), "median_defect_before": round(float(np.median(base[:, 1] + base[:, 2])), 3),
                                  "median_defect_after": round(float(np.median(p[:, 1] + p[:, 2])), 3),
                                  "read_crooked_before": int((base.argmax(1) == 2).sum()), "read_crooked_after": int((p.argmax(1) == 2).sum())}
    json.dump(res, open(os.path.join(out, "probes.json"), "w"), indent=1)
    return res


def twin_hole_floor(twin):
    """Read-only: the meshes under the G01 hole (tube base to plate top) and their bound material's diffuseColor."""
    from pxr import Usd, UsdGeom, UsdShade
    import twin_positions
    station = os.path.join(twin, "stations", "tray_inspection", "tray_inspection.usdc")
    st, anch = Usd.Stage.Open(station), twin_positions.load(station)
    base = np.array(anch["G01"]["tube_base"]); top = float(np.mean(np.array(anch["tray_good_top"])[:, 2]))
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    print(f"G01: tube base z {base[2]:.4f} m, plate top z {top:.4f} m")
    for p in st.Traverse():
        if not p.IsA(UsdGeom.Mesh):
            continue
        r = bc.ComputeWorldBound(p).ComputeAlignedRange(); lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
        if lo[0] <= base[0] <= hi[0] and lo[1] <= base[1] <= hi[1] and lo[2] <= top + 0.01 and hi[2] >= base[2] - 0.05 and hi[2] - lo[2] < 1:
            mat = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
            col = next((UsdShade.Shader(c).GetInput("diffuseColor").Get() for c in (mat.GetPrim().GetChildren() if mat else [])
                        if UsdShade.Shader(c) and UsdShade.Shader(c).GetInput("diffuseColor")), None)
            print(f"  {p.GetPath()}  z {lo[2]:.4f}..{hi[2]:.4f}  material {mat.GetPath() if mat else None}  diffuse {col}")


def main():
    ap = argparse.ArgumentParser(description="v2 preparation: G3 failure modes, real vs train_v1 synthetic crops")
    ap.add_argument("--out", default=OUT); ap.add_argument("--no-probes", action="store_true"); ap.add_argument("--twin")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.twin:
        twin_hole_floor(a.twin)
    real, block = real_crops()
    rf = []
    for k, im, geo, row in real:
        rf.append({"frame": k[0], "position": k[1], "run": row["run"], "gt": row["ground_truth"], "seq": row["gt_sequence"],
                   "pred": row["predicted_state"], "p_missing": row["p_missing_cap"], "p_crooked": row["p_crooked_cap"],
                   "defect": row["defect_score"], "top": k[1] in TOP_ROW, **features(im, geo)})
    rimg = {(r["frame"], r["position"]): (im, geo) for (k, im, geo, _), r in zip(real, rf)}
    pos_empty = {r["position"] for r in rf if r["gt"] == "no_vial"}; pos_pass = {r["position"] for r in rf if r["gt"] == "pass"}
    syn = synthetic_crops({"no_vial", "missing_cap", "pass", "crooked_cap"}, pos_empty | pos_pass)
    aug = T.CameraAug(json.load(open(os.path.join(HERE, "configs", "recipe_dev.json")))); rng = np.random.default_rng(0)
    sf, simg = [], {}
    for r, path, geo in syn:
        im = cv2.imread(path); ai = aug(im, rng)
        base = {"position": r["position"], "label": r["label"], "variant": r["variant"], "cap_group": r["cap_group"],
                "lean": r["lean_deg"], "offset_mm": r["offset_mm"], "tilt": r["tilt_deg"], "top": r["position"] in TOP_ROW, "file": path}
        sf.append({**base, "src": "synthetic", **features(im, geo)}); sf.append({**base, "src": "synthetic+aug", **features(ai, geo)})
        simg[path] = (im, ai, geo)
    keys_hole = ["hole_L", "hole_a", "hole_b", "hole_Lrel", "hole_grad", "hole_Lstd", "hole_Lp90", "mouth_L", "mouth_a", "mouth_b", "plate_L", "sharp", "sharp64", "noise", "chroma"]
    keys_cap = ["cap_L", "cap_a", "cap_b", "cap_area", "cap_off", "cap_dx", "cap_dy", "cap_elong", "cap_spec", "cap_grad", "cap_Lstd", "sharp", "sharp64", "noise", "chroma"]
    ea = [r for r in rf if r["gt"] == "no_vial"]
    res = {"real_observations": len(rf), "block_real_by_run": block,
           "block_synthetic_renders": float(np.median([blockiness(cv2.cvtColor(cv2.imread(os.path.join(TRAIN[0], "renders", f)), cv2.COLOR_BGR2GRAY))
                                                       for f in sorted(os.listdir(os.path.join(TRAIN[0], "renders")))[:40] if f.endswith(".png")]))}
    groups_a = {"real empty, read missing_cap": [r for r in ea if r["pred"] == "missing_cap"],
                "real empty, read otherwise": [r for r in ea if r["pred"] != "missing_cap"],
                "real empty, top row": [r for r in ea if r["top"]], "real empty, other rows": [r for r in ea if not r["top"]],
                "real uncapped (missing_cap)": [r for r in rf if r["gt"] == "missing_cap"]}
    for src in ("synthetic", "synthetic+aug"):
        for lbl, name in (("no_vial", "empty"), ("missing_cap", "uncapped")):
            s = [r for r in sf if r["src"] == src and r["label"] == lbl and r["position"] in pos_empty]
            groups_a[f"{src} {name}"] = s
            groups_a[f"{src} {name}, top row"] = [r for r in s if r["top"]]
            groups_a[f"{src} {name}, other rows"] = [r for r in s if not r["top"]]
    res["mode_a"] = {k: {"n": len(v), **summarize(v, keys_hole)} for k, v in groups_a.items()}
    res["mode_a"]["read_missing_cap_by_row"] = {"top": f"{sum(r['pred'] == 'missing_cap' for r in ea if r['top'])}/{sum(r['top'] for r in ea)}",
                                                "other": f"{sum(r['pred'] == 'missing_cap' for r in ea if not r['top'])}/{sum(not r['top'] for r in ea)}"}
    pb = [r for r in rf if r["gt"] == "pass"]
    groups_b = {"real pass": pb, "real pass, defect >= 0.5": [r for r in pb if r["defect"] >= 0.5],
                "real pass, defect < 0.5": [r for r in pb if r["defect"] < 0.5], "real crooked": [r for r in rf if r["gt"] == "crooked_cap"]}
    for src in ("synthetic", "synthetic+aug"):
        for lbl, var in (("pass", "seated"), ("pass", "offset"), ("crooked_cap", "tilt")):
            groups_b[f"{src} {lbl} {var}"] = [r for r in sf if r["src"] == src and r["label"] == lbl and r["variant"] == var and r["position"] in pos_pass]
        for grp in ("saturated", "pale"):
            groups_b[f"{src} pass seated, {grp} caps"] = [r for r in groups_b[f"{src} pass seated"] if r["cap_group"] == grp]
    res["mode_b"] = {k: {"n": len(v), **summarize(v, keys_cap)} for k, v in groups_b.items()}
    res["mode_b"]["real_pass_defect_median"] = round(float(np.median([r["defect"] for r in pb])), 3)
    res["mode_b"]["real_pass_read_crooked"] = f"{sum(r['pred'] == 'crooked_cap' for r in pb)}/{len(pb)}"
    res["mode_b"]["real_pass_units"] = len({(r["run"], r["position"]) for r in pb})
    syn_empty = [simg[r["file"]][0] for r in groups_a["synthetic empty"]][:600]
    syn_pass = [simg[r["file"]][0] for r in groups_b["synthetic pass seated"]][:600]
    res["blur_match"] = {"empty holes (crop)": blur_match(ea, syn_empty), "seated caps (crop)": blur_match(pb, syn_pass)}
    json.dump(res, open(os.path.join(a.out, "measurements.json"), "w"), indent=1)
    with open(os.path.join(a.out, "real_features.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, sorted({k for r in rf for k in r}), extrasaction="ignore"); w.writeheader(); w.writerows(rf)

    # contact sheets: per position, real crops (with v1 scores) next to synthetic crops of the same class and position
    def pick(rows, key, q):
        rows = sorted((r for r in rows if r.get(key) is not None), key=lambda r: r[key])
        return rows[min(len(rows) - 1, int(q * len(rows)))] if rows else None
    rows_a = []
    for pid in sorted(pos_empty, key=lambda p: (p not in TOP_ROW, p)):
        rr = [r for r in ea if r["position"] == pid]; mis = [r for r in rr if r["pred"] == "missing_cap"]
        ok = [r for r in rr if r["pred"] != "missing_cap"]
        se = [r for r in sf if r["src"] == "synthetic" and r["label"] == "no_vial" and r["position"] == pid]
        su = [r for r in sf if r["src"] == "synthetic" and r["label"] == "missing_cap" and r["position"] == pid]
        cells = []
        for r, cap in ((pick(mis, "hole_L", 0.5), "real->uncapped"), (pick(ok, "hole_L", 0.5), "real ok")):
            if r:
                cells.append((rimg[(r["frame"], r["position"])][0], f"{cap} L{r['hole_L']:.0f} p{r['p_missing']:.2f}"))
        for rs, cap in ((se, "syn empty"), (su, "syn uncapped")):
            for q in (0.1, 0.9):
                r = pick(rs, "hole_L" if "empty" in cap else "mouth_L", q)
                if r:
                    cells.append((simg[r["file"]][0], f"{cap} L{(r['hole_L'] if 'empty' in cap else r['mouth_L']):.0f}"))
        if cells and len(rows_a) < 14:
            rows_a.append((pid + (" top" if pid in TOP_ROW else ""), cells))
    sheet(rows_a, os.path.join(a.out, "sheet_a_empty_holes.png"),
          "(a) real empty holes (v1 read: uncapped / ok; L* = hole interior, p = P(missing_cap)) vs train_v1 empty and uncapped (p10, p90 L*)")
    rows_b = []
    for pid in sorted(pos_pass):
        rr = [r for r in pb if r["position"] == pid]
        if not rr:
            continue
        cells = []
        for r in (pick(rr, "defect", 0.9), pick(rr, "defect", 0.1)):
            if r:
                cells.append((rimg[(r["frame"], r["position"])][0], f"real d{r['defect']:.2f} off{r.get('cap_off', float('nan')):.1f}"))
        for lbl, var, cap in (("pass", "seated", "syn seated"), ("pass", "offset", "syn offset"), ("crooked_cap", "tilt", "syn crooked")):
            r = pick([s for s in sf if s["src"] == "synthetic" and s["label"] == lbl and s["variant"] == var and s["position"] == pid], "cap_off", 0.5)
            if r:
                cells.append((simg[r["file"]][0], f"{cap} off{r.get('cap_off', float('nan')):.1f}"))
        rows_b.append((pid, cells))
    sheet(rows_b, os.path.join(a.out, "sheet_b_seated_caps.png"),
          "(b) real seated caps (v1 defect score d, cap offset px; highest / lowest d) vs train_v1 seated, offset, crooked (median offset)")
    if not a.no_probes:
        syn_hole_L = res["mode_a"]["synthetic empty"]["hole_L"]["median"]
        pa = [rimg[(r["frame"], r["position"])] for r in ea if r["pred"] == "missing_cap"]
        pbb = [(*rimg[(r["frame"], r["position"])], np.array([r["cap_dx"], r["cap_dy"]])) for r in pb if r["defect"] >= 0.5 and r.get("cap_dx") is not None]
        ta = lab_stats([simg[r["file"]][1] for r in groups_a["synthetic empty"]][:600])
        tb = lab_stats([simg[r["file"]][1] for r in groups_b["synthetic pass seated"]][:600])
        res["probes"] = probes(pa, pbb, syn_hole_L, a.out, ta, tb)
        res["probe_targets"] = {"synthetic+aug empty": [np.round(t, 2).tolist() for t in ta],
                                "synthetic+aug seated": [np.round(t, 2).tolist() for t in tb],
                                "real empty misread": [np.round(t, 2).tolist() for t in lab_stats([im for im, _ in pa])],
                                "real seated, defect >= 0.5": [np.round(t, 2).tolist() for t in lab_stats([im for im, _, _ in pbb])]}
        json.dump(res, open(os.path.join(a.out, "measurements.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:200] + " ...")
    print(f"-> {a.out}/measurements.json, real_features.csv, sheet_a_empty_holes.png, sheet_b_seated_caps.png" + ("" if a.no_probes else ", probes.json"))


if __name__ == "__main__":
    main()
