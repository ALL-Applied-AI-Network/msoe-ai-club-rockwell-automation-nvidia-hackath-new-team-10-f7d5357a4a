"""Score Isaac Sim renders against real cap_seal frames: geometry, appearance, crooked-cap visibility.

    python compare_gate.py <results_dir> <real_frames_dir> <twin_dir> <out_dir>

results_dir holds renders/<variant>.png and variants/<variant>/labels.json (output of run_gate.sh).
Real frames used: DEV runs only (1-3), per DATA_BOUNDARY in HACKATHON_HANDOFF.md. Runs 4-5 stay held out.
  CAPS   run 2 last frame + run 1 last frame (capped vials)   EMPTY  run 3 last frame (empty G fixture)
  NOISE  two consecutive run-3 frames (static scene)           UNCAP  run 2, 19:26:57 (has an uncapped vial)
"""
import json, os, sys
import cv2, numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capsealcam import CapSealCam

RES, REAL, TWIN, OUT = [os.path.abspath(p) for p in sys.argv[1:5]]
os.makedirs(OUT, exist_ok=True)
F = lambda n: cv2.imread(os.path.join(REAL, n))
REAL_FULL = F("cap_seal_20260922_192843_993721.jpg"); REAL_FULL2 = F("cap_seal_20260922_171525_149283.jpg")
REAL_EMPTY = F("cap_seal_20260923_184939_562819.jpg")
REAL_N1, REAL_N2 = F("cap_seal_20260923_184935_548332.jpg"), F("cap_seal_20260923_184937_556995.jpg")
assert all(im is not None for im in (REAL_FULL, REAL_FULL2, REAL_EMPTY, REAL_N1, REAL_N2)), "missing a dev-run real frame"
REAL_UNCAP = F("cap_seal_20260922_192657_385060.jpg")
R = lambda v: cv2.imread(os.path.join(RES, "renders", f"{v}.png"))
L = lambda v: {r["position"]: r for r in json.load(open(os.path.join(RES, "variants", v, "labels.json")))["positions"]}
SYN = {v: R(v) for v in ("baseline", "t00", "t10", "t20", "t30", "liquid")}
missing = [v for v, im in SYN.items() if im is None]
assert SYN["baseline"] is not None, "baseline render missing"
cam = CapSealCam(os.path.join(TWIN, "cameras", "CapSealCam.json"))
st = Usd.Stage.Open(os.path.join(TWIN, "stations", "tray_inspection", "tray_inspection.usdc")); xc = UsdGeom.XformCache()

def world(path):
    prim = st.GetPrimAtPath(path); mesh = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Mesh)][0]
    P = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get()); M = np.array(xc.GetLocalToWorldTransform(mesh))
    return (np.c_[P, np.ones(len(P))] @ M)[:, :3]

PRED = {}
for i in range(1, 31):
    uv = cam.project(world(f"/TrayInspection/G{i:02d}_cap"))[0]
    m = cv2.moments(cv2.convexHull(uv.astype(np.float32))); PRED[f"G{i:02d}"] = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
names = list(PRED); gp = np.array([PRED[n] for n in names])
plate_uv = cam.project(np.array([(np.array([x, y, 0.125, 1]) @ np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath("/TrayInspection/tray_good_top"))))[:3]
                                 for x, y in [(-6.125, -6.125), (6.125, -6.125), (6.125, 6.125), (-6.125, 6.125)]]))[0]
PLATE = np.zeros((480, 640), np.uint8); cv2.fillConvexPoly(PLATE, plate_uv.astype(np.int32), 1)
PLATE_IN = cv2.erode(PLATE, np.ones((9, 9), np.uint8))

def caps(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (18, 110, 120), (38, 255, 255)) & (cv2.dilate(PLATE, np.ones((41, 41), np.uint8)) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        a = stats[k, cv2.CC_STAT_AREA]
        if not 120 < a < 900: continue
        ys, xs = np.nonzero(lab == k); cov = np.cov(np.vstack([xs, ys]))
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        d = np.linalg.norm(gp - cents[k], axis=1); j = d.argmin()
        out.append({"pos": names[j] if d[j] < 15 else None, "c": cents[k], "area": int(a),
                    "elong": float(np.sqrt(ev[0] / ev[1])), "err": float(d[j]), "mask": lab == k})
    return out, m

def lab_med(img, mask):
    px = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[mask.astype(bool)].astype(float)
    if len(px) == 0: return None
    l, a, b = np.median(px, 0); return [round(l * 100 / 255, 1), round(a - 128, 1), round(b - 128, 1)]

def dE(x, y): return None if x is None or y is None else round(float(np.linalg.norm(np.array(x) - np.array(y))), 1)
def hole_mask(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); v = g[PLATE_IN.astype(bool)]
    t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((g < t) & PLATE_IN.astype(bool)), t

rep = {"missing_renders": missing}

# ---------- A. GEOMETRY ----------
sc, _ = caps(SYN["baseline"]); m = [c for c in sc if c["pos"]]
e = np.array([c["err"] for c in m]) if m else np.array([np.nan])
rep["geometry"] = {"synthetic_caps_found": len(sc), "matched_to_twin_positions": len(m),
                   "synthetic_cap_vs_projection_px": {"median": round(float(np.median(e)), 2), "max": round(float(np.max(e)), 2)},
                   "note": "synthetic render vs our projection: tests that render + lens == capsealcam.py"}
rc, _ = caps(REAL_FULL); rm = [c for c in rc if c["pos"]]
rc2, _ = caps(REAL_FULL2); rm2 = [c for c in rc2 if c["pos"]]
off = np.array([c["c"] - PRED[c["pos"]] for c in rm]); bias = np.median(off, 0); res = np.linalg.norm(off - bias, axis=1)
rep["geometry"]["real_cap_vs_projection_px"] = {"bias_dx_dy": [round(float(b), 2) for b in bias], "median_after_bias": round(float(np.median(res)), 2),
                                                 "p90_after_bias": round(float(np.percentile(res, 90)), 2), "n": len(rm)}
if SYN["t00"] is not None:
    hs, ts = hole_mask(SYN["t00"]); hr, tr = hole_mask(REAL_EMPTY)
    rep["geometry"]["empty_hole_pattern_IoU_synthetic_t00_vs_real_empty"] = round(float((hs & hr).sum() / max(1, (hs | hr).sum())), 3)
    sx, sy = np.nonzero(hs.T); rx, ry = np.nonzero(hr.T)
    rep["geometry"]["hole_pattern_centroid_shift_px"] = [round(float(sx.mean() - rx.mean()), 1), round(float(sy.mean() - ry.mean()), 1)]

# ---------- B. APPEARANCE ----------
cap_s = np.any([c["mask"] for c in m], 0) if m else np.zeros((480, 640), bool)
cap_r = np.any([c["mask"] for c in rm], 0)
app = {}
app["cap_color_Lab"] = {"synthetic": lab_med(SYN["baseline"], cap_s), "real": lab_med(REAL_FULL, cap_r)}
app["cap_area_px_median"] = {"synthetic": int(np.median([c["area"] for c in m])) if m else None, "real": int(np.median([c["area"] for c in rm]))}
if SYN["t00"] is not None:
    hs, _ = hole_mask(SYN["t00"]); hr, _ = hole_mask(REAL_EMPTY)
    plate_s = PLATE_IN.astype(bool) & ~cv2.dilate(hs.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    plate_r = PLATE_IN.astype(bool) & ~cv2.dilate(hr.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    app["plate_Lab"] = {"synthetic": lab_med(SYN["t00"], plate_s), "real": lab_med(REAL_EMPTY, plate_r)}
    app["hole_interior_Lab"] = {"synthetic": lab_med(SYN["t00"], hs), "real": lab_med(REAL_EMPTY, hr)}
    g = lambda im, mk: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[mk].astype(float)
    app["plate_luminance_spread_std"] = {"synthetic": round(float(g(SYN["t00"], plate_s).std()), 1), "real": round(float(g(REAL_EMPTY, plate_r).std()), 1)}
WALL = np.zeros((480, 640), bool); WALL[15:95, 400:432] = True          # white panel above the scrap fixture
app["background_wall_Lab"] = {"synthetic": lab_med(SYN["baseline"], WALL), "real": lab_med(REAL_FULL, WALL)}
gs, gr = cv2.cvtColor(SYN["baseline"], cv2.COLOR_BGR2GRAY), cv2.cvtColor(REAL_FULL, cv2.COLOR_BGR2GRAY)
app["frame_luminance"] = {"synthetic_mean": round(float(gs.mean()), 1), "real_mean": round(float(gr.mean()), 1),
                          "synthetic_std": round(float(gs.std()), 1), "real_std": round(float(gr.std()), 1)}
nz = (REAL_N1.astype(float) - REAL_N2.astype(float))[PLATE_IN.astype(bool)]
app["real_frame_noise_sigma"] = round(float(nz.std() / np.sqrt(2)), 2)
if SYN["t00"] is not None and SYN["t10"] is not None:                    # PT noise: G01 unchanged between t00 and t10
    x0, y0, x1, y1 = L("t00")["G01"]["anchor_bbox_px"]
    d = SYN["t10"][y0:y1, x0:x1].astype(float) - SYN["t00"][y0:y1, x0:x1].astype(float)
    app["synthetic_render_noise_sigma"] = round(float(d.std() / np.sqrt(2)), 2)
lap = lambda im: float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F)[PLATE_IN.astype(bool)].var())
app["sharpness_laplacian_var_plate"] = {"synthetic": round(lap(SYN["t00"] if SYN["t00"] is not None else SYN["baseline"])), "real": round(lap(REAL_EMPTY))}
for k in ("cap_color_Lab", "background_wall_Lab", "plate_Lab", "hole_interior_Lab"):
    if k in app: app[k]["deltaE76"] = dE(app[k]["synthetic"], app[k]["real"])
rep["appearance"] = app

# ---------- C. CROOKED-CAP VISIBILITY (G03, same position, only the tilt differs) ----------
off2 = np.array([c["c"] - PRED[c["pos"]] for c in rm2]); res2 = np.linalg.norm(off2 - np.median(off2, 0), axis=1)
pool = rm + rm2; res = np.concatenate([res, res2])          # natural variation pooled over both dev frames
seated = [c for c, r_ in zip(pool, res) if r_ < 6]
nat = {"centroid_resid_px_std": round(float(np.std(res[res < 6])), 2), "area_std": round(float(np.std([c["area"] for c in seated])), 1),
       "elong_std": round(float(np.std([c["elong"] for c in seated])), 3), "elong_median": round(float(np.median([c["elong"] for c in seated])), 3)}
tilt = {"real_seated_cap_variation": nat, "by_tilt": {}}
base = SYN["t00"]
if base is not None:
    b0, _ = caps(base); c0 = [c for c in b0 if c["pos"] == "G03"]
    x0, y0, x1, y1 = L("t30")["G03"]["bbox_px"] if "bbox_px" in L("t30")["G03"] else L("t00")["G03"]["anchor_bbox_px"]
    x0, y0, x1, y1 = x0 - 6, y0 - 6, x1 + 6, y1 + 6
    for v in ("t10", "t20", "t30"):
        if SYN[v] is None: continue
        d = np.abs(SYN[v][y0:y1, x0:x1].astype(float) - base[y0:y1, x0:x1].astype(float)).max(2)
        bv, _ = caps(SYN[v]); cv_ = [c for c in bv if c["pos"] == "G03"]
        rec = {"pixels_changed_over_real_noise_p99(12)": int((d > 12).sum()), "pixels_changed_over_30": int((d > 30).sum()),
               "max_abs_diff": int(d.max())}
        if c0 and cv_:
            dc = float(np.linalg.norm(cv_[0]["c"] - c0[0]["c"])); da = cv_[0]["area"] - c0[0]["area"]; de = cv_[0]["elong"] - c0[0]["elong"]
            rec.update({"cap_centroid_shift_px": round(dc, 2), "cap_area_change_px": int(da), "cap_elongation_change": round(de, 3),
                        "centroid_shift_vs_real_seated_spread": round(dc / max(nat["centroid_resid_px_std"], 1e-6), 2),
                        "elongation_change_vs_real_seated_spread": round(abs(de) / max(nat["elong_std"], 1e-6), 2)})
        tilt["by_tilt"][v] = rec
rep["crooked_cap_visibility"] = tilt
json.dump(rep, open(os.path.join(OUT, "gate_report.json"), "w"), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))

# ---------- D. PANELS ----------
def lbl(im, t):
    im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(im, t, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1); return im
crop = lambda im: im[40:400, 150:470]
row1 = np.hstack([lbl(crop(SYN["baseline"]), "synthetic baseline (as shipped)"), lbl(crop(REAL_FULL), "real run 2, last frame")])
row2 = np.hstack([lbl(crop(SYN["t00"]), "synthetic t00 (4 states)") if SYN["t00"] is not None else np.zeros_like(crop(REAL_FULL)),
                  lbl(crop(REAL_EMPTY), "real run 3, empty G fixture")])
cv2.imwrite(os.path.join(OUT, "panel_full_frames.png"), np.vstack([row1, row2]))
tiles = []
def tile(im, box, t, s=4):
    x0, y0, x1, y1 = box; c = im[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8]
    c = cv2.resize(c, (44 * s, 44 * s), interpolation=cv2.INTER_NEAREST); return lbl(c, t)
A = lambda v, p: L(v)[p]["anchor_bbox_px"]
syn_row = [tile(SYN["t00"], A("t00", p), f"syn {p} {L('t00')[p]['state']}") for p in ("G01", "G02", "G04")] if SYN["t00"] is not None else []
syn_row += [tile(SYN[v], A(v, "G03"), f"syn G03 {v[1:]} deg") for v in ("t00", "t10", "t20", "t30") if SYN[v] is not None]
real_row = [tile(REAL_FULL, A("t00", p), f"real {p} run2") for p in ("G01", "G02", "G03", "G04", "G05", "G06", "G07")]
liq_row = [tile(SYN["liquid"], A("liquid", p), f"syn liquid {p}") for p in ("G01", "G02", "G03")] if SYN["liquid"] is not None else []
liq_row += [tile(REAL_UNCAP, A("t00", p), f"real run2 {p}") for p in ("G03", "G05", "G07", "G08")]
W = max(len(syn_row), len(real_row), len(liq_row))
pad = lambda r: np.hstack(r + [np.zeros_like(r[0])] * (W - len(r)))
cv2.imwrite(os.path.join(OUT, "panel_crops.png"), np.vstack([pad(r) for r in (syn_row, real_row, liq_row) if r]))
print("panels ->", OUT)
