"""Supplementary sim-vs-real evidence for the Isaac appearance gate. NOT part of the gate decision (GATE_v2.0.2.md).

    python gate_evidence.py <results_dir> <out_dir> [--labels labels/dev_labels.csv]

results_dir = run_gate.sh output (renders/<variant>.png). Real side: DEV runs 1-3 only, loaded through
real_data.Index (refuses runs 4-5), chosen with the DRAFT labels. Both domains are cut by positions.extract_crops +
crop_for_model (the 64x64 model input); synthetic uses the twin map with zero offset, real the DEV registration offset.
Writes evidence.json and one panel per question:
  A_pale_caps.png           cap saturation/brightness vs the yellow-cap detector, palest caps of each domain
  B_tilt_displacement.png   G03 at 0/10/20/30 deg (+ difference to 0 deg) vs real displaced caps
  C_empty_vs_uncapped.png   empty hole vs uncapped vial (synthetic t00 / liquid vs real)
  D_crop_context.png        same positions in synthetic baseline (all capped), t00 (mostly empty), fullest real frame
"""
import argparse, csv, json, os
import cv2, numpy as np
import positions as P
from real_data import Index

ap = argparse.ArgumentParser()
ap.add_argument("results"); ap.add_argument("out")
ap.add_argument("--labels", default=os.path.join(P.HERE, "labels", "dev_labels.csv"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SYN_MAP, REAL_MAP = P.PositionMap(offset=(0, 0)), P.PositionMap()
idx = Index()


def syn(v):
    im = cv2.imread(os.path.join(a.results, "renders", f"{v}.png"))
    return None if im is None else {c.position: c for c in P.extract_crops(im, SYN_MAP, f"{v}.png")}


SYN = {v: syn(v) for v in ("baseline", "t00", "t10", "t20", "t30", "liquid")}
rows = sorted((r for r in csv.DictReader(open(a.labels, newline="")) if r["usable"] == "1"),
              key=lambda r: (int(r["run"]), int(r["event"])))
_cache, used = {}, set()


def real(frame):
    if frame not in _cache:
        _cache[frame] = {c.position: c for c in P.extract_crops(idx.load(frame), REAL_MAP, frame)}
        used.add(frame)
    return _cache[frame]


def last_per_unit(cls):
    """Latest usable observation of each (run, position) with this label: one physical unit, one crop."""
    out = {(r["run"], r["position"]): r for r in rows if r["ground_truth"] == cls}
    return [(real(r["frame"])[r["position"]], f"r{r['run']} {r['position']} {cls}") for r in out.values()]


def cap_mask(c, grow=0):
    m = np.zeros(c.image.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.round(c.cap_poly).astype(np.int32), 1)
    return cv2.dilate(m, np.ones((2 * grow + 1,) * 2, np.uint8)) if grow else m


def cap_stats(c):
    m = cap_mask(c).astype(bool)
    hsv = cv2.cvtColor(c.image, cv2.COLOR_BGR2HSV)[m]
    yel = P.yellow_mask(c.image)[m] > 0
    return {"S_med": float(np.median(hsv[:, 1])), "V_med": float(np.median(hsv[:, 2])),
            "detector_miss_frac": float(1 - yel.mean()),                      # cap pixels outside the HSV cap rule
            "specular_frac": float(((hsv[:, 2] >= 245) & (hsv[:, 1] < 80)).mean())}


def summary(stats):
    if not stats:
        return None
    k = lambda key: np.array([s[key] for s in stats])
    pct = lambda x, q=(10, 50, 90): [round(float(v), 3) for v in np.percentile(x, q)]
    return {"n": len(stats), "S_med_p10_50_90": pct(k("S_med")), "V_med_p10_50_90": pct(k("V_med")),
            "detector_miss_frac_p10_50_90": pct(k("detector_miss_frac")),
            "caps_mostly_missed_by_detector(miss>0.5)": int((k("detector_miss_frac") > 0.5).sum()),
            "specular_frac_p50_90": pct(k("specular_frac"), (50, 90))}


def center_lab(c, r=5):
    """Median Lab in a small disk at the expected open-vial mouth (empty hole vs uncapped vial)."""
    m = np.zeros(c.image.shape[:2], np.uint8)
    cv2.circle(m, tuple(int(round(v)) for v in c.tube_top_center), r, 1, -1)
    px = cv2.cvtColor(c.image, cv2.COLOR_BGR2LAB)[m.astype(bool)].astype(float)
    l, aa, bb = np.median(px, 0)
    return [round(l * 100 / 255, 1), round(aa - 128, 1), round(bb - 128, 1)]


def neighbour_yellow(c):
    """Share of crop pixels that look like a cap but lie outside this position's own cap outline."""
    own = cap_mask(c, grow=3).astype(bool)
    return float(((P.yellow_mask(c.image) > 0) & ~own).mean())


def tile(x, caption, color=(255, 255, 255)):
    im = P.crop_for_model(x) if isinstance(x, P.PositionCrop) else x
    im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_NEAREST)
    im = cv2.copyMakeBorder(im, 16, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(im, caption[:22], (2, 12), FONT, 0.36, color, 1, cv2.LINE_AA)
    return im


def row(title, tiles, width=8):
    tiles = tiles[:width] + [np.zeros((144, 128, 3), np.uint8)] * (width - len(tiles[:width]))
    head = np.zeros((20, 128 * width, 3), np.uint8)
    cv2.putText(head, title, (4, 14), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, np.hstack(tiles)])


def save(name, rws, note):
    foot = np.zeros((22, rws[0].shape[1], 3), np.uint8)
    cv2.putText(foot, note, (4, 15), FONT, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(a.out, name), np.vstack(rws + [foot]))


ev = {"note": "Supplementary only; not part of the gate decision (GATE_v2.0.2.md). Real labels are DRAFT.",
      "results_dir": os.path.abspath(a.results), "renders_present": [v for v, s in SYN.items() if s]}
NOTE = "SUPPLEMENTARY - not the gate decision. Real side: DEV runs 1-3, DRAFT labels. Tiles = 64x64 model input."

# A. pale caps / highlights
pass_rows = [r for r in rows if r["ground_truth"] == "pass"]
real_pass = [(real(r["frame"])[r["position"]], f"r{r['run']}e{r['event']} {r['position']}") for r in pass_rows]
rs = [cap_stats(c) for c, _ in real_pass]
unit = [(r["run"], r["position"]) for r in pass_rows]
palest = {}                                             # one observation per physical cap: its palest event
for i, u in enumerate(unit):
    if u not in palest or rs[i]["S_med"] < rs[palest[u]]["S_med"]:
        palest[u] = i
syn_caps = [(c, f"syn {p}") for p, c in SYN["baseline"].items()] if SYN["baseline"] else []
ss = [cap_stats(c) for c, _ in syn_caps]
ev["A_pale_caps"] = {"real_seated_cap_observations": summary(rs),
                     "real_units": len(palest),
                     "real_units_ever_mostly_missed(miss>0.5)": sorted(f"r{u[0]} {u[1]}" for u, i in palest.items()
                                                                       if rs[i]["detector_miss_frac"] > 0.5),
                     "synthetic_baseline_caps": summary(ss),
                     "detector_rule_HSV": [P.YELLOW_LO, P.YELLOW_HI],
                     "caveat": "cap pixels = expected seated outline; real registration residual (~3 px) adds rim/background pixels"}
plot = np.full((276, 276, 3), 30, np.uint8)                                 # x = S, y = V (cap medians)
cv2.rectangle(plot, (10 + P.YELLOW_LO[1], 10), (10 + 255, 10 + 255 - P.YELLOW_LO[2]), (90, 90, 90), 1)
for s, col in [(x, (255, 255, 255)) for x in rs] + [(x, (255, 200, 0)) for x in ss]:
    cv2.circle(plot, (10 + int(s["S_med"]), 10 + 255 - int(s["V_med"])), 2, col, -1)
cv2.putText(plot, "S ->  (box = detector rule)", (14, 270), FONT, 0.35, (200, 200, 200), 1)
pale_r = [real_pass[i] for i in sorted(palest.values(), key=lambda i: rs[i]["S_med"])[:8]]
pale_s = [syn_caps[i] for i in np.argsort([s["S_med"] for s in ss])[:8]] if ss else []
A = ev["A_pale_caps"]
legend = np.zeros((276, 1024 - 276, 3), np.uint8)
for k, t in enumerate(["cap median S (x) vs V (y), one dot per observation", "white = real seated caps (DEV)",
                       "blue = synthetic baseline caps",
                       f"real observations mostly missed by detector: {A['real_seated_cap_observations']['caps_mostly_missed_by_detector(miss>0.5)'] if rs else '-'} / {len(rs)}",
                       f"real physical caps ever mostly missed: {len(A['real_units_ever_mostly_missed(miss>0.5)'])} / {A['real_units']}",
                       f"synthetic caps mostly missed: {A['synthetic_baseline_caps']['caps_mostly_missed_by_detector(miss>0.5)'] if ss else '-'} / {len(ss)}"]):
    cv2.putText(legend, t, (10, 30 + 26 * k), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
save("A_pale_caps.png", [np.hstack([plot, legend]),
                         row("palest real seated caps, one per physical cap (lowest S)", [tile(c, t) for c, t in pale_r]),
                         row("8 palest synthetic baseline caps", [tile(c, t) for c, t in pale_s])], NOTE)

# B. tilted / displaced caps
b = {"note": "gate renders G03 tilted about its cap edge only; it renders NO gross displacement (cap off the vial mouth)"}
syn_b, diff_b = [], []
if SYN["t00"]:
    base = P.crop_for_model(SYN["t00"]["G03"]).astype(float)
    for v in ("t00", "t10", "t20", "t30"):
        if not SYN[v]:
            continue
        syn_b.append(tile(SYN[v]["G03"], f"syn G03 {v[1:]} deg"))
        d = np.abs(P.crop_for_model(SYN[v]["G03"]).astype(float) - base).max(2)
        if v != "t00":
            b[v] = {"model_crop_px_changed_over_12": int((d > 12).sum()), "model_crop_max_abs_diff": int(d.max())}
            diff_b.append(tile(cv2.cvtColor(np.clip(d * 4, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), f"|{v}-t00| x4"))
real_b = last_per_unit("crooked_cap")
b["real_displaced_units"] = [t for _, t in real_b]
ev["B_tilt_displacement"] = b
save("B_tilt_displacement.png", [row("synthetic G03 tilt (only tilt renders in the gate)", syn_b + diff_b),
                                 row("real displaced caps (DEV, one tile per unit)", [tile(c, t) for c, t in real_b])], NOTE)

# C. empty hole vs uncapped vial
syn_c = []
for v, p, t in [("t00", "G02", "missing_cap"), ("t00", "G04", "no_vial missed"), ("t00", "G05", "no_vial waiting"),
                ("liquid", "G02", "uncapped+liquid"), ("baseline", "G02", "capped ref")]:
    if SYN[v]:
        syn_c.append((SYN[v][p], f"syn {v} {p} {t}"))
real_c = last_per_unit("missing_cap")
empties = [r for r in rows if r["ground_truth"] == "no_vial" and r["run"] in ("2", "3")]
seen = set()
for r in empties:                                       # two empty holes from each of runs 2 and 3
    if sum(k[0] == r["run"] for k in seen) < 2 and (r["run"], r["position"]) not in seen:
        seen.add((r["run"], r["position"])); real_c.append((real(r["frame"])[r["position"]], f"r{r['run']} {r['position']} empty"))
ev["C_empty_vs_uncapped"] = {"tube_mouth_Lab_median": {t: center_lab(c) for c, t in syn_c + real_c}}
save("C_empty_vs_uncapped.png", [row("synthetic: uncapped vial, empty holes, liquid", [tile(c, t) for c, t in syn_c]),
                                 row("real: uncapped vials, empty holes (DEV)", [tile(c, t) for c, t in real_c])], NOTE)

# D. crop context (neighbouring caps inside the crop)
D_POS = ["G01", "G03", "G07", "G08", "G12", "G13", "G17", "G30"]
full = max(rows, key=lambda r: int(r["K"] or 0))
fr = real(full["frame"])
d = {"positions": D_POS, "real_frame": f"run {full['run']} {full['frame']} (K={full['K']})", "neighbour_yellow_frac": {}}
rws = []
for name, crops in [("synthetic baseline (all 30 capped)", SYN["baseline"]), ("synthetic t00 (G05-G30 empty)", SYN["t00"]),
                    (f"real fullest DEV frame, K={full['K']}", fr)]:
    if not crops:
        continue
    d["neighbour_yellow_frac"][name] = {p: round(neighbour_yellow(crops[p]), 4) for p in D_POS}
    rws.append(row(name, [tile(crops[p], p) for p in D_POS]))
ev["D_crop_context"] = d
save("D_crop_context.png", rws, NOTE)

ev["real_frames_used"] = sorted(used)
ev["real_runs_used"] = sorted({idx.run_of_source(f) for f in used})
json.dump(ev, open(os.path.join(a.out, "evidence.json"), "w"), indent=1)
print(json.dumps({k: ev[k] for k in ("renders_present", "real_runs_used")}), "->", a.out)
