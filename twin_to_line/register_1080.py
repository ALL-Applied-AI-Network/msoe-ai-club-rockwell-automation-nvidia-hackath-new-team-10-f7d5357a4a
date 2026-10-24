"""Register CSI's 1920x1080 still of the empty Good fixture to the DEV run-3 empty frame (Sept 23, 640x480), image to
image, and write the frozen 1080 -> DEV 640 transform (frame_warp_1080.json) that live_inspect.py applies through
frame_warp.py. The 640 path (position_map_G.json + DEV offset, event gate, rule thresholds, crops) then runs unchanged on
1080 frames. Offline; DEV run 3 only (D011).

Method, fixed before the run:
  model      similarity (one scale, rotation, offset), x640 = M x1080
  criterion  masked intensity alignment with a gain and bias refitted every iteration (the ECC criterion, so the two
             captures' different exposure does not matter); Gauss-Newton with compositional updates. Every iteration
             resamples the still with frame_warp.warp, the runtime's own resampling; both images are smoothed with
             SIGMA px (640 grid) before comparing
  start      the size-only relation (centre 4:3 crop of the 1080 frame / 2.25), so the result does not lean on the twin
  masks      (640 grid, from the run-3 frame) the fixed station frame on the left, the G plate (its twin polygon + DEV
             offset, dilated), the S plate; the run-3 timestamp is excluded. The union is frozen; station-only,
             G-only and S-only fits are reported as agreement checks, OpenCV's findTransformECC (affine) as a
             cross-check
Checks: the 30 still holes (fit_map_1080's detector) mapped by each transform vs run 3's holes; register_dev.hole_check on
the warped still vs the same check on run 3; the rule inspector + LIVE_FILL on the warped still.

    python register_1080.py [still] [--out frame_warp_1080.json] [--force]
Outputs: frame_warp_1080.json (refuses to replace a different one without --force); results/csi_1080/
register_1080_report.json, warped_still_holes.png, warp_check_still_vs_run3.png (gitignored).
"""
import argparse, datetime, json, os
import cv2, numpy as np
import positions as P
import real_data as R
import frame_warp as W
from check_organizer_grid import fit_similarity, stats
from fit_map_1080 import STILL, fit_still, sha256
from inspect_frame import inspect
from register_dev import hole_check
from rules import RuleInspector

OUT_DIR = os.path.join(P.HERE, "results", "csi_1080")
RUN3_EMPTY = "cap_seal_20260923_184630_557477.jpg"
SIZE_IN, SIZE_OUT = (1920, 1080), (640, 480)
SIGMA = 1.0
STATION = [(0, 0, 206, 290), (150, 0, 206, 440)]          # x0, y0, x1, y1: station frame + block column, above the timestamp
S_PLATE = (404, 233, 567, 414)
PLATE_DILATE = 7


def masks(pm):
    m = {k: np.zeros(SIZE_OUT[::-1], np.uint8) for k in ("station", "G plate", "S plate")}
    for x0, y0, x1, y1 in STATION:
        m["station"][y0:y1, x0:x1] = 1
    cv2.fillConvexPoly(m["G plate"], np.round(pm.plate).astype(np.int32), 1)
    m["G plate"] = cv2.dilate(m["G plate"], np.ones((2 * PLATE_DILATE + 1,) * 2, np.uint8))
    x0, y0, x1, y1 = S_PLATE; m["S plate"][y0:y1, x0:x1] = 1
    m = {k: v.astype(bool) for k, v in m.items()}
    m["union"] = m["station"] | m["G plate"] | m["S plate"]
    return m


def similarity(s, deg, t):
    """x_out = s R(deg) x_in + t as a 2x3 matrix."""
    r = np.radians(deg)
    return np.array([[s * np.cos(r), -s * np.sin(r), t[0]], [s * np.sin(r), s * np.cos(r), t[1]]])


def invert(M):
    return np.linalg.inv(W.h3(M))[:2]


def forward_params(M):
    """M: 1080 -> 640. Returns the 640 -> 1080 view (scale, rotation, offset), the convention of fit_map_1080."""
    F = invert(M)
    return {"scale": round(float(np.sqrt(np.linalg.det(F[:, :2]))), 6),
            "rotation_deg": round(float(np.degrees(np.arctan2(F[1, 0], F[0, 0]))), 4), "offset_px": np.round(F[:, 2], 3).tolist()}


def apply(M, pts):
    return np.asarray(pts, float) @ np.asarray(M)[:, :2].T + np.asarray(M)[:, 2]


def smooth(g):
    return cv2.GaussianBlur(g.astype(np.float32), (0, 0), SIGMA)


def register(still, tmpl, mask, M0, iters=200, tol=1e-4):
    """still: 1080 grey; tmpl: 640 grey (DEV grid); mask: bool, 640 grid. Returns M (1080 -> 640) and fit info."""
    T = smooth(tmpl); h, w = T.shape; c = np.array([w / 2, h / 2])
    ys, xs = np.mgrid[0:h, 0:w]; u, v = (xs - c[0]).astype(np.float32), (ys - c[1]).astype(np.float32)
    ones = np.full(still.shape[:2], 255, np.uint8); M = np.asarray(M0, float)
    for it in range(1, iters + 1):
        J = smooth(W.warp(still, M, (w, h)))
        inside = cv2.erode((W.warp(ones, M, (w, h)) == 255).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        m = mask & inside
        gx, gy = cv2.Sobel(J, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(J, cv2.CV_32F, 0, 1, ksize=3) / 8
        Jm, Tm = J[m], T[m]
        g, b = np.polyfit(Jm, Tm, 1)                                          # T ~ g J + b
        A = g * np.stack([gx[m] * u[m] + gy[m] * v[m], gy[m] * u[m] - gx[m] * v[m], gx[m], gy[m]], 1)
        d, *_ = np.linalg.lstsq(A.astype(float), (g * Jm + b - Tm).astype(float), rcond=None)
        U = np.array([[1 + d[0], -d[1], 0.0], [d[1], 1 + d[0], 0.0]]); U[:, 2] = c - U[:, :2] @ c + d[2:]
        M = (W.h3(U) @ W.h3(M))[:2]                                            # x640' = U(M x1080)
        if max(abs(d[0]) * 400, abs(d[1]) * 400, abs(d[2]), abs(d[3])) < tol:     # < tol px at the frame edge
            break
    J = smooth(W.warp(still, M, (w, h)))
    a, bb = (M[0, 0] + M[1, 1]) / 2, (M[1, 0] - M[0, 1]) / 2              # exact similarity form
    M = np.array([[a, -bb, M[0, 2]], [bb, a, M[1, 2]]])
    return M, {"iterations": it, "converged": it < iters, "pixels": int(m.sum()),
               "zncc": round(float(np.corrcoef(J[m], T[m])[0, 1]), 4), "gain_bias": [round(float(g), 4), round(float(b), 2)]}


def ecc_affine(still, tmpl, mask, M):
    """Cross-check: OpenCV findTransformECC, affine, same mask, started from M. Returns the 1080 -> 640 affine."""
    s = np.sqrt(abs(np.linalg.det(invert(M)[:, :2])))
    inp = cv2.GaussianBlur(still.astype(np.float32), (0, 0), SIGMA * s)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-8)
    m_in = cv2.warpAffine(mask.astype(np.uint8), M, still.shape[1::-1], flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    cc, F = cv2.findTransformECC(smooth(tmpl), inp, invert(M).astype(np.float32), cv2.MOTION_AFFINE, crit, m_in, 1)
    return invert(F.astype(float)), round(float(cc), 4)


def run3_holes(img, pm):
    return {p: d + pm.plate_hole_centers[pm.ids.index(p)] for p, d in hole_check(img, pm).items()}


def hole_stats(res):
    r = np.array(list(res.values())); e = np.linalg.norm(r, axis=1)
    return {"holes_matched": len(res), "of": 30, **{k: v for k, v in stats(e).items()},
            "mean_dx_dy": np.round(r.mean(0), 2).tolist()}


def compare_at_holes(cands, frozen, B, H3, ids):
    """Each 1080 -> 640 transform applied to the 30 still holes B: vs the frozen one and vs run 3's holes H3."""
    out, ref = {}, apply(cands[frozen], B)
    h3 = np.array([H3[i] for i in ids])
    for k, M in cands.items():
        p = apply(M, B)
        out[k] = {"vs_frozen_640px": stats(np.linalg.norm(p - ref, axis=1)),
                  "vs_run3_holes_640px": {**stats(np.linalg.norm(p - h3, axis=1)), "mean_dx_dy": np.round((p - h3).mean(0), 2).tolist()}}
    return out


def check_image(ref, warped, path):
    panels = [ref.copy(), warped.copy(), cv2.addWeighted(ref, 0.5, warped, 0.5, 0)]
    for im, txt in zip(panels, ("DEV run 3 empty (Sept 23)", "CSI 1080 still, frame_warp", "50/50 blend")):
        cv2.putText(im, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.hstack(panels))


def main():
    ap = argparse.ArgumentParser(description="register the CSI 1080 still to the DEV 640 grid (run 3 empty frame)")
    ap.add_argument("still", nargs="?", default=STILL)
    ap.add_argument("--out", default=W.WARP_PATH); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    still = cv2.imread(a.still)
    if still is None or still.shape[1::-1] != SIZE_IN:
        raise SystemExit(f"{a.still}: missing or not {SIZE_IN[0]}x{SIZE_IN[1]}")
    ix = R.Index(); ref = ix.load(RUN3_EMPTY)                                     # refuses anything but DEV runs
    pm = P.PositionMap()                                                          # DEV offset, as at runtime
    sg, tg = cv2.cvtColor(still, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    mk = masks(pm)
    s0 = SIZE_IN[1] / SIZE_OUT[1]; M_size = invert(similarity(s0, 0, ((SIZE_IN[0] - s0 * SIZE_OUT[0]) / 2, 0)))
    fits = {k: register(sg, tg, mk[k], M_size) for k in ("union", "station", "G plate", "S plate")}
    M, info = fits["union"]
    M_ecc, cc = ecc_affine(sg, tg, mk["union"], M)

    m640 = json.load(open(P.MAP_PATH)); hf = fit_still(still, m640)                 # 30 still holes + the twin-hole fit
    s_h, R_h, t_h, _ = fit_similarity(hf["A"], hf["B"])
    dev = np.array(pm.offset)
    hole_fit = invert(similarity(hf["scale"], 0, hf["offset"]))
    hole_sim = invert(np.c_[s_h * R_h, t_h])
    cands = {"image, union (frozen)": M, "image, station only": fits["station"][0], "image, G plate only": fits["G plate"][0],
             "image, S plate only": fits["S plate"][0], "ECC affine (OpenCV, cross-check)": M_ecc,
             "twin-hole fit (scale+offset), twin grid": hole_fit,
             "twin-hole fit + DEV offset": np.c_[hole_fit[:, :2], hole_fit[:, 2] + dev],
             "twin-hole similarity (+rotation) + DEV offset": np.c_[hole_sim[:, :2], hole_sim[:, 2] + dev],
             "size only (2.25, (240, 0))": M_size}
    H3 = run3_holes(ref, pm)
    if len(H3) != 30:
        raise SystemExit(f"run 3 hole check found {len(H3)}/30")
    at_holes = compare_at_holes(cands, "image, union (frozen)", hf["B"], H3, pm.ids)

    fw_path = a.out
    doc = {"meta": {"generated_by": "register_1080.py", "date": datetime.date.today().isoformat(),
                    "method": "similarity, masked ECC-criterion Gauss-Newton (gain + bias), resampled with frame_warp.warp; "
                              f"both images smoothed sigma {SIGMA} px (640 grid); start = size-only relation",
                    "still": {"file": os.path.basename(a.still), "sha256": sha256(a.still), "bytes": os.path.getsize(a.still)},
                    "reference": {"frame": RUN3_EMPTY, "run": "3", "sha256": sha256(os.path.join(ix.root, "images", RUN3_EMPTY))},
                    "mask": {"station_xyxy": STATION, "s_plate_xyxy": S_PLATE,
                             "g_plate": f"position_map_G.json plate polygon + DEV offset, dilated {PLATE_DILATE} px"},
                    "fit": info, "forward_640_to_1080": forward_params(M),
                    "note": "applied by frame_warp.py to every 1920x1080 frame; then the 640 path with position_map_G.json "
                            "+ registration_dev.json offset. Numbers: results/csi_1080/register_1080_report.json"},
           "size_in": list(SIZE_IN), "size_out": list(SIZE_OUT),
           "matrix_in_to_out": np.round(M, 10).tolist()}
    text = json.dumps(doc, indent=1)
    if os.path.exists(fw_path) and not a.force:
        old = json.load(open(fw_path))
        if old["matrix_in_to_out"] != doc["matrix_in_to_out"]:
            raise SystemExit(f"{fw_path} exists with a different matrix; pass --force to replace the frozen transform")
    else:
        open(fw_path, "w").write(text)
    fw = W.FrameWarp(fw_path)

    warped = fw.apply(still)
    holes_w, holes_3 = hole_check(warped, pm), hole_check(ref, pm)
    insp = RuleInspector(); res = inspect(warped, os.path.basename(a.still) + " (warped)", pm, insp)
    states = [p["final_state"] for p in res["positions"]]
    os.makedirs(OUT_DIR, exist_ok=True)
    big = P.draw_positions(warped, pm, [(p, d + pm.plate_hole_centers[pm.ids.index(p)], 0) for p, d in holes_w.items()], scale=3)
    head = np.full((26, big.shape[1], 3), 25, np.uint8)
    hs = hole_stats(holes_w)
    cv2.putText(head, f"1080 still warped; map + DEV offset; holes {hs['holes_matched']}/30 median {hs['median']} px "
                f"mean d {hs['mean_dx_dy']}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT_DIR, "warped_still_holes.png"), np.vstack([head, big]))
    check_image(ref, warped, os.path.join(OUT_DIR, "warp_check_still_vs_run3.png"))
    report = {"transform_file": os.path.relpath(fw_path, P.HERE), "transform_sha256": fw.sha256,
              "fits": {k: {**forward_params(Mk), **i} for k, (Mk, i) in fits.items()},
              "ecc_affine": {"cc": cc, "matrix_in_to_out": np.round(M_ecc, 6).tolist()},
              "twin_hole_fit": {"scale": hf["scale"], "offset_px": hf["offset"].tolist(),
                                "similarity": {"scale": round(s_h, 4), "rotation_deg": round(float(np.degrees(np.arctan2(R_h[1, 0], R_h[0, 0]))), 2)}},
              "at_30_holes": at_holes,
              "hole_check": {"warped_still": hole_stats(holes_w), "run3": hole_stats(holes_3),
                             "run3_in_registration_dev_json": json.load(open(P.REG_PATH))["empty_hole_check_run3"]},
              "rule_on_warped_still": {"arm_px": res["arm_px"], "visual": {s: sum(p["visual_state"] == s for p in res["positions"]) for s in {p["visual_state"] for p in res["positions"]}},
                                       "final": {s: states.count(s) for s in set(states)},
                                       "max_cap_frac": max(p["cap_frac"] for p in res["positions"]),
                                       "per_position": {p["position"]: [p["visual_state"], p["final_state"], p["cap_frac"]] for p in res["positions"]}}}
    json.dump(report, open(os.path.join(OUT_DIR, "register_1080_report.json"), "w"), indent=1)

    print(f"transform -> {fw_path}  sha256 {fw.sha256}")
    for k, v in report["fits"].items():
        print(f"  {k:9s} s {v['scale']:.5f}  rot {v['rotation_deg']:+.3f} deg  t {v['offset_px']}  zncc {v['zncc']}  "
              f"px {v['pixels']}  iters {v['iterations']}")
    print(f"  ECC affine cc {cc}; twin-hole fit s {hf['scale']} t {np.round(hf['offset'], 2).tolist()}, similarity "
          f"{report['twin_hole_fit']['similarity']}")
    print("at the 30 still holes, 640 px (median / max):")
    for k, v in at_holes.items():
        a_, b_ = v["vs_frozen_640px"], v["vs_run3_holes_640px"]
        print(f"  {k:46s} vs frozen {a_['median']:5.2f} / {a_['max']:5.2f}   vs run-3 holes {b_['median']:5.2f} / {b_['max']:5.2f}  mean d {b_['mean_dx_dy']}")
    print(f"hole check, warped still: {hole_stats(holes_w)}")
    print(f"hole check, run 3       : {hole_stats(holes_3)}")
    print(f"rule on warped still: arm_px {res['arm_px']}  {report['rule_on_warped_still']['visual']} -> "
          f"{report['rule_on_warped_still']['final']}  max cap_frac {report['rule_on_warped_still']['max_cap_frac']}")
    print(f"-> {OUT_DIR}/register_1080_report.json, warped_still_holes.png, warp_check_still_vs_run3.png")


if __name__ == "__main__":
    main()
