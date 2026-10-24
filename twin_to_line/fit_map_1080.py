"""1080 position map: fit the twin's G01-G30 map (position_map_G.json, 640x480 cap_seal) to CSI's 1920x1080 still of
the empty Good fixture, then check it. Writes position_map_G_1080.json; position_map_G.json is only read.

View (decided before the fit, from the still): the 1080 still is the cap_seal view at a higher resolution. The 640x480
frames are the centre 4:3 part of this sensor (1440x1080) scaled by about 1/2.25, so the twin map lands on the still
under a scale + offset. The camera's k1 lens model is unchanged by that, so no new camera pose is solved.
Method, fixed before scoring:
  1. start: nominal_guess = that centre crop (s 2.25, t (240, 0)), from the two image sizes only, never from the
     organizer grid (D024)
  2. holes: register_dev.hole_check's rule (Otsu inside the eroded plate polygon, dark blobs, nearest predicted
     plate-hole centre) with its 640 px sizes times s; all 30 must match
  3. fit: least-squares scale + offset, pure twin plate_hole_center_px (no DEV offset) -> hole centroids, refit once
     after re-detecting with the fitted plate. Residuals are in-sample on the same 30 holes (the 640x480 reference,
     1.68 px median, took its offset from caps instead), so leave-one-out is reported next to them. Similarity, affine
     and homography residuals are reported for transparency only, never used for the map.
  4. map: every point field of position_map_G.json as s*p + t, anchor boxes rounded outwards. The fit already holds
     the real-image offset: load it with PositionMap(path, offset=(0, 0)), never with registration_dev.json.
  5. colour check: each 1080 crop (pad 6*s) resized to the size of its 640x480 crop, geometry scaled with it, then the
     current rule inspector and LIVE_FILL. An empty fixture should read 30 x no_vial -> 30 x WAITING.

    python fit_map_1080.py [still] [--out-map position_map_G_1080.json] [--out-dir results/csi_1080]
"""
import argparse, copy, datetime, hashlib, json, os, sys
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import positions as P
from check_organizer_grid import fit_similarity, apply as apply_similarity, stats
from fill_state import resolve_occupancy
from rules import RuleInspector, CAP_R

STILL = os.path.join(HERE, "results", "csi_1080", "csi_1080_empty_G.jpg")
OUT_MAP = os.path.join(HERE, "position_map_G_1080.json")
OUT_DIR = os.path.join(HERE, "results", "csi_1080")
SIZE = (1920, 1080)
RUN3_EMPTY = "cap_seal_20260923_184630_557477.jpg"      # DEV run 3 empty fixture, register_dev.py's hole check
POINT_FIELDS = ("cap_center_px", "cap_hull_px", "hole_center_px", "plate_hole_center_px", "tube_top_center_px")
ORGANIZER_REPORT = os.path.join(HERE, "results", "organizer_grid", "organizer_grid_report.json")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def nominal_guess(src=(640, 480), dst=SIZE):
    """Centre 4:3 part of the dst frame scaled to src: s = height ratio, x offset = the side margin."""
    s = dst[1] / src[1]
    return s, np.array([(dst[0] - s * src[0]) / 2, 0.0])


def fit_scale_offset(A, B):
    """B ~ s A + t, least squares (one scale, no rotation)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    a, b = A - A.mean(0), B - B.mean(0)
    s = float((a * b).sum() / (a ** 2).sum())
    return s, B.mean(0) - s * A.mean(0)


def hole_centroids(img, plate_poly, predicted, ids, s):
    """register_dev.hole_check with its 640 px sizes x s. Returns {position: dark-blob centroid, full-image px}."""
    k = lambda n: int(round(n * s)) | 1
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inside = np.zeros(g.shape, np.uint8); cv2.fillConvexPoly(inside, np.round(plate_poly).astype(np.int32), 1)
    inside = cv2.erode(inside, np.ones((k(9), k(9)), np.uint8)).astype(bool)
    t, _ = cv2.threshold(g[inside].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((g < t) & inside).astype(np.uint8)
    n, _, st, cents = cv2.connectedComponentsWithStats(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k(3), k(3)), np.uint8)))
    out = {}
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] < 150 * s * s:
            continue
        d = np.linalg.norm(predicted - cents[c], axis=1); j = int(d.argmin())
        if d[j] < 12 * s:
            out[ids[j]] = cents[c]
    return out


def fit_still(img, m640):
    """Steps 1-3 on one image. Raises ValueError unless all 30 holes match."""
    pos = sorted(m640["positions"], key=lambda p: p["index"])
    ids = [p["position"] for p in pos]
    A, plate = np.array([p["plate_hole_center_px"] for p in pos]), np.array(m640["plate_polygon_px"])
    s0, t0 = nominal_guess(tuple(m640["image_size"]), img.shape[1::-1])
    s, t, passes = s0, t0, []
    for _ in range(2):
        holes = hole_centroids(img, s * plate + t, s * A + t, ids, s)
        if len(holes) < 30:
            raise ValueError(f"{len(holes)}/30 holes matched (missing {sorted(set(ids) - set(holes))})")
        B = np.array([holes[i] for i in ids]); passes.append(B)
        s, t = fit_scale_offset(A, B)
    s, t = round(s, 6), np.round(t, 4)            # stored values rebuild the map exactly
    loo = []
    for i in range(30):
        si, ti = fit_scale_offset(np.delete(A, i, 0), np.delete(B, i, 0))
        loo.append(np.linalg.norm(si * A[i] + ti - B[i]))
    return {"ids": ids, "A": A, "B": B, "holes": holes, "scale": s, "offset": t, "guess": (s0, t0),
            "guess_err": np.linalg.norm(s0 * A + t0 - passes[0], axis=1), "err": np.linalg.norm(s * A + t - B, axis=1),
            "loo": np.array(loo), "pass2_moved_px": float(np.abs(passes[1] - passes[0]).max())}


def other_models(A, B):
    """Transparency only: residuals if the map were allowed rotation, shear or perspective."""
    s, R, t, mirror = fit_similarity(A, B)
    Xa = np.c_[A, np.ones(len(A))]; Ma, *_ = np.linalg.lstsq(Xa, B, rcond=None)
    H, _ = cv2.findHomography(A, B, 0)
    return {"similarity": {"scale": round(s, 4), "rotation_deg": round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 2),
                           "mirror": mirror, "residual_px": stats(np.linalg.norm(apply_similarity(s, R, t, A) - B, axis=1))},
            "affine": {"matrix": np.round(Ma.T, 4).tolist(), "residual_px": stats(np.linalg.norm(Xa @ Ma - B, axis=1))},
            "homography": {"residual_px": stats(np.linalg.norm(cv2.perspectiveTransform(A[None], H)[0] - B, axis=1))}}


def scale_map(m, s, t, size=SIZE, meta=None):
    """position_map_G.json -> the same fields for a frame where the twin's 640x480 pixel p lands at s*p + t."""
    f = lambda p: np.round(s * np.asarray(p, float) + t, 2).tolist()
    out = copy.deepcopy(m)
    out["image_size"], out["plate_polygon_px"] = list(size), f(m["plate_polygon_px"])
    for p in out["positions"]:
        for k in POINT_FIELDS:
            if k in p:
                p[k] = f(p[k])
        x0, y0, x1, y1 = p["anchor_bbox_px"]
        (u0, v0), (u1, v1) = s * np.array([x0, y0], float) + t, s * np.array([x1, y1], float) + t
        p["anchor_bbox_px"] = [int(np.floor(u0)), int(np.floor(v0)), int(np.ceil(u1)), int(np.ceil(v1))]
    problems = P.layout_problems(out)
    out["meta"] = {**(meta or {}), "layout_check": problems or "ok: G01-G05 rightmost column top->bottom, columns step leftwards"}
    return out, problems


def resize_crop(c, size):
    """A 1080 crop resized to size (w, h) = its 640x480 crop's size; the crop geometry follows the resize."""
    h, w = c.image.shape[:2]; k = np.array([size[0] / w, size[1] / h])
    f = lambda p: (np.asarray(p, float) + 0.5) * k - 0.5          # cv2.resize keeps pixel centres, not corners
    return P.PositionCrop(c.position, c.index, c.frame, c.box, c.clipped, tuple(f(c.cap_center)), tuple(f(c.hole_center)),
                          tuple(f(c.tube_top_center)), f(c.cap_poly), cv2.resize(c.image, tuple(size), interpolation=cv2.INTER_AREA))


def crop_sizes(pm640, pad=P.CROP_PAD_PX):
    return [(x1 - x0, y1 - y0) for (x0, y0, x1, y1), _ in (pm640.crop_box(i, pad) for i in range(len(pm640.ids)))]


def disk_colour(crop):
    """Colour inside the rule's cap disk (CAP_R px at the expected cap centre): median HSV and yellow px."""
    m = np.zeros(crop.image.shape[:2], np.uint8)
    cv2.circle(m, tuple(int(round(v)) for v in crop.cap_center), CAP_R, 1, -1)
    hsv = cv2.cvtColor(crop.image, cv2.COLOR_BGR2HSV)[m > 0]
    return {"hsv_median": np.median(hsv, 0).astype(int).tolist(), "yellow_px": int((P.yellow_mask(crop.image)[m > 0] > 0).sum()),
            "disk_px": int(m.sum())}


def colour_check(img, pm1080, pm640, s, inspector=None, frame=""):
    """Step 5: rule inspector on the 30 resized crops, then LIVE_FILL. Returns (crops_1080, resized, obs, states)."""
    insp = inspector or RuleInspector()
    crops = P.extract_crops(img, pm1080, frame, pad=int(round(P.CROP_PAD_PX * s)))
    small = [resize_crop(c, sz) for c, sz in zip(crops, crop_sizes(pm640))]
    obs = insp.inspect_batch(small)
    _, states = resolve_occupancy([o.visual_state for o in obs])
    return crops, small, obs, states


def overlay(img, pm, holes, header, path):
    big = P.draw_positions(img, pm, [(p, c, 0) for p, c in holes.items()], scale=2)
    head = np.full((26, big.shape[1], 3), 25, np.uint8)
    cv2.putText(head, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.vstack([head, big]))


def view_check(img, s, t, ref, path):
    """The still resampled to 640x480 by the inverse fit, next to (and blended with) the DEV run-3 empty frame."""
    still = cv2.warpAffine(img, np.array([[1 / s, 0, -t[0] / s], [0, 1 / s, -t[1] / s]]), (640, 480), flags=cv2.INTER_LINEAR)
    panels = [ref, still, cv2.addWeighted(ref, 0.5, still, 0.5, 0)]
    for im, txt in zip(panels, ("DEV run 3 empty, 640x480 (Sept 23)", "CSI 1080 still -> 640x480 by the fit", "50/50 blend")):
        cv2.putText(im, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.hstack(panels))


def main():
    ap = argparse.ArgumentParser(description="fit the twin G01-G30 map to the CSI 1080 empty-fixture still")
    ap.add_argument("still", nargs="?", default=STILL)
    ap.add_argument("--out-map", default=OUT_MAP); ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    if os.path.abspath(a.out_map) == os.path.abspath(P.MAP_PATH):
        raise SystemExit("refusing to write position_map_G.json")
    img = cv2.imread(a.still)
    if img is None or img.shape[1::-1] != SIZE:
        raise SystemExit(f"{a.still}: missing or not {SIZE[0]}x{SIZE[1]}")
    still = {"file": os.path.basename(a.still), "sha256": sha256(a.still), "bytes": os.path.getsize(a.still), "size": list(SIZE)}
    m640 = json.load(open(P.MAP_PATH))
    try:
        r = fit_still(img, m640)
    except ValueError as e:
        raise SystemExit(f"no map written: {e}")
    s, t, e = r["scale"], r["offset"], r["err"]
    fit = {"model": "scale+offset", "scale": s, "offset_px": t.tolist(),
           "on": "30 empty-hole openings: twin plate_hole_center_px (no DEV offset) -> dark-blob centroids in the still",
           "holes_matched": 30, "residual_px": stats(e), "residual_640_equiv_px": stats(e / s), "leave_one_out_px": stats(r["loo"]),
           "nominal_guess": {"scale": r["guess"][0], "offset_px": r["guess"][1].tolist(), "residual_px": stats(r["guess_err"])}}
    meta = {"generated_by": "fit_map_1080.py", "date": datetime.date.today().isoformat(),
            "source_map": os.path.basename(P.MAP_PATH), "source_map_sha256": sha256(P.MAP_PATH),
            **{k: m640["meta"][k] for k in ("twin_dir", "station_sha256", "camera_json_sha256")},
            "image": "1920x1080 cap_seal stream (same view as the 640x480 frames, higher resolution)", "still": still,
            "fit": fit, "offset_applied": "the fit (includes the real-image offset): load with PositionMap(path, offset=(0, 0)); "
                                          "never add registration_dev.json",
            "note": "position_map_G.json scaled by the fit; numbers in results/csi_1080/map_1080_report.json"}
    m1080, problems = scale_map(m640, s, t, SIZE, meta)
    if problems:
        raise SystemExit(f"no map written: layout problems {problems}")
    json.dump(m1080, open(a.out_map, "w"), indent=1)

    os.makedirs(a.out_dir, exist_ok=True)
    pm, pm640, insp = P.PositionMap(a.out_map, offset=(0, 0)), P.PositionMap(), RuleInspector()
    crops, small, obs, states = colour_check(img, pm, pm640, s, insp, still["file"])
    ref = None
    try:                                           # DEV run 3 (D011); a ProtectedDataError is an OSError
        import real_data as R
        ref_img = R.Index().load(RUN3_EMPTY)
        ref_crops = P.extract_crops(ref_img, pm640, RUN3_EMPTY)
        ref = {c.position: (c, o) for c, o in zip(ref_crops, insp.inspect_batch(ref_crops))}
        view_check(img, s, t, ref_img.copy(), os.path.join(a.out_dir, "view_check_still_vs_run3.png"))
    except (OSError, ValueError) as err:
        print(f"run-3 reference skipped: {err}")
    others = other_models(r["A"], r["B"])
    org = None
    if os.path.exists(ORGANIZER_REPORT):
        o = json.load(open(ORGANIZER_REPORT))["profiles"]
        org = {pid: {k: pl["plate_hole_center_px"]["fit"][k] for k in ("scale", "rotation_deg")} for pid, pl in o.items()}
    th = insp.t
    rows = []
    for c, sm, ob, (final, why) in zip(crops, small, obs, states):
        i = c.index - 1
        row = {"position": c.position, "crop_1080": list(c.image.shape[1::-1]), "crop_640": list(sm.image.shape[1::-1]),
               "clipped": c.clipped, "visual_state": ob.visual_state, "final_state": final, **ob.features, **disk_colour(sm),
               "hole_detected_px": np.round(r["B"][i], 2).tolist(), "hole_residual_px": round(float(e[i]), 2),
               "hole_residual_loo_px": round(float(r["loo"][i]), 2)}
        if ref:
            rc, ro = ref[c.position]
            row["run3_640"] = {"visual_state": ro.visual_state, **ro.features, **disk_colour(rc)}
        rows.append(row)
    not_empty = [x["position"] for x in rows if x["visual_state"] != "no_vial" or x["final_state"] != "WAITING"]
    report = {"still": still, "map": os.path.relpath(a.out_map, HERE), "fit": fit, "other_models": others,
              "organizer_view_d024": org, "pass2_moved_px": round(r["pass2_moved_px"], 3),
              "rule_thresholds": th, "rule_disks_px": {"CAP_R": CAP_R}, "crop_pad_1080_px": int(round(P.CROP_PAD_PX * s)),
              "states": {k: sum(x["final_state"] == k for x in rows) for k in ("PASS", "DEFECT", "MISSED", "WAITING", "UNSURE")},
              "not_empty": not_empty, "per_position": rows}
    json.dump(report, open(os.path.join(a.out_dir, "map_1080_report.json"), "w"), indent=1)
    ov = os.path.join(a.out_dir, "overlay_map_1080.png")
    overlay(img, pm, r["holes"], f"1080 map: s {s:.4f} t ({t[0]:.1f}, {t[1]:.1f})  holes 30/30  median {fit['residual_px']['median']} "
                                 f"max {fit['residual_px']['max']} px   green = anchor box, red = cap, blue = hole, yellow + = detected hole", ov)

    print(f"still {still['file']}  sha256 {still['sha256']}  {still['bytes']} bytes  {SIZE[0]}x{SIZE[1]}")
    print(f"nominal guess s {r['guess'][0]} t {r['guess'][1].tolist()}: residual {fit['nominal_guess']['residual_px']}")
    print(f"scale+offset s {s} t {t.tolist()}: 30/30  {fit['residual_px']}  (640-equiv {fit['residual_640_equiv_px']})  "
          f"leave-one-out {fit['leave_one_out_px']}  pass-2 moved {r['pass2_moved_px']:.3f} px")
    for k, v in others.items():
        print(f"  [transparency] {k}: {({kk: vv for kk, vv in v.items() if kk != 'matrix'})}")
    print(f"  organizer view (D024, 4-corner similarity): {org}")
    print(f"map -> {a.out_map}  layout {m1080['meta']['layout_check']}")
    print(f"rule thresholds {th}; CAP_R {CAP_R} px; 1080 crop pad {report['crop_pad_1080_px']} px -> resized to the 640 crop size")
    print(f"{'pos':4s} {'1080 crop':>9s} {'640':>7s} {'cap_frac':>8s} {'offset':>6s} {'H,S,V':>13s} {'yel':>4s} {'visual':>8s} {'state':>8s}"
          f" {'hole px':>7s} {'loo':>5s} | run3 640: {'cap_frac':>8s} visual")
    for x in rows:
        r3 = x.get("run3_640", {})
        print(f"{x['position']:4s} {'x'.join(map(str, x['crop_1080'])):>9s} {'x'.join(map(str, x['crop_640'])):>7s} {x['cap_frac']:8.4f} "
              f"{str(x['cap_offset']):>6s} {','.join(map(str, x['hsv_median'])):>13s} {x['yellow_px']:4d} {x['visual_state']:>8s} "
              f"{x['final_state']:>8s} {x['hole_residual_px']:7.2f} {x['hole_residual_loo_px']:5.2f} | "
              f"{r3.get('cap_frac', float('nan')):17.4f} {r3.get('visual_state', '-')}")
    print(f"states {report['states']}; not reading empty: {not_empty or 'none'}")
    print(f"-> {ov}\n-> {os.path.join(a.out_dir, 'map_1080_report.json')}")


if __name__ == "__main__":
    main()
