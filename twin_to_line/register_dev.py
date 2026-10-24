"""DEV-only registration of the twin position map to real cap_seal frames (registration.py method, reused).

One frame per independent fixture state (classifier-free events from events.py), never the 78 near-duplicate
run-2 stills: the global offset is the median over unique (run, position) caps, so a cap seen in 17 events of
run 1 counts once. Also checks the empty-hole openings on the run-3 empty fixture (Sept 23 camera pose).

    python register_dev.py        -> registration_dev.json + results/prototype/registration/*.png
"""
import csv, datetime, json, os
from collections import defaultdict
import cv2, numpy as np
import positions as P
import real_data as R

OUT = os.path.join(P.HERE, "results", "prototype", "registration")
EV = os.path.join(P.HERE, "results", "prototype", "events")
OUTLIER_PX = 6.0                       # as registration.py: >= 6 px after the offset = candidate displaced cap / lean
OLD_OFFSET = (-0.3, -3.6)              # Oct 13 estimate (12 frames, 8 of them from runs 4-5)


def event_sources():
    """(run, event, source) for every DEV event: run-1 video events, run-2 and run-3 still events."""
    out = []
    for name in ("events_run1_video.csv", "events_run2_stills.csv", "events_run3_stills.csv"):
        for r in csv.DictReader(open(os.path.join(EV, name))):
            out.append((r["run"], int(r["event"]), r["frame"]))
    return out


def load_source(ix, src):
    cached = os.path.join(EV, src + ".png")
    return ix.load(cached if os.path.exists(cached) else src)


def hole_check(img, pmap, min_area=150):
    """Dark hole openings on an empty plate (Otsu inside the plate, as compare_gate.hole_mask) vs the twin."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inside = np.zeros(g.shape, np.uint8); cv2.fillConvexPoly(inside, pmap.plate.astype(np.int32), 1)
    inside = cv2.erode(inside, np.ones((9, 9), np.uint8)).astype(bool)
    t, _ = cv2.threshold(g[inside].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((g < t) & inside).astype(np.uint8)
    n, _, st, cents = cv2.connectedComponentsWithStats(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    res = {}
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < min_area:
            continue
        d = np.linalg.norm(pmap.plate_hole_centers - cents[k], axis=1); j = int(d.argmin())
        if d[j] < 12:
            res[pmap.ids[j]] = cents[k] - pmap.plate_hole_centers[j]
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ix = R.Index(); raw = P.PositionMap(offset=(0, 0))
    srcs = event_sources()
    per_unit, per_frame, images = defaultdict(list), {}, {}
    for run, ev, src in srcs:
        R.check_allowed(run)
        img = load_source(ix, src); images[src] = img
        res = P.residuals(img, raw); per_frame[src] = (run, ev, res)
        for pid, d in res.items():
            per_unit[(run, pid)].append(d)
    unit_med = {k: np.median(np.array(v), 0) for k, v in per_unit.items()}
    first = np.median(np.array(list(unit_med.values())), 0)          # then refit on inliers (registration.py step):
    inl = [v for v in unit_med.values() if np.linalg.norm(v - first) < OUTLIER_PX]   # displaced caps must not drag it
    offset = np.median(np.array(inl), 0)
    pm = P.PositionMap(offset=tuple(offset))
    frames, all_r = [], []
    for src, (run, ev, res) in per_frame.items():
        r = np.array([np.linalg.norm(np.array(d) - offset) for d in res.values()])
        out = sorted(p for p, x in zip(res, r) if x >= OUTLIER_PX)
        all_r += list(r)
        frames.append({"run": run, "event": ev, "frame": src, "caps_matched": len(res),
                       "median_px": round(float(np.median(r)), 2) if len(r) else None,
                       "p90_px": round(float(np.percentile(r, 90)), 2) if len(r) else None,
                       "max_px": round(float(r.max()), 2) if len(r) else None, "outliers_ge6px": out})
    unit_r = {f"run{k[0]}:{k[1]}": round(float(np.linalg.norm(v - offset)), 2) for k, v in sorted(unit_med.items())}
    ur = np.array(list(unit_r.values()))
    empty_src = [s for run, ev, s in srcs if run == "3"][0]
    holes = hole_check(images[empty_src], pm)
    hr = np.array([np.linalg.norm(d) for d in holes.values()])
    rep = {"date": datetime.date.today().isoformat(), "method": "registration.py yellow-blob rule; offset = median over unique (run, position) caps",
           "offset_dx_dy_px": [round(float(x), 2) for x in offset], "previous_offset_dx_dy_px": list(OLD_OFFSET),
           "frames_used": [f["frame"] for f in frames], "runs": sorted({f["run"] for f in frames}),
           "unique_caps": len(unit_med), "inlier_caps_for_offset": len(inl),
           "unique_cap_residual_px": {"median": round(float(np.median(ur)), 2), "p90": round(float(np.percentile(ur, 90)), 2),
                                      "max": round(float(ur.max()), 2), "per_unit": unit_r},
           "all_frame_observations_residual_px": {"n": len(all_r), "median": round(float(np.median(all_r)), 2),
                                                  "p90": round(float(np.percentile(all_r, 90)), 2)},
           "empty_hole_check_run3": {"frame": empty_src, "holes_matched": len(holes), "of": 30,
                                     "median_px": round(float(np.median(hr)), 2) if len(hr) else None,
                                     "p90_px": round(float(np.percentile(hr, 90)), 2) if len(hr) else None,
                                     "mean_dx_dy": [round(float(x), 2) for x in np.mean(np.array(list(holes.values())), 0)] if holes else None},
           "per_frame": frames}
    json.dump(rep, open(os.path.join(P.HERE, "registration_dev.json"), "w"), indent=1)
    for f in frames:
        img = images[f["frame"]]
        vis = P.draw_positions(img, pm, P.cap_blobs(img, pm))
        cv2.imwrite(os.path.join(OUT, f"reg_run{f['run']}_e{f['event']:02d}.png"), vis)
        crops = P.extract_crops(img, pm, f["frame"])
        assert len(crops) == 30 and not any(c.clipped for c in crops), f"{f['frame']}: crop problem"
        cv2.imwrite(os.path.join(OUT, f"crops_run{f['run']}_e{f['event']:02d}.png"), P.contact_sheet(crops))
        print(f"run {f['run']} e{f['event']:02d} {f['frame'][-26:]:>26s}: caps {f['caps_matched']:2d}  median {f['median_px']}  p90 {f['p90_px']}  max {f['max_px']}  outliers {f['outliers_ge6px']}")
    print(f"\noffset {rep['offset_dx_dy_px']} (previous {list(OLD_OFFSET)}) from {len(inl)} inlier of {len(unit_med)} unique caps;"
          f" unique-cap residual median {rep['unique_cap_residual_px']['median']} p90 {rep['unique_cap_residual_px']['p90']} max {rep['unique_cap_residual_px']['max']}")
    print("per unique cap:", unit_r)
    print("run-3 empty holes:", rep["empty_hole_check_run3"])
