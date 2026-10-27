"""Organizer-grid consistency check: twin G01-G30 layout vs the NVIDIA vial labeler's position grid (Oct 14).

Read-only (D024): position_map_G.json is read, never written or tuned; the twin map stays the runtime /
commissioning source. NOT an independent 30-point physical validation: the organizer grid is 4 hand-placed corner
centres plus bilinear interpolation, and the twin map is aligned to those same 4 corners. The result shows whether the
twin's topology, spacing, orientation and G01-G30 order are consistent with the organizers' grid convention. Physical
accuracy evidence stays the real 640x480 empty-fixture hole check (30/30, ~1.68 px median).

The organizer grid belongs to a different camera view (1920x1080 fixed view, not the 640x480 cap_seal frames the
twin map is projected for), so pixels cannot be compared directly. Method, fixed before scoring:
  1. official centres = the labeler's own bilinear interpolation of 4 hand-placed tray-corner centres
     (G01 top-right, G05 bottom-right, G26 top-left, G30 bottom-left). Only these 4 are independent measurements;
     the other 26 are interpolated from them, so interior agreement tests layout shape, not 26 extra measurements.
  2. a 4-DOF similarity (scale, rotation, translation) is fitted on the 4 corners only, twin -> organizer grid.
  3. residuals are reported for all 30 positions, in official-view px and divided by the fitted scale (~ cap_seal px).
Primary plane: plate_hole_center_px (the labeler's circles mark the target insertion hole); the other twin planes
are reported for transparency. Numbering check: the fit must need no mirror and |rotation| < 45 deg, otherwise the
G01-G30 conventions differ (a 180-degree or mirrored numbering fits the same shape).
OUTLIER_PX is in official-view px (pitch is ~66 px between columns).

    python check_organizer_grid.py <labeler>/calibration.json [--example <labeler>/examples/tray_000001.jpg] [--out DIR]
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.path.join(HERE, "position_map_G.json")
OUT = os.path.join(HERE, "results", "organizer_grid")
PLANES = ("plate_hole_center_px", "hole_center_px", "tube_top_center_px", "cap_center_px")
PRIMARY = "plate_hole_center_px"
CORNERS = (1, 5, 26, 30)            # G01 top-right, G05 bottom-right, G26 top-left, G30 bottom-left
OUTLIER_PX = 5.0


def official_center(grid, position):
    """The labeler's labeler.center(): grid = [top-left, top-right, bottom-left, bottom-right] hole centres."""
    column, row = divmod(position - 1, 5)
    u, v = 1 - column / 5, row / 4
    tl, tr, bl, br = np.asarray(grid, dtype=float)
    return (1 - v) * ((1 - u) * tl + u * tr) + v * ((1 - u) * bl + u * br)


def official_centers(grid):
    return np.array([official_center(grid, i) for i in range(1, 31)])


def fit_similarity(A, B):
    """B ~ s R A + t (Umeyama, rotation only). Returns s, R, t and whether an unconstrained fit wanted a mirror."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a)
    mirror = np.linalg.det(U @ Vt) < 0
    D = np.diag([1.0, -1.0 if mirror else 1.0])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a ** 2).sum())
    return s, R, mb - s * R @ ma, bool(mirror)


def apply(s, R, t, A):
    return (s * (R @ np.asarray(A, float).T)).T + t


def stats(e):
    e = np.asarray(e, float)
    return {"median": round(float(np.median(e)), 2), "mean": round(float(e.mean()), 2),
            "p90": round(float(np.percentile(e, 90)), 2), "max": round(float(e.max()), 2)}


def validate(twin_px, grid):
    """twin_px: 30 twin points (G01..G30, any plane). grid: one organizer tray grid."""
    A, B = np.asarray(twin_px, float), official_centers(grid)
    if A.shape != (30, 2):
        raise ValueError(f"need 30 twin points, got {A.shape}")
    c = [i - 1 for i in CORNERS]
    s, R, t, mirror = fit_similarity(A[c], B[c])
    d = apply(s, R, t, A) - B                                           # signed, official-view px
    e = np.linalg.norm(d, axis=1)
    rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    interior = [i for i in range(30) if i not in c]
    return {
        "positions_matched": int(np.isfinite(e).sum()), "positions_expected": 30,
        "fit": {"on": "4 corners (G01, G05, G26, G30)", "scale": round(s, 4), "rotation_deg": round(rot, 2),
                "translation_px": [round(float(x), 1) for x in t]},
        "numbering": {"mirror_needed": mirror, "rotation_deg": round(rot, 2),
                      "convention_matches": (not mirror) and abs(rot) < 45},
        "corner_residual_px": {f"G{i:02d}": round(float(e[i - 1]), 2) for i in CORNERS},
        "residual_px_all30": stats(e), "residual_px_interior26": stats(e[interior]),
        "residual_capseal_px_all30": stats(e / s),
        "signed_bias_px": {"dx": round(float(d[:, 0].mean()), 2), "dy": round(float(d[:, 1].mean()), 2)},
        "signed_bias_capseal_px": {"dx": round(float(d[:, 0].mean() / s), 2), "dy": round(float(d[:, 1].mean() / s), 2)},
        "outliers": [{"position": f"G{i + 1:02d}", "residual_px": round(float(e[i]), 2)}
                     for i in np.argsort(-e) if e[i] > OUTLIER_PX],
        "per_position_px": {f"G{i + 1:02d}": [round(float(x), 2) for x in d[i]] for i in range(30)},
        "_pred": apply(s, R, t, A).tolist(), "_official": B.tolist(),
    }


def twin_points(pm, plane):
    pos = sorted(pm["positions"], key=lambda p: p["index"])
    if [p["position"] for p in pos] != [f"G{i:02d}" for i in range(1, 31)]:
        raise ValueError("position map must hold G01..G30")
    return [p[plane] for p in pos]


def overlay(img, res, path, scale=2):
    import cv2
    B, P = np.array(res["_official"]), np.array(res["_pred"])
    x0, y0 = (B.min(0) - 60).astype(int)
    x1, y1 = (B.max(0) + 60).astype(int)
    big = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    S = lambda p: (int(round((p[0] - x0) * scale)), int(round((p[1] - y0) * scale)))
    for i, (b, p) in enumerate(zip(B, P)):
        cv2.circle(big, S(b), 24 * scale // 2, (0, 200, 0), 2)                    # official ring
        cv2.drawMarker(big, S(p), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)         # twin, mapped
        cv2.putText(big, f"G{i + 1:02d}", (S(b)[0] - 18, S(b)[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(big, "green ring = organizer grid  magenta + = twin map (similarity fitted on 4 corners)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, big)


def main():
    ap = argparse.ArgumentParser(description="organizer-grid consistency check: twin G01-G30 layout vs organizer grid")
    ap.add_argument("calibration"); ap.add_argument("--example"); ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    pm, cal = json.load(open(MAP_PATH)), json.load(open(a.calibration))
    report = {"twin_map": os.path.basename(MAP_PATH), "calibration_image_size": cal.get("image_size"),
              "primary_plane": PRIMARY, "outlier_px": OUTLIER_PX, "profiles": {}}
    for prof in cal["profiles"]:
        grid = prof["areas"]["good"]["grid"]
        report["profiles"][prof["id"]] = {pl: validate(twin_points(pm, pl), grid) for pl in PLANES}
    os.makedirs(a.out, exist_ok=True)
    clean = {k: v for k, v in report.items() if k != "profiles"}
    clean["profiles"] = {pid: {pl: {k: v for k, v in r.items() if not k.startswith("_")} for pl, r in planes.items()}
                         for pid, planes in report["profiles"].items()}
    json.dump(clean, open(os.path.join(a.out, "organizer_grid_report.json"), "w"), indent=1)
    for pid, planes in report["profiles"].items():
        for pl in PLANES:
            r = planes[pl]; f = r["fit"]; m = r["residual_px_all30"]; c = r["residual_capseal_px_all30"]
            print(f"{pid} {pl:21s} {r['positions_matched']}/30  s={f['scale']:.3f} rot={f['rotation_deg']:+.2f}deg  "
                  f"median {m['median']:.2f} p90 {m['p90']:.2f} max {m['max']:.2f} px  (/s: median {c['median']:.2f} max {c['max']:.2f})  "
                  f"bias ({r['signed_bias_px']['dx']:+.2f}, {r['signed_bias_px']['dy']:+.2f})  "
                  f"convention {'OK' if r['numbering']['convention_matches'] else 'MISMATCH'}  outliers {len(r['outliers'])}")
    if a.example:
        import cv2
        img = cv2.imread(a.example)
        if img is None or img.shape[1::-1] != tuple(cal["image_size"]):
            raise SystemExit(f"example image missing or not {cal['image_size']}")
        name = os.path.basename(a.example)
        prof = next((p["id"] for p in cal["profiles"] if p["reference_image"] == name), cal["profiles"][0]["id"])
        overlay(img, report["profiles"][prof][PRIMARY], os.path.join(a.out, f"overlay_{prof}_{os.path.splitext(name)[0]}.png"))
    print(f"-> {a.out}/organizer_grid_report.json")


if __name__ == "__main__":
    main()
