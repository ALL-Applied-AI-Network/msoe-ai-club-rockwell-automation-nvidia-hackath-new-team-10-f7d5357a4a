"""Render randomization recipe from DEV runs 1-3 only: measure how the real camera's brightness, colour cast, plate
and caps vary, then map those ranges onto scene parameters with the render model validated by gate run 2.

    python calibrate_recipe.py [--render-dir results/isaac_gate/run2] [--out configs/recipe_dev.json]

Real side (DEV runs 1-3 only; real_data refuses runs 4-5): arm-clear frames (run 1 video every 5 s, runs 2-3
stills), the same estimators as on the renders:
  wall    median Lab of the white table panel (compare_gate.py's WALL box)
  plate   median Lab of the Good plate, holes (Otsu) and yellow removed
  caps    per physical seated cap (labels/dev_labels.csv, pass, usable): median Lab in a 7 px disk at the cap seat,
          median over its event frames. A disk, not the HSV cap detector, so pale caps count too.
  noise   consecutive unchanged arm-clear stills: std of the plate difference / sqrt(2)
  jpeg    quality of the DEV stills from their luminance quantisation table (IJG scaling)
Render side: gate run 2 renders (the work1 overlay twin, ACES tone map, /rtx/post/tonemap/op = 6), inverted to
scene-linear with the ACES (Narkowicz) curve. A global light change scales scene-linear light; a light colour
temperature T multiplies it by UsdLux.BlackbodyTemperatureAsRgb(T); a cap albedo change is applied per pixel by
splitting cap light into a diffuse part (albedo x light) and a white specular part (least squares, the model whose
predictions matched run 2 within 1 dE). Every range is a DEV percentile range mapped through this model; nothing is
fitted to runs 4-5 or to any image that is scored later.
"""
import argparse, csv, hashlib, json, os, sys
import cv2, numpy as np
from pxr import UsdLux
import positions as P
import real_data as R
import events as E
import inspect_frame as IF

WALL = (slice(15, 95), slice(400, 432))
LIGHT_RGB = np.array([1.0, 0.8941666, 0.8075703])          # twin RectLights' authored inputs:color (both lights)
CAP_ALBEDO = np.array([0.95, 0.90, 0.00])                  # work1 cap diffuseColor
CAP_DISK_PX, PALE_B = 7, 65.0                               # caps with DEV b* below PALE_B form the pale group


def srgb_dec(u):
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def srgb_enc(l):
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * np.power(np.clip(l, 0.0031308, None), 1 / 2.4) - 0.055)


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def aces_inv(y):
    y = np.clip(y, 0, 0.9999); a, b, c = 2.51 - 2.43 * y, 0.03 - 0.59 * y, -0.14 * y
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)


def to_scene(img):
    return aces_inv(srgb_dec(img[..., ::-1].astype(float) / 255))


def to_image(S):
    return np.clip(np.round(srgb_enc(aces(S)) * 255), 0, 255).astype(np.uint8)[..., ::-1]


def bb(T):
    return np.array(UsdLux.BlackbodyTemperatureAsRgb(float(T)))


def lab_of(img, mask):
    px = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[mask].astype(float)
    m = np.median(px, 0)
    return np.array([m[0] * 100 / 255, m[1] - 128, m[2] - 128])


def plate_mask(img, pm):
    m = np.zeros(img.shape[:2], np.uint8); cv2.fillConvexPoly(m, pm.plate.astype(np.int32), 1)
    m = cv2.erode(m, np.ones((9, 9), np.uint8)).astype(bool)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t, _ = cv2.threshold(g[m].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    holes = cv2.dilate(((g < t) & m).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return m & ~holes & ~(P.yellow_mask(img) > 0)


def disk(shape, c, r=CAP_DISK_PX):
    m = np.zeros(shape, np.uint8); cv2.circle(m, (int(round(c[0])), int(round(c[1]))), r, 1, -1)
    return m.astype(bool)


def jpeg_quality(path):
    """IJG quality from the first (luminance) quantisation table of a baseline JPEG."""
    b = open(path, "rb").read(); i = b.find(b"\xff\xdb")
    q = np.frombuffer(b[i + 5:i + 69], np.uint8).astype(float)
    std = np.array([16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40, 26, 24, 22, 22, 24, 49, 35, 37, 29, 40,
                    58, 51, 61, 60, 57, 51, 56, 55, 64, 72, 92, 78, 64, 68, 87, 69, 55, 56, 80, 109, 81, 87, 95, 98, 103,
                    104, 103, 62, 77, 113, 121, 112, 100, 120, 92, 101, 103, 99], float)   # zig-zag order
    s = np.median(q / std) * 100
    return float(100 - s / 2 if s <= 100 else 5000 / s)


# ------------------------------------------------------------------------------------------------------ real (DEV)
def measure_dev(ix, pm):
    plate_px = pm.plate_mask(); frames = []
    for run in ("1", "2", "3"):
        src = ((R.video_id(run, t), img) for t, img in R.iter_video(ix.video_path(run), 5.0)) if run == "1" else \
              ((f, ix.load(f)) for f in ix.frames(run))
        for name, img in src:
            if E.arm_pixels(img, pm, plate_px):
                continue
            frames.append({"run": run, "frame": name, "wall": lab_of(img, _wall(img)),
                           "plate": lab_of(img, plate_mask(img, pm))})
    caps = {}
    for r in csv.DictReader(open(os.path.join(P.HERE, "labels", "dev_labels.csv"))):
        if r["usable"] == "1" and r["ground_truth"] == "pass":
            caps.setdefault((r["run"], r["position"]), []).append(r["frame"])
    per_cap, cache = [], {}
    for (run, pos), frs in sorted(caps.items()):
        vals = []
        for f in frs:
            cache.setdefault(f, IF.load(ix, f))
            vals.append(lab_of(cache[f], disk(cache[f].shape[:2], pm.cap_centers[pm.ids.index(pos)])))
        per_cap.append({"run": run, "position": pos, "lab": np.median(vals, 0)})
    noise, quality = [], []
    for run in ("2", "3"):
        fr = ix.frames(run)
        for a, b in zip(fr, fr[1:]):
            A, B = ix.load(a), ix.load(b)
            if E.arm_pixels(A, pm, plate_px) or E.arm_pixels(B, pm, plate_px) or \
                    E.box_change(A, B, pm.anchor_boxes).max() >= E.CHANGE_T:
                continue
            noise.append(float((A.astype(float) - B.astype(float))[plate_px > 0].std() / np.sqrt(2)))
        quality += [jpeg_quality(os.path.join(ix.root, "images", f)) for f in fr]
    return frames, per_cap, noise, quality


def _wall(img):
    m = np.zeros(img.shape[:2], bool); m[WALL] = True
    return m


# ------------------------------------------------------------------------------------------------- render model
class RenderModel:
    """Gate run 2 renders (work1 twin) in scene-linear light, with the same estimators as the real side."""

    def __init__(self, render_dir):
        self.base = cv2.imread(os.path.join(render_dir, "renders", "baseline.png"))
        self.t00 = cv2.imread(os.path.join(render_dir, "renders", "t00.png"))
        self.pm = P.PositionMap(offset=(0, 0))                    # renders carry no DEV registration offset
        self.Sb, self.St = to_scene(self.base), to_scene(self.t00)
        self.wall = _wall(self.base); self.plate = plate_mask(self.t00, self.pm)
        self.cap_disks = [disk(self.base.shape[:2], c) for c in self.pm.cap_centers]
        hsv = cv2.cvtColor(self.base, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (10, 50, 40), (45, 255, 255)) & self.pm.plate_mask()
        self.cap_px = cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool)
        a = CAP_ALBEDO * LIGHT_RGB; M = np.stack([a, LIGHT_RGB], 1)
        co = np.linalg.lstsq(M, self.Sb[self.cap_px].T, rcond=None)[0].T
        D, Sp = co[:, 0], co[:, 1]
        n = Sp < 0; D[n] = (self.Sb[self.cap_px][n] @ a) / (a @ a); Sp[n] = 0
        n = D < 0; Sp[n] = (self.Sb[self.cap_px][n] @ LIGHT_RGB) / (LIGHT_RGB @ LIGHT_RGB); D[n] = 0
        self.D, self.Sp = D, Sp

    def image(self, S, ev=0.0, T=None):
        return to_image(S * 2 ** ev * (bb(T) if T else 1.0))

    def wall_lab(self, ev=0.0, T=None):
        return lab_of(self.image(self.Sb, ev, T), self.wall)

    def plate_lab(self, ev=0.0, T=None, scale=1.0):
        S = self.St.copy(); S[self.plate] *= scale
        return lab_of(self.image(S, ev, T), self.plate)

    def cap_lab(self, albedo, ev=0.0, T=None, spec=1.0):
        """spec scales the cap's specular light (F0 relative to the twin's ior 1.5, F0 0.04)."""
        S = self.Sb.copy()
        S[self.cap_px] = self.D[:, None] * (np.asarray(albedo) * LIGHT_RGB) + spec * self.Sp[:, None] * LIGHT_RGB
        im = self.image(S, ev, T)
        return np.median([lab_of(im, d) for d in self.cap_disks], 0)


def solve(f, target, lo, hi, increasing=True, it=40):
    """Bisection for a monotonic scalar function."""
    for _ in range(it):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def hsv_rgb(h, s, v):
    import colorsys
    return np.array(colorsys.hsv_to_rgb(h % 1.0, s, v))


def ior_for_spec(k):
    """UsdPreviewSurface ior whose normal-incidence reflectance is k x the twin's (ior 1.5 -> F0 0.04)."""
    f = np.sqrt(0.04 * k)
    return float((1 + f) / (1 - f))


def fit_cap_group(model, labs, ev, T, rng):
    """Box (diffuse HSV around the work1 cap + specular scale) spanned by the grid points whose rendered cap colour
    falls inside the group's DEV Lab envelope (min..max widened by 2 Lab units), and the share of random samples
    from that box that land inside the envelope too."""
    import colorsys
    lo, hi = labs.min(0) - 2, labs.max(0) + 2
    h0 = colorsys.rgb_to_hsv(*CAP_ALBEDO)[0]
    inside = lambda lab: bool(np.all(lab >= lo) and np.all(lab <= hi))
    hits = [(dh, s, v, k) for dh in np.linspace(-0.03, 0.03, 7) for s in np.linspace(0.75, 1.0, 6)
            for v in np.linspace(0.75, 1.0, 6) for k in (0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
            if inside(model.cap_lab(hsv_rgb(h0 + dh, s, v), ev, T, k))]
    if not hits:
        return None, 0, None
    a = np.array(hits)
    box = {"hue_shift": [r(a[:, 0].min()), r(a[:, 0].max())], "sat": [r(a[:, 1].min()), r(a[:, 1].max())],
           "val": [r(a[:, 2].min()), r(a[:, 2].max())], "spec_scale": [r(a[:, 3].min()), r(a[:, 3].max())]}
    smp = [(rng.uniform(*box["hue_shift"]), rng.uniform(*box["sat"]), rng.uniform(*box["val"]), rng.uniform(*box["spec_scale"]))
           for _ in range(60)]
    cover = np.mean([inside(model.cap_lab(hsv_rgb(h0 + dh, s, v), ev, T, k)) for dh, s, v, k in smp])
    return box, len(hits), r(cover, 2)


def r(x, n=3):
    return float(round(float(x), n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--render-dir", default=os.path.join(P.HERE, "results", "isaac_gate", "run2"))
    ap.add_argument("--out", default=os.path.join(P.HERE, "configs", "recipe_dev.json"))
    a = ap.parse_args()
    ix, pm = R.Index(), P.PositionMap()
    frames, per_cap, noise, quality = measure_dev(ix, pm)
    W = np.array([f["wall"] for f in frames]); PL = np.array([f["plate"] for f in frames])
    C = np.array([c["lab"] for c in per_cap]); pale = C[:, 2] < PALE_B
    pct = lambda x, q: [r(v, 1) for v in np.percentile(x, q)]
    Q = (2.5, 50, 97.5)
    m = RenderModel(a.render_dir)
    # exposure (wall L*) and colour temperature (wall b*), solved jointly: T first at the median exposure, then ev
    T_mid, ev_mid = 6500.0, 0.0
    for _ in range(3):
        T_mid = solve(lambda T: m.wall_lab(ev_mid, T)[2], np.median(W[:, 2]), 2000, 12000, increasing=False)
        ev_mid = solve(lambda e: m.wall_lab(e, T_mid)[0], np.median(W[:, 0]), -2, 2)
    ev_lo, ev_hi = (solve(lambda e: m.wall_lab(e, T_mid)[0], q, -2, 2) for q in np.percentile(W[:, 0], (2.5, 97.5)))
    T_warm, T_cool = (solve(lambda T: m.wall_lab(ev_mid, T)[2], q, 2000, 12000, increasing=False)
                      for q in np.percentile(W[:, 2], (97.5, 2.5)))
    sc = [solve(lambda s: m.plate_lab(ev_mid, T_mid, s)[0], q, 0.3, 4.0) for q in np.percentile(PL[:, 0], Q)]
    rng = np.random.default_rng(0)
    sat_box, n_sat, cov_sat = fit_cap_group(m, C[~pale], ev_mid, T_mid, rng)
    pale_box, n_pale, cov_pale = fit_cap_group(m, C[pale], ev_mid, T_mid, rng)
    a_wall = m.wall_lab(ev_mid, T_mid)[1]
    rec = {
        "source": "DEV runs 1-3 only (real_data guard); render model = gate run 2 renders (work1 twin, ACES)",
        "render_dir": os.path.relpath(a.render_dir, P.HERE),
        "render_baseline_sha256": hashlib.sha256(open(os.path.join(a.render_dir, "renders", "baseline.png"), "rb").read()).hexdigest(),
        "calibrate_recipe_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "dev": {"frames": {k: sum(f["run"] == k for f in frames) for k in ("1", "2", "3")},
                "wall_Lab_p2.5_50_97.5": {"L": pct(W[:, 0], Q), "a": pct(W[:, 1], Q), "b": pct(W[:, 2], Q)},
                "wall_L_median_by_run": {k: r(np.median(W[[f["run"] == k for f in frames], 0]), 1) for k in ("1", "2", "3")},
                "plate_Lab_p2.5_50_97.5": {"L": pct(PL[:, 0], Q), "a": pct(PL[:, 1], Q), "b": pct(PL[:, 2], Q)},
                "caps": [{"run": c["run"], "position": c["position"], "Lab": [r(x, 1) for x in c["lab"]]} for c in per_cap],
                "caps_pale": {"rule": f"b* < {PALE_B}", "n": int(pale.sum()), "of": len(C)},
                "noise_sigma_8bit_p5_50_95": pct(noise, (5, 50, 95)), "noise_pairs": len(noise),
                "jpeg_quality_p5_50_95": pct(quality, (5, 50, 95)),
                "position_error_px": "empty-hole check run 3 (registration_dev.json): median 1.68, p90 2.53"},
        "render_model_at_work1": {"wall_Lab": [r(x, 1) for x in m.wall_lab()], "plate_Lab": [r(x, 1) for x in m.plate_lab()],
                                  "cap_Lab_disk_median": [r(x, 1) for x in m.cap_lab(CAP_ALBEDO)]},
        "derived": {
            "exposure_ev": {"range": [r(ev_lo), r(ev_hi)], "mid": r(ev_mid),
                            "rule": "both lights x 2^ev on top of work1; ev maps DEV wall L* p2.5..p97.5 (at T mid)"},
            "color_temperature_k": {"range": [r(min(T_warm, T_cool), 0), r(max(T_warm, T_cool), 0)], "mid": r(T_mid, 0),
                                    "rule": "UsdLux colour temperature on both lights; maps DEV wall b* p97.5..p2.5 (at ev mid)"},
            "plate_val_scale": {"range": [r(sc[0]), r(sc[2])], "mid": r(sc[1]),
                                "rule": "plate diffuseColor x scale; maps DEV plate L* p2.5..p97.5 (at ev, T mid)"},
            "cap_saturated": {"box": sat_box, "grid_hits": n_sat, "box_samples_inside_dev_envelope": cov_sat,
                              "n_dev_caps": int((~pale).sum()), "dev_Lab_min": [r(x, 1) for x in C[~pale].min(0)],
                              "dev_Lab_max": [r(x, 1) for x in C[~pale].max(0)]},
            "cap_pale": {"box": pale_box, "grid_hits": n_pale, "box_samples_inside_dev_envelope": cov_pale,
                         "n_dev_caps": int(pale.sum()), "dev_Lab_min": [r(x, 1) for x in C[pale].min(0)],
                         "dev_Lab_max": [r(x, 1) for x in C[pale].max(0)]},
            "cap_rule": "diffuse = HSV(work1 hue + hue_shift, sat, val); ior = ior_for_spec(spec_scale) "
                        "(F0 = 0.04 x spec_scale); groups split at DEV b* 65",
            "cap_pale_fraction": r(pale.mean()),
            "image_aug": {"green_gain": _green_gain(m, ev_mid, T_mid, np.percentile(W[:, 1], (2.5, 97.5))),
                          "noise_sigma_8bit": pct(noise, (5, 95)), "jpeg_quality": pct(quality, (5, 95)),
                          "translate_px": 3, "rule": "applied to crops at training time; green gain maps DEV wall a* "
                                                       "p2.5..p97.5; translate covers the run 3 empty-hole p90 (2.53 px)"},
            "wall_a_at_mid": r(a_wall, 1)}}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("dev", "render_model_at_work1", "derived")}, indent=1)[:6000])


def _green_gain(m, ev, T, a_targets):
    """Per-image green-channel gain (linear light, after tone mapping, like a camera white balance) that moves the
    rendered wall a* to the DEV a* p2.5 and p97.5 (a* falls as green rises)."""
    img = m.image(m.Sb, ev, T)
    def a_of(g):
        L = srgb_dec(img[..., ::-1].astype(float) / 255); L[..., 1] *= g
        return lab_of(np.clip(np.round(srgb_enc(L) * 255), 0, 255).astype(np.uint8)[..., ::-1], m.wall)[1]
    return sorted(r(solve(a_of, t, 0.7, 1.4, increasing=False)) for t in a_targets)


if __name__ == "__main__":
    sys.exit(main())
