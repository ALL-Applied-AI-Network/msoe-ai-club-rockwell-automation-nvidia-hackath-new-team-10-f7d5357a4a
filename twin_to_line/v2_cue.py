"""v2 cue probes (diagnostic only; DEV runs 1-3, D033 development; never a threshold): what in a real empty-hole crop
makes ResNet-18 v1 read an uncapped vial, how seated-cap position compares between real and train_v1, and single-factor
image probes for the D040 candidates.

(a) the 166 real empty holes v1 read as missing_cap (G3, DEV runs 1-3)
  swap_context  real hole (disk HOLE_RIM px, feathered) + synthetic empty surroundings at the same position
  swap_hole     real surroundings + the synthetic empty hole (the true reverse)
                synthetic = N_SYN train_v1 empty crops at that position after train_resnet.CameraAug (what v1 trained on),
                aligned on the hole centre; a crop counts as read missing_cap when the mean of its N_SYN scores says so
  occlusion     10 misread top-row crops: a PATCH x PATCH window (stride STRIDE) replaced by the aligned synthetic empty
                crop; map = drop of P(missing_cap); summary inside / outside the hole rim and by side
  rows          what top-row crops hold that other rows don't: share of the crop outside the plate, L* of that part,
                plate L* top vs bottom of the crop (lighting gradient), real vs synthetic
(b) seated-cap position: cap-blob offset from the expected seat, real DEV pass vs train_v1 seated / offset / crooked,
  raw and with v1's training window shift (uniform integer +-3 px); the Gaussian translation jitter (sigma, px) that gives
  synthetic seated caps the real spread, and how well |offset| then separates crooked from seated (AUC)
Single-factor probes on real crops: a* moved to the synthetic+aug median (green gain), noise added at v1's training
level, Gaussian blur sigma 0.7, L* and b* whole-crop mean / std moved to the synthetic+aug seated median (and
v2_analysis' sharpen), so each D040 candidate has its own probe.

    python v2_cue.py [--out results/v2_analysis]
"""
import argparse, json, os
import cv2, numpy as np
import positions as P
import train_resnet as T
import v2_analysis as V

HOLE_RIM, FEATHER, N_SYN, PATCH, STRIDE = 11, 1.5, 5, 6, 2
SIGMAS = np.round(np.arange(1.0, 4.01, 0.25), 2)


class Scorer:
    def __init__(self):
        from resnet_inspector import ResNetInspector
        self.insp = ResNetInspector(); self.torch = self.insp.torch

    def __call__(self, imgs, bs=256):
        out = []
        for k in range(0, len(imgs), bs):
            x = self.torch.from_numpy(np.stack([T.to_tensor_array(im) for im in imgs[k:k + bs]])).to(self.insp.device)
            with self.torch.no_grad():
                out.append(self.torch.softmax(self.insp.model(x), 1).cpu().numpy())
        return np.concatenate(out)


def shift(im, d):
    return cv2.warpAffine(im, np.float32([[1, 0, d[0]], [0, 1, d[1]]]), im.shape[1::-1], flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def soft_disk(shape, c, r):
    m = V.disk(shape, c, r).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), FEATHER)[..., None]


def blend(a, b, m):
    return np.clip(a * m + b * (1 - m), 0, 255).astype(np.uint8)


def synthetic_empties(pos, geo_real):
    """N_SYN synthetic+aug empty crops at each position (spread over hole brightness), aligned to the real hole."""
    aug = T.CameraAug(json.load(open(os.path.join(P.HERE, "configs", "recipe_dev.json")))); rng = np.random.default_rng(0)
    by = {}
    for r, path, geo in V.synthetic_crops({"no_vial"}, pos):
        by.setdefault(r["position"], []).append((path, geo))
    out = {}
    for p, items in by.items():
        imgs = [(cv2.imread(f), g) for f, g in items]
        imgs.sort(key=lambda t: V.features(t[0], t[1])["hole_L"])
        pick = [imgs[int(q * (len(imgs) - 1))] for q in np.linspace(0.1, 0.9, N_SYN)]
        out[p] = [(shift(aug(im, rng), geo_real[p]["hole"] - g["hole"]), g) for im, g in pick]
    return out


def auc(pos, neg):
    """P(a random positive scores higher than a random negative)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def window_shift(d, rng, kind, sigma=None):
    """Cap offset as v1 saw it: the crop window moved by s, so the cap moves by -s in crop coordinates."""
    if kind == "uniform3":
        s = rng.integers(-3, 4, size=d.shape)
    else:
        s = np.clip(np.round(rng.normal(0, sigma, size=d.shape)), -8, 8)
    return np.hypot(*(d - s).T)


def main():
    ap = argparse.ArgumentParser(description="v2 cue probes (diagnostic, DEV runs 1-3)")
    ap.add_argument("--out", default=V.OUT); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    score = Scorer(); res = {}
    real, _ = V.real_crops()
    mis = [(k, im, geo) for k, im, geo, row in real if row["ground_truth"] == "no_vial" and row["predicted_state"] == "missing_cap"]
    geo_real = {k[1]: geo for k, _, geo in mis}
    syn = synthetic_empties({k[1] for k, _, _ in mis}, geo_real)

    # (a) context swap, hole swap
    def read_missing(build):
        n = 0; ps = []
        for k, im, geo in mis:
            imgs = [build(im, geo, s) for s, _ in syn[k[1]] if s.shape == im.shape]
            p = score(imgs).mean(0); ps.append(p[1]); n += int(p.argmax() == 1)
        return {"read_missing_cap": f"{n}/{len(mis)}", "median_p_missing": round(float(np.median(ps)), 3)}
    m_of = lambda im, geo: soft_disk(im.shape, geo["hole"], HOLE_RIM)
    res["a_baseline"] = {"read_missing_cap": f"{int((score([im for _, im, _ in mis]).argmax(1) == 1).sum())}/{len(mis)}"}
    res["a_swap_context (real hole + synthetic surroundings)"] = read_missing(lambda im, geo, s: blend(im, s, m_of(im, geo)))
    res["a_swap_hole (real surroundings + synthetic hole)"] = read_missing(lambda im, geo, s: blend(s, im, m_of(im, geo)))
    res["a_synthetic_empty_alone (control)"] = read_missing(lambda im, geo, s: s)

    # (a) occlusion maps on 10 misread top-row crops
    top = sorted([t for t in mis if t[0][1] in V.TOP_ROW], key=lambda t: t[0])
    top = [top[int(i)] for i in np.linspace(0, len(top) - 1, min(10, len(top)))]
    rows_img, inside, outside, sides = [], [], [], {"top": [], "bottom": [], "left": [], "right": []}
    for k, im, geo in top:
        s = next(x for x, _ in syn[k[1]][N_SYN // 2:] if x.shape == im.shape)
        h, w = im.shape[:2]; p0 = score([im])[0, 1]; boxes, imgs = [], []
        for y in range(0, h - PATCH + 1, STRIDE):
            for x in range(0, w - PATCH + 1, STRIDE):
                o = im.copy(); o[y:y + PATCH, x:x + PATCH] = s[y:y + PATCH, x:x + PATCH]; imgs.append(o); boxes.append((x, y))
        drop = p0 - score(imgs)[:, 1]
        heat = np.zeros((h, w)); cnt = np.zeros((h, w))
        for (x, y), d in zip(boxes, drop):
            heat[y:y + PATCH, x:x + PATCH] += d; cnt[y:y + PATCH, x:x + PATCH] += 1
        heat /= np.maximum(cnt, 1); rim = V.disk(im.shape, geo["hole"], HOLE_RIM)
        inside.append(float(heat[rim].mean())); outside.append(float(heat[~rim].mean()))
        cy, cx = geo["hole"][1], geo["hole"][0]; yy, xx = np.mgrid[0:h, 0:w]; out_rim = ~rim
        sides["top"].append(float(heat[out_rim & (yy < cy - HOLE_RIM)].mean()) if (out_rim & (yy < cy - HOLE_RIM)).any() else 0.0)
        sides["bottom"].append(float(heat[out_rim & (yy > cy + HOLE_RIM)].mean()) if (out_rim & (yy > cy + HOLE_RIM)).any() else 0.0)
        sides["left"].append(float(heat[out_rim & (xx < cx - HOLE_RIM)].mean()) if (out_rim & (xx < cx - HOLE_RIM)).any() else 0.0)
        sides["right"].append(float(heat[out_rim & (xx > cx + HOLE_RIM)].mean()) if (out_rim & (xx > cx + HOLE_RIM)).any() else 0.0)
        hm = cv2.applyColorMap(np.clip(heat / max(heat.max(), 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.circle(hm, tuple(int(round(v)) for v in geo["hole"]), HOLE_RIM, (255, 255, 255), 1)
        rows_img.append((k[1], [(im, f"real p{p0:.2f}"), (s, "syn empty"), (hm, f"drop max {heat.max():.2f}"),
                                (cv2.addWeighted(im, 0.55, hm, 0.45, 0), k[0][-12:])]))
    V.sheet(rows_img, os.path.join(a.out, "occlusion_a_top_row.png"),
            "(a) occlusion: drop of P(missing_cap) when a 6x6 patch is replaced by the aligned synthetic empty crop (white = hole rim)")
    res["a_occlusion_10_top_row"] = {"mean_drop_inside_rim": round(float(np.mean(inside)), 3), "mean_drop_outside_rim": round(float(np.mean(outside)), 3),
                                     "mean_drop_by_side_outside_rim": {k: round(float(np.mean(v)), 3) for k, v in sides.items()}}

    # (a) what top-row crops hold: plate share, outside-plate L*, plate gradient (real vs synthetic, same geometry)
    pm = P.PositionMap(); pm0 = P.PositionMap(offset=(0, 0))
    def row_stats(img, pmap, i):
        (x0, y0, x1, y1), _ = pmap.crop_box(i); poly = pmap.plate - [x0, y0]
        m = np.zeros(img.shape[:2], np.uint8); cv2.fillConvexPoly(m, np.round(poly).astype(np.int32), 1); m = m.astype(bool)
        L = V.lab(img)[..., 0]; h = img.shape[0]; plate_top = L[:h // 3][m[:h // 3]]; plate_bot = L[-h // 3:][m[-h // 3:]]
        return {"outside_plate_share": float((~m).mean()), "outside_plate_L": float(L[~m].mean()) if (~m).any() else None,
                "plate_L_top_minus_bottom": float(plate_top.mean() - plate_bot.mean()) if len(plate_top) and len(plate_bot) else None}
    rs = {"real top row": [], "real other rows": [], "synthetic top row": [], "synthetic other rows": []}
    for k, im, geo, row in real:
        if row["ground_truth"] == "no_vial":
            rs["real top row" if k[1] in V.TOP_ROW else "real other rows"].append(row_stats(im, pm, pm.ids.index(k[1])))
    for p, items in syn.items():
        for s, _ in items:
            rs["synthetic top row" if p in V.TOP_ROW else "synthetic other rows"].append(row_stats(s, pm0, pm0.ids.index(p)))
    res["a_rows"] = {g: {key: round(float(np.median([r[key] for r in v if r[key] is not None])), 3) for key in v[0]} for g, v in rs.items() if v}

    # (b) seated-cap position: real vs train_v1, raw and as v1 saw it; jitter that gives seated the real spread
    real_pass = np.array([[f["cap_dx"], f["cap_dy"]] for (k, im, geo, row) in real if row["ground_truth"] == "pass"
                          for f in [V.features(im, geo)] if "cap_dx" in f])
    syn_rows = V.synthetic_crops({"pass", "crooked_cap"}, set(P.PositionMap().ids))
    d = {"seated": [], "offset": [], "crooked": []}
    for r, path, geo in syn_rows:
        f = V.features(cv2.imread(path), geo)
        if "cap_dx" in f:
            d["crooked" if r["label"] == "crooked_cap" else r["variant"]].append([f["cap_dx"], f["cap_dy"]])
    d = {k: np.array(v) for k, v in d.items()}; seated = np.vstack([d["seated"], d["offset"]]); rng = np.random.default_rng(0)
    q = lambda v: {"p10": round(float(np.percentile(v, 10)), 2), "median": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2)}
    real_r = np.hypot(*real_pass.T)
    res["b_offset_px"] = {"real DEV pass": {"n": len(real_r), **q(real_r)},
                          "train_v1 seated+offset, raw": {"n": len(seated), **q(np.hypot(*seated.T))},
                          "train_v1 crooked, raw": {"n": len(d["crooked"]), **q(np.hypot(*d["crooked"].T))},
                          "train_v1 seated+offset, v1 window shift (uniform +-3)": q(window_shift(seated, rng, "uniform3")),
                          "train_v1 crooked, v1 window shift (uniform +-3)": q(window_shift(d["crooked"], rng, "uniform3"))}
    res["b_auc_crooked_vs_seated_by_abs_offset"] = {"raw": round(auc(np.hypot(*d["crooked"].T), np.hypot(*seated.T)), 3),
                                                    "v1 window shift": round(auc(window_shift(d["crooked"], rng, "uniform3"), window_shift(seated, rng, "uniform3")), 3)}
    search = {}
    for sg in SIGMAS:
        r_seat = window_shift(seated, np.random.default_rng(1), "gauss", sg)
        search[float(sg)] = {"seated_median": round(float(np.median(r_seat)), 2), "seated_p90": round(float(np.percentile(r_seat, 90)), 2),
                             "ks_vs_real": round(float(np.max(np.abs(np.searchsorted(np.sort(r_seat), np.sort(np.r_[r_seat, real_r]), "right") / len(r_seat)
                                                                     - np.searchsorted(np.sort(real_r), np.sort(np.r_[r_seat, real_r]), "right") / len(real_r)))), 3),
                             "auc_crooked_vs_seated": round(auc(window_shift(d["crooked"], np.random.default_rng(2), "gauss", sg), r_seat), 3)}
    best = min(search, key=lambda s: search[s]["ks_vs_real"])
    res["b_jitter_search"] = search; res["b_jitter_proposed_sigma_px"] = best

    # single-factor probes on real crops (b: seated, defect >= 0.5; a: the 166)
    pb = [(im, geo) for k, im, geo, row in real if row["ground_truth"] == "pass" and row["defect_score"] >= 0.5]
    m_ = json.load(open(os.path.join(a.out, "measurements.json")))
    a_target = m_["probe_targets"]["synthetic+aug seated"][0][1]; a_real = m_["probe_targets"]["real seated, defect >= 0.5"][0][1]
    def ashift(im, da):
        L = V.lab(im); L[..., 1] += da; L[..., 0] *= 255 / 100; L[..., 1:] += 128
        return cv2.cvtColor(np.clip(L, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    noise = lambda im, r: np.clip(im + r.normal(0, 2.7, im.shape), 0, 255).astype(np.uint8)
    tgt = m_["probe_targets"]["synthetic+aug seated"]
    def chan(im, c):                                   # one L*a*b* channel's whole-crop mean / std moved to the target
        L = V.lab(im); x = L[..., c]; L[..., c] = (x - x.mean()) / (x.std() + 1e-6) * tgt[1][c] + tgt[0][c]
        L[..., 0] *= 255 / 100; L[..., 1:] += 128
        return cv2.cvtColor(np.clip(L, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    probes = {"a* moved to the synthetic+aug median (green gain)": lambda im, r: ashift(im, a_target - a_real),
              "noise sigma 2.7 added (v1 training level)": noise,
              "Gaussian blur sigma 0.7": lambda im, r: cv2.GaussianBlur(im, (0, 0), 0.7),
              "L* mean / std moved to the synthetic+aug seated median (brightness, contrast)": lambda im, r: chan(im, 0),
              "b* mean / std moved to the synthetic+aug seated median": lambda im, r: chan(im, 2)}
    sp = {}
    for name, fn in probes.items():
        r = np.random.default_rng(0); pb_before = score([im for im, _ in pb]); pb_after = score([fn(im, r) for im, _ in pb])
        r = np.random.default_rng(0); pa_after = score([fn(im, r) for _, im, _ in mis])
        sp[name] = {"(b) read crooked": f"{int((pb_before.argmax(1) == 2).sum())} -> {int((pb_after.argmax(1) == 2).sum())} of {len(pb)}",
                    "(b) median defect": f"{np.median(pb_before[:, 1] + pb_before[:, 2]):.3f} -> {np.median(pb_after[:, 1] + pb_after[:, 2]):.3f}",
                    "(a) read missing_cap": f"{len(mis)} -> {int((pa_after.argmax(1) == 1).sum())} of {len(mis)}"}
    res["single_factor_probes"] = sp
    json.dump(res, open(os.path.join(a.out, "cue_probes.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
