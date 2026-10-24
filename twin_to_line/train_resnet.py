"""Synthetic-only ResNet-18 position classifier (D003, D037): trained on twin renders only, thresholds from synthetic
validation only (D033). No real image is read by this script.

    python train_resnet.py --data DIR [DIR ...] --out results/train/resnet18_v1 [--epochs 15] [--device mps]

Each DIR holds renders/<scene_id>.png and crops.csv (cut_train_crops.py). Crops are cut again from the full render
with the runtime's geometry (positions.PositionMap, offset 0) so a shifted window can be drawn. Classes: pass,
missing_cap, crooked_cap, no_vial. Split by scene as in crops.csv (every 5th scene = validation).

Fixed before any training render existed (Oct 23):
  input       crop -> positions-style square pad (edge pixels) -> 64 px (positions.MODEL_SIZE) -> 128 px for the net,
              ImageNet mean/std
  camera aug  from configs/recipe_dev.json image_aug (DEV runs 1-3): window shift +-3 px, green gain 1.10-1.32 in linear
              light, Gaussian noise sigma 2.6-2.8 (8-bit), JPEG quality 95. Applied to training crops at random and to
              validation crops with a fixed seed, so validation looks like what the camera delivers.
  model       torchvision ResNet-18, ImageNet weights, new 4-way head; AdamW lr 3e-4, wd 1e-4, cosine, batch 128,
              class-weighted cross entropy (inverse frequency), 15 epochs, the final epoch is the model (no epoch picking)
  threshold   defect score s = P(missing_cap) + P(crooked_cap). t = the highest threshold that flags >= 99.5 % of the
              validation defect crops (missing_cap + crooked_cap). Visual state at runtime: if s >= t, the more likely
              defect class; otherwise the more likely of pass / no_vial. Reported with it: validation false-alarm rate on
              pass and on no_vial, per-class recall, and the threshold that catches every validation defect (MAIC metric).
  D038 (Oct 23, after v1 trained, before any DEV evaluation): the deployed threshold is the midpoint of the empty gap
              between the highest non-defect and the lowest defect score on synthetic validation, because the 99.5 %
              rule degenerates when validation separates perfectly. thresholds_d038.json carries it and the
              pre-registered t_defect, reported side by side. `--d038 OUT` writes it for a finished run.
  D040 (v2)   --aug configs/aug_v2.json: v1's camera aug plus L* shift and contrast, b* shift and Gaussian blur, with
              noise sigma 0-2.8; each change backed by a DEV probe (v2_cue.py). If validation scores overlap (no empty
              gap), thresholds_d038.json deploys the pre-registered threshold instead (flagged d040_fallback).
Outputs (OUT, gitignored): model.pt, thresholds.json, thresholds_d038.json, train_log.csv, val_predictions.csv,
run.json (data and code hashes, versions, seed), and the sha256 of model.pt in run.json.
"""
import argparse, csv, hashlib, json, os, platform, subprocess, sys, time
import cv2, numpy as np
import positions as P

CLASSES = ("pass", "missing_cap", "crooked_cap", "no_vial")
DEFECTS = (1, 2)
INPUT_PX = 128
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
RECALL_TARGET = 0.995


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def square_resize(img, size=P.MODEL_SIZE):
    """positions.crop_for_model on a raw crop array: pad to square with edge pixels, resize to size."""
    h, w = img.shape[:2]; s = max(h, w); t, l = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(img, t, s - h - t, l, s - w - l, cv2.BORDER_REPLICATE)
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor_array(crop_bgr):
    """Model input from a BGR crop (renders now, live frames later): 3 x 128 x 128 float32, ImageNet-normalized."""
    x = cv2.resize(square_resize(crop_bgr), (INPUT_PX, INPUT_PX), interpolation=cv2.INTER_LINEAR)
    x = (x[..., ::-1].astype(np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1).astype(np.float32)


def srgb_dec(u):
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def srgb_enc(l):
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * np.power(np.clip(l, 0.0031308, None), 1 / 2.4) - 0.055)


class CameraAug:
    def __init__(self, recipe):
        a = recipe["derived"]["image_aug"]
        self.gain, self.noise, self.q, self.shift = a["green_gain"], a["noise_sigma_8bit"], int(a["jpeg_quality"][0]), int(a["translate_px"])

    def __call__(self, crop, rng):
        lin = srgb_dec(crop[..., ::-1].astype(np.float32) / 255)
        lin[..., 1] *= rng.uniform(*self.gain)
        x = srgb_enc(np.clip(lin, 0, 1))[..., ::-1] * 255 + rng.normal(0, rng.uniform(*self.noise), crop.shape)
        x = np.clip(np.round(x), 0, 255).astype(np.uint8)
        return cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, self.q])[1], cv2.IMREAD_COLOR)


class CameraAugV2(CameraAug):
    """D040: per crop, L* shift and contrast about the crop mean and a b* shift (CIELAB), then v1's green gain in
    linear light, a Gaussian blur, noise (sigma from the v2 range) and JPEG at v1's quality. Window shift as v1."""

    def __init__(self, recipe, v2):
        super().__init__(recipe)
        self.noise = tuple(v2["noise_sigma_8bit"]); self.blur = tuple(v2["blur_sigma"])
        self.dL, self.cL, self.db = tuple(v2["L_shift"]), tuple(v2["L_contrast"]), tuple(v2["b_shift"])

    def __call__(self, crop, rng):
        dl, c, db, g, s, n = (rng.uniform(*r) for r in (self.dL, self.cL, self.db, self.gain, self.blur, self.noise))
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0] * (100 / 255); m = L.mean()
        lab[..., 0] = np.clip((L - m) * c + m + dl, 0, 100) * (255 / 100); lab[..., 2] = np.clip(lab[..., 2] + db, 0, 255)
        x = cv2.cvtColor(np.round(lab).astype(np.uint8), cv2.COLOR_LAB2BGR)
        lin = srgb_dec(x[..., ::-1].astype(np.float32) / 255); lin[..., 1] *= g
        x = (srgb_enc(np.clip(lin, 0, 1))[..., ::-1] * 255).astype(np.float32)
        if s >= 0.05:
            x = cv2.GaussianBlur(x, (0, 0), s)
        x = np.clip(np.round(x + rng.normal(0, n, crop.shape)), 0, 255).astype(np.uint8)
        return cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, self.q])[1], cv2.IMREAD_COLOR)


def make_aug(recipe, aug_path=None):
    """v1 (no aug file) or D040's v2 (configs/aug_v2.json)."""
    return CameraAugV2(recipe, json.load(open(aug_path))) if aug_path else CameraAug(recipe)


class CropSet:
    """All crops of one split, held as (render index, position index, label); pixels cut on demand."""

    def __init__(self, dirs, split, aug, seed=None):
        self.pm = P.PositionMap(offset=(0, 0)); self.aug = aug; self.seed = seed
        self.renders, self.items, self.rows = [], [], []
        for d in dirs:
            rows = [r for r in csv.DictReader(open(os.path.join(d, "crops.csv"))) if r["split"] == split]
            index = {}
            for r in rows:
                if r["label"] == "crooked_cap" and r["bbox_in_crop"] == "False":
                    continue                                   # cap leaves the runtime window: not a learnable crop
                if r["scene_id"] not in index:
                    index[r["scene_id"]] = len(self.renders)
                    img = cv2.imread(os.path.join(d, "renders", r["scene_id"] + ".png"))
                    if img is None:
                        raise FileNotFoundError(os.path.join(d, "renders", r["scene_id"] + ".png"))
                    self.renders.append(img)
                self.items.append((index[r["scene_id"]], self.pm.ids.index(r["position"]), CLASSES.index(r["label"])))
                self.rows.append(r)

    def __len__(self):
        return len(self.items)

    def get(self, i, rng):
        ri, pi, y = self.items[i]
        (x0, y0, x1, y1), _ = self.pm.crop_box(pi)
        dx, dy = (rng.integers(-self.aug.shift, self.aug.shift + 1, 2) if self.aug else (0, 0))
        img = self.renders[ri]; H, W = img.shape[:2]
        x0, x1 = np.clip([x0 + dx, x1 + dx], 0, W); y0, y1 = np.clip([y0 + dy, y1 + dy], 0, H)
        crop = img[y0:y1, x0:x1]
        if self.aug:
            crop = self.aug(crop, rng)
        return to_tensor_array(crop), y


def batches(ds, idx, bs, rng_for):
    for k in range(0, len(idx), bs):
        xs, ys = zip(*(ds.get(i, rng_for(i)) for i in idx[k:k + bs]))
        yield np.stack(xs), np.array(ys)


def evaluate(model, ds, bs, device, torch):
    model.eval(); probs = []
    with torch.no_grad():
        for x, _ in batches(ds, np.arange(len(ds)), bs, lambda i: np.random.default_rng([ds.seed, i])):
            probs.append(torch.softmax(model(torch.from_numpy(x).to(device)), 1).cpu().numpy())
    return np.concatenate(probs), np.array([it[2] for it in ds.items])


def decide(probs, t):
    """Class index per row: s >= t -> the more likely defect class, else the more likely of pass / no_vial."""
    s = probs[:, DEFECTS].sum(1)
    return np.where(s >= t, np.where(probs[:, 1] >= probs[:, 2], 1, 2), np.where(probs[:, 0] >= probs[:, 3], 0, 3))


def choose_threshold(probs, y):
    s = probs[:, DEFECTS].sum(1); d = np.isin(y, DEFECTS)
    sd = np.sort(s[d])
    t = float(sd[int(np.floor((1 - RECALL_TARGET) * len(sd)))])    # >= RECALL_TARGET of defects have s >= t
    t_all = float(sd[0])
    far = lambda thr, c: float(np.mean(s[y == c] >= thr)) if np.any(y == c) else None
    pred = decide(probs, t)
    return {"rule": f"s = P(missing_cap) + P(crooked_cap); t = highest threshold flagging >= {RECALL_TARGET:.1%} of "
                    "validation defects; s >= t -> the more likely defect class, else the more likely of pass / no_vial",
            "t_defect": t, "val_defect_recall": float(np.mean(s[d] >= t)),
            "val_false_alarm_rate": {"pass": far(t, 0), "no_vial": far(t, 3)},
            "t_catch_every_val_defect": t_all,
            "val_false_alarm_rate_at_t_catch_every": {"pass": far(t_all, 0), "no_vial": far(t_all, 3)},
            "val_per_class_recall_after_rule": {c: float(np.mean(pred[y == k] == k)) for k, c in enumerate(CLASSES)},
            "val_confusion_after_rule": {c: {c2: int(np.sum((y == k) & (pred == k2))) for k2, c2 in enumerate(CLASSES)}
                                         for k, c in enumerate(CLASSES)},
            "n_val": int(len(y)), "n_val_defects": int(d.sum())}


def gap_threshold(probs, y, t_preregistered, source, fallback_preregistered=False):
    """D038 (Oct 23, before any DEV evaluation): t = midpoint of the empty gap between the highest non-defect score and
    the lowest defect score on synthetic validation. The 99.5 % rule degenerates when validation separates perfectly
    (it puts t at the top of the defect scores). D038 covers only that case: if the scores overlap, t is None and a
    new decision is needed. The pre-registered threshold is carried along so every result can report both."""
    s = probs[:, DEFECTS].sum(1); d = np.isin(y, DEFECTS)
    hi, lo = float(s[~d].max()), float(s[d].min())
    t = (hi + lo) / 2 if lo > hi else None
    fallback = t is None and fallback_preregistered and t_preregistered is not None
    if fallback:                                       # D040: no empty gap -> the pre-registered 99.5 % threshold
        t = t_preregistered
    out = {"decision": "D038", "rule": "t = midpoint of the empty gap between the highest non-defect score and the lowest "
                                       "defect score on synthetic validation; s = P(missing_cap) + P(crooked_cap); s >= t -> "
                                       "the more likely defect class, else the more likely of pass / no_vial",
           "t_defect": t, "gap": {"max_non_defect_score": hi, "min_defect_score": lo, "width": lo - hi},
           **({"d040_fallback": "no empty gap: the pre-registered 99.5 %-recall threshold is deployed (D040)"} if fallback else {}),
           "t_defect_preregistered": t_preregistered, "source": source, "n_val": int(len(y)), "n_val_defects": int(d.sum())}
    if t is not None:
        pred = decide(probs, t); mean = lambda v: float(np.mean(v)) if len(v) else None
        out.update({"val_defect_recall": mean(s[d] >= t),
                    "val_false_alarm_rate": {"pass": mean(s[y == 0] >= t), "no_vial": mean(s[y == 3] >= t)},
                    "val_per_class_recall_after_rule": {c: mean(pred[y == k] == k) for k, c in enumerate(CLASSES)}})
    return out


def write_d038(out):
    """thresholds_d038.json for a finished run, from its val_predictions.csv (probabilities rounded to 6 decimals)."""
    rows = list(csv.DictReader(open(os.path.join(out, "val_predictions.csv"))))
    probs = np.array([[float(r[f"p_{c}"]) for c in CLASSES] for r in rows]); y = np.array([CLASSES.index(r["label"]) for r in rows])
    pre = json.load(open(os.path.join(out, "thresholds.json")))["t_defect"]
    th = gap_threshold(probs, y, pre, {"val_predictions_csv_sha256": sha256(os.path.join(out, "val_predictions.csv")),
                                       "thresholds_json_sha256": sha256(os.path.join(out, "thresholds.json")),
                                       "note": "probabilities as rounded in val_predictions.csv"})
    json.dump(th, open(os.path.join(out, "thresholds_d038.json"), "w"), indent=1)
    return th


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs="+"); ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="mps"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recipe", default=os.path.join(P.HERE, "configs", "recipe_dev.json"))
    ap.add_argument("--d038", metavar="OUT", help="write thresholds_d038.json for a finished run and exit")
    ap.add_argument("--aug", help="v2 augmentation file (configs/aug_v2.json, D040); default: v1's camera aug")
    a = ap.parse_args()
    if a.d038:
        th = write_d038(a.d038)
        print(json.dumps({k: v for k, v in th.items() if k not in ("rule", "source")}, indent=1)); return
    if not (a.data and a.out):
        ap.error("--data and --out are required to train")
    import torch, torchvision
    torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "model.pt")):
        sys.exit(f"{a.out} already has a model; use a new output folder")
    device = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    aug = make_aug(json.load(open(a.recipe)), a.aug)
    t0 = time.time()
    train, val = CropSet(a.data, "train", aug, seed=None), CropSet(a.data, "val", aug, seed=a.seed + 1)
    counts = np.bincount([it[2] for it in train.items], minlength=4)
    print(f"train {len(train)} crops {dict(zip(CLASSES, counts.tolist()))} | val {len(val)} | loaded in {time.time() - t0:.0f}s", flush=True)
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES)); model.to(device)
    w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(train) / a.batch)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(a.seed)
    log = open(os.path.join(a.out, "train_log.csv"), "w", newline=""); lw = csv.writer(log)
    lw.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
    for ep in range(1, a.epochs + 1):
        model.train(); te = time.time(); tl = tc = n = 0
        for x, y in batches(train, rng.permutation(len(train)), a.batch, lambda i: rng):
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            out = model(x); loss = loss_fn(out, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(y); tc += int((out.argmax(1) == y).sum()); n += len(y)
        pv, yv = evaluate(model, val, a.batch, device, torch)
        vl = float(torch.nn.functional.nll_loss(torch.log(torch.from_numpy(pv).clamp_min(1e-9)), torch.from_numpy(yv),
                                                 weight=w.cpu())); va = float(np.mean(pv.argmax(1) == yv))
        lw.writerow([ep, round(tl / n, 4), round(tc / n, 4), round(vl, 4), round(va, 4), round(time.time() - te, 1)]); log.flush()
        print(f"epoch {ep:2d}  train loss {tl / n:.4f} acc {tc / n:.4f} | val loss {vl:.4f} acc {va:.4f} | {time.time() - te:.0f}s", flush=True)
    log.close()
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    th = choose_threshold(pv, yv)
    json.dump(th, open(os.path.join(a.out, "thresholds.json"), "w"), indent=1)
    json.dump(gap_threshold(pv, yv, th["t_defect"], {"note": "validation probabilities in memory"}, fallback_preregistered=bool(a.aug)),
              open(os.path.join(a.out, "thresholds_d038.json"), "w"), indent=1)
    with open(os.path.join(a.out, "val_predictions.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["scene_id", "position", "label", "variant", "cap_group"] + [f"p_{c}" for c in CLASSES])
        for r, p in zip(val.rows, pv):
            wr.writerow([r["scene_id"], r["position"], r["label"], r["variant"], r["cap_group"]] + [round(float(x), 6) for x in p])
    try:
        commit = subprocess.run(["git", "-C", P.HERE, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    run = {"model": "torchvision resnet18, ImageNet1K_V1 weights, 4-way head", "classes": CLASSES, "input_px": INPUT_PX,
           "model_size_px": P.MODEL_SIZE, "epochs": a.epochs, "batch": a.batch, "seed": a.seed, "device": str(device),
           "data": [{"dir": os.path.abspath(d), "crops_csv_sha256": sha256(os.path.join(d, "crops.csv"))} for d in a.data],
           "train_counts": dict(zip(CLASSES, counts.tolist())), "n_val": len(val),
           "recipe_sha256": sha256(a.recipe), "train_resnet_sha256": sha256(os.path.abspath(__file__)), "git_commit": commit,
           "aug": None if not a.aug else {"file": os.path.relpath(os.path.abspath(a.aug), P.HERE), "sha256": sha256(a.aug)},
           "torch": torch.__version__, "torchvision": torchvision.__version__, "python": platform.python_version(),
           "model_pt_sha256": sha256(os.path.join(a.out, "model.pt")), "thresholds_sha256": sha256(os.path.join(a.out, "thresholds.json")),
           "thresholds_d038_sha256": sha256(os.path.join(a.out, "thresholds_d038.json")),
           "wall_seconds": round(time.time() - t0), "real_images_used": 0}
    json.dump(run, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print(json.dumps({k: th[k] for k in ("t_defect", "val_defect_recall", "val_false_alarm_rate", "t_catch_every_val_defect",
                                         "val_false_alarm_rate_at_t_catch_every", "val_per_class_recall_after_rule")}, indent=1))


if __name__ == "__main__":
    main()
