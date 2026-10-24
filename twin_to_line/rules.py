"""Rule baseline inspector: the simplest visual rule, anchored on the twin-expected cap seat.

Features (per PositionCrop; the twin supplies cap_center, so nothing is learned about WHERE to look):
  cap_frac    yellow fraction in a disk of CAP_R px at the expected seated-cap center (registration.py HSV range)
  cap_offset  distance (px) from the expected cap center to the centroid of yellow inside a disk of OFF_R px
Decision (thresholds from rule_thresholds.json, fit on DEV labels only):
  cap_frac <  t_absent                               -> no_vial   ("no cap seen"; an uncapped vial is NOT
                                                                    separable from an empty hole by this rule)
  cap_frac >= t_seated and cap_offset <= t_offset    -> pass
  anything else                                      -> unknown  (partial / off-center cap -> UNSURE downstream)
It never outputs crooked_cap or missing_cap: on DEV, pale specular seated caps and displaced caps overlap.

    python rules.py --fit labels/dev_labels.csv     -> rule_thresholds.json (+ per-class feature ranges)
"""
import argparse, csv, json, os
from dataclasses import dataclass, field
import cv2, numpy as np
import positions as P

CAP_R, OFF_R = 11, 13
THRESHOLDS = os.path.join(P.HERE, "rule_thresholds.json")


@dataclass
class Observation:
    visual_state: str                    # pass | no_vial | unknown  (fill_state.VISUAL_STATES)
    score: float                         # defect score: 1 - cap_frac (higher = less cap where a cap should sit)
    features: dict = field(default_factory=dict)
    reason: str = ""


class Inspector:
    """Interface every inspector implements (rule now; ResNet-18 variants later)."""
    name = "inspector"

    def inspect(self, crop) -> Observation:
        raise NotImplementedError

    def inspect_batch(self, crops):
        """All 30 crops of a frame at once (a learned model runs them as one batch)."""
        return [self.inspect(c) for c in crops]

    def info(self):
        """Provenance for logs: {name, files: {relative path: sha256}, status}."""
        return {"name": self.name, "files": {}, "status": ""}


def _disk(shape, c, r):
    m = np.zeros(shape, np.uint8); cv2.circle(m, (int(round(c[0])), int(round(c[1]))), r, 1, -1)
    return m.astype(bool)


def features(crop):
    y = P.yellow_mask(crop.image) > 0
    frac = float(y[_disk(y.shape, crop.cap_center, CAP_R)].mean())
    ys, xs = np.nonzero(y & _disk(y.shape, crop.cap_center, OFF_R))
    off = float(np.hypot(xs.mean() - crop.cap_center[0], ys.mean() - crop.cap_center[1])) if len(xs) >= 20 else None
    return {"cap_frac": round(frac, 4), "cap_offset": None if off is None else round(off, 2)}


def decide(f, t):
    if f["cap_frac"] < t["t_absent"]:
        return "no_vial", "no cap seen at the seat"
    if f["cap_frac"] >= t["t_seated"] and f["cap_offset"] is not None and f["cap_offset"] <= t["t_offset"]:
        return "pass", ""
    return "unknown", f"partial/off-center cap (frac {f['cap_frac']:.2f}, offset {f['cap_offset']} px)"


class RuleInspector(Inspector):
    name = "rule_yellow_seat"

    def __init__(self, path=THRESHOLDS):
        self.path = path; self.t = json.load(open(path))["thresholds"]

    def inspect(self, crop):
        f = features(crop)
        state, why = decide(f, self.t)
        return Observation(state, round(1 - f["cap_frac"], 4), f, why)

    def info(self):
        import hashlib
        return {"name": self.name, "files": {os.path.relpath(self.path, P.HERE): hashlib.sha256(open(self.path, "rb").read()).hexdigest()},
                "status": "fitted on DEV draft labels: plumbing placeholder, not a scored arm"}


def fit(rows):
    """rows: dicts with ground_truth, cap_frac, cap_offset (usable labels only). Deterministic:
    t_absent = 0.8 x the smallest cap_frac of any labeled cap (so no labeled cap reads as 'no cap');
    (t_seated, t_offset) = the grid point that keeps the most labeled 'pass' caps while sending EVERY labeled
    displaced cap (crooked_cap) to unknown (catch every known defect); ties -> stricter thresholds."""
    caps = [r for r in rows if r["ground_truth"] in ("pass", "crooked_cap")]
    t_absent = round(0.8 * min(r["cap_frac"] for r in caps), 3)
    passes = [r for r in caps if r["ground_truth"] == "pass"]; bad = [r for r in caps if r["ground_truth"] == "crooked_cap"]
    best = None
    for ts in np.round(np.arange(0.20, 0.96, 0.02), 2):
        for to in np.round(np.arange(2.0, 8.01, 0.1), 1):
            t = {"t_absent": t_absent, "t_seated": float(ts), "t_offset": float(to)}
            if any(decide(r, t)[0] == "pass" for r in bad):
                continue
            kept = sum(decide(r, t)[0] == "pass" for r in passes)
            key = (kept, ts, -to)
            if best is None or key > best[0]:
                best = (key, t)
    return best[1]


def feature_rows(labels_path):
    import real_data as R
    from register_dev import load_source
    ix = R.Index(); pm = P.PositionMap(); out = []; cache = {}
    for L in csv.DictReader(open(labels_path)):
        if L["usable"] != "1":
            continue
        if L["frame"] not in cache:
            cache[L["frame"]] = {c.position: c for c in P.extract_crops(load_source(ix, L["frame"]), pm, L["frame"])}
        out.append({**L, **features(cache[L["frame"]][L["position"]])})
    return out, pm


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fit", required=True, help="DEV labels CSV"); a = ap.parse_args()
    rows, pm = feature_rows(a.fit)
    t = fit(rows)
    ranges = {}
    for g in sorted({r["ground_truth"] for r in rows}):
        v = [r for r in rows if r["ground_truth"] == g]
        fr = [r["cap_frac"] for r in v]; of = [r["cap_offset"] for r in v if r["cap_offset"] is not None]
        ranges[g] = {"n_observations": len(v), "n_physical_units": len({(r["run"], r["position"]) for r in v}),
                     "cap_frac": [min(fr), max(fr)], "cap_offset": [min(of), max(of)] if of else None}
    out = {"inspector": RuleInspector.name, "thresholds": t, "features": {"cap_frac_disk_px": CAP_R, "cap_offset_disk_px": OFF_R},
           "fit_on": {"labels": os.path.relpath(a.fit, P.HERE), "frames": sorted({r["frame"] for r in rows}),
                      "runs": sorted({r["run"] for r in rows}), "position_offset_px": list(pm.offset)},
           "per_class_ranges": ranges,
           "note": "DEV runs 1-3 only; in-sample fit on draft labels; ~5 physical displaced caps; not a performance claim"}
    json.dump(out, open(THRESHOLDS, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("thresholds", "per_class_ranges")}, indent=1))
