"""D040 check render: the 6 train_v1 scenes of configs/v2_check.json re-rendered with work2 and the train_v2 floor block,
judged against the frozen rules there. Runs where the renders are (instance or Mac); numpy + OpenCV (+ this repo).

  E1-E3  empty holes inside the real DEV ranges: core L* median, lit-floor (90th-percentile) L* median, b* median
  U1-U2  uncapped vials no worse than the same scenes in train_v1: mouth L* and b* medians within the tolerance
  U3     uncapped lit-floor median above the empty holes' 90th percentile (the two classes stay apart)
All six -> PASS (exit 0). Otherwise FAIL (exit 1) and, unless this was the last try, the next floor block from the
pre-declared rule (value factor from the empty lit floor, capped by the core L*; blue factor from the empty b*); --update writes it into the
given configs. Empty holes and uncapped vials are printed side by side with train_v1 and the real ranges.

    python v2_check.py --renders RENDERS --scenes SCENES --try N [--update configs/train_v2.json configs/tilt_train_v2.json]
"""
import argparse, json, os, sys
import cv2, numpy as np
import positions as P
from v2_analysis import features

HERE = P.HERE
CRITERIA = os.path.join(HERE, "configs", "v2_check.json")


def lstar_to_y(L):
    return ((L + 16) / 116) ** 3


def measure(renders, scenes, indices):
    """Feature rows of every empty-hole and uncapped-vial crop of the check scenes (labels from their labels.json)."""
    pm0 = P.PositionMap(offset=(0, 0)); out = {"empty": [], "uncapped": []}; seen = []
    for sid in sorted(os.listdir(scenes)):
        lab_path = os.path.join(scenes, sid, "labels.json")
        if not os.path.isfile(lab_path):
            continue
        lab = json.load(open(lab_path))
        if lab["index"] not in indices:
            continue
        img = cv2.imread(os.path.join(renders, sid + ".png"))
        if img is None:
            raise FileNotFoundError(os.path.join(renders, sid + ".png"))
        seen.append(lab["index"]); state = {o["position"]: o["state"] for o in lab["positions"]}
        for c in P.extract_crops(img, pm0, sid):
            k = {"no_vial": "empty", "missing_cap": "uncapped"}.get(state[c.position])
            if k:
                g = {"cap": np.array(c.cap_center), "hole": np.array(c.hole_center), "mouth": np.array(c.tube_top_center)}
                out[k].append(features(c.image, g))
    if sorted(seen) != sorted(indices):
        raise ValueError(f"check scenes {sorted(seen)} != the frozen indices {sorted(indices)}")
    return out


def judge(F, crit):
    r, base = crit["rules"], crit["train_v1_baseline_same_scenes"]
    med = lambda k, f: float(np.median([x[f] for x in F[k]]))
    m = {"empty": {f: med("empty", f) for f in ("hole_L", "hole_Lp90", "hole_b")},
         "uncapped": {f: med("uncapped", f) for f in ("mouth_L", "mouth_b", "hole_Lp90")}}
    m["empty"]["hole_Lp90_p90"] = float(np.percentile([x["hole_Lp90"] for x in F["empty"]], 90))
    inside = lambda v, lohi: lohi[0] <= v <= lohi[1]
    tol = r["U12_uncapped_no_worse_tolerance"]
    checks = {"E1 empty core L*": inside(m["empty"]["hole_L"], r["E1_empty_core_L_median"]),
              "E2 empty lit floor L*": inside(m["empty"]["hole_Lp90"], r["E2_empty_lit_floor_L_median"]),
              "E3 empty b*": inside(m["empty"]["hole_b"], r["E3_empty_b_median"]),
              "U1 uncapped mouth L* no worse": abs(m["uncapped"]["mouth_L"] - base["uncapped"]["mouth_L"]) <= tol,
              "U2 uncapped mouth b* no worse": abs(m["uncapped"]["mouth_b"] - base["uncapped"]["mouth_b"]) <= tol,
              "U3 uncapped apart from empty": m["uncapped"]["hole_Lp90"] > m["empty"]["hole_Lp90_p90"]}
    return m, checks


def next_floor(m, crit):
    c = crit["calibration"]
    f = min(lstar_to_y(53.0) / lstar_to_y(m["empty"]["hole_Lp90"]), lstar_to_y(32.0) / lstar_to_y(m["empty"]["hole_L"]))
    f = float(np.clip(f, 0.25, 4.0))
    b = m["empty"]["hole_b"]; blue = 0.85 if b < crit["rules"]["E3_empty_b_median"][0] else 1.15 if b > crit["rules"]["E3_empty_b_median"][1] else 1.0
    return {"value_factor": round(f, 4), "blue_factor": blue, "rule": c["f"] + "; " + c["blue"]}


def update_configs(paths, step, try_n):
    for p in paths:
        cfg = json.load(open(p)); fb = cfg["floor"]
        fb["val_scale"] = [round(fb["val_scale"][0] * step["value_factor"], 6), round(fb["val_scale"][1] * step["value_factor"], 6)]
        fb["blue_scale"] = round(fb["blue_scale"] * step["blue_factor"], 6)
        cfg["purpose"] = cfg["purpose"].split(" [check try")[0] + f" [check try {try_n + 1} floor]"
        json.dump(cfg, open(p, "w"), indent=1)
        print(f"updated {p}: floor {fb}")


def main():
    ap = argparse.ArgumentParser(description="D040 check render: work2 empty holes and uncapped vials vs the frozen rules")
    ap.add_argument("--renders", required=True); ap.add_argument("--scenes", required=True)
    ap.add_argument("--try", dest="try_n", type=int, required=True); ap.add_argument("--criteria", default=CRITERIA)
    ap.add_argument("--update", nargs="*", default=[]); ap.add_argument("--out")
    a = ap.parse_args()
    crit = json.load(open(a.criteria)); base = crit["train_v1_baseline_same_scenes"]; real = crit["real_dev_report_only"]
    F = measure(a.renders, a.scenes, crit["indices"]); m, checks = judge(F, crit)
    print(f"check try {a.try_n}: {len(F['empty'])} empty-hole crops, {len(F['uncapped'])} uncapped-vial crops")
    print(f"{'':28s} {'work2 (this try)':>17s} {'train_v1 same scenes':>21s} {'rule':>16s}")
    r = crit["rules"]
    for name, now, was, rule in (("empty core L*", m["empty"]["hole_L"], base["empty"]["hole_L"], r["E1_empty_core_L_median"]),
                                 ("empty lit floor L*", m["empty"]["hole_Lp90"], base["empty"]["hole_Lp90"], r["E2_empty_lit_floor_L_median"]),
                                 ("empty b*", m["empty"]["hole_b"], base["empty"]["hole_b"], r["E3_empty_b_median"]),
                                 ("uncapped mouth L*", m["uncapped"]["mouth_L"], base["uncapped"]["mouth_L"], f"±{r['U12_uncapped_no_worse_tolerance']:g} of train_v1"),
                                 ("uncapped mouth b*", m["uncapped"]["mouth_b"], base["uncapped"]["mouth_b"], f"±{r['U12_uncapped_no_worse_tolerance']:g} of train_v1"),
                                 ("uncapped lit floor L*", m["uncapped"]["hole_Lp90"], base["uncapped"]["hole_Lp90"], f"> {m['empty']['hole_Lp90_p90']:.1f}")):
        print(f"{name:28s} {now:17.1f} {was:21.1f} {str(rule):>16s}")
    print(f"real DEV (report only): empty {real['empty_all_p10_p90']}; uncapped mouth L* {real['uncapped']['mouth_L']}, "
          f"b* {real['uncapped']['mouth_b']} ({real['uncapped']['source']})")
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    ok = all(checks.values()); res = {"try": a.try_n, "measured": m, "checks": checks, "pass": ok,
                                       "n": {k: len(v) for k, v in F.items()}}
    if not ok and a.try_n < crit["calibration"]["max_tries"]:
        res["next_floor"] = step = next_floor(m, crit)
        print(f"next try: value x{step['value_factor']}, blue x{step['blue_factor']}")
        if a.update:
            update_configs(a.update, step, a.try_n)
    elif not ok:
        print(f"FAIL after {a.try_n} tries: no batch (D040)")
    json.dump(res, open(a.out or os.path.join(os.path.dirname(os.path.abspath(a.renders)), f"v2_check_try{a.try_n}.json"), "w"), indent=1)
    print("CHECK", "PASS: render the batch" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
