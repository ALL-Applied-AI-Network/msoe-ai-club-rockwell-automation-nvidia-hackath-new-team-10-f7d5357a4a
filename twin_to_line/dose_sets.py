"""Real-example dose sets for the data-efficiency experiment (ResNet-18 real-only vs synthetic-only vs
synthetic + real). No training here, no torch: only a deterministic, auditable choice of WHICH real examples.

Rules (from the plan, replacing the pass-heavy reading of D014):
  * the dose N counts real DEFECT examples (missing_cap / crooked_cap / damaged_vial): 0, 5, 10, 20;
  * defects are stratified round-robin across the defect classes, so dose 5 is not five of one class;
  * each dose also gets real companions at a fixed ratio (default: as many 'pass' as defects), so the real
    data cannot teach "real-looking = pass";
  * sets are nested (5 is inside 10 is inside 20) and fixed by the seed before any model is trained;
  * one example = one placement unit (session, event, position): never two frames of the same event;
  * only rows tagged split=adapt are allowed in; test / holdout rows (runs 4-5, the Session A final-test
    portion) make the call fail instead of being silently dropped.

Pool CSV columns: session, event, position, label, split, path
    python dose_sets.py pool.csv [--seed 0] [--out dose_sets.json]
"""
import csv, json, random, sys
from collections import Counter

DEFECT_CLASSES = ("missing_cap", "crooked_cap", "damaged_vial")
DOSES = (0, 5, 10, 20)


class InsufficientDefects(ValueError):
    pass


def unit(r):
    return (str(r["session"]), str(r["event"]), r["position"])


def check_pool(pool):
    bad = [unit(r) for r in pool if r.get("split") != "adapt"]
    if bad:
        raise ValueError(f"{len(bad)} pool rows are not split=adapt (test/holdout must never enter a dose set), e.g. {bad[:3]}")
    dup = [u for u, n in Counter(map(unit, pool)).items() if n > 1]
    if dup:
        raise ValueError(f"duplicate placement units (near-duplicate frames?) e.g. {dup[:3]}")


def interleave(pool, classes, rng):
    """One ordered list: each class shuffled once, then taken round-robin. Dose N = its first N items,
    which makes every smaller dose a subset of every larger one."""
    lists = []
    for c in classes:
        rows = sorted((r for r in pool if r["label"] == c), key=unit)
        rng.shuffle(rows); lists.append(rows)
    out = []
    while any(lists):
        for l in lists:
            if l:
                out.append(l.pop(0))
    return out


def make_dose_sets(pool, doses=DOSES, seed=0, defect_classes=DEFECT_CLASSES, companions=None):
    """Returns {N: {"defects": [rows], "companions": {label: [rows]}, "counts": {label: n}}}."""
    companions = {"pass": 1.0} if companions is None else companions
    check_pool(pool)
    rng = random.Random(seed)
    order = interleave(pool, defect_classes, rng)
    comp_order = {c: interleave(pool, (c,), rng) for c in sorted(companions)}
    top = max(doses)
    if len(order) < top:
        have = dict(Counter(r["label"] for r in order))
        raise InsufficientDefects(f"dose {top} needs {top} real defect examples; the pool has {len(order)} {have}")
    out = {}
    for n in sorted(doses):
        comp = {}
        for c, ratio in companions.items():
            k = round(ratio * n)
            if len(comp_order[c]) < k:
                raise InsufficientDefects(f"dose {n} needs {k} real '{c}' companions; the pool has {len(comp_order[c])}")
            comp[c] = comp_order[c][:k]
        chosen = order[:n]
        counts = Counter(r["label"] for r in chosen)
        counts.update({c: len(v) for c, v in comp.items()})
        out[n] = {"defects": chosen, "companions": comp, "counts": {k: v for k, v in counts.items() if v}}
    return out


def manifest(sets, seed):
    return {"seed": seed, "doses": {str(n): {"counts": s["counts"],
                                             "units": [list(unit(r)) for r in s["defects"]],
                                             "companion_units": {c: [list(unit(r)) for r in v] for c, v in s["companions"].items()}}
                                    for n, s in sets.items()}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pool"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out")
    a = ap.parse_args()
    pool = list(csv.DictReader(open(a.pool, newline="")))
    try:
        sets = make_dose_sets(pool, seed=a.seed)
    except InsufficientDefects as e:
        sys.exit(f"cannot build dose sets: {e}")
    m = manifest(sets, a.seed)
    for n, s in m["doses"].items():
        print(f"dose {n:>2}: {s['counts']}")
    if a.out:
        json.dump(m, open(a.out, "w"), indent=1); print("->", a.out)
