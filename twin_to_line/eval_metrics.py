"""Evaluation metrics for Twin-to-Line. No model, no training: scores a predictions CSV against ground truth.

Result format (one row per position per inspection EVENT, not per video frame):
  run, event, position, ground_truth, predicted_state, defect_score, gt_sequence, pred_sequence
    ground_truth / predicted_state: pass | missing_cap | crooked_cap | damaged_vial | no_vial
    gt_sequence / pred_sequence   : missed | waiting | (blank when a vial is present)
    defect_score                  : higher = more likely defective (blank if the method has none)
  optional: frame (source image of the event), final_state (PASS | DEFECT | MISSED | WAITING | UNSURE).
    When final_state is present it decides what was flagged: DEFECT / MISSED / UNSURE. UNSURE (and the visual
    state 'unknown') counts as flagged -- it goes to a person -- and is also reported as the escalated share.
A position is a DEFECT when its state is missing_cap / crooked_cap / damaged_vial, or no_vial with sequence
'missed'. 'no_vial' + 'waiting' is NOT a defect (a trailing empty position).

    python eval_metrics.py predictions.csv [threshold] [--labels labels.csv]
        --labels joins ground truth (run, event or frame, position, ground_truth, gt_sequence) onto predictions
    python eval_metrics.py --selftest
"""
import csv, sys
from collections import Counter, defaultdict

DEFECT_STATES = {"missing_cap", "crooked_cap", "damaged_vial"}

def is_defect(state, seq):
    return state in DEFECT_STATES or (state == "no_vial" and seq == "missed")

def gt_class(r):
    if r["ground_truth"] == "no_vial":
        return "missed_placement" if r["gt_sequence"] == "missed" else "waiting"
    return r["ground_truth"]

FLAGGED_FINAL = {"DEFECT", "MISSED", "UNSURE"}

def gt_final(r):
    """Ground truth as a process state."""
    if r["ground_truth"] == "no_vial":
        return {"missed": "MISSED", "waiting": "WAITING"}.get(r["gt_sequence"], "UNKNOWN_SEQUENCE")
    return "DEFECT" if r["ground_truth"] in DEFECT_STATES else "PASS"

def escalated(r):
    return r.get("final_state") == "UNSURE" or r["predicted_state"] == "unknown"

def flagged_state(r):
    if r.get("final_state"):
        return r["final_state"] in FLAGGED_FINAL
    return is_defect(r["predicted_state"], r.get("pred_sequence", "")) or r["predicted_state"] == "unknown"

def check_units(rows):
    """One row per (run, event, position): near-duplicate video frames must never become extra samples."""
    seen, frames = set(), defaultdict(set)
    for r in rows:
        k = (str(r["run"]), str(r["event"]), r["position"])
        if k in seen:
            raise ValueError(f"duplicate row for run {k[0]} event {k[1]} {k[2]}: one row per position per placement event")
        seen.add(k)
        if r.get("frame"):
            frames[k[:2]].add(r["frame"])
    multi = {k: v for k, v in frames.items() if len(v) > 1}
    if multi:
        raise ValueError(f"events built from more than one frame: {sorted(multi)[:3]}")

def evaluate(rows, threshold=None):
    """rows: dicts with the columns above. If threshold is given and defect_score is present,
    a row counts as flagged when score >= threshold OR the predicted state/sequence is a defect."""
    check_units(rows)
    def flagged(r):
        f = flagged_state(r)
        if threshold is not None and r.get("defect_score") not in (None, ""):
            f = f or float(r["defect_score"]) >= threshold
        return f
    out = {"n_positions": len(rows), "n_events": len({(r["run"], r["event"]) for r in rows})}
    gt_def = [r for r in rows if is_defect(r["ground_truth"], r["gt_sequence"])]
    gt_ok = [r for r in rows if not is_defect(r["ground_truth"], r["gt_sequence"])]
    fn = [r for r in gt_def if not flagged(r)]; fp = [r for r in gt_ok if flagged(r)]
    out["counts"] = {"defects": len(gt_def), "non_defects": len(gt_ok), "false_negatives": len(fn), "false_positives": len(fp),
                     "by_ground_truth_class": dict(Counter(gt_class(r) for r in rows))}
    out["defect_recall"] = round(1 - len(fn) / len(gt_def), 4) if gt_def else None
    out["false_positive_rate"] = round(len(fp) / len(gt_ok), 4) if gt_ok else None
    per = defaultdict(lambda: [0, 0, 0])            # class -> [n, flagged as any defect, exact class right]
    for r in gt_def:
        c = gt_class(r); per[c][0] += 1; per[c][1] += flagged(r)
        exact = (r["predicted_state"] == r["ground_truth"] and (c != "missed_placement" or r.get("pred_sequence") == "missed"))
        per[c][2] += exact
    out["per_class"] = {c: {"n": n, "detection_recall": round(d / n, 4), "exact_class_recall": round(e / n, 4), "false_negatives": n - d}
                        for c, (n, d, e) in sorted(per.items())}
    trailing = [r for r in rows if gt_class(r) == "waiting"]
    out["trailing_empty_false_alarms"] = sum(flagged(r) for r in trailing)
    empties = [r for r in rows if r["ground_truth"] == "no_vial" and r["predicted_state"] == "no_vial"]
    out["sequence_accuracy_on_detected_empties"] = (round(sum(r["gt_sequence"] == r.get("pred_sequence") for r in empties) / len(empties), 4)
                                                     if empties else None)
    scored = [r for r in rows if r.get("defect_score") not in (None, "")]
    if scored and gt_def and all(r.get("defect_score") not in (None, "") for r in gt_def):
        t_all = min(float(r["defect_score"]) for r in gt_def)              # MAIC spec: threshold that catches every known defect
        fa = sum(float(r["defect_score"]) >= t_all for r in gt_ok if r.get("defect_score") not in (None, ""))
        out["false_alarm_rate_at_full_recall"] = {"threshold": t_all, "rate": round(fa / len(gt_ok), 4) if gt_ok else None}
    out["escalated_share"] = round(sum(map(escalated, rows)) / len(rows), 4) if rows else None
    out["threshold_used"] = threshold
    return out

def confusion(rows, gt=lambda r: r["ground_truth"], pred=lambda r: r["predicted_state"]):
    """{ground truth: {prediction: count}}. Default: visual classes. Process states:
    confusion(rows, gt_final, lambda r: r["final_state"])."""
    m = defaultdict(Counter)
    for r in rows:
        m[gt(r)][pred(r)] += 1
    return {g: dict(c) for g, c in sorted(m.items())}

def format_confusion(m):
    cols = sorted({p for c in m.values() for p in c})
    w = max([len(c) for c in cols] + [len(g) for g in m] + [5])
    lines = ["gt \\ pred".ljust(w) + " " + " ".join(c.rjust(w) for c in cols)]
    lines += [g.ljust(w) + " " + " ".join(str(c.get(p, 0)).rjust(w) for p in cols) for g, c in m.items()]
    return "\n".join(lines)

def sweep(rows, positives, negatives, score_key="defect_score"):
    """Threshold sweep for one task, e.g. positives={'missing_cap'}, negatives={'pass'} (gt_class names, so
    'missed_placement' / 'waiting' for empty holes). Rows of other classes are ignored, so trailing empties do
    not pollute a cap-presence threshold. Returns per-threshold recall / false-alarm rate and the FAR at the
    lowest threshold that still catches every positive (MAIC spec)."""
    pos = [r for r in rows if gt_class(r) in positives]; neg = [r for r in rows if gt_class(r) in negatives]
    missing = [r for r in pos + neg if r.get(score_key) in (None, "")]
    if not pos or not neg or missing:
        return {"n_pos": len(pos), "n_neg": len(neg), "rows_without_score": len(missing), "table": [],
                "note": "need at least one positive, one negative and a score on every row"}
    sp = [float(r[score_key]) for r in pos]; sn = [float(r[score_key]) for r in neg]
    table = []
    for t in sorted(set(sp + sn), reverse=True):
        tp = sum(s >= t for s in sp); fp = sum(s >= t for s in sn)
        table.append({"threshold": t, "recall": round(tp / len(sp), 4), "false_alarm_rate": round(fp / len(sn), 4),
                      "tp": tp, "fn": len(sp) - tp, "fp": fp})
    t_all = min(sp)
    return {"n_pos": len(pos), "n_neg": len(neg), "threshold_full_recall": t_all,
            "far_at_full_recall": round(sum(s >= t_all for s in sn) / len(sn), 4),
            "separation_margin": round(min(sp) - max(sn), 4), "table": table}

def join_labels(preds, labels):
    """Attach ground_truth / gt_sequence from a labels CSV. Joins on (frame, position) when both files carry a
    frame column, else on (run, event, position). Labels marked usable=0 (ambiguous) are dropped, not guessed.
    Returns (joined rows, unlabeled prediction count, label keys with no prediction)."""
    labels = [r for r in labels if r.get("usable", "1") != "0"]
    by_frame = all(r.get("frame") for r in preds) and all(r.get("frame") for r in labels)
    key = (lambda r: (r["frame"], r["position"])) if by_frame else (lambda r: (str(r["run"]), str(r["event"]), r["position"]))
    lab = {}
    for r in labels:
        if key(r) in lab:
            raise ValueError(f"duplicate label for {key(r)}")
        lab[key(r)] = r
    out, unlabeled = [], 0
    for r in preds:
        g = lab.get(key(r))
        if g is None:
            unlabeled += 1; continue
        out.append({**r, "ground_truth": g["ground_truth"], "gt_sequence": g.get("gt_sequence", "")})
    unmatched = sorted(set(lab) - {key(r) for r in preds})
    return out, unlabeled, unmatched

def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def selftest():
    R = lambda run, ev, pos, gt, pred, score, gseq="", pseq="": dict(run=run, event=ev, position=pos, ground_truth=gt,
            predicted_state=pred, defect_score=score, gt_sequence=gseq, pred_sequence=pseq)
    rows = [
        R("5", 1, "G01", "pass", "pass", 0.10), R("5", 1, "G02", "pass", "missing_cap", 0.70),      # FP
        R("5", 1, "G03", "missing_cap", "missing_cap", 0.90), R("5", 1, "G04", "crooked_cap", "pass", 0.40),  # FN
        R("5", 1, "G05", "crooked_cap", "crooked_cap", 0.80),
        R("5", 1, "G06", "no_vial", "no_vial", 0.95, "missed", "missed"),                            # missed placement caught
        R("5", 1, "G07", "no_vial", "no_vial", 0.05, "waiting", "waiting"),                          # trailing empty, fine
        R("5", 1, "G08", "no_vial", "no_vial", 0.60, "waiting", "missed"),                           # trailing-empty false alarm
        R("5", 2, "G01", "pass", "pass", 0.20), R("5", 2, "G02", "missing_cap", "crooked_cap", 0.85), # detected, wrong class
    ]
    m = evaluate(rows)
    assert m["counts"] == {"defects": 5, "non_defects": 5, "false_negatives": 1, "false_positives": 2,
                           "by_ground_truth_class": {"pass": 3, "missing_cap": 2, "crooked_cap": 2, "missed_placement": 1, "waiting": 2}}, m["counts"]
    assert m["defect_recall"] == 0.8 and m["false_positive_rate"] == 0.4
    assert m["per_class"]["missing_cap"] == {"n": 2, "detection_recall": 1.0, "exact_class_recall": 0.5, "false_negatives": 0}
    assert m["per_class"]["crooked_cap"] == {"n": 2, "detection_recall": 0.5, "exact_class_recall": 0.5, "false_negatives": 1}
    assert m["per_class"]["missed_placement"]["detection_recall"] == 1.0
    assert m["trailing_empty_false_alarms"] == 1
    assert m["sequence_accuracy_on_detected_empties"] == round(2 / 3, 4)
    # every defect caught needs threshold 0.40 (G04); non-defects scoring >= 0.40: G02 0.70, G08 0.60 -> 2/5
    assert m["false_alarm_rate_at_full_recall"] == {"threshold": 0.40, "rate": 0.4}
    m2 = evaluate(rows, threshold=0.5)        # score >= 0.5 also flags; G04 (0.40) stays a FN
    assert m2["counts"]["false_negatives"] == 1 and m2["counts"]["false_positives"] == 2
    m3 = evaluate(rows, threshold=0.35)       # now G04 is caught; G01 run 5 ev 1 (0.10) and ev 2 (0.20) stay clean
    assert m3["counts"]["false_negatives"] == 0 and m3["defect_recall"] == 1.0
    # UNSURE goes to a person: flagged + escalated; final_state overrides the visual/sequence columns
    u = [dict(r, final_state=f) for r, f in zip(rows[:3], ("PASS", "UNSURE", "DEFECT"))]
    mu = evaluate(u)
    assert mu["counts"]["false_positives"] == 1 and mu["escalated_share"] == round(1 / 3, 4)
    assert confusion(rows)["crooked_cap"] == {"pass": 1, "crooked_cap": 1}
    sw = sweep(rows, {"missing_cap"}, {"pass"})                # missing caps 0.90, 0.85 vs passes 0.10, 0.70, 0.20
    assert sw["threshold_full_recall"] == 0.85 and sw["far_at_full_recall"] == 0.0 and sw["separation_margin"] == 0.15
    try:
        evaluate(rows + rows[:1]); raise AssertionError("duplicate event row not caught")
    except ValueError:
        pass
    print("selftest passed: recall, FP rate, per-class, trailing empties, sequence accuracy, FAR@full-recall, thresholds,"
          " UNSURE/escalation, confusion, sweep, duplicate-event guard")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    else:
        import argparse, json
        ap = argparse.ArgumentParser(description="Score per-event predictions against ground truth.")
        ap.add_argument("predictions"); ap.add_argument("threshold", nargs="?", type=float)
        ap.add_argument("--labels", help="ground-truth CSV to join onto the predictions")
        a = ap.parse_args()
        rows = load(a.predictions)
        if a.labels:
            rows, unlabeled, unmatched = join_labels(rows, load(a.labels))
            print(f"joined {len(rows)} labeled rows; {unlabeled} prediction rows have no label; {len(unmatched)} labels have no prediction")
        print(json.dumps(evaluate(rows, a.threshold), indent=1))
        print("\nvisual confusion (rows = ground truth):\n" + format_confusion(confusion(rows)))
        if all(r.get("final_state") for r in rows):
            print("\nprocess-state confusion:\n" + format_confusion(confusion(rows, gt_final, lambda r: r["final_state"])))
        for c in sorted(DEFECT_STATES & {r["ground_truth"] for r in rows}):
            sw = sweep(rows, {c}, {"pass"})
            print(f"\nsweep {c} vs pass:", json.dumps({k: v for k, v in sw.items() if k != "table"}))
