"""G3 go/no-go on DEV runs 1-3 (NEXT_SESSION_PLAN step 8): the synthetic-only ResNet-18 through the live pipeline.

    python g3_check.py [--model-dir DIR] [--labels labels/dev_labels.csv] [--out DIR]
    python g3_check.py --rule-only --out DIR      # plumbing check: rule baseline only, the model is not loaded
    python g3_check.py --rule-only --rule-thresholds rule_thresholds_synth.json --out DIR  # D033 arm B, once

Fixed before the first run (Oct 23, D038):
  pipeline   one frame per placement event (results/prototype/events/events_run{1,2,3}_*.csv, events.py), the runtime
             position map with the DEV offset, inspect_frame.inspect() (arm check, batch inspector, fill_state):
             LIVE_FILL, frontier WAITING, no SequenceTracker (live_inspect.py defaults)
  threshold  the D038 threshold (thresholds_d038.json) decides; the pre-registered 99.5 % threshold is run and reported
             next to it from the same model
  scoring    eval_metrics on labels/dev_labels.csv (usable rows; draft labels, DEV only). Defect observation = missing /
             crooked cap, or no_vial with sequence missed; flagged = final state DEFECT / MISSED / UNSURE
  G3         go if defect recall >= 0.90 and false-positive rate <= 0.05 at the D038 threshold. A development signal
             only (D033): no threshold or model change follows from it without Noor's OK, logged as a new decision.
Also reported: the rule baseline through the same pipeline (its thresholds were fitted on these DEV labels, so it is a
reference, not a comparison). Runs once per model: refuses if OUT already holds g3_report.json. Runs 4-5 are refused.
Outputs (gitignored): predictions_<arm>.csv (eval_metrics schema), probabilities.csv, annotated/<event>.png for the
D038 arm, g3_report.json.
"""
import argparse, csv, hashlib, json, os, subprocess, time
from datetime import datetime, timezone
import cv2, numpy as np
import positions as P
import real_data as R
from eval_metrics import confusion, evaluate, gt_final, join_labels, load as load_csv
from fill_state import LIVE_FILL
from inspect_frame import annotate, inspect, load as load_frame, prediction_rows, safe
from rules import RuleInspector

RUNS = ("1", "2", "3")
EVENTS = os.path.join(P.HERE, "results", "prototype", "events")
LABELS = os.path.join(P.HERE, "labels", "dev_labels.csv")
RECALL_MIN, FPR_MAX = 0.90, 0.05


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def event_jobs():
    jobs, files = [], {}
    for run in RUNS:
        R.check_allowed(run)
        names = sorted(n for n in os.listdir(EVENTS) if n.startswith(f"events_run{run}_") and n.endswith(".csv"))
        if len(names) != 1:
            raise FileNotFoundError(f"expected one events_run{run}_*.csv in {EVENTS}, found {names}")
        files[names[0]] = sha256(os.path.join(EVENTS, names[0]))
        jobs += [(run, int(r["event"]), r["frame"]) for r in sorted(csv.DictReader(open(os.path.join(EVENTS, names[0]))),
                                                                    key=lambda r: int(r["event"]))]
    return jobs, files


def predict(jobs, frames, insp, pm, plate):
    rows, results = [], []
    for (run, event, src), img in zip(jobs, frames):
        res = inspect(img, src, pm, insp, plate=plate, mode=LIVE_FILL, frontier="waiting")
        rows += prediction_rows(run, event, res); results.append(res)
    return rows, results


def score(rows, labels):
    joined, unlabeled, unmatched = join_labels(rows, labels)
    m = evaluate(joined)
    m["join"] = {"labeled_rows": len(joined), "prediction_rows_without_label": unlabeled, "labels_without_prediction": len(unmatched)}
    m["visual_confusion"] = confusion(joined)
    m["process_confusion"] = confusion(joined, gt_final, lambda r: r["final_state"])
    m["misses"] = [{k: r[k] for k in ("run", "event", "position", "ground_truth", "gt_sequence", "predicted_state", "final_state",
                                      "defect_score")} for r in joined
                   if gt_final(r) in ("DEFECT", "MISSED") and r["final_state"] not in ("DEFECT", "MISSED", "UNSURE")]
    m["false_alarms"] = [{k: r[k] for k in ("run", "event", "position", "ground_truth", "gt_sequence", "predicted_state",
                                            "final_state", "defect_score")} for r in joined
                         if gt_final(r) in ("PASS", "WAITING") and r["final_state"] in ("DEFECT", "MISSED", "UNSURE")]
    m["meets_g3_criteria"] = m["defect_recall"] >= RECALL_MIN and m["false_positive_rate"] <= FPR_MAX
    return m


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def git_state():
    try:
        head = subprocess.run(["git", "-C", P.HERE, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", P.HERE, "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout.split("\n")
        return head, sorted(l[3:] for l in dirty if l.strip())
    except Exception:
        return "unknown", []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir"); ap.add_argument("--labels", default=LABELS); ap.add_argument("--out")
    ap.add_argument("--rule-only", action="store_true", help="plumbing check without the model")
    ap.add_argument("--rule-thresholds", help="explicit rule threshold file; requires --rule-only and a new --out")
    a = ap.parse_args()
    if a.rule_thresholds and not (a.rule_only and a.out):
        ap.error("--rule-thresholds requires --rule-only and --out")
    from resnet_inspector import MODEL_DIR, ResNetInspector
    model_dir = a.model_dir or MODEL_DIR
    out = a.out or os.path.join(P.HERE, "results", "g3", os.path.basename(os.path.normpath(model_dir)))
    if os.path.exists(os.path.join(out, "g3_report.json")):
        raise SystemExit(f"{out} already holds a G3 report: G3 runs once per model (a rerun needs Noor's OK and a new --out)")
    os.makedirs(out, exist_ok=True)
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    jobs, event_files = event_jobs()
    ix = R.Index(); pm = P.PositionMap(); plate = pm.plate_mask()
    frames = [load_frame(ix, src) for _, _, src in jobs]
    labels = load_csv(a.labels)
    report = {"what": "G3 go/no-go, DEV runs 1-3, development signal only (D033, D038)", "started_utc": started,
              "pipeline": {"events": event_files, "n_events": len(jobs), "mode": LIVE_FILL, "frontier": "waiting", "tracker": None,
                           "position_map_sha256": sha256(P.MAP_PATH), "registration_sha256": sha256(P.REG_PATH),
                           "offset_px": list(pm.offset)},
              "labels": {"file": os.path.relpath(a.labels, P.HERE), "sha256": sha256(a.labels), "status": "draft, DEV only"},
              "arms": {}}
    rule = RuleInspector(a.rule_thresholds) if a.rule_thresholds else RuleInspector()
    rule_info = rule.info()
    rule_name, rule_note = "rule_baseline", "thresholds fitted on these DEV labels: reference only"
    if a.rule_thresholds:
        with open(a.rule_thresholds) as f:
            metadata = json.load(f)
        rule_name = "rule_synthetic" if metadata.get("fit_on", {}).get("split") == "val" else "rule_custom"
        rule_note = metadata.get("note", "explicitly selected rule thresholds")
        rule_info["status"] = rule_note
    rows, _ = predict(jobs, frames, rule, pm, plate)
    write_csv(os.path.join(out, "predictions_rule.csv"), rows)
    report["arms"][rule_name] = {"inspector": rule_info, "note": rule_note, **score(rows, labels)}
    if a.rule_only:
        m = report["arms"][rule_name]
        report["g3"] = {"decided_by": rule_name, "go": m["meets_g3_criteria"],
                        "rule": f"defect recall >= {RECALL_MIN} and false-positive rate <= {FPR_MAX}",
                        "defect_recall": m["defect_recall"], "false_positive_rate": m["false_positive_rate"]}
    if not a.rule_only:
        d038 = ResNetInspector(model_dir, "d038")
        pre = ResNetInspector(model_dir, "preregistered", device=str(d038.device), model=d038.model)
        rows_d, res_d = predict(jobs, frames, d038, pm, plate)
        rows_p, res_p = predict(jobs, frames, pre, pm, plate)
        pd = np.array([[p[f"p_{c}"] for c in ("pass", "missing_cap", "crooked_cap", "no_vial")] for r in res_d for p in r["positions"]])
        pp = np.array([[p[f"p_{c}"] for c in ("pass", "missing_cap", "crooked_cap", "no_vial")] for r in res_p for p in r["positions"]])
        write_csv(os.path.join(out, "predictions_resnet_d038.csv"), rows_d)
        write_csv(os.path.join(out, "predictions_resnet_preregistered.csv"), rows_p)
        write_csv(os.path.join(out, "probabilities.csv"),
                  [{"run": run, "event": ev, "frame": src, "position": p["position"], "arm_px": r["arm_px"],
                    **{k: p[k] for k in ("p_pass", "p_missing_cap", "p_crooked_cap", "p_no_vial")}, "defect_score": p["defect_score"]}
                   for (run, ev, src), r in zip(jobs, res_d) for p in r["positions"]])
        os.makedirs(os.path.join(out, "annotated"), exist_ok=True)
        for (run, ev, src), img, r in zip(jobs, frames, res_d):
            cv2.imwrite(os.path.join(out, "annotated", f"run{run}_e{ev:02d}_{safe(src)}.png"), annotate(img, r, pm))
        report["arms"]["resnet_d038"] = {"inspector": d038.info(), "device": str(d038.device), **score(rows_d, labels)}
        report["arms"]["resnet_preregistered"] = {"inspector": pre.info(), "device": str(pre.device),
                                                  "note": "reported next to D038; does not decide G3", **score(rows_p, labels)}
        report["probability_passes_max_abs_diff"] = float(np.abs(pd - pp).max())
        report["g3"] = {"decided_by": "resnet_d038", "go": report["arms"]["resnet_d038"]["meets_g3_criteria"],
                        "rule": f"defect recall >= {RECALL_MIN} and false-positive rate <= {FPR_MAX}",
                        "defect_recall": report["arms"]["resnet_d038"]["defect_recall"],
                        "false_positive_rate": report["arms"]["resnet_d038"]["false_positive_rate"]}
    head, dirty = git_state()
    report.update({"git_commit": head, "uncommitted_files": dirty, "seconds": round(time.time() - t0, 1),
                   "g3_check_sha256": sha256(os.path.abspath(__file__)), "real_images": "DEV runs 1-3 event frames only"})
    json.dump(report, open(os.path.join(out, "g3_report.json"), "w"), indent=1)
    for name, m in report["arms"].items():
        print(f"{name:22s} recall {m['defect_recall']:.4f} ({m['counts']['defects'] - m['counts']['false_negatives']}/{m['counts']['defects']})"
              f"  FPR {m['false_positive_rate']:.4f} ({m['counts']['false_positives']}/{m['counts']['non_defects']})"
              f"  escalated {m['escalated_share']}  meets G3 criteria: {m['meets_g3_criteria']}")
    if "g3" in report:
        print(f"G3 (decided by {report['g3']['decided_by']}): {'GO' if report['g3']['go'] else 'NO-GO'}")
    print(f"-> {out}/g3_report.json")


if __name__ == "__main__":
    main()
