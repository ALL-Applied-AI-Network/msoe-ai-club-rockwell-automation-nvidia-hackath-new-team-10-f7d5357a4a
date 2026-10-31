"""D044 post-hoc diagnostic: the five sealed arms on the sealed stills with ONLY the arm check bypassed.

Never the sealed test: D043's once-only result stands. Reuses score_sealed.py's label lock check, image loading,
arms and Session settings by import; the one change is events.arm_pixels -> 0, which clears both the event gate
and inspect()'s arm override. No lock ledger: writes its own folder and refuses to overwrite it. Before any metric,
frames the sealed run scored must be predicted identically (event numbers aside).

    python posthoc_nogate.py FOLDER --labels L --labels-sha256 SHA [--out DIR] [--sealed-out DIR] [--device cpu]
"""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import eval_metrics as M
import events as E
import live_inspect as L
import score_sealed as S
from fill_state import LIVE_FILL

HERE = Path(__file__).resolve().parent
CAP_DEFECTS = {"crooked_cap", "missing_cap"}


def arm_clear(img, pmap, plate=None):
    return 0


def score(folder, labels, labels_sha256, out, device="cpu"):
    folder, labels, out = (Path(p).resolve() for p in (folder, labels, out))
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; never overwrite")
    label_bytes, images, ix = S.inputs(folder, labels, labels_sha256)   # sha256 check precedes any pixel read
    hashes = {name: S.digest(data) for name, data, _, _ in images}
    out.mkdir(parents=True)
    (out / "labels.locked.csv").write_bytes(label_bytes)
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "decision": "D044 post-hoc, not the sealed test",
              "change": "events.arm_pixels -> 0 (arm check bypassed); everything else as score_sealed.py",
              "source": str(folder), "labels_sha256": S.digest(label_bytes), "images_sha256": hashes,
              "arms": list(S.ARMS), "gate": "each", "mode": LIVE_FILL, "track": False, "device": device,
              "script_sha256": S.digest(Path(__file__).read_bytes()),
              "score_sealed_sha256": S.digest(Path(S.__file__).read_bytes()), "status": "started", "coverage": {}}
    S.write_json(out / "evaluation.json", record)
    original, E.arm_pixels = E.arm_pixels, arm_clear
    try:
        inspectors = S.load_inspectors(device)
        for arm in S.ARMS:
            session = L.Session(str(out / arm), source=f"posthoc_d044:{folder}", gate="each", mode=LIVE_FILL,
                                track=False, labels=str(out / "labels.locked.csv"), ix=ix,
                                inspector=inspectors[arm], quiet=True)
            session.write_info(posthoc="D044: arm check bypassed", sealed_arm=arm,
                               locked_labels_sha256=S.digest(label_bytes), input_sha256=hashes)
            L.run(session, ((str(folder / name), img.copy(), t) for name, _, img, t in images))
            record["coverage"][arm] = dict(session.counts)
            if session.stop_reason != "end of source" or set(session.counts) != {"scored"}:
                raise RuntimeError(f"{arm} incomplete: {session.stop_reason}, {dict(session.counts)}")
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        E.arm_pixels = original
        S.write_json(out / "evaluation.json", record)
    return record


def same_as_sealed(out, sealed_out):
    """Rows of every frame the sealed run scored must match field for field, except the session's event counter."""
    report = {}
    for arm in S.ARMS:
        sealed = {(r["frame"], r["position"]): r for r in M.load(Path(sealed_out) / arm / "predictions.csv")}
        post = {(r["frame"], r["position"]): r for r in M.load(Path(out) / arm / "predictions.csv")}
        diff = [k for k, r in sealed.items()
                if k not in post or {c: v for c, v in r.items() if c != "event"} != {c: v for c, v in post[k].items() if c != "event"}]
        report[arm] = {"sealed_frames": sorted({f for f, _ in sealed}), "rows": len(sealed), "differ": diff}
    return report


def metrics(out):
    labels = M.load(Path(out) / "labels.locked.csv")
    lean = {(r["frame"], r["position"]) for r in labels if "lean" in r.get("note", "")}
    report = {}
    for arm in S.ARMS:
        rows, unlabeled, unmatched = M.join_labels(M.load(Path(out) / arm / "predictions.csv"), labels)
        if unlabeled or unmatched:
            raise ValueError(f"{arm}: {unlabeled} unlabeled predictions, {len(unmatched)} labels without prediction")
        e = M.evaluate(rows)
        defect = lambda r: M.is_defect(r["ground_truth"], r["gt_sequence"])
        flag = M.flagged_state
        caps = M.sweep(rows, CAP_DEFECTS, {"pass"})
        report[arm] = {
            "n_by_class": dict(Counter(M.gt_class(r) for r in rows)),
            "misses": {c: [f"{Path(r['frame']).stem}:{r['position']}" for r in rows if M.gt_class(r) == c and not flag(r)]
                       for c in ("crooked_cap", "missing_cap", "missed_placement")},
            "false_alarms": dict(Counter(M.gt_class(r) for r in rows if not defect(r) and flag(r))),
            "unsure": dict(Counter(M.gt_class(r) for r in rows if r["final_state"] == "UNSURE")),
            "far_full_recall_caps_vs_pass": {"threshold": caps.get("threshold_full_recall"),
                                             "rate": caps.get("far_at_full_recall")},
            "far_full_recall_all_defects": e.get("false_alarm_rate_at_full_recall"),
            "setup_06_crooked": {r["position"]: r["final_state"] for r in rows
                                 if r["frame"].startswith("setup_06") and r["ground_truth"] == "crooked_cap"},
            "lean": {f"{Path(r['frame']).stem}:{r['position']}": r["final_state"] for r in rows
                     if (r["frame"], r["position"]) in lean},
            "counts": e["counts"], "per_class": e["per_class"], "escalated_share": e["escalated_share"]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--labels-sha256", required=True)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "posthoc_d044")
    parser.add_argument("--sealed-out", type=Path, default=HERE / "results" / "sealed_2026-10-27")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    record = score(args.folder, args.labels, args.labels_sha256, args.out, args.device)
    check = same_as_sealed(args.out, args.sealed_out)
    S.write_json(args.out / "sealed_frames_check.json", check)
    if any(c["differ"] for c in check.values()):
        raise SystemExit(f"STOP: post-hoc predictions differ from the sealed run on its scored frames; see {args.out}")
    S.write_json(args.out / "metrics.json", metrics(args.out))
    print(f"D044 POST-HOC COMPLETE (not the sealed test): {len(record['images_sha256'])} images, {len(S.ARMS)} arms, "
          f"sealed frames identical -> {args.out}")


if __name__ == "__main__":
    main()
