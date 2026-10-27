"""1080 copies of DEV replay folders (runs 1-3 only) and a per-hole comparison of live sessions, to check the 1080 path
of live_inspect.py end to end on real caps.

    python replay_1080.py make results/live/replay_run1 results/live/replay_run1_1080
    python replay_1080.py compare --a <640 session dirs...> --b <1080 session dirs...> [--labels labels/dev_labels.csv]

make     every frame warped UP into a 1920x1080 frame with the inverse of the frozen transform (frame_warp_1080.json,
         INTER_CUBIC), saved under the same file name so frame ids, frame times and labels still join; outside the DEV
         640x480 view = 0. Stills keep their .jpg name (JPEG quality 100). Refused unless every frame is a DEV frame.
compare  event frames, then per (frame, position): final state, visual state, defect score; every difference listed.
         With --labels, eval_metrics on each side (all sessions of a side pooled).
"""
import argparse, json, os
import cv2, numpy as np
import real_data as R
import frame_warp as FW
from eval_metrics import evaluate, join_labels, load as load_csv
from live_inspect import PATH_1080, dataset_index, is_image, refusal


def warp_up(img, fw):
    """A 640x480 DEV-grid frame as the 1920x1080 frame it would come from: dst(x1080) = src(M x1080)."""
    return cv2.warpAffine(img, fw.M, fw.size_in, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)


def make(src, dst, fw=None):
    fw = fw or FW.FrameWarp(); ix = dataset_index()
    if ix is None:
        raise SystemExit("needs the dataset index to prove every frame is DEV (runs 1-3)")
    names = sorted(n for n in os.listdir(src) if is_image(n))
    for n in names:
        run = ix.run_of_source(n)
        if run is None or refusal(ix, n):
            raise R.ProtectedDataError(f"{n}: not a DEV dataset frame (run {run}); nothing written")
    os.makedirs(dst, exist_ok=True)
    for n in names:
        img = cv2.imread(os.path.join(src, n))
        if img is None or img.shape[1::-1] != fw.size_out:
            raise SystemExit(f"{n}: not a {fw.size_out[0]}x{fw.size_out[1]} frame")
        params = [cv2.IMWRITE_JPEG_QUALITY, 100] if n.lower().endswith((".jpg", ".jpeg")) else []
        cv2.imwrite(os.path.join(dst, n), warp_up(img, fw), params)
    print(f"{len(names)} frames {fw.size_out[0]}x{fw.size_out[1]} -> {fw.size_in[0]}x{fw.size_in[1]}: {dst}  (transform {fw.sha256[:12]})")


def session_rows(dirs):
    preds, trial = [], []
    for d in dirs:
        preds += load_csv(os.path.join(d, "predictions.csv")); trial += load_csv(os.path.join(d, "trial_log.csv"))
    return preds, trial


def compare(a_dirs, b_dirs, labels=None):
    pa, ta = session_rows(a_dirs); pb, tb = session_rows(b_dirs)
    ev = lambda t: [r["event_frame"] for r in t if r["status"] == "scored"]
    st = lambda t: sorted({r["status"] for r in t})
    out = {"events_a": len(ev(ta)), "events_b": len(ev(tb)), "same_event_frames": ev(ta) == ev(tb),
           "event_frames_only_a": sorted(set(ev(ta)) - set(ev(tb))), "event_frames_only_b": sorted(set(ev(tb)) - set(ev(ta))),
           "statuses_a": st(ta), "statuses_b": st(tb), "paths_b": sorted({r.get("path", "") for r in tb if r["status"] != "wrong_size"}),
           "same_status_every_frame": [(r["input_file"], r["status"]) for r in ta] == [(r["input_file"], r["status"]) for r in tb]}
    A = {(r["frame"], r["position"]): r for r in pa}; B = {(r["frame"], r["position"]): r for r in pb}
    common = sorted(set(A) & set(B)); diffs, ds = [], []
    for k in common:
        a, b = A[k], B[k]; ds.append(abs(float(a["defect_score"]) - float(b["defect_score"])))
        if (a["final_state"], a["predicted_state"]) != (b["final_state"], b["predicted_state"]):
            diffs.append({"frame": k[0], "position": k[1], "a": [a["predicted_state"], a["final_state"], a["defect_score"], a["cap_frac"], a["cap_offset"]],
                          "b": [b["predicted_state"], b["final_state"], b["defect_score"], b["cap_frac"], b["cap_offset"]]})
    ds = np.array(ds) if ds else np.zeros(1)
    out.update(holes_compared=len(common), holes_only_a=len(set(A) - set(B)), holes_only_b=len(set(B) - set(A)),
               state_differences=diffs, defect_score_abs_diff={"median": round(float(np.median(ds)), 4), "p99": round(float(np.percentile(ds, 99)), 4),
                                                               "max": round(float(ds.max()), 4)})
    if labels:
        lab = load_csv(labels)
        for side, p in (("a", pa), ("b", pb)):
            m = evaluate(join_labels(p, lab)[0]); c = m["counts"]
            out[f"metrics_{side}"] = {"recall": f"{c['defects'] - c['false_negatives']}/{c['defects']}",
                                      "false_positives": f"{c['false_positives']}/{c['non_defects']}",
                                      "defect_recall": m["defect_recall"], "false_positive_rate": m["false_positive_rate"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="1080 copies of DEV replays; compare live sessions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make"); mk.add_argument("src"); mk.add_argument("dst")
    cp = sub.add_parser("compare"); cp.add_argument("--a", nargs="+", required=True); cp.add_argument("--b", nargs="+", required=True)
    cp.add_argument("--labels"); cp.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.src, a.dst)
    else:
        res = compare(a.a, a.b, a.labels)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "state_differences"}, indent=1))
        print(f"{len(res['state_differences'])} per-hole state differences:")
        for d in res["state_differences"]:
            print(f"  {d['frame']} {d['position']}  a {d['a']}  b {d['b']}")
