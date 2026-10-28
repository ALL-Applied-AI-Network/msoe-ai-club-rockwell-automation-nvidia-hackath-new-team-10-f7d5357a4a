"""End-to-end real-frame inspection: frame -> twin G01-G30 map (+ DEV offset) -> 30 crops -> inspector ->
fill-order reasoning -> PASS / DEFECT / MISSED / WAITING / UNSURE -> annotated image + JSON / CSV.

    python inspect_frame.py SOURCE... [--mode live_fill|final_inspection] [--frontier waiting|unsure] [--track] [--debug] [--out DIR]
        [--inspector rule|resnet|staging5] [--threshold d038|preregistered] [--model-dir DIR]
    python inspect_frame.py --run 1 --events [--track | --k-from-events]

SOURCE = a still name/path or a DEV video frame id (cap_seal_video_1.mp4@t0612.0). Runs 4-5 are refused.
Default: MISSED / WAITING from occupancy only (fill_state.py, D021); the line has no placement count.
--events reads results/prototype/events/events_run{N}_*.csv (events.py): one frame per placement event.
--track: feed the frames in order through fill_state.SequenceTracker (--persist frames, default 1 for --events
  because each event frame is already a stable segment, else 2).
--mode live_fill (default): the fixture may still be filling; trailing empties are WAITING.
--mode final_inspection: the test protocol/operator says this fixture is expected complete now (a complete good order
  = 30 vials), so every empty position is MISSED. External context only: the equipment has no order-complete signal.
--fill-count K: OFFLINE metadata only (synthetic / staged ground truth), never a runtime input.
--k-from-events: K = placements counted by the classifier-free event detector (first event = baseline,
  its K from occupancy; each later event adds one placement). Kept for comparison with the default.
"""
import argparse, csv, json, os, time
import cv2, numpy as np
import positions as P
import real_data as R
import events as E
from fill_state import resolve_fixture, SequenceTracker, LIVE_FILL, FINAL_INSPECTION, MODES
from rules import RuleInspector

OUT = os.path.join(P.HERE, "results", "prototype", "inspect")
COLORS = {"PASS": (0, 170, 0), "DEFECT": (0, 0, 230), "MISSED": (200, 0, 200), "WAITING": (150, 150, 150), "UNSURE": (0, 150, 255)}


def inspect(img, src, pm, inspector, fill_count=None, plate=None, mode=LIVE_FILL, frontier="waiting", tracker=None,
            fixture_id="G"):
    if fixture_id != "G":
        raise ValueError("fixture S is not supported: this runtime has a calibrated position map for fixture G only")
    t0 = time.perf_counter()
    arm = E.arm_pixels(img, pm, plate)
    crops = P.extract_crops(img, pm, src)
    obs = inspector.inspect_batch(crops)
    if arm:
        for o in obs:
            o.visual_state, o.reason = "unknown", f"arm over the fixture ({arm} yellow px on the plate)"
    visuals = [o.visual_state for o in obs]
    if tracker is not None and fill_count is None:
        if all(v == "unknown" for v in visuals):            # arm frame: not a valid frame, history unchanged
            K, ksrc, states = resolve_fixture(visuals, mode=mode, frontier=frontier)
        else:
            K, _, states = tracker.update(visuals)
        ksrc = "tracked"
    else:
        K, ksrc, states = resolve_fixture(visuals, fill_count, mode, frontier)
    ms = (time.perf_counter() - t0) * 1000
    pos = []
    for c, o, (final, why) in zip(crops, obs, states):
        features = dict(o.features)
        pos.append({"position": c.position, "index": c.index, "box": list(c.box), "visual_state": o.visual_state,
                    "defect_score": o.score, "final_state": final,
                    "reason": o.reason if o.visual_state == "unknown" else (why or o.reason),
                    "vision_reason": o.reason, **features,
                    # Always present in the position record.  Learned inspectors populate this with the raw
                    # taxonomy argmax; deterministic/rule inspectors leave it blank.
                    "model_argmax_class": features.get("model_argmax_class", "")})
    return {"frame": src, "fixture_id": fixture_id, "mode": mode, "K": K, "K_source": ksrc,
            "arm_px": arm, "offset_px": list(pm.offset),
            "inspector": inspector.name, "latency_ms": round(ms, 1), "positions": pos}


def annotate(img, res, pm, scale=3):
    x0, y0, x1, y1 = P.fixture_window(pm)
    big = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    S = lambda u, v: (int(round((u - x0) * scale)), int(round((v - y0) * scale)))
    for p, (bx0, by0, bx1, by1) in zip(res["positions"], pm.anchor_boxes):
        col = COLORS[p["final_state"]]
        cv2.rectangle(big, S(bx0, by0), S(bx1, by1), col, 2)
        a, b = S(bx0, by1)
        cv2.rectangle(big, (a, b - 2), (a + 66, b + 12), col, -1)
        cv2.putText(big, f"{p['position']} {p['final_state']}", (a + 2, b + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    head = np.full((44, big.shape[1], 3), 25, np.uint8)
    cv2.putText(head, f"{res['frame']}", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    k = f"{res['K']} ({res['K_source']})"
    cv2.putText(head, f"fill frontier/count={k}  offset {res['offset_px']}  {res['inspector']}" + ("  ARM OVER FIXTURE" if res["arm_px"] else ""),
                (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    leg = np.full((22, big.shape[1], 3), 25, np.uint8); x = 6
    for s, c in COLORS.items():
        n = sum(p["final_state"] == s for p in res["positions"])
        cv2.rectangle(leg, (x, 5), (x + 12, 17), c, -1)
        cv2.putText(leg, f"{s} {n}", (x + 16, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA); x += 110
    return np.vstack([head, big, leg])


def safe(name):
    return os.path.basename(name).replace(".png", "").replace(".jpg", "").replace("@", "_")


def write(img, res, pm, out, debug=False):
    os.makedirs(out, exist_ok=True); stem = os.path.join(out, safe(res["frame"]))
    cv2.imwrite(stem + "_annotated.png", annotate(img, res, pm))
    crops = P.extract_crops(img, pm, res["frame"])
    cv2.imwrite(stem + "_contact.png", P.contact_sheet(crops, [f"{p['position']} {p['final_state']}" for p in res["positions"]],
                                                         [COLORS[p["final_state"]] for p in res["positions"]]))
    if debug:
        cv2.imwrite(stem + "_positions.png", P.draw_positions(img, pm, P.cap_blobs(img, pm)))
    json.dump(res, open(stem + ".json", "w"), indent=1)


def _expected_occupancy(final_state):
    """Conservative fallback when no more specific process context is available.

    An unresolved inspection stays ``unknown``: treating it as empty could silently pass a suspected defect, while
    treating it as occupied could turn a trailing unfilled position into a claimed missed placement.
    """
    if final_state in {"PASS", "DEFECT", "MISSED"}:
        return "occupied"
    if final_state == "WAITING":
        return "empty"
    return "unknown"


def _position_index(position):
    """Return a validated one-based fixture index, or ``None`` for an incomplete legacy record."""
    value = position.get("index")
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 30:
        return value
    label = str(position.get("position", ""))
    try:
        value = int(label[1:]) if len(label) == 3 and label[:1] in {"G", "S"} else None
    except ValueError:
        value = None
    return value if value is not None and 1 <= value <= 30 else None


def _expected_occupancy_audit(position, res):
    """Return ``(expectation, source)`` without representing inferred context as ground truth.

    An explicit fill count and final-inspection declaration are external process context.  Event counting, temporal
    tracking, and the visible frontier are deterministic inferences.  Older/incomplete records retain a clearly
    marked fallback instead of silently claiming an external signal.
    """
    k_source = str(res.get("K_source", ""))
    index, count = _position_index(position), res.get("K")
    if k_source in {"given", "events"} and index is not None and isinstance(count, int) and not isinstance(count, bool):
        source = ("external_context:fill_count" if k_source == "given"
                  else "deterministic_inference:event_count")
        return ("occupied" if index <= count else "empty"), source
    if res.get("mode") == FINAL_INSPECTION:
        return "occupied", "external_context:final_inspection"
    source = {
        "occupancy": "deterministic_inference:visual_frontier",
        "tracked": "deterministic_inference:temporal_tracker",
    }.get(k_source, "deterministic_fallback:final_state")
    return _expected_occupancy(position["final_state"]), source


def _decision_reason(p):
    final, visual = p["final_state"], p["visual_state"]
    if final == "PASS":
        return "visual_pass"
    if final == "DEFECT":
        return f"visual_{visual}"
    if final == "MISSED":
        return "no_vial_where_occupied_expected"
    if final == "WAITING":
        return "expected_trailing_empty"
    return "unsupported_or_unresolved_visual_evidence" if visual == "unknown" else "deterministic_sequence_unsure"


def prediction_rows(run, event, res, *, event_uid=None, inspection_timestamp_utc=""):
    """Flatten one event while retaining the legacy evaluation columns.

    The explicit aliases are the live safety/audit contract used by the five-class staging adapter.  ``event_uid``
    is always populated; a wall-clock timestamp is supplied by :class:`live_inspect.Session` when run live.
    """
    seq = {"MISSED": "missed", "WAITING": "waiting"}
    fixture = res.get("fixture_id", "G")
    try:
        event_label = f"{int(event):06d}"
    except (TypeError, ValueError):
        event_label = str(event)
    event_uid = event_uid or f"{run}:event_{event_label}"
    rows = []
    for p in res["positions"]:
        expected_occupancy, expected_occupancy_source = _expected_occupancy_audit(p, res)
        vision_code = str(p.get("reason_code", "")).strip()
        decision_code = _decision_reason(p)
        reason_code = ";".join(dict.fromkeys(filter(None, (vision_code, decision_code))))
        score = p["defect_score"]
        threshold = p.get("applicable_threshold", p.get("t_defect", ""))
        row = {
            # Five-class live audit contract.
            "source_image_id": res["frame"], "fixture_id": fixture, "position_id": p["position"],
            "expected_occupancy": expected_occupancy,
            "expected_occupancy_source": expected_occupancy_source,
            # This safety-gated class may be unknown even when the raw model has a five-class argmax.  Preserve
            # both without presenting an unsupported raw argmax as a defect detection.
            "predicted_visual_class": p["visual_state"],
            "model_argmax_class": p.get("model_argmax_class", ""), "model_score": score,
            "applicable_threshold": threshold, "inspection_decision": p["final_state"],
            "reason_code": reason_code, "inspection_timestamp_utc": inspection_timestamp_utc,
            "event_uid": event_uid,
            # Legacy columns consumed by eval_metrics.py and the existing replay tools.
            "run": run, "event": event, "frame": res["frame"], "position": p["position"],
            "predicted_state": p["visual_state"], "defect_score": score,
            "pred_sequence": seq.get(p["final_state"], ""), "final_state": p["final_state"],
            "K": res["K"], "K_source": res["K_source"], "cap_frac": p.get("cap_frac", ""),
            "cap_offset": "" if p.get("cap_offset") is None else p["cap_offset"], "reason": p["reason"],
        }
        rows.append(row)
    return rows


def load(ix, src):
    cached = os.path.join(os.path.dirname(OUT), "events", src + ".png")
    return ix.load(cached if os.path.exists(cached) else src)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="inspect real cap_seal frames end to end (DEV runs only)")
    ap.add_argument("sources", nargs="*"); ap.add_argument("--run"); ap.add_argument("--events", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fill-count", type=int, help="offline metadata only"); g.add_argument("--k-from-events", action="store_true")
    g.add_argument("--track", action="store_true")
    ap.add_argument("--mode", choices=MODES, default=LIVE_FILL); ap.add_argument("--frontier", choices=("waiting", "unsure"), default="waiting")
    ap.add_argument("--persist", type=int)
    ap.add_argument("--debug", action="store_true"); ap.add_argument("--out", default=OUT)
    ap.add_argument("--inspector", choices=("rule", "resnet", "staging5"), default="rule")
    ap.add_argument("--threshold", choices=("d038", "preregistered"), default="d038", help="ResNet threshold (D038)")
    ap.add_argument("--model-dir", help="model run directory; required for staging5")
    a = ap.parse_args()
    tracker = SequenceTracker(a.persist or (1 if a.events else 2), a.mode, a.frontier) if a.track else None
    if a.inspector == "resnet":
        from resnet_inspector import ResNetInspector
        insp = ResNetInspector(a.model_dir or None, a.threshold) if a.model_dir else ResNetInspector(threshold=a.threshold)
    elif a.inspector == "staging5":
        if not a.model_dir:
            ap.error("--inspector staging5 requires an explicit --model-dir")
        from resnet5_onnx_inspector import ResNet5OnnxInspector
        insp = ResNet5OnnxInspector(a.model_dir); insp.warmup()
    else:
        insp = RuleInspector()
    ix = R.Index(); pm = P.PositionMap(); plate = pm.plate_mask()
    if a.run:
        R.check_allowed(a.run)
        ev = [r for n in os.listdir(os.path.join(os.path.dirname(OUT), "events")) if n.startswith(f"events_run{a.run}_")
              for r in csv.DictReader(open(os.path.join(os.path.dirname(OUT), "events", n)))]
        jobs = [(a.run, int(r["event"]), r["frame"]) for r in sorted(ev, key=lambda r: int(r["event"]))]
    else:
        jobs = [(ix.run_of_source(s) or "external", i + 1, s) for i, s in enumerate(a.sources)]
    rows, prev, K0 = [], None, None
    out = os.path.join(a.out, (f"run{a.run}" if a.run else "frames") + (f"_resnet_{a.threshold}" if a.inspector == "resnet" else ""))
    for run, event, src in jobs:
        img = load(ix, src)
        if a.k_from_events:
            if K0 is None:
                K0 = resolve_fixture([insp.inspect(c).visual_state for c in P.extract_crops(img, pm, src)])[0]
            res = inspect(img, src, pm, insp, fill_count=min(30, K0 + event - jobs[0][1]), plate=plate)
            res["K_source"] = "events"
        else:
            res = inspect(img, src, pm, insp, a.fill_count, plate, a.mode, a.frontier, tracker)
        write(img, res, pm, out, a.debug)
        rows += prediction_rows(run, event, res)
        counts = {s: sum(p["final_state"] == s for p in res["positions"]) for s in COLORS}
        print(f"run {run} e{event:02d} {src[-30:]:>30s} K={res['K']}({res['K_source']}) {res['latency_ms']:5.1f} ms  " +
              " ".join(f"{s}={n}" for s, n in counts.items() if n))
        if prev:
            ch = [f"{p['position']} {q['final_state']}->{p['final_state']}" for p, q in zip(res["positions"], prev["positions"])
                  if p["final_state"] != q["final_state"]]
            if ch:
                print("      changes:", ", ".join(ch))
        prev = res
    with open(os.path.join(out, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> {out}/ (annotated, contact sheets, json, predictions.csv)")
