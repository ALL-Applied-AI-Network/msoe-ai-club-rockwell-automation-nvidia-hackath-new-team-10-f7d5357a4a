"""D041: one approved good-only DEV fit; G3 is a separate, once-only command.

    python fit_rule_dev_good_only.py

Uses the existing event/label join, crop features and rules.fit without changes.
Refuses any existing output or fit evidence directory, including interrupted fits.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import eval_metrics as E
import g3_check as G
import positions as P
import real_data as R
import rules

HERE = Path(P.HERE)
OUT = HERE / "rule_thresholds_dev_good_only.json"
EVIDENCE = HERE / "results/g3/rule_dev_good_only_fit"


def select_rows(jobs, labels, positions):
    if any(str(run) not in G.RUNS for run, _, _ in jobs):
        raise ValueError("only DEV runs 1-3 are allowed")
    eligible = [r for r in labels if r["usable"] == "1" and
                (r["ground_truth"] == "pass" or
                 (r["ground_truth"] == "no_vial" and r["gt_sequence"] == "waiting"))]
    predicted = [dict(run=run, event=event, frame=src, position=pos)
                 for run, event, src in jobs for pos in positions]
    rows, _, unmatched = E.join_labels(predicted, eligible)
    E.check_units(rows)
    counts = Counter(r["ground_truth"] for r in rows)
    if unmatched or counts != {"pass": 112, "no_vial": 439}:
        raise ValueError(f"expected 112 pass + 439 waiting; found {dict(counts)}, unmatched={len(unmatched)}")
    return rows


def main():
    if OUT.exists() or EVIDENCE.exists():
        raise FileExistsError("refusing an existing threshold file or fit attempt; D041 allows one fit")
    jobs, manifests = G.event_jobs()
    pm, ix = P.PositionMap(), R.Index()
    if pm.offset != (0.9, -1.34):
        raise ValueError(f"unexpected DEV offset: {pm.offset}")
    rows = select_rows(jobs, E.load(G.LABELS), pm.ids)
    head, dirty = G.git_state()
    if dirty:
        raise ValueError(f"commit before fitting: {dirty}")
    sources = {}
    for run, _, src in jobs:
        R.check_allowed(run, role_map=ix.role_map)
        if ix.run_of_source(src) != run:
            raise ValueError(f"event/source run mismatch: {src}")
        cached = Path(G.EVENTS) / (src + ".png")
        path = cached if cached.exists() else (Path(ix.video_path(run)) if R.VIDEO_RE.match(src) else Path(ix.root)/"images"/src)
        sources[str(path)] = path
    lock = {"decision": "D041", "started_utc": datetime.now(timezone.utc).isoformat(), "git_commit": head,
            "labels_sha256": G.sha256(G.LABELS), "event_manifests_sha256": manifests,
            "input_files_sha256": {p.name: G.sha256(p) for p in sources.values()},
            "code_sha256": {n: G.sha256(HERE/n) for n in
                ("rules.py", "positions.py", "g3_check.py", "eval_metrics.py", "real_data.py",
                 "inspect_frame.py", "fit_rule_dev_good_only.py", "ARM_B_DOUBLE_PRIME_PROPOSAL.md", "DECISIONS.md")},
            "position_map_sha256": G.sha256(P.MAP_PATH), "registration_sha256": G.sha256(P.REG_PATH),
            "selected_rows": rows}
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    with (EVIDENCE/"input_lock.json").open("x") as f:
        json.dump(lock, f, indent=1)
    print(f"Locked inputs before feature extraction: {EVIDENCE/'input_lock.json'}", flush=True)
    features = []
    for _, _, src in jobs:
        selected = [r for r in rows if r["frame"] == src]
        if not selected:
            continue
        crops = {c.position: c for c in P.extract_crops(G.load_frame(ix, src), pm, src)}
        features.extend({**r, **rules.features(crops[r["position"]])} for r in selected)
    if len(features) != 551:
        raise ValueError(f"expected 551 feature rows, got {len(features)}")
    G.write_csv(EVIDENCE/"features.csv", features)
    # Exactly one call; original grid, objective, tie-breaks and feature rounding.
    thresholds = rules.fit(features)
    artifact = {"inspector": rules.RuleInspector.name, "thresholds": thresholds,
                "features": {"cap_frac_disk_px": rules.CAP_R, "cap_offset_disk_px": rules.OFF_R},
                "fit_on": {"runs": list(G.RUNS), "position_offset_px": list(pm.offset), "n_observations": 551,
                           "classes": {"pass": 112, "no_vial_waiting": 439}, "defect_observations": 0,
                           "frames": sorted({r['frame'] for r in rows}), "git_commit": head},
                "provenance": {k: v for k, v in lock.items() if k != "selected_rows"},
                "input_lock_sha256": G.sha256(EVIDENCE/"input_lock.json"),
                "feature_rows_sha256": G.sha256(EVIDENCE/"features.csv"),
                "procedure": "D041 / ARM_B_DOUBLE_PRIME_PROPOSAL.md; unchanged rules.fit once; empty rows ignored by fitter; crooked constraint vacuous; no retuning",
                "note": "Arm B double prime: in-sample DEV reference fitted on 112 pass and 439 waiting observations only; organizer question remains open; live inspector stays B prime"}
    with OUT.open("x") as f:
        json.dump(artifact, f, indent=1); f.write("\n")
    print(json.dumps({"thresholds": thresholds, "fit_on": artifact['fit_on'], "sha256": G.sha256(OUT)}, indent=1))


if __name__ == "__main__":
    main()
