# Twin-to-Line kit v2 — sim-to-real gate (frozen: gate-v2.0.2, see MANIFEST.sha256 → 8 × OK)

Start here: `HACKATHON_HANDOFF.md`, then `DECISIONS.md` and `ORGANIZER_UPDATE_2026-10-16.md`. Gate decision rule, version history (v2.0 → v2.0.2) and the current
runbook (copies only the kit files to the instance): `GATE_v2.0.2.md`.

Goal: six tiny Isaac Sim renders through CapSealCam, to measure the sim-to-real gap before any dataset or training.

## What gets rendered (all at 768×576 pinhole → real lens → 640×480)
| Variant | Scene |
| --- | --- |
| baseline | tray station as shipped: all 30 G vials capped (geometry + appearance check vs a full real frame) |
| t00 | G01 pass · G02 missing cap · G03 cap 0° · G04 no vial (missed placement) · G05–G30 empty |
| t10 / t20 / t30 | same, G03 cap tilted 10° / 20° / 30° about its rim (only G03 differs) |
| liquid | t20 + liquid cylinders: G01 blue, G02 orange (uncapped), G03 blue |

Path tracing, 64 subframes per frame, 2 orchestrator steps (the twin's own render settings, `--subframes` raised from 32).

## Run on the Brev Isaac Sim launchable (about 20–30 min of instance time)
```bash
brev copy ~/code/crusaders/twin_to_line <instance-name>:/home/ubuntu   # from your Mac
brev shell <instance-name>
RATE_PER_HR=<$/hr shown in Brev> bash ~/twin_to_line/run_gate.sh   # downloads the twin, finds Isaac Sim, logs cost
#   if it cannot find Isaac Sim:  ISAAC_PY=/path/to/python.sh bash ~/twin_to_line/run_gate.sh
exit
brev copy <instance-name>:/home/ubuntu/ignite_gate/results.tar.gz ~/Downloads/   # then STOP the instance
```
Expected at the end of the log: six `exit=0` lines and `DONE -> .../results.tar.gz`.

## Files
- `run_gate.sh` — the whole run: twin download (gdown), a venv with usd-core for authoring, Isaac Sim detection,
  6 renders with timing, crops, tarball, `run_log.txt` (GPU, Isaac version, per-render wall time).
- `make_variants.py` — writes each variant as a tiny override layer over the untouched station file, plus
  `labels.json` (state, fill-sequence status, drawn box, seated-position anchor box for all 30 G positions).
  Asserts that labels derived from the composed USD equal the authored states.
- `render_keep.py` — the twin's `tools/render_example.py` with 3 marked changes: `--twin` path, keeps the
  undistorted `*_pinhole768.png`, writes `*.json` with render settings, Isaac version, GPU and timings.
- `cut_crops.py` — 4 labeled crops per render (empty holes use the anchor box) + overlay + `crops.csv`.
- `compare_gate.py` — runs on the results afterwards: geometry, appearance (Lab ΔE per region), noise,
  crooked-cap visibility against real seated-cap variation, comparison panels.
- `capsealcam.py` — CapSealCam projection (pinhole + k1), matches `apply_lens.py` to 0.001 px.
- `eval_metrics.py` — metrics for later (per-class recall, FN, FP rate, FAR at full recall, trailing-empty alarms,
  sequence accuracy); `--selftest` checks it on mock predictions. No model, no training.
- `real_splits.json` — which real runs are DEV vs HELD-OUT.
- Earlier tools: `registration.py`, `tilt_sensitivity.py`, `twin_positions.py`, `overlay.py`.
- `gen_scenes.py` + `configs/` — synthetic scene generator (not part of the frozen gate kit; see below).

Twin note: the twin README says it was tested with Isaac Sim 6.0; `run_log.txt` records the version actually used.

## Synthetic scenes (CPU only; scenes and labels, no renders)
`gen_scenes.py` writes USD override layers over the untouched station (same mechanism as `make_variants.py`) for
pass, missing_cap, no_vial (missed or waiting) and crooked_cap (rim-pivot tilt or lift), plus seated-cap offsets,
leaning vials, lighting and material variation. Labels are read back from each composed stage and checked against
the authored spec and physical rules (seat height, cap on its vial, vial inside its plate hole, no neighbour
collisions). `labels.json` uses the gate schema, so `cut_crops.py` reads it. Not rendered; no transfer claim.
```bash
python gen_scenes.py ~/Downloads/csi_digital_twin/csi_digital_twin_v1 --selftest            # -> "selftest passed"
python gen_scenes.py ~/Downloads/csi_digital_twin/csi_digital_twin_v1 out/smoke --config configs/smoke.json --seed 0
python gen_scenes.py ~/Downloads/csi_digital_twin/csi_digital_twin_v1 out/sweep --config configs/tilt_sweep.json
IGNITE_TWIN=~/Downloads/csi_digital_twin/csi_digital_twin_v1 python -m pytest tests/test_gen_scenes.py -q
```
- `smoke.json`: 300 scenes, pass / missing_cap / no_vial only, lighting + material variation (Oct 25 smoke test).
- `tilt_sweep.json`: 130 scenes, crooked caps 5-40° in 8 world directions on a checkerboard, twin's own lights.
- Scene `scene.usda` sublayers the station by relative path: generate on the machine that renders, or keep the
  twin and output folders in the same relative layout.
- Labels are physical truth ("any cap not fully seated" = crooked_cap). Visibility is reported, not assumed:
  `cap_shift_px` / `cap_max_shift_px`, and `bbox_in_crop` (False when a crooked cap leaves the runtime crop box).

## Real-image prototype (DEV runs 1-3 only; laptop, no GPU)
`real frame -> twin G01-G30 map (+ DEV offset) -> 30 crops -> rule inspector -> fill-order reasoning (occupancy, D021) -> PASS / DEFECT / MISSED / WAITING / UNSURE`
Setup: `python3.11 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt`. Data paths: `IGNITE_REAL` (default `~/Downloads/csi-vial-fill-dataset/cap_seal`), twin dir as an argument.
```bash
python export_position_map.py ~/Downloads/csi_digital_twin/csi_digital_twin_v1   # -> position_map_G.json (twin only)
python events.py --run 1                     # classifier-free placement events (video 1); --source stills for runs 2-3
python register_dev.py                       # -> registration_dev.json (DEV offset) + overlays / crop sheets
python rules.py --fit labels/dev_labels.csv  # -> rule_thresholds.json (DEV labels only)
python inspect_frame.py --run 1 --events --track                  # end to end, one frame per event, occupancy + tracker
python inspect_frame.py --run 1 --events --k-from-events          # comparison: K from the camera's own event counter
python inspect_frame.py cap_seal_20260922_192609_126406.jpg --debug   # single frame, LIVE_FILL; --mode final_inspection only when the test protocol declares the fixture complete
python check_organizer_grid.py <labeler>/calibration.json --example <labeler>/examples/tray_000001.jpg   # organizer-grid consistency check
python eval_metrics.py results/prototype/inspect/dev_events_predictions.csv --labels labels/dev_labels.csv
python -m pytest tests -q
```
- `real_data.py` refuses runs 4-5 and unassigned frames everywhere (stills, videos, cached copies of video frames).
- `fill_state.py` holds the manufacturing semantics; inspectors only report what a crop looks like. The line has no
  placement count or order-complete signal: MISSED / WAITING come from occupancy (+ `SequenceTracker`) in LIVE_FILL
  (default) or FINAL_INSPECTION (external protocol/operator context, never from the equipment); `--fill-count` is
  offline metadata only.
- G01-G30 is the twin's numbering (our convention), not CSI's internal position numbers (D022).
- Outputs go to `results/prototype/` (gitignored). `labels/dev_labels.csv` are draft visual labels (status=draft).

## Dataset contract for the ResNet-18 experiment (not trained yet; real-image fitting on hold, D025)
- One crop function for both domains: `positions.extract_crops` + `crop_for_model` (pad 6 px, 64x64). Synthetic crops
  must be cut from the same twin map, never re-derived.
- Visual classes: `pass`, `missing_cap`, `crooked_cap` (not fully seated; target displaced/tilted >= ~15-20 deg,
  D005-D006, D023), `no_vial`. `damaged_vial` is out of scope for v1 but still scored if it occurs (D026).
  MISSED vs WAITING is never a visual class; `fill_state.py` decides it from occupancy (D021).
- Arms: synthetic-only vs the rule baseline (always). Real-only and synthetic + real dose N in {5, 10, 20} only if
  organizers allow fitting on newly collected CSI data (D025; supplied runs 1-5 are never training data).
  Dose sets: N real DEFECT examples with matched real passes (`dose_sets.py`: nested, class-stratified, one unit per placement event, refuses test/holdout rows).
- Unit = placement event (`events.py`), not video frame. DEV runs 1-3 give 7 independent cap-defect units
  (5 displaced caps, 2 uncapped vials): `make_dose_sets` refuses 10 and 20 by design, so the dose curve needs the
  hand-staged CSI collection.
