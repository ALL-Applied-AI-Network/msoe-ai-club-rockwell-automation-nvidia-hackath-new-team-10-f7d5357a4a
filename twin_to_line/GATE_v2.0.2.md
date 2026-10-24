# Isaac appearance gate — kit gate-v2.0.2: frozen decision rule

Status: **frozen 2026-10-14, before any gate render exists.** Decision semantics clarified before the first Isaac
render; no synthetic gate result existed when this version was frozen. (The twin package ships a vendor preview,
`previews/tray_inspection_rtx.png`, 640×480, dated Oct 8. This kit did not produce it, it was not opened while
v2.0.1–v2.0.2 were written, and it is not gate evidence.) Any change to this rule = a new kit version + regenerated
`MANIFEST.sha256`, and only while no gate render exists.

Question: does the twin render G-fixture images whose geometry and appearance are compatible enough with the real
CSI `cap_seal` domain to justify *testing* synthetic defect generation? A PASS permits that experiment; it does not
show that synthetic training works.

## Version history
| Version | Change | Why |
| --- | --- | --- |
| v2.0 (Oct 13) | Six-scene kit and metrics (`compare_gate.py`); no written decision rule | — |
| v2.0.1 (Oct 14) | `render_keep.py` `--stage` default `os.path.join(HERE, "csi_line.usda")` → `"csi_line.usda"`; draft decision rule; `gate_evidence.py` | v2.0 raised `NameError: name 'HERE' is not defined` on every call; no rule existed |
| v2.0.2 (Oct 14) | Decision semantics only: INCONCLUSIVE = invalid execution only; render-vs-projection outside tolerance moved from INCONCLUSIVE to FAIL (G0, was E3); explicit four-way aggregation; undefined metrics fail their check; band boundaries explicit; limitations recorded. Checklist rewritten as enumerated failure conditions, same four items: V1 keeps "missing" and "solid black" but drops "cap-coloured / not white" (colour is scored by A1/B1, and real pale caps exist: 2 of 16 DEV caps); V3's exposure clause becomes "plate cannot be seen"; the liquid WORK item is pinned to the uncapped G02 (B6). **No numeric threshold changed.** | GPT review of v2.0.1 |

`render_keep.py` sha256: v2.0 `db64b98f…6736f` → v2.0.1/v2.0.2 `f0e8a174…a8e22` (the `--stage` default makes no
difference to the gate: relative stages resolve against `HERE` and `run_gate.sh` always passes `--stage`). Unchanged
since v2.0: scenes (`make_variants.py`), variants (baseline, t00, t10, t20, t30, liquid), 64 subframes, PathTracing,
CapSealCam 768×576 → `apply_lens` 640×480, real reference frames and metrics (`compare_gate.py`), crops
(`cut_crops.py`), projection (`capsealcam.py`).

## 1. Validity — INCONCLUSIVE only if one of these holds
| ID | The run is invalid when |
| --- | --- |
| X1 | `run_log.txt` does not report `kit files match MANIFEST.sha256` (it is not this kit) |
| X2 | the `make_variants.py` label self-check failed (the scenes are not the frozen scenes) |
| X3 | for any of the six variants: `render_keep.py` exit ≠ 0 in `run_log.txt`, or `renders/<v>.png` or `renders/<v>.json` is missing |
| X4 | any `renders/<v>.png` is unreadable, not 640×480, or blank (grayscale std ≤ 5) |
| X5 | `compare_gate.py` cannot produce `gate_report.json`, `panel_full_frames.png` and `panel_crops.png` (e.g. a real reference frame is missing or the script fails) |

Nothing else makes a run INCONCLUSIVE. A render that executed validly but shows the wrong geometry or appearance is
scientific evidence and is scored below. A metric that is undefined (`NaN`/`null`) in a valid run because the render
lacks the measured feature (e.g. no synthetic caps detected → no geometry value, cap ΔE or area ratio) makes the
check that uses it **FAIL**. (Verified on a yellow-free stand-in: `compare_gate.py` still writes all three artifacts.)
`gate_evidence.py` is supplementary; its failure never affects the outcome.

## 2. Hard checks — any failure in a valid run = FAIL
| ID | `gate_report.json` key | Passes when |
| --- | --- | --- |
| G0 | `geometry.synthetic_cap_vs_projection_px` (render + lens vs `capsealcam.py`) | median ≤ 2.0 px **and** max ≤ 4.0 px |
| G1 | `geometry.empty_hole_pattern_IoU_synthetic_t00_vs_real_empty` | ≥ 0.50 |
| A1 | `geometry.matched_to_twin_positions` (real-domain HSV cap detector on synthetic baseline) | ≥ 27 (of 30) |
| A2 | `appearance.cap_area_px_median` synthetic / real | 0.75 ≤ ratio ≤ 1.33 |
| V1–V4 | visual checklist (section 4) | no listed failure condition observed |

## 3. Appearance bands — scored in a valid run
| ID | Metric | OK | WORK | FAIL |
| --- | --- | --- | --- | --- |
| B1 | `appearance.cap_color_Lab.deltaE76` | ≤ 15 | > 15 and ≤ 30 | > 30 or undefined |
| B2 | `appearance.plate_Lab.deltaE76` | ≤ 20 | > 20 and ≤ 40 | > 40 or undefined |
| B3 | `appearance.hole_interior_Lab.deltaE76` | ≤ 20 | > 20 and ≤ 40 | > 40 or undefined |
| B4 | `appearance.background_wall_Lab.deltaE76` | ≤ 20 | > 20 and ≤ 40 | > 40 or undefined |
| B5 | `appearance.frame_luminance`: \|synthetic_mean − real_mean\| / real_mean | ≤ 0.25 | > 0.25 | — |
| B6 | liquid visible inside the uncapped G02 vial of the `liquid` render (capped vials hide liquid from above, D013) | visible | not visible | — |

Values are compared exactly as written in `gate_report.json` (ΔE rounded to 0.1; A2 from the two integer medians).

## 4. Visual checklist — mandatory, enumerated failure conditions
Done by Noor or the reviewer on `panel_full_frames.png`, `panel_crops.png` and the six renders, recorded as
fail / pass with one line each, **before** reading the ΔE values. An item fails only if its stated condition is
observed; anything not listed here is not a visual failure.
- **V1** baseline: at one or more of the 30 G positions no cap is visible, or a cap is rendered solid black.
- **V2** t00: no vial body is visible at G01, G02 or G03 (only plate/hole where the vial stands), or one is a solid black silhouette.
- **V3** t00: the G fixture plate and its holes cannot be seen (missing, or lost in black or blown-out white).
- **V4** any render: floating, duplicated or misplaced parts (a part away from where it sits in the other renders), or the fixture/plate/back wall seen in the other renders is missing. G positions hidden or tilted by design are excepted.

## 5. Outcome — evaluate in this order
| Outcome | Rule | Then |
| --- | --- | --- |
| **INCONCLUSIVE** | any of X1–X5 | no scientific result; fix the execution, rerun the same kit |
| **FAIL** | valid, and any of G0, G1, A1, A2, V1–V4 fails, or any of B1–B4 is in FAIL | stop; GPT reviews why before any change |
| **PASS WITH WORK** | valid, no FAIL condition, and ≥ 1 of B1–B6 in WORK | stop; WORK items are twin-only fixes (materials, lights), verified by re-running this same rule before any dataset |
| **PASS** | valid, all hard checks pass, B1–B6 all OK | stop before large-scale generation; GPT designs the synthetic experiment |

The crooked-cap result (section 6) and the informational values (section 7) never change the outcome.

## 6. Crooked-cap scope — reported separately
**SUPPORTED** if t20 or t30 has `centroid_shift_vs_real_seated_spread` ≥ 2 or
`elongation_change_vs_real_seated_spread` ≥ 2. **NOT SUPPORTED** otherwise: the headline narrows to missing cap +
missed placement (D006). **UNDETERMINED** if the G03 cap is not detected in t00 or the tilt render (fields absent):
report the pixel-change counts and the panel; do not infer support. t10 is expected below threshold (D005–D006).
Shallow-angle detectability is never claimed from synthetic images alone.

## 7. Informational only
`hole_pattern_centroid_shift_px`, `real_cap_vs_projection_px`, `plate_luminance_spread_std`, noise sigmas
(augmentation adds noise), `sharpness_laplacian_var_plate`, everything from `gate_evidence.py`.

## 8. Why these numbers (calibrated on DEV real frames before any render; unchanged since v2.0.1)
Execution smoke test (`results/isaac_gate/dryrun/`): `compare_gate.py` with **real DEV frames as stand-in renders**
(baseline = run 1 last frame, t00 = run 2 first frame with 7 vials, liquid = run 2 uncapped frame). Real-vs-real:
cap ΔE 5.3, wall 3.7, plate 7.4, hole interior 10.3; frame mean +7.5 %; cap area 311 vs 310 px; hole IoU 0.84; hole
centroid shift 9.3 px; real seated-cap spread 0.95 px centroid, 0.072 elongation. Every OK band sits above
real-vs-real variation. Two draft numbers were changed in v2.0.1, before any render:
- Hole centroid shift (draft ≤ 6 px) → informational: real-vs-real is 9.3 px because the shift depends on which
  holes are occupied (t00 has vials in G01–G03, the real reference none) and on the Sept 22/23 pose change.
- G0 (draft median ≤ 1.5 / max ≤ 3 px) → ≤ 2.0 / ≤ 4.0 px: from twin geometry alone the visible cap top sits
  0.67 px median / 1.34 px max from the projected full-hull centroid that `compare_gate.py` predicts; ~1 px is added
  for HSV segmentation and anti-aliasing.

## 9. Known limitations — carry forward to the synthetic-dataset design
- Real DEV cap defects are gross displacement (5 units: run 1 G12–G15, run 2 G07, angle unknown); the gate's defect
  variants are G03 rim tilts of 10/20/30°. No displacement scene is rendered.
- t00/t10/t20/t30/liquid hide G05–G30, so their crops are more isolated than many real crops (neighbour caps cover
  4–8 % of some DEV crops); only baseline has all 30 caps.
- One lighting configuration (the twin's own lights) and one camera. Pale/specular caps (2 of 16 real seated caps
  in DEV) are represented only if that single setup produces them.
- Crooked caps below ~15–20° are a camera-resolution limit (D005–D006).
- Real defect data is scarce (DEV: 7 independent cap-defect units, 5 displaced + 2 uncapped).
- Passing the gate does not show that synthetic training works; it only permits testing it.

## 10. Prohibited after any render exists
Changing thresholds, checklist items, reference frames, variants, subframes or renderer; re-rendering with tweaks
and reporting the best; running on or tuning with runs 4–5; generating the dataset; training (ResNet-18, ChangeNet,
TAO); Cosmos.

## 11. Runbook (team Brev or team DGX access only; RTX PRO 6000 preferred, L40S fallback; never a personal account)
```bash
# Mac, once:  brew install brevdev/homebrew-brev/brev && brev login
cd ~/code/ignite/twin_to_line && shasum -a 256 -c MANIFEST.sha256          # 8 x OK
tar czf ~/Downloads/ignite_kit_v2.0.2.tar.gz $(awk '{print $2}' MANIFEST.sha256) MANIFEST.sha256 KIT_VERSION
brev copy ~/Downloads/ignite_kit_v2.0.2.tar.gz <instance>:/home/ubuntu/
brev shell <instance>
mkdir -p ~/twin_to_line && tar xzf ~/ignite_kit_v2.0.2.tar.gz -C ~/twin_to_line
RATE_PER_HR=<$/hr> bash ~/twin_to_line/run_gate.sh     # log must say "kit files match MANIFEST.sha256", six exit=0
exit
brev copy <instance>:/home/ubuntu/ignite_gate/results.tar.gz ~/Downloads/      # then STOP the instance
# Mac:
R=results/isaac_gate/run1 && mkdir -p $R && tar xzf ~/Downloads/results.tar.gz -C $R
../.venv/bin/python compare_gate.py $R ~/Downloads/csi-vial-fill-dataset/cap_seal/images ~/Downloads/csi_digital_twin/csi_digital_twin_v1 $R/report
../.venv/bin/python gate_evidence.py $R $R/evidence
```
Only the kit files go to the instance (no real frames, labels or prototype outputs).

## 12. Report template
Kit version + manifest line from `run_log.txt` · GPU, Isaac version, renderer, subframes, per-render wall time, cost ·
X1–X5 · G0, G1, A1, A2 values · V1–V4 answers · B1–B6 values with band · crooked-cap SUPPORTED / NOT SUPPORTED /
UNDETERMINED with values · informational values · outcome (INCONCLUSIVE / FAIL / PASS WITH WORK / PASS) and the row
of section 5 that decided it · artifact paths (renders, crops, panels, evidence).
