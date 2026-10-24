# Arm B″: approved non-defect DEV observation procedure (D041)

**Approved by Noor on 2026-10-24, before fitting or G3; recorded in D041.** This is a new DEV-fit reference arm,
not a synthetic-only arm and not independent evidence. D033's organizer-permission question still applies.
Approval must precede both fitting and G3. Existing arms, operating points, and live inspector stay unchanged.

1. Lock the input to the existing DEV runs 1–3 placement-event frames and `labels/dev_labels.csv`, with the same
   event manifests and usable-label join as `g3_check.py`. Record their SHA256 before running. No runs 4–5,
   sealed/staged images, organizer release, or newly chosen frames.
2. Select only `usable == "1"` observations whose ground truth is `pass`, or `no_vial` with `gt_sequence == "waiting"`.
   Exclude missing_cap, crooked_cap, damaged_vial, ambiguous labels, and no_vial/missed. Expected pool from the saved
   G3 report: **112 pass + 439 trailing-empty = 551 observations**. If the count differs, stop and report.
   The unit is an event-position. “Good only” means these selected observations, not that every other position in
   the same full frame is defect-free; do not describe this as training on entirely defect-free fixture images.
3. Use the existing real-image geometry: `PositionMap()` with the unchanged DEV offset (0.9, −1.34) px.
   Extract the selected crops using `positions.extract_crops`; calculate only their original `rules.features`.
   Preserve the existing HSV mask, disk radii, feature rounding, crop size, and image handling. No augmentation,
   per-position adjustment, balancing, weighting, resampling, or synthetic observations.
4. Pass exactly those feature rows to the unchanged **`rules.fit`** once. Its rule remains:
   `t_absent = round(0.8 * min(pass cap_frac), 3)`; seated grid 0.20…0.94 by 0.02; offset grid 2.0…8.0 by 0.1;
   maximize accepted pass observations, then prefer larger seated threshold and smaller offset. With no crooked
   rows, the original reject-every-crooked constraint is vacuous. Empty rows are supplied but **the existing fitter
   does not use them to optimize thresholds**. Do not invent an empty-hole objective or add a defect-recall target.
5. Write a new `rule_thresholds_dev_good_only.json` using exclusive creation, recording input/code hashes,
   counts, exact procedure, DEV-fit status, and the absence of defect observations. Never overwrite an existing
   threshold file. Record the selected thresholds before reading any resulting G3 performance.
6. If explicitly approved along with fitting, run G3 once on the unchanged full DEV event/label set, into a new
   `results/g3/rule_dev_good_only/` directory. Same LIVE_FILL pipeline, no tracker, same recall ≥0.90 / FPR ≤0.05
   criteria. Report all classes, misses, false positives, escalation, and exact-class recall alongside A/B/B′.
   This reuses DEV examples for fitting and evaluation: label it an **in-sample development reference**. No retuning,
   no replacing the live inspector. Noor also approved B″ for one sealed-set score next to A, B and B′ (D041);
   the Sunday freeze record must include it.

The exact procedure and single subsequent G3 run are approved. Fitting and G3 must follow the D041 commit;
no retuning is authorized. The organizer-permission question remains open.
