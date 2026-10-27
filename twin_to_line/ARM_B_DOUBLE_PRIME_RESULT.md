# D041 — B″ good-only DEV reference result

Approved in `eea3e8b` before fitting. Adapter and boundary tests committed as `470720f`; the single fit used that commit. Thresholds were committed as `901022e` before the one G3 run. No retuning.

**In-sample development reference, not independent performance evidence.** Organizer permission remains unresolved. The live inspector stays B′. D041 authorizes one sealed-set score for B″ next to A (both operating points), B and B′ after labels are locked.

## Fixed thresholds and inputs

`rule_thresholds_dev_good_only.json`: **t_absent = 0.081, t_seated = 0.22, t_offset = 4.8 px**.

SHA256: `d0d78f6b2354b5e27a371eb306da91474258f7e8d88a6e3d6a074d3581896993`.

Exactly 551 event-position observations: 112 pass and 439 no_vial/waiting, from the existing 20 DEV runs 1–3 placement events and usable-label join. No defect observations. Original map, DEV offset (0.9, −1.34), crops, HSV features, rounding and `rules.fit` were unchanged. Empty rows are supplied but ignored by the fitter; with no crooked rows its crooked-rejection constraint is vacuous. “Good only” describes selected observations, not every position in each source frame.

Input/code hashes were locked before feature extraction in `results/g3/rule_dev_good_only_fit/input_lock.json`; the 551 feature rows are in `features.csv`. Their hashes and source hashes are embedded in the new threshold JSON. No runs 4–5 were accessed.

Executed exactly once each:

```bash
python fit_rule_dev_good_only.py
python g3_check.py --rule-only --rule-thresholds rule_thresholds_dev_good_only.json --out results/g3/rule_dev_good_only
```

## G3 comparison

Same 583 usable observations: 32 defects and 551 non-defects. LIVE_FILL, frontier WAITING, no tracker. Flagged means DEFECT/MISSED/UNSURE; detection includes escalation. Criteria unchanged: recall ≥0.90 and FPR ≤0.05. A/B/B′ are read from saved reports, not rerun.

| Metric | A v1 D038 | A v1 pre-registered | B synthetic | B′ DEV-fit | B″ good-only DEV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Crooked detected | 19/19 | 2/19 | 19/19 | 19/19 | 16/19 |
| Missing cap detected | 12/12 | 12/12 | 0/12 | 11/12 | 11/12 |
| Missed placement detected | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 |
| Total recall | 32/32 (100.00%) | 15/32 (46.88%) | 19/32 (59.38%) | 31/32 (96.88%) | 28/32 (87.50%) |
| False positives: pass | 89/112 | 10/112 | 112/112 | 10/112 | 4/112 |
| False positives: waiting | 394/439 | 298/439 | 20/439 | 4/439 | 4/439 |
| Total false positives | 483/551 (87.66%) | 308/551 (55.90%) | 132/551 (23.96%) | 14/551 (2.54%) | 8/551 (1.45%) |
| Escalated share | 0.00% | 0.00% | 25.90% | 5.66% | 4.12% |
| Meets G3 criteria | NO | NO | NO | Yes (in-sample) | NO |
| Exact-class recall: crooked_cap | 12/19 | 0/19 | 0/19 | 0/19 | 0/19 |
| Exact-class recall: missing_cap | 12/12 | 12/12 | 0/12 | 0/12 | 0/12 |
| Exact-class recall: missed_placement | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 |

`damaged_vial`: no observations, no recall estimate; remains out of scope. Pass and waiting are non-defect classes, reported as false positives above.

**B″ G3: NO-GO.** It misses three crooked observations and one missing cap; recall 28/32 = 87.5% is below 90%. Its 8 false positives split into 4 pass and 4 waiting. Escalation is 24/583 = 4.12%. No parameter changed after this result.

## B″ misses and false positives

| Kind | Run | Event | Position | Ground truth | Sequence | Visual | Final | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Miss | 1 | 8 | G07 | missing_cap |  | no_vial | WAITING | 1.0 |
| Miss | 1 | 14 | G13 | crooked_cap |  | pass | PASS | 0.4403 |
| Miss | 1 | 16 | G13 | crooked_cap |  | pass | PASS | 0.4721 |
| Miss | 1 | 18 | G13 | crooked_cap |  | pass | PASS | 0.4509 |
| False positive | 1 | 6 | G16 | no_vial | waiting | unknown | UNSURE | 0.9098 |
| False positive | 1 | 7 | G06 | pass |  | unknown | UNSURE | 0.8647 |
| False positive | 1 | 8 | G06 | pass |  | unknown | UNSURE | 0.8992 |
| False positive | 1 | 9 | G06 | pass |  | unknown | UNSURE | 0.8753 |
| False positive | 1 | 10 | G06 | pass |  | unknown | UNSURE | 0.8674 |
| False positive | 1 | 16 | G20 | no_vial | waiting | unknown | UNSURE | 0.8568 |
| False positive | 1 | 17 | G20 | no_vial | waiting | unknown | UNSURE | 0.8515 |
| False positive | 1 | 18 | G20 | no_vial | waiting | unknown | UNSURE | 0.8594 |

## Saved reports

- `results/g3/resnet18_v1/g3_report.json` — SHA256 `6a1e4825b8292d0034dbc0d68af775eb7fc022baecaaf57d6479d59d889ecae9`
- `results/g3/rule_synthetic_v1/g3_report.json` — SHA256 `132b4fbf2b60f9da64695a0f04ae65cf5cf67ab3bfdaab5eae4345f52ef7eedf`
- `results/g3/rule_dev_good_only/g3_report.json` — SHA256 `ce1900afd202c03aa4a4a5f8238575ed6d29262c5f327118b9f5b7837e109d37`

Complete command output: [RC transcript](results/rc_1024/COMMAND_OUTPUTS_CONTINUED.txt), commands `26_bpp_fit_once` and `28_bpp_g3_once`. Sunday freeze must include B″ and its hash; existing draft manifest is not a final freeze.
