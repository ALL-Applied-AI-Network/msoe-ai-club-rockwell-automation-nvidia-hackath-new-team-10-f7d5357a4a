# CSI hand-staged collection — staging kit (DRAFT, revised 2026-10-16)

Preparation only. Nothing here is booked: request time in `#data-and-testbed-requests` and tag Shamar Webster
(or webste63@uwm.edu). Collection happens only after CSI approves it and assigns a supervisor. Unit = independent
staged event (D011), never a frame.

## What changed on Oct 16 (organizer feedback)
- The line has **no placement count or order-complete signal**. MISSED vs WAITING is decided by `fill_state.py` from
  occupancy (D021). The stager still records the true sequence state as **ground truth** (`gt_sequence`), which is
  never a runtime input.
- Completion context comes from this protocol, not the equipment. Before capture, the stager records
  `inspection_mode`:
  - `final_inspection`: the fixture is declared complete, so every empty position is MISSED;
  - `live_fill`: a partial fixture, so trailing empties are WAITING.
- **Position names are the twin's G01–G30** (G01 top-right, down the column, then one column left). CSI's internal
  numbering is different and not available, so never write "position 7" for G07 (D022).
- **Crooked cap = any cap not fully seated** (organizers). Our refinement: target visible tilt or displacement of
  about 15–20° or more. Shallower tilt or lift is recorded and reported as the overhead camera's information limit (D023).
- **Training rule is unresolved** (D025). The organizer guides say supplied real data is "to look at, not to train on".
  Until organizers answer, Session A frames are for characterization, annotation checks and evaluation design.
  They are not a training set, and no `adaptation` designation is used.
- A **supervised live run in Session B counts as the real-hardware demo** if our system runs on the real line (D029).
- Teams may not move, re-aim or reconfigure an existing camera. A side phone photo of a crooked cap needs our
  own-equipment setup to be reviewed first (Collect-and-Test guide).

## Files
- `staging_sessions_template.csv`: one row per session (role, camera, hole check).
- `staging_log_template.csv`: one row per G position per staged event.

## 1. Start of every session: camera and empty-fixture check (record, never tune)
1. Record which physical overhead camera, feed and resolution CSI provides today; CSI has not confirmed this yet.
   The twin map is solved for the 640×480 cap_seal view. The organizers' labeling-tool examples use a different
   1920×1080 view. If today's feed is not the 640×480 cap_seal view, flag it to Noor before staging: the twin crop
   map cannot be used on that view until its camera projection is solved.
2. G fixture empty, arm clear. Save one frame and note its file name.
3. From `twin_to_line/`:
   ```bash
   ../.venv/bin/python -c "
   import sys, numpy as np, positions as P, register_dev as RD, real_data as R
   img = R.Index().load(sys.argv[1]); r = RD.hole_check(img, P.PositionMap()); d = [float(np.hypot(*v)) for v in r.values()]
   print(f'{len(r)}/30 holes matched, median {np.median(d):.2f} px, p90 {np.percentile(d, 90):.2f} px')" <empty_frame.jpg>
   ```
4. Record the result in `staging_sessions_template.csv`. DEV reference (run 3, Sept 23 pose): 30/30, median 1.68 px,
   p90 2.53 px. If it is clearly worse, flag it before staging. Never change the position map, offset or thresholds
   from session data.

## 2. Session A: development (first visit)
Purpose: camera-limit characterization, annotation verification, and the evaluation protocol. Model fitting happens
only if the training rule allows it, and is decided before the session.
- Known-angle crooked caps: about 10°, 15°, 20° and 30° tilt, plus displaced caps. A side photo and an approximate
  severity for each (side photo only if our own-equipment setup was approved).
- Missing caps (uncapped vials), skipped positions (internal empties), all-good complete fixtures.
- Hard negatives: pale or glary caps, a leaning vial with a properly seated cap, a seated cap slightly off-centre,
  neighbouring caps that spill into a crop.
- Partial fixtures with trailing empties, to check that WAITING is never raised as a false alarm.

## 3. Session B: final test (different day)
- Touch nothing from Session B until the model, thresholds, synthetic recipe and process logic are frozen and their
  hashes are recorded.
- Target, if CSI time allows: about 15 independent events per major class (missing_cap, crooked_cap, missed
  placement), plus all-good fixtures, hard negatives, partial fixtures and non-adjacent multiple defects.
- Independence: one event = one staged physical state. Adjacent or repeated frames of the same staged state are not
  separate examples, and re-seating the same cap at the same position counts once unless the cap is fully removed
  and re-staged.
- The supervised live run during Session B is the real-hardware demonstration. Keep a trial log of every attempt:
  software version, model hash, input condition, prediction, expected result, latency, operator intervention.

## 4. Before staging each event
- Write `designation` (`session_a_dev` or `session_b_final`) **before** staging, never after seeing a result.
- `session_b_final` frames go to their own folder and are never opened for development (D011).

## 5. Per staged event (`staging_log_template.csv`)
| Column | Meaning |
| --- | --- |
| `session` | session id (matches `staging_sessions_template.csv`) |
| `event_id` | staged event number within the session (the independence unit) |
| `frame_id` | camera file name of the first arm-clear frame of the staged state |
| `capture_time` | local ISO time of `frame_id` |
| `position` | Twin G01–G30 (not CSI numbering); record all 30 positions per event |
| `staged_state` | what the stager physically set: `pass`, `missing_cap`, `crooked_cap`, `displaced_cap`, `no_vial`, `damaged_vial` |
| `defect_class` | official visual class: `pass`, `missing_cap`, `crooked_cap`, `damaged_vial`, `no_vial` |
| `inspection_mode` | `live_fill` or `final_inspection` (fixture declared complete by this protocol), set before capture |
| `gt_sequence` | for `no_vial` only: `missed` (skipped inside the order) or `waiting` (not yet filled); stager ground truth, never a runtime input |
| `known_tilt_deg` | measured cap tilt for `crooked_cap` when known; blank otherwise |
| `approx_severity` | `shallow` (<15°), `medium` (15–25°), `strong` (>25°) or `displaced`, for `crooked_cap` |
| `side_photo_id` | side photo file name for crooked caps, if the setup was approved |
| `staging_start`, `staging_end` | local ISO time the staged state was set up and released |
| `designation` | `session_a_dev` or `session_b_final`, fixed before staging |
| `notes` | lighting/glare, neighbour interference, hard-negative type, anything unusual |

## 6. After the session
- Check: no blank `designation`; `known_tilt_deg` / `approx_severity` only on `crooked_cap`; `gt_sequence` only on
  `no_vial`; classes from the lists above; 30 rows per event.
- Keep Session A and Session B frames in separate folders, and record which is which in the session notes.
- Raw files stay unchanged. Labels and notes go in a separate location (not git). Record who collected them.
