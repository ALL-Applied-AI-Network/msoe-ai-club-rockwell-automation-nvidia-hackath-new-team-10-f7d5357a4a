# Organizer feedback integration — 2026-10-16

What changed after Derek Riley (NVIDIA) reviewed the Team Ignite plan, what was checked locally, and what stays
blocked. The Isaac appearance gate is still **gate-v2.0.2, frozen, not run; verdict INCONCLUSIVE** (no NVIDIA compute).
Runs 4–5 were not opened.

**Gatekeeper state (GPT review, Oct 16): RUN FROZEN ISAAC GATE WHEN COMPUTE ARRIVES.** The gate does not depend on
the training-data rule. Model training and the real-data dose experiment remain blocked until the organizer clarifies
whether newly collected CSI real data may be used for fitting, fine-tuning or calibration. Repository migration and
push remain blocked until Tom clarifies team ownership.

| Status | Items |
| --- | --- |
| Confirmed | gate-v2.0.2 frozen; no valid Isaac render yet; runs 4–5 protected; supplied real images are not training data; the organizer labeler is manual review, not ML autolabeling; the organizer grid convention matches our Twin G01–G30 orientation and order; no placement count or order-complete signal from the equipment; CSI numbering differs and is unavailable |
| Strong physical mapping evidence | real 640×480 empty fixture: 30/30 holes, ~1.68 px median, no per-position manual teaching |
| Supporting evidence | organizer-grid consistency check (section 3): layout, orientation, numbering, topology |
| Unknown | the Session A/B camera and feed; whether newly collected real data may be used for fitting or calibration; final repository ownership after the team reassignment |

## Sources
- Derek Riley's review of the submitted Team Ignite plan, relayed by Noor on Oct 16 (summarized in `HACKATHON_HANDOFF.md`).
- Organizer guides, read Oct 16: the event overview `msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026`, Dataset 1
  `...-Vial-Inspection`, and Collect-and-Test `...-Data-Collection`.
- `~/Downloads/vial-labeler-participants-20261014.zip` (sha256 `7fddb29e…0072d0`, version `participant-1.0`,
  `SHA256SUMS` 80/80 OK). It was extracted outside the repo, and nothing from it is committed.

## 1. Dataset 1 assets (Task 1)
**The new labeled/unlabeled crop release is not on this Mac.** The Dataset 1 page says "Dataset and labeling tool: the
download will be linked in #dataset-1-vial-inspection on Slack" (also in the "Innovation Lab data folder"). The only
local real data is `csi-vial-fill-dataset` (Oct 5): 640×480 cap_seal frames, with labels for vial_fill video 5 only.
Everything below comes from the labeler package. Items marked BLOCKED need the release.

| # | Question | Answer |
| --- | --- | --- |
| 1 | Files found | `vial-labeler-participants-v1/`: `labeler.py`, `rules.py`, `report.py`, `calibration.json`, `README.md`, `PACKAGE.json`, `SHA256SUMS`, `requirements.txt`, `tests/`, `examples/tray_000001.jpg`, `examples/tray_000122.jpg`, `demo/` (one unlabeled review of tray_000001: 60 crops, `labels.json`, `slots.csv`, `manifest.json`, `index.html`, annotated scene, contact sheets) |
| 2 | Formats | JPEG scenes 1920×1080 (no EXIF, no burned-in timestamp); crops JPEG; `labels.json`, `slots.csv` (`image,slot,area,position,class,presence,review_status,reason`), `manifest.json` |
| 3 | Label schema | `pass`, `missing_cap`, `crooked_cap`, `damaged_vial`, `no_vial`; `null` = unresolved (not a sixth class); `review_status` draft/confirmed; multi-defect priority damaged > crooked > missing |
| 4 | Samples per class | **BLOCKED** (release not local; the demo crops are all unresolved) |
| 5 | Unique source frames | **BLOCKED**; at least 122 tray images exist (`tray_000122`); the README says "keep related public images 1 and 6 together" |
| 6 | Unique positions | 60 per image: G01–G30 + S01–S30 |
| 7 | Crop → full scene | Yes: each slot stores `center`, `bbox` and `crop_path`, and the result stores the source image and its sha256 |
| 8 | Coordinate representation | Pixel centre per position, from **4 hand-placed tray-corner hole centres** per tray, bilinearly interpolated; crop box = centre + fixed offsets (good −38/−47/+42/+43; scrap −28/−34/+143/+96). The README calls the boxes "review context crops, not tight object boxes" |
| 9 | Image dimensions | 1920×1080 only (the tool rejects anything else); two calibration profiles `view_001`, `view_122` (translation ≈ 2/18 px apart) |
| 10 | Position-ID convention | Six columns of five; start at the top right, count down, then one column left; G = good, S = scrap |
| 11 | Matches our G01–G30? | **Yes**: same rule, and the organizer-grid consistency check shows the same orientation and order (+3.6° rotation, no mirror or flip; section 3). CSI's internal numbering is still different and unavailable (D022) |
| 12 | `damaged_vial` meaning | Guide: "The target vial body is visibly damaged." Labeler: "crushed, torn, broken or visibly deformed". No confirmed example seen (section 7) |
| 13 | Temporal duplication | Likely: "related public images 1 and 6" must be kept together. Splits must be by image group, not by crop |
| 14 | Overlap with runs 4–5 | No evidence of overlap: different camera view (1920×1080, rotated about 3.6° and a different scale vs the 640×480 cap_seal frames), no timestamp overlay. The capture dates are unknown, so treat the release as a separate source; ask organizers before any evaluation claim |

**View difference.** The newly supplied organizer labeling-tool examples use a different 1920×1080 overhead view of
the same station: wider, and rotated about 3.6° relative to the 640×480 cap_seal view that the twin's `CapSealCam`
(and our position map) was solved for. **CSI must confirm which physical camera and resolution will be available for
staged Session A/B capture**; nothing yet says which one it will be.
- If CSI uses the existing 640×480 cap_seal view: the organizer release implies no camera re-solve.
- If CSI uses the newer 1920×1080 view: the camera registration/projection must be solved for that physical view
  before the twin-derived crop map is used. It is not being solved now, because the Session B camera is unknown.

The frozen gate is unaffected either way: it compares renders with 640×480 cap_seal frames.

`tray_000122` (my visual read, not labels) looks like a deliberately staged defect scene. G01, G14, G16 and G23 have
strongly tilted or knocked-over caps. G04, G05, G13 and G22 are empty while later positions are filled. That makes it
the kind of scene our staged-session plan wanted.

## 2. NVIDIA labeling tool (Task 2)
- **Not an autolabeler.** "This standalone tool creates an offline browser page for manual labeling … It makes no
  class predictions". In code: `backend='manual_annotation'`, `suggested_class=None`, every slot starts `class: null`.
- Python 3.10+, NumPy + Pillow. No model weights, no API key, no network calls. The review page is self-contained
  HTML (images embedded as base64) and stores state only in browser `localStorage`. Labels round-trip through
  "Export review JSON" and `labeler.py import-review`, which checks hashes, geometry and states.
- Outputs use relative paths and contain no absolute paths or machine names. `model_hash` is a fingerprint of the
  calibration plus `labeler.py` and `rules.py`.
- Alignment: a translation search of ±18 px over two anchor patches against the reference images. No rotation or
  scale. "Unfamiliar alignment is rejected." The README says to visually check every ring.
- `rules.py` has the same fill-order semantics as ours. It says a last-slot drop is unobservable without a placement
  counter.
- Licence: "This package does not add a new software or dataset license; any redistribution remains subject to the
  event's supplied terms." It is not redistributed here.
- **The DEV comparison is not possible.** The tool refuses 640×480 images and produces no predictions, so there is
  nothing to reconcile with `labels/dev_labels.csv`. Agreement, confusion and mismatch clustering: N/A.
- Workflow: use it as the **human adjudication** step (Noor or teammates confirm classes in the browser), and keep
  our draft labels as a separate opinion. Its labels are only as good as the reviewer.
- Risks: crops are context windows (neighbours spill in, and tall vials lean visibly in this view, so caps sit well
  off the hole centre near the edges). The 4-corner bilinear grid ignores the lens's barrel distortion. Browser
  storage can be lost unless the JSON is exported.

## 3. Organizer-grid consistency check (Task 3)
`check_organizer_grid.py` (read-only; `position_map_G.json` is never written).

**This is not an independent 30-point physical validation of the twin map.** The organizer tool builds its 30-position
grid from 4 manually defined corner centres and interpolates the rest bilinearly. Our comparison aligns the frozen twin
map to those same 4 corners, so the residuals measure how consistent the twin's layout is with the organizers'
grid convention.

The method was fixed before scoring:
- The organizer centres come from the labeler's own bilinear formula.
- A 4-DOF similarity (scale, rotation, translation) is fitted twin → organizer grid on the 4 tray corners.
- The other 26 positions are interpolated by the organizer tool, so agreement there reflects layout shape, not 26
  extra measurements.
- Primary twin plane: `plate_hole_center_px`, because the labeler's rings mark the target insertion hole.

| Profile | Matched | Scale | Rotation | Median | p90 | Max | Median ÷ scale (≈ cap_seal px) | Signed bias (dx, dy) | Outliers (> 5 px) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| view_001 (tray_000001) | 30/30 | 2.252 | +3.60° | 1.23 px | 2.09 | 3.80 | 0.54 | +0.93, +0.22 px | 0 |
| view_122 (tray_000122) | 30/30 | 2.262 | +3.59° | 1.50 px | 2.42 | 3.80 | 0.67 | +0.93, +0.22 px | 0 |

- Corner residuals: view_001 G01 1.16, G05 0.90, G26 0.91, G30 0.54 px. view_122 0.68 / 1.22 / 1.86 / 2.12 px.
- Other twin planes are in the same range (median 1.22–1.67 px, max ≤ 4.94 px).
- Numbering: no mirror needed and rotation +3.6°, so the conventions match. The unit tests show that a 180° or
  mirrored numbering is caught.
- Overlays (gitignored): `results/organizer_grid/overlay_view_*.png`. They show the twin's G01–G30 crosses inside the
  organizer rings.

**What this does and does not show.**
- **Shows (supporting evidence):** the frozen twin fixture topology, spacing (row/column pitch ratio 1.177 vs 1.178),
  orientation and G01–G30 ordering are highly consistent with the organizer labeling tool's grid convention.
- **Does not show:** that organizer coordinates independently validate our physical position-map accuracy to 1.23 px.
  The fit uses the organizers' own 4 corners, and the absolute accuracy of our 640×480 map cannot be tested in a
  different camera's view.
- **The main evidence for the no-teach claim is unchanged:** on a real 640×480 cap_seal image of the empty fixture,
  the twin-derived map found 30/30 holes at about 1.68 px median error, with no manual per-position teaching. That is
  why twin-derived geometry substantially removes manual position teaching. The consistency check supports layout,
  orientation, numbering and topology only.
- Contrast: the organizer tool needs 4 hand-placed corners per tray, per view, plus an alignment reference image.
- The twin map was not changed.

## 4. Sequence logic without a placement count (Task 4)
Derek: placement count and order-complete are not available from the manufacturing equipment, so the runtime cannot
know by itself that a fixture is complete. `fill_state.py` runs from occupancy, with optional temporal history, in two
explicit operating contexts (D021, `mode=`):

**LIVE_FILL** (default; the fixture may still be filling):
- **MISSED**: an empty position before the highest visibly occupied position F (production has passed it).
- **WAITING**: empty positions after F. The first one (F+1) is the ambiguous frontier, either not attempted yet or just
  skipped. The default is WAITING with a "fill frontier" reason; `frontier="unsure"` makes it UNSURE.
- **UNSURE**: unreadable positions (occlusion, glare, crop-overlap ghosts). They are flagged on their own and never
  move F.
- **Temporal** (`SequenceTracker`): an empty position becomes MISSED once a later position has been occupied for
  `persist` valid frames (default 2, or 1 for event frames). Until then it is UNSURE, so a one-frame flicker never
  latches. A confirmed vial that later reads empty is UNSURE.
- Example: G01–G08 occupied, G09 empty, G10 occupied → G09 = MISSED. But G01–G08 occupied and G09–G30 empty → G09 is
  the frontier (WAITING) and the rest are WAITING. Nothing is promoted to MISSED while filling may continue.

**FINAL_INSPECTION** (`mode="final_inspection"`, CLI `--mode final_inspection`): used only when the workflow or test
protocol explicitly says "this fixture is now expected to be complete". In CSI Session B that context comes from the
controlled staging protocol or the operator, never from the manufacturing equipment, and the system never infers it
from the image. A complete good order has 30 vials (Dataset 1 guide), so every empty position, trailing ones
included, is MISSED. There is no PLC or `order_complete` signal, and none is assumed.

A placement count K is optional, in either context. It is offline metadata (synthetic scenes, staged ground truth)
or the camera's own classifier-free event counter (`events.py`, video only), and never comes from the equipment.

DEV check (LIVE_FILL): runs 1–3, the rule baseline, 20 events, 583 usable draft-labeled position observations (32
defect, 551 non-defect: 112 pass + 439 trailing empties; 17 `usable=0` rows dropped by `eval_metrics`). Thresholds
and labels are unchanged; this is reporting, not tuning, and not a holdout result. A false positive is a non-defect
position flagged DEFECT, MISSED or UNSURE.

| Sequence mode | Defect recall (obs.) | False positives / non-defect obs. | Good-vial false-reject rate | Trailing-empty false alarms |
| --- | --- | --- | --- | --- |
| `--k-from-events` (previous mode, video event count) | 32/32 | 14/551 = 2.5% | 10/112 = 8.9% | 4/439 = 0.9% |
| occupancy only (new default) | 31/32 | 14/551 = 2.5% | 10/112 = 8.9% | 4/439 = 0.9% |
| occupancy + tracker | 31/32 | 14/551 = 2.5% | 10/112 = 8.9% | 4/439 = 0.9% |
| tracker, frontier = unsure | 31/32 | 26/551 = 4.7% | 10/112 = 8.9% | 16/439 = 3.6% |
| first draft: UNSURE spread to every empty before an unreadable position | 31/32 | 33/551 = 6.0% | 10/112 = 8.9% | 23/439 = 5.2% |

**Independence caveat.** Observations are not independent. Each event re-observes the whole fixture, so the same
physical vial or empty hole appears in many events.
- By unique physical position-state (run, position, ground-truth state), the default mode flags 4 of 98 non-defect
  units at least once: 2 specular top-row caps (run 1 G06, G11) and 2 empty holes with a neighbour's displaced cap
  intruding (run 1 G16, G20). They account for all 14 false-positive observations.
- Defect units: 8 (7 cap-defect units + 1 missed placement). All 8 are flagged at least once, and 7 of 8 at every
  observation.
- Event-level false-alarm rate: 11 of 20 events contain at least one false alarm. Events within a run share the same
  fixture, so this is not an independent-event rate. No reliable event-level independence can be computed from the
  DEV data.

- The last row is why unreadable positions no longer spread. On DEV that rule flagged 19 extra trailing positions
  behind crop-overlap ghosts. The change was made after seeing DEV results and is recorded here.
- The single FN is run 1, event 8, G07: an uncapped vial that the rule baseline reads as `no_vial`, next to a pale
  specular cap (G06, `unknown`). Without a count it is indistinguishable from waiting. The tracker marks it MISSED at
  the next placement (event 9).
- This is the "uncapped vial vs empty hole" case the organizers list, and it is a target for the learned model.

## 5. Training-data rule (Task 5)
Exact organizer wording (read Oct 16):
- Event overview: "All of the data and digital twins are in the Innovation Lab data folder. The real data from the
  line is there to look at, not to train on: teams generate their training data with the digital twins, as a
  high-mix, low-volume line would have to."
- Dataset 1, "Look at the real data, but train on the digital twin": "The sample images come from the real line. Look
  through them to understand the station, the defects, and what your generated images need to match, but avoid
  training your models on them. The goal of this lab is to generate your training data with the digital twin."
- Dataset 1, Testing: "Keep a test set that is not used for prompt examples, augmentation decisions, or training.
  Report results per class, not only overall accuracy." "Test at least a few complete fixture scenes, not only
  individual crops."
- Collect-and-Test, Synthetic data: "It should extend real data, not replace all real testing." "Compare performance
  on real-only, synthetic-only, and combined training data. A synthetic improvement counts only if it carries over to
  held-out real examples."
- Labeler README: "Use only confirmed labels for training or evaluation."
- Derek (Oct 16): "Please confirm the training-data rules before proceeding with the proposed real-only and mixed
  real/synthetic training comparisons."

| Use of real images | Supplied Dataset 1 data (runs 1–5, public release) | Newly collected CSI data (Session A/B) |
| --- | --- | --- |
| A. Real-only training (5/10/20) | **PROHIBITED** ("not to train on"; "avoid training your models on them") | **UNCLEAR** (Collect-and-Test suggests real-only vs synthetic vs combined; the overview says train on the twins) |
| B. Fine-tune a synthetic model with 5/10/20 real | **PROHIBITED** (same wording) | **UNCLEAR** |
| C. Threshold tuning / calibration | **UNCLEAR** (not addressed) | **UNCLEAR** |
| D. Development / debugging | **ALLOWED** ("Look through them to understand the station, the defects, and what your generated images need to match") | ALLOWED (collection exists to answer questions) |
| E. Final evaluation only | **ALLOWED** ("Keep a test set …"; improvements must carry over "to held-out real examples") | **ALLOWED** |

**BLOCKED — ORGANIZER CLARIFICATION REQUIRED** for A, B and C on newly collected data, and C on supplied data.
Question for Noor to send (one message):

> For Dataset 1, the guides say not to train on the supplied real images, while the Collect-and-Test page suggests comparing real-only, synthetic-only and combined training. May we train or fine-tune on real images we newly collect at CSI (for example 5/10/20 staged defects), and may we calibrate decision thresholds on real development images, or must all model fitting and calibration be synthetic with real images used only for development and evaluation?

Camera question for CSI (Noor sends it; also not sent automatically):

> For our supervised Dataset 1 staging sessions, which physical camera/feed and resolution should we expect to use? Our existing twin registration is solved against the 640×480 cap_seal view, while the new organizer labeling-tool examples appear to use a different 1920×1080 overhead view.

Consequences already visible:
- D003's and D014's reliance on "any data mix is allowed" (Tanner, Oct 13) is superseded for supplied data (D025).
- The runs 1–3 "adaptation" permission in `real_splits.json` is void until the answer arrives.
- `rule_thresholds.json` was fitted on DEV draft labels (`rules.py --fit`). If calibration on supplied data is ruled
  out, report the rule baseline as DEV-calibrated, or re-derive its thresholds from synthetic renders.
- The project still works in all three of Noor's cases. If real fitting is barred, the experiment becomes: rule
  baseline vs synthetic-trained ResNet-18, with real images for validation and evaluation only.

## 6. Session A / B changes (Task 6)
See `staging/STAGING_CHECKLIST.md`:
- The placement-count column is replaced by stager ground truth `gt_sequence`.
- Twin Gxx naming is stated as our own convention.
- The crooked-cap definition and our severity refinement are added.
- Each session starts with a camera/resolution check.
- Session B targets about 15 independent events per major class, plus hard negatives.
- The Session B live run is the hardware demo.
- Designations are now `session_a_dev` / `session_b_final`; no `adaptation` while D025 is open.
- Booking goes through `#data-and-testbed-requests`, tagging Shamar, with nothing assumed approved.

## 7. Damaged vial (Task 7) — **EXPLICITLY OUT OF SCOPE for v1**
1. Count: **BLOCKED** (release not local). Zero confirmed examples in the two public images; nothing in them is clearly
   a crushed or torn body.
2. Appearance: "crushed, torn, broken or visibly deformed" transparent bodies. From overhead, the cap hides most of
   the body, and neighbouring transparent bodies overlap (the guide warns of this).
3. Camera: the 640×480 cap_seal view shows a cap top plus a thin body ring (cap ≈ 21 px wide). Body damage below the
   cap is unlikely to be resolvable. Unverified until real examples exist.
4. Twin: defects are authored as USD override layers on rigid tube meshes (D012). Crushing or tearing needs new
   mesh-deformation authoring and new materials, which is a new capability.
5. Value: low for the core thesis (commissioning effort and data efficiency). The cap and occupancy classes carry it.
6. Cost and risk: high (new authoring, no real examples to check against).

`damaged_vial` stays in the label vocabulary, `fill_state.py` and `eval_metrics.py`, so any real occurrence is still
reported as DEFECT or UNSURE and counted. Revisit only if the release shows clear, frequent examples.

## 8. GB10 — see `GB10_PREFLIGHT.md`
No access yet (Shamar administers UWM GB10s). Brev remains the fallback. The frozen `run_gate.sh` may not run
unmodified on aarch64 (`usd-core` wheel). The preflight checks this, and if it fails we stop and report rather than
edit the kit.

## 9. Nemotron (Task 9, note only)
There is no measured reason to use NVIDIA-Nemotron-3.5-Bolt yet. One candidate failure mode: on usable DEV rows the
rule baseline flags 10 PASS and 4 WAITING positions as UNSURE. The causes are specular or off-centre top-row caps,
and a neighbour's displaced cap intruding into an empty crop.
- Benchmark: send only UNSURE crops plus full-scene context, then measure the share resolved and the change in
  false-alarm rate at 100% defect recall, against the ResNet-18 alone, on a fixed set.
- Keep it only if it cuts false alarms without losing a single defect, at an acceptable latency.
- Not installed, integrated or run.

## 10. Team reassignment impact (Noor → Code Crusaders, Oct 16)
- **Portable unchanged:**
  - `positions.py`, `fill_state.py`, `eval_metrics.py`, `real_data.py` (data guards), `events.py`, `rules.py`,
    `dose_sets.py`, `capsealcam.py`, `export_position_map.py`, `check_organizer_grid.py`;
  - the gate kit as a unit (`MANIFEST.sha256` paths are relative);
  - `staging/`.
  These need numpy, OpenCV and the twin/data paths only.
- **Repo-coupled:** `HACKATHON_HANDOFF.md` (Team Ignite roster, repo URL, deadlines), the `README.md` run commands
  (paths relative to `twin_to_line/`), the repo-root organizer README, `.gitignore`, the decision-doc artifact link,
  and `real_splits.json` (Team Ignite's split policy).
- **Preserve exactly:**
  - git history, including `c673212` (gate kit v2.0) and `2e56ab0` (gate-v2.0.2 frozen before any render). The
    pre-registration claim depends on these timestamps;
  - `MANIFEST.sha256`, `KIT_VERSION`, `GATE_v2.0.2.md`;
  - `PROTOTYPE_SNAPSHOT.sha256` and its four files;
  - `real_splits.json`, including the `already_touched_held_out` list;
  - `labels/dev_labels.csv` (draft) and `DECISIONS.md`.
- **Do not migrate yet:** anything. A copy without history would lose the gate's freeze timestamps, and a merge
  could mix incompatible data boundaries.
- **Possible conflicts:**
  - the camera/view (640×480 cap_seal vs 1920×1080);
  - position numbering;
  - model family (ResNet-18 crop classifier vs, for example, TAO ChangeNet, a detector or a VLM);
  - the training-data source (they may already train on supplied real images, which the guide discourages);
  - the holdout definition and use of runs 4–5;
  - process logic (the fill-order states);
  - the primary claim (commissioning effort vs detection accuracy).
- **Need from Code Crusaders:**
  - their repo and plan;
  - Derek's feedback on their plan;
  - which camera, resolution and dataset version they use;
  - model and training data;
  - test-set discipline;
  - what currently runs end to end, and on what hardware;
  - compute access (Brev, GB10);
  - Isaac Sim / Replicator status;
  - position naming;
  - who owns what and their weekly hours.
- **Safest next Git action:** keep `feat/organizer-feedback-integration` local and unpushed. Optionally make an
  offline backup with `git bundle create ~/ignite-$(date +%F).bundle --all`. Wait for Tom's answer on repository
  ownership.
