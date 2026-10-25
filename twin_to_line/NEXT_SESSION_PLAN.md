# Twin-to-Line — Next-Session Plan (Thu Oct 22 → Mon Oct 26 live test)
Updated 2026-10-22, 2:45 pm CT. Two copies: Project doc `claude/next-session-plan.md` and repo
`twin_to_line/NEXT_SESSION_PLAN.md` (added in step 0). Read this first, then `HACKATHON_HANDOFF.md` and `DECISIONS.md`
(D001–D036). The handoff's standing rules apply (no secrets in files, no AI attribution, no push, draft messages for
Noor, never touch his `dose_sets.py` edit). Everything about CSI is **unconfirmed** until Shamar confirms.

## The short version
On **Mon Oct 26 at 1 pm** (requested, not confirmed) Noor and Tony stage 22 test images on the real Good fixture, then
run our software live on the real camera. That gives us the two things the submission cannot do without:
1. a **sealed real test set** that shows whether an inspector trained only on twin images works on the real line, and
2. the **hardware demo** the judges require (Derek: a supervised live run counts if our software runs on real hardware).

For that, by **Sun Oct 25, 9 pm** we need, all frozen: the camera answer from CSI, a reviewed GPU gate, training renders,
a trained and thresholded inspector, and a live runtime tested on saved images. **Today's only job is the GPU gate.**

## 0. Where to run the next session
- **Claude Code on Noor's Mac, in `~/code/crusaders`**: anything that runs commands (git, tests, Brev CLI, renders,
  training, the runtime). The Claude app in the cloud cannot type into the Mac's Terminal or reach the CSI VPN.
- **Claude app, Project "hackathon"**: messages to CSI, team and organizers; reviews; doc updates.
- Start prompt for Claude Code (after step 0):
  > Read twin_to_line/NEXT_SESSION_PLAN.md, then twin_to_line/HACKATHON_HANDOFF.md and twin_to_line/DECISIONS.md.
  > Run `git status -sb`. Do step 1 of the plan (the GPU gate) exactly as the handoff's NEXT ACTION says. Ask me before
  > anything that costs money, pushes to GitHub or sends a message.

## 1. Where things stand (Oct 22, 2:45 pm)
- **Done:** scene generator verified (91/91 tests); Brev org access (no instance yet); YAAD team joined, VPN + SSH work;
  Shamar emailed a short request; staging list doc written; Discord update posted; handoff, D036 and this plan written.
- **Not done:** GPU gate (no renders exist); training renders; any trained model; the live runtime; the freeze.
- **Waiting on:** Shamar (session + camera resolution + camera access); Vang (what "the m8 inspection pipeline" is);
  a live-runtime owner.

## 2. Requirements for the Oct 26 live test
### 2.1 Derek's guidance (Oct 16 email)
- Our direction is aligned; "don't hold yourself to this strategy only".
- Confirm training-data rules before any real-only or mixed real/synthetic training. We now train on twin renders only.
- Dataset 1 already has full overhead images of both fixtures, labeled/unlabeled crops, coordinates/position IDs, labels
  `pass`, `missing_cap`, `crooked_cap`, `damaged_vial`, `no_vial`, and a browser labeling tool.
- CSI requests: post in `#data-and-testbed-requests` tagging Shamar or email him, with test plan, duration, attendees and
  dates. CSI must confirm the booking and supervision.
- A supervised live run counts as the real-hardware demonstration "as long as your implementation is running on real
  hardware".
- No placement count or order-complete signal exists. CSI's hole numbering differs; use the twin's G01–G30.
- Crooked cap = any cap not fully seated; a defined refinement is allowed. Ours: **≥20°, pre-registered** (D036).

### 2.2 CSI Data Collection guide (fetched Oct 22)
- **May:** visit, record with our own equipment after the setup is reviewed, ask CSI to collect specific images, request
  supervised time to test a finished system.
- **May not:** move, re-aim, disconnect or reconfigure a camera; change a PLC, robot controller, FactoryTalk, safety
  system, recipe or program; connect hardware to the industrial network without approval; place objects where they could
  collide with the robot; film people or sensitive information.
- **Before any physical test:** freeze the code and model version; document required inputs and expected outputs;
  prepare a start and stop procedure; decide the metrics; test failure handling with saved data; make sure it runs
  without changing plant controls.
- **Trial log:** record every attempt, not only successes: software version, model, input condition, prediction,
  expected result, latency, operator intervention.
- Testing happens only after CSI confirms the time and assigns a supervisor. Extra collection is not guaranteed.
- Synthetic data: keep the source and generation settings; a synthetic improvement counts only if it carries over to
  held-out real examples. Data handling: raw data unchanged; labels and generated files stored separately; record who
  collected each set; no datasets, checkpoints or credentials in GitHub.

### 2.3 Dataset 1 page (scoring rules)
- Real images are for understanding the station and what renders must match: "avoid training your models on them".
- No test set is provided: keep our own, separate from prompts, augmentation and training.
- Report per class; report false alarms and misses separately; show false negatives per defect class; test complete
  fixture scenes; report the **false-alarm rate at the threshold that catches every known defect**; check the model
  reacts to defects, not pose, glare or lighting; change one thing at a time. **A missed defect matters more than a
  false alarm.**
- `crooked_cap`: visibly tilted or not seated; a leaning vial alone is not enough. `missing_cap`: don't confuse the clear
  vial body with an empty hole. `no_vial`: whether it's an error depends on fill progress (our MISSED vs WAITING).

### 2.4 What the session must produce
- **Staged test set:** 22 images (orientation check + 21 setups) per the staging list doc
  https://claude.ai/artifact/P1WRYUf2vJb2kUR2XPHTKh — 14 tilted caps ≥20° (7 at 20°, 7 at 30°), 3 tilted below 20°,
  14 missing caps, 14 missed placements, 6 leaning vials with seated caps, 37 waiting holes. Plus the log CSV (ground
  truth), side photos of tilted caps (if approved), session start/end and per-setup times (business case).
- **Live run:** our frozen software reads the camera while the robot fills the fixture; trial log; screen recording;
  phone video of the line without people (if approved). Optional live-defect part (if CSI agrees): robot stopped after
  the fill, we remove 2 caps, tilt 1 and take out 1 vial; the software flags them live.

### 2.5 The live system — requirements and acceptance checks
| ID | Requirement | Acceptance check (expected output) |
| --- | --- | --- |
| R1 | **Frame input.** Reads new images from a folder as they appear (`--source folder:<dir>`) and replays saved frames (`--source replay:<dir>`), same code path. | Replay of DEV run 1 processes every frame in order; files copied into a watched folder one by one are each picked up within ~1 s. |
| R2 | **Geometry.** Position map for the camera actually used. 640×480: existing `position_map_G.json` with its recorded offset. 1920×1080: re-solved (step 5). | Overlay of all 30 boxes on a real frame from that camera: 30/30 land on their holes or caps; median error reported. |
| R3 | **Event gating.** Score only frames with the arm clear (reuse the `inspect_frame.py --events` logic). Staged stills are always clear. | On DEV runs 1–3 replay, no frame with the arm over the fixture produces a hole result. |
| R4 | **Inspector.** Frozen model gives each hole a class (`pass`, `missing_cap`, `crooked_cap`, `no_vial`) and a score; thresholds come from synthetic validation only (D033). `damaged_vial` out of scope. | Same input → same output; weights sha256 matches the freeze record. |
| R5 | **Fill state.** `fill_state.py` LIVE_FILL (D021): empty before the highest occupied hole = MISSED, after it = WAITING; unreadable = UNSURE. | On synthetic partial fills, every empty hole after the last filled one reads WAITING and every empty hole before it reads MISSED. |
| R6 | **Outputs.** Predictions CSV in the schema `eval_metrics.py` reads (check its docstring/selftest); trial log CSV (time, git sha, model sha256, input file, per-hole prediction + score, expected if known, latency ms, intervention note); operator view (frame with 30 coloured boxes + legend). | One replay run writes all three; `eval_metrics.py` reads the predictions CSV with no errors. |
| R7 | **Speed.** Faster than the frame interval (DEV frames are ~2 s apart). | Median latency per frame on Noor's laptop logged; target ≤1 s. |
| R8 | **Robustness.** Never crashes on: half-written file, unreadable image, wrong resolution, arm in view, empty folder, missing model file. Bad input → logged, hole UNSURE or frame skipped. | Each case tested on saved data before the freeze; each logged; process keeps running. |
| R9 | **Start/stop.** One command to start, Ctrl-C to stop, logs flushed on stop. Written in `LIVE_RUNBOOK.md` with the expected output of each step. | Sunday dry run: start → replay → Ctrl-C → complete logs. |
| R10 | **No plant changes.** Reads images only. No PLC, robot or plant-network connection. | Runbook states the camera-access route CSI approved. |
| R11 | **Freeze.** Code commit, weights, thresholds, config, position map and render recipe recorded with sha256 in `FREEZE_2026-10-25.md` + local git tag. No change after. A crash fix on the day = a new logged software version; model and thresholds unchanged. | `shasum -a 256 -c FREEZE.sha256` all OK on Oct 26 morning. |
| R12 | **Portability** (only if CSI says "run it on our PC"). Runs from one folder with its own environment (CPU PyTorch) on the OS CSI names. | Tested in a clean environment before Sunday. |

### 2.6 Still to settle with CSI
- **Asked, waiting:** camera resolution (640×480 or 1920×1080); session Mon Oct 26 1 pm (or next slot); how our laptop
  gets camera images without the plant network (shared folder or a CSI PC); backup slot Oct 28/29.
- **Raise once he confirms:** side photos with a phone; filming the line for the video (no people); the optional
  live-defect part; who saves the images, their file names, and how files reach us (USB or shared folder); ~32 vials
  and ~34 caps; the robot stop procedure is CSI's.

### 2.7 What we bring
Laptop (charged + charger) with the frozen software and `LIVE_RUNBOOK.md`; printed staging list + numbering map; log CSV
pre-filled from the staging list; phone; pen; USB stick; Tony.

## 3. Schedule to Mon Oct 26
| When | Work | Done when | Who |
| --- | --- | --- | --- |
| Thu Oct 22 pm | Step 0 sync docs; step 1 GPU gate + 6 check renders; stop the instance | Gate verdict recorded; `check.tar.gz` on the Mac; instance stopped | Noor + Claude Code |
| Thu Oct 22 | Step 2 message to Vang | Vang says what "m8" is and whether it's the live runtime | Noor |
| Fri Oct 23 am | Gate review; organizer release download; labeler check (step 3) | Go/no-go for training renders; notes on release + labeler | Noor + ChatGPT |
| Fri Oct 23 noon | Shamar follow-up in Slack if no reply | Session + camera answer | Noor |
| Fri Oct 23 | Batch renderer (step 6); live runtime skeleton (step 4) | 30 scenes in one Isaac session; DEV run 1 replay end to end | Claude Code + runtime owner |
| Fri Oct 23 eve | Camera decision (step 5) | Map and render camera fixed | Noor |
| Sat Oct 24 | Training renders; train ResNet-18; thresholds (step 7); integrate + robustness tests (step 8) | Weights + thresholds; R1–R10 pass on replay | Claude Code |
| Sun Oct 25 | G3 check; pick the live inspector; dry run; **freeze by 9 pm** (step 9); print; message Tony | Freeze record + tag; dry-run log | Noor |
| Mon Oct 26 1 pm | CSI visit (step 10) | 22 images + log; live trial log; recording | Noor + Tony |
| Mon Oct 26 eve | Adjudicate labels, then score every arm once (step 11) | Labels locked; metrics CSV; failure gallery | Noor + Tony |
| Tue–Thu Oct 27–29 | Error analysis; backup slot if needed; business-case numbers | Results tables | Team |
| Fri–Sun Oct 30–Nov 1 | Final write-up; video; one-page summary; **submit by Sun 11:59 pm CT** | Submitted | Team |

## 4. Step by step
**Step 0 — Sync docs (Mac, 5 min).** Save the three files Claude sent on Oct 22 into `~/code/crusaders/twin_to_line/`
(`HACKATHON_HANDOFF.md` and `DECISIONS.md` replace the old ones; `NEXT_SESSION_PLAN.md` is new). Then:
```bash
cd ~/code/crusaders && git status -sb   # expect: ## feat/scene-generator, " M twin_to_line/dose_sets.py", the 2 changed docs and 1 new
git add twin_to_line/HACKATHON_HANDOFF.md twin_to_line/DECISIONS.md twin_to_line/NEXT_SESSION_PLAN.md
git commit -m "docs: Oct 22 handoff, D036, next-session plan"   # local only; dose_sets.py stays unstaged
```

**Step 1 — GPU gate (Thu).** Handoff NEXT ACTION 1–9, exactly. Ask Noor before creating the instance (state $/hr).
While it's up, note the per-render seconds: it sizes Saturday (430 scenes × s/render ÷ 3600 × $/hr). Stop the instance
as soon as files are copied. Then stop for review.

**Step 2 — Vang (Thu).** Send the draft in §8. If his pipeline is the live runtime, step 4 becomes an adapter: his
output → our predictions CSV + trial log, holes numbered G01–G30, same position map. If not, ask the team for an owner
(draft in §8); if none by Fri evening, Noor + Claude Code build the minimum runtime.

**Step 3 — Organizer release + labeler (Fri, ~45 min).**
- Download the Dataset 1 organizer release from `#dataset-1-vial-inspection` into `~/Downloads/` (never the repo).
  Note: resolution, number of images, label counts per class, whether it's the same view as the 1920×1080 labeler
  examples.
- Unzip `~/Downloads/vial-labeler-participants-20261014.zip` to a scratch folder outside the repo; open it per its README;
  load one release image. *Expected:* G01–G30 grid, no class predictions, review-JSON export (D027). Keep one exported
  JSON as a format sample (not committed).

**Step 4 — Live runtime skeleton (Fri).** New local branch `feat/live-runtime` off `feat/scene-generator`. New
`twin_to_line/live_inspect.py` reusing `positions.py`, `fill_state.py` and the `inspect_frame.py` event logic; replay
mode first; rule baseline as the first inspector so the pipeline runs end to end before any model exists. Write
`LIVE_RUNBOOK.md`. *Done when:* R1, R3, R5, R6 pass on a DEV run 1 replay. Tests for each part; commit locally.

**Step 5 — Camera decision (Fri, when Shamar answers).**
- **640×480** (the cap_seal view of runs 1–5): keep `CapSealCam` and `position_map_G.json`; render as planned.
- **1920×1080:** check whether the organizer release is that view; re-solve the camera for it from release frames
  (detect cap/hole centres, solve pose against the twin's 3D positions; never fit to the organizer's grid coordinates,
  D024); validate 30/30 by overlay; add the camera to the renderer so training renders match the live view; resize crops
  to the training crop size.
- **No answer by Fri evening:** render for 640×480 (the view we can validate), keep the 1080 re-solve as Saturday
  contingency, and get the answer in person or by phone before Sunday.

**Step 6 — Batch renderer + training renders (Fri–Sat).** One Isaac session renders many `scene.usda` files. Check it
against `render_keep.py` on 3 scenes (same framing; differences only from sampling noise). After the gate review says
go: generate and render `smoke` (300) + `tilt_sweep` (130) on Brev; cut crops; copy back; stop the instance.

**Step 7 — Train + thresholds (Sat).** Split crops by scene (no scene in both train and validation); rebalance (58% of
smoke labels are empty holes); classes `pass`, `missing_cap`, `crooked_cap`, `no_vial`; ResNet-18 on the Mac (MPS), the
Spark or Brev. Thresholds from synthetic validation only, same procedure for the rule baseline. Record the recipe:
render `run.json`, seeds, split, epochs, crop size.

**Step 8 — Integrate + robustness (Sat–Sun).** Plug the model into the runtime; R4, R7, R8 checks; G3 on DEV runs 1–3
(go if ≥90% of DEV defect observations caught at ≤5% false alarms; development signal only; never tune on runs 4–5).

**Step 9 — Freeze (Sun by 9 pm).** Choose the live inspector before freezing: ResNet-18 if G3 is go, otherwise the rule
baseline (record why). Write `FREEZE_2026-10-25.md` (§5) and tag locally. Dry run: start → replay → Ctrl-C → logs. Print
the staging list; pre-fill the log CSV; confirm time and place with Tony.

**Step 10 — Session day.** Arrive early. Orientation check (setup 00) first. Follow the staging doc's per-setup
procedure. Live run per `LIVE_RUNBOOK.md`; log every attempt. Copy every file (images, log, side photos, trial log,
recording) before leaving. Note start and end times.

**Step 11 — Score (Mon evening).**
1. Adjudicate labels from the log, the images and side photos — two people, **before looking at any model output**.
   Lock the labels file (sha256).
2. Run every frozen arm once over the 22 images → predictions CSVs.
3. `eval_metrics.py`: per class misses; false-alarm rate at the threshold that catches every known defect (reported,
   never used to set anything); complete fixture scenes; sub-20° caps reported separately as the camera limit; WAITING
   holes falsely flagged.
4. Failure gallery: every miss and false alarm with its crop.
5. Live run summary from the trial log: latency, wrong calls, interventions.

## 5. Arms, thresholds and the freeze record
| Arm | What it is | Thresholds | Scored on |
| --- | --- | --- | --- |
| A — synthetic-only ResNet-18 (proposed primary) | Trained on twin renders only | Synthetic validation | Oct 26 images, runs 4–5, organizer release |
| B — rule baseline | Colour/geometry rules at twin positions | Synthetic validation | Same |
| C — Vang's pipeline / counterfactual (if it writes our CSV by the freeze) | Team approach | Its own, frozen | Eligible events + coverage (D034) |
Not run: real-only and synthetic + real (no permission, no Session A; D025, D036). Every arm is frozen Sunday and scored
once.

**Freeze record contents:** date/time; git commit + tag; weights sha256; thresholds sha256; config; position map file +
camera; render recipe (`gen_scenes.py` sha256, configs, seeds, `run.json`, GPU, Isaac version); training seed, split,
epochs; Python/PyTorch versions; chosen live inspector + reason; G3 result; gate verdict.

**Business-case numbers to collect** (for Zara): real defect images used for training (0); holes taught by hand (0);
staging minutes per setup and total session time; hours spent commissioning from the twin (map, renders, training); and
from CSI/organizers: today's inspection setup time for a new product and the cost of an escaped defect vs a false reject
(our Oct 14 plan asked; never answered).

## 6. The vial labeling tool (Slack: "auto labeling tool")
- It's the same file we already have: `~/Downloads/vial-labeler-participants-20261014.zip`. Our Oct 16 check (D027):
  despite the name, it does **not** predict labels. It's a browser review tool for the **1920×1080** tray view with the
  same G01–G30 numbering as ours (verified: +3.6° rotation, no mirror); it rejects 640×480 images; it exports review JSON.
- **Use 1 — adjudicating the Oct 26 labels** if the camera is 1920×1080: load each image, set each hole from our log, fix
  anything that slipped, export JSON. Our log stays the source; the tool is the second check. Never one source alone.
- **Use 2 — organizer release labels** for development and evaluation (never training): good job for Tony.
- **Use 3 (optional)** — export our predictions in its JSON so reviewers can browse our results in the organizers' tool.
- **640×480 camera:** the tool can't load the images; adjudicate with our own overlay images + the log instead.
- Never commit the zip, images or exported JSON with image data.

## 7. Risks and fallbacks
| Risk | Trigger | Fallback |
| --- | --- | --- |
| CSI can't do Oct 26 | Shamar's reply | Backup Oct 28/29; freeze moves to the night before; same plan |
| Camera is 1920×1080 | Shamar's reply | Step 5 re-solve Fri–Sat; render with the new camera |
| No laptop access to the camera | Shamar's reply | Run on a CSI PC (R12) or a shared folder; if neither, stage only and ask organizers whether a same-day replay counts as the demo |
| Gate FAIL / INCONCLUSIVE | Step 1 review | Proposal-doc fallbacks F1–F4; rule baseline becomes the live inspector |
| No usable renders by Sat noon | Step 6 | Rule baseline needs thresholds without renders → send the narrowed organizer question (§8) |
| G3 no-go | Sunday | Rule baseline runs live; ResNet-18 still scored offline as arm A |
| No runtime owner | Fri evening | Noor + Claude Code build it (most parts exist offline) |
| Two pipelines in the team | Vang's reply | One runtime; the other becomes an arm writing the same CSV (D034) |
| Caps won't hold a set tilt | On the day | Stage the natural tilt; measure from the side photo; log measured angle |
| Brev credits | Any instance | Stop when idle; record GPU, hours, cost per run |

## 8. Messages (drafts — Noor sends)
**Vang (Discord, now):**
> @Vang what's the m8 inspection pipeline? Is it the live part (camera image → result for each hole)? If yes, can it
> give one row per hole (G01–G30) like our eval CSV and be ready to run at CSI Mon 1 pm? Just making sure we don't
> build the same thing twice.

**Team (Discord, only if Vang's pipeline isn't the runtime):**
> Who can own the live software for Monday? It reads the camera images and shows pass/defect for each hole. It has to
> be done and frozen by Sunday night. If nobody can, I'll build it.

**Shamar (Slack `#data-and-testbed-requests`, Fri noon, only if no reply):**
> @Shamar Webster following up on my email from Thursday (Code Crusaders, Dataset 1). Can we book about 90 min on
> Mon Oct 26 at 1 pm (or your next slot)? With the robot stopped we place vials by hand in the Good fixture, CSI saves one
> image per setup, then a short supervised run with our software reading the camera. Also: what resolution does that
> camera record at? Thanks!

**Shamar (after he confirms):**
> Thanks! A few details for Monday: (1) OK if we take phone photos from the side of tilted caps? (2) OK to film the line
> for our video (no people)? (3) At the end, with the robot stopped, can we take off a few caps and one vial to show our
> software catching them live? (4) How will we get the image files, USB or a shared folder? (5) We'll need about 32 vials
> and 34 caps.

**Organizers (`#dataset-1-vial-inspection`, only if renders fail by Sat noon):**
> Quick rules question (Code Crusaders, Dataset 1): our baseline is simple colour rules at each hole, not a trained
> model. Is it OK to set its thresholds using the supplied runs 1–3 images, or should those also come only from twin
> images?

## 9. Open questions for Noor
1. Freeze at Sun Oct 25, 9 pm — OK?
2. If no runtime owner by Fri evening, OK for you + Claude Code to build it Fri–Sat?
3. Send the narrowed organizer question now as insurance, or only if renders fail?
4. Tony confirmed for Mon 1 pm?
