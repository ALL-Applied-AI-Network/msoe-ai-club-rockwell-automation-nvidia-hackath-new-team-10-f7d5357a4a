# Twin-to-Line: Commissioning Final Vial Inspection from the Digital Twin

**Code Crusaders** · Rockwell Automation + NVIDIA Physical AI Innovation Lab, Fall 2026 · Challenge 1, Dataset 1 (Final Vial Inspection)

Twin-to-Line sets up the overhead inspection at CSI's final vial station from the digital twin instead of from
hand-teaching and real defect images. On a hand-staged test at CSI it located all 30 fixture holes with no
hand-teaching and caught 14 of 14 missed placements. Its twin-trained cap checks did not handle crooked caps; we
traced why (see [Results](#results)).

## Problem

At CSI's final station the robot places vials into a 30-hole Good fixture (G). The side inspection misses some cap
defects, caps can move after inspection, and the robot can miss a placement. On a high-mix, low-volume line the
overhead inspection has to be re-commissioned at every changeover: hole positions taught by hand, and real defect
images that only exist once someone makes defects on purpose. Our question: how much of that commissioning can the
digital twin do? A missed defect counts more than a false alarm.

## Approach

| Part | What it does | Code |
|---|---|---|
| Position map from the twin | Projects G01–G30 through the twin's solved cap_seal camera; no per-hole teaching. 640×480 practice runs: 30/30 holes, ~1.68 px median error. CSI's 1920×1080 stream is warped into the same grid (D039). | `twin_to_line/positions.py`, `capsealcam.py`, `frame_warp.py`, `position_map_G.json` |
| Synthetic data | Isaac Sim renders from USD override layers over the untouched twin: pass, missing cap, crooked cap, no vial, leaning vials (568 scenes, 17,040 crops). Labels are read back from the composed stage. | `gen_scenes.py`, `render_batch.py`, `cut_train_crops.py` |
| Inspectors (frozen Oct 25) | **A:** ResNet-18 trained only on renders (zero real images). **B / B′ / B″:** a three-threshold colour rule as the baseline, with thresholds from renders (B), practice runs 1–3 (B′, ran live) or good practice images only (B″). | `train_resnet.py`, `resnet_inspector.py`, `rules.py`, `rule_thresholds*.json` |
| Fill-order logic | An empty hole before the last filled one is MISSED, after it WAITING. No PLC or robot signal needed. | `fill_state.py` |
| Live runtime | Reads camera images from a folder, waits for the arm to clear, scores all 30 holes, writes predictions and a trial log. | `live_inspect.py`, `events.py`, `LIVE_RUNBOOK.md` |
| AI-run CSI session | An AI agent in a terminal on CSI's GB10 ran setup, staging, sealing, the live run and export from a written guide; the team only placed vials. | `csi_day2/`, `monday/` |
| Operator UI | Read-only web view of the checker's output. | `ignite_core/operator_ui/` |

Every decision, with its reason and date, is in [`twin_to_line/DECISIONS.md`](twin_to_line/DECISIONS.md) (D001–D045).

## Results

**Test set.** On Oct 27, with the robot stopped, we hand-placed 22 fixture setups at CSI (17 crooked caps, 14 missing
caps, 14 missed placements, 65 waiting holes, 550 good positions). The AI agent captured each from the live cap_seal
camera on CSI's GB10 and sealed the set. Code, models and thresholds were frozen on Oct 25 (tag `freeze-2026-10-25`).
Labels were drafted from the staging log, checked against every image, confirmed by a teammate and locked by checksum
before any model ran (D043). The set was scored once.

**Pre-registered result (D043).** The frozen arm check mistook a yellow strip beside the fixture for the robot arm and
skipped 16 of 22 frames. On the 6 it kept, the live rule (B′) caught 7 of 7 missed placements and 1 of 1 missing cap,
and missed 3 of 4 crooked caps.

**After fixing only the arm check (D044/D045, v1.1 built without the test images; post-hoc), all 22 frames:**

| Inspector | Crooked caps caught (/17) | Missing caps caught (/14) | Missed placements caught (/14) | Good holes flagged (/550) | Waiting holes flagged (/65) |
|---|---|---|---|---|---|
| B′ rule, practice-fit (ran live) | 5 | 12 | 14 | 109 | 13 |
| B″ rule, good images only | 3 | 12 | 14 | 41 | 13 |
| A ResNet-18, twin only (D038 threshold) | 17 | 14 | 14 | 412 | 54 |
| A ResNet-18, pre-registered threshold | 8 | 8 | 9 | 57 | 37 |
| B rule, twin-set thresholds | 17 | 11 | 6 | 538 | 18 |

Rule flags go to a person as UNSURE; ResNet flags are rejects. False-alarm rate at the threshold that catches every
cap defect: 63.8% (A), 81.5% (rules).

**What worked:** the twin's geometry carried over to a new camera with no teaching, and fill-order logic told missed
placements from waiting holes 14/14. **What failed:** crooked caps. The twin-trained ResNet catches all 17 only by
flagging 75% of good holes.

**Why (diagnosis after scoring, not a result).**
- A tilted cap pivots about the vial mouth, so its yellow centre moves a median 0.9 px from a good reference and the
  rule's position feature cannot see it. Its orange side does show: a hue check against one all-good image of the
  fixture separated all 17 crooked caps from most good caps (threshold set on the test images, so optimistic).
- On the 1080 view each hole has a fixed offset (0.8–4.9 px), so one global threshold flags the same good holes.
- The 1080 stream's warm tint makes empty holes and open vial rims read partly yellow; the ResNet learned the twin's
  lighting.

Takeaway: the twin carries geometry and fill order; appearance needs one reference image of a good fixture per camera,
not a defect set.

**On hardware.** On Oct 27 the frozen live inspector ran for about an hour on CSI's GB10 off the real camera. The line
was not filling, so every hole read waiting: no live defect was caught. A second session on Oct 29 was lost to a line
fault.

**Not part of these results:** the `staging5` ResNet (PR #7) was trained and thresholded on the sealed Oct 27 images,
so it cannot be scored on them; it is reported separately if at all.

## How to run

Requirements: Python 3.11+, `pip install -r twin_to_line/requirements.txt` (NumPy, OpenCV, pytest; usd-core only for
twin authoring), plus PyTorch/torchvision for the ResNet arm. Data is not in git: get the Dataset 1 images and the
digital twin from the Innovation Lab data folder (see the event guide) and keep them outside the repo.

```bash
cd twin_to_line
python -m pytest tests -q                          # unit tests (a few need git-ignored data and skip or fail without it)

# Live/replay inspection: folder of cap_seal images in, predictions + trial log out
python live_inspect.py --source folder:/path/to/incoming_images --out results/live/run_$(date +%Y%m%d_%H%M%S) --inspector rule
python live_inspect.py --source replay:/path/to/saved_frames   --out results/live/replay_test --inspector rule

# Operator UI on the checker's output folder (from the repo root)
python -m ignite_core.operator_ui --output-dir twin_to_line/results/live/<run> --port 8765

# Score a labelled image folder (once-only lock ledger; see D043)
python score_sealed.py <image_folder> --labels <labels.csv> --labels-sha256 <sha256> --out <out_dir>
```

Full start/stop procedure with expected output for each step: [`twin_to_line/LIVE_RUNBOOK.md`](twin_to_line/LIVE_RUNBOOK.md).
Rendering on Brev and the sim-to-real gate: [`twin_to_line/README.md`](twin_to_line/README.md) and `GATE_v2.0.2.md`.

## Repository map

- `twin_to_line/`: position map, scene generation and rendering, inspectors, fill-order logic, live runtime, scoring,
  decisions, freeze record (`FREEZE_2026-10-25.md`) and CSI session guides (`csi_day2/`).
- `ignite_core/`: operator UI, fill-state and position modules, CSI staging plan and report tools.
- Not in git (by design): raw CSI images, twin renders, model weights, results.

## Team

Code Crusaders: Prabhnoor (Noor) Bal, Vang, Zara, Matthew, Trevor, Anthony Harvey, Jamie Gill.

## Credits and references

- **Data and digital twin:** Dataset 1 (Final Vial Inspection) and the CSI digital twin, provided by UWM's Connected
  Systems Institute, NVIDIA, Rockwell Automation and the MSOE AI Club. Staged test images collected at CSI on Oct 27,
  2026, with CSI staff supervision (Shamar Webster).
- **Software:** NVIDIA Isaac Sim 6.0.1 (on NVIDIA Brev), OpenUSD (`usd-core`), PyTorch and torchvision (ResNet-18 with
  ImageNet-1K weights), OpenCV, NumPy, pytest.
- **Compute:** NVIDIA Brev (L40S), CSI's YAAD DGX Spark lab and the CSI GB10 at the line.
- **AI coding tools:** Claude (Claude Code), OpenAI Codex and ChatGPT were used for code, reviews, documents and to run
  the CSI session.
- **Event guide:** <https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026>
