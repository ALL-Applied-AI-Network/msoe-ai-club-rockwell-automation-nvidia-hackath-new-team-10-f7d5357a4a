# Live runbook: `live_inspect.py`
Draft for the Sunday dry run (plan §2.5 R9). The software is **not frozen yet**. §0–§5 run on Noor's Mac in
`~/code/crusaders/twin_to_line`; **Monday runs on CSI's GB10 from the package: §7.** The camera streams 1920×1080 (and
320×240); 1920×1080 frames are warped into the DEV 640×480 grid and then follow the same path (D039). How image files
reach our process is **not confirmed yet** (§6).

**What it does:** it reads image files from a folder. Each time the fixture is still for 4 s with the arm away, it
gives all 30 holes (G01–G30) a state and logs every image it saw. It connects to no PLC, robot or plant network, and it
never writes into the image folder (R10).

| Colour on screen | State | Meaning |
|---|---|---|
| green | PASS | vial with a seated cap |
| red | DEFECT | cap problem seen |
| magenta | MISSED | empty hole before the last filled hole |
| grey | WAITING | empty hole after the last filled hole (not filled yet) |
| orange | UNSURE | can't tell (displaced or partial cap, or a hole read empty that was filled before); goes to a person |

**Inspector** (`--inspector`, chosen at the freeze, step 9):
- `rule` (default): the rule baseline. Its `rule_thresholds.json` was fitted on DEV draft labels (a placeholder).
- `resnet`: the synthetic-only ResNet-18 v1 at the D038 threshold (`--threshold preregistered` gives the
  pre-registered one). **G3 on DEV runs 1–3 was NO-GO** (87.7 % false alarms), so per the plan the rule baseline runs
  live and the ResNet is scored offline as arm A (§5). Needs `results/train/resnet18_v1/` (gitignored, never
  committed) and torch in `.venv`.

## 0. Before the session (Sunday dry run, then again Monday morning)
1. **Version.** `git -C ~/code/crusaders log -1 --oneline` should show the frozen commit/tag. After the freeze:
   `shasum -a 256 -c FREEZE.sha256` → every line `OK` (R11; that file is written at the freeze).
2. **Tests.**
   `IGNITE_TWIN=~/Downloads/csi_digital_twin/csi_digital_twin_v1 ../.venv/bin/python -m pytest tests -q`
   → `166 passed` (or the count in the freeze record).
3. **Replay folder** (once; 569 MB, stays in `results/`, never committed):
   `../.venv/bin/python live_inspect.py --export-replay 1 --out results/live/replay_run1`
   → `run 1: 1123 frames (video) -> results/live/replay_run1`
4. **Dry run on DEV run 1:**
   ```bash
   ../.venv/bin/python live_inspect.py --source replay:results/live/replay_run1 \
       --labels labels/dev_labels.csv --out results/live/dryrun_$(date +%Y%m%d_%H%M) --note "dry run"
   ```
   Expect `inspector rule_yellow_seat  rule_thresholds.json 322c29ec2ea9`, then 18 lines from
   `event  1       cap_seal_video_1.mp4@t0000.0 K= 0 delay   4.0s  WAITING=30` to
   `event 18       cap_seal_video_1.mp4@t1050.0 K=17 delay   4.0s  MISSED=1 PASS=10 UNSURE=7 WAITING=12`, then
   `1123 frames, 18 events, {'settling': 75, 'scored': 18, 'unchanged': 964, 'arm': 66} -> results/live/dryrun_…`
5. **Outputs readable:** `../.venv/bin/python eval_metrics.py <dryrun dir>/predictions.csv --labels labels/dev_labels.csv`
   → first line `joined 523 labeled rows; 17 prediction rows have no label; 60 labels have no prediction`, no error.
   This is a plumbing check only (draft, in-sample labels), not a result.
6. **Folder rehearsal:**
   - Start step 1 below on an empty test folder, then copy a few `results/live/replay_run1` frames into it, one every
     second or two.
   - Each copied file should get a trial-log line within ~1 s. Ctrl-C should stop it as in step 3.
7. **Keep the laptop awake.** In a second terminal: `caffeinate -dims` (stop it with Ctrl-C after the session).

## 1. Start (live run)
1. **Set the folders.** `IN` = the folder where the camera images arrive (route to be approved by CSI). `OUT` = a
   **new** folder for every start; reusing one overwrites its CSV files.
   ```bash
   IN=/path/to/incoming_images
   OUT=results/live/csi_$(date +%Y%m%d_%H%M%S)
   ../.venv/bin/python live_inspect.py --source folder:"$IN" --out "$OUT" --note "CSI live run, Oct 26"
   ```
2. **Expected:**
   - One line names the inspector and its file hash, e.g. `inspector rule_yellow_seat  rule_thresholds.json
     322c29ec2ea9`; with `--inspector resnet`: `inspector resnet18_synth_v1[d038] t=0.50224 (pre-registered 0.99968)
     on mps  results/train/resnet18_v1/model.pt 8380851f8b4e`.
   - `watching <IN>: N image(s) already there are ignored; new images only`: images already in `IN` at the start
     (unchanged files only) are not read; `session.json` records N.
   - `operator view: this console (one grid per event), <OUT>/overlay_latest.png, or <OUT>/view.html in a browser`.
     Then nothing until the first event.
   - Within ~2 s, `$OUT` contains `overlays/`, `predictions.csv`, `session.json`, `runtime_status.json`,
     `trial_log.csv` and `view.html`. `runtime_status.json` is replaced atomically after every input. It starts
     `INSPECTION_UNAVAILABLE`, changes to reason `input_processing` before decode/inference, and remains unavailable
     after an unscored input, any input/source fault, during `session_closing`, or after session end. The closing
     state is published before either CSV is closed, so a slow or blocked filesystem close cannot leave an old
     available state on disk. Only a newly completed `scored` event publishes
     `INSPECTION_AVAILABLE`. It is shadow-only status and never grants PASS, release, or actuation authority.
     Its `availability_scope` is `processing_runtime_only` and `source_liveness_guaranteed` is false: folder silence
     can be normal, so the runtime does not invent a universal idle timeout. Any consumer must configure a maximum
     age or heartbeat appropriate to its camera route and treat an older-than-configured `status_updated_utc` as
     unavailable.
   - In `session.json`, `"git"` is the frozen commit **without** `+modified`.
3. **Watch the log** in a second terminal: `tail -f "$OUT/trial_log.csv" | cut -d, -f6,8,77`. Each new image adds one
   `file,status,path` line within ~1 s.
4. **Check the first few images:**
   - No status `wrong_size`. Column 77 (`path`) is `1080->640` for 1920×1080 frames (warped into the DEV grid with
     `frame_warp_1080.json`, D039) or `640` for 640×480 frames. Any other size (e.g. the 320×240 stream) is
     `wrong_size`: see §4.
   - Column 7 (`frame_t`) prefers time parsed from the filename when it is later than the previous frame's.
     Otherwise it advances by elapsed arrival time since the previous frame, keeping the same timebase
     (`cut -d, -f6,7 "$OUT/trial_log.csv"`). Repeated names still advance; timestamped bursts retain capture intervals.
     With no initial timestamp it starts from monotonic arrival time. Replay keeps filename timestamps.
   - Images already in `IN` when you start are ignored (the start line says how many); only images that arrive
     after the start are read, in arrival order.

## 2. While it runs
- **Normal statuses:**
  - `arm`: arm over the fixture, nothing is scored.
  - `settling`: waiting for 4 s of stillness.
  - `unchanged`: same fixture state as the last event.
  - `scored`: an event.
- **Event lines.** One per event: `event N <frame> K=<last filled hole> delay <s>  <state counts>`. The delay is
  normally 4.0 s (the stillness rule).
- **Operator view (primary): the console grid** printed under every event line, the 30 holes as the camera sees
  them (G26 … G01 across the top row, as in the staging doc), then the counts. DEV run 1, event 18:
  ```
      G26 .     G21 .     G16 ok    G11 ??    G06 ok    G01 ??
      G27 .     G22 .     G17 ok    G12 ??    G07 MISS  G02 ok
      G28 .     G23 .     G18 .     G13 ??    G08 ok    G03 ok
      G29 .     G24 .     G19 .     G14 ??    G09 ok    G04 ok
      G30 .     G25 .     G20 ??    G15 ??    G10 ok    G05 ok
      PASS 10  DEFECT 0  MISSED 1  UNSURE 7  WAITING 12    (ok = PASS  DEF = DEFECT  MISS = MISSED  ?? = UNSURE  . = WAITING)
  ```
  In a terminal the codes are coloured (green ok, red DEF, magenta MISS, yellow ??, dim .).
- **Operator view (extra, if there is a browser):** open `$OUT/view.html` (`xdg-open "$OUT/view.html"`, or drag it into
  the browser). It reloads `overlay_latest.png` every second over `file://`; the image is replaced atomically (temp
  file + rename), so a half-written overlay is never shown. The page explicitly labels it as the **last successful
  event**, not the current input or a product-release decision. Check `runtime_status.json` for current health.
  Earlier events stay in `$OUT/overlays/eNN_<frame>.png`.
- **Interventions.** Anything a person does goes on the paper log with the clock time: robot stopped, part moved,
  restart, cable, re-aim. After the session, copy those notes into the `intervention` column of the matching
  `trial_log.csv` rows. Never edit any other column.

## 3. Stop
1. Press **Ctrl-C once** in the terminal running `live_inspect.py`.
2. **Expected:** one line `N frames, M events, {...} -> <OUT>`. The prompt returns with no traceback.
   `session.json` shows `"stop_reason": "ctrl-c"`. Every row was already written as it happened, so nothing is lost.
   It also reports separately named all-record and scored-event p50/p95/p99 latency; use the scored-event values for
   inference performance because cheap gating/source records can make the all-record distribution look faster.
3. **Copy all of `$OUT`** to the USB stick, and note the start and stop times on the paper log.

## 4. Errors: what you see and what to do
| What you see | What it means | What to do |
|---|---|---|
| `live_inspect.py: error: --source must be replay:DIR or folder:DIR with an existing DIR` | Typo, or the folder isn't there / not mounted | Fix the path; start again |
| `ProtectedDataError: N protected frame(s) in DIR` (replay only; nothing was processed) | The folder holds dataset frames from runs 4–5 or unassigned | Stop; don't open or move them; use the right folder; tell Noor |
| `can't open file ... live_inspect.py` | Not in the project folder | `cd ~/code/crusaders/twin_to_line`; start again |
| A traceback at start naming `rule_thresholds.json` or `position_map_G.json` | A file is missing from this checkout | Check the version and hashes (§0.1); don't run until they match the freeze |
| `FileNotFoundError: ... results/train/resnet18_v1/run.json` (`--inspector resnet`; no `OUT` created) | The model folder isn't on this Mac (gitignored) | Copy it back from the backup and check its hashes against the freeze record; or run `--inspector rule` |
| `ValueError: ... model.pt does not match the sha256 recorded in run.json` | The model file was changed or damaged | Don't run the ResNet; restore the folder from the backup |
| `ModuleNotFoundError: No module named 'torch'` | `.venv` lacks torch | `../.venv/bin/pip install torch torchvision` before the session (not on the day); or run `--inspector rule` |
| Status `unreadable` | A version remains unacceptable after 3 s: permission denied, corrupt image, empty/incomplete file, missing JPEG/PNG end marker, or extra bytes after the end marker (JPEG trailing 0x00 bytes are accepted; any nonzero JPEG trailer or any PNG trailer is rejected) | Logged once per version and never scored; a later complete version is accepted. One-off: note it. Repeated: Ctrl-C, check permissions and the copy route with CSI |
| Status `wrong_size`, note e.g. `320x240, expected 640x480 or 1920x1080` | The images are neither 1920×1080 nor 640×480 (e.g. the 320×240 stream) | Ctrl-C. Ask CSI for the 1920×1080 images. Never resize or crop by hand |
| `FileNotFoundError: ... frame_warp_1080.json` at startup | The required warp file is missing; startup stops while hashing runtime files | Restore the frozen package and check its hashes (§0.1 or §7) before restarting |
| `WARNING folder ...`, status `source_error` | The watched directory cannot be listed (e.g. disconnected mount or permissions) | The runtime retries every `--poll` seconds and logs `source_recovered` on recovery; restore access. If `--idle-exit` ends during the outage, `session.json` records the source error |
| Status `refused` | A dataset holdout frame (runs 4–5) arrived in the folder | Ctrl-C; take that folder out of use; tell Noor |
| Status `error` (the note has the message) | A recoverable pre-stateful frame operation failed, or a gate/score/evidence transaction failed | Any note starting `restart_required_after_` is fatal: the session stops and refuses more inputs because gate/tracker state or audit evidence may have changed. Preserve the entire `OUT`; restart with a new one and never reuse its gate/tracker state. Predictions plus an archival overlay may be retained if only publishing `overlay_latest.png` failed; the prior latest overlay remains and health is unavailable. For another one-off error, note it; repeated errors require a restart |
| `runtime_status.json` says `INSPECTION_UNAVAILABLE` | No event has succeeded yet, the latest input/source fault is unresolved, shutdown is in `session_closing`, or the session ended | Do not treat the last overlay as current. Restore the input route and wait for a new `scored` event; recovery alone does not clear this state. Never restart into the same `OUT` after a shutdown/evidence failure |
| `SessionRestartRequired`, `runtime_status.json` disappears, or shutdown reports an evidence-close failure | Gate, score, prediction, overlay, trial/health status, or close evidence could not be completed safely | Fatal even if no final trial row was writable. Preserve the entire `OUT`, never reuse its gate/tracker state, and restart with a new `OUT`. A failed CSV append is rolled back when possible; rollback failure is reported explicitly. If predictions and the archival overlay committed before latest-overlay publication failed, they remain as partial evidence, event numbering stays reserved, the prior `overlay_latest.png` remains, and health is unavailable |
| A new image is in `IN` but no trial-log line appears | It isn't `.png` / `.jpg` / `.jpeg`, or its name starts with `.` (ignored) | `ls -la "$IN"`; ask CSI for the file type |
| Trial-log lines keep coming but no event for minutes | The arm stays over the fixture, the scene never holds still for 4 s, or `frame_t` barely advances (e.g. a burst with missing, invalid or repeated filename timestamps) | Check `arm` / `settling` and column 7. If `frame_t` is flat, stop and tell Noor (§6); for independent stills saved on request, restart with `--gate each` (§7.5) |
| Boxes on the overlay don't sit on the holes; many holes UNSURE or WAITING that look filled | The camera view differs from the position map | Ctrl-C; take one photo of the screen; tell Noor. Never re-aim or move the camera (CSI rule) |
| The program crashed, or the terminal closed | — | Start again with a **new** `OUT` (§1). Images already in `IN` are ignored by the new session, so images that arrived while it was down are not read. Log the restart and the gap as an intervention |

## 5. Staged test images: don't run the software on them on the day
- The 22 staged images are the sealed test (D036). On the day they are only **copied off to the USB stick with their
  sha256** (§7 step 4). Monday evening their labels are adjudicated and locked (the labels CSV and its sha256 written
  down) before any output is looked at. Then each arm is scored once.
- They stay out of every run on the day: CSI saves them in the separate `STAGED` folder. After verified USB
  backup and a fixture reset, live uses a fresh folder/feed (§7); existing-file skipping alone is not a seal.
- Scoring later (after labels are locked, on the Mac): run `score_sealed.py` once for all five declared operating
  points (A D038, A pre-registered, B synthetic, B′ DEV-fit, B″ good-only DEV). This wrapper injects each inspector
  into the unchanged `live_inspect.Session`; never swap threshold files to select an arm.
  ```bash
  read -r -p 'Staged images directory: ' SEALED
  read -r -p 'Adjudicated labels CSV: ' LABELS
  read -r -p 'Previously recorded full labels SHA256: ' LABELS_SHA
  ../.venv/bin/python score_sealed.py "$SEALED" --labels "$LABELS" --labels-sha256 "$LABELS_SHA" \
      --out "results/sealed/evaluation_$(date +%Y%m%dT%H%M%S)" --device cpu
  ```
  Labels must contain exactly one row for every input frame × G01–G30, with `frame`, `position`, `ground_truth`,
  `gt_sequence`; use runtime frame IDs (ordinary filename including extension; cached video IDs omit the final `.png`).
  Keep exclusions explicit via `usable=0`; arm-occluded images are logged and not scored. The wrapper snapshots
  images and locked labels, records each arm's provenance/coverage, and refuses existing outputs or image bytes
  already attempted in `results/sealed_score_locks`. A failed attempt remains locked; do not delete the ledger
  or rerun in another checkout. The wrapper produces predictions, not fitted thresholds or a selected winner.
- `--gate each` scores every arm-clear image on its own: each staged setup is an independent fixture, and its fill
  frontier comes from that image alone (tested: two different fixtures in a row give the same result as each alone).
  **Never add `--track`** here: the tracker carries MISSED latches from one image to the next. An image with the arm
  over the fixture is logged `arm` and not scored.

## 6. Open before the freeze
- CSI: how images reach our process on the GB10 (folder path or stream URL, and capture cadence); a screen for the
  overlay. Folder mode prefers advancing filename times, otherwise elapsed arrival time; replay uses filename
  times, falling back to modification times.
- 1920×1080 is handled by the D039 warp; the 320×240 stream is not supported (`wrong_size`).
- Operator view: the console grid (primary) and `view.html` (needs a browser on the GB10; checked in Chrome over
  http://127.0.0.1 because the test browser could not open `file://`).
- How frames arrive (new files, one rewritten file, a live stream, stills on request) decides the §7 step 5 command.
  Folder mode re-reads a file written again under the same name; stream mode is untested on a live camera.
- The frozen inspector and thresholds (R4); the full robustness tests (R8); the freeze record and hashes (R11).
- Measured on this Mac (DEV run 1 replay): rule median 27 ms per scored image, ResNet (MPS) 74 ms; 1.3 ms per other
  image (R7 target ≤ 1 s); a 1920×1080 frame adds ~3.4 ms for the warp. ResNet start-up (torch, model load, warm-up)
  takes about 2.5 s before its first line.

## 7. Monday Oct 26 — CSI's GB10, 12:30–2:00

Bring an **exFAT USB labelled TTL**: one Sunday-attested `ttl_gb10_*.tar`, its `.sha256`, five scripts,
`monday_ops.py`, `monday.sha256`; allow several GB free. Bring printed staging/numbering sheets and paper logs.
Confirm CSI's supervisor and capture route. Never change cameras, controls or network settings.
Live rule B′ is DEV-fit; organizer permission remains required. Labels and paper logs stay on paper and go to the Mac.

1. **12:30–12:45 — Setup:** `bash /media/$USER/TTL/start.sh`
   No prompts: USB and the single package are found automatically. If the mount differs, use
   `lsblk -o NAME,SIZE,MOUNTPOINT`, then type `bash <mount>/start.sh`.
   Expect checksums **OK → PREFLIGHT OK → BOOTSTRAP OK → SETUP OK**. Check the printed commit against
   Sunday's attestation. Repeating start with the same verified package resumes the visit without bootstrap:
   **SETUP OK: resumed … (N staged, sealed yes/no)**. Missing/failed setup starts a new visit. A different
   package gives **STOP**; choose `bash /media/$USER/TTL/start.sh --new-visit` for an intentional new visit.
   Old visits are kept and exported. **STOP:** fix the reported checksum/mount problem before retrying.
   No successful preflight means no live run. Context survives a closed terminal.

2. **Confirm delivery.** Start staging before CSI's first save. CSI may use the **same camera folder** for
   staging and live, including one repeatedly rewritten filename. Supported sizes: **1920×1080 or 640×480**.
   Ask CSI for supported captures; never resize, crop or re-aim.

3. **12:45–1:30 — Stage:** `bash ~/ttl/stage.sh`
   Enter the camera folder once. At each setup: **Enter** after CSI saves; **s** skip; **r** retake the last
   setup after a new save; **q** done. Expect filename, size, SHA256 and `staged`/`skipped`/`retake`, ending
   **STAGING OK**. Retakes become `setup_NN_rK`; all attempts stay. Duplicate bytes or wrong sizes are
   **REFUSED** without advancing: ask for a new supported capture. **STOP:** preserve files and reconcile
   the reported problem before resuming. Record setup numbers, labels, tilt evidence and timing on paper.

4. **1:30–1:35 — Seal:** `bash ~/ttl/seal.sh`
   Expect counts, final/earlier-attempt lists, checksum **OK**, then **SEALED USB OK**. Last attempt per
   setup is final. Zero captures prints **WARNING** and still seals. USB failure gives **SEALED DISK OK**;
   live is allowed, but export must back up the disk copy. **STOP:** reconcile log/files/hashes; never erase
   evidence. An already sealed visit refuses resealing or more staging. Reset the fixture before live.

5. **1:35–1:50 — Live:** `bash ~/ttl/live.sh`
   Enter the camera folder (the staging camera folder is allowed) or approved RTSP/HTTP(S) URL.
   **Enter = n (continuous camera feed); y only for stills CSI saves on request.** Confirm fixture reset and
   cleared buffered staged frames. Expect new **LIVE OUT**, inspector/hash, `watching ...` for folders,
   then console grids. Existing files are ignored; CSI saves new live frames after `watching`.
   Staged/sealed copies and overlapping folders are refused. Never recopy staged captures into live.
   `??` means check by eye. **STOP:** correct the reported source/seal problem; restart into a new OUT.

6. **1:50 — Stop:** Ctrl-C once. Expect frame/event summary and **LIVE OK** (`stop_reason: ctrl-c`).
   **STOP:** retain evidence and record the abnormal stop on paper. Log every restart/gap/intervention.
   CSI refuses live or there is no usable terminal: preserve captures; record staging-only, not a live demo.

7. **1:50–2:00 — Export:** stop writers, restore USB, then `bash ~/ttl/export.sh`.
   Expect archive checksums **OK → RESULTS USB OK: N visits verified**; includes every retry and attempt.
   **STOP:** keep GB10 files and arrange CSI-assisted recovery. Re-verify on the Mac before removing anything.
   On the Mac: adjudicate paper labels without predictions, lock labels by SHA256, score each arm once (§5).

Cut optional live defects, then filming, then live duration; never labels or backups. Real camera streaming and
capped 1080 performance remain unverified. Appendix A retains the long blocks for reference; these steps govern.

## Appendix A. Original manual procedure (reference only)

These pre-script blocks preserve the earlier procedure. §7 scripts are authoritative: they allow zero captures, retain retakes, reuse the camera folder and permit a verified disk seal when USB fails.
The old 22-image, USB-only and separate-camera-folder checks below do not govern Monday.


**12:30–2:00 only; no installation before the slot.** Frozen rule B′ runs offline on aarch64 Linux;
no torch or Isaac is needed. Organizer permission for its DEV-fit thresholds remains a condition.
Bring the Sunday-attested package and checksum on USB, printed staging/numbering sheets, label log,
paper intervention log, laptop, charger and pen. Confirm CSI's supervisor and approved file/feed route;
never change cameras, plant controls or network settings.

**Time limits:** 12:30–12:45 setup; 12:45–1:30 staging; 1:30–1:35 sealed backup;
1:35–1:50 live; 1:50–2:00 export. These are allowances, not measured staging times.
Cut optional live defects first, filming second, then live duration. Never cut labels or backups.

1. **12:30 setup: identify the USB and set the paths once.** Use Bash. `lsblk -o NAME,SIZE,MOUNTPOINT`
   shows mounts; CSI must mount an unmounted stick. Enter actual values at the prompts (no angle-bracket
   substitutions). `PKG` is the package basename printed in Sunday's attestation, without `.tar`.
   `IN` is a new folder that CSI must agree to write into after staging; if CSI cannot do so, use the
   approved fresh stream or on-request stills route below. Do not point it at the staged folder.

   ```bash
   read -r -p 'Mounted USB directory: ' USB
   read -r -p 'Sunday package basename, without .tar: ' PKG
   DAY="$HOME/ttl/visit_$(date +%Y%m%dT%H%M%S)_$$"
   STAGED="$DAY/staged"
   IN="$DAY/live_in"
   OUT="$DAY/attempts/live_$(date +%Y%m%dT%H%M%S)_$$"
   INSTALL="$DAY/install_$(date +%Y%m%dT%H%M%S)_$$"
   (
     set -euo pipefail
     trap 'echo "STOP: setup failed; preserve this visit folder. Do not start live." >&2' ERR
     [[ "$PKG" =~ ^ttl_gb10_[0-9a-f]+$ ]]
     test -d "$USB" && test -w "$USB"
     mkdir -p "$HOME/ttl"
     mkdir "$DAY"
     mkdir "$STAGED" "$IN" "$DAY/attempts" "$DAY/records" "$INSTALL"
     printf '%s\n' 'time,action,operator,notes' > "$DAY/records/interventions.csv"
     printf '%s\n' 'setup,filename,capture_time,collector,measured_tilt,notes' > "$DAY/records/staging_log.csv"
     printf '%s\n' 'No live attempt yet.' > "$DAY/records/live_status.txt"
     cp "$USB/$PKG.tar" "$USB/$PKG.tar.sha256" "$INSTALL/"
     cd "$INSTALL"
     sha256sum -c "$PKG.tar.sha256"
     tar -xf "$PKG.tar"
     cd "$PKG"
     bash bootstrap.sh 2>&1 | tee "$DAY/records/bootstrap_$(date +%Y%m%dT%H%M%S).log"
     printf 'USB=%q\nPKG=%q\nDAY=%q\nSTAGED=%q\nIN=%q\nOUT=%q\nINSTALL=%q\n' \
       "$USB" "$PKG" "$DAY" "$STAGED" "$IN" "$OUT" "$INSTALL" > "$DAY/operator.env"
     echo "SETUP OK; context: $DAY/operator.env"
   )
   ```

   **Proceed only after `PREFLIGHT OK`, `BOOTSTRAP OK` and `SETUP OK`. Any checksum failure is STOP.**
   The subshell can fail without closing the terminal. Check the archive commit against Sunday's attestation.
   Expected rehearsal: offline bootstrap about 5 s; all still-image states equal the Mac, score difference 0.
   Video parity is not covered (RC MP4: 3/540 state differences). The RC package is not the Sunday release.
   A failed attempt stays in `DAY`. For an installation retry repeat step 1, creating a new timestamped visit;
   retain every visit under `~/ttl` for step 7. Never unpack over an installation.
   Bundled-Python failure falls back automatically to system Python 3.10–3.12 with venv (container-tested).
   No usable Python or failed preflight: no live run. Offline-install failure: CSI may permit a fresh attempt
   using `bash bootstrap.sh --online`; record it. No permission means staging only.

2. **Confirm capture delivery before staging.** Ask CSI to save each setup once under a unique name,
   `setup_00.jpg` through `setup_21.jpg` (PNG is also accepted), in `STAGED`.
   A camera that overwrites one filename requires CSI to save a separate immutable copy after every setup,
   before changing the fixture. Check image dimensions without an inspector:

   ```bash
   PY="$INSTALL/$PKG/venv/bin/python"
   read -r -p 'Full path of a CSI capture to check dimensions: ' CHECK_IMAGE
   "$PY" -c 'import cv2,sys; im=cv2.imread(sys.argv[1]); print(None if im is None else im.shape)' "$CHECK_IMAGE"
   ```

   Expect `(1080, 1920, 3)` or `(480, 640, 3)`. Wrong size, including 320×240: ask CSI for supported
   stills; never resize, crop, re-aim or recalibrate. If unavailable, preserve the collection and log that
   live inference could not run. Real capped 1080 images, CSI's own GB10 and its camera stream remain unverified.

3. **12:45–1:30 staging: no inspector runs.** Noor and Tony reconcile each setup 00–21 to its unique saved
   filename in `records/staging_log.csv`; record capture time, collector, actual positions/classes, tilt evidence
   and deviations on the label sheet. Do not infer ground truth from model output. Keep per-setup timing and
   interventions. Side photos/filming need CSI's approval. Save label sheets and a legible photo or transcription
   of the paper log under `records` before backup; retain the originals. Geometry-only overlays are permitted
   for later adjudication; predictions, scores and model-coloured overlays are not.

4. **1:30–1:35 seal and verify on USB BEFORE live.** Reconcile filenames and exactly 22 unique setups against
   the log, check no image is still being written, and ask CSI to stop writing to `STAGED`. This block checks
   the log, packages images plus records without inference, and verifies the copied archive. Count alone is
   insufficient. On failure: STOP; do not start live or discard the source files.

   ```bash
   (
     set -euo pipefail
     trap 'echo "STOP: sealed backup failed; no live run." >&2' ERR
     "$PY" - "$STAGED" "$DAY/records/staging_log.csv" <<'PY'
   import csv, pathlib, sys
   folder = pathlib.Path(sys.argv[1])
   rows = list(csv.DictReader(open(sys.argv[2], newline="")))
   names = [r["filename"] for r in rows]
   assert len(rows) == 22 and {r["setup"] for r in rows} == {f"{i:02d}" for i in range(22)}
   assert len(set(names)) == 22 and all(pathlib.Path(n).name == n for n in names)
   images = {p.name for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
   assert set(names) == images, "filename-to-setup mismatch"
   assert all((folder / n).is_file() and (folder / n).stat().st_size for n in names)
   print("RECONCILED: 22 setups, 22 unique captures")
   PY
     ARCHIVE=$(mktemp "$DAY/sealed_$(date +%Y%m%dT%H%M%S)_XXXXXX.tar")
     tar -cf "$ARCHIVE" -C "$DAY" staged records
     (cd "$DAY" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
     DEST=$(mktemp -d "$USB/sealed_$(date +%Y%m%dT%H%M%S)_XXXXXX")
     cp "$ARCHIVE" "$ARCHIVE.sha256" "$DEST/"
     sync
     (cd "$DEST" && sha256sum -c "$(basename "$ARCHIVE").sha256")
     printf '%s\n' "$DEST" > "$DAY/records/sealed_usb_verified.txt"
     echo 'SEALED USB OK; CSI may now reset the fixture for a separate live run.'
   )
   ```

5. **1:35–1:50 live, after `SEALED USB OK` and CSI's fixture reset.** CSI must use `IN`, a fresh live-only
   folder, or a fresh feed after resetting the fixture and clearing buffered staged frames. Never reuse the staged
   feed/folder: existing-file skipping does not protect late or rewritten staged captures. Log reset/start times.
   The first live arrival must be after the watcher starts. Use the folder command below; independent stills
   saved on request add `--gate each`. New filenames and a rewritten live-only filename are tested.

   ```bash
   (
     set -euo pipefail
     trap 'echo "STOP: live attempt failed; keep evidence and record the intervention." >&2' ERR
     test -s "$DAY/records/sealed_usb_verified.txt"
     test "$IN" != "$STAGED"
     mkdir "$OUT"
     cp "$DAY/operator.env" "$OUT/operator.env"
     cd "$INSTALL/$PKG/code/twin_to_line"
     "$PY" live_inspect.py --source "folder:$IN" --out "$OUT" 2>&1 | tee "$OUT/console.log"
   )
   ```

   A stream requires CSI's approved RTSP/HTTP URL: `read -r -p 'Approved fresh live URL: ' SRC`, then replace
   `--source "folder:$IN"` with `--source "stream:$SRC" --every 1.0` in the block. Real-camera streaming is
   untested; local MJPEG drop/reconnect was tested. Raw devices are refused. Drops/timeouts: use CSI-saved
   files; no reachable source: staging only. Keep original live captures/recording where CSI permits; the runtime
   saves predictions and overlays, not a raw stream archive. Do not put credential-bearing URLs in paper notes.
   Folder arrivals should log within about 1 s; events follow 4 s of arm-clear stillness. Sparse/changing stills
   may remain `settling`: stop, request independent new captures, and restart with `--gate each`.
   Before EVERY restart set a new timestamped output and update the context (never reuse an output):
   `OUT="$DAY/attempts/live_$(date +%Y%m%dT%H%M%S)_${RANDOM}"`; append `printf 'OUT=%q\n' "$OUT" >> "$DAY/operator.env"`.
   Log the gap: arrivals before restart are ignored. Do not replay them or any staged images on the day.

   **Second terminal: its own directory and variables.** Paste the absolute `operator.env` path inside the
   current `OUT`, printed/located by the first operator; this snapshot belongs to that attempt.

   ```bash
   read -r -p 'Current attempt operator.env absolute path: ' CONTEXT
   source "$CONTEXT"
   cd "$INSTALL/$PKG/code/twin_to_line"
   tail -f "$OUT/trial_log.csv" | cut -d, -f6,8,77
   ```

   Ctrl-C here stops only `tail`. The first terminal prints the primary console grid; optional browser view is
   `$OUT/view.html`. No browser: use the grid. No screen or terminal: CSI saves captures, labels and logs directly
   to USB; review on the laptop afterwards, keeping staged predictions hidden until labels are locked. If an
   already-running system is inaccessible, ask CSI to stop it and copy its evidence; do not assume a clean stop.
   CSI refuses live: staging only, record the refusal and missing live demo. No USB mount: CSI must provide an
   approved transfer route or retain the files; this fallback is not tested. Never claim an unverified backup.

6. **1:50 stop.** Ctrl-C once in the first terminal; expect the frame/event summary and
   `session.json` with `stop_reason: ctrl-c`. If the shell reports interruption before that summary, inspect the
   session/logs and record an abnormal stop. Save the final label sheets and paper-log photo/transcription in
   `records`, log every attempt/refusal/restart and actual start/stop times, and update `live_status.txt`.
   On a staging-only day write the reason; no `results/live` directory is required for export.

7. **1:50–2:00 export ALL visits/attempts and verify.** Stop all writers first. This archives every timestamped
   visit under `~/ttl`, including failed installations, preflights, outputs, live captures, staged backups and
   records. Retry visit folders are included. Ensure label sheets and the paper log are present; copy any
   separately saved CSI evidence into a visit's `records` first. Space includes installation packages: bring a
   USB with adequate free space (several GB plus recordings). No archive or USB destination is reused.

   ```bash
   (
     set -euo pipefail
     trap 'echo "STOP: final export failed; keep GB10 files and arrange CSI-assisted recovery." >&2' ERR
     shopt -s nullglob
     VISITS=("$HOME"/ttl/visit_*)
     test "${#VISITS[@]}" -gt 0
     EXPORT=$(mktemp -d "$HOME/ttl/export_$(date +%Y%m%dT%H%M%S)_XXXXXX")
     ARCHIVE="$EXPORT/ttl_results_$(date +%Y%m%dT%H%M%S).tar"
     NAMES=(); for v in "${VISITS[@]}"; do NAMES+=("$(basename "$v")"); done
     tar -cf "$ARCHIVE" -C "$HOME/ttl" "${NAMES[@]}"
     (cd "$EXPORT" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
     DEST=$(mktemp -d "$USB/results_$(date +%Y%m%dT%H%M%S)_XXXXXX")
     cp "$ARCHIVE" "$ARCHIVE.sha256" "$DEST/"
     sync
     (cd "$DEST" && sha256sum -c "$(basename "$ARCHIVE").sha256")
     echo "RESULTS USB OK: $DEST"
   )
   ```

   Keep GB10 files until the Mac verifies the copied checksum (`shasum -a 256 -c` from its export directory).
   Delete nothing unless CSI asks. Monday evening: adjudicate without predictions, lock labels by SHA256,
   then use `score_sealed.py` (§5), once for all declared arms. No threshold changes or selection from test results.
