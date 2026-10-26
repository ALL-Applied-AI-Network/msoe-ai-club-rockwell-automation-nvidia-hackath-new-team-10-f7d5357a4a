# CSI Tuesday Oct 27, 2026 · noon–2 pm
Read only this guide and the session log. Replies to Noor: 1–2 plain lines.

## Hard rules
Run only the supplied kit scripts as written. Never write capture/inspection code,
edit code, thresholds or settings, score or open staged images, touch the robot,
camera, PLC or network, delete anything, use git, or explore the repo.
Today's `codecrusaders/data` pictures are trial-only: preserve, never use them.
Only CSI operates camera delivery; Noor and Tony move vials and report events.
Shell commands below control terminals and record evidence; no helper scripts.
On ANY `STOP`: report the entire line verbatim to Noor, log it, and wait.
On `REFUSED` or no new image: report it; do not advance the setup.
Do not auto-retry failed seal/setup or invent a repair.

## First five minutes
Run each command block as a separate non-interactive call. Do not assume shell
variables survive calls. Replace uppercase placeholders with observed values.
```bash
ls -ld ~/ttl_kit ~/ttl
command -v tmux; command -v mkfifo
```
Expected: kit directory and tool paths; `ttl/` may be absent before first setup.
If kit is missing, ask Noor for the download command. Do not search for it.
Ask Noor for CSI's camera folder path, continuous feed versus requested stills,
and confirmation this is the FIRST Tuesday start. Never use Monday's trial folder.
Check an existing Tuesday log before starting anything:
```bash
cat ~/ttl_kit/csi_day2_session.log
```
Expected: prior state, or missing file on first start. Append, never truncate:
```bash
printf '%s\n' 'BEGIN Tuesday Oct 27; first start confirmed; mode=A; camera=CAMERA_PATH; phase=setup pending' >> ~/ttl_kit/csi_day2_session.log
```
Select A if tmux works. B if only mkfifo works (live requires human key fallback).
Otherwise C: use HUMAN_FALLBACK.md; tell Noor the exact pending command/key.
A launch failure means stop and record it before changing mode; never run two writers.

## A — persistent terminal (preferred)
Create one terminal; inspect before sending any keys:
```bash
tmux new-session -d -s ttl-day2 -x 160 -y 48
```
Expected: exit 0. If session already exists, attach logically by capturing it; do
not replace it. This terminal survives an agent usage limit.
After EVERY send, use this separate call (repeat while working):
```bash
tmux capture-pane -p -S -200 -t ttl-day2:0.0
```
Expected: current prompt or final OK/STOP. Do not batch prompt answers blindly.
Send a literal string, then Enter separately; this preserves spaces/metacharacters:
```bash
tmux send-keys -t ttl-day2:0.0 -l 'TEXT'
tmux send-keys -t ttl-day2:0.0 Enter
```
For empty Enter, omit the literal-string command. Never send shell commands while
a script is awaiting input. Log every issued action before sending, and its result
afterwards (phase, pending prompt, setup, paths, time, OK/STOP). Example:
```bash
printf '%s %s\n' "$(date -Is)" 'phase=stage; action=Enter setup 00 pending' >> ~/ttl_kit/csi_day2_session.log
```

## Setup: clean Tuesday visit, once only
Send this as TEXT in A:
```bash
bash ~/ttl_kit/start.sh --new-visit 2>&1 | tee -i -a ~/ttl_kit/csi_day2_terminal.log
```
Expected: `SETUP OK: preflight passed; context .../operator.env`.
Log the exact context path and `phase=setup OK; next=stage`.
`--new-visit` creates a unique `~/ttl/visit_TIMESTAMP_RANDOM/`; it preserves older
visits. Plain `start.sh` resumes READY=1 regardless of date; it would mix Monday
and Tuesday. On a successful Tuesday resume use plain `bash ~/ttl_kit/start.sh`
only if no script is active: expected `SETUP OK: resumed ...`.
If first setup failed, STOP and wait: a retry can create another retained visit.

## Stage
Send as TEXT:
```bash
bash ~/ttl/stage.sh 2>&1 | tee -i -a ~/ttl_kit/csi_day2_terminal.log
```
At `Camera folder (ask CSI; before the first save):`, send the confirmed path.
Expected: `Existing camera images ignored. Ask CSI to save setup 00 now.`
The script hashes existing images as baseline; CSI must now save fresh images.
This baseline is not permission to use Monday's trial pictures.
At each `setup NN: Enter ...` prompt:
- Noor says “setup NN saved”: verify NN matches; send Enter.
- “retake”: confirm CSI has saved a NEW image for the LAST setup; send `r` + Enter.
  The displayed prompt names the next setup; `r` retakes the preceding one.
- “skip: REASON”: log reason first; send `s` + Enter for the current setup.
- “done”: send `q` + Enter.
Expected capture: `setup_NN[ _rN].EXT: WIDTHxHEIGHT; sha256 ...` (no space in name),
then `setup NN: staged` or `setup NN: retake`; skip: `setup NN: skipped`.
Confirm the result before logging completion. For a successful capture, use the
exact filename printed (metadata only; do not display/decode the image):
```bash
bash -c 'source ~/ttl/operator.env; stat -c "%n | %s bytes | %y" "$STAGED/EXACT_FILENAME"; tail -n 2 "$DAY/records/staging_log.csv"'
```
Expected: copied filename, bytes, file time, and CSV capture time/hash.
Reply and append ONE line with setup, filename, bytes, CSV time, dimensions,
status and next setup. Use `file=none size=0` for skips, plus reason and current time:
```bash
printf '%s\n' 'setup=NN file=EXACT_FILENAME bytes=COUNT time=CSV_TIME dimensions=WxH status=staged next=NN' >> ~/ttl_kit/csi_day2_session.log
```
End expected: `STAGING OK: N setups staged; skips and retakes logged; no inference`.
Log `phase=staging OK; next=seal`.
Staging ends by **1:20 pm**. Reduce planned coverage by dropping 18, then 15, then
17, then 16. The script visits in numeric order: decide this skip set early with
Noor, then send `s` when each chosen setup arrives. Never skip 00, 01, 06, 19–21.
Reserve time for 19–21; warn Noor early if the deadline/required coverage conflict.
Do not silently skip required setups or run beyond 1:20; report incomplete coverage.

## Seal
Send as TEXT:
```bash
bash ~/ttl/seal.sh 2>&1 | tee -i -a ~/ttl_kit/csi_day2_terminal.log
```
Expected: `RECONCILED: ...`, `FINAL: ...`, then `SEALED USB OK: ...; reset fixture before live`.
`SEALED DISK OK: USB FAILED ...` means disk seal succeeded; tell Noor and log the
copy failure. Do not reseal; recover through export. Zero staged prints WARNING.
Log seal path/result and `phase=sealed; next=reset/live`. Staging is now closed.

## Live
Ask Noor to have CSI reset the fixture and clear staged/buffered frames. Send:
```bash
bash ~/ttl/live.sh 2>&1 | tee -i -a ~/ttl_kit/csi_day2_terminal.log
```
At `Live camera folder or approved RTSP/HTTP(S) stream URL:`, send the confirmed
camera folder (same staging source allowed; never staged/sealed copies).
At `Enter = n (continuous camera feed); y only for stills CSI saves on request. [y/n]:`,
send Enter for continuous or `y` + Enter for requested stills, as confirmed by Noor.
At `CSI reset the fixture and cleared staged/buffered frames? [y/n]:`, send `y`
ONLY after Noor confirms. Otherwise wait; never invent the reset.
Expected: `LIVE OUT: ...`, then `watching ... new images only` for folder input.
Log output path, gate, reset confirmation, `phase=live running` and start time.
Grid: `ok` PASS, `DEF` DEFECT, `MISS` MISSED, `??` UNSURE, `.` WAITING.
Stop on Noor's instruction, leaving time to export before 2 pm. Send once:
```bash
tmux send-keys -t ttl-day2:0.0 C-c
```
Capture output. Expected: summary, then `LIVE OK: stopped cleanly; evidence ...; export every visit`.
If still running after 30 seconds, run in a separate call:
```bash
pkill -TERM -f live_inspect.py
```
This can produce STOP instead of LIVE OK: report verbatim and wait for Noor's
recovery direction. Never claim a clean run after forced termination.

## Export and handoff
Once every writer has stopped, send as TEXT:
```bash
bash ~/ttl/export.sh 2>&1 | tee -i -a ~/ttl_kit/csi_day2_terminal.log
```
Expected: `RESULTS USB OK: N visits verified at ...; keep GB10 files until Mac verification`.
Log exact destination and `phase=export OK; next=CSI copy`; retain all files.
Say exactly: **“CSI, please copy the entire ~/ttl_kit/ folder for us, including the
sealed folders, visit_export folders and session logs; please keep the GB10 files
until we verify the copy on our Mac.”**
Exports contain ALL `~/ttl/visit_*`, including Monday visits, in distinct directories.
They do not include `codecrusaders/data`: leave those trial pictures untouched.
Extra logs in kit root are safe: manifests check named files, seal checks staged
files/logs and archives, export verifies archive hashes locally and at destination.
Do not put extra files in `staged/`; exact membership is checked. Kit-root logs
travel with the requested whole-folder copy, not inside evidence.tar.

## B — FIFO fallback; stdin only
All kit questions use stdin (`input()`), not `/dev/tty`. A FIFO does NOT create a
terminal and cannot answer `/dev/tty`; A can. FIFO EOF/lifetime must be checked.
Use a fresh control directory, record its path as CONTROL in the session log:
```bash
mktemp -d "$HOME/ttl_kit/control_XXXXXX"
mkfifo CONTROL/input
nohup bash -c 'exec 3<>"$1"; sleep 10800' _ CONTROL/input </dev/null >CONTROL/holder.log 2>&1 & echo $!
```
Expected: directory, exit 0, holder PID; log PID. Check `kill -0 HOLDER_PID`.
Run setup directly, using the exact setup command above; expect SETUP OK.
Launch stage in its own call:
```bash
nohup bash ~/ttl/stage.sh <CONTROL/input >>~/ttl_kit/csi_day2_terminal.log 2>&1 & echo $!
```
Read `tail -n 80 ~/ttl_kit/csi_day2_terminal.log`; expect the same prompts as A.
Each answer is a separate call; empty answer is `printf '\n'`:
```bash
printf '%s\n' 'ANSWER' >CONTROL/input
```
Follow the same staging/logging rules. Wait for STAGING OK, then run seal directly.
For live, use C: Noor runs the exact live command in a foreground terminal and
answers the three prompts, then presses Ctrl-C once on instruction. Background
shell jobs can inherit ignored SIGINT; FIFO byte 0x03 is not a terminal Ctrl-C.
Do not claim FIFO-only clean live control is proven. Export directly after stop.
If the agent environment kills detached jobs or FIFO stalls, use C; preserve files.

## Resume after usage limit
Read this guide and `~/ttl_kit/csi_day2_session.log`, then capture the existing tmux
pane (or tail the terminal log in B). Do not repeat `--new-visit`.
If an action is pending, inspect its output before resending. Metadata recovery:
```bash
bash -c 'source ~/ttl/operator.env; printf "DAY=%s\nSEALED=%s\nOUT=%s\n" "$DAY" "${SEALED:-}" "${OUT:-}"; tail -n 5 "$DAY/records/staging_log.csv"; cat "$DAY/records/live_status.txt"'
```
Expected: current visit, seal/live state, recent staging rows, live status.
Log any reconciliation with the pending action. Do not duplicate captures, reseal,
start another live process, or export while a writer runs. If uncertain, tell Noor
what the last confirmed step was and wait. A stopped staging process resumes with
`bash ~/ttl/stage.sh`; CAMERA and next setup persist. Keep all retake attempts.
