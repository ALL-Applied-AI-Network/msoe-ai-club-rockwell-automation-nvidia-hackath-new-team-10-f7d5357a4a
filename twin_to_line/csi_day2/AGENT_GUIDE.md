# CSI Tue Oct 27 · noon–2 pm · FIFO only
This guide is pasted with the start prompt; no guide file is needed on the GB10.
Reply to Noor in 1–2 plain lines. Run EACH command line below as a separate tool call.

## Rules and first five minutes
Run the supplied kit scripts as written. Never write capture/inspection code,
edit code/thresholds/settings, open or score staged images, touch robot/camera/PLC/
network, use git, or explore the repo. Noor/Tony move vials; CSI delivers images.
Preserve everything, including Monday's trial-only `codecrusaders/data`; never use
those pictures. Only the temporary FIFO may be unlinked, after its script exits.
On STOP: report the full line verbatim, log it, and wait. A fallback TERM stop
can produce STOP instead of LIVE OK; Noor must acknowledge it before export.
```bash
ls -ld ~/ttl_kit ~/ttl
command -v bash mkfifo setsid nohup tail pkill flock sha256sum
cat ~/ttl_kit/agent_log.md
```
Expected: kit and tools present; `ttl/` and log may be absent before first setup.
Missing kit: ask Noor for the download command. Missing tools: report and wait.
Ask Noor for camera folder, continuous/requested-still mode, and first start versus
resume. Never use the trial folder. No shell variable survives between tool calls.
Record pending actions BEFORE issuing them, and completion ONLY after reading output:
```bash
printf '%s %s\n' "$(date -Is)" 'PENDING setup; first Tuesday start confirmed' >> ~/ttl_kit/agent_log.md
```
Use this same append command with observed state after every step: last completed
step, next action/prompt, setup, filename/bytes/time, skip reason, paths and PIDs.
## Setup — once only for Tuesday
```bash
bash ~/ttl_kit/start.sh --new-visit >> ~/ttl_kit/setup.log 2>&1
tail -n 30 ~/ttl_kit/setup.log
```
Expected: `SETUP OK: preflight passed; context .../operator.env`. Log exact context
and `COMPLETED setup; NEXT stage`. Explicit new visit preserves Monday. Plain
`start.sh` resumes READY=1 regardless of date; never use it to begin Tuesday.
On resume only, if no writer runs: `bash ~/ttl_kit/start.sh` → `SETUP OK: resumed ...`.
## Stage — create FIFO and persistent holder
If `~/ttl_in` already exists, inspect resume state; do not replace it blindly.
```bash
mkfifo ~/ttl_in
nohup setsid bash -c 'exec 3<> "$HOME/ttl_in"; exec sleep 10800' </dev/null >> ~/ttl_kit/fifo_holder.log 2>&1 & echo $! > ~/ttl_kit/fifo_holder.pid
cat ~/ttl_kit/fifo_holder.pid
kill -0 "$(cat ~/ttl_kit/fifo_holder.pid)"
nohup setsid bash ~/ttl/stage.sh < ~/ttl_in >> ~/ttl_kit/stage.log 2>&1 & echo $! > ~/ttl_kit/stage.pid
tail -n 30 ~/ttl_kit/stage.log
```
Expected: holder PID, exit 0, then `Camera folder (ask CSI; before the first save):`.
The holder opens read/write, preventing EOF when individual answer writers close.
All prompts use stdin; no terminal is required. Do not send multiple answers together.
Replace only the path below with Noor's confirmed camera folder (quote it safely):
```bash
printf '%s\n' '/CAMERA/FOLDER/FROM/NOOR' > ~/ttl_in
tail -n 30 ~/ttl_kit/stage.log
```
Expected: `Existing camera images ignored. Ask CSI to save setup 00 now.` and
`setup 00: Enter after CSI saves it, s = skip, r = retake the last setup, q = done:`.
Noor says “setup NN saved”: verify NN is the current prompt, then send:
```bash
printf '\n' > ~/ttl_in
tail -n 12 ~/ttl_kit/stage.log
```
Expected: `setup_NN.EXT: WIDTHxHEIGHT; sha256 ...`, then `setup NN: staged` and next
prompt. “retake”: first confirm a NEW save for the LAST setup, then:
```bash
printf 'r\n' > ~/ttl_in
tail -n 12 ~/ttl_kit/stage.log
```
Expected: `setup_NN_r1.EXT: ...`, `setup NN: retake`; earlier copy retained.
The displayed prompt is the NEXT setup; `r` retakes the preceding one.
“skip: REASON”: append reason to agent_log.md BEFORE sending:
```bash
printf 's\n' > ~/ttl_in
tail -n 12 ~/ttl_kit/stage.log
```
Expected: `setup NN: skipped`, next prompt. “done”:
```bash
printf 'q\n' > ~/ttl_in
tail -n 20 ~/ttl_kit/stage.log
```
Expected: `STAGING OK: N setups staged; skips and retakes logged; no inference`.
If no new image/still changing/REFUSED: report it; setup has not advanced.
After each successful capture, read metadata only (replace EXACT_FILENAME):
```bash
bash -c 'source ~/ttl/operator.env; stat -c "%n | %s bytes | %y" "$STAGED/EXACT_FILENAME"; tail -n 2 "$DAY/records/staging_log.csv"'
```
Reply and append one line: `setup=NN file=NAME bytes=N time=CSV_TIME status=staged
COMPLETED setup NN; NEXT NN`. For retake include attempt; skip uses file=none,
bytes=0, reason and current time. Do not open/decode images yourself.
Staging ends by **1:20 pm**. Plan omissions in priority 18, 15, 17, 16; the script
is sequential, so decide early and send `s` at each selected numeric prompt.
Never skip 00, 01, 06, 19–21. Reserve time for 19–21; warn Noor early of any
coverage/deadline conflict. Do not silently skip requirements or overrun 1:20.
After STAGING OK, verify no operator writer, then clean only holder and FIFO:
```bash
flock -n ~/ttl/operator.lock true
kill -TERM "$(cat ~/ttl_kit/fifo_holder.pid)"
unlink ~/ttl_in
```
Expected: exit 0 each. Preserve PID files/logs; log `COMPLETED stage; NEXT seal`.
## Seal
```bash
bash ~/ttl/seal.sh >> ~/ttl_kit/seal.log 2>&1
tail -n 30 ~/ttl_kit/seal.log
```
Expected: `RECONCILED: ...`, `FINAL: ...`, `SEALED USB OK: ...; reset fixture before live`.
`SEALED DISK OK: USB FAILED ...`: tell Noor; keep disk seal, do not reseal. Zero
staged gives WARNING. Log seal destination and `COMPLETED seal; NEXT live reset`.
Seal checks all retained capture hashes/names/log order/exact staged membership,
archive capture bytes and archive copy hashes. Never put logs in `staged/`.
## Live — fresh FIFO and holder; launch WITHOUT &
Get CSI reset/cleared-buffer confirmation through Noor; never invent it.
```bash
mkfifo ~/ttl_in
nohup setsid bash -c 'exec 3<> "$HOME/ttl_in"; exec sleep 10800' </dev/null >> ~/ttl_kit/fifo_holder.log 2>&1 & echo $! > ~/ttl_kit/fifo_holder.pid
kill -0 "$(cat ~/ttl_kit/fifo_holder.pid)"
setsid -f nohup bash ~/ttl/live.sh < ~/ttl_in >> ~/ttl_kit/live_console.log 2>&1
pgrep -f '[m]onday_ops.py live$' > ~/ttl_kit/live.pid
tail -n 20 ~/ttl_kit/live_console.log
```
Expected: `Live camera folder or approved RTSP/HTTP(S) stream URL:`. Send folder:
```bash
printf '%s\n' '/CAMERA/FOLDER/FROM/NOOR' > ~/ttl_in
tail -n 20 ~/ttl_kit/live_console.log
```
Same camera source allowed; never staged/sealed copies. Expected gate question:
`Enter = n (continuous camera feed); y only for stills CSI saves on request. [y/n]:`.
For continuous send `printf '\n' > ~/ttl_in`; for requested stills send:
```bash
printf 'y\n' > ~/ttl_in
tail -n 20 ~/ttl_kit/live_console.log
```
Expected: `CSI reset the fixture and cleared staged/buffered frames? [y/n]:`.
ONLY after Noor confirms reset:
```bash
printf 'y\n' > ~/ttl_in
tail -n 40 ~/ttl_kit/live_console.log
```
Expected: `LIVE OUT: ...`, `watching ... new images only`, then grids on new images.
Log OUT path, gate/reset confirmation and `COMPLETED live launch; NEXT stop`.
`ok` PASS; `DEF` DEFECT; `MISS` MISSED; `??` UNSURE; `.` WAITING.
Prompts/staging lines flush explicitly; live child uses `-u`, wrapper flushes lines.
Python `-I` ignores PYTHONUNBUFFERED. Spark proof: prompts/confirmations appeared
while stage stayed alive; clean INT stop retained 600 prediction and 21 trial rows.
On Noor's stop instruction, ask CSI to finish the current save and pause delivery.
Then run this standalone command (never put it inside a launch wrapper):
```bash
pkill -INT -f live_inspect.py
tail -n 8 ~/ttl_kit/live_console.log
```
Expected: `N frames, M events, {...} -> ...`, then
`LIVE OK: stopped cleanly; evidence ...; export every visit`.
Launching WITHOUT `&` preserves SIGINT handling; `setsid -f` detaches and returns.
Do not replace this with an asynchronous shell launch or send a Ctrl-C byte to FIFO.
If not stopped within **30 seconds**, use the fallback in a separate call:
```bash
pkill -TERM -f live_inspect.py
tail -n 15 ~/ttl_kit/live_console.log
```
Fallback can give `STOP: live: live runtime failed (-15); preserve ...` instead of
LIVE OK, with incomplete final metadata/in-flight frame. Report verbatim and wait
for Noor's acknowledgement before export. Do not start another live run.
After wrapper exits (lock test succeeds), clean holder/FIFO, retaining every log:
```bash
flock -n ~/ttl/operator.lock true
kill -TERM "$(cat ~/ttl_kit/fifo_holder.pid)"
unlink ~/ttl_in
```
After LIVE OK, check metadata (no image viewing):
```bash
bash -c 'source ~/ttl/operator.env; grep -E "ended_utc|stop_reason|frames|events" "$OUT/session.json"; tail -n 1 "$DAY/records/live_status.txt"'
```
Expected: end time, `stop_reason: "ctrl-c"`, counts and `live exit=0`.
Log `COMPLETED live INT stop; LIVE OK; OUT=...; NEXT export` only after verification.
## Export and handoff
```bash
bash ~/ttl/export.sh >> ~/ttl_kit/export.log 2>&1
tail -n 20 ~/ttl_kit/export.log
```
Expected: `RESULTS USB OK: N visits verified at ...; keep GB10 files until Mac verification`.
Log destination and `COMPLETED export; NEXT CSI copy`. Export hashes all visit_*
archives locally and at destination; keep the separate live-stop evidence.
Extra kit-root logs are safe and travel with the whole-folder copy, outside the
visit archives. Monday visits stay separate; external trial pictures stay untouched.
Say exactly: “CSI, please copy the entire ~/ttl_kit/ folder for us, including the
sealed folders, visit_export folders and session logs; please keep the GB10 files
until we verify the copy on our Mac.” Leave time to finish before 2 pm.
## Resume after usage limit
Fresh agent reads this pasted guide plus `cat ~/ttl_kit/agent_log.md`, then tails
stage.log/live_console.log/export.log as appropriate. Check saved PID with `ps -p PID`.
Do not recreate FIFO/holder while active; continue answering the current prompt.
Inspect current visit metadata, never images:
```bash
bash -c 'source ~/ttl/operator.env; printf "DAY=%s\nSEALED=%s\nOUT=%s\n" "$DAY" "${SEALED:-}" "${OUT:-}"; tail -n 5 "$DAY/records/staging_log.csv"; cat "$DAY/records/live_status.txt"'
```
If an action is pending, reconcile its result before resending. Never repeat
--new-visit, reseal, duplicate live, or export while a writer runs. If stage died,
log that fact and repeat its FIFO launch only after confirming the prior writer
exited; CAMERA/next setup persist. If uncertain, tell Noor the last completed step.
