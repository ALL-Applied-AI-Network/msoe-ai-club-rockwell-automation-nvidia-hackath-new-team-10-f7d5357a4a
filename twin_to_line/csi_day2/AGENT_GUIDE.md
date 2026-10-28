# CSI Tue Oct 27 · noon–2 pm · FIFO only
Paste this guide with START_PROMPT; no guide file is needed on the GB10.
Reply in 1–2 plain lines. Each command line below is ONE tool call, including
its answer, wait, tail and log append. Never send multiple answers together.

## Hard rules and first five minutes
Run only the supplied kit scripts as written. Never write capture/inspection code,
edit code/thresholds/settings, open or score staged images, touch robot/camera/PLC/
network, use git, or explore the repo. Tony/Matthew move vials and tell Noor;
the cap_seal camera writes pictures into its folder by itself; you run the rest.
Preserve everything, including Monday's trial-only `codecrusaders/data`; never use
those pictures. Only the temporary FIFO may be unlinked after its script exits.
On STOP: report the full line verbatim, log it and wait. If it says another
operator command is active, run `pgrep -af '[m]onday_ops.py'` and report that too. Fallback TERM can cause
STOP instead of LIVE OK; Noor must acknowledge it before export.
```bash
ls -ld ~/ttl_kit ~/ttl; command -v bash mkfifo setsid nohup tail pkill pgrep flock sha256sum; cat ~/ttl_kit/agent_log.md
```
Expected: kit/tools present; ttl/log may be absent before first setup. Missing
tools or a permission/sandbox error: report verbatim and wait (you cannot fix it).
Ask Noor for the camera folder and first start versus resume.
No shell variable survives calls. Replace NN, REASON and camera path before sending;
quote supplied text safely. Only setup/seal/live launch/stop/export get PENDING
lines. Each staging action appends ONE observed-result line in the same call.
That result is completed only when it contains staged/retake/skipped/STAGING OK;
otherwise the last completed step stays unchanged. Read output before advancing.
Missing kit only: ask Noor for the download command (on his printout), run it, then:
```bash
sha256sum ~/ttl_kit.tar
```
Expected exactly `c5374baebdb47a75e1ca2a4ca31a625f735675babb4e974359eefd479e0eb71a`.
Different: download again; never unpack a mismatch. Match only:
```bash
cd ~ && tar -xf ttl_kit.tar && ls ~/ttl_kit
```
Expected: 10 kit files (a tar "unknown extended header keyword" warning is harmless).

## Camera check — BEFORE stage.sh (file listing only, never open images)
The camera writes pictures into its folder continuously; nobody presses save.
```bash
F='/CAMERA/FOLDER/FROM/NOOR'; ls "$F" | wc -l; ls -lt --time-style=full-iso "$F" | head -n 4; sleep 10; ls -lt --time-style=full-iso "$F" | head -n 4
```
Expected: newest entries are FILES ending .jpg/.jpeg/.png directly in that folder,
and the second listing has newer files than the first. Note the interval between
files and set WAIT = 2 x interval, minimum 3 s (staging takes the newest picture,
so waiting WAIT after "ready" guarantees no hands in it). Subfolders, another file
type or no new file in 10 s: tell Noor, do not start stage.sh. Over 5,000 files:
tell Noor each capture may take longer (the script fingerprints the whole folder).
Tell Noor in one line: folder OK, interval, WAIT, live mode = continuous.

## Setup — once only for Tuesday
```bash
printf '%s PENDING setup; first Tuesday start confirmed\n' "$(date -Is)" >> ~/ttl_kit/agent_log.md; bash ~/ttl_kit/start.sh --new-visit >> ~/ttl_kit/setup.log 2>&1; tail -n 20 ~/ttl_kit/setup.log; printf '%s setup result=%s\n' "$(date -Is)" "$(tail -n 1 ~/ttl_kit/setup.log)" >> ~/ttl_kit/agent_log.md
```
Expected: `SETUP OK: preflight passed; context .../operator.env`. Explicit new
visit preserves Monday; record the context shown. Plain start resumes READY=1
regardless of date. Never repeat --new-visit on Tuesday resume. If no writer runs:
```bash
bash ~/ttl_kit/start.sh >> ~/ttl_kit/setup.log 2>&1; tail -n 6 ~/ttl_kit/setup.log
```
Expected on resume: `SETUP OK: resumed ...`.

## Stage — FIFO and persistent holder
If ~/ttl_in already exists, inspect resume state; never replace it blindly.
```bash
mkfifo ~/ttl_in
nohup setsid bash -c 'exec 3<> "$HOME/ttl_in"; exec sleep 10800' </dev/null >> ~/ttl_kit/fifo_holder.log 2>&1 & echo $! > ~/ttl_kit/fifo_holder.pid
kill -0 "$(cat ~/ttl_kit/fifo_holder.pid)" && cat ~/ttl_kit/fifo_holder.pid
nohup setsid bash ~/ttl/stage.sh < ~/ttl_in >> ~/ttl_kit/stage.log 2>&1 & echo $! > ~/ttl_kit/stage.pid; sleep 3; tail -n 6 ~/ttl_kit/stage.log
```
Expected: holder PID, then `Camera folder (ask CSI; before the first save):`.
Holder opens read/write, preventing EOF between answers. Every prompt uses stdin.
Send Noor's confirmed folder:
```bash
printf '%s\n' '/CAMERA/FOLDER/FROM/NOOR' > ~/ttl_in; sleep 3; tail -n 6 ~/ttl_kit/stage.log
```
Expected: `Existing camera images ignored. Ask CSI to save setup 00 now.` and
`setup 00: Enter after CSI saves it, s = skip, r = retake the last setup, q = done:`.
Noor says “setup NN ready” (hands are out): check NN matches the prompt, then ONE
call (WAIT from the camera check). Any note Noor adds goes into the log line:
```bash
sleep WAIT; printf '\n' > ~/ttl_in; sleep 3; tail -n 6 ~/ttl_kit/stage.log; printf '%s setup=NN action=save note=NOTE result=%s\n' "$(date -Is)" "$(tail -n 3 ~/ttl_kit/stage.log | tr '\n' ' ')" >> ~/ttl_kit/agent_log.md
```
Expected: `setup_NN.EXT: WIDTHxHEIGHT; sha256 ...`, `setup NN: staged`, next prompt.
“retake” (hand in view or setup wrong; fixed and hands out again); NN below is the
LAST setup:
```bash
sleep WAIT; printf 'r\n' > ~/ttl_in; sleep 3; tail -n 6 ~/ttl_kit/stage.log; printf '%s setup=NN action=retake result=%s\n' "$(date -Is)" "$(tail -n 3 ~/ttl_kit/stage.log | tr '\n' ' ')" >> ~/ttl_kit/agent_log.md
```
Expected: `setup_NN_r1.EXT: ...`, `setup NN: retake`; earlier attempt retained.
“skip: REASON”: NN is the CURRENT prompt; include Noor's reason in the same call:
```bash
printf 's\n' > ~/ttl_in; sleep 3; tail -n 6 ~/ttl_kit/stage.log; printf '%s setup=NN action=skip reason=REASON result=%s\n' "$(date -Is)" "$(tail -n 2 ~/ttl_kit/stage.log | tr '\n' ' ')" >> ~/ttl_kit/agent_log.md
```
Expected: `setup NN: skipped`, next prompt. “done”:
```bash
printf 'q\n' > ~/ttl_in; sleep 3; tail -n 6 ~/ttl_kit/stage.log; printf '%s stage done result=%s\n' "$(date -Is)" "$(tail -n 1 ~/ttl_kit/stage.log)" >> ~/ttl_kit/agent_log.md
```
Expected: `STAGING OK: N setups staged; skips and retakes logged; no inference`.
`Image still changing`: send the same call again once. No new image/REFUSED:
report it; setup has not advanced. If output
is still pending after 3 seconds, wait/tail; do not resend the answer blindly.
Reply using the same observed line: setup, filename, size in pixels, timestamp,
status. stage.log has dimensions/hash, not bytes; do not add a stat tool call.
The appended result contains filename/dimensions/hash/status/next prompt and time.
Spark verified this combined call for save/retake/skip/done through STAGING OK.
Staging ends by **1:20 pm**. Plan omissions in priority 18, 15, 17, 16; decide
this set early, then send s at each selected numeric prompt. Never skip 00, 01,
06, 19–21. Reserve time for 19–21; warn Noor early of any coverage/deadline conflict.
After STAGING OK, no writer, clean ONLY holder/FIFO in one call:
```bash
flock -n ~/ttl/operator.lock true && kill -TERM "$(cat ~/ttl_kit/fifo_holder.pid)" && unlink ~/ttl_in
```
Expected: exit 0. Retain logs/PID files; last completed result is STAGING OK.

## Seal
```bash
printf '%s PENDING seal\n' "$(date -Is)" >> ~/ttl_kit/agent_log.md; bash ~/ttl/seal.sh >> ~/ttl_kit/seal.log 2>&1; tail -n 20 ~/ttl_kit/seal.log; printf '%s seal result=%s\n' "$(date -Is)" "$(tail -n 1 ~/ttl_kit/seal.log)" >> ~/ttl_kit/agent_log.md
```
Expected: RECONCILED, FINAL, then `SEALED USB OK: ...; reset fixture before live`.
`SEALED DISK OK: USB FAILED ...`: tell Noor; keep disk seal, do not reseal. Zero
staged gives WARNING. Seal verifies capture names/order/hashes/exact membership,
archived capture bytes and archive-copy hashes. Never put logs in staged/.

## Live — fresh FIFO/holder; launch WITHOUT &
Ask Noor to confirm the fixture is empty and the robot is ready; never invent it.
```bash
mkfifo ~/ttl_in
nohup setsid bash -c 'exec 3<> "$HOME/ttl_in"; exec sleep 10800' </dev/null >> ~/ttl_kit/fifo_holder.log 2>&1 & echo $! > ~/ttl_kit/fifo_holder.pid
kill -0 "$(cat ~/ttl_kit/fifo_holder.pid)"
printf '%s PENDING live launch\n' "$(date -Is)" >> ~/ttl_kit/agent_log.md; setsid -f nohup bash ~/ttl/live.sh < ~/ttl_in >> ~/ttl_kit/live_console.log 2>&1; sleep 3; pgrep -f '[m]onday_ops.py live$' > ~/ttl_kit/live.pid; tail -n 6 ~/ttl_kit/live_console.log
```
Expected: `Live camera folder or approved RTSP/HTTP(S) stream URL:`. Send folder:
```bash
printf '%s\n' '/CAMERA/FOLDER/FROM/NOOR' > ~/ttl_in; sleep 3; tail -n 6 ~/ttl_kit/live_console.log
```
Same camera source allowed; never staged/sealed copies. Expected gate prompt:
`Enter = n (continuous camera feed); y only for stills CSI saves on request. [y/n]:`.
For continuous:
```bash
printf '\n' > ~/ttl_in; sleep 3; tail -n 6 ~/ttl_kit/live_console.log
```
For requested stills instead:
```bash
printf 'y\n' > ~/ttl_in; sleep 3; tail -n 6 ~/ttl_kit/live_console.log
```
Expected: `CSI reset the fixture and cleared staged/buffered frames? [y/n]:`.
ONLY after Noor confirms reset:
```bash
printf 'y\n' > ~/ttl_in; sleep 3; tail -n 10 ~/ttl_kit/live_console.log; printf '%s live launch gate=CONFIRMED_MODE reset=confirmed result=%s\n' "$(date -Is)" "$(tail -n 5 ~/ttl_kit/live_console.log | tr '\n' ' ')" >> ~/ttl_kit/agent_log.md
```
Replace CONFIRMED_MODE with each/events. Expected: LIVE OUT path and
`watching ... new images only`, then grids. Log result confirms completed launch.
`ok` PASS; `DEF` DEFECT; `MISS` MISSED; `??` UNSURE; `.` WAITING.
During live, Noor reports what the team sees ("G13 ?? = cap tilted"). One call each:
```bash
printf '%s OBSERVATION live: %s\n' "$(date -Is)" 'NOOR_TEXT' >> ~/ttl_kit/agent_log.md; tail -n 12 ~/ttl_kit/live_console.log
```
On "show grid": `tail -n 12 ~/ttl_kit/live_console.log` and give Noor a one-line summary of non-ok holes.
Prompts/staging lines flush explicitly; live child uses -u and wrapper flushes.
Python -I ignores PYTHONUNBUFFERED. Spark proof: stage results visible while
alive; clean INT stop retained 600 prediction/21 trial rows and finalized metadata.
On Noor's “stop live”, run ONE standalone call:
```bash
printf '%s PENDING live stop INT\n' "$(date -Is)" >> ~/ttl_kit/agent_log.md; pkill -INT -f '[l]ive_inspect.py'; sleep 3; tail -n 8 ~/ttl_kit/live_console.log; printf '%s live stop result=%s\n' "$(date -Is)" "$(tail -n 2 ~/ttl_kit/live_console.log | tr '\n' ' ')" >> ~/ttl_kit/agent_log.md
```
Expected: frame/event summary and `LIVE OK: stopped cleanly; evidence ...; export every visit`.
WITHOUT & preserves SIGINT handling; setsid -f detaches/returns. Never substitute
an asynchronous live launch or a Ctrl-C byte. If still running after **30 seconds**:
```bash
printf '%s PENDING live stop TERM fallback\n' "$(date -Is)" >> ~/ttl_kit/agent_log.md; pkill -TERM -f '[l]ive_inspect.py'; sleep 3; tail -n 15 ~/ttl_kit/live_console.log; printf '%s live fallback result=%s\n' "$(date -Is)" "$(tail -n 2 ~/ttl_kit/live_console.log | tr '\n' ' ')" >> ~/ttl_kit/agent_log.md
```
Fallback may give `STOP: live: live runtime failed (-15); preserve ...`, incomplete
metadata/in-flight frame. Report verbatim and wait for Noor before export.
After wrapper exits, clean holder/FIFO and verify metadata in one call:
```bash
flock -n ~/ttl/operator.lock true && kill -TERM "$(cat ~/ttl_kit/fifo_holder.pid)" && unlink ~/ttl_in && bash -c 'source ~/ttl/operator.env; grep -E "ended_utc|stop_reason|frames|events" "$OUT/session.json"; tail -n 1 "$DAY/records/live_status.txt"'
```
After LIVE OK expect end time, stop_reason ctrl-c, counts and live exit=0.

## Export and handoff
```bash
printf '%s PENDING export\n' "$(date -Is)" >> ~/ttl_kit/agent_log.md; bash ~/ttl/export.sh >> ~/ttl_kit/export.log 2>&1; tail -n 10 ~/ttl_kit/export.log; printf '%s export result=%s\n' "$(date -Is)" "$(tail -n 1 ~/ttl_kit/export.log)" >> ~/ttl_kit/agent_log.md
```
Expected: `RESULTS USB OK: N visits verified at ...; keep GB10 files until Mac verification`.
Retain destination from that result. Export hashes all visit_* archives locally and
at destination. Kit-root logs are safe, outside visit archives; Monday visits stay
separate. Tell Noor: “Done. Ask Shamar to copy the whole ~/ttl_kit/ folder for us
and keep the GB10 files until we verify the copy.” Finish before 2 pm.

## Resume after usage limit
Read this pasted guide and agent_log.md; tail the applicable script log. Last
confirmed successful result is the last completed step; PENDING alone is not success.
Check saved PID with ps -p PID. Do not recreate an active FIFO/holder or resend an
uncertain answer. Inspect metadata only:
```bash
bash -c 'source ~/ttl/operator.env; printf "DAY=%s\nSEALED=%s\nOUT=%s\n" "$DAY" "${SEALED:-}" "${OUT:-}"; tail -n 5 "$DAY/records/staging_log.csv"; cat "$DAY/records/live_status.txt"'
```
Never repeat --new-visit/reseal/duplicate live or export with a writer running.
If stage died, confirm no prior writer, then repeat its FIFO launch; CAMERA/next
setup persist. If uncertain, tell Noor the last confirmed result and wait.
