# CSI Tuesday · human terminal fallback
Use only if both agents are down, with the supervisor's agreement to human operation.
Read AGENT_GUIDE.md first. Preserve Monday trial pictures; never use them. Delete
nothing, edit nothing, never view/score staged images or touch camera/robot/PLC/network.
Every STOP: read it verbatim to Noor/supervisor and wait. Do not auto-retry.
Have Noor place these three docs in `~/ttl_kit/` before the session; no repo needed.

1. Check files: `ls -ld ~/ttl_kit ~/ttl` → directories (`ttl` may not exist yet).
   Missing kit: ask Noor for the download command. Read
   `cat ~/ttl_kit/csi_day2_session.log` → last state, or missing on first start.
2. FIRST Tuesday start only:
   `bash ~/ttl_kit/start.sh --new-visit 2>&1 | tee -i -a ~/ttl_kit/csi_day2_terminal.log`
   → `SETUP OK: preflight passed; context ...`.
   If Tuesday already started successfully and no writer runs:
   `bash ~/ttl_kit/start.sh` → `SETUP OK: resumed ...`. Never repeat `--new-visit`
   on resume. Old visits are retained; plain start otherwise resumes Monday too.
3. `bash ~/ttl/stage.sh 2>&1 | tee -i -a ~/ttl_kit/csi_day2_terminal.log`
   → camera-folder question on first use. Type the CSI-confirmed path, Enter.
   → `Existing camera images ignored ... setup 00 now`.
   After CSI saves each setup, Enter → filename/dimensions/hash and `setup NN: staged`.
   New save for LAST setup: `r`, Enter → `setup NN: retake` (prior copy kept).
   Skip current setup: log reason, `s`, Enter → `setup NN: skipped`.
   Never skip 00, 01, 06, 19–21. Plan omissions 18, 15, 17, 16 in that priority;
   send skips in numeric order as prompts arrive. Finish staging by **1:20 pm**;
   reserve time for 19–21, report any coverage/time conflict early.
   `q`, Enter → `STAGING OK: N setups staged; skips and retakes logged; no inference`.
4. `bash ~/ttl/seal.sh 2>&1 | tee -i -a ~/ttl_kit/csi_day2_terminal.log`
   → `RECONCILED ...`, then `SEALED USB OK ...; reset fixture before live`.
   `SEALED DISK OK: USB FAILED ...` → report backup failure; keep disk seal, do not reseal.
5. `bash ~/ttl/live.sh 2>&1 | tee -i -a ~/ttl_kit/csi_day2_terminal.log`
   → folder prompt: type confirmed camera source, never staged/sealed folders.
   → gate prompt: Enter for continuous; `y`, Enter only for requested stills.
   → reset prompt: `y`, Enter ONLY after CSI confirms reset and cleared buffered frames.
   → `LIVE OUT: ...`, `watching ... new images only`, then grid.
   Stop with **one Ctrl-C** → summary and `LIVE OK: stopped cleanly ...`.
   Still running after 30 seconds: second terminal `pkill -TERM -f live_inspect.py`
   → may give STOP in first terminal; report it and wait, never claim LIVE OK.
6. After writers stop: `bash ~/ttl/export.sh 2>&1 | tee -i -a ~/ttl_kit/csi_day2_terminal.log`
   → `RESULTS USB OK: N visits verified at ...; keep GB10 files until Mac verification`.
   Say: “CSI, please copy the entire ~/ttl_kit/ folder for us, including the sealed
   folders, visit_export folders and session logs; please keep the GB10 files until
   we verify the copy on our Mac.” Finish before 2 pm; delete nothing.

After each action, append state (replace TEXT; do not put secrets in logs):
`printf '%s %s\n' "$(date -Is)" 'TEXT' >> ~/ttl_kit/csi_day2_session.log`
→ exit 0. TEXT: phase, OK/STOP, current visit, pending prompt, last/next setup,
filename, bytes, capture time, retake/skip reason, seal/live/export paths.
Capture metadata only: `bash -c 'source ~/ttl/operator.env; stat -c "%n %s bytes %y" "$STAGED/EXACT_FILENAME"; tail -n 2 "$DAY/records/staging_log.csv"'`
→ filename, size, time and CSV row. No image opening. Resume from log and actual
prompt; never resend an uncertain action. Kit-root logs do not break verification.
