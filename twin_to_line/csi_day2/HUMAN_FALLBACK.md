# CSI Tuesday · human fallback if both agents are down
Use the normal foreground commands below; no FIFO. Read the pasted agent guide
and `cat ~/ttl_kit/agent_log.md` first → last completed step, or missing on first start.
Tell the supervisor if human operation is needed. Preserve Monday's trial pictures;
never use them. No code/settings edits, image opening/scoring, git or deletion.
Never touch camera, robot, PLC or network. Every STOP: report verbatim and wait.

1. `ls -ld ~/ttl_kit ~/ttl` → kit directory; ttl may not exist before first setup.
   Missing kit: ask Noor for the download command.
2. FIRST Tuesday start only:
   `bash ~/ttl_kit/start.sh --new-visit 2>&1 | tee -i -a ~/ttl_kit/setup.log`
   → `SETUP OK: preflight passed; context .../operator.env`.
   Already-started Tuesday, no active writer: `bash ~/ttl_kit/start.sh`
   → `SETUP OK: resumed ...`. Never repeat --new-visit on resume; old visits stay.
3. `bash ~/ttl/stage.sh 2>&1 | tee -i -a ~/ttl_kit/stage.log`
   → camera-folder prompt on first use. Type CSI's confirmed path, Enter.
   → `Existing camera images ignored. Ask CSI to save setup 00 now.`
   After each fresh save: Enter → filename/dimensions/hash, `setup NN: staged`.
   Fresh save for LAST setup: r, Enter → `setup NN: retake`; prior copy retained.
   Skip CURRENT setup: log reason, s, Enter → `setup NN: skipped`.
   Finish by **1:20 pm**. Plan omissions 18, 15, 17, 16 in that priority; send
   selected skips as their numeric prompts arrive. Never skip 00, 01, 06, 19–21.
   Reserve time for 19–21; report coverage/deadline conflict early.
   q, Enter → `STAGING OK: N setups staged; skips and retakes logged; no inference`.
4. `bash ~/ttl/seal.sh 2>&1 | tee -i -a ~/ttl_kit/seal.log`
   → `RECONCILED: ...`, `FINAL: ...`, `SEALED USB OK: ...; reset fixture before live`.
   `SEALED DISK OK: USB FAILED ...` → tell Noor, preserve disk seal, do not reseal.
5. `bash ~/ttl/live.sh 2>&1 | tee -i -a ~/ttl_kit/live_console.log`
   → folder prompt: type CSI-confirmed source, never staged/sealed folders.
   → gate: Enter for continuous; y, Enter only for requested stills.
   → reset: y, Enter ONLY after CSI confirms reset and cleared buffered frames.
   → `LIVE OUT: ...`, `watching ... new images only`, then grid on new frames.
   On stop instruction, have CSI finish the current save and pause delivery.
   In a SECOND terminal run `pkill -INT -f live_inspect.py`.
   → summary, then `LIVE OK: stopped cleanly; evidence ...; export every visit`.
   If still running after 30 seconds: `pkill -TERM -f live_inspect.py`.
   → may give `STOP: live: live runtime failed (-15); preserve ...` instead of OK;
   report it verbatim and wait for Noor's acknowledgement before export. TERM
   may leave final metadata/in-flight frame incomplete; never claim LIVE OK.
6. After LIVE OK and all writers stop (or Noor acknowledges fallback STOP):
   `bash ~/ttl/export.sh 2>&1 | tee -i -a ~/ttl_kit/export.log`
   → `RESULTS USB OK: N visits verified at ...; keep GB10 files until Mac verification`.
   Say: “CSI, please copy the entire ~/ttl_kit/ folder for us, including the sealed
   folders, visit_export folders and session logs; please keep the GB10 files until
   we verify the copy on our Mac.” Finish before 2 pm; retain everything.

Before each action and after its observed result, append state (replace TEXT):
`printf '%s %s\n' "$(date -Is)" 'TEXT' >> ~/ttl_kit/agent_log.md` → exit 0.
TEXT includes PENDING or COMPLETED, last/next step/setup, file/bytes/time, skip
reason, visit/seal/live/export paths, STOP and acknowledgement. Metadata only:
`bash -c 'source ~/ttl/operator.env; stat -c "%n %s bytes %y" "$STAGED/EXACT_FILENAME"; tail -n 2 "$DAY/records/staging_log.csv"'`
→ filename, size, time and CSV row. No image viewing. Resume from actual state;
never repeat an uncertain action. Export success verifies bytes, not live completeness.
