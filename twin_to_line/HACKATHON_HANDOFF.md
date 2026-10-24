# Twin-to-Line — Handoff (read first)
Updated: 2026-10-24 (Sat) 2:30 pm CT. Two identical copies: `twin_to_line/HACKATHON_HANDOFF.md` (repo, canonical) and
the Claude Project doc `claude/hackathon-handoff.md`. Decisions: repo `twin_to_line/DECISIONS.md` (D001–D041,
canonical). **The Project copy of DECISIONS.md stops at D038**; D039–D041 are summarised below.
`claude/next-session-plan.md` is superseded except §2 (requirements) and §8 (message style).
New session: work from `~/code/ignite`, run `git status -sb` first, then go to **NEXT ACTION**. Do not reopen settled
decisions without new evidence. Replace stale facts; no diary. **Never put passwords, tokens, invitation links, team
codes or auth URLs in this file or in memory** (that includes the CSI VPN password and the YAAD team code).

## Standing rules
- Branches stay **local and unpushed** until Noor authorizes a push. Never self-merge. Don't change `main` of the
  Ignite repo. Don't modify the frozen gate.
- Never touch Noor's unstaged edit in `twin_to_line/dose_sets.py`. Stage files by name, never `git add -A` / `-u`.
- No AI/assistant attribution anywhere (commits, PRs, docs). Tool credits go in the final submission's references.
- Never commit raw data, the twin, the labeler, renders, results, CSI images or model weights. Never open runs 4–5.
- Ask Noor before spending money. Draft messages for Noor; he sends them. Short, plain language for CSI and organizers.
- Agents: **Claude Code usage is limited until Wed** (main tasks only); **Codex** does the rest and reviews. One agent
  per branch: Codex works in worktree `../ignite-fix` on a new branch from `feat/live-runtime`; Noor fast-forwards it
  (`git merge --ff-only <branch>`, then the full tests). The implementer never reviews its own work. Claude in the
  Project "hackathon" reviews outputs and writes the prompts. Noor is a Code Crusaders member, not captain.

## State (Sat Oct 24, 2:30 pm)
Branch `feat/live-runtime` @ `1b49565` (local only). `git status -sb` → `## feat/live-runtime` +
` M twin_to_line/dose_sets.py`. Full tests: **247 passed**; one test needs the gitignored
`results/rc_1024/inputs/staged_3/` (3 DEV stills, copied from `../ignite-fix` on Oct 24).

### Arms (all frozen Sun 9 pm; each scored once on the sealed set with `score_sealed.py`, labels locked by sha256 first)
DEV G3 = runs 1–3, 20 placement events, 32 defect / 551 non-defect observations, LIVE_FILL, no tracker.

| Arm | What | DEV G3 |
|---|---|---|
| A | Synthetic-only ResNet-18 v1; D038 t 0.50224 / pre-registered t 0.99968 | 32/32, 483 FP (NO-GO) / 15/32, 308 FP |
| B | Rule, thresholds from synthetic validation (`rule_thresholds_synth.json`: 0.032 / 0.94 / 2.0) | 19/32, 132 FP (NO-GO) |
| B′ | Rule, DEV-fit (`rule_thresholds.json`: 0.081 / 0.22 / 4.4) — **Monday's live inspector** | 31/32, 14 FP (in-sample) |
| B″ | Rule, DEV good-only (D041, `rule_thresholds_dev_good_only.json`: 0.081 / 0.22 / 4.8) | 28/32, 8 FP (NO-GO, in-sample) |

- B″ differs from B′ only in the offset threshold (4.8 vs 4.4 px). Its 3 extra misses are **one physical crooked cap**
  (G13, run 1, events 14/16/18); its missing-cap miss (G07) is the same one B′ misses.
- The rule arms never output `crooked_cap` / `missing_cap` exactly: they catch them as UNSURE ("??" on the console =
  check by eye). B′ is DEV-fit: disclose that everywhere unless organizers approve (question under CSI and organizers).
- v2 (D040) stopped at its check render (3 tries failed; the rendered hole floor is flatter than the real one): no v2,
  arm A = v1. Finding for the write-up: the main sim-to-real gap for empty holes is the hole floor's lit contrast.
- Gate gate-v2.0.2: PASS WITH WORK (D037). Training set `train_v1`: 568 scenes, 17,040 crops. Details: repo reports.

### Live runtime (`live_inspect.py`) and package
- Sizes: 640×480 as is; 1920×1080 warped into the DEV 640 grid (D039, `frame_warp_1080.json` sha256 `9837b298…`);
  anything else `wrong_size` (the 320×240 stream is not supported).
- Sources: `folder:` (ignores files present at start; re-reads a rewritten name; time = filename time when it increases,
  else arrival time; an unacceptable version is logged `unreadable` once after 3 s and never scored; JPEG zero padding
  accepted), `stream:` RTSP/HTTP with open/read timeouts and a 10 s watchdog (**untested on a real camera**; raw
  devices refused), `replay:`. Staged stills: `--gate each`, never `--track`.
- Each run writes a new `--out`: `predictions.csv` (30 rows per event, flushed before the overlay), `trial_log.csv`,
  `session.json`, `overlay_latest.png` (atomic replace), `overlays/`, `view.html`. The console grid per event is the
  primary operator view.
- Package: `gb10/build_package.py` → `ttl_gb10_<sha>.tar` (~243 MB): git archive, bundled CPython 3.11 aarch64,
  wheels cp310/311/312, `bootstrap.sh` (refuses to run over existing results). The RC (`e597c4b`) passed on the YAAD
  Spark Oct 24: bootstrap 4.9 s → PREFLIGHT OK; replays at 640 and 1080: 0/540 holes differ from the Mac; folder mode
  1/s and burst OK; 3 staged DEV stills 0/60 differences; mp4 stream 3/540 differ (aarch64 video decode), so the freeze
  record uses the **Spark** outputs as the stream reference. The system-Python fallback passed in Ubuntu 22.04/24.04
  arm64 containers.
- **Not verified:** real 1080 frames with caps; CSI's own GB10; how CSI's camera delivers frames.
- Freeze draft: `FREEZE_2026-10-25.md` + `FREEZE.sha256` (83 OK, 0 mismatches on Oct 24); commit, tag and package
  fields are Sunday placeholders. `score_sealed.py` scores a folder of staged stills once per operating point (A ×2,
  B, B′, B″) into separate outputs and refuses to run unless the labels file matches a given sha256.
- `LIVE_RUNBOOK.md` §7 (Monday) is correct but too long to type at CSI's keyboard: it is being turned into 5 scripts
  (NEXT ACTION 1).

### Team
- **Vang:** "M8" = an operator UI around our checker (reads our output folder; standard library only; 127.0.0.1 only).
  Code Crusaders repo PR #2 (`feat/m8-operator-ui`). Noor sent a review requesting changes: remove `samples/`
  (dataset-derived images) and all `__pycache__` from the branch so they never reach main; re-fetch the overlay on
  every poll (it can show the previous event's picture); "STALE" must not hide an action-required result, default
  `--stale-after` 30. Vang accepted and is fixing it. Joint test on the Spark **Sun 3 pm**; if it passes, a pinned copy
  goes into the package with its commit and sha256 in the freeze record, otherwise Monday uses the console grid.
  On the day: start the checker first, then the UI with the folder path the checker prints; restart the UI whenever
  the checker restarts into a new folder.
- Code Crusaders repo `main` already contains PR #1 (`9040ded`).
- **Tony** (Anthony Harvey): attends Monday; staging and logging. Still needs the updated staging doc and his brief.

### CSI and organizers
- CSI session **Mon Oct 26, 12:30–2 pm, confirmed** (Noor + Tony). Camera streams 320×240 and 1920×1080; our software
  runs on CSI's GB10; no setup time before the slot. 1080 empty-fixture still: `results/csi_1080/csi_1080_empty_G.jpg`.
- **Shamar has not answered** the Oct 23 email (OS/Python/internet, USB copy, how frames reach the GB10, staged pictures
  at 1080, a screen, a full-fixture 1080 picture). Noor will not send another request: ask the supervisor in person
  at 12:15 Monday. Never confirmed: vials and caps for staging (~32 + 34), side photos, filming, the backup slot.
- **Organizer question** (may the rule's 3 thresholds be set on runs 1–3, or on good images only?), drafted Oct 24 for
  `#dataset-1-vial-inspection` tagging Derek Riley. **Posting status unknown: check with Noor.**
- Organizer 1920×1080 release: not on the Mac (development and evaluation only).

### Compute
- **Brev: done.** `cc-isaac-gate` is deleted. `cc-isaac-gate-2` (Mumbai, L40S, $4.38/hr) is stopped; every pack was
  copied back and checksum-verified. Delete it after Nov 1, or now if the console shows a stopped instance still
  billing. No more renders planned. Org `UWM/MSOE-Team-25`, $500 shared; create instances from the console.
- **YAAD Spark:** `ssh yt-codecrusaders6286@172.30.102.94` with the FortiClient VPN connected (the Oct 24 timeouts were
  the VPN being off). Bare container (no python3, tmux, scp or rsync): copy with `tar -cf - … | ssh … 'tar -xf - -C …'`,
  run long jobs with `setsid nohup … &`. Keep `/workspace/ttl_rehearsal_1023` and `/workspace/ttl_rc_1024` for
  Sunday; delete them after the event.

## NEXT ACTION
1. **Sat 6 pm, Codex (`freeze-prep-3`):** turn §7 into 5 scripts copied to the USB root (`start.sh`, `stage.sh`,
   `seal.sh`, `live.sh`, `export.sh`). Each ends in one OK/STOP line; `stage.sh` copies the newest camera image to
   `setup_NN.<ext>` and writes the log row itself; `seal.sh` counts the setups actually staged (skips logged); nothing
   deletes. Rewrite §7 to these commands with expected outputs; end-to-end test in an arm64 container; record
   `results/rc_1024/inputs/staged_3` and its sha256 in the freeze manifest. Then Noor fast-forwards and runs the tests.
2. **Sat night, Noor:** update the staging doc (12:30 confirmed; stills at 1920×1080; GB10 setup inside the slot; cut
   order: optional live defects, filming, live duration; never labels or backups) and send it to Tony with his role
   and what "??" means. Post the organizer question if it isn't posted.
3. **Sun:** morning: review Vang's fixed PR · 3 pm: joint Spark test (checker + UI) · ~5 pm: freeze build (final commit
   + local tag, package rebuilt from it, `FREEZE` completed, Spark dry run) · **6–8 pm: Noor types the 5 commands
   himself on the Spark from the real USB stick, timed, no agent** · **9 pm: freeze**.
4. **Mon:** arrive 12:15; ask how frames reach the GB10, USB, screen. Bring an exFAT USB (package `.tar` + `.sha256`,
   runbook PDF, empty results folder, several GB free), laptop + charger, printed staging list + numbering map, label
   sheet, paper log, phone, pen. Staged images are copied and sealed, never scored on the day; the live run starts
   after the seal, in a fresh folder.
5. **Mon evening:** adjudicate labels without predictions (two people), lock them by sha256, then run
   `score_sealed.py` once for all arms.

## Milestones
Oct 25 freeze by 9 pm · **Mon Oct 26 12:30–2 pm CSI** · Oct 26–27 adjudicate, score every arm once · Oct 28–29 backup
slot if needed (unconfirmed), error analysis, business case · Oct 30 evaluation write-up · Oct 31 video, one-page
summary, docs · **Sun Nov 1 11:59 pm CT final submission** (~5-min video, one-page summary, repo, references) ·
Thu Nov 5 5–8 pm finals, MSOE Diercks Hall.

## Write-up headline (draft; every number in-sample until the sealed set)
From the digital twin alone: all 30 positions located with no per-hole teaching (640: 30/30, ~1.68 px; 1080 via D039),
and missed placements told apart from waiting holes with no PLC signal. Synthetic-only appearance did not transfer
(ResNet v1 and synthetic rule thresholds fail G3); the main gap was traced to the hole floor's contrast. A 3-number
rule needs real images; on DEV, setting them from good images only lands within one crooked cap of defect-tuned
thresholds; the sealed Oct 26 set tests B″ against B′ out of sample. Distinctiveness: a novel integration, not a new
algorithm (cap checks are commodity, D002). Likely pushback: DEV-fit live rule, "detection" via UNSURE, no measured
changeover time. Verify any prior-art link before citing it.

## Decisions since D038 (full text in repo DECISIONS.md)
- **D039:** 1920×1080 frames are warped into the DEV 640×480 grid; the unchanged 640 path runs; other sizes `wrong_size`.
- **D040:** one ResNet-18 v2 attempt (new hole-floor material + augmentation), pre-declared; its check render failed,
  so no v2; arm A stays v1.
- **D041:** arm B″, rule thresholds fitted once on good DEV observations only; one G3; scored once on the sealed set;
  live stays B′; organizer question still open.

## Real-data boundary (D011, D025, D033, D036)
Runs 1–3: development/reference only. Runs 4–5: holdout, never for training, thresholds, prompts, recipe or model
choice. Organizer release: development + evaluation only. **Oct 26 staged images: sealed final test; labels adjudicated
before any model runs on them; scored once after the freeze; never training or tuning.** Unit = placement event or
arrangement-position, never a frame. `labels/dev_labels.csv` = draft labels.

## Organizer facts
- Derek Riley (Oct 16): direction aligned, not exclusive; confirm training-data rules first (D025); crooked cap = any cap
  not fully seated, a defined refinement allowed (ours: ≥20°, D036); a supervised live run counts as the hardware demo
  if our implementation runs on real hardware (D029); no PLC/robot/placement-count data (D019, D021); CSI numbering
  differs, G01–G30 is our twin convention (D022).
- CSI Data Collection guide: requests in `#data-and-testbed-requests` tagging Shamar (email fallback); extra collection
  not guaranteed. May not: move/re-aim/reconfigure a camera; change PLC, robot, FactoryTalk, safety, recipe or program;
  connect hardware to the plant network without approval. Before a physical test: freeze code and model, document
  inputs/outputs, start/stop procedure, metrics, failure handling tested on saved data. Trial log: every attempt.
- Dataset 1 page: avoid training on the real images; keep your own test set; report per class, misses and false alarms
  separately, the false-alarm rate at the threshold that catches every known defect, full fixture scenes; missed
  defects matter more than false alarms. `damaged_vial` out of scope for v1, still counted (D026). The NVIDIA labeler
  is a 1920×1080 manual review tool, no predictions (D027).

## Team and contacts
- Code Crusaders: Zara, Matthew, Trevor, Vang, Tony Harvey, Jamie Gill, Noor ("Jatt" on Discord).
- CSI/testbed: Shamar Webster (`#data-and-testbed-requests`, webste63@uwm.edu) · dataset: Brett Storoe, Derek Riley,
  Tanner Cellio (`#dataset-1-vial-inspection`) · compute: Dan Schneider (`#compute-questions`) · workspace: Brett
  Storoe (`#hackathon-general`) · program: Tom Hershberger.

## Paths
- Ignite repo (local clone `~/code/ignite`, project dir `twin_to_line/`, venv `~/code/ignite/.venv`, Python 3.11):
  https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-new-team-11-b020278d03
  · `main` @ `2a935de` unchanged · work branch `feat/live-runtime` (local only). Codex worktree `~/code/ignite-fix`.
- Code Crusaders repo: ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-new-team-10-f7d5357a4a.
- Data (not in git): twin `~/Downloads/csi_digital_twin/csi_digital_twin_v1/`; real `~/Downloads/csi-vial-fill-dataset/cap_seal/`;
  labeler `~/Downloads/vial-labeler-participants-20261014.zip`.
- Docs: staging list https://claude.ai/artifact/P1WRYUf2vJb2kUR2XPHTKh (private; update before sharing) · proposal
  https://claude.ai/code/artifact/7e742133-103c-483e-8409-8aca3b5e131e · repo reports `RC_2026-10-24.md`,
  `ARM_B_DOUBLE_PRIME_RESULT.md`, `FREEZE_2026-10-25.md`, `LIVE_RUNBOOK.md`, `GATE_v2.0.2.md`.

## Known-working commands (in `~/code/ignite`)
```bash
git status -sb                         # expect: ## feat/live-runtime  +  " M twin_to_line/dose_sets.py"
IGNITE_TWIN=~/Downloads/csi_digital_twin/csi_digital_twin_v1 .venv/bin/python -m pytest twin_to_line/tests -q   # expect: 247 passed
git merge --ff-only <codex-branch>     # after Codex reports clean; if it refuses, stop and ask
cd twin_to_line && shasum -a 256 -c FREEZE.sha256    # expect: all OK
../.venv/bin/python gb10/build_package.py             # package from HEAD -> results/gb10_pkg/
ssh yt-codecrusaders6286@172.30.102.94                # YAAD Spark; FortiClient VPN first
```

## Known risks
- How frames reach our program on CSI's GB10 is unknown; covered by folder / rewritten file / stream / stills-on-request.
- CSI's GB10 itself is untested (user rights, USB mount, screen). No real 1080 frame with caps yet.
- B′ is DEV-fit; organizer ruling pending. Every model arm failed G3.
- Staging time is unmeasured (~2 min per setup needed). Cut order: optional live defects, filming, live duration.
- The rule flags crooked caps as "??" (UNSURE), never as "crooked": brief Tony and explain it to the supervisor.
