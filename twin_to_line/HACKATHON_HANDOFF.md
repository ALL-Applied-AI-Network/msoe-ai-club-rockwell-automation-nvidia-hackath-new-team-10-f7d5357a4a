# Twin-to-Line — Handoff (read first)
Updated: 2026-10-24 (Sat) 3:00 pm CT. **All work now lives in the Code Crusaders team repo**: local clone
`~/code/crusaders`, branch `feat/twin-to-line` (pushed; PR to `main` awaiting Vang's review). Two identical copies of
this file: `twin_to_line/HACKATHON_HANDOFF.md` in that repo (canonical) and the Claude Project doc
`claude/hackathon-handoff.md`. Decisions: repo `twin_to_line/DECISIONS.md` (D001–D041, canonical; D042 pending).
**The Project copy of DECISIONS.md stops at D038**; D039–D041 are summarised below. `claude/next-session-plan.md` is
superseded except §2 (requirements) and §8 (message style). The Ignite repo (`~/code/ignite`) is the read-only archive
holding the full commit history.
New session: work from `~/code/crusaders`, run `git status -sb` first, then go to **NEXT ACTION**. Do not reopen settled
decisions without new evidence. Replace stale facts; no diary. **Never put passwords, tokens, invitation links, team
codes or auth URLs in this file or in memory** (that includes the CSI VPN password and the YAAD team code).

## Standing rules
- Work on `feat/twin-to-line` in `~/code/crusaders`. Push only with Noor's OK. `main` changes only through a PR that a
  teammate reviews; never self-merge. Don't modify the frozen gate.
- Stage files by name, never `git add -A` / `-u`. Always run tests with the path: `pytest twin_to_line/tests`. A bare
  `pytest` from the repo root fails (ignite_core/tests and twin_to_line/tests share file names).
- No AI/assistant attribution anywhere (commits, PRs, docs). Tool credits go in the final submission's references.
- Never commit raw data, the twin, the labeler, renders, results, CSI images or model weights. Never open runs 4–5.
  New files must not use ignored names (`outputs/`, `models/`, `results*/`, `*.zip`, `*.tar.gz` match at any depth).
- Ask Noor before spending money. Draft messages for Noor; he sends them. Short, plain language for CSI and organizers.
- Agents: **Claude Code usage is limited until Wed** (main tasks only); **Codex** does the rest and reviews. Only one
  agent works at a time, on its own new branch from `feat/twin-to-line` in `~/code/crusaders`; Noor fast-forwards it
  (`git switch feat/twin-to-line && git merge --ff-only <branch>`, then the full tests). The implementer never
  reviews its own work. Claude in the Project "hackathon" reviews outputs and writes the prompts. Noor is a Code
  Crusaders member, not captain.
- Keep `~/code/ignite` until after Nov 1: `~/code/crusaders/.venv` is a symlink to `~/code/ignite/.venv`.

## State (Sat Oct 24, 3:00 pm)
`~/code/crusaders`, branch `feat/twin-to-line` @ `8402039` (+ this handoff). `twin_to_line/` is byte-identical to
Ignite `feat/live-runtime` @ `1a0601e` (folder hash `7a59a199`). Root `.gitignore` = Ignite's rules + `.venv`.
`twin_to_line/results/` (4 GB, gitignored) was copied from the Ignite clone and checked identical (21,948 files;
sha256 of the model, threshold files, 1080 still and staged_3 match). Full tests here: **247 passed**.
Move check by Claude Code (Oct 24): no runtime code assumes the Ignite repo; a package built here has the same 42
files as the Spark-tested RC, byte-identical runtime files, `ARCHIVE_COMMIT` = this repo's commit, Mac preflight OK; it
also carries `ignite_core/` and the README, which bootstrap never imports. live_inspect.py and score_sealed.py always
load twin_to_line's own modules.

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
- `score_sealed.py` keeps its once-only lock ledger in the clone's own `results/`: run Monday evening's scoring only
  from `~/code/crusaders`.

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
- Package: `gb10/build_package.py` → `ttl_gb10_<sha>.tar` (~243 MB): git archive of the repo root, bundled CPython
  3.11 aarch64, wheels cp310/311/312, `bootstrap.sh` (refuses to run over existing results). The RC (Ignite `e597c4b`)
  passed on the YAAD Spark Oct 24: bootstrap 4.9 s → PREFLIGHT OK; replays at 640 and 1080: 0/540 holes differ from the
  Mac; folder mode 1/s and burst OK; 3 staged DEV stills 0/60 differences; mp4 stream 3/540 differ (aarch64 video
  decode), so the freeze record uses the **Spark** outputs as the stream reference. The system-Python fallback passed
  in Ubuntu 22.04/24.04 arm64 containers.
- **Not verified:** real 1080 frames with caps; CSI's own GB10; how CSI's camera delivers frames.
- Freeze draft: `FREEZE_2026-10-25.md` + `FREEZE.sha256`; commit, tag and package fields are Sunday placeholders.
  Here it checks 81 OK / 2 FAILED until the evidence below is copied over (expect 83 OK). It still cites Ignite shas:
  Sunday's record needs the provenance line "Ignite 1a0601e twin_to_line tree 7a59a199 = Crusaders 8402039".
- **Evidence still only in `~/code/ignite-fix`:** `results/freeze_prep_2/` (2 test logs) and `results/rc_1024/` (RC
  package, `spark_e597c4b/` = the stream reference, `MAC_VS_SPARK_e597c4b.json`). Copy before 6 pm (NEXT ACTION 1).
- `LIVE_RUNBOOK.md` §7 (Monday) is correct but too long to type at CSI's keyboard, and lines 3, 27, 121 still say
  `~/code/ignite`: both fixed by Codex tonight (NEXT ACTION 2).

### Team
- **Vang:** "M8" = an operator UI around our checker (reads our output folder; standard library only; 127.0.0.1 only).
  PR #2 (`feat/m8-operator-ui`, adds `ignite_core/operator_ui/`). Noor's review requested changes: remove `samples/`
  (dataset-derived images) and all `__pycache__`; re-fetch the overlay on every poll; "STALE" must not hide an
  action-required result, default `--stale-after` 30 (still 5 at `c60bb04`). Vang accepted and is fixing it.
  - **Merge order matters:** the package archives the whole repo, so whatever is in the freeze commit ships to CSI's
    GB10. Merge PR #2 only after his fix; then the UI ships automatically under `code/ignite_core/operator_ui` with the
    freeze sha (no separate pinned copy). If it isn't fixed by the freeze, leave it out; Monday uses the console grid.
  - PR #2 and our PR both add `.gitignore` (add/add conflict): keep ours and add `.pytest_cache/`.
  - Run the UI from the package's `code/` folder (not `code/twin_to_line`):
    `cd <pkg>/code && ../venv/bin/python -m ignite_core.operator_ui --output-dir <checker --out>`. Start the checker
    first; restart the UI whenever the checker restarts into a new folder. Claude Code checked side by side: his UI
    served our 1080 preflight output on 127.0.0.1. Add this to §7 after the Sun 3 pm joint test.
- Code Crusaders `main` contains PR #1 (`9040ded`: `ignite_core/` with older copies of positions, fill_state,
  eval_metrics). Our runtime never imports them.
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
1. **Before 6 pm, Noor (in `~/code/crusaders`):** copy the evidence and re-check the manifest:
   `rsync -a --ignore-existing --exclude pip_cache ~/code/ignite-fix/twin_to_line/results/freeze_prep_2 ~/code/ignite-fix/twin_to_line/results/rc_1024 twin_to_line/results/`
   then `cd twin_to_line && shasum -a 256 -c FREEZE.sha256` (expect 83 OK). Commit and push this handoff.
2. **Sat 6 pm, Codex (`freeze-prep-3`, new branch from `feat/twin-to-line` in `~/code/crusaders`):**
   - turn §7 into 5 scripts copied to the USB root (`start.sh`, `stage.sh`, `seal.sh`, `live.sh`, `export.sh`). Each
     ends in one OK/STOP line; `stage.sh` copies the newest camera image to `setup_NN.<ext>` and writes the log row
     itself; `seal.sh` counts the setups actually staged (skips logged); nothing deletes. Rewrite §7 to these commands
     with expected outputs; end-to-end test in an arm64 container;
   - `LIVE_RUNBOOK.md` lines 3, 27, 121: `~/code/ignite` → `~/code/crusaders`; any other Ignite path in docs;
   - D042 in DECISIONS.md: active work moved to the Code Crusaders repo (supersedes D028); Ignite = archive; identity
     proven by folder hash `7a59a199`;
   - FREEZE draft: the provenance line above; record `results/rc_1024/inputs/staged_3` and its sha256.
   Then Noor fast-forwards, runs the tests, pushes.
3. **Sat night, Noor:** update the staging doc (12:30 confirmed; stills at 1920×1080; GB10 setup inside the slot; cut
   order: optional live defects, filming, live duration; never labels or backups) and send it to Tony with his role
   and what "??" means. Post the organizer question if it isn't posted.
4. **Sun:** morning: review Vang's fixed PR #2; merge only if fixed · 3 pm: joint Spark test (checker + UI) · ~5 pm:
   freeze build (final commit + tag in this repo, package rebuilt from it, `FREEZE` completed, Spark dry run) ·
   **6–8 pm: Noor types the 5 commands himself on the Spark from the real USB stick, timed, no agent** ·
   **9 pm: freeze** · then push the branch and the tag.
5. **Mon:** arrive 12:15; ask how frames reach the GB10, USB, screen. Bring an exFAT USB (package `.tar` + `.sha256`,
   runbook PDF, empty results folder, several GB free), laptop + charger, printed staging list + numbering map, label
   sheet, paper log, phone, pen. Staged images are copied and sealed, never scored on the day; the live run starts
   after the seal, in a fresh folder.
6. **Mon evening:** adjudicate labels without predictions (two people), lock them by sha256, then run
   `score_sealed.py` once for all arms, from `~/code/crusaders`.

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
- **D042 (pending, Codex tonight):** work moved to the Code Crusaders repo; Ignite is the archive (supersedes D028).

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
- **Code Crusaders repo (active):** https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-new-team-10-f7d5357a4a
  · local `~/code/crusaders` · project dir `twin_to_line/` · `.venv` → `~/code/ignite/.venv` (Python 3.11) · branches
  `feat/twin-to-line` (ours), `feat/m8-operator-ui` (Vang, PR #2), `main` @ `9040ded`.
- **Ignite repo (archive):** https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-new-team-11-b020278d03
  · local `~/code/ignite` (history; `feat/live-runtime` @ `1a0601e`, local only) · old Codex worktree `~/code/ignite-fix`.
- Data (not in git): twin `~/Downloads/csi_digital_twin/csi_digital_twin_v1/`; real `~/Downloads/csi-vial-fill-dataset/cap_seal/`;
  labeler `~/Downloads/vial-labeler-participants-20261014.zip`.
- Docs: staging list https://claude.ai/artifact/P1WRYUf2vJb2kUR2XPHTKh (private; update before sharing) · proposal
  https://claude.ai/code/artifact/7e742133-103c-483e-8409-8aca3b5e131e · repo reports `RC_2026-10-24.md`,
  `ARM_B_DOUBLE_PRIME_RESULT.md`, `FREEZE_2026-10-25.md`, `LIVE_RUNBOOK.md`, `GATE_v2.0.2.md`.

## Known-working commands (in `~/code/crusaders`)
```bash
git status -sb                         # expect: ## feat/twin-to-line...origin/feat/twin-to-line
IGNITE_TWIN=~/Downloads/csi_digital_twin/csi_digital_twin_v1 .venv/bin/python -m pytest twin_to_line/tests -q   # expect: 247 passed
git switch feat/twin-to-line && git merge --ff-only <agent-branch>   # after the agent reports clean; if it refuses, stop and ask
cd twin_to_line && shasum -a 256 -c FREEZE.sha256    # expect: all OK (83 after the evidence copy)
../.venv/bin/python gb10/build_package.py             # package from HEAD -> results/gb10_pkg/
ssh yt-codecrusaders6286@172.30.102.94                # YAAD Spark; FortiClient VPN first
```

## Known risks
- How frames reach our program on CSI's GB10 is unknown; covered by folder / rewritten file / stream / stills-on-request.
- CSI's GB10 itself is untested (user rights, USB mount, screen). No real 1080 frame with caps yet.
- B′ is DEV-fit; organizer ruling pending. Every model arm failed G3.
- PR #2 merged before Vang's fix would ship dataset-derived images to CSI's GB10 inside the package.
- Staging time is unmeasured (~2 min per setup needed). Cut order: optional live defects, filming, live duration.
- The rule flags crooked caps as "??" (UNSURE), never as "crooked": brief Tony and explain it to the supervisor.
