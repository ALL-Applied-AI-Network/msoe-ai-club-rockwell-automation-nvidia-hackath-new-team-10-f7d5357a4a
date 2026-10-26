# Twin-to-Line — Handoff (read first)
Updated: 2026-10-25 (Sun) 8:00 pm CT. **FROZEN: commit `4c24ec2`, tag `freeze-2026-10-25` (pushed).** All work lives in
the Code Crusaders team repo: local clone `~/code/crusaders`, branch `feat/twin-to-line` (pushed; PR #3 to `main`
awaiting Vang's review after Monday). Two identical copies of this file: `twin_to_line/HACKATHON_HANDOFF.md` in the
repo (canonical) and the Claude Project doc `claude/hackathon-handoff.md`. Decisions: repo `twin_to_line/DECISIONS.md`
(D001–D042, canonical); the Project copy of DECISIONS.md stops at D038 (D039–D042 summarised below).
`claude/next-session-plan.md` is superseded except §2 (requirements) and §8 (message style). The Ignite repo
(`~/code/ignite`) is the read-only archive holding the full commit history (D042).
New session: work from `~/code/crusaders`, run `git status -sb` first, then go to **NEXT ACTION**. Do not reopen settled
decisions without new evidence. Replace stale facts; no diary. **Never put passwords, tokens, invitation links, team
codes, auth URLs or the Drive kit's share link in this file or in memory.**

## Standing rules
- Work on `feat/twin-to-line` in `~/code/crusaders`. Push only with Noor's OK. `main` changes only through a PR that a
  teammate reviews; never self-merge. Don't modify the frozen gate. **After the freeze, nothing that ships changes:
  only docs and results commits until Monday's scoring is done.**
- Stage files by name, never `git add -A` / `-u`. Always run tests with the path: `pytest twin_to_line/tests`.
- No AI/assistant attribution anywhere (commits, PRs, docs). Tool credits go in the final submission's references.
- Never commit raw data, the twin, the labeler, renders, results, CSI images or model weights. Never open runs 4–5.
  New files must not use ignored names (`outputs/`, `models/`, `results*/`, `*.zip`, `*.tar.gz`).
- Ask Noor before spending money. Draft messages for Noor; he sends them. Short, plain language.
- Agents: Claude Code usage limited until Wed (main tasks only); **Codex** does the rest. One agent at a time, on its
  own new branch from `feat/twin-to-line`; Noor fast-forwards it and runs the full tests. The implementer never reviews
  its own work. Claude in the Project "hackathon" reviews outputs and writes the prompts. **Check agent claims against
  `git diff`, not the report.** Agents never edit this file (Claude writes it; Codex may only copy it in).
- Noor works best one step at a time, each step with its expected output. When he types on the Spark, the first
  command is `ssh …` alone; check the prompt says `root@codecrusaders6286` before pasting anything else.
- Keep `~/code/ignite` until after Nov 1: `~/code/crusaders/.venv` is a symlink to `~/code/ignite/.venv`.

## State (Sun Oct 25, 8:00 pm): frozen
- **Freeze:** `4c24ec2` = Vang's PR #2 head `c31a0ce` merged into our branch (only the `.gitignore` add/add conflict;
  resolved as ours + `.pytest_cache/`). Tag `freeze-2026-10-25` pushed 7:52 pm. Tests 274 passed; `FREEZE.sha256` 106
  OK at the tag. The freeze record (`FREEZE_2026-10-25.md`) is filled in a docs-only commit after the tag.
- **Package (what runs Monday):** `results/gb10_freeze/ttl_gb10_4c24ec2.tar`, 243,507,200 bytes, sha256
  `89a55f4977d0a44c65c65ec67d8fb753a070242b3aeb7043d204c5e10e110294`, built 6:50 pm from the tagged commit (Mac
  preflight OK). An earlier build of the same commit (`results/gb10_joint`, sha256 `a8072d84…`) was used for the
  joint test; same code, different tar bytes.
- **Drive kit (Monday's way in):** `results/gb10_freeze/ttl_kit.tar` = folder `ttl_kit/` with the 10 files (package
  `.tar` + `.tar.sha256`, 5 scripts, `monday_ops.py`, `monday.sha256`, `MONDAY_ONE_PAGE.pdf`); 243,578,880 bytes,
  sha256 **`c5374baebdb47a75e1ca2a4ca31a625f735675babb4e974359eefd479e0eb71a`**. Uploaded to Noor's Drive folder
  "TTL Monday kit (CSI Oct 26)", shared "anyone with the link: viewer" (Noor has the link; **turn sharing off after
  2 pm Mon**). A `curl` download with no login matched the hash. GNU tar may print a harmless "unknown extended header
  keyword" warning (macOS quarantine metadata).
- **USB stick: not used for the package.** Noor's new USB-A stick (via USB-C adapter) corrupted the 243 MB file on two
  separate copies (same size, wrong hash); the 6 small scripts copied fine. It may only carry results home: `export.sh`
  verifies every file. Tony was asked to bring a stick (answer unknown).
- **Spark rehearsal (7:35–7:50 pm, Drive-kit route, Noor typing):** kit sha OK → `start.sh` SETUP OK (bootstrap 5 s;
  preflight 0 holes differ from the Mac at 640 and 1080) → `stage.sh` 3 staged, 1 skip, 1 retake → `seal.sh`
  SEALED USB OK (retake final, earlier attempt retained) → `live.sh` 21 frames, 20 scored, 1 arm → **foreground Ctrl-C
  → LIVE OK** → `export.sh` RESULTS USB OK.
- **Joint test (Codex, 4 pm):** checker on the Spark OK (600 rows = 20 × 30, 1 arm); Vang's UI served HTTP 200 and its
  latest event matched predictions.csv 30/30. A background-launched checker ignores SIGINT (non-interactive shells
  give async jobs SIGINT = ignored): a launcher artifact, not a Monday bug. Monday fallback if Ctrl-C doesn't finish
  within 30 s: `pkill -TERM -f live_inspect.py` in a second terminal (wrapper then prints STOP and frees export).

### Arms (all frozen; each scored once on the sealed set with `score_sealed.py`, labels locked by sha256 first)
DEV G3 = runs 1–3, 20 placement events, 32 defect / 551 non-defect observations, LIVE_FILL, no tracker.

| Arm | What | DEV G3 |
|---|---|---|
| A | Synthetic-only ResNet-18 v1; D038 t 0.50224 / pre-registered t 0.99968 | 32/32, 483 FP (NO-GO) / 15/32, 308 FP |
| B | Rule, thresholds from synthetic validation (`rule_thresholds_synth.json`: 0.032 / 0.94 / 2.0) | 19/32, 132 FP (NO-GO) |
| B′ | Rule, DEV-fit (`rule_thresholds.json`: 0.081 / 0.22 / 4.4; sha256 `322c29ec…`) — **Monday's live inspector** | 31/32, 14 FP (in-sample) |
| B″ | Rule, DEV good-only (D041, `rule_thresholds_dev_good_only.json`: 0.081 / 0.22 / 4.8) | 28/32, 8 FP (NO-GO, in-sample) |

- B″ differs from B′ only in the offset threshold. Its 3 extra misses are one physical crooked cap (G13, run 1); its
  missing-cap miss (G07) is the same one B′ misses.
- The rule arms never name `crooked_cap` / `missing_cap`: they flag them UNSURE. B′ is DEV-fit: disclose everywhere.
- v2 (D040) stopped at its check render: no v2; arm A = v1. Gate gate-v2.0.2: PASS WITH WORK (D037). `train_v1`: 568
  scenes, 17,040 crops.
- `score_sealed.py` keeps its once-only lock ledger in the clone's own `results/`: run Monday evening's scoring only
  from `~/code/crusaders`.

### Live runtime and Monday kit
- Sizes: 640×480 as is; 1920×1080 warped into the 640 grid (D039); anything else `wrong_size` (320×240 refused).
- Sources: `folder:` (ignores files present at start; re-reads a rewritten name), `stream:` (untested on a real
  camera), `replay:`. Staged stills never scored on the day.
- Console legend: `ok` PASS · `DEF` DEFECT · `MISS` MISSED · `??` UNSURE (check by eye) · `.` WAITING.
- Monday commands: `bash ~/ttl_kit/start.sh` (rerun resumes the current visit; `--new-visit` only when a STOP says so)
  → `bash ~/ttl/stage.sh` (camera folder once; Enter after CSI saves · `s` skip · `r` retake the last setup after a new
  save · `q` done) → `bash ~/ttl/seal.sh` (0 staged allowed with WARNING) → `bash ~/ttl/live.sh` (camera folder, the
  staging one is allowed; gate Enter = continuous feed, `y` = stills saved on request; confirm reset) → Ctrl-C once →
  `bash ~/ttl/export.sh`. Without a stick, the kit folder `~/ttl_kit/` plays the USB: seal copies and exports land
  there; ask CSI to copy it for us, or carry it home on a verified stick.
- Optional UI (Vang): second terminal, `source ~/ttl/operator.env && cd "$INSTALL/$PKG/code" && "$PY" -m
  ignite_core.operator_ui --output-dir <LIVE OUT>`, then http://127.0.0.1:8765 in the GB10's browser. Stop it before
  export. The console grid is the demo.
- Not verified: real 1080 frames with caps; CSI's own GB10 (internet, browser, curl, user rights, USB ports); how CSI's
  camera delivers frames.

### Team
- **Vang:** M8 operator UI, PR #2 (`feat/m8-operator-ui` @ `c31a0ce`, all 5 requested fixes verified) is merged into
  our branch, not into `main`; PR #2 and PR #3 both go to `main` after Monday (Vang reviews #3; merge by Oct 31). Asked
  him not to push more before Monday.
- **Tony** (Anthony Harvey): attends Monday; has the staging doc and his brief (reads setups aloud, checks holes vs the
  map, paper log, side photos if allowed, logs the live run). Asked to bring a USB stick.
- **Zara:** offered help Oct 25; tasks drafted (sourced business-case numbers and video storyboard by Oct 28; one-page
  summary draft by Oct 29; optional backup labeler Monday evening). Whether Noor sent it: unknown.
- Code Crusaders `main` @ `9040ded` (PR #1, `ignite_core/` older copies; our runtime never imports them).

### CSI and organizers
- CSI session **Mon Oct 26, 12:30–2 pm, confirmed** (Noor + Tony; arrive 12:15). Camera streams 320×240 and
  1920×1080; our software runs on CSI's GB10; no setup time before the slot.
- Shamar never answered the Oct 23 questions (internet, USB, how frames reach the GB10, screen). Ask the supervisor at
  12:15 (staging doc checklist). Never confirmed: vials/caps (~32 + 34), side photos, filming, backup slot.
- Organizer question (rule thresholds on runs 1–3 vs good images only) posted; no reply as of Sun 8 pm. Live stays B′,
  disclosed as DEV-fit; B″ is the out-of-sample comparison.
- Docs: staging list https://claude.ai/artifact/P1WRYUf2vJb2kUR2XPHTKh (shared with Tony) · study guide
  https://claude.ai/artifact/HXuXWc83jkGNVYN1x3Vqeu · proposal https://claude.ai/code/artifact/7e742133-103c-483e-8409-8aca3b5e131e

### Compute
- Brev: done; `cc-isaac-gate-2` stopped (delete after Nov 1, or now if a stopped instance still bills).
- YAAD Spark: `ssh yt-codecrusaders6286@172.30.102.94` with the FortiClient VPN connected (timeouts = VPN off). Bare
  container: no python3, curl, tmux, scp or rsync; copy with `tar -cf - … | ssh … 'tar -xf - -C …'`. Test leftovers to
  delete after the event: `/workspace/ttl_joint`, `kit_1025`, `ttl_cam_src`, `cam_joint`, `stage_cam_1025`, `cam_1025`,
  `home_joint`, `home_rehearsal`, `ttl_rehearsal_1023`, `ttl_rc_1024`; joint-test runtime PID 2157 may still be running
  (`kill -TERM 2157`).

## NEXT ACTION
1. **Sun night:** Codex docs-only commit after the tag: this file copied into the repo + `FREEZE_2026-10-25.md` filled
   (commit, tag, package and kit sha256, rehearsal and joint-test results; no Drive link). Noor fast-forwards and
   pushes. Print: staging doc with the numbering map (2 copies), paper-log sheets, `MONDAY_ONE_PAGE.pdf` with the kit
   sha256 `c5374bae…71a` and the start command `bash ~/ttl_kit/start.sh` written on it. Charge phone and laptop.
2. **Mon 12:15:** ask the supervisor how frames reach the GB10, whether it has internet and a browser/terminal, and
   for a USB stick if Tony has none. Get the kit onto the GB10:
   `curl -L -o ~/ttl_kit.tar "https://drive.usercontent.google.com/download?id=<FILE_ID>&export=download&confirm=t"`
   (or the browser, then `mv ~/Downloads/ttl_kit.tar ~/`) → `sha256sum ~/ttl_kit.tar` = `c5374bae…71a` →
   `cd ~ && tar -xf ttl_kit.tar` → `bash ~/ttl_kit/start.sh` → SETUP OK. Then staging, seal, live, export as above.
   Never score staged images on the day; never change code or thresholds.
3. **Mon 2 pm:** turn the Drive link sharing off. Get results off the GB10 (verified stick, or CSI copies `~/ttl_kit/`).
4. **Mon evening:** adjudicate labels without predictions (two people; Zara as backup), lock them by sha256, then run
   `score_sealed.py` once for all arms, from `~/code/crusaders`.
5. **After Monday:** Vang reviews PR #3; merge #2 and #3 to `main` by Oct 31; error analysis; Zara's business case and
   video plan; write-up Oct 30; video and one-page summary Oct 31; submit Sun Nov 1 11:59 pm CT.

## Milestones
**Mon Oct 26 12:30–2 pm CSI** · Oct 26–27 adjudicate, score every arm once · Oct 28–29 backup slot if needed
(unconfirmed), error analysis, business case · Oct 30 write-up · Oct 31 video, one-page summary, both PRs on `main` ·
**Sun Nov 1 11:59 pm CT final submission** (~5-min video, one-page summary, repo, references) · Thu Nov 5 5–8 pm
finals, MSOE Diercks Hall.

## Write-up headline (draft; every number in-sample until the sealed set)
From the digital twin alone: all 30 positions located with no per-hole teaching (640: 30/30, ~1.68 px; 1080 via D039),
and missed placements told apart from waiting holes with no PLC signal. Synthetic-only appearance did not transfer
(ResNet v1 and synthetic rule thresholds fail G3); the main gap was traced to the hole floor's contrast. A 3-number
rule needs real images; on DEV, setting them from good images only lands within one crooked cap of defect-tuned
thresholds; the sealed Oct 26 set tests B″ against B′ out of sample. Distinctiveness: a novel integration, not a new
algorithm (cap checks are commodity, D002). Likely pushback: DEV-fit live rule, "detection" via UNSURE, no measured
changeover time. Business case: no CSI setup-time or defect-cost numbers; Zara sources them. Verify any prior-art link
before citing it.

## Decisions since D038 (full text in repo DECISIONS.md)
- **D039:** 1920×1080 frames are warped into the DEV 640×480 grid; other sizes `wrong_size`.
- **D040:** one ResNet-18 v2 attempt, pre-declared; its check render failed, so no v2; arm A stays v1.
- **D041:** arm B″, rule thresholds fitted once on good DEV observations only; scored once on the sealed set.
- **D042:** active work moved to the Code Crusaders repo; Ignite is the read-only archive; supersedes D028.

## Real-data boundary (D011, D025, D033, D036)
Runs 1–3: development/reference only. Runs 4–5: holdout, never for training, thresholds, prompts, recipe or model
choice. Organizer release: development + evaluation only. **Oct 26 staged images: sealed final test; labels adjudicated
before any model runs on them; scored once after the freeze; never training or tuning.** Unit = placement event or
arrangement-position, never a frame.

## Organizer facts
- Derek Riley (Oct 16): confirm training-data rules first (D025); crooked cap = any cap not fully seated, a defined
  refinement allowed (ours: ≥20°, D036); a supervised live run counts as the hardware demo if our implementation runs on
  real hardware (D029); no PLC/robot/placement-count data (D019, D021); G01–G30 is our twin convention (D022).
- CSI Data Collection guide: may not move/re-aim/reconfigure a camera; change PLC, robot, FactoryTalk, safety, recipe
  or program; connect hardware to the plant network without approval. Before a physical test: freeze code and model,
  documented start/stop, metrics, failure handling tested on saved data. Trial log: every attempt.
- Dataset 1 page: avoid training on the real images; report per class, misses and false alarms separately, the
  false-alarm rate at the threshold that catches every known defect, full fixture scenes; missed defects matter more
  than false alarms. `damaged_vial` out of scope for v1, still counted (D026).

## Team and contacts
- Code Crusaders: Zara, Matthew, Trevor, Vang, Tony Harvey, Jamie Gill, Noor ("Jatt" on Discord).
- CSI/testbed: Shamar Webster (`#data-and-testbed-requests`, webste63@uwm.edu) · dataset: Brett Storoe, Derek Riley,
  Tanner Cellio (`#dataset-1-vial-inspection`) · compute: Dan Schneider (`#compute-questions`) · program: Tom
  Hershberger.

## Paths
- Code Crusaders repo: https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-new-team-10-f7d5357a4a
  · local `~/code/crusaders` · project dir `twin_to_line/` · branches `feat/twin-to-line` (ours, PR #3),
  `feat/m8-operator-ui` (Vang, PR #2), `main` @ `9040ded`.
- Ignite repo (archive): …-new-team-11-b020278d03 · local `~/code/ignite` (`feat/live-runtime` @ `1a0601e`).
- Data (not in git): twin `~/Downloads/csi_digital_twin/csi_digital_twin_v1/`; real `~/Downloads/csi-vial-fill-dataset/cap_seal/`.
- Rehearsal kit: `~/ttl_rehearsal_kit/` (frames, feeder, REHEARSAL.txt). Freeze evidence: `twin_to_line/results/`
  (`gb10_freeze/`, `joint_1025/`, `freeze_prep_5/`, `gb10_freeze/DRIVE_KIT_VERIFICATION.md`).

## Known-working commands (in `~/code/crusaders`)
```bash
git status -sb                         # expect: ## feat/twin-to-line...origin/feat/twin-to-line
IGNITE_TWIN=~/Downloads/csi_digital_twin/csi_digital_twin_v1 .venv/bin/python -m pytest twin_to_line/tests -q   # expect: 274 passed
git switch feat/twin-to-line && git merge --ff-only <agent-branch>   # after Claude reviews the diff
git log --oneline -1 freeze-2026-10-25                # expect: 4c24ec2 …
ssh yt-codecrusaders6286@172.30.102.94                # YAAD Spark; FortiClient VPN first
```

## Known risks
- CSI's GB10 may have no internet or no browser/curl: then ask the supervisor for a USB stick and copy the 10 kit files
  from the laptop (`results/gb10_freeze/ttl_kit/`) onto it on the spot; check the tar's sha256 from the stick first.
- How frames reach our program on CSI's GB10 is unknown (folder / rewritten file / stream / stills on request).
- B′ is DEV-fit; organizer question unanswered. Every model arm failed G3.
- Staging time is unmeasured (~2 min per setup needed). Cut order: optional live defects, filming, live duration;
  never the paper log or the copy of results.
- The rule flags crooked and missing caps as `??`, never by name: explain it to the supervisor.
