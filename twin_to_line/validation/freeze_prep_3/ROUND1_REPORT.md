# Freeze-prep 3 — review round 1

Implementation/package tested: `1ac85cfd0fe9fb6fe09729177850697a387dc339` on `freeze-prep-3`.
New commits only; no amend, rebase, squash, tag or push. Sunday release fields stay placeholders.
One-page operator sheet: [MONDAY_ONE_PAGE.pdf](../../monday/MONDAY_ONE_PAGE.pdf).

## 1. Round-0 preconditions and limits of the historical evidence

Current `git ls-tree 8402039 twin_to_line` output:

```text
040000 tree 7a59a199b1d9507b5ba3140166c5156378b04542    twin_to_line
```

Ignite `feat/live-runtime` at `1a0601e` is recorded as tree `7a59a199` in the inherited handoff and D042.
The Ignite commit object is not present in this checkout. No new query was made against the protected archive
repo, so a second original Git command transcript cannot be supplied. The recorded identity is preserved;
only the Crusaders object has been queried directly during this review.

The original 83-entry manifest was reconstructed from baseline Git bytes for tracked files and the existing
local ignored evidence, then checked with `shasum -a 256 -c FREEZE.sha256`: **83 OK, 0 mismatches, exit 0**.
This is a reconstruction checked now, not a claim that a missing round-0 console log was recovered.
`round1_preconditions.log` contains every check line.

The inherited original evidence has `245 passed in 67.63s` plus `2 passed in 0.40s` = **247 passed**.
The handoff separately records 247 after the repository move. No single original 247-test transcript exists
in the available evidence; these two actual logs are `results/freeze_prep_2/tests.log` and `runbook_tests.log`.

## 2. Why three live stills produced 60 prediction rows

The round-0 summary was `3 frames, 2 events, {'scored': 2, 'arm': 1}`. Each scored event produces 30 hole rows.
`cap_seal_20260922_171520_389286.jpg` was logged `arm` (7421 arm pixels) and correctly produced no predictions.
The other two stills were scored: 2 × 30 = **60**, not 90. No frame or prediction output was missing.

## 3. Exact live command and both gate answers

The working directory is `$INSTALL/$PKG/code/twin_to_line`. `$PY` is that package's `venv/bin/python`.
The helper invokes these argument vectors (shell notation for readability):

```bash
# Enter or n: continuous camera, runtime default --gate events
"$PY" -u live_inspect.py --source "folder:$CAMERA" --out "$OUT"
# y: independent stills saved on request
"$PY" -u live_inspect.py --source "folder:$CAMERA" --out "$OUT" --gate each
# Stream equivalents
"$PY" -u live_inspect.py --source "stream:$URL" --out "$OUT" --every 1.0
"$PY" -u live_inspect.py --source "stream:$URL" --out "$OUT" --gate each --every 1.0
```

No `--track`, inspector override, threshold override or scoring-logic changes are introduced.
The prompt is `Enter = n (continuous camera feed); y only for stills CSI saves on request. [y/n]:`.

## 4. Complete package delta against RC e597c4b

Compared `results/rc_1024/package/ttl_gb10_e597c4b` with the actual `ttl_gb10_1ac85cf` rehearsal package.
All 14 `live_inspect.RUNTIME_FILES` are byte-identical; **0 runtime differences**. Bootstrap, wheels, bundled
Python and preflight image/prediction bytes also match. The metadata and file inventory changes are below.
The full machine-readable result is `round1_package_diff.json`. No removed files in either archive layer.
Final evidence commits after the tested implementation add reports and refreshed hashes only.

### Outer package

Added (0):

None.

Changed (5):

```text
FILES.txt
README.txt
SHA256SUMS
code.tar
preflight/expected/mac.json
```

Removed (0):

None.

### Inside code.tar

Added (31):

```text
ignite_core/eval_metrics.py
ignite_core/fill_state.py
ignite_core/position_map_G.json
ignite_core/positions.py
ignite_core/requirements.txt
ignite_core/tests/test_eval.py
ignite_core/tests/test_fill_state.py
ignite_core/tests/test_positions.py
twin_to_line/ARM_B_DOUBLE_PRIME_RESULT.md
twin_to_line/fit_rule_dev_good_only.py
twin_to_line/monday/MONDAY_ONE_PAGE.pdf
twin_to_line/monday/export.sh
twin_to_line/monday/live.sh
twin_to_line/monday/monday_ops.py
twin_to_line/monday/seal.sh
twin_to_line/monday/stage.sh
twin_to_line/monday/start.sh
twin_to_line/rule_thresholds_dev_good_only.json
twin_to_line/score_sealed.py
twin_to_line/tests/test_fit_rule_dev_good_only.py
twin_to_line/tests/test_operator_scripts.py
twin_to_line/tests/test_runbook_freeze.py
twin_to_line/tests/test_score_sealed.py
twin_to_line/validation/freeze_prep_3/README.md
twin_to_line/validation/freeze_prep_3/build_rehearsal.py
twin_to_line/validation/freeze_prep_3/manifest_check.log
twin_to_line/validation/freeze_prep_3/package_diff.py
twin_to_line/validation/freeze_prep_3/print_runbook.py
twin_to_line/validation/freeze_prep_3/rehearse.py
twin_to_line/validation/freeze_prep_3/section7.md
twin_to_line/validation/freeze_prep_3/tests.log
```

Changed (14):

```text
.gitignore
twin_to_line/ARCHIVE_COMMIT
twin_to_line/ARM_B_DOUBLE_PRIME_PROPOSAL.md
twin_to_line/DECISIONS.md
twin_to_line/FREEZE.sha256
twin_to_line/FREEZE_2026-10-25.md
twin_to_line/HACKATHON_HANDOFF.md
twin_to_line/LIVE_RUNBOOK.md
twin_to_line/NEXT_SESSION_PLAN.md
twin_to_line/RC_2026-10-24.md
twin_to_line/README.md
twin_to_line/V2_RENDER.md
twin_to_line/gb10/build_package.py
twin_to_line/tests/test_rc_1024.py
```

Removed (0):

None.

USB-root additions beside the tar:

```text
start.sh
stage.sh
seal.sh
live.sh
export.sh
monday_ops.py
monday.sha256
```

## 5. Interpreters and absence of system Python

All five `.sh` scripts run under **Bash**. `start.sh` and `export.sh` need no system Python.
Start calls the unchanged Bash bootstrap, which unpacks bundled **CPython 3.11.17 aarch64** and creates
`$INSTALL/$PKG/venv/bin/python`. Its fallback supports system Python 3.10–3.12 with venv if the bundle fails.
Stage/seal/live use the saved absolute `$PY`, with `-I`, to execute `monday_ops.py`; they do not look up `python3`.
The helper uses Python 3.10+ syntax/stdlib; staging additionally uses packaged NumPy/OpenCV to read dimensions.
Live starts the unchanged inspector with that same package interpreter. Export remains Bash-only even when setup fails.
The PDF builder/package comparison/rehearsal harness are development tools, not Monday dependencies.

Actual container probe, before start:

```text
$ command -v python3
(no output; exit 1; before start.sh)
```

The harness itself was launched by an absolute path to a separately extracted bundled interpreter outside PATH.
The image was arm64 `ubuntu:24.04`, UID/GID 1000:1000. Container exit 0; both setup runs used bundled Python.

## 6. The 20 round-0 manifest additions

These are the exact paths added between the inherited 83-entry manifest and round-0 `edb5858` (103 entries).
The two private raw-log entries were subsequently dropped in round 1; the inventory below records history.

```text
NEXT_SESSION_PLAN.md
README.md
V2_RENDER.md
monday/export.sh
monday/live.sh
monday/monday_ops.py
monday/seal.sh
monday/stage.sh
monday/start.sh
results/rc_1024/inputs/staged_3/cap_seal_20260922_170747_881863.jpg
results/rc_1024/inputs/staged_3/cap_seal_20260922_171520_389286.jpg
results/rc_1024/inputs/staged_3/cap_seal_20260922_171525_149283.jpg
tests/test_operator_scripts.py
validation/freeze_prep_3/README.md
validation/freeze_prep_3/build_rehearsal.py
validation/freeze_prep_3/container_transcript.log
validation/freeze_prep_3/package_build_final.log
validation/freeze_prep_3/rehearse.py
validation/freeze_prep_3/section7.md
validation/freeze_prep_3/tests.log
```

## 7. Remaining old repository path hits

`round1_remaining_paths.txt` records the source hits before this report was added. The protected frozen-gate
command remains verbatim. The handoff retains archive/symlink identities and stale historical next-action text.
One substring match is the valid package module path `code/ignite_core/operator_ui`, not an old checkout path.
There are no old checkout paths in the active runbook or Monday scripts. Report artifacts repeat the search
terms as evidence; they are not commands for CSI to run.

## Timed non-root arm64 container rehearsal

Times include command execution, fixed safety waits and scripted keyboard input; they are not estimates of CSI's
physical setup time. Raw logs are local under `results/freeze_prep_3/round1/container.log`; no machine identifiers
are shipped. `round1_steps.tsv` contains each complete final line with only HOME/USB paths replaced by variables.

| Step | Seconds | Final outcome |
| --- | ---: | --- |
| start: no typing | 6.565 | SETUP OK |
| start: again, no typing | 6.699 | SETUP OK |
| live before seal | 0.035 | STOP |
| stage: six setups, one skip, one retake, two refusals | 4.563 | STAGING OK |
| seal separate visit: zero staged | 0.480 | SEALED USB OK |
| live allowed after zero seal: Enter gate | 0.257 | LIVE OK |
| seal: final and earlier attempts | 0.119 | SEALED USB OK |
| seal again | 0.036 | STOP |
| stage after seal | 0.037 | STOP |
| live refuses staged | 0.039 | STOP |
| live refuses disk seal | 0.039 | STOP |
| live refuses USB seal | 0.039 | STOP |
| live same camera: Enter gate, 640 + 1080 + rewritten 640, SIGINT | 1.165 | LIVE OK |
| live same camera: y gate, 640 + 1080 + rewritten 640, SIGINT | 1.345 | LIVE OK |
| export every visit | 9.738 | RESULTS USB OK |
| re-verify exported archive | 1.499 | evidence.tar |
| start with 0 tars | 0.008 | STOP |
| start with 2 tars | 0.004 | STOP |

Staging: six setup numbers, one skipped, five final captures, one retained earlier attempt; seven log rows and
six staged files. Duplicate SHA256 and 320×240 inputs were refused without advancing. Every capture came through
one rewritten camera filename. Zero-capture seal printed WARNING and permitted live. Resealing and post-seal
staging stopped. Live reused the same camera folder; both gate answers read 640, 1080→640, then rewritten 640
and ended cleanly on SIGINT. Both visits and all attempts were exported; a separate SHA256 recheck passed.

## Current validation

**265 passed**, 0 skipped, 79.83 seconds, using only the requested full-suite command; see `round1_tests.log`.
The current draft manifest check is saved in `round1_manifest_check.log` and excluded from itself.
The one-page PDF was rendered and visually checked with no clipping. All protected runtime, scorer, map,
threshold, warp, gate and dose files remain byte-identical to the starting branch. No holdout frames were read.
Raw logs with machine/account paths were moved out of Git, not merely hidden by ignore rules.
