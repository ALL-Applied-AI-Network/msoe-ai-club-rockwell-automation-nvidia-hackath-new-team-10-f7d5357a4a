# Freeze-prep 4 — resume the current Monday visit

## 1. Preconditions (all passed before creating the branch)

```text
$ git status -sb
## feat/twin-to-line...origin/feat/twin-to-line
```

The working tree was clean. The exact required test command was run from the repository root:

```text
$ IGNITE_TWIN=~/Downloads/csi_digital_twin/csi_digital_twin_v1 .venv/bin/python -m pytest twin_to_line/tests -q
265 passed in 79.58s (0:01:19)
```

```text
$ cd twin_to_line && shasum -a 256 -c FREEZE.sha256
104 OK lines, 0 failed; exit 0
$ git switch -c freeze-prep-4
Switched to a new branch 'freeze-prep-4'
```

## 2. Commits

- `48dbb239b8ae9f7aba9a57b4a41c4db0c6cde979` — Resume successful Monday visits unless a new visit is requested.
- The subsequent report-only commit is named **Record freeze-prep-4 resume validation**. Its hash is listed in
  the final delivery message; it is not embedded in its own content.

Only local commits. No push, merge, PR, tag, amend, rebase or squash.

## 3. Changes against feat/twin-to-line

```text
 twin_to_line/FREEZE.sha256                        |  10 +-
 twin_to_line/LIVE_RUNBOOK.md                      |   7 +-
 twin_to_line/monday/MONDAY_ONE_PAGE.pdf           | Bin 37347 -> 37584 bytes
 twin_to_line/monday/start.sh                      |  64 +++++-
 twin_to_line/tests/test_start_resume.py           | 162 +++++++++++++++
 twin_to_line/validation/freeze_prep_3/section7.md |   7 +-
 twin_to_line/validation/freeze_prep_4/REPORT.md   | 241 ++++++++++++++++++++++
 twin_to_line/validation/freeze_prep_4/rehearse.py | 169 +++++++++++++++
 8 files changed, 648 insertions(+), 12 deletions(-)
```

Start verifies the USB tar and records its full SHA256 in each new visit. READY=1 remains the durable
successful-setup flag; new visits also retain `records/setup_ok.txt`. For a successful visit, the same verified
archive resumes without bootstrap or a new directory; the final line includes the reconciled final setup count
and seal status. Legacy successful visits can use their retained package checksum file. A different archive
stops with the explicit `--new-visit` command. Failed setup automatically gets a fresh visit; explicit new visits
retain old evidence. Corrupt USB bytes are rejected before changing the current context.

Resuming without a USB mount change writes nothing to the visit or active context. A remount appends the new
USB path so export can find it. Creating a new visit updates the active-context pointer and, if necessary, the
convenience launcher copies; all prior visit files and packaged code remain retained. No evidence is deleted.

## 4. Tests

Same full-suite command as above: **274 passed**, 0 skipped, **81.85 seconds**. Raw log:
`results/freeze_prep_4/tests.log` (not committed).

Nine new Mac tests execute the actual start script against a tiny fake package: no current visit; same-package
resume unsealed and sealed; failed setup; differing package then explicit new; explicit new with same package;
corrupt then repaired tar; legacy recorded checksum; invalid arguments. Resume tests compare complete visit
snapshots and the active context before/after to ensure no bootstrap or evidence writes. The Mac fixture stubs
only Linux flock; the container uses real flock and the real unchanged bootstrap.

## 5. Manifest and one-page PDF

`shasum -a 256 -c FREEZE.sha256`: **106 OK, 0 failed, exit 0**. Raw output:
`results/freeze_prep_4/manifest_check.log`. The count rose from 104 because the new decision-table tests and
container-rehearsal source were added. Existing changed-file hashes were refreshed.

The PDF was regenerated with the existing `print_runbook.py` and existing `results/freeze_prep_3/pdf_tools`.
**One Letter page**, 10.2-point body, 12.4-point leading; rendered and visually checked with no clipping.
No new tools or dependencies were installed. PDF: `monday/MONDAY_ONE_PAGE.pdf`.

## 6. Container sequence: final lines and seconds

arm64 `ubuntu:24.04`, UID/GID 1000:1000, exit 0. USB directory includes a space. Before start:

```text
$ command -v python3
(no output; exit 1; before start)
```

The harness uses a separately extracted bundled interpreter by absolute path, outside PATH. The real bootstrap
selects the bundled CPython 3.11.17; no system Python or new downloads are needed. Timings include scripted
keyboard input and checksum/IO time, not CSI's physical staging time. HOME and USB in the final lines below are
replaced with `$HOME` and `$USB` to avoid shipping machine/account paths.

| Step | Seconds | Final line |
| --- | ---: | --- |
| 1 start: new | 7.037 | `========== SETUP OK: preflight passed; context $HOME/ttl/visit_20261025T051323_LVZ35d/operator.env ==========` |
| 2 stage two setups, q | 1.334 | `========== STAGING OK: 2 setups staged; skips and retakes logged; no inference ==========` |
| 3 start again: resume | 0.443 | `========== SETUP OK: resumed $HOME/ttl/visit_20261025T051323_LVZ35d (2 staged, sealed no) ==========` |
| 3 stage resumes at setup 02 | 0.039 | `========== STAGING OK: 2 setups staged; skips and retakes logged; no inference ==========` |
| 4 seal first visit | 0.490 | `========== SEALED USB OK: $USB/sealed_wh5rejod; reset fixture before live ==========` |
| 4 start again: sealed resume | 0.445 | `========== SETUP OK: resumed $HOME/ttl/visit_20261025T051323_LVZ35d (2 staged, sealed yes) ==========` |
| 4 live after sealed resume, SIGINT | 0.576 | `========== LIVE OK: stopped cleanly; evidence $HOME/ttl/visit_20261025T051323_LVZ35d/attempts/live_20261025T051333_630uklkp; export every visit ==========` |
| 5 explicit new visit, same package | 6.859 | `========== SETUP OK: preflight passed; context $HOME/ttl/visit_20261025T051334_eD8Px9/operator.env ==========` |
| 5 seal zero staged | 0.537 | `========== SEALED USB OK: $USB/sealed_lzkoxuhg; reset fixture before live ==========` |
| 6 different tar: refuse | 0.502 | `========== STOP: current $HOME/ttl/visit_20261025T051334_eD8Px9 used package f3d6823def0d508a239e972f5e5d550641ede0e5d80f9df5a34066905fcd460a; USB has abeda300f8de420633b9c98321017e05a0953cef186324dd5ebcd78f06d82b8f -> bash start.sh --new-visit ==========` |
| 6 explicit new visit, different tar | 7.052 | `========== SETUP OK: preflight passed; context $HOME/ttl/visit_20261025T051342_m74fRG/operator.env ==========` |
| 7 corrupt tar: refuse | 0.239 | `========== STOP: USB script or package checksum failed; preserve the current visit, fix the USB and retry ==========` |
| 7 fixed tar: resume | 0.428 | `========== SETUP OK: resumed $HOME/ttl/visit_20261025T051342_m74fRG (0 staged, sealed no) ==========` |
| 8 export all three visits | 17.211 | `========== RESULTS USB OK: 3 visits verified at $USB/visit_export_20261025T051358_F8QYow; keep GB10 files until Mac verification ==========` |
| 8 independent export recheck | 3.278 | `evidence.tar: OK` |

The first visit retained two staged captures; stage resumed at setup 02 without asking for the camera again.
Resuming after sealing preserved the context byte-for-byte and reported `sealed yes`. Live accepted a newly
saved DEV frame and stopped cleanly on SIGINT. The second visit sealed zero captures with WARNING.
For the different-tar test, only outer tar padding changed; runtime and bootstrap bytes remained untouched.
The bad-checksum test preserved the current successful visit, so repairing the tar resumed that visit rather
than creating a fourth one. Every original/changed/corrupt USB test archive was retained in local history.
All three visits were exported, checked again with sha256sum, and inspected for both original staged files.

## 7. Package comparison against RC e597c4b

Built by the unchanged `gb10/build_package.py`, invoked through the existing cache-reuse wrapper, from committed
HEAD `48dbb23` into a fresh output directory. Cached runtime/wheel bytes were SHA256-verified before reuse.
Package: `ttl_gb10_48dbb23.tar`, original SHA256:
`f3d6823def0d508a239e972f5e5d550641ede0e5d80f9df5a34066905fcd460a`.
Mac and container preflight passed with exact 640/1080 state and score parity.

**All 14 live_inspect.RUNTIME_FILES match the RC byte-for-byte: 0 differences.** Bootstrap, bundled Python,
wheels and preflight images/prediction CSVs also match. Every other package change is listed below, including
changes inherited from earlier work; the list does not imply those protected files changed in this task.
The report-only commit after the tested package adds this document and changes no operator/runtime bytes.

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

Added (40):

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
twin_to_line/tests/test_start_resume.py
twin_to_line/validation/freeze_prep_3/README.md
twin_to_line/validation/freeze_prep_3/ROUND1_REPORT.md
twin_to_line/validation/freeze_prep_3/build_rehearsal.py
twin_to_line/validation/freeze_prep_3/manifest_check.log
twin_to_line/validation/freeze_prep_3/package_diff.py
twin_to_line/validation/freeze_prep_3/print_runbook.py
twin_to_line/validation/freeze_prep_3/rehearse.py
twin_to_line/validation/freeze_prep_3/round0_manifest_additions.txt
twin_to_line/validation/freeze_prep_3/round1_package_diff.json
twin_to_line/validation/freeze_prep_3/round1_preconditions.log
twin_to_line/validation/freeze_prep_3/round1_remaining_paths.txt
twin_to_line/validation/freeze_prep_3/round1_steps.tsv
twin_to_line/validation/freeze_prep_3/round1_tests.log
twin_to_line/validation/freeze_prep_3/section7.md
twin_to_line/validation/freeze_prep_3/tests.log
twin_to_line/validation/freeze_prep_4/rehearse.py
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

## 8. Not done / boundaries

No requested implementation or validation step remains undone. Real CSI hardware was not exercised; the
requested Ubuntu rehearsal was used. Docker Desktop was stopped after the rehearsal.

No protected file changed: runtime modules, bootstrap, package builder, model/thresholds/maps/warp, sealed scorer,
frozen gate, ignite_core, .gitignore and HACKATHON_HANDOFF remain unchanged from feat/twin-to-line. No runs 4–5
were opened; no YAAD Spark or Brev connection was made. Raw test/build/container logs, packages and captures
remain local under `results/freeze_prep_4/` and were not committed. This report contains no machine/account paths.
