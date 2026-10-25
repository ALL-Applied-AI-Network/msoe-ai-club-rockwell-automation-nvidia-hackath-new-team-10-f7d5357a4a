# Freeze-prep 3 validation — Oct 24, 2026

Active checkout: `~/code/crusaders`, branch `freeze-prep-3`. Local commits only; no tag or push.
The Sunday freeze fields remain placeholders. This is rehearsal evidence, not a release attestation.

- Full suite: **259 passed**, 0 skipped, 76.04 s; exact command and result in `tests.log`.
- `container_transcript.log`: all five scripts in `ubuntu:24.04`, Linux aarch64, UID/GID **1000:1000**.
  Docker inspect: `platform=linux user=1000:1000 image=ubuntu:24.04 exit=0 running=false`.
- Three DEV captures copied without source changes; setup 01 skipped; staged setups 00/02/03 reconcile by SHA256.
- USB seal verified; an intentionally unavailable USB produces an explicit verified disk seal and permits live.
- A staged-folder symlink is refused before creating an attempt. Live reads three DEV stills, produces 60 prediction
  rows (two events; one arm refusal), then exits on Ctrl-C. Restart creates a distinct OUT and ignores existing files.
- A deliberately corrupt package checksum produces a second failed visit. Python-independent export includes
  both visits, records and every attempt. Two exports produce different archive destinations; both verify SHA256.
- Every command runs in a fresh process with terminal-backed keyboard input. The shared context survives them.
- Docker Desktop was started for this rehearsal and quit after the container exited successfully.

`build_rehearsal.py` invokes the actual builder, replacing only downloads with SHA256-verified cached RC runtime
and wheel files already in this checkout. The builder copies committed scripts from `code.tar` to the USB root,
with `monday.sha256`. Final tested runtime/script commit: `a5b38b0b82295a0660f38bcaedc7669f4abe4790`.
`package_build_final.log` records the rehearsal tar SHA256 and Mac preflight. Ubuntu bootstrap took 6 seconds;
640 and 1080 preflight comparisons both have 0 state differences and maximum score difference 0.

The first container harness used a pipe for Bash `read -p` (which only prompts on a terminal); the corrected
harness uses a PTY. The first operator launch then exposed a helper named `operator.py` shadowing Python's
standard library. The helper was renamed `monday_ops.py`, isolated Python launches added, and a fresh-process
regression added. Only the final successful transcript is the validation reference.

Raw rehearsal packages, fake camera captures and exported archives remain local in `scratch/` and `rehearsal/`;
none are committed. The 3 RC DEV inputs are referenced by hash in `FREEZE.sha256`, not added to Git.
`manifest_check.log` contains the complete final `shasum -a 256 -c FREEZE.sha256` output and is excluded from
that manifest to avoid a cycle. `section7.md` is the exact new §7, also present in `LIVE_RUNBOOK.md`; the old
manual blocks remain Appendix A.

Active command paths were changed to `~/code/crusaders`. Archive/symlink identities in the handoff remain literal;
`GATE_v2.0.2.md` retains its historical path because the frozen gate was explicitly protected from edits.
No live/scorer logic, thresholds, maps, warp, gate, dose code or holdout data was changed or evaluated.
