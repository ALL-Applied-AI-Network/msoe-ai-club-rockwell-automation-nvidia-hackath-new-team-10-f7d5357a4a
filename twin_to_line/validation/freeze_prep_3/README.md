# Freeze-prep 3 validation

Active checkout: `~/code/crusaders`, branch `freeze-prep-3`. Local commits only; no tag or push.
The Sunday freeze fields remain placeholders. These are rehearsal records, not release attestations.

Round 1 replaces the initial operator workflow: automatic USB/package discovery, camera-folder reuse,
retakes with all attempts retained, duplicate/size refusal, zero-capture seals, and Enter for continuous live.
The printable operator sheet is `../../monday/MONDAY_ONE_PAGE.pdf`: one Letter page, generated from `section7.md`.
`print_runbook.py` is development-only PDF tooling; it is never imported by an operator script.

`build_rehearsal.py` invokes the actual builder, replacing only downloads with SHA256-verified cached RC runtime
and wheel files already in this checkout. `package_diff.py` compares every outer-package and nested code-archive
file, and stops if a live runtime file differs from the RC. `rehearse.py` drives all operator commands as fresh
processes with terminal-backed input in non-root arm64 Ubuntu; start receives no input.

Raw round-0 container and package-build logs were moved to `results/freeze_prep_3/round0/` because they contain
machine/account paths (or hostname strings). Their entries were dropped from FREEZE.sha256. Raw round-1 logs
and generated packages stay in `results/freeze_prep_3/round1/`; they are not shipped in the code archive.
Only reports without machine/account identifiers belong here. `manifest_check.log` and `tests.log` preserve
safe round-0 evidence; current round-1 reports use the `round1_` prefix.

The initial rehearsal processed three DEV stills: two scored events (30 rows each) and one arm refusal,
hence 60 prediction rows. This was not missing output. Retakes and staged captures are never scored by these
collection scripts. Monday labels stay on paper and go to the Mac for adjudication before sealed evaluation.

The earlier scripts were tested at `a5b38b0`, with 259 full-suite tests passing. Their raw evidence remains
local. `HACKATHON_HANDOFF.md` records the repository-move identity and the original 247-test precondition;
the inherited part-2 evidence has 245 full-suite tests plus 2 runbook tests. No missing historical transcript
is presented as a new execution. The frozen gate retains its historical path because it is protected from edits.
