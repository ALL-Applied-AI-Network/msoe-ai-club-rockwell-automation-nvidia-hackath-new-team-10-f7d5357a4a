VIAL GUARD STAGING COLLECTION KIT (2026-10-29)

This small offline .run file launches the inference-free staging image
collector. It uses only Ubuntu/Linux Python 3.10+ standard-library modules. It
does not contain or run a model and does not control the camera, robot, PLC,
conveyor, actuator, or product-release decision.

COPY AND VERIFY

1. Copy this entire directory to the USB drive.
2. On the compute module, open a terminal in this directory and verify:

     sha256sum -c PACKAGE_SHA256SUMS

3. Review STAGING_COLLECTION_V2.md, CODEX_TERMINAL_PROMPT.txt, and
   ALL_STAGING_EVENTS.md.

ALL_STAGING_EVENTS.md is the single read-only fixture-G reference containing
every shared rehearsal, training, and validation tray condition plus the public
aggregate for all protected-test events. It never contains the session's exact
target slots. At collection time, the terminal display and immutable session
plan.csv are authoritative. The collector generates exact protected-test cards
only inside the protected custodian's session after its secret plan is
initialized.

Keep the launchable on the USB, but use a local compute-module camera folder
and a local evidence disk while collecting. Do not use the kit directory as
either input or output. After sealing and verification, copy the sealed
evidence to approved storage and verify its SHA-256 again.

CURRENT OFFICIAL-COLLECTION BLOCKER

This repository currently contains only
`synthetic_baseline_freeze.example.json`, whose values are placeholders. It is
not an authentic baseline freeze. Do not edit it with invented hashes, do not
substitute a model trained on prior staging data, and do not use it for an
official role. Until an authorized synthetic-only baseline freeze is produced,
run `rehearsal` only; train, validation, and protected_test must STOP closed.

REHEARSAL FIRST

Run a six-event rehearsal before official collection:

     bash vial_guard_staging_collector_2026-10-29.run collect \
       --role rehearsal \
       --camera-dir /path/to/camera/images \
       --output-root /path/to/rehearsal-output \
       --fixture G \
       --expected-resolution 1920x1080 \
       --operator-id YOUR_NAME \
       --camera-id CAPSEAL_CAMERA_PROFILE

If arguments are omitted, the launchable enters an interactive wizard.

OFFICIAL TRAIN VISIT

     bash vial_guard_staging_collector_2026-10-29.run collect \
       --role train \
       --camera-dir /path/to/camera/images \
       --output-root /path/to/development-evidence \
       --fixture G \
       --expected-resolution 1920x1080 \
       --operator-id YOUR_NAME \
       --camera-id CAPSEAL_CAMERA_PROFILE \
       --baseline-freeze /secure/evidence/synthetic_baseline_freeze.json

Use a new physical visit and session for validation, changing only `--role` to
`validation`. Train and validation are both development data.

FREEZE DEVELOPMENT, THEN START THE PROTECTED VISIT

Complete, seal, and verify every train and validation session first. Export the
review packets, finish independent adjudication, and score development once
with the frozen baseline. Those review/scoring gates are operator-controlled;
the freeze command itself can mechanically verify only COMPLETE sealed raw
sessions. After all gates pass, use the development account to irreversibly
close development collection and emit a pixel-free metadata handoff:

     bash vial_guard_staging_collector_2026-10-29.run freeze-development \
       --development-root /path/to/development-evidence \
       --out-dir /path/to/new-protected-handoff

This command refuses PARTIAL/unsealed development sessions and requires both
train and validation. Once it succeeds, that development campaign cannot accept
more captures. Copy only the five-file handoff directory to the protected
custodian and verify its `SHA256SUMS`; it contains hashes, sealed-selection
commitments, and split commitments—not images. Its exact members are
`development_campaign.json`,
`development_capture_sha256_reservations.csv`,
`development_sealed_sessions.json`,
`protected_handoff_receipt.json`, and `SHA256SUMS`.

Run protected collection under a separate custodian/account with a separate
non-overlapping storage root and camera-source folder. The protected account
does not need access to the development root. Its source folder, output, and
review files must be inaccessible to the development Codex and development
operators—not merely different pathnames:

     bash vial_guard_staging_collector_2026-10-29.run collect \
       --role protected_test \
       --camera-dir /path/to/camera/images \
       --output-root /secure/protected-evidence \
       --protected-handoff /secure/transfer/new-protected-handoff \
       --fixture G \
       --expected-resolution 1920x1080 \
       --operator-id CUSTODIAN_NAME \
       --camera-id CAPSEAL_CAMERA_PROFILE \
       --baseline-freeze /secure/evidence/synthetic_baseline_freeze.json \
       --protected-custodian-ack PROTECTED-CUSTODIAN

PAUSE / RESUME / VERIFY

Type `q` at an event prompt to pause safely. Resume with:

     bash vial_guard_staging_collector_2026-10-29.run collect \
       --resume /path/to/output/SESSION_DIRECTORY

Verify a session or sealed archive:

     bash vial_guard_staging_collector_2026-10-29.run verify \
       --session-dir /path/to/output/SESSION_DIRECTORY

Seal from a separate invocation:

     bash vial_guard_staging_collector_2026-10-29.run seal \
       --session-dir /path/to/output/SESSION_DIRECTORY

Incomplete sessions are refused unless `--allow-partial` is supplied, in which
case the seal is visibly marked PARTIAL. A PARTIAL collection does not satisfy
the complete scenario plan.

After sealing, create derived review worksheets outside the immutable session:

     bash vial_guard_staging_collector_2026-10-29.run export-review \
       --session-dir /path/to/output/SESSION_DIRECTORY \
       --out-dir /path/to/new-empty-review-directory

Never edit the sealed originals. Give the two blinded reviewer files to
different reviewers under separate access, then adjudicate only after both
reviews are locked.

IMPORTANT

- Existing camera files are ignored when a session starts. The approved camera
  application must write a new JPEG or PNG for every capture attempt.
- Confirm the approved physical G01-G30 numbering guide is present and matches
  the frozen fixture map before staging. STOP on any numbering ambiguity; do
  not invent a row/column order.
- The source folder should contain only the full-fixture camera stills. Do not
  put phone/side photos into that folder. If more than one new image appears for
  one attempt, the collector refuses the ambiguous capture.
- Configure the approved camera application for baseline Huffman JPEG (SOF0) or
  non-interlaced PNG. Progressive/arithmetic/multiscan JPEG and interlaced PNG
  are refused so the offline standard-library validator can verify the complete
  raster/coefficient structure before retention.
- Type READY only after the exact displayed setup is complete, it is a fresh
  placement, motion has stopped, hands/arm/tools are clear, the approved
  camera/mount/settings are unchanged, and exactly one new image was saved.
- A retake is allowed only while the physical arrangement is unchanged. If any
  vial or cap is moved or restaged, pause and use a new physical event/session;
  do not relabel the moved state as a retake of the old event.
- Record primary-target angles at capture time with the approved measurement
  method and uncertainty. A later pixel measurement is acceptable only if its
  method is validated and documented. Otherwise leave actual angles blank and
  make no angle-stratified performance claim.
- The launcher records staging intent, not ground truth. Two independent human
  reviewers must label actual visible content afterward.
- Fixture S remains unavailable until its own calibrated map is present in the
  baseline freeze and differs from fixture G.
- The protected set must never enter a development or YAAD training bundle.
- Never initialize protected collection early. Finish and seal train and
  validation, verify them, then run `freeze-development` exactly once. The
  successful freeze permanently closes additional development capture for that
  campaign.

