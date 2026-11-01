# Vial Guard staging collection v2

Status: collection launcher implemented for a future real-data candidate. This
does not modify the closed 2026-10-28 model, protected result, or live-demo kit.

## Scope and safety boundary

The launcher reads newly created JPEG/PNG files from a folder populated by the
approved camera application. It does not open a camera device, change camera
settings, run inference, inspect pixels for labels, contact a network service,
or control a robot, PLC, conveyor, actuator, or product-release decision.

The operator and approved testbed supervisor remain responsible for physical
staging, safe handling, camera/fixture alignment, fresh robot placements,
hands/arm clearance, actual angle measurements, expected occupancy, and whether
the retained image corresponds to the stated setup.

## Evidence prerequisites

Official `train`, `validation`, and `protected_test` sessions require a valid
synthetic-only baseline-freeze JSON before the session directory is created.
The freeze must preserve the exact class order:

1. `pass`
2. `missing_cap`
3. `crooked_cap`
4. `damaged_vial`
5. `no_vial`

It must record model, threshold, preprocessing, policy, code, and calibrated
fixture-map SHA-256 values. Fixture S is refused unless its separately
calibrated map is present and differs from fixture G. If the freeze does not yet
exist, use `rehearsal`; rehearsal output is explicitly not official retraining
evidence.

The repository currently has only a placeholder
`synthetic_baseline_freeze.example.json`; it is not valid evidence. Never
fabricate its hashes, substitute a staging-trained candidate, or rename the
example to bypass this gate. Official collection remains blocked until the
authorized synthetic-only baseline is actually frozen.

Use independent visits and session IDs for train, validation, and protected
test. Train and validation may use one marked development storage root, but not
one physical visit. Complete and seal both roles, verify every session, export
the independent review packets, finish adjudication, and score development once
with the frozen baseline. After those operator-controlled gates pass, run
`freeze-development`. The command mechanically verifies COMPLETE sealed raw
sessions; it cannot itself prove that external review/scoring work is finished.
It permanently closes the campaign to new development captures and emits an
exact five-file, checksum-bound handoff containing campaign metadata, capture
digests, and sealed-selection hashes but no pixels. Do not initialize protected
collection early and do not transfer development session directories or images.

Protected test requires the transferred handoff, a separate non-overlapping
storage root, camera source, and custodian/account. Protected collection imports
the handoff and never needs to read, write, or lock the development root. The
development agent must not browse, copy, or summarize protected pixels, labels,
filenames, or event identifiers. The protected camera source, evidence root,
review files, and account must be inaccessible to the development agent;
different pathnames under one shared accessible account are not separate
custody.

Before collection, confirm the approved physical G01-G30 position-numbering
guide is present and agrees with the frozen G fixture map. If fixture holes are
not unambiguously identified, STOP. Do not invent an ordering. Fixture S stays
blocked until its own approved numbering and calibrated map are frozen.

## Angle and label semantics

The plan defines 90 degrees as upright relative to the tray:

- An intact, correctly capped vial whose body and aligned cap both lean from
  80–90 degrees is intended `pass`.
- `crooked_cap` requires visible improper seating or cap-to-vial misalignment.
  Absolute cap angle alone is not sufficient.
- Vial-body lean below 80 degrees has no supervised five-class mapping and is
  excluded from this plan.
- `no_vial` plus expected occupied is intended `missed`; `no_vial` plus expected
  empty is intended `waiting`. The visual class remains `no_vial` in both cases.
- Only safe, approved, visibly representative damage may be staged as
  `damaged_vial`. If none is available, skip with a reason and accept a PARTIAL
  seal; do not substitute glare, lean, or invented damage.

The launcher writes planned and actual angle fields separately. Planned values
are intent only. Measure primary-target angles contemporaneously while the
captured physical state still exists, using the approved method; retain the
event UID, position, value, uncertainty, method, measurer, and time. Transcribe
those records only into the exported measurement working copy. A later
pixel-derived value is acceptable only when the pixel-measurement method has
been validated and its uncertainty is retained. If neither source exists,
leave actual values blank and do not make angle-stratified claims.

## Recommended event plan

| Family | Train | Validation | Protected test |
| --- | ---: | ---: | ---: |
| Acceptable PASS lean ladder | 12 | 6 | 6 |
| Crooked-cap ladder | 15 | 5 | 5 |
| Missing cap | 8 | 4 | 4 |
| Approved damaged vial | 10 | 5 | 5 |
| `no_vial` MISSED/WAITING | 8 | 4 | 4 |
| Mixed/confounder | 12 | 6 | 6 |
| **Total independent events** | **65** | **30** | **30** |

One event is one freshly staged physical placement state. A retake of the same
unchanged state remains the same event. All 30 fixture positions receive intent
rows, but those correlated position crops do not become 30 independent events.
If a vial, cap, fixture, camera, or mount is moved, it is not a retake of the old
physical event. Pause and assign the changed state to a new event/session.

## Collection behavior

At session creation the launcher:

- freezes the role, fixture, session ID, camera folder/profile, resolution,
  operator, plan, and applicable baseline commitment;
- snapshots and ignores every pre-existing camera file;
- creates all 30 intent rows for each event with `usable=false` and
  `review_state=pending_independent_review`;
- locks 640x480 or 1920x1080 for the entire session;
- creates append-only capture, skip, and refusal logs.

For each event it displays the exact physical setup, then waits for the human to
type uppercase `READY`. This confirmation means a fresh placement was made,
the exact displayed setup is complete, motion stopped, hands/arm/tools cleared,
the approved camera/mount/settings remain unchanged, and exactly one new camera
image was saved. The
collector waits for a stable regular file, structurally validates JPEG/PNG
bytes, verifies resolution, exclusively writes a retained copy, fsyncs it, and
checks its SHA-256. Supported encodings are baseline Huffman JPEG (SOF0) with a
complete decodable coefficient stream and non-interlaced PNG with an exact
raster length and valid scanline filters. Progressive/arithmetic/multiscan JPEG
and interlaced PNG fail closed. The collector never changes the source camera
file. Structural decoding does not establish focus, visibility, or semantic
class; human review remains mandatory.

The human then confirms whether the retained frame corresponds to the staged
setup. Rejected frames and every unchanged-state retake remain preserved. More
than one new candidate camera file is ambiguous and refused. A duplicate digest
is refused as a new event. `q` pauses safely; the same session can be resumed.

## Sealing and review

A COMPLETE seal requires one selected capture for every planned event and no
skips. A collection with skips or unfinished events can only receive an
explicit PARTIAL seal. A zero-capture session cannot be sealed.

The seal creates:

- `sealed/evidence.tar`
- `sealed/evidence.tar.sha256`
- a selected/earlier-attempt inventory
- a tamper-evident session inventory
- an independent-review template covering all 30 positions for each selected
  event

The sealed review template deliberately leaves actual visible class,
confidence, and reviewer fields empty. Run `export-review` to create derived
working files outside the immutable session. The export contains two separately
shuffled blinded worksheets that omit intent, expected occupancy, sequence
context, and the other review. Give them to different authorized reviewers
under separate access. Only after both reviews are locked may a coordinator
open the adjudication worksheet and resolve disagreements. Intended setup is
never silently promoted to visual ground truth. Raw captures, sealed templates,
and the sealed archive must remain unchanged.

Keep the `.run` on the USB but use local compute-module paths for the active
camera source and evidence output. Do not collect into the kit directory. Copy
sealed output to approved storage only after verification, then verify its
SHA-256 again.

After development captures are adjudicated, score train and validation once
with the frozen baseline before any real-data adaptation. Do not open or score
the protected set until the final candidate, thresholds, preprocessing, maps,
code, and shared decision policy are frozen.

The custody transition is:

1. finish, seal, verify, export review packets, adjudicate, and baseline-score
   development;
2. run `freeze-development --development-root DEV --out-dir NEW_HANDOFF` under
   the development account;
3. transfer only `development_campaign.json`,
   `development_capture_sha256_reservations.csv`,
   `development_sealed_sessions.json`,
   `protected_handoff_receipt.json`, and `SHA256SUMS`;
4. verify the handoff, then run protected collection under the custodian with
   `--protected-handoff NEW_HANDOFF`.

## Stop conditions

Pause the collection for camera/mount/settings drift, unsupported resolution,
fixture-map uncertainty, hands/arm occlusion, blur or incomplete writes, missing
occupancy/sequence context, unsafe damage handling, clock/session-ID failure,
protected-root exposure, or inability to establish the actual visible state.
Preserve existing files, record the reason, and start a new session after any
camera geometry change.

