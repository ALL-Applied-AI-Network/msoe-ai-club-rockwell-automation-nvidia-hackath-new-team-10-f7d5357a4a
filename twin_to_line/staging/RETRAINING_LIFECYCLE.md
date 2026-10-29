# Synthetic-to-real staging retraining lifecycle

Status: metadata evidence gate implemented and tested. No candidate was trained,
no closed staging5 assignment/result was changed, and no protected image was
opened by this work.

## Authorization and evidence meaning

The project owner has clarified that newly collected staging images may be used
to train and validate a real-adapted model. The required order is:

1. Freeze the synthetic-trained baseline, its threshold, preprocessing,
   fixture maps, decision policy, and code hashes.
2. Run that frozen baseline once on the staging development scenarios. Preserve
   complete per-position outputs as synthetic-to-real transfer evidence.
3. Only after baseline scoring is complete may authorized staging examples
   influence fitting, augmentation, checkpoint selection, threshold selection,
   or design choices.
4. From that moment, every influencing example is development evidence for the
   new candidate. It cannot be final-test evidence for that candidate.
5. Seal a different event/session/visit before adaptation starts. Keep its
   pixels, labels, and identifiers out of all training and selection inputs.
   Evaluate it once only after the candidate is frozen.

The staging plan records the intended scenario. A review record must also state
whether the visible image was adjudicated or is only a protocol reference; do
not silently present intended state as independent visual ground truth.

## Version boundaries

The 2026-10-28 staging5 model, split lock, protected result, and USB kit remain
closed historical evidence. Do not edit or overwrite them.

The old 2026-10-27 visit may be explicitly used as development data for a new
candidate, including events that previously appeared in the old protected
result, but doing so does not erase or relabel the old result. Those examples
are no longer final-test evidence for the new candidate. A preferred future
layout is:

- old visit: training;
- fresh visit: validation and model/threshold selection;
- another fresh, separately sealed visit: protected final evaluation.

Each physical placement event remains indivisible. Retakes and captures sharing
one staged physical state stay in one event and one split. Sessions/visits must
not cross training, validation, and protected partitions.

## Metadata gate

`prepare_retraining_evidence.py` checks the boundary before training. It reads
manifest metadata with image verification disabled, so it neither hashes nor
decodes protected pixels. It requires:

- exact class order: `pass`, `missing_cap`, `crooked_cap`, `damaged_vial`,
  `no_vial`;
- complete 30-position events with recorded lowercase SHA-256 values;
- development-only train and validation manifests;
- a protected-test manifest from a disjoint event, visit, source ID, and raw
  digest;
- a synthetic-only baseline freeze inventory;
- an authenticated baseline score receipt and exact baseline prediction
  coverage for every train/validation target, with scoring completed before
  adaptation;
- an explicit staging-data authorization record whose `allowed_uses` is limited
  to `baseline_scoring`, `train`, and `validation` and includes both development
  uses;
- a calibrated map hash for every represented fixture, with fixture S using a
  different map from fixture G.

All manifest capture timestamps are parsed with required UTC offsets and
normalized before comparison. The synthetic freeze must precede every
development and protected capture, with this strict chronology:

```text
synthetic freeze < each development capture < its baseline prediction
each baseline prediction <= score-receipt completion < development start
synthetic freeze < each protected capture <= protected seal < development start
```

The score receipt also attests the frozen decision-policy version and hash.
Predictions must agree with deterministic output semantics: `unknown` maps to
`UNSURE`, a supported defect maps to `DEFECT`, `pass` maps to `PASS`, and
`no_vial` maps to `MISSED` or `WAITING` from expected occupancy. Supported
defects must meet their applicable threshold; `pass` and `no_vial` must remain
below it. Duplicate raw digests cannot represent different physical events.

Example input shapes are in:

- `synthetic_baseline_freeze.example.json`
- `baseline_score_receipt.example.json`
- `retraining_authorization.example.json`

After replacing example placeholders with actual evidence, run from
`twin_to_line/`:

```bash
../.venv/bin/python prepare_retraining_evidence.py \
  --candidate-id staging5-real-v2 \
  --baseline-freeze /secure/evidence/synthetic_baseline_freeze.json \
  --baseline-predictions /secure/evidence/baseline_predictions.csv \
  --baseline-score-receipt /secure/evidence/baseline_score_receipt.json \
  --authorization /secure/evidence/staging_training_authorization.json \
  --train-manifest /secure/manifests/train_manifest.csv \
  --validation-manifest /secure/manifests/validation_manifest.csv \
  --protected-manifest /secure/protected/test_manifest.csv \
  --development-start-utc 2026-11-10T15:00:00Z \
  --protected-sealed-at-utc 2026-11-10T14:00:00Z \
  --development-lock-out /jobs/staging5-real-v2/development_lock.json \
  --protected-lock-out /secure/protected/staging5-real-v2/protected_reservation_lock.json
```

Only `development_lock.json`, the train manifest, and the validation manifest
may accompany the training job. The development lock contains only the SHA-256
commitment to the private reservation lock; it contains no protected event,
session, source, path, or label identifiers. The reservation lock and protected
manifest stay on the evaluation side.

The development lock retains the authorization-record SHA-256, normalized
authorization timestamp, authorization boolean, and exact strict-allowlist
uses. Free-form authority, dataset-scope, and statement text stays only in the
separately retained authorization record, so it cannot carry protected
identifiers into a training job.

Both locks are staged before publication. If either publication fails, the
other is rolled back; an immutable orphaned private lock is not left behind.

The baseline predictions CSV must contain the repository's required audit
fields for every development target:

```text
source_image_id,fixture_id,position_id,expected_occupancy,
predicted_visual_class,model_score,applicable_threshold,
inspection_decision,reason_code,inspection_timestamp_utc,event_uid
```

## Existing staging-tool audit

The existing `staging_dataset.py` and `prepare_staging_splits.py` already:

- enforce the exact taxonomy and G/S position naming;
- require all 30 positions per event;
- keep events atomic and default to session/visit isolation;
- prevent source-ID and raw-digest leakage across splits;
- retain source-image hashes and reject protected rows in the training crop
  adapter;
- require a fixture-specific map before fixture-S crops can be opened.

They must remain unchanged for the closed staging5 provenance. Their existing
`split_lock.json` includes protected event/session identifiers and class counts,
and the current YAAD training-bundle builder copies that lock. Therefore that
old lock is not the training-side contract for the new lifecycle. A future
trainer/bundle must explicitly consume the new development-only lock and refuse
private protected artifacts.

`build_staging_manifest.py` is also specific to the 22-setup, fixture-G Tuesday
plan. A future fixture-S collection needs its own calibrated map and direct
reviewed manifest or a separately validated plan expander; never reuse G
coordinates.

## Remaining gates before training

- Collect and adjudicate a fresh validation visit and a different protected
  visit. Neither exists merely because the metadata validator exists.
- Hash raw captures during ingestion and record those hashes before sealing.
  This gate intentionally trusts recorded manifest hashes so it does not touch
  protected image bytes.
- Make the future trainer and YAAD bundler authenticate
  `development_lock.json` and fail if any protected manifest, lock, label,
  identifier, or pixel is present.
- Keep deterministic occupancy, placement, sequence, and completion rules
  separate from visual fitting.
- Keep the candidate shadow/evaluation only; do not add PLC, robot, actuator,
  automatic product-release, or language-model override behavior.
- Freeze the candidate, threshold, preprocessing, maps, code, and shared
  decision policy before the private evaluation side opens protected pixels.

Focused validation command:

```bash
../.venv/bin/pytest -q tests/test_retraining_evidence.py
```

Expected current result: `23 passed`.
