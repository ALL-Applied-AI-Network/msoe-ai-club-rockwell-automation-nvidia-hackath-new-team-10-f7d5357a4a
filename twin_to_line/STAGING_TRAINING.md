# Five-class staging training on YAAD

This is a new real-staging training path. It does not modify the frozen
four-class synthetic model or its inspector.

## Data-use decision and metric scope

D036 historically designated this collection as a sealed final test. On
2026-10-28, the project owner explicitly instructed that the selected staging
images be used for development and that the staging plan's intended states be
the scoring reference. The derived split lock records that policy change; the
historical decision and original evidence repository remain unchanged.

One physical setup event remains indivisible: all 30 positions and its selected
image stay in one split. The leakage-safe default also keeps every event from a
collection visit together. This collection has only one visit, so the current
protocol uses an explicit, acknowledged event-level override: 12 train events,
5 validation events and 5 protected test events. The test events are chosen and
locked before training and their pixels never enter the training bundle.

Because train, validation and test share a visit, camera, lighting and fixture,
the result is **within-visit held-out setup-event accuracy**, not independent
deployment or new-visit accuracy. The labels have
`review_state=protocol_reference`: they measure agreement with intended staging
states and do not claim that image contents were independently adjudicated.

The exact class order is always:

1. `pass`
2. `missing_cap`
3. `crooked_cap`
4. `damaged_vial`
5. `no_vial`

If development data contain zero `damaged_vial` examples, the fifth-output
model preserves its index but records it as unsupported and is not considered
ready for the full five-class claim. A high unsupported damage probability is
returned as `unknown` / `UNSURE`; it is never reported as detected damage.

## Files

- `build_staging_manifest.py`: expands only human-confirmed captures that match
  the predeclared Tuesday setup plan. It never reads pixels for labels.
- `staging/manifest_template.csv`: normalized full-manifest schema for direct
  labeling of any setup that deviated from the plan.
- `prepare_staging_splits.py`: validates every source hash and creates locked,
  event-atomic, visit-isolated train, validation and protected-test manifests.
- `staging_dataset.py`: guarded data loader using the runtime's 1080-to-640
  warp, real registration offset and fixed target crops.
- `train_resnet5.py`: five-output ResNet-18 trainer. It can open only TRAIN and
  VALIDATION manifests.
- `resnet5_inspector.py`: model adapter that conservatively escalates an
  unsupported class.
- `make_yaad_training_bundle.py`: builds a complete YAAD input from an explicit
  training-code allowlist plus TRAIN/VALIDATION data. Staging reviews, event
  lists, test pixels and the test manifest are excluded.
- `yaad_train_resnet5.sh`: container entry point and artifact checks.
- `evaluate_resnet5_test.py`: separate post-training evaluator that verifies
  frozen model/threshold/split hashes before opening the protected test.
- `make_yaad_test_bundle.py`: builds the separate frozen-model, test-only YAAD
  input and rejects development manifests or extra files.
- `yaad_evaluate_resnet5.sh`: one-time protected-evaluation container entry
  point and artifact checks.
- `staging/RESULT_2026-10-28.md`: recorded YAAD job IDs, hashes, results, and
  deployment limitations for the completed run.

Only fixture G has a calibrated map in this checkout. Any S row fails closed
until a calibrated `position_map_S.json` is supplied explicitly.

## 1. Review the selected captures

For independently adjudicated labels, copy `staging/capture_review_template.csv`
and require `confirmed_matches_plan=true`, an annotator, confidence and
`review_state=adjudicated`. If a setup deviated, write its 30 authoritative rows
directly using `manifest_template.csv`.

The current owner-directed intended-state protocol instead uses
`staging/capture_review_2026-10-27_protocol_reference.csv` and the explicit
`--allow-intended-plan-reference` flag. It uses only the sealed final selection;
the superseded originals for setups 02 and 03 are excluded.

Valid designations are:

- `development`: eligible for TRAIN or VALIDATION, and possibly a newly locked
  TEST allocation if the requested ratio needs one.
- `protected_test`: immutable TEST assignment.

Build the normalized labels:

```bash
cd twin_to_line
../.venv/bin/python build_staging_manifest.py \
  --review staging/capture_review_2026-10-27_protocol_reference.csv \
  --image-root ../data/capseal-staging-dataset-2026-10-27/data/images \
  --out ../data/capseal-training-derived-2026-10-28/position_manifest.csv \
  --allow-intended-plan-reference
```

The command refuses missing images, symlinks, path traversal, duplicate events,
unconfirmed deviations, noncanonical classes and incomplete 30-position data.

## 2. Freeze the event-atomic split

Create the split once in a new destination. Reusing an existing output path is
an error.

```bash
../.venv/bin/python prepare_staging_splits.py \
  --manifest ../data/capseal-training-derived-2026-10-28/position_manifest.csv \
  --image-root ../data/capseal-staging-dataset-2026-10-27/data/images \
  --out ../data/capseal-training-derived-2026-10-28/splits_v1 \
  --seed 20261028 \
  --validation-fraction 0.20 \
  --test-fraction 0.20 \
  --validation-events staging/validation_events_2026-10-27.txt \
  --test-events staging/test_events_2026-10-27.txt \
  --split-unit event \
  --acknowledge-within-visit-evaluation \
  --leakage-justification "Owner-directed staging development; report only within-visit held-out event metrics" \
  --source-repo-url https://github.com/vang838/capseal-staging-dataset-2026-10-27.git \
  --source-git-commit "$(git -C ../data/capseal-staging-dataset-2026-10-27 rev-parse HEAD)"
```

The output contains (and the lock records both session and event membership):

```text
splits_v1/
  train_manifest.csv
  validation_manifest.csv
  split_lock.json
  protected_test/test_manifest.csv   # read-only; final evaluation only
```

Inspect `split_lock.json` before training. It records the exact event IDs,
class counts, hashes, within-visit scope and warning. Never change event
membership after seeing validation or test results.

## 3. Build a test-free, allowlisted YAAD job bundle

```bash
../.venv/bin/python make_yaad_training_bundle.py \
  --split-root ../data/capseal-training-derived-2026-10-28/splits_v1 \
  --image-root ../data/capseal-staging-dataset-2026-10-27/data/images \
  --out ../data/capseal-yaad-input-2026-10-28-v3
```

This copies only the runtime files named in `TRAINING_CODE_ALLOWLIST` and only
TRAIN and VALIDATION images. The protected test event IDs and lock digest remain
in `split_lock.json`, because the trainer needs the lock to authenticate the two
development manifests. No staging review CSV, standalone test-event list, test
manifest, or test pixels enter the training job. The output is complete; do not
copy the repository into it afterward.

```text
capseal-yaad-input/
  team_repo/
    twin_to_line/
      tests/test_yaad_training_preflight.py
      train_resnet5.py
      yaad_train_resnet5.sh
      ...explicitly allowlisted runtime files only...
  dataset/
    images/
    splits/
```

## 4. Stage the input through SSH

Connect the CSI VPN first. Use a new remote folder so an earlier job input is
never overwritten:

```bash
ssh yt-codecrusaders6286@172.30.102.94 \
  'mkdir /workspace/capseal-train-20261028-v3'
tar -C /path/to/capseal-yaad-input -cf - team_repo dataset | \
  ssh yt-codecrusaders6286@172.30.102.94 \
  'tar -xf - -C /workspace/capseal-train-20261028-v3'
```

If SSH reports `Permission denied (publickey)`, register this computer's
`~/.ssh/id_ed25519.pub` in YAAD before retrying.

## 5. Submit one Spark through YAAD

SSH stages `/workspace`; the pooled GPU job is submitted through the YAAD
scheduler, not as a long-running interactive SSH process. Use:

```json
{
  "name": "capseal-five-output-20261028-v3",
  "input_path": "capseal-train-20261028-v3",
  "image": "nvcr.io/nvidia/pytorch:26.09-py3",
  "gpu": true,
  "nodes": 1,
  "memory_gb": 16,
  "cpus": 4,
  "max_minutes": 60,
  "command": "bash /job/in/team_repo/twin_to_line/yaad_train_resnet5.sh"
}
```

Only `/job/out` is retained. The entry point runs the focused boundary and
adapter preflight first, checks CUDA, trains with `--device cuda`, and saves the
model, validation-only threshold, validation predictions, provenance and
SHA-256 inventory.

Never run final-test evaluation inside this training job. Thresholds and model
parameters come from TRAIN/VALIDATION only.

## 6. Evaluate the protected events once

After training completes and its model and validation threshold hashes are
frozen, build a separate test-only input:

```bash
../.venv/bin/python make_yaad_test_bundle.py \
  --split-root ../data/capseal-training-derived-2026-10-28/splits_v1 \
  --image-root ../data/capseal-staging-dataset-2026-10-27/data/images \
  --model-dir ../data/yaad-results/3c875c39-3bb5-4f17-bb07-46e30679c520/out/resnet18_staging5 \
  --out ../data/capseal-yaad-test-input-2026-10-28-v1
```

Stage that directory as `capseal-protected-test-20261028-v1`, then submit
exactly one job whose command is:

```bash
bash /job/in/team_repo/twin_to_line/yaad_evaluate_resnet5.sh
```

Save `test_predictions.csv` and `test_metrics.json`. Report visual confusion,
per-class recall, missed-defect recall, missed-versus-waiting state accuracy and
UNSURE rate. Overall position accuracy alone is not sufficient because 549 of
660 planned crops are `pass`. The completed run and its single evaluation job
are recorded in `staging/RESULT_2026-10-28.md`.

This collection contains no `damaged_vial` examples. The model keeps the exact
five-output taxonomy, but `damaged_vial` is recorded as unsupported and resolves
to UNKNOWN/UNSURE. Do not describe it as a validated five-class model until
damage evidence exists in train, validation and test.
