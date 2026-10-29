---
name: staging-real-retraining
description: Improve the five-class vial inspector through synthetic pretraining and authorized real staging-data retraining while preserving event-level evidence boundaries, deterministic safety rules, and a fresh untouched evaluation set.
---

# Staging real-data retraining

Use this workflow when improving the vial inspector with staging scenarios and
newly collected real staging images.

## Evidence lifecycle

1. Freeze and identify the synthetic-trained baseline before opening a new
   staging collection. Record its model, thresholds, preprocessing, maps, code,
   and decision-policy hashes.
2. Score the newly collected staging scenarios once with that baseline. Preserve
   those predictions as baseline-transfer evidence.
3. Preserve raw captures unchanged. Derive labels and manifests separately, keep
   placement events atomic, and record the intended scenario plus independently
   reviewed visible content.
4. Authorized staging examples may then enter a new model's training or
   validation sets. Once an example influences fitting, augmentation,
   checkpoint selection, thresholds, prompts, or design choices, it is
   development data for that model version and is not its final test evidence.
5. Before retraining, seal a different event/session/visit as the new model's
   protected test. Keep its pixels, labels, and identifiers out of all training
   and selection inputs. Evaluate it once only after the candidate is frozen.

An older protected set may be deliberately retired and reclassified as
development data for a later model version. Preserve its original result and
role in history, record the reclassification, and seal a fresh protected set
before using it. It can no longer support a final-test claim for the later
version.

Do not overwrite the staging5 baseline, its closed protected result, or its USB
kit. Improvements create a separately versioned candidate and release record.

## Model and rule invariants

- Preserve the exact class order: `pass`, `missing_cap`, `crooked_cap`,
  `damaged_vial`, `no_vial`.
- Use fixed fixture coordinates and target-aware crops. Fixture S requires its
  own calibrated map and evidence; never reuse fixture-G coordinates.
- Keep visual inference separate from deterministic occupancy, placement,
  sequence, and order-complete rules.
- Trailing empty positions remain `WAITING` unless event/sequence context
  supports `MISSED`.
- Unsupported, ambiguous, invalid, low-quality, or out-of-distribution evidence
  resolves conservatively to `UNKNOWN`/`UNSURE`, never automatic `PASS`.
- A language model may launch, inspect, and summarize the system but may not
  override deterministic decisions.
- Remain shadow/evaluation only. Do not add PLC, robot, actuator, or automatic
  product-release output.

## Implementation gates

- Use one shared, versioned score-to-visual decision policy in training,
  validation, protected evaluation, export verification, and every runtime.
- Add automated tests for each acceptance criterion, including decision-policy
  boundary cases, split isolation, preprocessing parity, required audit fields,
  ONNX/runtime equivalence, and fail-closed input faults.
- Select augmentations, checkpoints, seeds, and thresholds from training and
  validation only. Keep validation/test augmentation disabled.
- Report independent event counts rather than treating 30 correlated crops from
  one frame as 30 independent defect trials.
- Freeze hashes before protected evaluation. If the protected gate fails, retain
  that result. Either leave its data protected, or explicitly retire it into a
  later development pool and use a fresh future protected set; never keep both
  roles for the same model version.

## Required evidence

For every inspected target, retain the fields required by the repository
`AGENTS.md`: source image, fixture and position, expected occupancy, visual
class, score and threshold, deterministic decision, reason code, and timestamp
or placement-event identifier.

Release reports must include per-class defect-to-PASS escapes, conservative
non-PASS intervention recall, false alarms, `UNSURE` rate, `MISSED`/`WAITING`
accuracy, complete-event results, latency/coverage, data provenance, hashes, and
known limitations. Overall position accuracy is supporting evidence, not the
primary safety metric.
