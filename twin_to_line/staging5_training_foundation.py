"""Deterministic, data-independent foundations for five-class v2 training.

This module deliberately does not read manifests, images, split locks, model
artifacts, or protected-test evidence.  Callers must supply already-validated
``ManifestRow`` objects and in-memory crops.  The additional checks here form a
second boundary before any future trainer uses the rows.

Two assumptions are explicit:

* a physical placement event is the smallest independent sampling unit; rows
  from the same event are correlated and must not be mistaken for independent
  trials;
* a non-identity augmentation is only a *candidate* transform.  This module
  cannot prove that it preserves a visual label.  Its immutable policy must be
  justified by a development-only evidence attestation before it can run.  It
  must not rotate or flip a vial because those operations can directly change
  five-class label semantics.  The default policy is identity.

The primitives are not wired into the frozen staging5 trainer yet.  A future
versioned trainer can consume the returned schedule and policy fingerprint
without changing or re-evaluating the existing model.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import cv2
import numpy as np

from staging_dataset import (
    CANONICAL_CLASSES,
    ManifestError,
    ManifestRow,
    ProtectedTestError,
    SPLITS,
)


EXPECTED_TAXONOMY = (
    "pass",
    "missing_cap",
    "crooked_cap",
    "damaged_vial",
    "no_vial",
)
SCHEDULE_VERSION = "staging5_event_class_schedule_v2"
AUGMENTATION_VERSION = "staging5_candidate_train_augmentation_v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_TRAIN_CLASSES = (
    "pass",
    "missing_cap",
    "crooked_cap",
    "no_vial",
)

if tuple(CANONICAL_CLASSES) != EXPECTED_TAXONOMY:
    raise RuntimeError(
        "staging five-class taxonomy changed; refuse to construct a training schedule"
    )


@dataclass(frozen=True)
class TrainingSchedule:
    """Auditable event-weighted indices for one epoch, without replacement."""

    version: str
    taxonomy: tuple[str, ...]
    active_classes: tuple[str, ...]
    seed: int
    epoch: int
    samples_per_class: int
    indices: tuple[int, ...]
    sample_ids: tuple[str, ...]
    event_uids: tuple[str, ...]
    visual_classes: tuple[str, ...]
    class_counts: Mapping[str, int]
    distinct_event_support: Mapping[str, int]
    event_counts: Mapping[str, int]
    sha256: str


def _require_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ManifestError(f"{name} must be a non-negative integer")
    output = int(value)
    if output < 0:
        raise ManifestError(f"{name} must be a non-negative integer")
    return output


def _stable_key(seed: int, epoch: int, namespace: str, value: str) -> str:
    payload = json.dumps(
        [SCHEDULE_VERSION, seed, epoch, namespace, value],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_training_rows(rows: Sequence[ManifestRow]) -> list[tuple[int, ManifestRow]]:
    """Validate schedule inputs without opening a source image.

    ``read_manifest`` normally enforces these constraints.  Repeating the
    checks prevents a future caller from bypassing the manifest boundary by
    constructing dataclass instances directly.
    """

    if not rows:
        raise ManifestError("cannot build a training schedule from no rows")

    sample_ids: set[str] = set()
    event_positions: set[tuple[str, str, str]] = set()
    source_events: dict[str, str] = {}
    source_metadata: dict[str, tuple[str, str]] = {}
    digest_events: dict[str, str] = {}
    physical_events: dict[tuple[str, str], str] = {}
    event_metadata: dict[str, tuple[str, str]] = {}
    event_fixture_sources: dict[tuple[str, str], str] = {}
    usable: list[tuple[int, ManifestRow]] = []

    for index, row in enumerate(rows):
        if not isinstance(row, ManifestRow):
            raise ManifestError(
                f"training row {index} is not a validated ManifestRow"
            )
        row.validate()
        if row.designation != "development":
            raise ProtectedTestError(
                f"{row.sample_id}: protected-test evidence reached the training scheduler"
            )
        if row.split != "train":
            raise ManifestError(
                f"{row.sample_id}: training schedule accepts split='train' only, "
                f"got {row.split!r}"
            )
        if row.sample_id in sample_ids:
            raise ManifestError(f"duplicate sample_id {row.sample_id!r}")
        sample_ids.add(row.sample_id)

        position = (row.event_uid, row.fixture_id, row.position_id)
        if position in event_positions:
            raise ManifestError(f"duplicate event/fixture/position {position!r}")
        event_positions.add(position)

        event_meta = (row.session_id, row.event_id)
        prior_meta = event_metadata.setdefault(row.event_uid, event_meta)
        if prior_meta != event_meta:
            raise ManifestError(
                f"event {row.event_uid!r} has inconsistent session/event IDs: "
                f"{prior_meta!r} and {event_meta!r}"
            )

        physical_key = (row.session_id, row.event_id)
        prior_uid = physical_events.setdefault(physical_key, row.event_uid)
        if prior_uid != row.event_uid:
            raise ManifestError(
                "one physical session/event is assigned multiple event_uids: "
                f"{physical_key!r} -> {prior_uid!r}, {row.event_uid!r}"
            )

        prior_event = source_events.setdefault(row.source_image_id, row.event_uid)
        if prior_event != row.event_uid:
            raise ManifestError(
                f"source image {row.source_image_id!r} is assigned to multiple events: "
                f"{prior_event!r}, {row.event_uid!r}"
            )
        image_metadata = (row.source_image_relpath, row.source_image_sha256)
        prior_image_metadata = source_metadata.setdefault(
            row.source_image_id, image_metadata
        )
        if prior_image_metadata != image_metadata:
            raise ManifestError(
                f"source image {row.source_image_id!r} has inconsistent path/hash metadata"
            )
        fixture_key = (row.event_uid, row.fixture_id)
        prior_fixture_source = event_fixture_sources.setdefault(
            fixture_key, row.source_image_id
        )
        if prior_fixture_source != row.source_image_id:
            raise ManifestError(
                f"event/fixture {fixture_key!r} references multiple source images: "
                f"{prior_fixture_source!r}, {row.source_image_id!r}"
            )
        prior_digest_event = digest_events.setdefault(
            row.source_image_sha256, row.event_uid
        )
        if prior_digest_event != row.event_uid:
            raise ManifestError(
                "byte-identical source content is assigned to multiple events: "
                f"{prior_digest_event!r}, {row.event_uid!r}"
            )
        if row.is_usable:
            usable.append((index, row))

    if not usable:
        raise ManifestError("training split has no usable rows")
    return usable


def _class_sequence(
    entries: Sequence[tuple[int, ManifestRow]],
    *,
    target: int,
    seed: int,
    epoch: int,
    visual_class: str,
) -> list[tuple[int, ManifestRow]]:
    """Choose at most one row per event and rotate positions across epochs."""

    by_event: dict[str, list[tuple[int, ManifestRow]]] = defaultdict(list)
    for entry in entries:
        by_event[entry[1].event_uid].append(entry)

    event_order = sorted(
        by_event,
        key=lambda event: (
            _stable_key(seed, epoch, f"{visual_class}:event", event),
            event,
        ),
    )
    if target > len(event_order):
        raise ManifestError(
            f"class {visual_class!r} has only {len(event_order)} distinct events; "
            f"cannot schedule {target} event-weighted samples"
        )

    output: list[tuple[int, ManifestRow]] = []
    for event in event_order[:target]:
        # Sort independently of manifest order, then use an epoch increment so
        # repeated epochs rotate the chosen position within the event instead
        # of repeatedly drawing a single correlated crop.
        ordered = sorted(by_event[event], key=lambda entry: entry[1].sample_id)
        base_offset = int(
            _stable_key(
                seed,
                0,
                f"{visual_class}:position-offset",
                event,
            ),
            16,
        ) % len(ordered)
        output.append(ordered[(base_offset + epoch) % len(ordered)])
    return output


def build_balanced_epoch_schedule(
    rows: Sequence[ManifestRow],
    *,
    seed: int,
    epoch: int = 0,
    samples_per_class: int | None = None,
) -> TrainingSchedule:
    """Build a deterministic class-balanced, event-weighted epoch.

    Every usable class present in ``rows`` contributes the same number of
    samples, capped at one row per physical event/class.  The default count is
    the smallest distinct-event support among active classes, never the number
    of correlated position rows.  The epoch deterministically rotates which
    position represents an event/class.  This prevents one densely annotated
    event from dominating while retaining the exact five-class taxonomy in
    schedule metadata.

    ``pass``, ``missing_cap``, ``crooked_cap``, and ``no_vial`` are mandatory.
    ``damaged_vial`` is the only class that may be absent; absence is recorded
    as zero support and never filled with a fabricated sample.

    The schedule is invariant to input row order when compared by sample IDs or
    ``sha256``.  ``indices`` always point into the caller's original sequence.
    """

    seed = _require_nonnegative_integer("seed", seed)
    epoch = _require_nonnegative_integer("epoch", epoch)
    usable = _validate_training_rows(rows)

    by_class: dict[str, list[tuple[int, ManifestRow]]] = {
        name: [] for name in EXPECTED_TAXONOMY
    }
    for entry in usable:
        by_class[entry[1].visual_class].append(entry)
    active = tuple(name for name in EXPECTED_TAXONOMY if by_class[name])
    missing_required = [
        name for name in REQUIRED_TRAIN_CLASSES if not by_class[name]
    ]
    if missing_required:
        raise ManifestError(
            "training schedule is missing required classes "
            f"{missing_required}; only damaged_vial may be absent"
        )

    distinct_event_support = {
        name: len({row.event_uid for _, row in by_class[name]})
        for name in EXPECTED_TAXONOMY
    }

    smallest = min(distinct_event_support[name] for name in active)
    if samples_per_class is None:
        target = smallest
    else:
        target = _require_nonnegative_integer(
            "samples_per_class", samples_per_class
        )
        if target == 0:
            raise ManifestError("samples_per_class must be positive")
        if target > smallest:
            support = {name: distinct_event_support[name] for name in active}
            raise ManifestError(
                "samples_per_class exceeds distinct-event capacity; each event/class "
                "is capped at one contribution per epoch "
                f"(requested={target}, distinct_event_support={support})"
            )

    per_class = {
        name: deque(
            _class_sequence(
                by_class[name],
                target=target,
                seed=seed,
                epoch=epoch,
                visual_class=name,
            )
        )
        for name in active
    }
    class_order = sorted(
        active,
        key=lambda name: (_stable_key(seed, epoch, "class", name), name),
    )

    # Interleave classes without reordering each class queue.  _class_sequence
    # guarantees at most one contribution per event/class; cross-class
    # adjacency preferences must never undo that guarantee.
    selected: list[tuple[int, ManifestRow]] = []
    while any(per_class[name] for name in class_order):
        progressed = False
        for name in class_order:
            queue = per_class[name]
            if not queue:
                continue
            entry = queue.popleft()
            selected.append(entry)
            progressed = True
        if not progressed:  # defensive: the loop condition should make this impossible
            raise RuntimeError("training schedule interleave made no progress")

    sample_ids = tuple(row.sample_id for _, row in selected)
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("internal error: schedule selected a sample more than once")
    event_uids = tuple(row.event_uid for _, row in selected)
    visual_classes = tuple(row.visual_class for _, row in selected)
    event_class_pairs = tuple(zip(event_uids, visual_classes))
    if len(event_class_pairs) != len(set(event_class_pairs)):
        raise RuntimeError(
            "internal error: one event contributed multiple rows to a class"
        )
    class_counts = {
        name: int(Counter(visual_classes).get(name, 0))
        for name in EXPECTED_TAXONOMY
    }
    event_counts = dict(sorted(Counter(event_uids).items()))
    receipt = {
        "version": SCHEDULE_VERSION,
        "taxonomy": list(EXPECTED_TAXONOMY),
        "active_classes": list(active),
        "seed": seed,
        "epoch": epoch,
        "samples_per_class": target,
        "sample_ids": list(sample_ids),
        "event_uids": list(event_uids),
        "visual_classes": list(visual_classes),
        "class_counts": class_counts,
        "distinct_event_support": distinct_event_support,
        "event_counts": event_counts,
    }
    digest = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TrainingSchedule(
        version=SCHEDULE_VERSION,
        taxonomy=EXPECTED_TAXONOMY,
        active_classes=active,
        seed=seed,
        epoch=epoch,
        samples_per_class=target,
        indices=tuple(index for index, _ in selected),
        sample_ids=sample_ids,
        event_uids=event_uids,
        visual_classes=visual_classes,
        class_counts=class_counts,
        distinct_event_support=distinct_event_support,
        event_counts=event_counts,
        sha256=digest,
    )


def _finite_range(name: str, value: object, *, lower: float) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ManifestError(f"{name} must be a two-value [low, high] range")
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must contain finite numbers") from exc
    if not math.isfinite(low) or not math.isfinite(high):
        raise ManifestError(f"{name} must contain finite numbers")
    if low < lower or high < low:
        raise ManifestError(
            f"{name} must satisfy {lower} <= low <= high, got {(low, high)!r}"
        )
    return low, high


@dataclass(frozen=True)
class CandidateTrainAugmentationPolicy:
    """Candidate train-only transform ranges, not a label-preservation claim.

    Any non-identity policy requires the SHA-256 of an immutable, versioned
    development-only attestation that records why those exact ranges are
    acceptable.  The digest authenticates the evidence record; it does not by
    itself prove that a transform preserves labels.  Protected-test evidence
    must never be used to create or revise that record.

    Defaults are identity, which remains usable without an attestation.
    Rotation and reflection are intentionally absent from the contract.
    """

    exposure_gain: tuple[float, float] = (1.0, 1.0)
    contrast_gain: tuple[float, float] = (1.0, 1.0)
    blur_sigma_px: tuple[float, float] = (0.0, 0.0)
    noise_sigma_8bit: tuple[float, float] = (0.0, 0.0)
    max_translation_px: int = 0
    development_attestation_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exposure_gain",
            _finite_range("exposure_gain", self.exposure_gain, lower=0.0),
        )
        if self.exposure_gain[0] == 0:
            raise ManifestError("exposure_gain must be strictly positive")
        object.__setattr__(
            self,
            "contrast_gain",
            _finite_range("contrast_gain", self.contrast_gain, lower=0.0),
        )
        if self.contrast_gain[0] == 0:
            raise ManifestError("contrast_gain must be strictly positive")
        object.__setattr__(
            self,
            "blur_sigma_px",
            _finite_range("blur_sigma_px", self.blur_sigma_px, lower=0.0),
        )
        object.__setattr__(
            self,
            "noise_sigma_8bit",
            _finite_range("noise_sigma_8bit", self.noise_sigma_8bit, lower=0.0),
        )
        translation = _require_nonnegative_integer(
            "max_translation_px", self.max_translation_px
        )
        object.__setattr__(self, "max_translation_px", translation)
        attestation = self.development_attestation_sha256
        if attestation is not None and (
            not isinstance(attestation, str) or not SHA256_RE.fullmatch(attestation)
        ):
            raise ManifestError(
                "development_attestation_sha256 must be 64 lowercase hex digits"
            )
        if not self.is_identity and attestation is None:
            raise ManifestError(
                "non-identity candidate augmentation requires an immutable "
                "development_attestation_sha256"
            )

    @property
    def is_identity(self) -> bool:
        return (
            self.exposure_gain == (1.0, 1.0)
            and self.contrast_gain == (1.0, 1.0)
            and self.blur_sigma_px == (0.0, 0.0)
            and self.noise_sigma_8bit == (0.0, 0.0)
            and self.max_translation_px == 0
        )

    @property
    def sha256(self) -> str:
        payload = {
            "version": AUGMENTATION_VERSION,
            "policy": asdict(self),
            "train_only": True,
            "candidate_not_label_preserving_by_construction": True,
            "rotation": "disabled",
            "reflection": "disabled",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


def _validate_crop(crop: np.ndarray) -> None:
    if not isinstance(crop, np.ndarray):
        raise ManifestError("crop must be a NumPy array")
    if crop.dtype != np.uint8:
        raise ManifestError(f"crop dtype must be uint8, got {crop.dtype}")
    if crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) < 2:
        raise ManifestError(
            f"crop must have non-empty HxWx3 BGR shape, got {crop.shape!r}"
        )


def _augmentation_rng(
    *, seed: int, epoch: int, sample_id: str, occurrence: int
) -> np.random.Generator:
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ManifestError("sample_id must be a non-blank string")
    payload = json.dumps(
        [AUGMENTATION_VERSION, seed, epoch, sample_id, occurrence],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    entropy = np.frombuffer(digest, dtype="<u4")
    return np.random.default_rng(entropy)


def apply_candidate_split_augmentation(
    crop: np.ndarray,
    *,
    split: str,
    policy: CandidateTrainAugmentationPolicy,
    seed: int,
    epoch: int,
    sample_id: str,
    occurrence: int = 0,
) -> np.ndarray:
    """Apply an attested candidate augmentation to TRAIN only.

    Validation and test crops are returned byte-identically (as independent
    arrays), which makes it impossible for this primitive to introduce random
    evaluation-time transforms.  A non-identity result is not evidence that the
    original class label remains correct; that claim belongs in the referenced
    development-only attestation.  The function never opens a file.
    """

    _validate_crop(crop)
    if split not in SPLITS:
        raise ManifestError(f"split must be one of {SPLITS}, got {split!r}")
    if not isinstance(policy, CandidateTrainAugmentationPolicy):
        raise ManifestError("policy must be a CandidateTrainAugmentationPolicy")
    if split != "train":
        return crop.copy()

    seed = _require_nonnegative_integer("seed", seed)
    epoch = _require_nonnegative_integer("epoch", epoch)
    occurrence = _require_nonnegative_integer("occurrence", occurrence)
    rng = _augmentation_rng(
        seed=seed,
        epoch=epoch,
        sample_id=sample_id,
        occurrence=occurrence,
    )
    height, width = crop.shape[:2]
    if policy.max_translation_px * 2 >= min(height, width):
        raise ManifestError(
            "max_translation_px must be less than half the crop's shorter side"
        )
    if policy.blur_sigma_px[1] > max(height, width):
        raise ManifestError(
            "blur_sigma_px cannot exceed the crop's longer side"
        )

    numeric_limit = float(np.finfo(np.float32).max) / 1024.0
    maximum_signal = (
        255.0
        * max(policy.exposure_gain)
        * (max(policy.contrast_gain) + 1.0)
    )
    if maximum_signal > numeric_limit or max(policy.noise_sigma_8bit) > numeric_limit:
        raise ManifestError("candidate augmentation ranges exceed numeric safety limits")

    output = crop
    if policy.max_translation_px:
        dx, dy = rng.integers(
            -policy.max_translation_px,
            policy.max_translation_px + 1,
            size=2,
        )
        matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
        output = cv2.warpAffine(
            output,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    exposure = rng.uniform(*policy.exposure_gain)
    contrast = rng.uniform(*policy.contrast_gain)
    blur = rng.uniform(*policy.blur_sigma_px)
    noise = rng.uniform(*policy.noise_sigma_8bit)
    work = output.astype(np.float32)
    mean = work.mean(axis=(0, 1), keepdims=True)
    work = (work - mean) * contrast + mean
    work *= exposure
    if not np.isfinite(work).all():
        raise ManifestError("candidate augmentation produced non-finite photometric values")
    if blur > 0:
        work = cv2.GaussianBlur(work, (0, 0), float(blur))
    if noise > 0:
        work += rng.normal(0.0, float(noise), size=work.shape).astype(np.float32)
    if work.shape != crop.shape or not np.isfinite(work).all():
        raise ManifestError("candidate augmentation produced an invalid output array")
    result = np.clip(np.rint(work), 0, 255).astype(np.uint8)
    if result.shape != crop.shape:
        raise ManifestError("candidate augmentation changed the crop shape")
    return result
