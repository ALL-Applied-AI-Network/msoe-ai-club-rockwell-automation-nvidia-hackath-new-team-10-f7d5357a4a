"""Canonical safety policy for five-class staging-model probabilities.

Every training, evaluation, export-verification, and runtime path must use
``decide_one``.  The model always retains the exact five-output taxonomy, but
classes without development evidence are not asserted.  Invalid or ambiguous
evidence fails closed to ``unknown`` so deterministic manufacturing rules can
escalate it to ``UNSURE``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from staging_dataset import CANONICAL_CLASSES, DEFECT_CLASSES


POLICY_VERSION = "staging5_probability_to_visual_v1"
EXPECTED_CLASS_ORDER = (
    "pass",
    "missing_cap",
    "crooked_cap",
    "damaged_vial",
    "no_vial",
)
NON_DEFECT_CLASSES = ("pass", "no_vial")
PROBABILITY_SUM_ATOL = 1e-4

if CANONICAL_CLASSES != EXPECTED_CLASS_ORDER:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "staging decision policy requires the exact five-class taxonomy "
        f"{EXPECTED_CLASS_ORDER}, got {CANONICAL_CLASSES}"
    )

DEFECT_INDICES = tuple(CANONICAL_CLASSES.index(name) for name in DEFECT_CLASSES)
NON_DEFECT_INDICES = tuple(
    CANONICAL_CLASSES.index(name) for name in NON_DEFECT_CLASSES
)


def _validated_configuration(
    threshold: float, unsupported_classes: Sequence[str]
) -> tuple[float, frozenset[str]]:
    """Return normalized policy configuration or refuse unsafe metadata."""
    if isinstance(threshold, bool):
        raise ValueError("defect threshold must be a finite number in [0, 1]")
    try:
        normalized_threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("defect threshold must be a finite number in [0, 1]") from exc
    if not math.isfinite(normalized_threshold) or not 0.0 <= normalized_threshold <= 1.0:
        raise ValueError("defect threshold must be a finite number in [0, 1]")

    unsupported = tuple(unsupported_classes)
    if len(set(unsupported)) != len(unsupported):
        raise ValueError("unsupported_classes contains duplicates")
    unknown = set(unsupported) - set(CANONICAL_CLASSES)
    if unknown:
        raise ValueError(f"unknown unsupported classes: {sorted(unknown)}")
    return normalized_threshold, frozenset(unsupported)


def _unknown(reason_code: str, score: float | None = None) -> tuple[str, float | None, str]:
    return "unknown", score, reason_code


def decide_one(
    probabilities: np.ndarray | Sequence[float],
    threshold: float,
    unsupported_classes: Sequence[str] = (),
) -> tuple[str, float | None, str]:
    """Map one five-class probability vector to a safety-gated visual class.

    The returned tuple is ``(visual_class, defect_score, reason_code)``.
    ``defect_score`` is ``None`` only when no trustworthy score can be computed.
    Invalid evidence never falls through to ``pass`` or ``no_vial``.
    """
    normalized_threshold, unsupported = _validated_configuration(
        threshold, unsupported_classes
    )
    try:
        values = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _unknown("invalid_probability_vector")
    if values.shape != (len(CANONICAL_CLASSES),):
        return _unknown("invalid_probability_shape")
    if not np.isfinite(values).all():
        return _unknown("non_finite_probability")
    if np.any(values < 0.0) or np.any(values > 1.0):
        return _unknown("probability_out_of_range")
    if not math.isclose(
        float(values.sum()), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_ATOL
    ):
        return _unknown("probabilities_not_normalized")

    score = float(values[list(DEFECT_INDICES)].sum())
    global_maximum = float(values.max())
    global_indices = tuple(int(index) for index in np.flatnonzero(values == global_maximum))
    global_classes = {CANONICAL_CLASSES[index] for index in global_indices}

    # Unsupported evidence is rejected before thresholding.  In particular, an
    # unsupported damaged_vial global argmax may not become pass merely because
    # the aggregate defect score is below the validation-only threshold.
    if global_classes & unsupported:
        return _unknown("unsupported_class_is_global_argmax", score)
    if len(global_indices) != 1:
        return _unknown("ambiguous_global_argmax", score)

    if score < normalized_threshold:
        non_defect_maximum = max(float(values[index]) for index in NON_DEFECT_INDICES)
        candidates = tuple(
            index
            for index in NON_DEFECT_INDICES
            if float(values[index]) == non_defect_maximum
        )
        if len(candidates) != 1:
            return _unknown("ambiguous_non_defect_class", score)
        return CANONICAL_CLASSES[candidates[0]], score, "below_defect_threshold"

    defect_maximum = max(float(values[index]) for index in DEFECT_INDICES)
    candidates = tuple(
        index for index in DEFECT_INDICES if float(values[index]) == defect_maximum
    )
    candidate_classes = {CANONICAL_CLASSES[index] for index in candidates}
    if candidate_classes & unsupported:
        return _unknown("unsupported_class_has_highest_defect_probability", score)
    if len(candidates) != 1:
        return _unknown("ambiguous_defect_class", score)
    return CANONICAL_CLASSES[candidates[0]], score, "at_or_above_defect_threshold"
