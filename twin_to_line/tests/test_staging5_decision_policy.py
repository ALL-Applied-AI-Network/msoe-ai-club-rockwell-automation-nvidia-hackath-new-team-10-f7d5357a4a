"""Boundary and cross-path tests for the canonical staging5 safety policy."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate_resnet5_test as protected_evaluation
import export_resnet5_onnx as onnx_export
import resnet5_inspector as pytorch_runtime
import resnet5_onnx_inspector as onnx_runtime
import staging5_decision_policy as policy
import train_resnet5 as training
from staging_dataset import CANONICAL_CLASSES


def test_exact_taxonomy_and_all_paths_share_one_policy_function():
    assert CANONICAL_CLASSES == (
        "pass",
        "missing_cap",
        "crooked_cap",
        "damaged_vial",
        "no_vial",
    )
    assert policy.POLICY_VERSION == "staging5_probability_to_visual_v1"
    assert training.decide_one is policy.decide_one
    assert protected_evaluation.decide_one is policy.decide_one
    assert onnx_export.decide_one is policy.decide_one
    assert pytorch_runtime.decide_one is policy.decide_one
    assert onnx_runtime.decide_one is policy.decide_one


@pytest.mark.parametrize(
    ("probabilities", "threshold", "expected_class", "expected_reason"),
    [
        ([0.70, 0.10, 0.05, 0.05, 0.10], 0.50, "pass", "below_defect_threshold"),
        ([0.10, 0.05, 0.05, 0.05, 0.75], 0.50, "no_vial", "below_defect_threshold"),
        # Aggregate defect score equals the threshold: the supported defect wins.
        ([0.22, 0.32, 0.10, 0.08, 0.28], 0.50, "missing_cap", "at_or_above_defect_threshold"),
    ],
)
def test_supported_decision_boundaries(
    probabilities, threshold, expected_class, expected_reason
):
    predicted, score, reason = policy.decide_one(
        probabilities, threshold, ["damaged_vial"]
    )
    assert predicted == expected_class
    assert score is not None
    assert reason == expected_reason


def test_unsupported_global_argmax_is_unknown_below_aggregate_threshold():
    probabilities = np.array([0.29, 0.08, 0.07, 0.30, 0.26], np.float32)
    predicted, score, reason = policy.decide_one(
        probabilities, 0.90, ["damaged_vial"]
    )
    assert predicted == "unknown"
    assert score == pytest.approx(0.45)
    assert reason == "unsupported_class_is_global_argmax"


def test_unsupported_highest_defect_is_unknown_above_threshold():
    predicted, score, reason = policy.decide_one(
        [0.40, 0.15, 0.14, 0.16, 0.15], 0.40, ["damaged_vial"]
    )
    assert predicted == "unknown"
    assert score == pytest.approx(0.45)
    assert reason == "unsupported_class_has_highest_defect_probability"


@pytest.mark.parametrize(
    ("probabilities", "threshold", "reason"),
    [
        ([0.35, 0.35, 0.10, 0.05, 0.15], 0.90, "ambiguous_global_argmax"),
        ([0.20, 0.30, 0.15, 0.15, 0.20], 0.90, "ambiguous_non_defect_class"),
        ([0.30, 0.20, 0.20, 0.10, 0.20], 0.50, "ambiguous_defect_class"),
    ],
)
def test_exact_ties_are_never_resolved_by_class_order(probabilities, threshold, reason):
    predicted, score, actual_reason = policy.decide_one(probabilities, threshold)
    assert predicted == "unknown"
    assert score is not None
    assert actual_reason == reason


@pytest.mark.parametrize(
    ("probabilities", "reason"),
    [
        ([0.8, 0.1, np.nan, 0.0, 0.1], "non_finite_probability"),
        ([0.8, 0.1, np.inf, 0.0, 0.1], "non_finite_probability"),
        ([1.1, -0.1, 0.0, 0.0, 0.0], "probability_out_of_range"),
        ([0.4, 0.1, 0.1, 0.1, 0.1], "probabilities_not_normalized"),
        ([0.8, 0.1, 0.1, 0.0], "invalid_probability_shape"),
    ],
)
def test_invalid_evidence_fails_closed_without_a_synthetic_score(probabilities, reason):
    predicted, score, actual_reason = policy.decide_one(probabilities, 0.5)
    assert predicted == "unknown"
    assert score is None
    assert actual_reason == reason


@pytest.mark.parametrize("threshold", [-0.01, 1.01, np.nan, np.inf, True])
def test_invalid_threshold_configuration_is_refused(threshold):
    with pytest.raises(ValueError, match="threshold"):
        policy.decide_one([0.8, 0.05, 0.05, 0.05, 0.05], threshold)


def test_invalid_unsupported_class_configuration_is_refused():
    probabilities = [0.8, 0.05, 0.05, 0.05, 0.05]
    with pytest.raises(ValueError, match="unknown unsupported"):
        policy.decide_one(probabilities, 0.5, ["not_a_class"])
    with pytest.raises(ValueError, match="duplicates"):
        policy.decide_one(probabilities, 0.5, ["damaged_vial", "damaged_vial"])


def test_protected_evaluation_uses_same_unsupported_below_threshold_decision():
    row = SimpleNamespace(
        source_image_id="image-1",
        fixture_id="G",
        position_id="G01",
        expected_occupancy="occupied",
        visual_class="pass",
        capture_time="2026-10-28T12:00:00-05:00",
        event_uid="visit-1:event-1",
        review_state="adjudicated",
    )
    probabilities = np.array([[0.29, 0.08, 0.07, 0.30, 0.26]], np.float32)
    metrics, records = protected_evaluation.compute_metrics(
        [row], probabilities, 0.90, ["damaged_vial"]
    )
    assert records[0]["predicted_visual_class"] == "unknown"
    assert records[0]["inspection_decision"] == "UNSURE"
    assert records[0]["defect_score"] == pytest.approx(0.45)
    assert records[0]["reason_code"].startswith(
        "unsupported_class_is_global_argmax;"
    )
    assert metrics["unsure_rate"] == 1.0


def test_onnx_runtime_observation_matches_canonical_policy_without_local_logic():
    inspector = object.__new__(onnx_runtime.ResNet5OnnxInspector)
    inspector.t_defect = 0.90
    inspector.unsupported = ("damaged_vial",)
    probabilities = np.array([[0.29, 0.08, 0.07, 0.30, 0.26]], np.float32)

    predicted, score, reason = policy.decide_one(
        probabilities[0], inspector.t_defect, inspector.unsupported
    )
    observation = inspector.observations(probabilities)[0]

    assert observation.visual_state == predicted == "unknown"
    assert observation.score == pytest.approx(score)
    assert observation.features["reason_code"] == reason
    assert observation.features["decision_policy_version"] == policy.POLICY_VERSION
