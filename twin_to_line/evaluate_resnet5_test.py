#!/usr/bin/env python3
"""Evaluate one frozen five-output model on its locked protected TEST split.

This is deliberately separate from training.  It verifies model, threshold,
split-lock and test-manifest hashes before reading test pixels.  Test results
are descriptive only and must never be used to refit the model or threshold.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np

import frame_warp as FW
from resnet5_inspector import ResNet5Inspector
from staging_dataset import (
    CANONICAL_CLASSES,
    DEFECT_CLASSES,
    ManifestError,
    ManifestRow,
    ProtectedTestCropSet,
    read_manifest,
    sha256_file,
    verify_test_manifest,
)
from train_resnet5 import (
    _parse_maps,
    decide_one,
    evaluate,
    final_decision,
    verify_preprocessing_sha256,
)


def _ground_truth_decision(row: ManifestRow) -> str:
    if row.visual_class in DEFECT_CLASSES:
        return "DEFECT"
    if row.visual_class == "no_vial":
        return "MISSED" if row.expected_occupancy == "occupied" else "WAITING"
    return "PASS"


def compute_metrics(
    rows: Sequence[ManifestRow],
    probabilities: np.ndarray,
    threshold: float,
    unsupported_classes: Sequence[str] = (),
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if probabilities.shape != (len(rows), len(CANONICAL_CLASSES)):
        raise ManifestError(
            f"probability shape {probabilities.shape} does not match "
            f"{len(rows)} rows x {len(CANONICAL_CLASSES)} classes"
        )
    unsupported = set(unsupported_classes)
    predictions: list[dict[str, object]] = []
    for row, probability in zip(rows, probabilities):
        predicted, score, visual_reason = decide_one(
            probability, threshold, unsupported_classes
        )
        decision, decision_reason = final_decision(predicted, row)
        record: dict[str, object] = {
            "source_image_id": row.source_image_id,
            "fixture_id": row.fixture_id,
            "position_id": row.position_id,
            "expected_occupancy": row.expected_occupancy,
            "ground_truth_visual_class": row.visual_class,
            "predicted_visual_class": predicted,
            "defect_score": round(float(score), 8),
            "applicable_threshold": threshold,
            "inspection_decision": decision,
            "ground_truth_decision": _ground_truth_decision(row),
            "reason_code": f"{visual_reason};{decision_reason}",
            "capture_time": row.capture_time,
            "event_uid": row.event_uid,
            "review_state": row.review_state,
        }
        record.update(
            {
                f"p_{name}": round(float(value), 8)
                for name, value in zip(CANONICAL_CLASSES, probability)
            }
        )
        predictions.append(record)

    n = len(predictions)
    exact = np.array(
        [
            item["predicted_visual_class"] == item["ground_truth_visual_class"]
            for item in predictions
        ],
        dtype=bool,
    )
    process_exact = np.array(
        [
            item["inspection_decision"] == item["ground_truth_decision"]
            for item in predictions
        ],
        dtype=bool,
    )
    confusion_names = list(CANONICAL_CLASSES) + ["unknown"]
    confusion = {
        truth: {
            predicted: sum(
                item["ground_truth_visual_class"] == truth
                and item["predicted_visual_class"] == predicted
                for item in predictions
            )
            for predicted in confusion_names
        }
        for truth in CANONICAL_CLASSES
    }
    per_class: dict[str, dict[str, object]] = {}
    recalls: list[float] = []
    for name in CANONICAL_CLASSES:
        indices = [
            index
            for index, row in enumerate(rows)
            if row.visual_class == name
        ]
        supported = name not in unsupported
        recall = float(exact[indices].mean()) if indices and supported else None
        if recall is not None:
            recalls.append(recall)
        per_class[name] = {
            "n": len(indices),
            "supported": supported,
            "exact_visual_recall": recall,
        }

    defect_indices = [
        index for index, row in enumerate(rows) if row.visual_class in DEFECT_CLASSES
    ]
    supported_defect_indices = [
        index
        for index in defect_indices
        if rows[index].visual_class not in unsupported
    ]
    detected_defect = np.array(
        [item["inspection_decision"] == "DEFECT" for item in predictions], dtype=bool
    )
    nondefect_indices = [
        index for index, row in enumerate(rows) if row.visual_class not in DEFECT_CLASSES
    ]
    missed_indices = [
        index
        for index, row in enumerate(rows)
        if row.visual_class == "no_vial" and row.expected_occupancy == "occupied"
    ]
    waiting_indices = [
        index
        for index, row in enumerate(rows)
        if row.visual_class == "no_vial" and row.expected_occupancy == "empty"
    ]
    events: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        events.setdefault(row.event_uid, []).append(index)
    event_exact = {
        event: bool(exact[indices].all()) for event, indices in sorted(events.items())
    }
    metrics: dict[str, object] = {
        "n_positions": n,
        "n_events": len(events),
        "visual_exact_accuracy": float(exact.mean()),
        "macro_visual_recall_supported_classes": (
            float(np.mean(recalls)) if recalls else None
        ),
        "process_decision_accuracy": float(process_exact.mean()),
        "supported_visual_defect_recall": (
            float(detected_defect[supported_defect_indices].mean())
            if supported_defect_indices
            else None
        ),
        "visual_false_defect_alarm_rate": (
            float(detected_defect[nondefect_indices].mean())
            if nondefect_indices
            else None
        ),
        "missed_placement_recall": (
            float(
                np.mean(
                    [
                        predictions[index]["inspection_decision"] == "MISSED"
                        for index in missed_indices
                    ]
                )
            )
            if missed_indices
            else None
        ),
        "waiting_state_accuracy": (
            float(
                np.mean(
                    [
                        predictions[index]["inspection_decision"] == "WAITING"
                        for index in waiting_indices
                    ]
                )
            )
            if waiting_indices
            else None
        ),
        "unsure_rate": float(
            np.mean([item["inspection_decision"] == "UNSURE" for item in predictions])
        ),
        "per_class": per_class,
        "visual_confusion": confusion,
        "event_all_positions_exact": event_exact,
        "events_all_positions_exact_rate": float(np.mean(list(event_exact.values()))),
        "unsupported_classes": sorted(unsupported),
        "protocol_reference_rows": sum(
            row.review_state == "protocol_reference" for row in rows
        ),
    }
    return metrics, predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--position-map", action="append", default=[], metavar="FIXTURE=PATH")
    parser.add_argument("--frame-warp", type=Path, default=Path(FW.WARP_PATH))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite test evaluation {args.out}")

    lock = verify_test_manifest(args.test_manifest, args.split_lock)
    inspector = ResNet5Inspector(args.model_dir, device=args.device)
    run_lock_sha = inspector.run.get("split_lock", {}).get("sha256")
    if run_lock_sha != sha256_file(args.split_lock):
        raise ManifestError("model was not trained from this split lock")
    if inspector.run.get("threshold_source") != "validation_only":
        raise ManifestError("model threshold was not frozen from validation only")

    fixture_maps, fixture_map_paths = _parse_maps(args.position_map)
    warp = FW.FrameWarp(args.frame_warp) if args.frame_warp else None
    verified_preprocessing = verify_preprocessing_sha256(
        inspector.run.get("preprocessing_sha256"),
        args.frame_warp,
        fixture_map_paths,
    )

    # Do not hash or decode protected pixels until all frozen-model and
    # preprocessing provenance checks above have passed.
    rows = read_manifest(args.test_manifest, image_root=args.image_root, verify_images=True)
    if {row.split for row in rows} != {"test"}:
        raise ManifestError("protected test manifest contains a non-test row")
    dataset = ProtectedTestCropSet(
        rows,
        args.image_root,
        fixture_maps=fixture_maps,
        warp=warp,
    )
    probabilities, _ = evaluate(
        inspector.model,
        dataset,
        args.batch,
        inspector.device,
        inspector.torch,
    )
    metrics, predictions = compute_metrics(
        dataset.rows,
        probabilities,
        inspector.threshold,
        inspector.unsupported,
    )
    args.out.mkdir(parents=True)
    prediction_path = args.out / "test_predictions.csv"
    with prediction_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    report = {
        "evaluation_started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "once-only descriptive evaluation of frozen model on locked test events",
        "evaluation_scope": lock.get("evaluation_scope"),
        "independent_visit_test": lock.get("independent_visit_test"),
        "metric_warning": lock.get("metric_warning"),
        "label_interpretation": (
            "protocol_reference means agreement with intended staging state, not "
            "independently adjudicated physical ground truth"
        ),
        "test_used_for_training": False,
        "test_used_for_threshold_selection": False,
        "model_dir": str(args.model_dir.resolve()),
        "model_pt_sha256": inspector.run["model_pt_sha256"],
        "thresholds_sha256": inspector.run["thresholds_sha256"],
        "split_lock_sha256": sha256_file(args.split_lock),
        "test_manifest_sha256": sha256_file(args.test_manifest),
        "preprocessing_sha256": verified_preprocessing,
        "evaluation_code_sha256": sha256_file(__file__),
        "test_predictions_sha256": sha256_file(prediction_path),
        "metrics": metrics,
    }
    report_path = args.out / "test_metrics.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
