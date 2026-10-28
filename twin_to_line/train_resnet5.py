#!/usr/bin/env python3
"""Train a safety-first five-output ResNet-18 on locked staging splits.

Only TRAIN and VALIDATION manifests are accepted.  The protected TEST
manifest is never opened.  The exact visual taxonomy is:

    pass, missing_cap, crooked_cap, damaged_vial, no_vial

If the development data contain no damaged-vial evidence, the output head and
index are retained but the class is recorded as unsupported.  A high damaged
logit then becomes UNKNOWN/UNSURE, never an evidence-free damage claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Mapping, Sequence

import cv2
import numpy as np

import frame_warp as FW
import positions as P
from staging_dataset import (
    CANONICAL_CLASSES,
    DEFECT_CLASSES,
    ManifestError,
    ManifestRow,
    StagingCropSet,
    read_manifest,
    sha256_file,
    verify_training_manifests,
)


RECALL_TARGET = 0.995
DEFECT_INDICES = tuple(CANONICAL_CLASSES.index(name) for name in DEFECT_CLASSES)
NON_DEFECT_INDICES = (
    CANONICAL_CLASSES.index("pass"),
    CANONICAL_CLASSES.index("no_vial"),
)


def class_counts(rows: Sequence[ManifestRow]) -> dict[str, int]:
    return {
        name: sum(row.is_usable and row.visual_class == name for row in rows)
        for name in CANONICAL_CLASSES
    }


def validate_class_support(
    train_rows: Sequence[ManifestRow], validation_rows: Sequence[ManifestRow]
) -> list[str]:
    """Return zero-global-support classes, refusing partial/leaky support."""
    train = class_counts(train_rows)
    validation = class_counts(validation_rows)
    unsupported = [name for name in CANONICAL_CLASSES if train[name] + validation[name] == 0]
    illegal = set(unsupported) - {"damaged_vial"}
    if illegal:
        raise ManifestError(
            f"development data have no examples for required classes {sorted(illegal)}"
        )
    for name in CANONICAL_CLASSES:
        if name in unsupported:
            continue
        if train[name] == 0 or validation[name] == 0:
            raise ManifestError(
                f"class {name!r} exists in development data but is not independently "
                f"represented in both train and validation (train={train[name]}, "
                f"validation={validation[name]})"
            )
    return unsupported


def class_weights(counts: Mapping[str, int]) -> np.ndarray:
    """Inverse-frequency weights with a finite zero for unsupported classes."""
    values = np.array([counts[name] for name in CANONICAL_CLASSES], dtype=np.float64)
    present = values > 0
    if not present.any():
        raise ManifestError("training split has no usable examples")
    weights = np.zeros(len(CANONICAL_CLASSES), dtype=np.float32)
    weights[present] = values[present].sum() / (present.sum() * values[present])
    if not np.isfinite(weights).all():
        raise ManifestError("class weights are not finite")
    return weights


def defect_scores(probabilities: np.ndarray) -> np.ndarray:
    return probabilities[:, DEFECT_INDICES].sum(axis=1)


def choose_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    unsupported_classes: Sequence[str] = (),
) -> dict[str, object]:
    """Choose one conservative operating point using validation data only."""
    scores = defect_scores(probabilities)
    unsupported = set(unsupported_classes)
    supported_defects = [name for name in DEFECT_CLASSES if name not in unsupported]
    per_class: dict[str, float | None] = {}
    for name in DEFECT_CLASSES:
        mask = labels == CANONICAL_CLASSES.index(name)
        if name in unsupported:
            per_class[name] = None
            continue
        if not mask.any():
            raise ManifestError(f"validation has no examples for supported defect class {name}")
        ordered = np.sort(scores[mask])
        index = int(np.floor((1 - RECALL_TARGET) * len(ordered)))
        per_class[name] = float(ordered[index])
    if not supported_defects:
        raise ManifestError("validation has no supported defect class")
    threshold = min(float(per_class[name]) for name in supported_defects)
    defect_mask = np.isin(
        labels, [CANONICAL_CLASSES.index(name) for name in supported_defects]
    )
    non_defect_mask = np.isin(labels, NON_DEFECT_INDICES)
    if not non_defect_mask.any():
        raise ManifestError("validation has no pass/no_vial examples for false-alarm measurement")
    predicted_names = [
        decide_one(probability, threshold, unsupported_classes)[0]
        for probability in probabilities
    ]
    predicted_array = np.asarray(predicted_names, dtype=object)
    flag_recall = {}
    visual_recall = {}
    for name in CANONICAL_CLASSES:
        mask = labels == CANONICAL_CLASSES.index(name)
        visual_recall[name] = (
            float(np.mean(predicted_array[mask] == name)) if mask.any() else None
        )
        if name in DEFECT_CLASSES:
            flag_recall[name] = (
                float(np.mean(scores[mask] >= threshold)) if mask.any() else None
            )
    false_alarm_by_class = {}
    for name in ("pass", "no_vial"):
        mask = labels == CANONICAL_CLASSES.index(name)
        false_alarm_by_class[name] = (
            float(np.mean(scores[mask] >= threshold)) if mask.any() else None
        )
    return {
        "rule": (
            "s=P(missing_cap)+P(crooked_cap)+P(damaged_vial); t is the highest "
            "common validation-only threshold meeting the recall target for every "
            "supported defect class; unsupported damaged evidence becomes UNSURE"
        ),
        "threshold_source": "validation_only",
        "recall_target": RECALL_TARGET,
        "t_defect": threshold,
        "per_supported_defect_threshold": per_class,
        "t_catch_every_validation_defect": float(scores[defect_mask].min()),
        "validation_supported_defect_recall": float(np.mean(scores[defect_mask] >= threshold)),
        "validation_false_alarm_rate": float(np.mean(scores[non_defect_mask] >= threshold)),
        "validation_flag_recall_by_defect_class": flag_recall,
        "validation_visual_class_recall": visual_recall,
        "validation_false_alarm_rate_by_non_defect_class": false_alarm_by_class,
        "validation_unsure_rate": float(np.mean(predicted_array == "unknown")),
        "unsupported_classes": list(unsupported_classes),
        "n_validation": int(len(labels)),
        "n_validation_supported_defects": int(defect_mask.sum()),
    }


def decide_one(
    probabilities: np.ndarray,
    threshold: float,
    unsupported_classes: Sequence[str] = (),
) -> tuple[str, float, str]:
    score = float(probabilities[list(DEFECT_INDICES)].sum())
    if score < threshold:
        index = max(NON_DEFECT_INDICES, key=lambda value: float(probabilities[value]))
        return CANONICAL_CLASSES[index], score, "below_defect_threshold"
    defect_index = max(DEFECT_INDICES, key=lambda value: float(probabilities[value]))
    predicted = CANONICAL_CLASSES[defect_index]
    if predicted in set(unsupported_classes):
        return "unknown", score, "unsupported_class_has_highest_defect_probability"
    return predicted, score, "at_or_above_defect_threshold"


def final_decision(predicted: str, row: ManifestRow) -> tuple[str, str]:
    if predicted == "unknown":
        return "UNSURE", "unsupported_visual_evidence"
    if predicted in DEFECT_CLASSES:
        return "DEFECT", f"visual_{predicted}"
    if predicted == "no_vial":
        if row.expected_occupancy == "occupied":
            return "MISSED", "no_vial_where_occupied_expected"
        return "WAITING", "expected_trailing_empty"
    return "PASS", "visual_pass"


def batches(dataset: StagingCropSet, indices: Sequence[int], batch_size: int):
    for start in range(0, len(indices), batch_size):
        examples = [dataset.get(int(index)) for index in indices[start : start + batch_size]]
        features, labels = zip(*examples)
        yield np.stack(features), np.asarray(labels, dtype=np.int64)


def evaluate(model, dataset, batch_size, device, torch):
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in batches(dataset, np.arange(len(dataset)), batch_size):
            tensor = torch.from_numpy(features).to(device)
            output.append(torch.softmax(model(tensor), dim=1).cpu().numpy())
    probabilities = np.concatenate(output)
    labels = np.array(
        [CANONICAL_CLASSES.index(row.visual_class) for row in dataset.rows], dtype=np.int64
    )
    return probabilities, labels


def _parse_maps(
    values: Sequence[str],
) -> tuple[dict[str, P.PositionMap], dict[str, Path]]:
    if not values:
        return {"G": P.PositionMap()}, {"G": Path(P.MAP_PATH)}
    output: dict[str, P.PositionMap] = {}
    paths: dict[str, Path] = {}
    for value in values:
        try:
            fixture, path = value.split("=", 1)
        except ValueError as exc:
            raise ManifestError("--position-map must use FIXTURE=/path/to/map.json") from exc
        if fixture not in {"G", "S"}:
            raise ManifestError(f"invalid fixture in --position-map: {fixture!r}")
        pmap = P.PositionMap(path)
        if any(not position.startswith(fixture) for position in pmap.ids):
            raise ManifestError(f"{path}: map positions do not belong to fixture {fixture}")
        output[fixture] = pmap
        paths[fixture] = Path(path)
    return output, paths


def preprocessing_sha256(
    frame_warp: os.PathLike[str] | str,
    fixture_map_paths: Mapping[str, os.PathLike[str] | str],
) -> dict[str, str | None]:
    """Hash every file that determines staging crops and model inputs.

    ``registration_dev.json`` is optional in the legacy position-map loader.
    Record its absence explicitly so adding or removing a registration offset
    after training cannot silently change protected-test crops.
    """
    paths = {
        "train_resnet5.py": Path(__file__),
        "staging_dataset.py": Path(__file__).with_name("staging_dataset.py"),
        "positions.py": Path(P.__file__),
        "frame_warp.py": Path(FW.__file__),
        "frame_warp_1080.json": Path(frame_warp),
        "train_resnet.py": Path(__file__).with_name("train_resnet.py"),
    }
    paths.update(
        {
            f"position_map_{fixture}": Path(path)
            for fixture, path in sorted(fixture_map_paths.items())
        }
    )
    output: dict[str, str | None] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ManifestError(f"required preprocessing file is missing: {path}")
        output[name] = sha256_file(path)
    registration = Path(P.REG_PATH)
    output["registration_dev.json"] = (
        sha256_file(registration) if registration.is_file() else None
    )
    return output


def verify_preprocessing_sha256(
    recorded: object,
    frame_warp: os.PathLike[str] | str,
    fixture_map_paths: Mapping[str, os.PathLike[str] | str],
) -> dict[str, str | None]:
    """Fail closed unless evaluation preprocessing matches the training run."""
    if not isinstance(recorded, dict):
        raise ManifestError("model run.json lacks preprocessing_sha256")
    actual = preprocessing_sha256(frame_warp, fixture_map_paths)
    recorded_keys = set(recorded)
    actual_keys = set(actual)
    if recorded_keys != actual_keys:
        missing = sorted(actual_keys - recorded_keys)
        unexpected = sorted(recorded_keys - actual_keys)
        raise ManifestError(
            "model preprocessing inventory differs from evaluation "
            f"(missing={missing}, unexpected={unexpected})"
        )
    for name in sorted(actual):
        if recorded[name] != actual[name]:
            raise ManifestError(
                f"preprocessing SHA-256 differs from training for {name}"
            )
    return actual


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", P.HERE, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--position-map", action="append", default=[], metavar="FIXTURE=PATH")
    parser.add_argument("--frame-warp", type=Path, default=Path(FW.WARP_PATH))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch < 1:
        parser.error("--epochs and --batch must be positive")

    lock = verify_training_manifests(
        args.train_manifest, args.validation_manifest, args.split_lock
    )
    train_rows = read_manifest(
        args.train_manifest, image_root=args.image_root, verify_images=True
    )
    validation_rows = read_manifest(
        args.validation_manifest, image_root=args.image_root, verify_images=True
    )
    if {row.split for row in train_rows} != {"train"}:
        raise ManifestError("train manifest contains a non-train row")
    if {row.split for row in validation_rows} != {"validation"}:
        raise ManifestError("validation manifest contains a non-validation row")
    train_events = {row.event_uid for row in train_rows}
    validation_events = {row.event_uid for row in validation_rows}
    if train_events & validation_events:
        raise ManifestError("train and validation events overlap")
    train_sessions = {row.session_id for row in train_rows}
    validation_sessions = {row.session_id for row in validation_rows}
    overlapping_sessions = train_sessions & validation_sessions
    if overlapping_sessions and lock.get("split_unit") != "event":
        raise ManifestError("train and validation collection sessions/visits overlap")
    if overlapping_sessions and not lock.get("within_visit_evaluation_acknowledged"):
        raise ManifestError("within-visit event split was not explicitly acknowledged")
    unsupported = validate_class_support(train_rows, validation_rows)
    counts = class_counts(train_rows)
    weights_array = class_weights(counts)

    import torch
    import torchvision

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    fixture_maps, fixture_map_paths = _parse_maps(args.position_map)
    warp = FW.FrameWarp(args.frame_warp) if args.frame_warp else None
    preprocessing = preprocessing_sha256(args.frame_warp, fixture_map_paths)
    train_set = StagingCropSet(
        train_rows,
        args.image_root,
        allowed_split="train",
        fixture_maps=fixture_maps,
        warp=warp,
        seed=args.seed,
    )
    validation_set = StagingCropSet(
        validation_rows,
        args.image_root,
        allowed_split="validation",
        fixture_maps=fixture_maps,
        warp=warp,
        seed=args.seed + 1,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "model.pt"
    if model_path.exists():
        raise FileExistsError(f"refusing to replace existing model {model_path}")
    weights = (
        None
        if args.no_pretrained
        else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CANONICAL_CLASSES))
    model.to(device)
    loss_weights = torch.tensor(weights_array, dtype=torch.float32, device=device)
    loss_function = torch.nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(train_set) / args.batch))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    rng = np.random.default_rng(args.seed)

    started = time.time()
    log_path = args.out / "train_log.csv"
    with log_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_started = time.time()
            total_loss = 0.0
            total_correct = 0
            total = 0
            for features, labels in batches(
                train_set, rng.permutation(len(train_set)), args.batch
            ):
                x = torch.from_numpy(features).to(device)
                y = torch.from_numpy(labels).to(device)
                logits = model(x)
                loss = loss_function(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += float(loss.item()) * len(labels)
                total_correct += int((logits.argmax(dim=1) == y).sum())
                total += len(labels)
            probabilities, validation_labels = evaluate(
                model, validation_set, args.batch, device, torch
            )
            validation_loss = float(
                torch.nn.functional.nll_loss(
                    torch.log(torch.from_numpy(probabilities).clamp_min(1e-9)),
                    torch.from_numpy(validation_labels),
                    weight=loss_weights.cpu(),
                )
            )
            validation_accuracy = float(
                np.mean(probabilities.argmax(axis=1) == validation_labels)
            )
            elapsed = time.time() - epoch_started
            writer.writerow(
                [
                    epoch,
                    round(total_loss / total, 6),
                    round(total_correct / total, 6),
                    round(validation_loss, 6),
                    round(validation_accuracy, 6),
                    round(elapsed, 2),
                ]
            )
            stream.flush()
            print(
                f"epoch {epoch:02d} train_loss={total_loss / total:.4f} "
                f"train_acc={total_correct / total:.4f} "
                f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f} "
                f"seconds={elapsed:.1f}",
                flush=True,
            )

    torch.save(model.state_dict(), model_path)
    threshold = choose_threshold(probabilities, validation_labels, unsupported)
    threshold_path = args.out / "thresholds.json"
    with threshold_path.open("x", encoding="utf-8") as stream:
        json.dump(threshold, stream, indent=2, sort_keys=True)
        stream.write("\n")

    prediction_path = args.out / "validation_predictions.csv"
    fields = [
        "source_image_id",
        "fixture_id",
        "position_id",
        "expected_occupancy",
        "ground_truth_visual_class",
        "predicted_visual_class",
        "defect_score",
        "applicable_threshold",
        "inspection_decision",
        "reason_code",
        "capture_time",
        "event_uid",
    ] + [f"p_{name}" for name in CANONICAL_CLASSES]
    with prediction_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, probability in zip(validation_set.rows, probabilities):
            predicted, score, visual_reason = decide_one(
                probability, float(threshold["t_defect"]), unsupported
            )
            decision, decision_reason = final_decision(predicted, row)
            output = {
                "source_image_id": row.source_image_id,
                "fixture_id": row.fixture_id,
                "position_id": row.position_id,
                "expected_occupancy": row.expected_occupancy,
                "ground_truth_visual_class": row.visual_class,
                "predicted_visual_class": predicted,
                "defect_score": round(score, 8),
                "applicable_threshold": threshold["t_defect"],
                "inspection_decision": decision,
                "reason_code": f"{visual_reason};{decision_reason}",
                "capture_time": row.capture_time,
                "event_uid": row.event_uid,
            }
            output.update(
                {
                    f"p_{name}": round(float(value), 8)
                    for name, value in zip(CANONICAL_CLASSES, probability)
                }
            )
            writer.writerow(output)

    run = {
        "model": "torchvision ResNet-18 with five-output head",
        "classes": list(CANONICAL_CLASSES),
        "defect_classes": list(DEFECT_CLASSES),
        "unsupported_classes": unsupported,
        "five_class_model_ready": not unsupported,
        "safety_note": (
            "unsupported visual classes must resolve to UNKNOWN/UNSURE; no positive "
            "damage claim is allowed without supporting development evidence"
        ),
        "train_counts": counts,
        "validation_counts": class_counts(validation_rows),
        "class_weights": {
            name: float(value) for name, value in zip(CANONICAL_CLASSES, weights_array)
        },
        "train_events": len(train_events),
        "validation_events": len(validation_events),
        "real_images_used": len(
            {row.source_image_sha256 for row in train_rows + validation_rows}
        ),
        "test_rows_loaded": 0,
        "test_images_used": 0,
        "final_test_used": False,
        "threshold_source": "validation_only",
        "evaluation_scope": lock.get("evaluation_scope", "unknown"),
        "independent_visit_test": lock.get("independent_visit_test"),
        "split_lock": {
            "path": str(args.split_lock.resolve()),
            "sha256": sha256_file(args.split_lock),
            "protected_test_events": len(lock.get("events", {}).get("test", [])),
        },
        "manifests": {
            "train": sha256_file(args.train_manifest),
            "validation": sha256_file(args.validation_manifest),
        },
        "source": lock.get("source", {}),
        "preprocessing_sha256": preprocessing,
        "model_pt_sha256": sha256_file(model_path),
        "thresholds_sha256": sha256_file(threshold_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "train_log_sha256": sha256_file(log_path),
        "training_code_sha256": sha256_file(__file__),
        "git_commit": _git_commit(),
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": args.seed,
        "device": str(device),
        "pretrained_weights": "none" if args.no_pretrained else "IMAGENET1K_V1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "wall_seconds": round(time.time() - started, 2),
    }
    run_path = args.out / "run.json"
    with run_path.open("x", encoding="utf-8") as stream:
        json.dump(run, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"threshold": threshold, "run": run}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
