from dataclasses import replace
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import staging5_training_foundation as F
import staging_dataset as D


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


ATTESTATION_SHA256 = _digest("immutable synthetic development-policy receipt")


def _row(
    event: int,
    visual_class: str,
    position: int,
    *,
    usable: str = "true",
) -> D.ManifestRow:
    event_uid = f"session-{event}:event-{event}"
    no_vial = visual_class == "no_vial"
    return D.ManifestRow(
        sample_id=f"{event_uid}:G{position:02d}:{visual_class}",
        event_uid=event_uid,
        session_id=f"session-{event}",
        event_id=f"event-{event}",
        source_image_id=f"image-{event}",
        source_image_relpath=f"event-{event}.png",
        source_image_sha256=_digest(f"source-{event}"),
        capture_time=f"2026-10-28T12:{event:02d}:00-05:00",
        fixture_id="G",
        position_id=f"G{position:02d}",
        expected_occupancy="empty" if no_vial else "occupied",
        visual_class=visual_class,
        inspection_mode="final_inspection",
        gt_sequence="waiting" if no_vial else "",
        designation="development",
        usable=usable,
        split="train",
        split_reason="synthetic unit-test development event",
        label_source="synthetic unit-test fixture",
        annotator_id="unit-test",
        annotation_confidence="1.0",
        review_state="adjudicated",
        notes="no image is created or opened",
    )


def _balanced_rows() -> list[D.ManifestRow]:
    """Four events contain every class; pass has extra correlated positions."""
    rows = []
    for event in range(4):
        for position, visual_class in enumerate(D.CANONICAL_CLASSES, 1):
            rows.append(_row(event, visual_class, position))
        rows.append(_row(event, "pass", 6))
        rows.append(_row(event, "pass", 7))
    return rows


def _pattern_crop() -> np.ndarray:
    y, x = np.indices((24, 32))
    return np.stack(
        [
            (x * 7 + y * 3) % 256,
            (x * 2 + y * 11) % 256,
            (x * 13 + y * 5) % 256,
        ],
        axis=2,
    ).astype(np.uint8)


def test_taxonomy_is_exact_and_schedule_is_balanced_by_distinct_events():
    rows = _balanced_rows()
    schedule = F.build_balanced_epoch_schedule(rows, seed=838, epoch=2)

    assert schedule.taxonomy == (
        "pass",
        "missing_cap",
        "crooked_cap",
        "damaged_vial",
        "no_vial",
    )
    assert schedule.active_classes == schedule.taxonomy
    assert schedule.samples_per_class == 4
    assert schedule.class_counts == {name: 4 for name in schedule.taxonomy}
    assert schedule.distinct_event_support == {
        name: 4 for name in schedule.taxonomy
    }
    assert len(schedule.indices) == 20
    assert len(schedule.sample_ids) == len(set(schedule.sample_ids))
    # Each class draws once from each physical event before any repetition.
    for name in schedule.taxonomy:
        events = {
            event
            for event, actual in zip(schedule.event_uids, schedule.visual_classes)
            if actual == name
        }
        assert len(events) == 4
    assert all(count == 5 for count in schedule.event_counts.values())
    assert [rows[index].sample_id for index in schedule.indices] == list(
        schedule.sample_ids
    )


def test_schedule_is_deterministic_and_independent_of_manifest_row_order():
    rows = _balanced_rows()
    first = F.build_balanced_epoch_schedule(rows, seed=17, epoch=0)
    reversed_rows = list(reversed(rows))
    second = F.build_balanced_epoch_schedule(reversed_rows, seed=17, epoch=0)

    assert first.sample_ids == second.sample_ids
    assert first.event_uids == second.event_uids
    assert first.visual_classes == second.visual_classes
    assert first.sha256 == second.sha256
    assert [reversed_rows[index].sample_id for index in second.indices] == list(
        second.sample_ids
    )
    later = F.build_balanced_epoch_schedule(rows, seed=17, epoch=1)
    assert later.sha256 != first.sha256
    first_pass_by_event = {
        event: sample
        for event, visual, sample in zip(
            first.event_uids, first.visual_classes, first.sample_ids
        )
        if visual == "pass"
    }
    later_pass_by_event = {
        event: sample
        for event, visual, sample in zip(
            later.event_uids, later.visual_classes, later.sample_ids
        )
        if visual == "pass"
    }
    # Pass has three position rows in every event.  Epoch selection rotates the
    # representative position instead of repeatedly using one correlated crop.
    assert first_pass_by_event.keys() == later_pass_by_event.keys()
    assert all(
        first_pass_by_event[event] != later_pass_by_event[event]
        for event in first_pass_by_event
    )


def test_schedule_undersamples_without_replacement_and_rejects_oversampling():
    rows = _balanced_rows()
    schedule = F.build_balanced_epoch_schedule(
        rows, seed=5, samples_per_class=3
    )
    assert len(schedule.indices) == 15
    assert schedule.class_counts == {name: 3 for name in D.CANONICAL_CLASSES}
    assert len(set(schedule.sample_ids)) == 15

    with pytest.raises(D.ManifestError, match="distinct-event capacity"):
        F.build_balanced_epoch_schedule(rows, seed=5, samples_per_class=5)
    with pytest.raises(D.ManifestError, match="must be positive"):
        F.build_balanced_epoch_schedule(rows, seed=5, samples_per_class=0)


def test_skewed_row_counts_cannot_make_one_event_dominate_a_class():
    # Regression for event IDs 0, 1 and 3 under seed 2.  Event 3 contains many
    # more pass positions, but it still receives one pass contribution.
    required = ("pass", "missing_cap", "crooked_cap", "no_vial")
    position = {
        "pass": 1,
        "missing_cap": 2,
        "crooked_cap": 3,
        "no_vial": 5,
    }
    rows = [
        _row(event, visual, position[visual])
        for event in (0, 1, 3)
        for visual in required
    ]
    rows.extend(_row(3, "pass", extra) for extra in range(6, 26))
    schedule = F.build_balanced_epoch_schedule(rows, seed=2)

    assert schedule.samples_per_class == 3
    assert schedule.distinct_event_support == {
        "pass": 3,
        "missing_cap": 3,
        "crooked_cap": 3,
        "damaged_vial": 0,
        "no_vial": 3,
    }
    for visual in required:
        events = [
            event
            for event, actual in zip(schedule.event_uids, schedule.visual_classes)
            if actual == visual
        ]
        assert len(events) == len(set(events)) == 3
        assert set(events) == {
            "session-0:event-0",
            "session-1:event-1",
            "session-3:event-3",
        }


def test_unsupported_class_remains_in_taxonomy_without_fabricated_samples():
    rows = [
        row
        for row in _balanced_rows()
        if row.visual_class != "damaged_vial"
    ]
    schedule = F.build_balanced_epoch_schedule(rows, seed=23)

    assert schedule.taxonomy == tuple(D.CANONICAL_CLASSES)
    assert schedule.active_classes == (
        "pass",
        "missing_cap",
        "crooked_cap",
        "no_vial",
    )
    assert schedule.class_counts["damaged_vial"] == 0
    assert "damaged_vial" not in schedule.visual_classes
    assert set(schedule.sample_ids).issubset({row.sample_id for row in rows})


def test_schedule_rejects_pass_only_or_any_missing_required_class():
    with pytest.raises(D.ManifestError, match="missing required classes") as pass_only:
        F.build_balanced_epoch_schedule([_row(0, "pass", 1)], seed=1)
    assert "missing_cap" in str(pass_only.value)
    assert "crooked_cap" in str(pass_only.value)
    assert "no_vial" in str(pass_only.value)

    without_no_vial = [
        row for row in _balanced_rows() if row.visual_class != "no_vial"
    ]
    with pytest.raises(D.ManifestError, match="only damaged_vial may be absent"):
        F.build_balanced_epoch_schedule(without_no_vial, seed=1)


def test_schedule_rejects_nontrain_and_protected_rows_before_sampling():
    row = _row(1, "pass", 1)
    with pytest.raises(D.ManifestError, match="split='train' only"):
        F.build_balanced_epoch_schedule(
            [replace(row, split="validation")], seed=1
        )
    with pytest.raises(D.ProtectedTestError, match="protected-test evidence"):
        F.build_balanced_epoch_schedule(
            [replace(row, designation="protected_test", split="test")], seed=1
        )
    with pytest.raises(D.ManifestError, match="split='train' only"):
        F.build_balanced_epoch_schedule(
            [row, replace(_row(1, "missing_cap", 2), split="validation")],
            seed=1,
        )


def test_schedule_rejects_malformed_or_leakage_like_event_assignments():
    first = _row(1, "pass", 1)
    second = _row(2, "missing_cap", 2)

    with pytest.raises(D.ManifestError, match="exact five-class taxonomy"):
        F.build_balanced_epoch_schedule(
            [replace(first, visual_class="almost_pass")], seed=1
        )
    with pytest.raises(D.ManifestError, match="duplicate sample_id"):
        F.build_balanced_epoch_schedule(
            [first, replace(second, sample_id=first.sample_id)], seed=1
        )
    with pytest.raises(D.ManifestError, match="multiple event_uids"):
        F.build_balanced_epoch_schedule(
            [
                first,
                replace(
                    second,
                    event_uid="alias-for-event-1",
                    session_id=first.session_id,
                    event_id=first.event_id,
                ),
            ],
            seed=1,
        )
    with pytest.raises(D.ManifestError, match="byte-identical source content"):
        F.build_balanced_epoch_schedule(
            [
                first,
                replace(second, source_image_sha256=first.source_image_sha256),
            ],
            seed=1,
        )
    with pytest.raises(D.ManifestError, match="inconsistent path/hash metadata"):
        F.build_balanced_epoch_schedule(
            [first, replace(_row(1, "missing_cap", 2), source_image_relpath="other.png")],
            seed=1,
        )
    with pytest.raises(D.ManifestError, match="multiple source images"):
        F.build_balanced_epoch_schedule(
            [
                first,
                replace(
                    _row(1, "missing_cap", 2),
                    source_image_id="second-image-same-event",
                ),
            ],
            seed=1,
        )
    with pytest.raises(D.ManifestError, match="no usable rows"):
        F.build_balanced_epoch_schedule(
            [replace(first, usable="false")], seed=1
        )


def test_train_augmentation_is_seeded_per_sample_and_order_independent():
    crop = _pattern_crop()
    policy = F.CandidateTrainAugmentationPolicy(
        exposure_gain=(0.85, 1.15),
        contrast_gain=(0.9, 1.1),
        blur_sigma_px=(0.2, 0.8),
        noise_sigma_8bit=(1.0, 3.0),
        max_translation_px=2,
        development_attestation_sha256=ATTESTATION_SHA256,
    )
    arguments = dict(
        split="train",
        policy=policy,
        seed=44,
        epoch=3,
        sample_id="session-1:event-1:G01",
    )
    np.random.seed(999)
    first = F.apply_candidate_split_augmentation(crop, **arguments)
    np.random.seed(1)
    second = F.apply_candidate_split_augmentation(crop, **arguments)

    assert np.array_equal(first, second)
    assert first.shape == crop.shape
    assert first.dtype == np.uint8
    assert not np.array_equal(first, crop)
    assert not np.array_equal(
        first,
        F.apply_candidate_split_augmentation(crop, **{**arguments, "epoch": 4}),
    )
    assert not np.array_equal(
        first,
        F.apply_candidate_split_augmentation(
            crop, **{**arguments, "sample_id": "session-1:event-1:G02"}
        ),
    )


def test_validation_and_test_paths_are_byte_identical_and_not_augmented():
    crop = _pattern_crop()
    policy = F.CandidateTrainAugmentationPolicy(
        exposure_gain=(0.5, 1.5),
        contrast_gain=(0.5, 1.5),
        blur_sigma_px=(1.0, 2.0),
        noise_sigma_8bit=(10.0, 20.0),
        max_translation_px=3,
        development_attestation_sha256=ATTESTATION_SHA256,
    )
    for split in ("validation", "test"):
        output = F.apply_candidate_split_augmentation(
            crop,
            split=split,
            policy=policy,
            seed=0,
            epoch=0,
            sample_id="synthetic-only",
        )
        assert np.array_equal(output, crop)
        assert output is not crop
        assert not np.shares_memory(output, crop)


def test_identity_is_default_and_policy_fingerprint_is_auditable():
    crop = _pattern_crop()
    identity = F.CandidateTrainAugmentationPolicy()
    output = F.apply_candidate_split_augmentation(
        crop,
        split="train",
        policy=identity,
        seed=0,
        epoch=0,
        sample_id="synthetic-only",
    )
    assert np.array_equal(output, crop)
    assert len(identity.sha256) == 64
    assert identity.is_identity is True
    assert identity.development_attestation_sha256 is None
    assert identity.sha256 == F.CandidateTrainAugmentationPolicy().sha256
    assert identity.sha256 != F.CandidateTrainAugmentationPolicy(
        noise_sigma_8bit=(0.0, 1.0),
        development_attestation_sha256=ATTESTATION_SHA256,
    ).sha256


def test_nonidentity_candidate_requires_valid_attestation_in_fingerprint():
    with pytest.raises(D.ManifestError, match="requires an immutable"):
        F.CandidateTrainAugmentationPolicy(noise_sigma_8bit=(0.0, 1.0))
    for malformed in ("", "0" * 63, "A" * 64, 123):
        with pytest.raises(D.ManifestError, match="64 lowercase hex digits"):
            F.CandidateTrainAugmentationPolicy(
                noise_sigma_8bit=(0.0, 1.0),
                development_attestation_sha256=malformed,
            )

    first = F.CandidateTrainAugmentationPolicy(
        noise_sigma_8bit=(0.0, 1.0),
        development_attestation_sha256=ATTESTATION_SHA256,
    )
    second = F.CandidateTrainAugmentationPolicy(
        noise_sigma_8bit=(0.0, 1.0),
        development_attestation_sha256=_digest("different immutable receipt"),
    )
    assert first.is_identity is False
    assert first.sha256 != second.sha256


def test_augmentation_fails_loudly_on_malformed_inputs_and_ranges():
    crop = _pattern_crop()
    with pytest.raises(D.ManifestError, match="strictly positive"):
        F.CandidateTrainAugmentationPolicy(exposure_gain=(0.0, 1.0))
    with pytest.raises(D.ManifestError, match="low <= high"):
        F.CandidateTrainAugmentationPolicy(noise_sigma_8bit=(2.0, 1.0))
    with pytest.raises(D.ManifestError, match="non-negative integer"):
        F.CandidateTrainAugmentationPolicy(max_translation_px=-1)
    with pytest.raises(D.ManifestError, match="uint8"):
        F.apply_candidate_split_augmentation(
            crop.astype(np.float32),
            split="train",
            policy=F.CandidateTrainAugmentationPolicy(),
            seed=0,
            epoch=0,
            sample_id="synthetic-only",
        )
    with pytest.raises(D.ManifestError, match="split must be one of"):
        F.apply_candidate_split_augmentation(
            crop,
            split="holdout",
            policy=F.CandidateTrainAugmentationPolicy(),
            seed=0,
            epoch=0,
            sample_id="synthetic-only",
        )
    with pytest.raises(D.ManifestError, match="less than half"):
        F.apply_candidate_split_augmentation(
            crop,
            split="train",
            policy=F.CandidateTrainAugmentationPolicy(
                max_translation_px=12,
                development_attestation_sha256=ATTESTATION_SHA256,
            ),
            seed=0,
            epoch=0,
            sample_id="synthetic-only",
        )
    with pytest.raises(D.ManifestError, match="longer side"):
        F.apply_candidate_split_augmentation(
            crop,
            split="train",
            policy=F.CandidateTrainAugmentationPolicy(
                blur_sigma_px=(33.0, 33.0),
                development_attestation_sha256=ATTESTATION_SHA256,
            ),
            seed=0,
            epoch=0,
            sample_id="synthetic-only",
        )
    with pytest.raises(D.ManifestError, match="numeric safety limits"):
        F.apply_candidate_split_augmentation(
            crop,
            split="train",
            policy=F.CandidateTrainAugmentationPolicy(
                exposure_gain=(1e38, 1e38),
                development_attestation_sha256=ATTESTATION_SHA256,
            ),
            seed=0,
            epoch=0,
            sample_id="synthetic-only",
        )
