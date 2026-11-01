"""Safety and evidence tests for the portable staging image collector."""

from __future__ import annotations

import binascii
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import threading
import time
import zlib

import pytest


TTL = Path(__file__).resolve().parents[1]
REPO = TTL.parent
sys.path.insert(0, str(TTL))

import staging_collector as C  # noqa: E402
from gb10 import build_staging_collection_kit as B  # noqa: E402


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
        ">I", binascii.crc32(kind + payload) & 0xFFFFFFFF
    )


def png_bytes(width: int = 640, height: int = 480, value: int = 0) -> bytes:
    pixel = bytes((value % 256, (value * 3) % 256, (value * 7) % 256))
    raw = b"".join(b"\x00" + pixel * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, level=1))
        + _chunk(b"IEND", b"")
    )


def _jpeg_segment(marker: int, payload: bytes) -> bytes:
    return b"\xff" + bytes((marker,)) + struct.pack(">H", len(payload) + 2) + payload


def framed_but_truncated_jpeg(width: int = 640, height: int = 480) -> bytes:
    counts = b"\x01" + b"\x00" * 15
    return (
        b"\xff\xd8"
        + _jpeg_segment(0xDB, b"\x00" + b"\x01" * 64)
        + _jpeg_segment(0xC4, b"\x00" + counts + b"\x00" + b"\x10" + counts + b"\x00")
        + _jpeg_segment(0xC0, b"\x08" + struct.pack(">HH", height, width) + b"\x01\x01\x11\x00")
        + _jpeg_segment(0xDA, b"\x01\x01\x00\x00\x3f\x00")
        + b"\x3f\xff\xd9"
    )


def write_baseline(path: Path, *, fixture_s: bool = False, same_map: bool = False) -> Path:
    prior = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    maps = {"G": "a" * 64}
    if fixture_s:
        maps["S"] = maps["G"] if same_map else "b" * 64
    value = {
        "schema_version": 1,
        "baseline_id": "synthetic-five-output-v1",
        "canonical_classes": list(C.CANONICAL_CLASSES),
        "training_data_scope": "synthetic_only",
        "frozen_at_utc": prior,
        "model_sha256": "1" * 64,
        "thresholds_sha256": "2" * 64,
        "decision_policy_sha256": "3" * 64,
        "decision_policy_version": "policy-v1",
        "code_sha256": {"inference": "4" * 64},
        "preprocessing_sha256": {"warp": "5" * 64},
        "fixture_map_sha256": maps,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def init_rehearsal(tmp_path: Path) -> tuple[Path, Path, Path]:
    camera = tmp_path / "camera"
    camera.mkdir()
    camera.joinpath("old.png").write_bytes(png_bytes(value=99))
    output = tmp_path / "rehearsal"
    session = C.initialize_session(
        role="rehearsal",
        camera_dir=camera,
        output_root=output,
        fixture="G",
        expected_resolution=(640, 480),
        operator_id="operator",
        camera_id="camera-profile",
        baseline_freeze=None,
        protected_handoff=None,
        session_id="rehearsal-session",
    )
    return session, camera, output


def accept_new(session: Path, camera: Path, event: dict[str, str], value: int) -> dict[str, str]:
    source = camera / f"new-{event['event_id']}-{value}.png"
    source.write_bytes(png_bytes(value=value))
    before = source.read_bytes()
    pending = C.ingest_capture(session, event, stability_seconds=0.01, wait_seconds=0.5)
    result = C.finalize_capture(session, accepted=True, note="physical setup confirmed")
    assert source.read_bytes() == before
    assert pending["sha256"] == hashlib.sha256(before).hexdigest()
    return result


@pytest.mark.parametrize("role,total", [("train", 65), ("validation", 30),
                                         ("protected_test", 30), ("rehearsal", 6)])
def test_plan_is_event_atomic_complete_and_preserves_taxonomy(role, total):
    plan, intents = C.build_plan(role, "G", "session-a", "f" * 64)
    assert len(plan) == total
    assert len(intents) == total * 30
    assert {row["intended_visual_class"] for row in intents} == set(C.CANONICAL_CLASSES)
    for event in plan:
        rows = [row for row in intents if row["event_id"] == event["event_id"]]
        assert len(rows) == 30
        assert {row["position_id"] for row in rows} == {f"G{i:02d}" for i in range(1, 31)}
        assert {row["usable"] for row in rows} == {"false"}
        assert {row["review_state"] for row in rows} == {"pending_independent_review"}
    passes = [row for row in intents if row["intended_visual_class"] == "pass"]
    assert all(80 <= int(row["planned_vial_axis_angle_deg"]) <= 90 for row in passes)
    assert all(row["planned_vial_axis_angle_deg"] == row["planned_cap_axis_angle_deg"] for row in passes)
    empties = [row for row in intents if row["intended_visual_class"] == "no_vial"]
    assert all(
        (row["expected_occupancy"], row["intended_gt_sequence"])
        in {("occupied", "missed"), ("empty", "waiting")}
        for row in empties
    )


def test_baseline_freeze_and_fixture_s_fail_closed(tmp_path):
    baseline = write_baseline(tmp_path / "baseline.json")
    result = C.validate_baseline_freeze(baseline, "G")
    assert result["baseline_id"] == "synthetic-five-output-v1"
    with pytest.raises(C.CollectionError, match="no calibrated S"):
        C.validate_baseline_freeze(baseline, "S")
    write_baseline(baseline, fixture_s=True, same_map=True)
    with pytest.raises(C.CollectionError, match="own calibrated map"):
        C.validate_baseline_freeze(baseline, "S")
    write_baseline(baseline, fixture_s=True)
    assert C.validate_baseline_freeze(baseline, "S")["fixture_map_sha256"] == "b" * 64


def test_image_validator_rejects_incomplete_rasters_and_coefficient_streams():
    tiny_png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(b"x"))
        + _chunk(b"IEND", b"")
    )
    with pytest.raises(C.CollectionError, match="incomplete|safety limit"):
        C.image_dimensions(tiny_png)
    with pytest.raises(C.CollectionError, match="truncated JPEG coefficient stream"):
        C.image_dimensions(framed_but_truncated_jpeg())


def test_role_roots_and_protected_storage_are_disjoint_before_write(tmp_path):
    development = C.prepare_output_root(tmp_path / "development", "train")
    assert json.loads((development / ".vial_guard_collection_root.json").read_text())["root_kind"] == "development"
    protected = C.prepare_output_root(tmp_path / "protected", "protected_test")
    assert json.loads((protected / ".vial_guard_collection_root.json").read_text())["root_kind"] == "protected_test"
    with pytest.raises(C.CollectionError, match="already reserved"):
        C.prepare_output_root(development, "rehearsal")


def test_runtime_event_cards_match_frozen_plan_and_are_commitment_bound(tmp_path):
    session, _, output = init_rehearsal(tmp_path)
    state = json.loads((session / "session.json").read_text(encoding="utf-8"))
    plan = C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)
    cards_dir = session / "records/event_cards"
    manifest_path = cards_dir / C.EVENT_CARD_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_names = {f"{event['event_id']}_{event['family']}.md" for event in plan}
    assert {path.name for path in cards_dir.iterdir()} == (
        expected_names | {C.EVENT_CARD_MANIFEST_NAME}
    )
    assert manifest["card_count"] == state["event_cards_count"] == len(plan) == 6
    assert manifest["source_plan_sha256"] == state["plan_sha256"]
    assert manifest["source_intended_positions_sha256"] == state["intended_positions_sha256"]
    assert C.sha256_file(manifest_path) == state["event_cards_manifest_sha256"]
    assert cards_dir.stat().st_mode & 0o777 == 0o750
    assert all(path.stat().st_mode & 0o777 == 0o440 for path in cards_dir.iterdir())
    commitment = json.loads(
        (output / ".session_commitments" / f"{session.name}.json").read_text(encoding="utf-8")
    )
    assert commitment["event_cards_manifest_sha256"] == state["event_cards_manifest_sha256"]
    assert commitment["event_cards_count"] == len(plan)
    first = plan[0]
    card = (cards_dir / f"{first['event_id']}_{first['family']}.md").read_text(encoding="utf-8")
    assert first["event_uid"] in card and first["primary_targets"] in card
    assert "INTENDED PHYSICAL SETUP ONLY" in card and "NOT VISIBLE GROUND TRUTH" in card
    assert "plan.csv" in card and "intended_positions.csv" in card and "typing `READY`" in card
    assert "Operator record" not in card and "________________" not in card
    C.load_session(session)

    s_case = tmp_path / "fixture-s"
    s_case.mkdir()
    camera = s_case / "camera"
    camera.mkdir()
    baseline = write_baseline(s_case / "baseline.json", fixture_s=True)
    s_session = C.initialize_session(
        role="train",
        camera_dir=camera,
        output_root=s_case / "development",
        fixture="S",
        expected_resolution=(640, 480),
        operator_id="operator",
        camera_id="camera-profile",
        baseline_freeze=baseline,
        protected_handoff=None,
        session_id="train-fixture-s",
    )
    s_plan = C.read_csv(s_session / "plan.csv", C.PLAN_COLUMNS)
    s_card = (
        s_session / "records/event_cards" /
        f"{s_plan[0]['event_id']}_{s_plan[0]['family']}.md"
    ).read_text(encoding="utf-8")
    assert "`S01`–`S30`" in s_card and s_plan[0]["primary_targets"] in s_card
    assert "G01" not in s_card
    C.load_session(s_session)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("edit", "event card changed"),
        ("delete", "directory membership"),
        ("extra", "directory membership"),
        ("symlink", "symlink inside session evidence"),
        ("manifest", "manifest differs from the immutable session state"),
    ],
)
def test_runtime_event_card_tamper_fails_closed(tmp_path, mutation, error):
    session, _, _ = init_rehearsal(tmp_path)
    cards_dir = session / "records/event_cards"
    card = next(path for path in sorted(cards_dir.glob("E*.md")))
    manifest = cards_dir / C.EVENT_CARD_MANIFEST_NAME
    if mutation == "edit":
        card.chmod(0o640)
        card.write_bytes(card.read_bytes() + b"\ntamper\n")
        card.chmod(0o440)
    elif mutation == "delete":
        card.unlink()
    elif mutation == "extra":
        extra = cards_dir / "EXTRA.md"
        extra.write_text("unexpected", encoding="utf-8")
        extra.chmod(0o440)
    elif mutation == "symlink":
        card.unlink()
        card.symlink_to(manifest.name)
    else:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["warning"] = "changed"
        manifest.chmod(0o640)
        manifest.write_text(json.dumps(value), encoding="utf-8")
        manifest.chmod(0o440)
    with pytest.raises(C.CollectionError, match=error):
        C.load_session(session)


def test_runtime_event_card_hash_chain_ends_at_external_commitment(tmp_path):
    session, _, _ = init_rehearsal(tmp_path)
    cards_dir = session / "records/event_cards"
    card = next(path for path in sorted(cards_dir.glob("E*.md")))
    card.chmod(0o640)
    card.write_bytes(card.read_bytes() + b"\ntamper\n")
    card.chmod(0o440)
    manifest_path = cards_dir / C.EVENT_CARD_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(value for value in manifest["files"] if value["path"] == card.name)
    entry["bytes"] = card.stat().st_size
    entry["sha256"] = C.sha256_file(card)
    manifest_path.chmod(0o640)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.chmod(0o440)
    state_path = session / "session.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["event_cards_manifest_sha256"] = C.sha256_file(manifest_path)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(C.CollectionError, match="session.json differs from its external session commitment"):
        C.load_session(session)


def test_new_stable_capture_preserves_source_and_retake_selection(tmp_path):
    session, camera, _ = init_rehearsal(tmp_path)
    plan = C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)
    card_hashes = {
        path.name: C.sha256_file(path)
        for path in (session / "records/event_cards").iterdir()
    }
    first = accept_new(session, camera, plan[0], 1)
    second = accept_new(session, camera, plan[0], 2)
    attempts = C.read_csv(session / "records/capture_attempts.csv", C.ATTEMPT_COLUMNS)
    assert [row["attempt"] for row in attempts] == ["1", "2"]
    assert all(row["ready_confirmed_at_utc"] for row in attempts)
    assert all(
        C.parse_utc(row["ready_confirmed_at_utc"], "ready")
        <= C.parse_utc(row["capture_time_utc"], "capture")
        for row in attempts
    )
    assert C.selected_attempts(attempts)[plan[0]["event_id"]]["capture_name"] == second["capture_name"]
    assert (session / "captures" / first["capture_name"]).exists()
    report = C.verify_session(session, verify_archive=False)
    assert report["events_selected"] == 1 and report["capture_attempts"] == 2
    assert card_hashes == {
        path.name: C.sha256_file(path)
        for path in (session / "records/event_cards").iterdir()
    }


def test_duplicate_digest_and_wrong_resolution_do_not_advance(tmp_path):
    session, camera, _ = init_rehearsal(tmp_path)
    plan = C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)
    first_bytes = png_bytes(value=4)
    camera.joinpath("first.png").write_bytes(first_bytes)
    C.ingest_capture(session, plan[0], stability_seconds=0.01, wait_seconds=0.5)
    C.finalize_capture(session, accepted=True, note="ok")
    camera.joinpath("duplicate.png").write_bytes(first_bytes)
    with pytest.raises(C.CollectionError, match="duplicate image"):
        C.ingest_capture(session, plan[1], stability_seconds=0.01, wait_seconds=0.25)
    camera.joinpath("wrong.png").write_bytes(png_bytes(320, 240, 5))
    with pytest.raises(C.CollectionError, match="differs from locked"):
        C.ingest_capture(session, plan[1], stability_seconds=0.01, wait_seconds=0.25)
    assert len(C.read_csv(session / "records/capture_attempts.csv", C.ATTEMPT_COLUMNS)) == 1


def test_multiple_new_camera_files_are_refused_without_being_consumed(tmp_path):
    session, camera, _ = init_rehearsal(tmp_path)
    event = C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)[0]
    camera.joinpath("one.png").write_bytes(png_bytes(value=31))
    camera.joinpath("two.png").write_bytes(png_bytes(value=32))
    with pytest.raises(C.CollectionError, match="multiple new camera images"):
        C.ingest_capture(session, event, stability_seconds=0.01, wait_seconds=0.5)
    assert not list((session / "captures").iterdir())
    seen = json.loads((session / "records/camera_seen.json").read_text())
    assert "one.png" not in seen and "two.png" not in seen


def test_second_file_during_stability_window_and_camera_directory_replacement_are_refused(tmp_path):
    session, camera, _ = init_rehearsal(tmp_path)
    event = C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)[0]
    camera.joinpath("one.png").write_bytes(png_bytes(value=34))

    def add_second() -> None:
        time.sleep(0.03)
        camera.joinpath("two.png").write_bytes(png_bytes(value=35))

    writer = threading.Thread(target=add_second)
    writer.start()
    with pytest.raises(C.CollectionError, match="multiple new camera images|camera folder changed"):
        C.ingest_capture(session, event, stability_seconds=0.1, wait_seconds=0.5)
    writer.join()
    assert not list((session / "captures").iterdir())

    original = tmp_path / "camera-original"
    camera.rename(original)
    camera.mkdir()
    camera.joinpath("replacement.png").write_bytes(png_bytes(value=36))
    with pytest.raises(C.CollectionError, match="camera directory identity changed"):
        C.ingest_capture(session, event, stability_seconds=0.01, wait_seconds=0.5)


def test_recovery_preserves_unconfirmed_capture_without_selecting_it(tmp_path):
    session, camera, _ = init_rehearsal(tmp_path)
    event = C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)[0]
    camera.joinpath("interrupted.png").write_bytes(png_bytes(value=8))
    pending = C.ingest_capture(session, event, stability_seconds=0.01, wait_seconds=0.5)
    C.recover_pending(session)
    attempts = C.read_csv(session / "records/capture_attempts.csv", C.ATTEMPT_COLUMNS)
    assert attempts[0]["status"] == "unconfirmed_recovered"
    assert not C.selected_attempts(attempts)
    assert (session / "captures" / pending["capture_name"]).exists()


def test_recovery_cannot_mutate_before_writer_lock_and_quarantines_orphans(tmp_path):
    session, camera, _ = init_rehearsal(tmp_path)
    event = C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)[0]
    camera.joinpath("interrupted.png").write_bytes(png_bytes(value=33))
    C.ingest_capture(session, event, stability_seconds=0.01, wait_seconds=0.5)
    attempts_path = session / "records/capture_attempts.csv"
    before = attempts_path.read_bytes()
    with (session / "records/writer.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(C.CollectionError, match="another collector"):
            C.recover_pending(session)
    assert attempts_path.read_bytes() == before
    assert (session / "records/pending_capture.json").is_file()
    C.recover_pending(session)
    orphan = session / "captures/.capture-building-test"
    orphan.write_bytes(b"interrupted")
    C.recover_pending(session)
    assert not orphan.exists()
    assert (session / "records/interrupted_capture_copies/.capture-building-test").is_file()


def test_complete_rehearsal_seal_and_archive_tamper_detection(tmp_path):
    session, camera, _ = init_rehearsal(tmp_path)
    plan = C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)
    for index, event in enumerate(plan, start=1):
        accept_new(session, camera, event, index)
    result = C.seal_session(session)
    assert result["seal_status"] == "COMPLETE"
    assert Path(result["archive"]).is_file()
    assert C.verify_session(session)["collection_complete"] is True
    with tarfile.open(result["archive"], "r") as bundle:
        archive_names = {member.name for member in bundle.getmembers() if member.isfile()}
    assert "records/event_cards/MANIFEST.json" in archive_names
    assert len({name for name in archive_names if name.startswith("records/event_cards/E")}) == 6
    review = C.read_csv(session / "records/independent_review_template.csv", C.REVIEW_COLUMNS)
    assert len(review) == 6 * 30
    assert {row["actual_visual_class"] for row in review} == {""}
    assert {row["usable"] for row in review} == {"false"}
    capture = next((session / "captures").iterdir())
    capture.chmod(0o644)
    capture.write_bytes(capture.read_bytes() + b"tamper")
    with pytest.raises(C.CollectionError, match="byte count changed|SHA-256 changed"):
        C.verify_session(session)


def test_incomplete_collection_requires_explicit_partial_seal(tmp_path):
    session, camera, _ = init_rehearsal(tmp_path)
    plan = C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)
    accept_new(session, camera, plan[0], 21)
    C.skip_event(session, plan[1], "approved damaged specimen unavailable")
    with pytest.raises(C.CollectionError, match="incomplete"):
        C.seal_session(session)
    result = C.seal_session(session, allow_partial=True)
    assert result["seal_status"] == "PARTIAL"
    assert result["events_pending"] == 4
    assert result["events_skipped"] == 1


def test_sealed_session_refuses_all_collection_mutators_and_exports_blinded_packets(tmp_path):
    session, camera, _ = init_rehearsal(tmp_path)
    event = C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)[0]
    accept_new(session, camera, event, 41)
    C.seal_session(session, allow_partial=True)
    evidence_hash = C.sha256_file(session / "sealed/evidence.tar")
    camera.joinpath("after-seal.png").write_bytes(png_bytes(value=42))
    with pytest.raises(C.CollectionError, match="sealed"):
        C.ingest_capture(session, event, stability_seconds=0.01, wait_seconds=0.5)
    with pytest.raises(C.CollectionError, match="sealed"):
        C.finalize_capture(session, accepted=True, note="not allowed")
    with pytest.raises(C.CollectionError, match="sealed"):
        C.skip_event(session, C.read_csv(session / "plan.csv", C.PLAN_COLUMNS)[1], "not allowed")
    with pytest.raises(C.CollectionError, match="sealed"):
        C.recover_pending(session)

    review_root = tmp_path / "review-export"
    result = C.export_review_working_copy(session, review_root)
    assert set(result) == {"reviewer_a_packet", "reviewer_b_packet", "coordinator_only", "manifest"}
    assert not any("event_cards" in path.parts for path in review_root.rglob("*"))
    assert not list(review_root.rglob("*.md"))
    for packet_name in ("reviewer_a_packet", "reviewer_b_packet"):
        packet = review_root / packet_name
        rows = C.read_csv(packet / "review.csv", C.BLINDED_REVIEW_COLUMNS)
        assert rows and all(row["review_image_id"].startswith("img-") for row in rows)
        assert all("event" not in key and "source" not in key and "intent" not in key
                   for key in rows[0])
        assert all((packet / row["review_image_relpath"]).is_file() for row in rows)
    assert (review_root / "coordinator_only/reviewer_join_map.csv").is_file()
    assert C.sha256_file(session / "sealed/evidence.tar") == evidence_hash


def test_official_session_records_freeze_and_never_runs_inference(tmp_path):
    camera = tmp_path / "camera"
    camera.mkdir()
    baseline = write_baseline(tmp_path / "baseline.json")
    session = C.initialize_session(
        role="train",
        camera_dir=camera,
        output_root=tmp_path / "development",
        fixture="G",
        expected_resolution=(1920, 1080),
        operator_id="operator",
        camera_id="fixed-overhead",
        baseline_freeze=baseline,
        protected_handoff=None,
        session_id="train-visit-a",
    )
    state = json.loads((session / "session.json").read_text())
    assert state["baseline_freeze"]["freeze_file_sha256"] == C.sha256_file(baseline)
    assert C.sha256_file(session / "records/baseline_freeze.json") == C.sha256_file(baseline)
    assert state["inference_performed"] is False
    assert state["manufacturing_control_performed"] is False
    assert state["role"] == "train" and state["designation"] == "development"


def test_package_build_checksum_extract_and_help(tmp_path):
    out = tmp_path / "kit"
    run = B.build(out)
    subprocess.run(["sha256sum", "-c", "PACKAGE_SHA256SUMS"], cwd=out, check=True,
                   capture_output=True, text=True)
    poisoned = dict(os.environ)
    poisoned.update({
        "PATH": str(tmp_path),
        "TAR_OPTIONS": "--this-option-must-be-unset",
        "PYTHONPATH": "/untrusted/python/path",
        "PYTHONHOME": "/untrusted/python/home",
    })
    result = subprocess.run(["/usr/bin/bash", str(run), "--help"], input="", capture_output=True,
                            text=True, timeout=10, env=poisoned)
    assert result.returncode == 0
    assert "inference-free Vial Guard staging" in result.stdout
    extracted = tmp_path / "extracted"
    subprocess.run(["bash", str(run), "--extract-only", str(extracted)], check=True,
                   capture_output=True, text=True, timeout=10)
    assert (extracted / "staging_collector.py").is_file()
    assert (extracted / "SHA256SUMS").is_file()
    assert {path.name for path in out.iterdir()} == {
        B.RUN_NAME,
        f"{B.RUN_NAME}.sha256",
        "README_USB.txt",
        "CODEX_TERMINAL_PROMPT.txt",
        "STAGING_COLLECTION_V2.md",
        B.ALL_EVENTS_NAME,
        "PACKAGE_CONTENTS.txt",
        "PACKAGE_SHA256SUMS",
    }
    assert {path.name for path in extracted.iterdir()} == (
        set(B.FILES) | {"SHA256SUMS", B.ALL_EVENTS_NAME}
    )
    guide = out / B.ALL_EVENTS_NAME
    text = guide.read_text(encoding="utf-8")
    assert guide.read_bytes() == (extracted / B.ALL_EVENTS_NAME).read_bytes()
    assert sum(B.ROLE_CARD_COUNTS.values()) == 131
    assert sum(B.SHARED_ROLE_CARD_COUNTS.values()) == 101
    assert text.count("### Rehearsal E") == 6
    assert text.count("### Training E") == 65
    assert text.count("### Validation E") == 30
    assert "### Protected" not in text and "Protected test E" not in text
    assert "## Protected test — public aggregate only (30 custodian-generated events)" in text
    for quantity in ("| Correctly capped/aligned whole-vial lean | 6 |",
                     "| Visibly crooked/unseated cap | 5 |",
                     "| Missing cap | 4 |", "| Approved visible vial damage | 5 |",
                     "| `no_vial` MISSED/WAITING | 4 |", "| Mixed/confounder | 6 |"):
        assert quantity in text
    assert "records/event_cards/" in text and "NOT RUNNABLE" in text
    assert "plan.csv" in text and "read-only fixture-g" in text.lower()
    assert not any(path.is_dir() for path in out.iterdir())
    package_paths = {
        line.split("\t", 1)[0]
        for line in (out / "PACKAGE_CONTENTS.txt").read_text(encoding="utf-8").splitlines()
    }
    assert package_paths == {path.name for path in out.iterdir()}
    checksum_paths = {
        line.split(maxsplit=1)[1]
        for line in (out / "PACKAGE_SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert checksum_paths == package_paths - {"PACKAGE_SHA256SUMS"}
    assert not any(path.suffix.lower() in {".onnx", ".pt", ".pth", ".engine"}
                   for path in extracted.rglob("*"))
    with pytest.raises(FileExistsError):
        B.build(out)


def test_cli_noninteractive_missing_inputs_fails_without_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TTL / "staging_collector.py"), "collect", "--role", "rehearsal"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 1
    assert "Camera image folder is required" in result.stderr
    assert "Traceback" not in result.stderr


def _init_official(tmp_path: Path, *, role: str, session_id: str, baseline: Path,
                   output: Path, handoff: Path | None = None) -> tuple[Path, Path]:
    camera = tmp_path / f"camera-{session_id}"
    camera.mkdir()
    session = C.initialize_session(
        role=role,
        camera_dir=camera,
        output_root=output,
        fixture="G",
        expected_resolution=(640, 480),
        operator_id="operator",
        camera_id=f"camera-{session_id}",
        baseline_freeze=baseline,
        protected_handoff=handoff,
        session_id=session_id,
    )
    return session, camera


def test_official_campaign_is_immutable_and_baseline_bound(tmp_path):
    baseline_a = write_baseline(tmp_path / "baseline-a.json")
    baseline_b = write_baseline(tmp_path / "baseline-b.json")
    changed = json.loads(baseline_b.read_text(encoding="utf-8"))
    changed["model_sha256"] = "9" * 64
    baseline_b.write_text(json.dumps(changed), encoding="utf-8")
    development = tmp_path / "development"

    train, _ = _init_official(
        tmp_path, role="train", session_id="train-a", baseline=baseline_a, output=development
    )
    validation, _ = _init_official(
        tmp_path, role="validation", session_id="validation-a", baseline=baseline_a,
        output=development,
    )
    campaign = json.loads((development / ".vial_guard_campaign.json").read_text())
    assert campaign["baseline_freeze_sha256"] == C.sha256_file(baseline_a)
    assert C.load_session(train)[0]["campaign"]["campaign_id"] == campaign["campaign_id"]
    assert C.load_session(validation)[0]["campaign"]["campaign_id"] == campaign["campaign_id"]

    with pytest.raises(C.CollectionError, match="different baseline freeze"):
        _init_official(
            tmp_path, role="validation", session_id="validation-wrong-freeze",
            baseline=baseline_b, output=development,
        )
    assert not (development / "validation-wrong-freeze").exists()


def test_protected_campaign_uses_metadata_handoff_and_refuses_cross_split_duplicate(
        tmp_path, monkeypatch):
    baseline = write_baseline(tmp_path / "baseline.json")
    development = tmp_path / "development"
    protected = tmp_path / "protected"
    train, train_camera = _init_official(
        tmp_path, role="train", session_id="train-a", baseline=baseline, output=development
    )
    _init_official(
        tmp_path, role="validation", session_id="validation-a", baseline=baseline,
        output=development,
    )
    train_event = C.read_csv(train / "plan.csv", C.PLAN_COLUMNS)[0]
    duplicate = png_bytes(value=61)
    train_camera.joinpath("development.png").write_bytes(duplicate)
    C.ingest_capture(train, train_event, stability_seconds=0.01, wait_seconds=0.5)
    C.finalize_capture(train, accepted=True, note="confirmed")

    def frozen_entry(session_id: str, role: str, digit: str) -> dict[str, str]:
        return {
            "session_id": session_id,
            "role": role,
            "seal_status": "COMPLETE",
            "session_commitment_sha256": digit * 64,
            **{field: digit * 64 for field, _ in C.SEALED_SESSION_HASH_PATHS},
        }

    monkeypatch.setattr(C, "_verify_development_sessions_for_freeze", lambda _root: {
        "development_roles": ["train", "validation"],
        "development_session_count": 2,
        "development_session_commitments_sha256": "c" * 64,
        "sealed_session_entries": [
            frozen_entry("train-a", "train", "1"),
            frozen_entry("validation-a", "validation", "2"),
        ],
    })
    handoff = tmp_path / "protected-handoff"
    frozen = C.freeze_development_for_protected(development, handoff)
    assert frozen["capture_reservation_count"] == 1
    assert {path.name for path in handoff.iterdir()} == C.HANDOFF_FILES
    development_campaign = json.loads((development / ".vial_guard_campaign.json").read_text())
    development_closed = json.loads(
        (development / ".development_frozen_for_protected.json").read_text()
    )
    assert development_closed["capture_reservations_sha256"] == C.sha256_file(
        development / ".capture_sha256_reservations.csv"
    )
    assert development_closed["development_sealed_sessions_manifest_sha256"] == C.sha256_file(
        development / ".development_sealed_sessions.json"
    )
    sealed_handoff = handoff / C.HANDOFF_SEALED_SESSIONS_NAME
    assert sealed_handoff.is_file()
    assert json.loads(sealed_handoff.read_text())["sessions"][0]["session_id"] == "train-a"

    unavailable_development = tmp_path / "development-not-visible-to-protected"
    development.rename(unavailable_development)
    protected_session, protected_camera = _init_official(
        tmp_path, role="protected_test", session_id="protected-a", baseline=baseline,
        output=protected, handoff=handoff,
    )
    protected_cards = protected_session / "records/event_cards"
    assert protected_cards.stat().st_mode & 0o777 == 0o700
    assert len(list(protected_cards.glob("E*.md"))) == 30
    assert all(path.stat().st_mode & 0o777 == 0o400 for path in protected_cards.iterdir())
    protected_manifest = protected_cards / C.EVENT_CARD_MANIFEST_NAME
    protected_manifest.chmod(0o440)
    with pytest.raises(C.CollectionError, match="manifest permissions differ"):
        C.load_session(protected_session)
    protected_manifest.chmod(0o400)
    protected_campaign = json.loads((protected / ".vial_guard_campaign.json").read_text())
    assert protected_campaign["campaign_id"] == development_campaign["campaign_id"]
    assert protected_campaign["baseline_freeze_sha256"] == C.sha256_file(baseline)
    assert (protected / ".development_handoff" / C.HANDOFF_DIGESTS_NAME).is_file()
    protected_event = C.read_csv(protected_session / "plan.csv", C.PLAN_COLUMNS)[0]
    protected_card = (
        protected_cards / f"{protected_event['event_id']}_{protected_event['family']}.md"
    ).read_text(encoding="utf-8")
    assert protected_event["event_uid"] in protected_card
    assert protected_event["primary_targets"] in protected_card
    protected_camera.joinpath("same-pixels.png").write_bytes(duplicate)
    with pytest.raises(C.CollectionError, match="collection campaign"):
        C.ingest_capture(
            protected_session, protected_event, stability_seconds=0.01, wait_seconds=0.5
        )
    assert not list((protected_session / "captures").iterdir())

    installed = protected / ".development_handoff"
    installed_manifest = installed / C.HANDOFF_SEALED_SESSIONS_NAME
    installed_manifest.chmod(0o600)
    altered = json.loads(installed_manifest.read_text())
    altered["sessions"][0]["seal_selection_sha256"] = "f" * 64
    installed_manifest.write_text(json.dumps(altered), encoding="utf-8")
    installed_checksums = installed / C.HANDOFF_CHECKSUMS_NAME
    installed_checksums.chmod(0o600)
    installed_checksums.write_text("".join(
        f"{C.sha256_file(installed / name)}  {name}\n"
        for name in sorted(C.HANDOFF_FILES - {C.HANDOFF_CHECKSUMS_NAME})
    ), encoding="utf-8")
    with pytest.raises(C.CollectionError, match="receipt hashes differ"):
        C.load_session(protected_session)


def test_freeze_development_requires_complete_sealed_sessions(tmp_path):
    baseline = write_baseline(tmp_path / "baseline.json")
    development = tmp_path / "development"
    _init_official(
        tmp_path, role="train", session_id="train-a", baseline=baseline, output=development
    )
    handoff = tmp_path / "handoff"
    with pytest.raises(C.CollectionError, match="complete and sealed"):
        C.freeze_development_for_protected(development, handoff)
    assert not handoff.exists()
    assert not (development / ".development_frozen_for_protected.json").exists()


def test_frozen_development_load_authenticates_all_selected_seal_artifacts(
        tmp_path, monkeypatch):
    baseline = write_baseline(tmp_path / "baseline.json")
    development = tmp_path / "development"
    train, _ = _init_official(
        tmp_path, role="train", session_id="train-a", baseline=baseline, output=development
    )
    validation, _ = _init_official(
        tmp_path, role="validation", session_id="validation-a", baseline=baseline,
        output=development,
    )
    for index, session in enumerate((train, validation), start=1):
        (session / "sealed/evidence.tar").write_bytes(f"archive-{index}".encode())
        (session / "sealed/evidence.tar.sha256").write_text(
            f"{'a' * 64}  evidence.tar\n", encoding="utf-8"
        )
        (session / "records/seal_manifest.json").write_text(
            json.dumps({"seal_status": "COMPLETE"}), encoding="utf-8"
        )
        (session / "records/seal_selection.json").write_text(
            json.dumps({"selected": [index]}), encoding="utf-8"
        )

    entries = [
        C._sealed_session_entry(
            session,
            session.name,
            "train" if session is train else "validation",
            C.sha256_file(development / ".session_commitments" / f"{session.name}.json"),
        )
        for session in (train, validation)
    ]
    monkeypatch.setattr(C, "_verify_development_sessions_for_freeze", lambda _root: {
        "development_roles": ["train", "validation"],
        "development_session_count": 2,
        "development_session_commitments_sha256": "d" * 64,
        "sealed_session_entries": entries,
    })
    handoff = tmp_path / "handoff"
    C.freeze_development_for_protected(development, handoff)
    frozen = json.loads((handoff / C.HANDOFF_SEALED_SESSIONS_NAME).read_text())
    train_frozen = next(row for row in frozen["sessions"] if row["session_id"] == "train-a")
    for field, relative in C.SEALED_SESSION_HASH_PATHS:
        assert train_frozen[field] == C.sha256_file(train / relative)
    assert C.load_session(train)[0]["role"] == "train"

    selection = train / "records/seal_selection.json"
    selection.write_text(json.dumps({"selected": [999]}), encoding="utf-8")
    with pytest.raises(C.CollectionError, match="sealed artifacts differ"):
        C.load_session(train)


def test_cross_session_duplicate_is_refused_inside_development_root(tmp_path):
    baseline = write_baseline(tmp_path / "baseline.json")
    development = tmp_path / "development"
    train, train_camera = _init_official(
        tmp_path, role="train", session_id="train-a", baseline=baseline, output=development
    )
    validation, validation_camera = _init_official(
        tmp_path, role="validation", session_id="validation-a", baseline=baseline,
        output=development,
    )
    duplicate = png_bytes(value=71)
    train_camera.joinpath("train.png").write_bytes(duplicate)
    C.ingest_capture(
        train, C.read_csv(train / "plan.csv", C.PLAN_COLUMNS)[0],
        stability_seconds=0.01, wait_seconds=0.5,
    )
    C.finalize_capture(train, accepted=True, note="confirmed")
    validation_camera.joinpath("validation.png").write_bytes(duplicate)
    with pytest.raises(C.CollectionError, match="collection campaign"):
        C.ingest_capture(
            validation, C.read_csv(validation / "plan.csv", C.PLAN_COLUMNS)[0],
            stability_seconds=0.01, wait_seconds=0.5,
        )


def test_external_commitment_detects_role_edit_and_symlink_paths(tmp_path):
    baseline = write_baseline(tmp_path / "baseline.json")
    development = tmp_path / "development"
    session, _ = _init_official(
        tmp_path, role="train", session_id="train-a", baseline=baseline, output=development
    )
    state_path = session / "session.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["role"] = "validation"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(C.CollectionError, match="external commitment"):
        C.load_session(session)

    symlink_case = tmp_path / "symlink-case"
    symlink_case.mkdir()
    rehearsal, _, _ = init_rehearsal(symlink_case)
    session_link = tmp_path / "linked-session"
    session_link.symlink_to(rehearsal, target_is_directory=True)
    with pytest.raises(C.CollectionError, match="symlink path component"):
        C.load_session(session_link)

    real_captures = rehearsal / "captures-real"
    (rehearsal / "captures").rename(real_captures)
    (rehearsal / "captures").symlink_to(real_captures, target_is_directory=True)
    with pytest.raises(C.CollectionError, match="unsafe session directory|symlink inside"):
        C.load_session(rehearsal)

