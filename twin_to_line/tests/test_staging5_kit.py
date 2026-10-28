"""Focused offline-package tests for the staging5 GB10 demo kit.

These tests deliberately avoid downloading dependencies or running the ARM64
runtime.  They exercise the package's trust boundary, protected-split
selection, golden-output contract, and fixed shadow-mode launch command.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


TTL = Path(__file__).resolve().parents[1]
REPO = TTL.parent
sys.path.insert(0, str(TTL))

from gb10 import build_staging5_kit as B  # noqa: E402
from gb10 import staging5_preflight as P  # noqa: E402


EXPECTED_GOLDEN_FIELDS = (
    "source_image_id",
    "fixture_id",
    "position_id",
    "expected_occupancy",
    "expected_occupancy_source",
    "predicted_visual_class",
    "model_argmax_class",
    "model_score",
    "applicable_threshold",
    "inspection_decision",
    "reason_code",
)
EXPECTED_LIVE_FIELDS = EXPECTED_GOLDEN_FIELDS + (
    "inspection_timestamp_utc",
    "event_uid",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for frame_number, frame in enumerate(B.PREFLIGHT_FRAME_NAMES, start=1):
        for position in range(1, 31):
            rows.append(
                {
                    "source_image_id": f"/nonportable/source/path/{frame}",
                    "fixture_id": "G",
                    "position_id": f"G{position:02d}",
                    "expected_occupancy": "occupied",
                    "expected_occupancy_source": "deterministic_inference:visual_frontier",
                    "predicted_visual_class": "pass",
                    "model_argmax_class": "pass",
                    "model_score": f"{0.01 * frame_number:.8f}",
                    "applicable_threshold": "0.9292464256286621",
                    "inspection_decision": "PASS",
                    "reason_code": "visual_pass_expected_occupied",
                    "inspection_timestamp_utc": "2026-10-28T12:00:00+00:00",
                    "event_uid": f"development-event-{frame_number}",
                }
            )
    return rows


def test_golden_schema_and_two_frame_sixty_row_comparison(tmp_path):
    """The shipped golden is exactly the auditable nine-field subset."""
    assert P.GOLDEN_FIELDS == EXPECTED_GOLDEN_FIELDS
    assert P.REQUIRED_LIVE_FIELDS == EXPECTED_LIVE_FIELDS

    rows = _prediction_rows()
    assert len(rows) == 2 * 30 == 60
    expected = tmp_path / "expected.csv"
    P._write_expected(expected, rows)

    with expected.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected_rows = list(reader)
    assert tuple(reader.fieldnames or ()) == EXPECTED_GOLDEN_FIELDS
    assert len(expected_rows) == 60
    assert {row["source_image_id"] for row in expected_rows} == set(
        B.PREFLIGHT_FRAME_NAMES
    )
    assert P._compare(expected, rows, score_tolerance=0.0) == {
        "ok": True,
        "rows": 60,
        "missing_keys": [],
        "extra_keys": [],
        "exact_differences": [],
        "max_score_difference": 0.0,
        "score_tolerance": 0.0,
        "max_threshold_difference": 0.0,
        "decisions": {"PASS": 60},
        "visual_classes": {"pass": 60},
    }

    # A golden file is immutable once recorded, and duplicate target keys are
    # refused rather than silently collapsing to fewer than 60 predictions.
    with pytest.raises(FileExistsError, match="refusing to replace"):
        P._write_expected(expected, rows)
    with pytest.raises(ValueError, match="duplicate"):
        P._compare(expected, rows[:-1] + [rows[0]], score_tolerance=0.0)


def _write_validation_fixture(root: Path) -> tuple[Path, Path]:
    image_root = root / "images"
    image_root.mkdir()
    manifest = root / "validation_manifest.csv"
    fieldnames = (
        "source_image_relpath",
        "source_image_sha256",
        "split",
        "designation",
        "event_uid",
        "position_id",
    )
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frame_number, name in enumerate(B.PREFLIGHT_FRAME_NAMES, start=1):
            image = image_root / name
            image.write_bytes(f"development-frame-{frame_number}".encode())
            for position in range(1, 31):
                writer.writerow(
                    {
                        "source_image_relpath": name,
                        "source_image_sha256": _sha256(image),
                        "split": "validation",
                        "designation": "development",
                        "event_uid": f"validation-event-{frame_number}",
                        "position_id": f"G{position:02d}",
                    }
                )
    return manifest, image_root


def _rewrite_cell(path: Path, row_index: int, field: str, value: str) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    rows[row_index][field] = value
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_preflight_selection_is_exactly_two_complete_validation_events(tmp_path):
    manifest, image_root = _write_validation_fixture(tmp_path)
    selected = B.select_preflight_frames(manifest, image_root, _sha256(manifest))
    assert [item["filename"] for item in selected] == list(B.PREFLIGHT_FRAME_NAMES)
    assert all(item["split"] == "validation" for item in selected)
    assert all(item["designation"] == "development" for item in selected)
    assert all(item["sha256"] == _sha256(image_root / item["filename"]) for item in selected)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("designation", "protected_test", "designated development"),
        ("split", "test", "assigned to validation"),
        ("source_image_sha256", "0" * 64, "image hash differs"),
    ],
)
def test_preflight_selection_rejects_nondevelopment_or_mismatched_rows(
    tmp_path, field, bad_value, message
):
    manifest, image_root = _write_validation_fixture(tmp_path)
    _rewrite_cell(manifest, 0, field, bad_value)
    with pytest.raises(ValueError, match=message):
        B.select_preflight_frames(manifest, image_root, _sha256(manifest))


def test_preflight_selection_rejects_manifest_changed_after_training(tmp_path):
    manifest, image_root = _write_validation_fixture(tmp_path)
    frozen_digest = _sha256(manifest)
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="differs from the frozen training run"):
        B.select_preflight_frames(manifest, image_root, frozen_digest)


def test_arm64_wheel_allowlist_is_exact_and_rejects_tampered_bytes(tmp_path):
    expected = {
        "numpy-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl":
            "54139e0eb219f52f60656d163cbe67c31ede51d13236c950145473504fa208cb",
        "opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_aarch64."
        "manylinux2014_aarch64.whl":
            "46071015ff9ab40fccd8a163da0ee14ce9846349f06c6c8c0f2870856ffa45db",
    }
    assert B.ARM64_WHEEL_SHA256 == expected

    for name in expected:
        (tmp_path / name).write_bytes(b"tampered wheel")
    with pytest.raises(ValueError, match="SHA-256 differs from the pinned ARM64 wheel"):
        B.authenticate_wheelhouse(tmp_path)


def test_arm64_wheel_allowlist_accepts_only_exact_names_and_hashes(tmp_path, monkeypatch):
    for name in B.ARM64_WHEEL_SHA256:
        (tmp_path / name).write_bytes(b"stand-in for a large wheel")
    monkeypatch.setattr(
        B,
        "sha256",
        lambda path: B.ARM64_WHEEL_SHA256[Path(path).name],
    )

    authenticated = B.authenticate_wheelhouse(tmp_path)
    assert [path.name for path in authenticated] == sorted(B.ARM64_WHEEL_SHA256)


def test_arm64_wheel_allowlist_rejects_missing_or_unexpected_files(tmp_path):
    first_name = next(iter(B.ARM64_WHEEL_SHA256))
    (tmp_path / first_name).write_bytes(b"placeholder")
    (tmp_path / "unexpected-1.0-py3-none-any.whl").write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="filenames differ from the pinned allowlist") as error:
        B.authenticate_wheelhouse(tmp_path)
    assert "missing=" in str(error.value)
    assert "unexpected=" in str(error.value)


def test_real_frozen_model_and_onnx_inputs_authenticate_when_present():
    """Exercise the actual downloaded artifacts without making them a CI fixture."""
    model_dir = (
        REPO
        / "data/yaad-results/3c875c39-3bb5-4f17-bb07-46e30679c520/out/resnet18_staging5"
    )
    onnx_dir = REPO / "data/onnx-export-staging5-2026-10-28"
    required = [
        model_dir / "model.pt",
        model_dir / "run.json",
        model_dir / "thresholds.json",
        onnx_dir / "model.onnx",
        onnx_dir / "onnx_export.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("large frozen training/ONNX artifacts are not installed")

    run, threshold, export = B.authenticate_inputs(model_dir, onnx_dir)
    assert run["classes"] == [
        "pass",
        "missing_cap",
        "crooked_cap",
        "damaged_vial",
        "no_vial",
    ]
    assert run["unsupported_classes"] == ["damaged_vial"]
    assert run["final_test_used"] is False
    assert export["verification"]["rows_total"] == 510
    assert export["verification"]["protected_test_used"] is False
    assert threshold["t_defect"] == pytest.approx(0.9292464256286621)


def test_self_extractor_checks_embedded_bytes_and_refuses_existing_target(tmp_path):
    payload = tmp_path / "payload.tar.gz"
    payload.write_bytes(b"not extracted in this refusal test")
    installer = tmp_path / "staging5.run"
    B._build_self_extractor(payload, installer)

    existing = tmp_path / "already-there"
    existing.mkdir()
    refused = subprocess.run(
        ["bash", str(installer), "--install", str(existing), "--extract-only"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "INSTALL REFUSED" in refused.stderr
    assert "already exists" in refused.stderr

    # Corruption after construction must be caught before the install
    # directory is created or tar is invoked on the embedded bytes.
    with installer.open("ab") as stream:
        stream.write(b"tampered")
    fresh = tmp_path / "fresh-target"
    corrupted = subprocess.run(
        ["bash", str(installer), "--install", str(fresh), "--extract-only"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert corrupted.returncode != 0
    assert "embedded payload checksum mismatch" in corrupted.stderr
    assert not fresh.exists()


def test_live_launcher_is_fixed_to_authenticated_shadow_configuration():
    script = (TTL / "gb10/staging5_live_demo.sh").read_text(encoding="utf-8")
    assert '--inspector staging5' in script
    assert '--model-dir "$ROOT/model/resnet18_staging5"' in script
    assert '--fixture G' in script
    assert "SHADOW / EVALUATION ONLY" in script
    assert "No PLC actuation" in script
    assert "not a product-release signal" in script
    assert "--inspector \"$" not in script
    assert "--model-dir \"$MODEL" not in script
    assert "--fixture \"$" not in script


def test_bootstrap_is_arm64_only_and_installs_dependencies_offline():
    script = (TTL / "gb10/staging5_bootstrap.sh").read_text(encoding="utf-8")
    assert '"$(uname -m)" = "aarch64"' in script
    assert "sha256sum --quiet -c SHA256SUMS" in script
    assert "runtime/cpython-aarch64.tar.gz" in script
    assert "--no-index" in script
    assert "--find-links wheels/cp311" in script
    assert "gb10/staging5_preflight.py" in script
    assert "shadow-demo" in script
    assert "never releases product or actuates the line" in script
    for network_token in ("curl ", "wget ", "http://", "https://"):
        assert network_token not in script


def test_staging5_shell_scripts_have_valid_bash_syntax():
    scripts = [
        TTL / "gb10/staging5_bootstrap.sh",
        TTL / "gb10/staging5_live_demo.sh",
        TTL / "gb10/staging5_run_sample.sh",
    ]
    for script in scripts:
        result = subprocess.run(
            ["bash", "-n", str(script)], text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"

