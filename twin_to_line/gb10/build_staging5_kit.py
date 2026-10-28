#!/usr/bin/env python3
"""Build one self-extracting, offline ARM64 staging5 live-demo kit.

The builder copies an explicit working-tree allowlist instead of using
``git archive`` because the staging adapter is a reviewed, uncommitted
experiment.  Every copied byte is listed in the inner SHA256SUMS file and the
self-extractor authenticates its embedded payload before installation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request


HERE = Path(__file__).resolve().parent
TTL = HERE.parent
REPO = TTL.parent
PACKAGE_VERSION = "staging5-2026-10-28"
KIT_BASENAME = "ttl_kit_staging5_2026-10-28"

PYTHON_RUNTIME = {
    "release": "20261009",
    "filename": "cpython-3.11.17+20261009-aarch64-unknown-linux-gnu-install_only.tar.gz",
    "sha256": "693ef5c3064e13af98178fba648c4757be92b842546a0ee2dae19ce1e3d863ba",
    "max_bytes": 200_000_000,
}
WHEEL_PINS = ("numpy==2.1.0", "opencv-python-headless==4.10.0.84")
WHEEL_PLATFORMS = (
    "manylinux_2_17_aarch64",
    "manylinux2014_aarch64",
)
ARM64_WHEEL_SHA256 = {
    "numpy-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl":
        "54139e0eb219f52f60656d163cbe67c31ede51d13236c950145473504fa208cb",
    "opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_aarch64."
    "manylinux2014_aarch64.whl":
        "46071015ff9ab40fccd8a163da0ee14ce9846349f06c6c8c0f2870856ffa45db",
}

APP_FILES = (
    "live_inspect.py",
    "inspect_frame.py",
    "positions.py",
    "fill_state.py",
    "rules.py",
    "events.py",
    "real_data.py",
    "frame_warp.py",
    "eval_metrics.py",
    "staging_dataset.py",
    "train_resnet.py",
    "train_resnet5.py",
    "resnet_inspector.py",
    "resnet5_onnx_inspector.py",
    "position_map_G.json",
    "registration_dev.json",
    "rule_thresholds.json",
    "frame_warp_1080.json",
    "gb10/staging5_preflight.py",
)
MODEL_FILES = (
    "model.pt",
    "run.json",
    "thresholds.json",
    "train_log.csv",
    "validation_predictions.csv",
)
DOC_FILES = (
    "STAGING_TRAINING.md",
    "staging/PROTOCOL_2026-10-28.md",
    "staging/RESULT_2026-10-28.md",
)
PREFLIGHT_FRAME_NAMES = ("setup_03_r1.jpg", "setup_20.jpg")


def sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"required regular file is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    shutil.copymode(source, destination)


def authenticate_inputs(model_dir: Path, onnx_dir: Path) -> tuple[dict, dict, dict]:
    run = _read_json(model_dir / "run.json")
    threshold = _read_json(model_dir / "thresholds.json")
    export = _read_json(onnx_dir / "onnx_export.json")
    expected_classes = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]
    if run.get("classes") != expected_classes or export.get("classes") != expected_classes:
        raise ValueError("model/export class order differs from the exact five-class taxonomy")
    if run.get("five_class_model_ready") is not False:
        raise ValueError("this builder is pinned to the current not-yet-ready staging run")
    if run.get("unsupported_classes") != ["damaged_vial"]:
        raise ValueError("unexpected unsupported-class inventory")
    if run.get("final_test_used") is not False or run.get("test_rows_loaded") != 0:
        raise ValueError("training provenance does not prove protected-test isolation")
    if sha256(model_dir / "model.pt") != run.get("model_pt_sha256"):
        raise ValueError("model.pt differs from run.json")
    if sha256(model_dir / "thresholds.json") != run.get("thresholds_sha256"):
        raise ValueError("thresholds.json differs from run.json")
    verification = export.get("verification", {})
    if not isinstance(verification, dict) or verification.get("status") != "passed":
        raise ValueError("ONNX verification did not pass")
    if verification.get("scope") != "train_and_validation_only":
        raise ValueError("ONNX verification was not limited to development splits")
    if verification.get("protected_test_used") is not False or verification.get("test_rows_loaded") != 0:
        raise ValueError("ONNX report does not prove protected-test isolation")
    if verification.get("decision_mismatches") != 0 or verification.get("rows_total") != 510:
        raise ValueError("ONNX report lacks full 510-crop decision equivalence")
    if sha256(onnx_dir / "model.onnx") != export.get("model_onnx_sha256"):
        raise ValueError("model.onnx differs from its export report")
    source = export.get("source", {})
    if source.get("model_pt_sha256") != run.get("model_pt_sha256"):
        raise ValueError("ONNX source model differs from the frozen training model")
    if source.get("run_json_sha256") != sha256(model_dir / "run.json"):
        raise ValueError("ONNX source run.json differs from the frozen training run")
    if source.get("thresholds_sha256") != sha256(model_dir / "thresholds.json"):
        raise ValueError("ONNX source threshold differs from the frozen threshold")
    if export.get("tools", {}).get("numpy") != "2.1.0":
        raise ValueError("ONNX verification did not use deployment NumPy 2.1.0")
    if export.get("tools", {}).get("opencv") != "4.10.0":
        raise ValueError("ONNX verification did not use deployment OpenCV 4.10.0")
    return run, threshold, export


def select_preflight_frames(
    validation_manifest: Path,
    image_root: Path,
    expected_manifest_sha256: str,
) -> list[dict[str, str]]:
    """Select named validation frames from the exact manifest frozen by training.

    The digest check precedes CSV parsing and image access.  A stale, edited, or
    substituted manifest therefore cannot self-designate protected pixels as
    development preflight data.
    """
    actual_manifest_sha256 = sha256(validation_manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "validation manifest SHA-256 differs from the frozen training run: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    with validation_manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected: list[dict[str, str]] = []
    for name in PREFLIGHT_FRAME_NAMES:
        matches = [row for row in rows if row.get("source_image_relpath") == name]
        if len(matches) != 30:
            raise ValueError(f"{name}: expected 30 validation position rows, found {len(matches)}")
        if {row.get("split") for row in matches} != {"validation"}:
            raise ValueError(f"{name}: not exclusively assigned to validation")
        if {row.get("designation") for row in matches} != {"development"}:
            raise ValueError(f"{name}: not exclusively designated development")
        if len({row.get("event_uid") for row in matches}) != 1:
            raise ValueError(f"{name}: rows do not form one placement event")
        image = image_root / name
        digest = sha256(image)
        if {row.get("source_image_sha256") for row in matches} != {digest}:
            raise ValueError(f"{name}: image hash differs from validation manifest")
        selected.append(
            {
                "filename": name,
                "sha256": digest,
                "split": "validation",
                "designation": "development",
                "event_uid": matches[0]["event_uid"],
            }
        )
    return selected


def download_python_runtime(destination: Path) -> None:
    info = PYTHON_RUNTIME
    escaped = urllib.parse.quote(info["filename"], safe="._-")
    url = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{info['release']}/{escaped}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "vial-guard-kit-builder/1"})
    with urllib.request.urlopen(request) as response, destination.open("xb") as stream:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > int(info["max_bytes"]):
            raise ValueError("standalone Python download exceeds the pinned size ceiling")
        shutil.copyfileobj(response, stream)
    if destination.stat().st_size > int(info["max_bytes"]):
        raise ValueError("standalone Python download exceeds the pinned size ceiling")
    if sha256(destination) != info["sha256"]:
        raise ValueError("standalone Python SHA-256 differs from the pinned release")


def authenticate_wheelhouse(destination: Path) -> list[Path]:
    """Fail closed unless the wheelhouse is the pinned ARM64 byte allowlist."""
    wheels = sorted(destination.glob("*.whl"))
    actual_names = {path.name for path in wheels}
    expected_names = set(ARM64_WHEEL_SHA256)
    if actual_names != expected_names:
        raise ValueError(
            "ARM64 wheel filenames differ from the pinned allowlist "
            f"(missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)})"
        )
    for path in wheels:
        actual = sha256(path)
        expected = ARM64_WHEEL_SHA256[path.name]
        if actual != expected:
            raise ValueError(
                f"{path.name}: SHA-256 differs from the pinned ARM64 wheel "
                f"(expected {expected}, got {actual})"
            )
    return wheels


def download_wheels(destination: Path) -> None:
    destination.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--disable-pip-version-check",
        "--only-binary=:all:",
        "--no-deps",
        "--implementation",
        "cp",
        "--python-version",
        "3.11",
        "--dest",
        str(destination),
    ]
    for platform_name in WHEEL_PLATFORMS:
        command.extend(("--platform", platform_name))
    subprocess.run([*command, *WHEEL_PINS], check=True)
    wheels = authenticate_wheelhouse(destination)
    if any("aarch64" not in path.name for path in wheels):
        raise ValueError(f"wheelhouse contains a non-aarch64 wheel: {[p.name for p in wheels]}")
    (destination / "requirements.txt").write_text(
        "\n".join(WHEEL_PINS) + "\n", encoding="utf-8"
    )


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_payload_docs(payload: Path, deployment: dict) -> None:
    readme = f"""VIAL GUARD AI — staging5 offline live-demo kit
Version: {PACKAGE_VERSION}

This payload is installed by the self-extracting .run file. It is fully
offline after copying to the NVIDIA GB10 compute module.

First run (performed automatically by the .run installer):
  bash bootstrap.sh

Bundled sample:
  bash run_sample.sh

Live camera folder (new files are processed; Ctrl-C stops):
  bash live_demo.sh folder:/path/to/camera/images

Saved-frame replay:
  bash live_demo.sh replay:/path/to/images /path/to/output each

The output directory contains predictions.csv, trial_log.csv, session.json,
overlays, and a browser-refreshing view.html. Every target row includes source
image, G fixture/position, expected-occupancy context, five-class prediction,
score/threshold, deterministic inspection decision, reason, timestamp and
event identifier.

This is a SHADOW/EVALUATION demo. It has no PLC or product-release output.
See SAFETY_AND_LIMITATIONS.txt before operating it.
"""
    safety = """SAFETY AND LIMITATIONS — READ BEFORE THE DEMO

* Shadow/evaluation only. A PASS is not authority to release product.
* The kit never writes to a PLC, robot, actuator, or manufacturing controller.
* Fixture G only. Fixture S is explicitly refused because it has no calibrated
  map or collected staging evidence.
* The network has a five-output head in the exact class order:
  pass, missing_cap, crooked_cap, damaged_vial, no_vial.
* damaged_vial had zero development examples. Its global-maximum evidence is
  forced to UNKNOWN, and deterministic rules resolve that conservatively to
  UNSURE/manual review.
* The one-visit protected event holdout is not independent deployment proof.
  It showed 98% exact visual/process agreement, but exact supported-defect
  recall was 5/8 (62.5%) and one crooked-cap defect escaped as PASS.
* Trailing empty positions are WAITING unless sequence/order-complete context
  supports MISSED. A language model cannot override these deterministic rules.
* Do not tune the model, threshold, or package using final protected-test data.

Use the demo to display predictions and audit records, not to control product.
"""
    (payload / "README.txt").write_text(readme, encoding="utf-8")
    (payload / "SAFETY_AND_LIMITATIONS.txt").write_text(safety, encoding="utf-8")
    (payload / "deployment.json").write_text(
        json.dumps(deployment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_inventory(payload: Path) -> None:
    files = sorted(path for path in payload.rglob("*") if path.is_file())
    with (payload / "FILES.txt").open("x", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{path.stat().st_size:12d}  {path.relative_to(payload)}\n")
    files = sorted(path for path in payload.rglob("*") if path.is_file())
    with (payload / "SHA256SUMS").open("x", encoding="utf-8") as stream:
        for path in files:
            relative = path.relative_to(payload)
            stream.write(f"{sha256(path)}  {relative}\n")


def _tar_payload(payload: Path, destination: Path) -> None:
    """Create a stable gzip stream containing payload contents at archive root."""
    with destination.open("xb") as raw:
        with gzip.GzipFile(filename="payload.tar", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in sorted(payload.rglob("*")):
                    relative = path.relative_to(payload)
                    info = archive.gettarinfo(str(path), arcname=str(relative))
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
                    else:
                        archive.addfile(info)


def self_extractor_header(payload_sha256: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$HOME/ttl_staging5_2026-10-28"
EXTRACT_ONLY=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --install)
            [ "$#" -ge 2 ] || {{ echo "--install requires a directory" >&2; exit 2; }}
            INSTALL_DIR="$2"; shift 2 ;;
        --extract-only) EXTRACT_ONLY=1; shift ;;
        --help|-h)
            echo "Usage: bash $0 [--install DIRECTORY] [--extract-only]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ ! -e "$INSTALL_DIR" ] || {{
    echo "INSTALL REFUSED: $INSTALL_DIR already exists; choose a new directory" >&2
    exit 1
}}
command -v awk >/dev/null
command -v tail >/dev/null
command -v tar >/dev/null
command -v sha256sum >/dev/null

PAYLOAD_LINE="$(awk '/^__VIAL_GUARD_PAYLOAD_BELOW__$/ {{ print NR + 1; exit }}' "$0")"
[ -n "$PAYLOAD_LINE" ] || {{ echo "embedded payload marker not found" >&2; exit 1; }}
TEMP_PAYLOAD="$(mktemp "${{TMPDIR:-/tmp}}/ttl-staging5.XXXXXX.tar.gz")"
trap 'rm -f "$TEMP_PAYLOAD"' EXIT
tail -n "+$PAYLOAD_LINE" "$0" > "$TEMP_PAYLOAD"
printf '%s  %s\n' '{payload_sha256}' "$TEMP_PAYLOAD" | sha256sum --quiet -c - || {{
    echo "INSTALL REFUSED: embedded payload checksum mismatch" >&2
    exit 1
}}

mkdir -p "$INSTALL_DIR"
tar -xzf "$TEMP_PAYLOAD" -C "$INSTALL_DIR"
(cd "$INSTALL_DIR" && sha256sum --quiet -c SHA256SUMS) || {{
    echo "INSTALL REFUSED: extracted payload checksum mismatch" >&2
    exit 1
}}
echo "Authenticated payload extracted to $INSTALL_DIR"

if [ "$EXTRACT_ONLY" -eq 1 ]; then
    echo "Extraction-only requested; bootstrap was not run."
    exit 0
fi
bash "$INSTALL_DIR/bootstrap.sh"
echo "Installation and model preflight completed successfully."
exit 0
__VIAL_GUARD_PAYLOAD_BELOW__
"""


def _build_self_extractor(payload_archive: Path, destination: Path) -> None:
    header = self_extractor_header(sha256(payload_archive)).encode("utf-8")
    with destination.open("xb") as stream, payload_archive.open("rb") as payload:
        stream.write(header)
        shutil.copyfileobj(payload, stream)
    destination.chmod(0o755)


def _external_readme(run_name: str) -> str:
    return f"""VIAL GUARD staging5 USB kit ({PACKAGE_VERSION})

The single .run file contains the application, authenticated ONNX model,
source model/provenance, ARM64 CPython, ARM64 NumPy/OpenCV wheels, development
golden preflight frames and launch scripts. Internet access is not required on
the compute module.

1. Copy this entire directory to the USB drive.
2. After copying, verify it on this computer or the compute module:

     sha256sum -c {run_name}.sha256

3. On the NVIDIA compute module:

     bash {run_name} --install "$HOME/ttl_staging5_2026-10-28"

   The installer verifies the embedded archive, installs only from bundled
   artifacts, and runs a model-aware golden preflight.

4. Run the bundled sample:

     bash "$HOME/ttl_staging5_2026-10-28/run_sample.sh"

5. Run the live demo against the folder receiving new camera JPEGs:

     bash "$HOME/ttl_staging5_2026-10-28/live_demo.sh" folder:/path/to/images

   Use Ctrl-C to stop. Results are written under ~/ttl_staging5_runs/.

USB warning: a matching file size is not enough. Always verify the SHA-256
after copying. FAT/exFAT may drop execute bits, so invoking the file with
"bash" is intentional.

SAFETY: shadow/evaluation only, fixture G only, no automatic release or PLC
actuation. Read SAFETY_AND_LIMITATIONS.txt.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO / "data/yaad-results/3c875c39-3bb5-4f17-bb07-46e30679c520/out/resnet18_staging5",
    )
    parser.add_argument(
        "--onnx-dir", type=Path, default=REPO / "data/onnx-export-staging5-2026-10-28"
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=REPO / "data/capseal-training-derived-2026-10-28/splits_v1/validation_manifest.csv",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=REPO / "data/capseal-staging-dataset-2026-10-27/data/images",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO / "data" / KIT_BASENAME
    )
    parser.add_argument(
        "--preflight-python",
        type=Path,
        default=Path(sys.executable),
        help="Python with deployment-matching numpy==2.1.0 and opencv==4.10.0",
    )
    args = parser.parse_args(argv)

    model_dir = args.model_dir.resolve()
    onnx_dir = args.onnx_dir.resolve()
    validation_manifest = args.validation_manifest.resolve()
    image_root = args.image_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to replace output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    run, threshold, export = authenticate_inputs(model_dir, onnx_dir)
    manifests = run.get("manifests")
    if not isinstance(manifests, dict) or not isinstance(manifests.get("validation"), str):
        raise ValueError("training run lacks a frozen validation-manifest SHA-256")
    preflight_frames = select_preflight_frames(
        validation_manifest, image_root, manifests["validation"]
    )

    with tempfile.TemporaryDirectory(prefix="ttl-staging5-build-", dir=out.parent) as temp_name:
        temporary = Path(temp_name)
        payload = temporary / "payload"
        payload.mkdir()

        for relative in APP_FILES:
            _copy(TTL / relative, payload / "app/twin_to_line" / relative)
        for relative in MODEL_FILES:
            _copy(model_dir / relative, payload / "model/resnet18_staging5" / relative)
        _copy(onnx_dir / "model.onnx", payload / "model/resnet18_staging5/model.onnx")
        _copy(
            onnx_dir / "onnx_export.json",
            payload / "model/resnet18_staging5/onnx_export.json",
        )
        for relative in DOC_FILES:
            _copy(TTL / relative, payload / "docs" / relative)
        _copy(HERE / "staging5_bootstrap.sh", payload / "bootstrap.sh")
        _copy(HERE / "staging5_live_demo.sh", payload / "live_demo.sh")
        _copy(HERE / "staging5_run_sample.sh", payload / "run_sample.sh")

        frames_dir = payload / "preflight/frames"
        frames_dir.mkdir(parents=True)
        for metadata in preflight_frames:
            _copy(image_root / metadata["filename"], frames_dir / metadata["filename"])

        runtime_dir = payload / "runtime"
        runtime_dir.mkdir()
        print("Downloading pinned standalone ARM64 CPython ...", flush=True)
        download_python_runtime(runtime_dir / "cpython-aarch64.tar.gz")
        print("Downloading pinned ARM64 NumPy/OpenCV wheels ...", flush=True)
        download_wheels(payload / "wheels/cp311")

        source_hashes = {
            relative: sha256(TTL / relative) for relative in APP_FILES
        }
        deployment = {
            "schema_version": 1,
            "package_version": PACKAGE_VERSION,
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "source_git_revision": _git_revision(),
            "operation_mode": "shadow_evaluation_only",
            "automatic_release_enabled": False,
            "actuation_enabled": False,
            "target": {
                "architecture": "aarch64",
                "platform": "NVIDIA GB10 / DGX Spark class",
                "python": "3.11.17 standalone",
                "numpy": "2.1.0",
                "opencv": "4.10.0",
                "inference": "OpenCV DNN CPU / ONNX opset 17",
            },
            "classes": run["classes"],
            "unsupported_classes": run["unsupported_classes"],
            "five_class_model_ready": run["five_class_model_ready"],
            "fixture_support": ["G"],
            "model": {
                "model_pt_sha256": run["model_pt_sha256"],
                "model_onnx_sha256": export["model_onnx_sha256"],
                "thresholds_sha256": run["thresholds_sha256"],
                "t_defect": threshold["t_defect"],
                "run_json_sha256": sha256(model_dir / "run.json"),
                "onnx_export_json_sha256": sha256(onnx_dir / "onnx_export.json"),
                "training_job_id": "3c875c39-3bb5-4f17-bb07-46e30679c520",
                "protected_evaluation_job_id": "14160052-ea75-4861-bf4d-46fd90b82b4b",
            },
            "onnx_equivalence": export["verification"],
            "conversion_benchmark": export["benchmark"],
            "preflight_frames": preflight_frames,
            "protected_test_pixels_included": False,
            "protected_test_manifests_accessed_by_builder": False,
            "runtime_source_sha256": source_hashes,
            "limitations": export["limitations"],
        }
        _write_payload_docs(payload, deployment)

        expected = payload / "preflight/expected.csv"
        with tempfile.TemporaryDirectory(prefix="ttl-golden-", dir=temporary) as golden_name:
            command = [
                # Do not resolve the venv's python symlink: executing its base
                # interpreter path would bypass pyvenv.cfg and lose the exact
                # deployment-matching NumPy/OpenCV environment.
                str(args.preflight_python.absolute()),
                str(payload / "app/twin_to_line/gb10/staging5_preflight.py"),
                "--frames",
                str(frames_dir),
                "--model-dir",
                str(payload / "model/resnet18_staging5"),
                "--expected",
                str(expected),
                "--out",
                str(Path(golden_name) / "scored"),
                "--record-development-golden",
            ]
            subprocess.run(
                command,
                cwd=payload / "app/twin_to_line",
                check=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

        _write_inventory(payload)
        payload_archive = temporary / "payload.tar.gz"
        _tar_payload(payload, payload_archive)

        out.mkdir()
        run_name = f"{KIT_BASENAME}.run"
        run_path = out / run_name
        _build_self_extractor(payload_archive, run_path)
        (out / f"{run_name}.sha256").write_text(
            f"{sha256(run_path)}  {run_name}\n", encoding="utf-8"
        )
        external_safety = (payload / "SAFETY_AND_LIMITATIONS.txt").read_text(encoding="utf-8")
        (out / "SAFETY_AND_LIMITATIONS.txt").write_text(external_safety, encoding="utf-8")
        (out / "README_USB.txt").write_text(_external_readme(run_name), encoding="utf-8")
        (out / "PACKAGE_CONTENTS.txt").write_text(
            f"{run_path.stat().st_size:12d}  {run_name}\n"
            f"{(out / (run_name + '.sha256')).stat().st_size:12d}  {run_name}.sha256\n"
            "Embedded payload inventory is available as FILES.txt after extraction.\n",
            encoding="utf-8",
        )

    print(f"KIT OK: {out}")
    print(f"RUN: {out / (KIT_BASENAME + '.run')}")
    print(f"SHA256: {sha256(out / (KIT_BASENAME + '.run'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
