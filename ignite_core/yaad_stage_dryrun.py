#!/usr/bin/env python3
"""Drive the frozen stage script with bundled preflight frames on isolated YAAD state.

This helper is for a rehearsal only. It never edits the frozen kit. It creates 21
new camera files, stages setups 00-20 through the kit's normal prompt, skips setup
21 through the normal prompt, and exits at setup 22.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}: line {line_number} has no equals sign")
        key, encoded = line.split("=", 1)
        words = shlex.split(encoded)
        if len(words) > 1:
            raise ValueError(f"{path}: line {line_number} has multiple values")
        values[key] = words[0] if words else ""
    return values


def preflight_images(context: dict[str, str]) -> list[Path]:
    package = Path(context["INSTALL"]) / context["PKG"] / "preflight"
    candidates = sorted(
        path for path in package.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    by_digest: dict[str, Path] = {}
    import hashlib
    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_digest.setdefault(digest, path)
    images = list(by_digest.values())
    if len(images) < 21:
        raise ValueError(f"need 21 SHA-distinct bundled preflight images, found {len(images)}")
    return images[:21]


def durable_copy(source: Path, destination: Path) -> None:
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def drive_stage(home: Path, camera: Path) -> str:
    home = home.resolve(strict=True)
    context_path = home / "ttl" / "operator.env"
    context = read_env(context_path)
    if context.get("READY") != "1":
        raise ValueError("frozen kit setup is not READY=1")
    stage_script = home / "ttl" / "stage.sh"
    if not stage_script.is_file():
        raise ValueError(f"missing stage script: {stage_script}")
    camera.mkdir(parents=True, exist_ok=False)
    if any(camera.iterdir()):
        raise ValueError("dry-run camera directory must start empty")
    images = preflight_images(context)

    environment = dict(os.environ)
    environment["HOME"] = str(home)
    process = subprocess.Popen(
        ["bash", str(stage_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        bufsize=0,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("failed to open stage subprocess pipes")

    transcript = bytearray()
    prompt_buffer = bytearray()
    camera_sent = False
    answered: set[str] = set()
    setup_prompt = re.compile(rb"setup (\d\d): .* q = done: $")
    while True:
        byte = process.stdout.read(1)
        if not byte:
            break
        transcript.extend(byte)
        prompt_buffer.extend(byte)
        if len(prompt_buffer) > 600:
            del prompt_buffer[:-600]
        sys.stdout.buffer.write(byte)
        sys.stdout.buffer.flush()

        if (not camera_sent and
                prompt_buffer.endswith(b"Camera folder (ask CSI; before the first save): ")):
            process.stdin.write((str(camera) + "\n").encode())
            process.stdin.flush()
            camera_sent = True
            prompt_buffer.clear()
            continue

        match = setup_prompt.search(prompt_buffer)
        if match is None:
            continue
        setup = match.group(1).decode()
        if setup in answered:
            process.kill()
            raise RuntimeError(f"stage script repeated setup {setup}; inspect the preserved dry run")
        answered.add(setup)
        if 0 <= int(setup) <= 20:
            source = images[int(setup)]
            destination = camera / f"dry_{setup}{source.suffix.lower()}"
            durable_copy(source, destination)
            response = b"\n"
        elif setup == "21":
            response = b"s\n"
        elif setup == "22":
            response = b"q\n"
        else:
            process.kill()
            raise RuntimeError(f"unexpected stage prompt for setup {setup}")
        process.stdin.write(response)
        process.stdin.flush()
        prompt_buffer.clear()

    return_code = process.wait()
    text = transcript.decode("utf-8", errors="replace")
    if return_code != 0:
        raise RuntimeError(f"stage script exited {return_code}")
    if "STAGING OK: 21 setups staged" not in text:
        raise RuntimeError("stage script did not report the expected successful reconciliation")

    visit = Path(context["DAY"])
    with (visit / "records" / "staging_log.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 22:
        raise RuntimeError(f"expected 22 setup rows, found {len(rows)}")
    if sum(row["status"] == "skipped" for row in rows) != 1:
        raise RuntimeError("expected exactly one skipped setup")
    if sum(row["status"] in {"staged", "retake"} for row in rows) != 21:
        raise RuntimeError("expected exactly 21 staged setup rows")
    print(f"\nDRY STAGING OK: {visit}")
    return str(visit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True, help="isolated HOME used by start.sh")
    parser.add_argument("--camera", type=Path, required=True,
                        help="new, nonexistent simulated camera directory")
    args = parser.parse_args(argv)
    try:
        drive_stage(args.home, args.camera)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
