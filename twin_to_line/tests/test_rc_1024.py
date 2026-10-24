"""RC regressions from the independent review; synthetic inputs only."""
import builtins
import csv
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import live_inspect as L


def image_bytes(ext=".png"):
    return cv2.imencode(ext, np.full((480, 640, 3), 90, np.uint8))[1].tobytes()


@pytest.mark.parametrize("names", [
    [f"cap_seal_20261026_12300{k}_000000.png" for k in (0, 2, 4, 6)],
    [f"cap_seal_video_1.mp4@t{k:06.1f}.png" for k in (0, 2, 4, 6)],
])
def test_timestamped_burst_settles_by_filename_time(tmp_path, monkeypatch, names):
    folder = tmp_path / "in"; folder.mkdir()
    clock = [100.0]
    monkeypatch.setattr(L.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(L.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    frames = L.watch(str(folder), poll=.01, idle_exit=1)
    for n in names: (folder / n).write_bytes(image_bytes())
    s = L.Session(str(tmp_path / "out"), quiet=True, ix=None)
    try:
        rows = []
        for _ in names:
            clock[0] += .04  # all four arrive within 0.2 seconds
            rows.append(s.process_frame(*next(frames)))
    finally:
        frames.close(); s.close()
    assert [r["frame_t"] for r in rows] == [L.frame_time(str(folder/n)) for n in names]
    assert [r["status"] for r in rows] == ["settling", "settling", "scored", "unchanged"]


def test_invalid_or_backward_names_fall_back_without_clock_jumps(tmp_path, monkeypatch):
    folder = tmp_path / "in"; folder.mkdir()
    clock = [100.0]
    monkeypatch.setattr(L.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(L.time, "sleep", lambda seconds: None)
    frames = L.watch(str(folder), poll=.01, idle_exit=10)
    names = ["cap_seal_video_1.mp4@t0010.0.png", "latest.png",
             "cap_seal_20261399_999999_000000.png", "cap_seal_video_1.mp4@t0004.0.png",
             "cap_seal_video_1.mp4@t0020.0.png"]
    times = []
    try:
        for i, n in enumerate(names):
            clock[0] = 100.0 + 2*i
            (folder/n).write_bytes(image_bytes())
            times.append(next(frames)[2])
    finally:
        frames.close()
    assert times == [10., 12., 14., 16., 20.]


@pytest.mark.parametrize("bad", ["permission_read", "permission_stat", "empty", "missing_jpeg_end",
                                 "jpeg_trailing_bytes", "jpeg_mixed_trailer", "png_trailing_bytes"])
def test_unacceptable_version_logs_once_after_three_seconds_then_accepts_repair(tmp_path, monkeypatch, bad):
    folder = tmp_path / "in"; folder.mkdir()
    path = folder / ("latest.png" if bad == "png_trailing_bytes" else "latest.jpg")
    valid = image_bytes(path.suffix)
    content = b"" if bad == "empty" else valid[:-2] if bad == "missing_jpeg_end" else valid
    if bad.endswith("trailing_bytes"): content += b"incomplete trailer"
    if bad == "jpeg_mixed_trailer": content += b"\x00x\x00"
    clock = [0.0]; denied = [True]
    monkeypatch.setattr(L.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(L.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    frames = L.watch(str(folder), poll=.25, idle_exit=10)
    path.write_bytes(content)
    original_open, original_version = builtins.open, L.file_version
    def opening(p, *args, **kwargs):
        if denied[0] and bad == "permission_read" and os.fspath(p) == str(path):
            raise PermissionError("simulated read denial")
        return original_open(p, *args, **kwargs)
    def version(p):
        if denied[0] and bad == "permission_stat" and os.fspath(p) == str(path):
            raise PermissionError("simulated stat denial")
        return original_version(p)
    monkeypatch.setattr(builtins, "open", opening)
    monkeypatch.setattr(L, "file_version", version)
    s = L.Session(str(tmp_path / "out"), gate="each", quiet=True, ix=None)
    try:
        first = next(frames, None)
        assert first is not None, "unacceptable version was silently dropped"
        assert clock[0] == 3.0, "version must wait for the 3-second give-up, then log immediately"
        assert s.process_frame(*first)["status"] == "unreadable"
        # Let the same version stay around for another poll, then replace it while the watcher is waiting.
        def repair(seconds):
            clock[0] += seconds
            if clock[0] >= 4.0 and denied[0]:
                denied[0] = False
                path.write_bytes(valid)
        monkeypatch.setattr(L.time, "sleep", repair)
        assert s.process_frame(*next(frames))["status"] == "scored"
        assert list(frames) == []
    finally:
        frames.close(); s.close()
    rows = list(csv.DictReader((tmp_path / "out/trial_log.csv").open()))
    assert [r["status"] for r in rows] == ["unreadable", "scored"]
    assert s.events == 1


def test_runbook_checksum_failure_stops_setup_without_exiting_terminal(tmp_path):
    doc = (HERE / "LIVE_RUNBOOK.md").read_text()
    block = doc.split("## 7. Monday Oct 26", 1)[1].split("```bash", 1)[1].split("```", 1)[0]
    block = "\n".join(line[3:] if line.startswith("   ") else line for line in block.splitlines())
    block = block.replace("$HOME/ttl", str(tmp_path / "ttl"))
    usb = tmp_path / "usb"; usb.mkdir()
    (usb / "ttl_gb10_abcdef0.tar").write_bytes(b"not an archive")
    (usb / "ttl_gb10_abcdef0.tar.sha256").write_text("0" * 64 + "  ttl_gb10_abcdef0.tar\n")
    result = subprocess.run(["bash", "-c", block + '\nprintf "TERMINAL_ALIVE\\n"\n'],
                            input=str(usb) + "\nttl_gb10_abcdef0\n", capture_output=True, text=True)
    assert result.returncode == 0 and "TERMINAL_ALIVE" in result.stdout
    assert "STOP: setup failed" in result.stderr
    assert "SETUP OK" not in result.stdout and "BOOTSTRAP OK" not in result.stdout
    assert not list((tmp_path / "ttl").glob("visit_*/install_*/ttl_gb10_abcdef0"))


@pytest.mark.parametrize("padding", [0, 1, 2, 12, 4096])
def test_jpeg_zero_padding_is_complete_and_decodable(tmp_path, padding):
    path = tmp_path / "padded.jpg"
    path.write_bytes(image_bytes(".jpg") + b"\x00" * padding)
    assert L.complete(str(path))
    image, finished, unchanged = L.read_version(str(path))
    assert finished and unchanged
    assert image is not None and image.shape == (480, 640, 3)


@pytest.mark.parametrize("trailer", [b"x", b"\x00x\x00", b"\x00" * 4096 + b"x"])
def test_jpeg_nonzero_trailer_stays_incomplete(tmp_path, trailer):
    path = tmp_path / "invalid.jpg"
    path.write_bytes(image_bytes(".jpg") + trailer)
    assert not L.complete(str(path))
    image, finished, unchanged = L.read_version(str(path))
    assert image is None and not finished and unchanged
