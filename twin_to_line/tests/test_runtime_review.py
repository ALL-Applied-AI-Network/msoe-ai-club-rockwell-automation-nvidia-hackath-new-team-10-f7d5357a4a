"""Regression checks for the ten Monday runtime review findings. No real holdout data."""
import csv
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from unittest import mock

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import live_inspect as L

spec = importlib.util.spec_from_file_location("review_package", HERE / "gb10/build_package.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def frame():
    return np.full((480, 640, 3), 90, np.uint8)


def encoded(ext=".png"):
    return cv2.imencode(ext, frame())[1].tobytes()


def session(tmp_path, **kw):
    return L.Session(str(tmp_path / "out"), ix=None, quiet=True, **kw)


def test_01_incomplete_versions_are_unreadable_then_complete_scores(tmp_path):
    incoming = tmp_path / "in"; incoming.mkdir()
    f = incoming / "latest.jpg"
    complete = encoded(".jpg")
    versions = [b"", complete[:-2], complete[:len(complete) * 2 // 3], complete]
    frames = L.watch(str(incoming), poll=.005, give_up_s=.02, idle_exit=.2)
    s = session(tmp_path, gate="each")
    rows = []
    try:
        for data in versions:
            f.write_bytes(data)
            try:
                item = next(frames)
            except StopIteration:
                pytest.fail("an incomplete version disappeared without a logged attempt")
            rows.append(s.process_frame(*item))
    finally:
        frames.close(); s.close()
    assert [r["status"] for r in rows] == ["unreadable"] * 3 + ["scored"]
    assert s.events == 1


def test_01_replay_does_not_score_a_jpeg_without_end_marker(tmp_path):
    incoming = tmp_path / "in"; incoming.mkdir()
    (incoming / "capture.jpg").write_bytes(encoded(".jpg")[:-2])
    s = session(tmp_path, gate="each")
    L.run(s, L.replay(str(incoming)))
    rows = list(csv.DictReader((tmp_path / "out/trial_log.csv").open()))
    assert [r["status"] for r in rows] == ["unreadable"]
    assert s.events == 0


def test_01_replay_rejects_a_version_changed_during_read(tmp_path, monkeypatch):
    incoming = tmp_path / "in"; incoming.mkdir()
    path = incoming / "capture.jpg"; data = encoded(".jpg"); path.write_bytes(data)
    original = builtins.open
    class ChangingRead:
        def __enter__(self):
            self.f = original(path, "rb")
            return self
        def seek(self, *args): return self.f.seek(*args)
        def read(self, *args):
            result = self.f.read(*args)
            with original(path, "wb") as out: out.write(data[:-2])
            return result
        def __exit__(self, *args): self.f.close()
    def opening(p, *args, **kwargs):
        return ChangingRead() if os.fspath(p) == str(path) and args == ("rb",) else original(p, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", opening)
    assert next(L.replay(str(incoming)))[1] is None


def test_02_reader_rejects_bytes_when_identity_changes(tmp_path, monkeypatch):
    path = tmp_path / "capture.png"; path.write_bytes(encoded())
    version = L.file_version(str(path))
    monkeypatch.setattr(L, "file_version", lambda p: (0, 0, 0, 0, 0))
    img, _, unchanged = L.read_version(str(path), version)
    assert img is None and not unchanged


def test_02_complete_replacements_do_not_need_a_whole_poll(tmp_path, monkeypatch):
    incoming = tmp_path / "in"; incoming.mkdir()
    f = incoming / "latest.png"
    frames = L.watch(str(incoming), poll=.01, idle_exit=.05)
    f.write_bytes(encoded())
    real_sleep = time.sleep
    def replace_on_every_poll(seconds):
        other = incoming / ".next.png"
        other.write_bytes(encoded()); os.replace(other, f)
        real_sleep(seconds)
    monkeypatch.setattr(L.time, "sleep", replace_on_every_poll)
    try:
        item = next(frames, None)
        assert item is not None, "complete atomic replacements starved the reader"
        assert item[1] is not None
    finally:
        frames.close()


def test_02_replacement_preserving_mtime_and_size_is_new(tmp_path):
    f = tmp_path / "latest.png"; f.write_bytes(encoded())
    before = L.file_version(str(f)); stat = f.stat()
    other = tmp_path / ".next.png"; other.write_bytes(encoded())
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns)); os.replace(other, f)
    assert L.file_version(str(f)) != before


def test_03_same_timestamp_rewrites_keep_advancing_replay_keeps_filename_time(tmp_path, monkeypatch):
    incoming = tmp_path / "in"; incoming.mkdir()
    f = incoming / "cap_seal_20261026_123000_000000.png"
    clock = [100.0]
    monkeypatch.setattr(L.time, "monotonic", lambda: clock[0])
    frames = L.watch(str(incoming), poll=.001, idle_exit=10)
    s = session(tmp_path)
    rows = []
    try:
        for k in range(4):
            clock[0] = 100.0 + 2*k
            f.write_bytes(encoded()); os.utime(f, (42, 42))
            # Atomic replacement also distinguishes files on coarse timestamp filesystems.
            other = incoming / ".next.png"; other.write_bytes(encoded()); os.replace(other, f)
            item = next(frames)
            assert len(item) == 3, "watched version did not carry its observation time"
            rows.append(s.process_frame(*item))
    finally:
        frames.close(); s.close()
    named = L.frame_time(str(f))
    assert [r["frame_t"] for r in rows] == [named, named + 2, named + 4, named + 6]
    assert [r["status"] for r in rows] == ["settling", "settling", "scored", "unchanged"]
    assert len(next(L.replay(str(incoming)))) == 2
    assert L.frame_time(str(f)) == named


def test_04_ffmpeg_open_and_read_deadlines_are_requested():
    cap = mock.Mock()
    cap.isOpened.return_value = True
    cap.getBackendName.return_value = "FFMPEG"
    with mock.patch.object(L.cv2, "VideoCapture", return_value=cap) as create:
        assert L.open_capture("rtsp://example.invalid/camera") is cap
    args = create.call_args.args
    assert len(args) == 3 and args[1] == cv2.CAP_FFMPEG
    params = dict(zip(args[2][::2], args[2][1::2]))
    assert 0 < params[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC] <= 10000
    assert 0 < params[cv2.CAP_PROP_READ_TIMEOUT_MSEC] <= 10000


def test_04_unbounded_device_has_an_explicit_fallback():
    with mock.patch.object(L.cv2, "VideoCapture"):
        with pytest.raises(ValueError, match="folder.*stills"):
            L.open_capture("/dev/video0")


def test_04_watchdog_releases_and_reconnects_without_concurrent_release(monkeypatch):
    monkeypatch.setattr(L, "STREAM_STALL_S", .03, raising=False)
    monkeypatch.setattr(L, "CAPTURE_JOIN_S", .3, raising=False)
    released = threading.Event()
    class Stalled:
        reading = False
        def isOpened(self): return True
        def read(self):
            self.reading = True
            time.sleep(.08)
            self.reading = False
            return True, frame()
        def release(self):
            assert not self.reading, "release must not race an FFmpeg read"
            released.set()
    class Good:
        def isOpened(self): return True
        def read(self): time.sleep(.002); return True, frame()
        def release(self): pass
    opens = []
    def opener(src):
        opens.append(src)
        return Stalled() if len(opens) == 1 else Good()
    logs = []
    reader = L.LatestFrame("mock", reconnect_s=.005, opener=opener, log=logs.append)
    try:
        deadline = time.monotonic()+.5
        while len(opens) < 2 and time.monotonic() < deadline: time.sleep(.005)
        assert len(opens) >= 2, "stalled source was not reconnected"
        assert released.is_set()
        assert any("WARNING" in x and "no new frame" in x for x in logs)
    finally:
        reader.stop()
    assert not reader.thread.is_alive()


def test_04_backend_that_ignores_deadlines_stops_instead_of_leaking_readers(monkeypatch):
    monkeypatch.setattr(L, "STREAM_STALL_S", .02, raising=False)
    monkeypatch.setattr(L, "CAPTURE_JOIN_S", .03, raising=False)
    unblock = threading.Event(); calls = []; logs = []
    def opener(src):
        calls.append(src); unblock.wait()
        return mock.Mock(isOpened=lambda: False, release=lambda: None)
    reader = L.LatestFrame("mock", reconnect_s=.005, opener=opener, log=logs.append)
    try:
        deadline = time.monotonic()+.4
        while not reader.stopped.is_set() and time.monotonic() < deadline: time.sleep(.005)
        assert reader.stopped.is_set()
        assert len(calls) == 1
        assert any("folder" in x and "stills" in x for x in logs)
    finally:
        unblock.set(); reader.stop()


@pytest.mark.parametrize("relative", ["code/twin_to_line/results/live/previous/trial_log.csv", "preflight_out/preflight.json"])
def test_05_bootstrap_refuses_existing_results_before_any_cleanup(tmp_path, relative):
    script = tmp_path / "bootstrap.sh"; shutil.copyfile(HERE / "gb10/bootstrap.sh", script)
    result = tmp_path / relative
    result.parent.mkdir(parents=True); result.write_text("previous evidence\n")
    # Checksum and environment shims prevent any Python installation in this test.
    shim = tmp_path / "tools"; shim.mkdir()
    bodies = {"sha256sum": "exit 0", "uname": "echo review-test"}
    bodies.update({f"python{v}": "exit 1" for v in ("3.10", "3.11", "3.12")})
    for name, body in bodies.items():
        p = shim / name; p.write_text("#!/bin/sh\n"+body+"\n"); p.chmod(0o755)
    (tmp_path / "SHA256SUMS").write_text("")
    with tarfile.open(tmp_path / "code.tar", "w") as tar:
        data = b"test\n"; entry = tarfile.TarInfo("twin_to_line/ARCHIVE_COMMIT"); entry.size = len(data)
        tar.addfile(entry, io.BytesIO(data))
    proc = subprocess.run(["bash", str(script)], cwd=tmp_path,
                          env={**os.environ, "PATH": str(shim)+":/usr/bin:/bin"}, capture_output=True, text=True)
    assert result.exists(), "bootstrap deleted trial evidence"
    assert result.read_text() == "previous evidence\n"
    assert proc.returncode != 0 and "fresh folder" in proc.stdout.lower()


def test_06_directory_error_is_logged_and_recovers(tmp_path, monkeypatch, capsys):
    incoming = tmp_path / "in"; incoming.mkdir()
    frames = L.watch(str(incoming), poll=.005, idle_exit=.04)
    (incoming / "new.png").write_bytes(encoded())
    real_listdir = os.listdir; calls = []
    def listing(path):
        if os.fspath(path) == str(incoming):
            calls.append(path)
            if len(calls) == 1: raise PermissionError("temporary mount permission")
        return real_listdir(path)
    monkeypatch.setattr(L.os, "listdir", listing)
    s = session(tmp_path, gate="each"); L.run(s, frames)
    rows = list(csv.DictReader((tmp_path / "out/trial_log.csv").open()))
    assert [r["status"] for r in rows] == ["source_error", "source_recovered", "scored"]
    assert "PermissionError" in rows[0]["note"]
    assert "WARNING" in capsys.readouterr().out
    assert json.loads((tmp_path / "out/session.json").read_text())["stop_reason"] == "end of source"


def test_06_directory_error_stop_reason_is_not_normal_eof(tmp_path, monkeypatch):
    incoming = tmp_path / "in"; incoming.mkdir()
    frames = L.watch(str(incoming), poll=.005, idle_exit=.02)
    real_listdir = os.listdir
    def listing(path):
        if os.fspath(path) == str(incoming): raise FileNotFoundError("camera folder gone")
        return real_listdir(path)
    monkeypatch.setattr(L.os, "listdir", listing)
    s = session(tmp_path); L.run(s, frames)
    info = json.loads((tmp_path / "out/session.json").read_text())
    assert "source error" in info["stop_reason"]
    assert "FileNotFoundError" in info["stop_reason"]


def test_06_initial_directory_error_recovers_without_reading_staged_files(tmp_path, monkeypatch):
    incoming = tmp_path / "in"; incoming.mkdir()
    (incoming / "staged.png").write_bytes(encoded())
    original = os.listdir; calls = []
    def listing(path):
        if os.fspath(path) == str(incoming):
            calls.append(path)
            if len(calls) == 1: raise PermissionError("initial permission error")
        return original(path)
    monkeypatch.setattr(L.os, "listdir", listing)
    stats = {}
    frames = L.watch(str(incoming), poll=.005, idle_exit=.02, stats=stats)
    s = session(tmp_path); L.run(s, frames)
    assert s.events == 0
    assert stats["ignored_existing_images"] == 1
    assert s.counts == {"source_error": 1, "source_recovered": 1}
    assert s.stop_reason == "end of source"


def test_06_unexpected_source_failure_records_its_reason(tmp_path):
    def source():
        raise RuntimeError("capture timeout ignored")
        yield
    s = session(tmp_path)
    with pytest.raises(RuntimeError, match="capture timeout ignored"):
        L.run(s, source())
    info = json.loads((tmp_path / "out/session.json").read_text())
    assert info["stop_reason"] == "source error: RuntimeError: capture timeout ignored"


def test_07_video_guard_works_without_an_index():
    assert L.refusal(None, "cap_seal_video_4.mp4@t0000.0.png") is not None
    assert L.refusal(None, "cap_seal_video_5.mp4@t0000.0.png") is not None
    assert L.refusal(None, "cap_seal_video_1.mp4@t0000.0.png") is None
    assert L.refusal(None, "new_camera_image.png") is None


def test_07_packaged_metadata_protects_stills_without_real_dataset(tmp_path, monkeypatch):
    here = tmp_path / "code/twin_to_line"; here.mkdir(parents=True)
    data = tmp_path / "data_index"; data.mkdir()
    (data / "image_index.csv").write_text("filename,timestamp,video_id\nprotected.jpg,2026-09-23 19:10:00,4\n")
    monkeypatch.setattr(L.P, "HERE", str(here))
    monkeypatch.setattr(L.R, "REAL_ROOT", str(tmp_path / "absent"))
    ix = L.dataset_index()
    assert ix is not None and L.refusal(ix, "protected.jpg") is not None
    assert L.refusal(ix, "live.jpg") is None


def test_07_package_index_contains_only_metadata(tmp_path, monkeypatch):
    from datetime import datetime
    ix = mock.Mock(run_of={"dev.jpg": "1", "protected.jpg": "4"},
                   time_of={name: datetime(2026, 9, 23) for name in ("dev.jpg", "protected.jpg")})
    monkeypatch.setattr(B.R, "Index", lambda: ix)
    B.package_index(str(tmp_path / "data_index"))
    rows = list(csv.DictReader((tmp_path / "data_index/image_index.csv").open()))
    assert {r["filename"]: r["video_id"] for r in rows} == ix.run_of
    assert set(rows[0]) == {"filename", "timestamp", "video_id"}
    ix.load.assert_not_called()


def test_07_watch_refuses_protected_names_before_reading_pixels(tmp_path, monkeypatch):
    incoming = tmp_path / "in"; incoming.mkdir()
    frames = L.watch(str(incoming), poll=.005, idle_exit=.02)
    path = incoming / "cap_seal_video_4.mp4@t0000.0.png"
    path.write_bytes(b"synthetic guard fixture, not real data")
    with mock.patch.object(L, "read_version", side_effect=AssertionError("protected pixels read")):
        item = next(frames)
    s = session(tmp_path)
    try:
        assert s.process_frame(*item)["status"] == "refused"
    finally:
        frames.close(); s.close()


@pytest.mark.parametrize("score", ["0.100001", "nan", "inf"])
def test_08_preflight_requires_identical_finite_still_scores(tmp_path, monkeypatch, score):
    expected_dir = tmp_path / "data/expected"; expected_dir.mkdir(parents=True)
    out = tmp_path / "out"; out.mkdir()
    expected = dict(frame="synthetic.png", position="G01", predicted_state="pass", final_state="PASS", defect_score="0.1")
    for name, _ in L.PREFLIGHT_SETS:
        with (expected_dir / (name+".csv")).open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(expected)); w.writeheader(); w.writerow(expected)
    def score_folder(folder, output):
        path = dict(L.PREFLIGHT_SETS)[Path(folder).name]
        return [dict(expected, defect_score=score)], [{"status": "scored", "path": path}]
    monkeypatch.setattr(L, "score_folder", score_folder)
    assert not L.preflight(str(tmp_path / "data"), str(out))


def test_09_runbook_describes_missing_warp_as_startup_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(L.FW, "load", lambda: None)
    original = L.sha256
    def missing(path):
        if str(path).endswith("frame_warp_1080.json"): raise FileNotFoundError(path)
        return original(path)
    monkeypatch.setattr(L, "sha256", missing)
    with pytest.raises(FileNotFoundError): session(tmp_path)
    rows = (HERE / "LIVE_RUNBOOK.md").read_text().splitlines()
    assert any("FileNotFoundError" in r and "frame_warp_1080.json" in r and "startup" in r for r in rows)
    assert not any("Status `wrong_size`" in r and "frame_warp_1080.json" in r for r in rows)


def test_10_relative_output_is_resolved_before_git_changes_directory(tmp_path, monkeypatch):
    class ArchiveReached(Exception): pass
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_package.py", "--out", "relative-output"])
    def git(*args):
        if args == ("rev-parse", "HEAD"): return "a" * 40
        if args == ("rev-parse", "--show-toplevel"): return str(tmp_path / "repo")
        return ""
    monkeypatch.setattr(B, "git", git)
    def archive(args, **kw):
        destination = Path(args[args.index("-o")+1])
        assert destination.is_absolute()
        assert destination == tmp_path / "relative-output/ttl_gb10_aaaaaaa/code.tar"
        assert destination.parent.exists()
        raise ArchiveReached()
    monkeypatch.setattr(B.subprocess, "run", archive)
    with pytest.raises(ArchiveReached): B.main()
