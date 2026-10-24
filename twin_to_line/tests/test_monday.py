import csv, itertools, json, os, shutil, subprocess, sys, tempfile, threading, time, unittest, unittest.mock
from datetime import datetime, timedelta
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import positions as P
import real_data as R
import frame_warp as FW
import live_inspect as L
from replay_1080 import warp_up

PM = P.PositionMap()
FWARP = FW.FrameWarp()
YELLOW = (0, 220, 230)
T0 = datetime(2026, 10, 26, 12, 30, 0)
EVENTS = os.path.join(P.HERE, "results", "prototype", "events")
DEV_STILLS = [os.path.join(EVENTS, "cap_seal_video_1.mp4@t1050.0.png"), os.path.join(EVENTS, "cap_seal_video_1.mp4@t0464.0.png")]
RUN3_EMPTY = "cap_seal_20260923_184630_557477.jpg"
REPLAY1 = os.path.join(P.HERE, "results", "live", "replay_run1")
VIDEO1 = os.path.join(R.REAL_ROOT, "videos", "cap_seal_video_1.mp4")
EVENTS1 = os.path.join(EVENTS, "events_run1_video.csv")
PY = sys.executable
DEV = os.path.exists(os.path.join(R.REAL_ROOT, "image_index.csv")) and all(map(os.path.exists, DEV_STILLS))


def fixture(filled=()):
    img = np.full((480, 640, 3), 90, np.uint8)
    for i in filled:
        x, y = PM.cap_centers[i - 1]
        cv2.circle(img, (int(round(x)), int(round(y))), 10, YELLOW, -1)
    return img


def name(k):
    return f"staged_{(T0 + timedelta(seconds=30 * k)).strftime('%Y%m%d_%H%M%S_%f')}.png"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def score(frames, track=False):
    """Session --gate each over (name, image) pairs; predictions keyed by (frame, position), without the event number."""
    out = tempfile.mkdtemp(); s = L.Session(out, gate="each", track=track, ix=None, quiet=True)
    for n, img in frames:
        s.process_frame(n, img)
    s.close()
    return {(r["frame"], r["position"]): {k: v for k, v in r.items() if k != "event"} for r in read_csv(os.path.join(out, "predictions.csv"))}


class StagedStillsIndependent(unittest.TestCase):
    """--gate each (no --track): every staged still is resolved from that image alone (fill frontier included)."""

    def check(self, a, b):
        together, reverse = score([a, b]), score([b, a])
        alone = {**score([a]), **score([b])}
        self.assertEqual(together, alone)
        self.assertEqual(reverse, alone)
        return alone

    def test_synthetic_fixtures(self):
        a = (name(0), fixture([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]))           # G04 empty inside the prefix -> MISSED
        b = (name(1), fixture([1, 2]))                                         # G04 after the frontier -> WAITING
        alone = self.check(a, b)
        self.assertEqual(alone[(a[0], "G04")]["final_state"], "MISSED")
        self.assertEqual(alone[(b[0], "G04")]["final_state"], "WAITING")
        self.assertEqual({alone[(a[0], "G12")]["K"], alone[(b[0], "G12")]["K"]}, {"12", "2"})

    def test_tracker_would_carry_state(self):                                   # why --track is not in the staged command
        a, b = (name(0), fixture(range(1, 13))), (name(1), fixture([1, 2]))
        self.assertNotEqual(score([a, b], track=True)[(b[0], "G05")]["final_state"], score([b])[(b[0], "G05")]["final_state"])

    @unittest.skipUnless(DEV, "needs the DEV dataset and the cached run-1 event frames")
    def test_two_dev_fixtures_in_a_row(self):
        ix = R.Index()
        a = (DEV_STILLS[0], cv2.imread(DEV_STILLS[0]))
        b = (RUN3_EMPTY, ix.load(RUN3_EMPTY))
        c = (DEV_STILLS[1], cv2.imread(DEV_STILLS[1]))
        self.check(a, b); self.check(a, c)
        self.check((a[0], warp_up(a[1], FWARP)), (c[0], warp_up(c[1], FWARP)))  # the same at 1080


class Preflight(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); os.makedirs(os.path.join(self.d, "expected"))
        frames = {"run1_640": [(name(0), fixture([1, 2, 3])), (name(1), fixture(range(1, 9)))]}
        frames["run1_1080"] = [(n, warp_up(im, FWARP)) for n, im in frames["run1_640"]]
        frames["still_1080"] = [("still.png", warp_up(fixture(), FWARP))]
        for k, fs in frames.items():
            os.makedirs(os.path.join(self.d, k))
            for n, im in fs:
                cv2.imwrite(os.path.join(self.d, k, n), im)
            preds, _ = L.score_folder(os.path.join(self.d, k), os.path.join(self.d, "_mac", k))
            shutil.copy2(os.path.join(self.d, "_mac", k, "predictions.csv"), os.path.join(self.d, "expected", k + ".csv"))

    def test_ok_then_fails_on_a_changed_state_or_wrong_path(self):
        self.assertTrue(L.preflight(self.d, tempfile.mkdtemp()))
        exp = os.path.join(self.d, "expected", "run1_640.csv"); rows = read_csv(exp)
        rows[0]["final_state"] = "DEFECT"
        with open(exp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        self.assertFalse(L.preflight(self.d, tempfile.mkdtemp()))

    def test_fails_when_a_1080_set_runs_the_640_path(self):
        st = os.path.join(self.d, "still_1080"); os.remove(os.path.join(st, "still.png"))
        cv2.imwrite(os.path.join(st, "still.png"), fixture())
        self.assertFalse(L.preflight(self.d, tempfile.mkdtemp()))


class Provenance(unittest.TestCase):
    def test_archive_commit_without_git(self):
        f = tempfile.NamedTemporaryFile("w", delete=False); f.write("0123456789abcdef\n"); f.close()
        no_git = unittest.mock.patch.object(L.subprocess, "run", side_effect=OSError("no git"))
        with no_git, unittest.mock.patch.object(L, "ARCHIVE_COMMIT", f.name):
            self.assertEqual(L.code_version(), "0123456 (archive)")
        with no_git:                                                            # the repo file is still a placeholder
            self.assertEqual(L.code_version(), "unknown")

    def test_git_archive_fills_the_commit(self):
        head = subprocess.run(["git", "-C", P.HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        top = subprocess.run(["git", "-C", P.HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
        out = subprocess.run(["git", "-C", top, "archive", "HEAD", "twin_to_line/ARCHIVE_COMMIT"], capture_output=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        import io, tarfile
        with tarfile.open(fileobj=io.BytesIO(out.stdout)) as t:
            self.assertEqual(t.extractfile("twin_to_line/ARCHIVE_COMMIT").read().decode().strip(), head)

    def test_runtime_files_and_bootstrap(self):
        for f in L.RUNTIME_FILES:
            self.assertTrue(os.path.exists(os.path.join(P.HERE, f)), f)
        b = open(os.path.join(P.HERE, "gb10", "bootstrap.sh")).read()
        self.assertIn("--no-index", b); self.assertIn("--preflight", b); self.assertIn("sha256sum --quiet -c SHA256SUMS", b)
        self.assertTrue(os.access(os.path.join(P.HERE, "gb10", "bootstrap.sh"), os.X_OK))


class FolderIgnoresOldImages(unittest.TestCase):
    """CSI's camera folder will still hold the 22 staged images when the live run starts: folder mode reads new images only."""

    def test_three_old_two_new(self):
        d = tempfile.mkdtemp()
        for k in range(3):
            cv2.imwrite(os.path.join(d, name(k)), fixture(range(1, k + 2)))
        frames = L.watch(d, poll=0.05, idle_exit=0.8)                          # the folder is looked at here
        for k in (3, 4):
            cv2.imwrite(os.path.join(d, name(k)), fixture(range(1, k + 2)))
        out = tempfile.mkdtemp(); L.run(L.Session(out, gate="each", ix=None, quiet=True), frames)
        self.assertEqual([r["input_file"] for r in read_csv(os.path.join(out, "trial_log.csv"))], [name(3), name(4)])

    def test_cli_logs_and_counts_the_ignored_images(self):
        d, out = tempfile.mkdtemp(), os.path.join(tempfile.mkdtemp(), "o")
        for k in range(3):
            cv2.imwrite(os.path.join(d, name(k)), fixture([1]))
        p = subprocess.Popen([PY, "live_inspect.py", "--source", f"folder:{d}", "--gate", "each", "--idle-exit", "1.5",
                              "--out", out], cwd=P.HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(1.0)
        for k in (3, 4):
            cv2.imwrite(os.path.join(d, name(k)), fixture([1, 2]))
        log, _ = p.communicate(timeout=60)
        self.assertEqual(p.returncode, 0, log)
        self.assertIn("3 image(s) already there are ignored", log)
        self.assertEqual([r["input_file"] for r in read_csv(os.path.join(out, "trial_log.csv"))], [name(3), name(4)])
        self.assertEqual(json.load(open(os.path.join(out, "session.json")))["ignored_existing_images"], 3)


class OperatorView(unittest.TestCase):
    def test_console_grid_camera_layout(self):
        res = {"positions": [{"position": f"G{i:02d}", "final_state": st} for i, st in
                             zip(range(1, 31), ["PASS"] * 3 + ["MISSED", "PASS", "DEFECT", "UNSURE"] + ["WAITING"] * 23)]}
        g = L.console_grid(res).splitlines()
        self.assertEqual(len(g), 6)
        self.assertEqual(g[0].split()[::2], ["G26", "G21", "G16", "G11", "G06", "G01"])
        self.assertEqual(g[4].split()[::2], ["G30", "G25", "G20", "G15", "G10", "G05"])
        self.assertTrue(g[0].rstrip().endswith("G06 DEF   G01 ok"))
        self.assertIn("G04 MISS", g[3]); self.assertIn("G07 ??", g[1]); self.assertIn("G26 .", g[0])
        self.assertIn("PASS 4  DEFECT 1  MISSED 1  UNSURE 1  WAITING 23", g[5])
        self.assertNotIn("\x1b", "\n".join(g)); self.assertIn("\x1b[1;31m", L.console_grid(res, color=True))

    def test_view_page_written_with_the_session(self):
        out = os.path.join(tempfile.mkdtemp(), "sess1"); L.Session(out, ix=None, quiet=True).close()
        page = open(os.path.join(out, L.VIEW_HTML)).read()
        self.assertIn('"overlay_latest.png?t=" + Date.now()', page); self.assertIn("setInterval(tick, 1000)", page)
        self.assertIn("sess1", page); self.assertNotIn("http", page)               # no network, file:// only

    @unittest.skipUnless(os.path.isdir(REPLAY1), "DEV run 1 replay folder not exported on this machine")
    def test_dev_run1_replay_grid_and_atomic_overlay(self):
        """DEV run 1 replay through the CLI: one grid per event on the console, and overlay_latest.png is always a
        whole PNG while a reader polls it (written to a temp name, then renamed)."""
        out = os.path.join(tempfile.mkdtemp(), "dev_run1_view"); latest = os.path.join(out, "overlay_latest.png")
        reads, bad, stop = [0], [0], threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    data = open(latest, "rb").read()
                except OSError:
                    time.sleep(0.002); continue
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                reads[0] += 1; bad[0] += img is None

        th = threading.Thread(target=reader); th.start()
        try:
            r = subprocess.run([PY, "live_inspect.py", "--source", f"replay:{REPLAY1}", "--out", out], cwd=P.HERE,
                               capture_output=True, text=True, timeout=600)
        finally:
            stop.set(); th.join()
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(sum(l.startswith("event ") for l in lines), 18)
        self.assertEqual(sum(l.split()[:1] == ["G26"] for l in lines), 18)
        self.assertEqual(sum(l.strip().startswith("PASS ") for l in lines), 18)
        self.assertGreater(reads[0], 0); self.assertEqual(bad[0], 0)
        self.assertFalse(os.path.exists(os.path.join(out, ".overlay_latest.png")))
        self.assertTrue(os.path.exists(os.path.join(out, L.VIEW_HTML)))


class OverwrittenFile(unittest.TestCase):
    def test_one_name_rewritten_three_times_gives_three_frames(self):
        d = tempfile.mkdtemp(); f = os.path.join(d, "latest.png")
        frames = L.watch(d, poll=0.05, idle_exit=1.0)

        def camera():                                                          # rewrites the same file in place
            for k in (1, 2, 3):
                time.sleep(0.4)
                data = cv2.imencode(".png", fixture(range(1, k + 1)))[1].tobytes()
                with open(f, "wb") as fh:
                    fh.write(data)

        th = threading.Thread(target=camera); th.start()
        out = tempfile.mkdtemp(); L.run(L.Session(out, gate="each", ix=None, quiet=True), frames); th.join()
        log = read_csv(os.path.join(out, "trial_log.csv"))
        self.assertEqual([r["input_file"] for r in log], ["latest.png"] * 3)
        self.assertEqual([r["K"] for r in log], ["1", "2", "3"])                # each version read once, whole

    def test_an_old_file_rewritten_after_the_start_is_read(self):
        d = tempfile.mkdtemp(); f = os.path.join(d, "latest.png"); cv2.imwrite(f, fixture([1]))
        frames = L.watch(d, poll=0.05, idle_exit=0.8)                           # the old version is ignored
        time.sleep(0.05); cv2.imwrite(f, fixture([1, 2]))
        out = tempfile.mkdtemp(); L.run(L.Session(out, gate="each", ix=None, quiet=True), frames)
        self.assertEqual([r["K"] for r in read_csv(os.path.join(out, "trial_log.csv"))], ["2"])


class FakeCap:
    def __init__(self, frames=(), opened=True):
        self.frames, self.opened = list(frames), opened

    def isOpened(self):
        return self.opened

    def read(self):
        time.sleep(0.01)
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        pass


class StreamSource(unittest.TestCase):
    def test_drop_cannot_open_and_exception_are_logged_and_reconnected(self):
        hd = warp_up(fixture([1, 2]), FWARP)                                    # the camera sends 1920x1080
        caps = iter([FakeCap([hd] * 5), FakeCap(opened=False), RuntimeError("driver hiccup"), FakeCap([hd] * 2000)])

        def opener(src):
            c = next(caps)
            if isinstance(c, Exception):
                raise c
            return c

        logs, stats = [], {}
        g = L.stream("rtsp://fake/cam", every=0.1, reconnect_s=0.05, log=logs.append, stats=stats, opener=opener)
        out = tempfile.mkdtemp(); L.run(L.Session(out, gate="each", ix=None, quiet=True), itertools.islice(g, 6)); g.close()
        text = "\n".join(logs)
        for want in ("connected, 1920x1080", "dropped after 5 frames", "cannot open", "RuntimeError: driver hiccup"):
            self.assertIn(want, text)
        self.assertEqual((stats["connects"], stats["drops"]), (2, 1))
        log = read_csv(os.path.join(out, "trial_log.csv"))
        self.assertEqual(len(log), 6)
        self.assertEqual({(r["path"], r["status"]) for r in log}, {("1080->640", "scored")})
        self.assertTrue(all(r["input_file"].startswith("stream_") for r in log))
        t = [float(r["frame_t"]) for r in log]; self.assertEqual(t, sorted(t))

    def test_holdout_video_refused_before_decoding(self):
        root = tempfile.mkdtemp(); os.makedirs(os.path.join(root, "images"))
        open(os.path.join(root, "image_index.csv"), "w").write("filename,timestamp,video_id\n")
        ix, d = R.Index(root), tempfile.mkdtemp()
        for run in ("1", "4", "5"):
            open(os.path.join(d, f"cap_seal_video_{run}.mp4"), "wb").close()
        self.assertIsNone(L.stream_refusal(ix, os.path.join(d, "cap_seal_video_1.mp4")))
        for run in ("4", "5"):
            self.assertIn("DEV runs 1-3", L.stream_refusal(ix, os.path.join(d, f"cap_seal_video_{run}.mp4")))
        self.assertIsNone(L.stream_refusal(ix, "rtsp://cam/stream"))           # live sources are never dataset frames

    @unittest.skipUnless(os.path.exists(VIDEO1) and os.path.exists(EVENTS1), "needs the DEV run 1 video")
    def test_dev_run1_video_file_same_events_as_the_replay(self):
        out = tempfile.mkdtemp(); ix = R.Index()
        self.assertIsNone(L.stream_refusal(ix, VIDEO1))
        L.run(L.Session(out, ix=ix, quiet=True), L.stream(VIDEO1, every=1.0))
        log = read_csv(os.path.join(out, "trial_log.csv"))
        self.assertEqual(len(log), 1123)
        self.assertEqual([r["event_frame"] for r in log if r["status"] == "scored"], [r["frame"] for r in read_csv(EVENTS1)])


if __name__ == "__main__":
    unittest.main()
