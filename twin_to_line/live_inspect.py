"""Live runtime: camera frames -> event gating (arm clear, scene stable) -> 30 position crops -> inspector ->
fill-sequence state (LIVE_FILL, D021) -> predictions CSV + trial log + operator overlay. Reads image files only (R10).

    python live_inspect.py --source replay:DIR [--labels labels/dev_labels.csv] [--out DIR]
    python live_inspect.py --source folder:DIR [--poll 0.25] [--idle-exit S]
    python live_inspect.py --source stream:URL|DEVICE|FILE [--every 1.0] [--idle-exit S]
        [--inspector rule|resnet|staging5] [--threshold d038|preregistered] [--model-dir DIR] [--device mps|cpu]
    python live_inspect.py --export-replay 1 --out results/live/replay_run1 [--from video|stills] [--step 1.0]
    python live_inspect.py --preflight PKG/preflight --out DIR      (GB10 package self-check, LIVE_RUNBOOK.md section 7)

Sources (one code path after decoding: Session.process_frame):
  replay:DIR  every image in DIR in frame-time order, as fast as it runs
  folder:DIR  watches DIR; a new image is decoded only when unchanged during its read and its end marker
              (JPEG FFD9 / PNG IEND) is present. Incomplete versions are logged unreadable after 3 s. Images already in DIR at
              start are ignored (the camera folder still holds the staged test images); session.json counts them.
              A file written again under the same name (a camera overwriting one file) is read again
  stream:SRC  an RTSP / HTTP URL via FFmpeg with bounded open/read calls, or a video file. A live source is read
              continuously and its newest frame is passed on every --every s (default 1.0); drops reconnect and a
              10 s watchdog detects stalls. Unbounded backends/devices require folder/stills fallback.
              A video file gives one frame per --every s of its own
              timestamps (real_data.iter_video: the same frames as --export-replay), as fast as it decodes
Frame time: folder sessions prefer increasing filename time, otherwise elapsed arrival time on the same clock;
replay uses a video frame id
(cap_seal_video_1.mp4@t0612.0.png) or a CSI still name
(cap_seal_20260922_192843_993721.jpg); otherwise the file's modification time.
Frame size (the 'path' column of the trial log):
  640x480    the DEV cap_seal grid: processed as is ('640')
  1920x1080  warped into the DEV 640x480 grid with the frozen transform frame_warp_1080.json (frame_warp.py,
             INTER_AREA), then the same 640 path ('1080->640'): position_map_G.json + DEV offset, event gate, rule
             thresholds and crops are all unchanged. session.json records the transform file's sha256
  any other  wrong_size, nothing processed
Gating (--gate):
  events (default)  events.py's rule, online: arm-clear frames with no position changing by >= CHANGE_T form a
                    segment; one that lasts >= STABLE_S and differs from the previous event is an event, and its
                    FIRST frame is scored (the same frames as events.segment_events, so frame ids join DEV labels)
  each              every arm-clear frame is its own event (staged stills: one image per setup)
Frames with the arm over the fixture are never scored. Fill state from occupancy (LIVE_FILL default), or
fill_state.SequenceTracker with --track (persist 1: each event is already a stable segment).
Outputs in --out (default results/live/<UTC time>):
  predictions.csv     one row per position per event (inspect_frame.py schema; eval_metrics.py reads it)
  trial_log.csv       one row per input frame, every attempt: status, event, per-position state + score, expected
                      state (--labels), latency, notes; an empty 'intervention' column for the operator
  console             operator view: per event, the 30 holes in the camera layout (G26 ... G01 across the top row)
                      with a short state each, then the counts
  overlay_latest.png  the latest event's overlay (30 coloured boxes + legend), replaced atomically (temp + rename);
                      overlays/eNN_<frame>.png; view.html reloads overlay_latest.png every second over file://
  session.json        configuration, code / thresholds / map hashes, counts
Data rule (D011): a frame that is in the dataset index but not in DEV runs 1-3 is refused, so runs 4-5 are never
processed; images from outside the dataset (staged / live captures) are accepted.
Inspectors (--inspector):
  rule (default)  rule baseline; rule_thresholds.json was fitted on DEV draft labels (a placeholder, not a scored arm)
  resnet          synthetic-only ResNet-18 v1 (resnet_inspector.py), threshold D038 (default) or the pre-registered one.
                  G3 on DEV runs 1-3 was NO-GO (87.7 % false positives at D038), so per the plan the rule baseline
                  runs live and the ResNet is scored offline as arm A; the step-9 choice is Noor's.
  staging5        five-output staging-data ResNet-18 exported to ONNX (resnet5_onnx_inspector.py). An explicit
                  --model-dir is required and model provenance is verified before any frame is read. This is a
                  shadow/evaluation-only demo: it never actuates, releases, or automatically passes product.
Preflight (--preflight): prints OS, CPU, glibc, Python, numpy / OpenCV versions and the sha256 of every runtime file,
then scores the package's bundled frames (the 1080 still, 10 DEV run-1 event frames at 640 and as 1080 copies) with
--gate each and compares every hole with the Mac's predictions shipped next to them. Exit 0 = PREFLIGHT OK.
"""
import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys, threading, time
from collections import Counter
from datetime import datetime, timezone
import cv2, numpy as np
import positions as P
import real_data as R
import events as E
import frame_warp as FW
from eval_metrics import gt_final
from fill_state import SequenceTracker, LIVE_FILL, MODES
from inspect_frame import inspect, annotate, prediction_rows, safe
from rules import RuleInspector

IMAGE_EXT = (".png", ".jpg", ".jpeg")
STILL_RE = re.compile(r"(\d{8})_(\d{6})_(\d{6})")
FLAGGED = ("DEFECT", "MISSED", "UNSURE")
RUNTIME_FILES = ("live_inspect.py", "inspect_frame.py", "positions.py", "fill_state.py", "rules.py", "events.py",
                 "real_data.py", "position_map_G.json", "registration_dev.json", "rule_thresholds.json",
                 "resnet_inspector.py", "resnet5_onnx_inspector.py", "staging_dataset.py", "train_resnet.py",
                 "train_resnet5.py", "frame_warp.py", "frame_warp_1080.json")
PRED_FIELDS = ["source_image_id", "fixture_id", "position_id", "expected_occupancy", "expected_occupancy_source",
               "predicted_visual_class", "model_argmax_class", "model_score", "applicable_threshold",
               "inspection_decision", "reason_code",
               "inspection_timestamp_utc", "event_uid", "run", "event", "frame", "position", "predicted_state",
               "defect_score", "pred_sequence", "final_state", "K", "K_source", "cap_frac", "cap_offset", "reason"]
IDS = [f"G{i:02d}" for i in range(1, 31)]
TRIAL_FIELDS = (["time_utc", "session", "git", "inspector", "model_sha256", "input_file", "frame_t", "status", "arm_px",
                 "event", "event_frame", "event_delay_s", "K", "n_flagged", "n_differ_from_expected", "latency_ms"]
                + [f"pred_{p}" for p in IDS] + [f"exp_{p}" for p in IDS] + ["path", "note", "intervention"])
PATH_640, PATH_1080 = "640", "1080->640"
ARCHIVE_COMMIT = os.path.join(P.HERE, "ARCHIVE_COMMIT")      # filled in by `git archive` (export-subst)
GRID_CODE = {"PASS": "ok", "DEFECT": "DEF", "MISSED": "MISS", "UNSURE": "??", "WAITING": "."}
GRID_ANSI = {"PASS": "32", "DEFECT": "1;31", "MISSED": "1;35", "UNSURE": "1;33", "WAITING": "2"}
VIEW_HTML = "view.html"
SHADOW_BANNER = ("SAFETY: STAGING5 SHADOW/EVALUATION ONLY — no automatic product release, line actuation, "
                 "or deterministic-rule override.")
VIEW_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Twin-to-Line live view: {session}</title>
<style>body{{margin:0;background:#111;color:#ddd;font:14px sans-serif}}#s{{padding:6px 10px}}
img{{display:block;max-width:100vw;max-height:calc(100vh - 32px);margin:auto}}</style></head>
<body><div id="s">{session}: waiting for the first event</div><img id="v" alt="">
<script>
// live_inspect.py replaces overlay_latest.png atomically after every event; reload it every second.
// A load that fails (no event yet) keeps the last image on screen.
const v = document.getElementById("v"), s = document.getElementById("s");
function tick() {{
  const n = new Image();
  n.onload = () => {{ v.src = n.src; s.textContent = "{session}: overlay_latest.png, checked " + new Date().toLocaleTimeString(); }};
  n.src = "overlay_latest.png?t=" + Date.now();
}}
tick(); setInterval(tick, 1000);
</script></body></html>
"""
PREFLIGHT_SETS = (("still_1080", PATH_1080), ("run1_640", PATH_640), ("run1_1080", PATH_1080))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def code_version():
    """Short git sha, '+modified' when a runtime source file differs from HEAD (others, e.g. dose_sets.py, ignored).
    Without git (a `git archive` package): the archived commit, '<sha> (archive)'; the preflight prints file hashes."""
    try:
        g = lambda *a: subprocess.run(["git", "-C", P.HERE, *a], capture_output=True, text=True, check=True).stdout.strip()
        return g("rev-parse", "--short", "HEAD") + ("+modified" if g("status", "--porcelain", "--", *RUNTIME_FILES) else "")
    except (OSError, subprocess.CalledProcessError):
        try:
            c = open(ARCHIVE_COMMIT).read().strip()
            return "unknown" if not c or c.startswith("$Format") else f"{c[:7]} (archive)"
        except OSError:
            return "unknown"


def frame_id(path):
    """Frame id as the labels and inspect_frame.py use it: a cached video frame 'cap_seal_video_1.mp4@t0612.0.png'
    is the video frame id without '.png'; any other image keeps its file name."""
    name = os.path.basename(path)
    return name[:-4] if R.VIDEO_RE.match(name) and name.endswith(".png") else name


def filename_time(path):
    """Parsed capture time, or None when the name carries no timestamp (never use mtime for live arrivals)."""
    name = os.path.basename(path)
    m = R.VIDEO_RE.match(name)
    if m:
        return float(m.group(2))
    m = STILL_RE.search(name)
    if m:
        return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S%f").replace(tzinfo=timezone.utc).timestamp()
    return None


def frame_time(path):
    t = filename_time(path)
    return os.path.getmtime(path) if t is None else t


def event_timestamp_utc(path, event_time):
    """Stable UTC capture/arrival time when the event clock is an epoch; video-relative clocks stay blank."""
    try:
        captured = filename_time(path)
    except ValueError:
        captured = None
    value = event_time if captured is None else captured
    return (datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds")
            if value is not None and value >= 946684800 else "")


class FolderClock:
    """Prefer forward filename time. Otherwise advance by elapsed arrival time in the previous frame's timebase."""
    def __init__(self):
        self.previous = self.arrival = None

    def time(self, path, arrival):
        try:
            t = filename_time(path)
        except ValueError:                         # malformed date/time: use arrival, not a per-frame error
            t = None
        if t is None or (self.previous is not None and t <= self.previous):
            t = arrival if self.previous is None else self.previous + max(0., arrival - self.arrival)
        self.previous, self.arrival = t, arrival
        return t


def dataset_index():
    # GB10 carries only filename/run metadata, never holdout pixels.
    for root in (R.REAL_ROOT, os.path.join(P.HERE, "..", "..", "data_index")):
        try:
            return R.Index(root)
        except FileNotFoundError:
            pass
    return None


def refusal(ix, path):
    """Why a frame must not be processed (dataset frame outside DEV runs 1-3), or None."""
    try:
        video = R.VIDEO_RE.match(os.path.basename(path))
        run = video.group(1) if video else ix.run_of_source(path) if ix else None
        if run is not None:
            R.check_allowed(run, role_map=ix.role_map if ix else None)
    except R.ProtectedDataError as e:
        return str(e)
    return None


def is_image(name):
    return name.lower().endswith(IMAGE_EXT) and not name.startswith(".")


def complete(path):
    """The file ends with its format's end marker (a half-written JPEG still decodes, as a partly grey image)."""
    try:
        with open(path, "rb") as f:
            tail = f.read()
    except OSError:
        return False
    if path.lower().endswith(".png"):
        return tail.endswith(b"IEND\xaeB`\x82")
    return tail.rstrip(b"\x00").endswith(b"\xff\xd9")


def replay(folder, ix=None):
    """(path, image) for every image in folder, in frame-time order. The folder is checked when replay() is
    called: one protected frame refuses all of it before anything is read."""
    paths = [os.path.join(folder, n) for n in os.listdir(folder) if is_image(n)]
    bad = [r for r in (refusal(ix, p) for p in paths) if r]
    if bad:
        raise R.ProtectedDataError(f"{len(bad)} protected frame(s) in {folder}, e.g. {bad[0]}")
    paths.sort(key=lambda p: (frame_time(p), os.path.basename(p)))
    return ((p, read_version(p)[0]) for p in paths)


def file_version(path):
    """Include identity and ctime to detect replacements even when size and mtime are preserved."""
    st = os.stat(path)
    return st.st_mtime_ns, st.st_size, st.st_dev, st.st_ino, st.st_ctime_ns


def read_version(path, version=None):
    """Decode only complete bytes from a version unchanged throughout the read. No second file open for decoding."""
    try:
        version = file_version(path) if version is None else version
        with open(path, "rb") as f:
            data = f.read()
        unchanged = file_version(path) == version
    except OSError:
        try:
            unchanged = version is not None and file_version(path) == version
        except OSError:
            unchanged = False
        return None, False, unchanged
    finished = (data.endswith(b"IEND\xaeB`\x82") if path.lower().endswith(".png")
                else data.rstrip(b"\x00").endswith(b"\xff\xd9"))
    img = None
    if unchanged and finished:
        try:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            pass
    return img, finished, unchanged


def watch(folder, poll=0.25, idle_exit=None, give_up_s=3.0, skip=None, stats=None, ix=None):
    """(path, image, time) for each acceptable version: advancing filename time, otherwise elapsed arrival time.
    An unacceptable version (including permission errors) is logged unreadable once after give_up_s.
    Replacements under the same name are read again.
    Runs until Ctrl-C, or until nothing new has arrived for idle_exit seconds.
    Images already in the folder when watch() is called are not read unless they are written again (CSI's camera
    folder still holds the staged test images); skip = the names to ignore instead (() reads everything)."""
    try:
        names = [n for n in os.listdir(folder) if is_image(n)] if skip is None else skip
    except OSError as e:
        return _watch(folder, poll, idle_exit, give_up_s, None, stats, ix, f"{type(e).__name__}: {e}")
    seen = {}
    for n in names:
        try:
            seen[n] = file_version(os.path.join(folder, n))
        except OSError:
            pass
    if stats is not None:
        stats["ignored_existing_images"] = len(seen)
    return _watch(folder, poll, idle_exit, give_up_s, seen, stats, ix)


def _watch(folder, poll, idle_exit, give_up_s, seen, stats, ix, initial_error=None):
    """seen: {name: version} of what was read (or ignored at start); pending: {name: (version, first seen)}."""
    pending, last = {}, time.monotonic()
    clock = FolderClock()
    failed = False
    if initial_error:
        print(f"WARNING folder {folder}: {initial_error}; retrying in {poll:g} s", flush=True)
        yield folder, None, last, initial_error
        failed = True
        time.sleep(poll)
    while True:
        try:
            names = sorted(os.listdir(folder))
        except OSError as e:
            note = f"{type(e).__name__}: {e}"
            print(f"WARNING folder {folder}: {note}; retrying in {poll:g} s", flush=True)
            yield folder, None, time.monotonic(), note
            failed = True
            if idle_exit is not None and time.monotonic() - last > idle_exit:
                return
            time.sleep(poll)
            continue
        if seen is None:                                 # first accessible snapshot still excludes staged images
            seen = {}
            for n in names:
                if is_image(n):
                    try:
                        seen[n] = file_version(os.path.join(folder, n))
                    except OSError:
                        pass
            if stats is not None:
                stats["ignored_existing_images"] = len(seen)
        if failed:
            yield folder, None, time.monotonic(), ""
            failed = False
        for n in names:
            if not is_image(n):
                continue
            p = os.path.join(folder, n)
            stat_failed = False
            try:
                v = file_version(p)
            except FileNotFoundError:
                continue
            except OSError as e:
                v, stat_failed = ("unavailable", type(e).__name__), True
            if seen.get(n) == v:
                continue
            now = time.monotonic()
            if n not in pending or pending[n][0] != v:
                pending[n] = (v, now)
            refused = refusal(ix, p)
            img, finished, unchanged = ((None, False, True) if stat_failed or refused else read_version(p, v))
            if not unchanged:                           # rewritten while being read: take the next version instead
                pending.pop(n, None); continue
            if refused or (finished and img is not None) or now - pending[n][1] >= give_up_s:
                seen[n] = v; del pending[n]; last = now
                yield p, img if finished and unchanged else None, clock.time(p, now)
        if idle_exit is not None and time.monotonic() - last > idle_exit:
            return
        time.sleep(poll)


CAPTURE_TIMEOUT_MS = 5000
STREAM_STALL_S = 10.0
CAPTURE_JOIN_S = CAPTURE_TIMEOUT_MS / 1000 + 0.5
STREAM_FALLBACK = "use folder mode or stills saved by CSI (--source folder:DIR --gate each)"


def open_capture(src):
    """Live URLs require FFmpeg's open-time timeout parameters; never fall back to an unbounded backend."""
    if src.isdigit() or src.startswith("/dev/"):
        raise ValueError(f"device reads cannot be bounded; {STREAM_FALLBACK}")
    try:
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CAPTURE_TIMEOUT_MS,
                                cv2.CAP_PROP_READ_TIMEOUT_MSEC, CAPTURE_TIMEOUT_MS])
    except (AttributeError, cv2.error) as e:
        raise ValueError(f"FFmpeg timeout parameters unavailable; {STREAM_FALLBACK}") from e


class LatestFrame:
    """One capture worker owns open/read/release; a supervisor detects stalls. Only the newest frame is retained.
    A backend ignoring its deadlines stops the source, without spawning more blocked workers or racing release."""

    def __init__(self, src, reconnect_s=2.0, log=print, opener=open_capture):
        self.src, self.reconnect_s, self.log, self.opener = src, reconnect_s, log, opener
        self.lock, self.stopped = threading.Lock(), threading.Event()
        self.frame = self.t = None
        self.failure = None
        self.n = self.connects = self.drops = 0
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def _capture(self, cancel, done, progress):
        cap, got = None, 0
        try:
            cap = self.opener(self.src)
            if not cap.isOpened():
                self.log(f"WARNING stream {self.src}: cannot open with FFmpeg timeouts; retrying in {self.reconnect_s:g} s; {STREAM_FALLBACK}")
            else:
                self.connects += 1
                while not self.stopped.is_set() and not cancel.is_set():
                    ok, img = cap.read()
                    if not ok or img is None or cancel.is_set() or self.stopped.is_set():
                        break
                    got += 1
                    if got == 1:
                        self.log(f"stream {self.src}: connected, {img.shape[1]}x{img.shape[0]}")
                    with self.lock:
                        self.frame, self.t, self.n = img, time.time(), self.n + 1
                        progress[0] = time.monotonic()
                if not self.stopped.is_set():
                    self.drops += 1
                    self.log(f"WARNING stream {self.src}: dropped after {got} frames; reconnecting in {self.reconnect_s:g} s")
        except ValueError as e:
            self.failure = str(e)
            self.log(f"WARNING stream {self.src}: {e}; {STREAM_FALLBACK}")
            self.stopped.set()
        except Exception as e:
            self.log(f"WARNING stream {self.src}: {type(e).__name__}: {e}; reconnecting in {self.reconnect_s:g} s")
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            done.set()

    def _run(self):
        while not self.stopped.is_set():
            cancel, done = threading.Event(), threading.Event()
            progress = [time.monotonic()]
            worker = threading.Thread(target=self._capture, args=(cancel, done, progress), daemon=True)
            worker.start()
            while not done.wait(min(.1, STREAM_STALL_S / 4)):
                if self.stopped.is_set():
                    break
                with self.lock:
                    stalled = time.monotonic() - progress[0] >= STREAM_STALL_S
                if stalled:
                    self.log(f"WARNING stream {self.src}: no new frame for {STREAM_STALL_S:g} s; releasing and reconnecting")
                    break
            cancel.set()
            worker.join(timeout=CAPTURE_JOIN_S)
            if worker.is_alive():
                self.failure = f"capture did not stop within its deadline; {STREAM_FALLBACK}"
                self.log(f"WARNING stream {self.src}: {self.failure}")
                self.stopped.set()
                return
            self.stopped.wait(self.reconnect_s)

    def latest(self):
        with self.lock:
            return self.n, self.frame, self.t

    def stop(self):
        self.stopped.set(); self.thread.join(timeout=CAPTURE_JOIN_S + .2)


def stream(src, every=1.0, idle_exit=None, reconnect_s=2.0, log=print, stats=None, opener=open_capture):
    """(name, image, t) for stream:SRC. A video file: one frame per `every` s of its own timestamps (the frames of
    real_data.iter_video, named like the replay frames). A live source: the newest frame every `every` s of wall
    time, named stream_<UTC time>; a tick without a new frame passes nothing (idle_exit: stop after that many s)."""
    if os.path.isfile(src):
        base = os.path.basename(src)
        for t, img in R.iter_video(src, every):
            yield f"{base}@t{t:06.1f}", img, t
        return
    reader = LatestFrame(src, reconnect_s, log, opener)
    last_n, tick = 0, time.monotonic(); last_new = tick
    try:
        while True:
            tick = max(tick + every, time.monotonic()); time.sleep(max(0.0, tick - time.monotonic()))
            n, img, t = reader.latest()
            if reader.stopped.is_set() and reader.failure:
                raise RuntimeError(reader.failure)
            if n == last_n:
                if idle_exit is not None and time.monotonic() - last_new > idle_exit:
                    return
                continue
            last_n, last_new = n, time.monotonic()
            yield "stream_" + datetime.fromtimestamp(t, timezone.utc).strftime("%Y%m%d_%H%M%S_%f"), img, t
    finally:
        reader.stop()
        if stats is not None:
            stats.update(connects=reader.connects, drops=reader.drops, frames_read=reader.n)


def stream_refusal(ix, src):
    """A dataset video outside DEV runs 1-3 is refused before anything is decoded (its frames carry the run)."""
    return refusal(ix, f"{os.path.basename(src)}@t0000.0") if os.path.isfile(src) else None


class EventGate:
    """events.segment_events, online. update() once per frame, in time order. Returns (status, frame, changed):
    'arm' (arm over the fixture), 'settling' (segment not yet stable for stable_s), 'unchanged' (segment already
    decided, or the same as the last event) or 'event' with frame = (name, t, image) of the segment's first frame."""

    def __init__(self, boxes, ids, change_t=E.CHANGE_T, stable_s=E.STABLE_S, each=False):
        self.boxes, self.ids, self.change_t, self.stable_s, self.each = boxes, ids, change_t, stable_s, each
        self.first = self.prev = self.last_event = None
        self.decided = False

    def _changed(self, img):
        if self.last_event is None:
            return list(self.ids)
        return [p for p, x in zip(self.ids, E.box_change(img, self.last_event, self.boxes)) if x >= self.change_t]

    def update(self, name, t, img, arm):
        if arm:
            self.first = self.prev = None
            return "arm", None, None
        if self.each:
            changed = self._changed(img); self.last_event = img
            return "event", (name, t, img), changed
        if self.first is not None and E.box_change(img, self.prev[2], self.boxes).max() >= self.change_t:
            self.first = None
        if self.first is None:
            self.first, self.decided = (name, t, img), False
        self.prev = (name, t, img)
        if self.decided:
            return "unchanged", None, None
        if t - self.first[1] < self.stable_s:
            return "settling", None, None
        self.decided = True
        changed = self._changed(self.first[2])
        if not changed:
            return "unchanged", None, None
        self.last_event = self.first[2]
        return "event", self.first, changed


def console_grid(res, color=False):
    """Operator view for the console: the 30 holes in the camera layout (G26 ... G01 across the top row, as in the
    staging doc), one short state each, then the counts."""
    st = {p["position"]: p["final_state"] for p in res["positions"]}
    rows = []
    for r in range(5):
        cells = []
        for c in range(6):
            pid = f"G{(5 - c) * 5 + r + 1:02d}"; cell = f"{pid} {GRID_CODE[st[pid]]:<4s}"
            cells.append(f"\x1b[{GRID_ANSI[st[pid]]}m{cell}\x1b[0m" if color else cell)
        rows.append("    " + "  ".join(cells))
    n = Counter(st.values())
    rows.append("    " + "  ".join(f"{s} {n.get(s, 0)}" for s in ("PASS", "DEFECT", "MISSED", "UNSURE", "WAITING"))
                + "    (" + "  ".join(f"{GRID_CODE[s]} = {s}" for s in GRID_CODE) + ")")
    return "\n".join(rows)


def expected_states(path):
    """{(frame, position): expected final state} from a labels CSV (eval_metrics ground-truth columns)."""
    with open(path, newline="") as f:
        return {(r["frame"], r["position"]): gt_final(r) for r in csv.DictReader(f) if r.get("usable", "1") != "0"}


class CsvLog:
    """CSV written and flushed row by row, so Ctrl-C or a crash never loses a logged row."""

    def __init__(self, path, fields):
        # Session output directories are fresh.  Exclusive creation closes the remaining race between the
        # directory check and opening evidence files: an existing log is never truncated.
        self.f = open(path, "x", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore"); self.w.writeheader(); self.f.flush()

    def write(self, rows):
        self.w.writerows(rows); self.f.flush()

    def close(self):
        self.f.close()


class Session:
    def __init__(self, out, source="", run_id=None, gate="events", mode=LIVE_FILL, track=False, labels=None,
                 note="", ix="auto", pmap=None, inspector=None, quiet=False, warp="auto", fixture_id="G"):
        if fixture_id != "G":
            raise ValueError("fixture S is refused: no calibrated fixture-S position map is available")
        if os.path.exists(out):
            if not os.path.isdir(out):
                raise FileExistsError(f"refusing nonempty live output directory: {out}; preserve existing evidence")
            with os.scandir(out) as entries:
                if next(entries, None) is not None:
                    raise FileExistsError(
                        f"refusing nonempty live output directory: {out}; preserve existing evidence"
                    )
        else:
            os.makedirs(out, exist_ok=False)
        self.out, self.run_id, self.mode, self.quiet = out, run_id, mode, quiet
        self.fixture_id = fixture_id
        self.pm = pmap or P.PositionMap(); self.plate = self.pm.plate_mask()
        self.warp = FW.load() if warp == "auto" else warp           # None: 1080 frames are wrong_size
        if self.warp is not None and self.warp.size_out != tuple(self.pm.size):
            raise ValueError(f"frame warp outputs {self.warp.size_out}, the position map is {self.pm.size}")
        self.insp = inspector or RuleInspector()
        self.gate = EventGate(self.pm.anchor_boxes, self.pm.ids, each=gate == "each")
        self.tracker = SequenceTracker(1, mode) if track else None
        self.ix = dataset_index() if ix == "auto" else ix
        self.expected = expected_states(labels) if labels else {}
        self.name = os.path.basename(os.path.normpath(out))
        insp_info = self.insp.info()
        files = insp_info.get("files", {})
        if not isinstance(files, dict):
            raise ValueError(f"{self.insp.name}.info()['files'] must be a path-to-sha256 mapping")
        model_sha = insp_info.get("model_sha256", "")
        if not model_sha:
            model_sha = next((digest for path, digest in files.items()
                              if os.path.basename(path) in {"model.onnx", "model.pt"}), "")
        self.git, self.model_sha = code_version(), model_sha or next(iter(files.values()), "")
        self.events, self.counts, self.latency, self.stop_reason = 0, Counter(), [], "end of source"
        self.source_error = None
        os.mkdir(os.path.join(out, "overlays"))
        with open(os.path.join(out, VIEW_HTML), "x") as f:
            f.write(VIEW_PAGE.format(session=self.name))
        self.preds = CsvLog(os.path.join(out, "predictions.csv"), PRED_FIELDS)
        self.trial = CsvLog(os.path.join(out, "trial_log.csv"), TRIAL_FIELDS)
        self.info = {"session": self.name, "started_utc": self.now(), "source": source, "fixture_id": fixture_id,
                     "gate": gate, "mode": mode,
                     "track": track, "run_id": run_id, "labels": labels, "note": note, "git": self.git,
                     "inspector": insp_info,
                     "operation": {"mode": "shadow_evaluation_only", "actuation_enabled": False,
                                   "automatic_release_enabled": False},
                     "position_map": {"file": os.path.relpath(P.MAP_PATH, P.HERE), "sha256": sha256(P.MAP_PATH),
                                      "image_size": list(self.pm.size), "offset_px": list(self.pm.offset),
                                      "offset_source": self.pm.offset_source},
                     "frame_warp": self.warp.info() if self.warp else None,
                     "event_rule": {"change_t": E.CHANGE_T, "stable_s": E.STABLE_S, "arm_min_area": E.ARM_MIN_AREA},
                     "code_sha256": {f: sha256(os.path.join(P.HERE, f)) for f in RUNTIME_FILES
                                     if os.path.isfile(os.path.join(P.HERE, f))}}
        self.write_info()

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def write_info(self, **extra):
        self.info.update(extra)
        json.dump(self.info, open(os.path.join(self.out, "session.json"), "w"), indent=1)

    def run_of(self, name):
        if self.run_id:
            return self.run_id
        try:
            return (self.ix.run_of_source(name) if self.ix else None) or "live"
        except R.ProtectedDataError:
            return "live"

    def process_frame(self, path, img, t=None, source_error=None):
        """One input frame (any source). Logs exactly one trial-log row and returns it."""
        t0 = time.perf_counter(); name = frame_id(path)
        row = {"time_utc": self.now(), "session": self.name, "git": self.git, "inspector": self.insp.name,
               "model_sha256": self.model_sha, "input_file": os.path.basename(path)}
        try:
            why = refusal(self.ix, path)
            if source_error is not None:
                self.source_error = source_error or None
                row.update(status="source_error" if source_error else "source_recovered", note=source_error)
            elif why:
                row.update(status="refused", note=why)
            elif img is None:
                row.update(status="unreadable", note="could not decode the image")
            elif img.shape[1::-1] == tuple(self.pm.size):
                row["path"] = PATH_640
            elif self.warp is not None and img.shape[1::-1] == self.warp.size_in:
                img, row["path"] = self.warp.apply(img), PATH_1080
            else:
                ok = [self.pm.size] + ([self.warp.size_in] if self.warp else [])
                row.update(status="wrong_size", note=f"{img.shape[1]}x{img.shape[0]}, expected " + " or ".join(f"{w}x{h}" for w, h in ok))
            if row.get("path"):
                t = frame_time(path) if t is None else t
                arm = E.arm_pixels(img, self.pm, self.plate)
                status, ev, changed = self.gate.update(name, t, img, arm > 0)
                row.update(frame_t=round(t, 3), status=status, arm_px=arm)
                if status == "event":
                    row.update(self.score(ev, changed, t))
        except Exception as e:                                  # never let one bad frame stop the session
            row.update(status="error", note=f"{type(e).__name__}: {e}")
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self.counts[row["status"]] += 1; self.latency.append(row["latency_ms"])
        self.trial.write([row])
        return row

    def score(self, ev, changed, t_now):
        name, t, img = ev
        self.events += 1
        res = inspect(img, name, self.pm, self.insp, plate=self.plate, mode=self.mode, tracker=self.tracker,
                      fixture_id=self.fixture_id)
        run_id = self.run_of(name)
        self.preds.write(prediction_rows(run_id, self.events, res,
                                         event_uid=f"{run_id}:{name}:{t:.6f}",
                                         inspection_timestamp_utc=event_timestamp_utc(name, t)))
        view = annotate(img, res, self.pm)
        cv2.imwrite(os.path.join(self.out, "overlays", f"e{self.events:02d}_{safe(name)}.png"), view)
        tmp = os.path.join(self.out, ".overlay_latest.png"); cv2.imwrite(tmp, view)
        os.replace(tmp, os.path.join(self.out, "overlay_latest.png"))   # a viewer never sees half a file
        pos = res["positions"]
        row = {"status": "scored", "event": self.events, "event_frame": name, "event_delay_s": round(t_now - t, 1),
               "K": res["K"], "n_flagged": sum(p["final_state"] in FLAGGED for p in pos),
               "note": "changed: " + (" ".join(changed) if len(changed) < 30 else "ALL")}
        for p in pos:
            row[f"pred_{p['position']}"] = f"{p['final_state']} {p['defect_score']:.3f}"
            exp = self.expected.get((name, p["position"]), "")
            row[f"exp_{p['position']}"] = exp
        if any(self.expected.get((name, p["position"])) for p in pos):
            row["n_differ_from_expected"] = sum(self.expected.get((name, p["position"])) not in (None, p["final_state"]) for p in pos)
        if not self.quiet:
            c = Counter(p["final_state"] for p in pos)
            print(f"event {self.events:2d} {name[-34:]:>34s} K={res['K']:2d} delay {row['event_delay_s']:5.1f}s  "
                  + " ".join(f"{s}={n}" for s, n in sorted(c.items())))
            print(console_grid(res, color=sys.stdout.isatty()), flush=True)
        return row

    def close(self):
        self.preds.close(); self.trial.close()
        if self.source_error and self.stop_reason == "end of source":
            self.stop_reason = f"source error: {self.source_error}"
        lat = sorted(self.latency)
        self.write_info(ended_utc=self.now(), stop_reason=self.stop_reason, frames=sum(self.counts.values()),
                        events=self.events, status_counts=dict(self.counts),
                        latency_ms_median=lat[len(lat) // 2] if lat else None)
        if not self.quiet:
            print(f"{sum(self.counts.values())} frames, {self.events} events, {dict(self.counts)} -> {self.out}")


def make_inspector(kind="rule", threshold="d038", model_dir=None, device=None, quiet=False):
    """The inspector for a session, loaded (and, for the ResNet, warmed up) before the first frame arrives."""
    if kind == "rule":
        insp = RuleInspector()
    elif kind == "resnet":
        from resnet_inspector import MODEL_DIR, ResNetInspector
        insp = ResNetInspector(model_dir or MODEL_DIR, threshold, device)
        insp.warmup()
    elif kind == "staging5":
        if not model_dir:
            raise ValueError("--inspector staging5 requires an explicit --model-dir; no model fallback is allowed")
        if device not in (None, "cpu"):
            raise ValueError("the OpenCV ONNX staging5 runtime supports --device cpu only")
        from resnet5_onnx_inspector import ResNet5OnnxInspector
        insp = ResNet5OnnxInspector(model_dir)
        insp.warmup()
    else:
        raise ValueError(f"unknown inspector {kind!r}")
    if not quiet:
        i = insp.info(); f, h = next(iter(i["files"].items()), ("", ""))
        if kind == "resnet":
            extra = f" t={i['t_defect']:.5f} (pre-registered {i['t_defect_preregistered']:.5f}) on {insp.device}"
        elif kind == "staging5":
            extra = f" t={i['t_defect']:.5f} on {getattr(insp, 'device', 'cpu')}"
        else:
            extra = ""
        print(f"inspector {i['name']}{extra}  {f} {h[:12]}")
        if kind == "staging5":
            print(SHADOW_BANNER)
    return insp


def run(session, frames):
    """Feed every (path, image[, t]) of a source through the session; Ctrl-C stops cleanly with all logs flushed."""
    try:
        for f in frames:
            session.process_frame(*f)
    except KeyboardInterrupt:
        session.stop_reason = "ctrl-c"
    except Exception as e:
        session.stop_reason = f"source error: {type(e).__name__}: {e}"
        raise
    finally:
        getattr(frames, "close", lambda: None)()                # stops a stream's reader thread
        session.close()


def environment():
    import importlib.util, platform
    try:
        osr = dict(l.rstrip("\n").split("=", 1) for l in open("/etc/os-release") if "=" in l)
    except OSError:
        osr = {}
    return {"os": osr.get("PRETTY_NAME", "").strip('"') or platform.platform(), "kernel": platform.release(),
            "arch": platform.machine(), "glibc": " ".join(platform.libc_ver()).strip() or "n/a",
            "python": f"{platform.python_version()} ({sys.executable})", "numpy": np.__version__, "opencv": cv2.__version__,
            "torch": "installed, not used" if importlib.util.find_spec("torch") else "not installed (not needed)"}


def score_folder(folder, out):
    """Every image in folder scored on its own (--gate each, no tracker, no dataset index). Returns (predictions, trial log)."""
    run(Session(out, f"replay:{folder}", gate="each", ix=None, quiet=True), replay(folder))
    rd = lambda n: list(csv.DictReader(open(os.path.join(out, n), newline="")))
    return rd("predictions.csv"), rd("trial_log.csv")


def preflight(data, out):
    """Package self-check: environment and runtime file hashes, then every bundled set scored and compared hole by hole
    with the Mac's predictions (data/expected/<set>.csv). True when every set matches."""
    env = {**environment(), "code": code_version()}
    for k, v in env.items():
        print(f"{k:8s} {v}")
    hashes = {f: sha256(os.path.join(P.HERE, f)) for f in RUNTIME_FILES if os.path.isfile(os.path.join(P.HERE, f))}
    for f, h in hashes.items():
        print(f"sha256   {h}  {f}")
    ok, sets = True, {}
    for name, path in PREFLIGHT_SETS:
        preds, trial = score_folder(os.path.join(data, name), os.path.join(out, name))
        exp = {(r["frame"], r["position"]): r for r in csv.DictReader(open(os.path.join(data, "expected", name + ".csv"), newline=""))}
        got = {(r["frame"], r["position"]): r for r in preds}
        differ = [k for k in exp if k not in got or (got[k]["predicted_state"], got[k]["final_state"]) != (exp[k]["predicted_state"], exp[k]["final_state"])]
        diffs = [abs(float(got[k]["defect_score"]) - float(exp[k]["defect_score"])) for k in exp if k in got]
        dmax = max(diffs, default=float("inf")) if all(np.isfinite(d) for d in diffs) else float("inf")
        st, paths = Counter(r["status"] for r in trial), {r["path"] for r in trial}
        good = not differ and set(got) == set(exp) and set(st) == {"scored"} and paths == {path} and dmax == 0
        ok &= good
        sets[name] = {"frames": len(trial), "holes": len(got), "final_states": dict(Counter(r["final_state"] for r in preds)),
                      "differ_from_mac": [list(k) for k in differ], "max_score_diff": dmax if np.isfinite(dmax) else None, "statuses": dict(st),
                      "paths": sorted(paths), "ok": good}
        print(f"{name:10s} {len(trial):2d} frames, path {'/'.join(sorted(paths))}, {dict(Counter(r['final_state'] for r in preds))}; "
              f"vs Mac: {len(differ)} holes differ, max |score diff| {dmax:.2g} -> {'OK' if good else 'FAIL'}")
    json.dump({"environment": env, "sha256": hashes, "sets": sets, "ok": ok}, open(os.path.join(out, "preflight.json"), "w"), indent=1)
    print("PREFLIGHT OK" if ok else "PREFLIGHT FAILED", f"-> {os.path.join(out, 'preflight.json')}")
    return ok


def export_replay(run_id, out, source="video", step=1.0):
    """A DEV run as a replay folder: video frames every `step` s (PNG, named by video frame id) or the run's stills
    (copied unchanged). Runs 4-5 are refused by real_data."""
    ix = R.Index(); os.makedirs(out, exist_ok=True); n = 0
    if source == "video":
        for t, img in R.iter_video(ix.video_path(run_id), step):
            cv2.imwrite(os.path.join(out, R.video_id(run_id, t) + ".png"), img); n += 1
    else:
        for f in ix.frames(run_id):
            shutil.copy2(os.path.join(ix.root, "images", f), out); n += 1
    print(f"run {run_id}: {n} frames ({source}) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="live inspection runtime (replay or watched folder)")
    ap.add_argument("--source", help="replay:DIR or folder:DIR")
    ap.add_argument("--out"); ap.add_argument("--run-id"); ap.add_argument("--labels", help="expected states (trial log)")
    ap.add_argument("--gate", choices=("events", "each"), default="events")
    ap.add_argument("--mode", choices=MODES, default=LIVE_FILL); ap.add_argument("--track", action="store_true")
    ap.add_argument("--poll", type=float, default=0.25); ap.add_argument("--idle-exit", type=float)
    ap.add_argument("--every", type=float, default=1.0, help="stream: seconds between frames passed on")
    ap.add_argument("--note", default="")
    ap.add_argument("--export-replay", metavar="RUN"); ap.add_argument("--from", dest="from_", choices=("video", "stills"), default="video")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--inspector", choices=("rule", "resnet", "staging5"), default="rule")
    ap.add_argument("--threshold", choices=("d038", "preregistered"), default="d038", help="ResNet threshold (D038)")
    ap.add_argument("--model-dir", help="model run folder (required for staging5; no fallback)")
    ap.add_argument("--device", help="ResNet device (default mps if available, else cpu)")
    ap.add_argument("--fixture", choices=("G", "S"), default="G", help="fixture to inspect (only G is calibrated)")
    ap.add_argument("--preflight", metavar="DIR", help="package self-check on the bundled frames in DIR")
    a = ap.parse_args()
    if a.fixture != "G":
        ap.error("fixture S is refused: no calibrated fixture-S position map is available")
    if a.inspector == "staging5" and not a.model_dir:
        ap.error("--inspector staging5 requires an explicit --model-dir; no model fallback is allowed")
    if a.preflight:
        sys.exit(0 if preflight(a.preflight, a.out or os.path.join(P.HERE, "results", "preflight")) else 1)
    if a.export_replay:
        if not a.out:
            ap.error("--export-replay needs --out")
        export_replay(a.export_replay, a.out, a.from_, a.step)
    else:
        kind, _, folder = (a.source or "").partition(":")
        if not (kind in ("replay", "folder") and os.path.isdir(folder) or kind == "stream" and folder):
            ap.error("--source must be replay:DIR or folder:DIR with an existing DIR, or stream:URL|DEVICE|FILE")
        ix = dataset_index()
        stats = {}
        if kind == "stream":
            why = stream_refusal(ix, folder)
            if why:
                raise R.ProtectedDataError(why)
            frames = stream(folder, a.every, a.idle_exit, stats=stats)
        else:
            frames = replay(folder, ix) if kind == "replay" else watch(folder, a.poll, a.idle_exit, stats=stats, ix=ix)
        out = a.out or os.path.join(P.HERE, "results", "live", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        insp = make_inspector(a.inspector, a.threshold, a.model_dir, a.device)
        s = Session(out, a.source, a.run_id, a.gate, a.mode, a.track, a.labels, a.note, ix=ix, inspector=insp,
                    fixture_id=a.fixture)
        if kind == "folder":
            s.write_info(**stats)
            print(f"watching {folder}: {stats.get('ignored_existing_images', 'unknown')} image(s) already there are ignored; new images only")
        elif kind == "stream":
            print(f"stream {folder}: " + (f"video file, one frame per {a.every:g} s of its own timestamps" if os.path.isfile(folder)
                  else f"live source, newest frame every {a.every:g} s; drops are logged and reconnected (Ctrl-C to stop)"))
        print(f"operator view: this console (one grid per event), {os.path.join(out, 'overlay_latest.png')}, "
              f"or {os.path.join(os.path.abspath(out), VIEW_HTML)} in a browser")
        run(s, frames)
        if kind == "stream":
            s.write_info(stream={"source": folder, "every_s": a.every, **stats})
        elif kind == "folder":
            s.write_info(**stats)
