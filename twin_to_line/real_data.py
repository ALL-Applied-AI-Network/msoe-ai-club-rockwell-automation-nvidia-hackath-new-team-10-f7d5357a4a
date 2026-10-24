"""Real cap_seal data access with the D011 split enforced.

    image_index.csv: filename, timestamp, video_id   (video_id = run 1..5 or 'unassigned'; stills ~2 s apart)
    videos/cap_seal_video_{run}.mp4                  (30 fps, 640x480)

Run roles come from real_splits.json. Everything here refuses MODEL_HOLDOUT (runs 4-5) and IGNORE
(unassigned) unless allow_holdout=True. No development command passes that flag: a future, explicitly
approved evaluation path is the only intended caller.

A frame source is either a still file name, a path to a still, or a video frame id 'cap_seal_video_1.mp4@t0612.0'.
Images outside the dataset (future staged / live captures) are allowed with a warning.
"""
import csv, json, os, re, sys, warnings
from datetime import datetime
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_ROOT = os.environ.get("IGNITE_REAL", os.path.expanduser("~/Downloads/csi-vial-fill-dataset/cap_seal"))
SPLITS = os.path.join(HERE, "real_splits.json")
VIDEO_RE = re.compile(r"cap_seal_video_(\w+)\.mp4@t(\d+(?:\.\d+)?)(?:\.png)?$")   # .png = a cached copy of that frame


class ProtectedDataError(PermissionError):
    pass


def roles(path=SPLITS):
    return {run: r["role"] for run, r in json.load(open(path))["runs"].items()}


def check_allowed(run, allow_holdout=False, role_map=None):
    role = (role_map or roles()).get(str(run))
    if role == "DEV":
        return run
    if allow_holdout and role == "MODEL_HOLDOUT":
        return run
    raise ProtectedDataError(f"run {run} is {role or 'not in real_splits.json'}: development code may only use DEV runs 1-3 (D011)")


class Index:
    def __init__(self, root=REAL_ROOT, role_map=None):
        self.root = root; self.role_map = role_map or roles()
        rows = list(csv.DictReader(open(os.path.join(root, "image_index.csv"), newline="")))
        self.run_of = {r["filename"]: r["video_id"] for r in rows}
        self.time_of = {r["filename"]: datetime.fromisoformat(r["timestamp"]) for r in rows}

    def frames(self, run, allow_holdout=False):
        """Stills of one run, time ordered."""
        check_allowed(run, allow_holdout, self.role_map)
        return sorted((f for f, r in self.run_of.items() if r == str(run)), key=self.time_of.get)

    def video_path(self, run, allow_holdout=False):
        check_allowed(run, allow_holdout, self.role_map)
        return os.path.join(self.root, "videos", f"cap_seal_video_{run}.mp4")

    def run_of_source(self, source):
        """Run id for a source, or None for an image from outside the dataset."""
        m = VIDEO_RE.match(os.path.basename(source))
        if m:
            return m.group(1)
        name = os.path.basename(source)
        if name in self.run_of:
            return self.run_of[name]
        if os.path.abspath(source).startswith(os.path.abspath(self.root)):
            raise ProtectedDataError(f"{source}: inside the dataset but not in image_index.csv")
        return None

    def load(self, source, allow_holdout=False):
        """BGR image for a still name/path or a video frame id; enforces the split first."""
        run = self.run_of_source(source)
        if run is None:
            warnings.warn(f"{source}: not in image_index.csv; treated as an external (staged/live) frame")
            img = cv2.imread(source)
        else:
            check_allowed(run, allow_holdout, self.role_map)
            m = VIDEO_RE.match(os.path.basename(source))
            if m and os.path.exists(source):
                img = cv2.imread(source)
            elif m:
                img = read_video_frame(self.video_path(run, allow_holdout), float(m.group(2)))
            else:
                path = source if os.path.exists(source) else os.path.join(self.root, "images", source)
                img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(source)
        return img


def video_id(run, t):
    return f"cap_seal_video_{run}.mp4@t{t:06.1f}"


def read_video_frame(path, t, fps=None):
    cap = cv2.VideoCapture(path)
    fps = fps or cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, img = cap.read()
    return img if ok else None


def iter_video(path, step_s=1.0):
    """(t, frame) every step_s seconds, decoding sequentially (faster and exact vs seeking)."""
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS); step = max(1, int(round(step_s * fps))); k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            return
        if k % step == 0:
            yield k / fps, img
        k += 1


if __name__ == "__main__":
    ix = Index()
    for run, role in sorted(ix.role_map.items()):
        n = sum(1 for r in ix.run_of.values() if r == run)
        print(f"run {run}: {role}, {n} stills")
    if len(sys.argv) > 1:
        for s in sys.argv[1:]:
            print(s, "->", ix.run_of_source(s), ix.load(s).shape)
