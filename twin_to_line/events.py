"""Placement events from a DEV frame sequence, without the cap classifier (so events cannot be biased by it).

A sample is (t, image). Two pixel signals only:
  arm      yellow connected component > ARM_MIN_AREA px touching the (dilated) fixture plate: the FANUC arm is yellow
  change   per position, mean over its anchor box of max-channel |difference| to the previous sample
A STABLE segment = consecutive arm-free samples with no position changing by >= CHANGE_T, lasting >= STABLE_S.
An EVENT = the first frame of a stable segment whose appearance differs (some position >= CHANGE_T) from the
previous event's frame; the first stable segment of a run is event 1. Transient occlusions (arm, tooling,
hands, shadows) that do not persist STABLE_S are ignored; a persistent occlusion becomes an event and has to be
caught at labeling time (it is labeled as not usable rather than hidden).

Thresholds from DEV static footage (runs 2-3): per-box mean |d| p99.9 = 6.8-9.7 (stills), p99 = 8.0 (video, 1 fps).
    python events.py --run 1 [--source video|stills] [--step 1.0] [--out DIR]
"""
import argparse, csv, os
from dataclasses import dataclass
import cv2, numpy as np
import positions as P
import real_data as R

ARM_MIN_AREA = 2000
CHANGE_T = 15.0
STABLE_S = 4.0


def arm_pixels(img, pmap, plate=None):
    plate = pmap.plate_mask() if plate is None else plate
    y = (P.yellow_mask(img) > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(y)
    big = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > ARM_MIN_AREA]
    return int(np.isin(lab, big)[plate > 0].sum()) if big else 0


def box_change(a, b, boxes):
    return np.array([np.abs(a[y0:y1, x0:x1].astype(np.int16) - b[y0:y1, x0:x1].astype(np.int16)).max(2).mean()
                     for x0, y0, x1, y1 in boxes])


@dataclass
class Event:
    event: int
    t: float                 # first frame of the stable segment
    t_end: float             # last frame of that segment
    n_samples: int
    changed: list            # positions that differ from the previous event (all 'new' for event 1)
    image: np.ndarray = None


def segment_events(samples, boxes, ids, arm_flags, change_t=CHANGE_T, stable_s=STABLE_S):
    """samples: list of (t, img) in time order; arm_flags: bool per sample. Pure function (testable)."""
    segs, cur = [], []
    for k, ((t, img), arm) in enumerate(zip(samples, arm_flags)):
        if arm:
            if cur: segs.append(cur)
            cur = []; continue
        if cur and box_change(img, samples[cur[-1]][1], boxes).max() >= change_t:
            segs.append(cur); cur = []
        cur.append(k)
    if cur: segs.append(cur)
    events, last = [], None
    for s in segs:
        t0, t1 = samples[s[0]][0], samples[s[-1]][0]
        if t1 - t0 < stable_s:
            continue
        img = samples[s[0]][1]
        if last is None:
            changed = list(ids)
        else:
            d = box_change(img, last, boxes); changed = [p for p, x in zip(ids, d) if x >= change_t]
            if not changed:
                continue
        events.append(Event(len(events) + 1, t0, t1, len(s), changed, img)); last = img
    return events


def run_samples(ix, run, source="video", step=1.0):
    if source == "video":
        return list(R.iter_video(ix.video_path(run), step))
    fr = ix.frames(run); t0 = ix.time_of[fr[0]]
    return [((ix.time_of[f] - t0).total_seconds(), ix.load(f)) for f in fr]


def source_id(run, source, t, ix=None, frames=None):
    if source == "video":
        return R.video_id(run, t)
    return frames[[round((ix.time_of[f] - ix.time_of[frames[0]]).total_seconds(), 3) for f in frames].index(round(t, 3))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="classifier-free placement events for one DEV run")
    ap.add_argument("--run", required=True); ap.add_argument("--source", default="video", choices=("video", "stills"))
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(P.HERE, "results", "prototype", "events"))
    a = ap.parse_args()
    ix = R.Index(); pm = P.PositionMap(); plate = pm.plate_mask()
    samples = run_samples(ix, a.run, a.source, a.step)
    arm = [arm_pixels(img, pm, plate) > 0 for _, img in samples]
    evs = segment_events(samples, pm.anchor_boxes, pm.ids, arm)
    os.makedirs(a.out, exist_ok=True)
    frames = ix.frames(a.run) if a.source == "stills" else None
    path = os.path.join(a.out, f"events_run{a.run}_{a.source}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "event", "frame", "t", "t_end", "n_samples", "changed"])
        for e in evs:
            sid = source_id(a.run, a.source, e.t, ix, frames)
            if a.source == "video":
                cv2.imwrite(os.path.join(a.out, sid + ".png"), e.image)
            w.writerow([a.run, e.event, sid, f"{e.t:.1f}", f"{e.t_end:.1f}", e.n_samples, " ".join(e.changed) if len(e.changed) < 30 else "ALL"])
            print(f"run {a.run} event {e.event:2d}  t={e.t:7.1f}-{e.t_end:7.1f}s  n={e.n_samples:4d}  changed: {' '.join(e.changed) if len(e.changed) < 30 else 'ALL (first)'}")
    print(f"{len(samples)} samples, {sum(arm)} with the arm on the plate, {len(evs)} events -> {path}")
