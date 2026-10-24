import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from events import segment_events

BOXES = [(10, 10, 30, 30), (40, 10, 60, 30), (70, 10, 90, 30)]
IDS = ["G01", "G02", "G03"]


def frame(changed=(), value=200):
    im = np.full((40, 100, 3), 100, np.uint8)
    for i in changed:
        x0, y0, x1, y1 = BOXES[i]; im[y0:y1, x0:x1] = value
    return im


class Events(unittest.TestCase):
    def test_static_arm_placement_transient(self):
        seq = [frame()] * 6 + [frame()] * 2 + [frame([1])] * 6 + [frame([1, 2])] + [frame([1])] * 6
        arm = [False] * 6 + [True] * 2 + [False] * 13
        samples = list(zip(range(len(seq)), seq))
        ev = segment_events(samples, BOXES, IDS, arm, change_t=15, stable_s=4)
        self.assertEqual([(e.event, e.t, e.changed) for e in ev], [(1, 0, IDS), (2, 8, ["G02"])])

    def test_near_duplicate_frames_are_one_event(self):
        rng = np.random.default_rng(0)
        seq = [np.clip(frame().astype(int) + rng.integers(-5, 6, (40, 100, 3)), 0, 255).astype(np.uint8) for _ in range(78)]
        ev = segment_events(list(zip(range(78), seq)), BOXES, IDS, [False] * 78, change_t=15, stable_s=4)
        self.assertEqual(len(ev), 1)                             # run 2: 78 stills, one fixture state

    def test_short_segment_is_not_an_event(self):
        seq = [frame()] * 5 + [frame([0])] * 2 + [frame()] * 5    # a 2 s blip (e.g. a hand) then back
        ev = segment_events(list(zip(range(12), seq)), BOXES, IDS, [False] * 12, change_t=15, stable_s=4)
        self.assertEqual(len(ev), 1)


if __name__ == "__main__":
    unittest.main()
