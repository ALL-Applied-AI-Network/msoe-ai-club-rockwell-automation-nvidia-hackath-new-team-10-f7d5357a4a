import os, sys, unittest
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rules
from positions import PositionCrop

T = {"t_absent": 0.08, "t_seated": 0.22, "t_offset": 4.4}


def crop(cap_at=None):
    im = np.full((40, 36, 3), 90, np.uint8)
    if cap_at is not None:
        cv2.circle(im, cap_at, 10, (0, 220, 255), -1)            # yellow cap, BGR
    return PositionCrop("G05", 5, "f.png", (0, 0, 36, 40), False, (18.0, 18.0), (16.0, 22.0), (18.0, 18.0),
                        np.zeros((0, 2)), im)


class Rules(unittest.TestCase):
    def test_seated_cap_passes(self):
        f = rules.features(crop((18, 18)))
        self.assertGreater(f["cap_frac"], 0.8); self.assertLess(f["cap_offset"], 0.5)
        self.assertEqual(rules.decide(f, T)[0], "pass")

    def test_no_yellow_is_no_cap(self):
        f = rules.features(crop())
        self.assertEqual((f["cap_frac"], f["cap_offset"]), (0.0, None))
        self.assertEqual(rules.decide(f, T)[0], "no_vial")

    def test_displaced_cap_is_unsure_not_a_crooked_claim(self):
        f = rules.features(crop((27, 24)))
        self.assertGreater(f["cap_offset"], T["t_offset"])
        self.assertEqual(rules.decide(f, T)[0], "unknown")

    def test_fit_catches_every_labeled_defect(self):
        rows = [dict(ground_truth="pass", cap_frac=0.9, cap_offset=1.0), dict(ground_truth="pass", cap_frac=0.12, cap_offset=5.0),
                dict(ground_truth="crooked_cap", cap_frac=0.55, cap_offset=4.6), dict(ground_truth="crooked_cap", cap_frac=0.3, cap_offset=7.5),
                dict(ground_truth="no_vial", cap_frac=0.0, cap_offset=None)]
        t = rules.fit(rows)
        self.assertAlmostEqual(t["t_absent"], 0.096)            # 0.8 x the faintest labeled cap
        self.assertFalse(any(rules.decide(r, t)[0] == "pass" for r in rows if r["ground_truth"] == "crooked_cap"))
        self.assertEqual(rules.decide(rows[0], t)[0], "pass")


if __name__ == "__main__":
    unittest.main()
