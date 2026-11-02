import os, sys, unittest
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import events as E
import positions as P

YELLOW = (0, 220, 230)          # BGR, inside positions.YELLOW_LO..YELLOW_HI


def blank():
    return np.full((480, 640, 3), 100, np.uint8)


def yellow_components(img):
    n, lab, st, _ = cv2.connectedComponentsWithStats((P.yellow_mask(img) > 0).astype(np.uint8))
    return lab, [st[k, cv2.CC_STAT_AREA] for k in range(1, n)]


class ArmCheckV11(unittest.TestCase):
    """D045: only arm-sized, not-thin yellow blobs touching the fixture plate count as the arm."""

    @classmethod
    def setUpClass(cls):
        cls.pm = P.PositionMap(); cls.plate = cls.pm.plate_mask()

    def touches_plate(self, img):
        return int(((P.yellow_mask(img) > 0) & (self.plate > 0)).sum())

    def test_thin_strip_beside_the_fixture_is_not_the_arm(self):
        img = blank(); img[30:450, 200:207] = YELLOW                    # D044: ~7 px wide, ~420 px tall, grazes the plate
        _, areas = yellow_components(img)
        self.assertGreater(max(areas), 2000); self.assertGreater(self.touches_plate(img), 0)   # tripped the frozen check
        self.assertEqual(E.arm_pixels(img, self.pm, self.plate), 0)

    def test_long_thin_band_over_the_fixture_is_not_the_arm(self):
        img = blank(); cv2.line(img, (0, 0), (639, 479), YELLOW, 12)    # > ARM_MIN_AREA but only ~12 px wide
        _, areas = yellow_components(img)
        self.assertGreater(max(areas), E.ARM_MIN_AREA); self.assertGreater(self.touches_plate(img), 0)
        self.assertEqual(E.arm_pixels(img, self.pm, self.plate), 0)

    def test_merged_neighbouring_caps_are_not_the_arm(self):
        img = blank(); idx = [0, 1, 2, 5, 6, 7]                         # G01-G03 and G06-G08, bridged into one blob
        for i in idx:
            cv2.fillPoly(img, [np.round(self.pm.cap_hulls[i]).astype(np.int32)], YELLOW)
        c = np.round(self.pm.cap_centers[idx]).astype(int)
        for a, b in [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]:
            cv2.line(img, tuple(map(int, c[a])), tuple(map(int, c[b])), YELLOW, 12)
        lab, areas = yellow_components(img)
        self.assertEqual(len(areas), 1); self.assertGreater(areas[0], 2000)
        self.assertTrue(all(lab[y, x] == 1 for x, y in c))
        self.assertEqual(E.arm_pixels(img, self.pm, self.plate), 0)

    def test_arm_sized_blob_over_the_fixture_is_the_arm(self):
        img = blank(); img[170:300, 250:370] = YELLOW
        self.assertGreater(E.arm_pixels(img, self.pm, self.plate), 1000)

    def test_arm_grazing_the_plate_edge_is_still_the_arm(self):
        img = blank(); img[150:300, 60:207] = YELLOW                    # DEV: arm frames touching the plate by 20-62 px
        arm = E.arm_pixels(img, self.pm, self.plate)
        self.assertGreater(arm, 0); self.assertEqual(arm, self.touches_plate(img))

    def test_mean_width(self):
        m = np.zeros((480, 640), bool); m[100:400, 300:310] = True
        self.assertTrue(9.5 < E.mean_width(m) < 10.5)                   # a 10 x 300 px bar is ~10 px wide


if __name__ == "__main__":
    unittest.main()
