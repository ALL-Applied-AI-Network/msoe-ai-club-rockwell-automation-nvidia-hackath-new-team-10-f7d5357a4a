import json, os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import check_organizer_grid as V

# organizer calibration view_001, good tray (public numbers from the labeler's calibration.json / demo labels.json)
GRID = [[750, 398], [1078, 406], [744, 710], [1067, 715]]
TWIN = V.twin_points(json.load(open(V.MAP_PATH)), V.PRIMARY)


def transform(A, s, deg, t):
    r = np.radians(deg)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return V.apply(s, R, np.asarray(t, float), A)


class OfficialGrid(unittest.TestCase):
    def test_bilinear_matches_labeler_demo(self):
        demo = {1: (1078.0, 406.0), 5: (1067.0, 715.0), 13: (942.3, 557.9), 18: (877.2, 556.6),
                26: (750.0, 398.0), 30: (744.0, 710.0)}
        for pos, xy in demo.items():
            np.testing.assert_allclose(np.round(V.official_center(GRID, pos), 1), xy)

    def test_known_similarity_recovered(self):
        B = transform(TWIN, 2.2, 3.6, (500, 100))
        s, R, t, mirror = V.fit_similarity(np.asarray(TWIN)[[0, 4, 25, 29]], B[[0, 4, 25, 29]])
        self.assertAlmostEqual(s, 2.2, places=6)
        self.assertAlmostEqual(np.degrees(np.arctan2(R[1, 0], R[0, 0])), 3.6, places=6)
        np.testing.assert_allclose(V.apply(s, R, t, TWIN), B, atol=1e-6)
        self.assertFalse(mirror)

    def test_real_grid_convention_and_size(self):
        r = V.validate(TWIN, GRID)
        self.assertEqual((r["positions_matched"], r["positions_expected"]), (30, 30))
        self.assertTrue(r["numbering"]["convention_matches"])

    def test_180_degree_numbering_detected(self):
        r = V.validate(TWIN[::-1], GRID)              # G01 <-> G30 ...
        self.assertFalse(r["numbering"]["convention_matches"])

    def test_mirrored_numbering_detected(self):
        cols = [TWIN[c * 5:(c + 1) * 5] for c in range(6)]
        mirrored = [p for col in cols[::-1] for p in col]          # G01 column swapped with G26 column
        r = V.validate(mirrored, GRID)
        self.assertTrue(r["numbering"]["mirror_needed"])
        self.assertFalse(r["numbering"]["convention_matches"])

    def test_input_not_mutated_and_wrong_size(self):
        before = [list(p) for p in TWIN]
        V.validate(TWIN, GRID)
        self.assertEqual(before, [list(p) for p in TWIN])
        with self.assertRaises(ValueError):
            V.validate(TWIN[:29], GRID)


if __name__ == "__main__":
    unittest.main()
