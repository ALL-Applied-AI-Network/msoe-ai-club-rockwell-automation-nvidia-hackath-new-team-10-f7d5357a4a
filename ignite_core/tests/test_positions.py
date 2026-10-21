import json, os, sys, tempfile, unittest
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import positions as P


def fake_map(edge=False):
    """6 x 5 grid laid out like the real fixture: G01-G05 rightmost column top->bottom."""
    pos = []
    for i in range(1, 31):
        r, c = (i - 1) % 5, (i - 1) // 5
        u, v = 500 - 40 * c, 100 + 40 * r
        if edge and i == 30:
            u, v = 8, 470
        hull = [[u + 10 * np.cos(a), v + 10 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        pos.append({"position": f"G{i:02d}", "index": i, "cap_center_px": [u, v], "cap_hull_px": hull,
                    "hole_center_px": [u, v + 8], "tube_top_center_px": [u, v + 4],
                    "anchor_bbox_px": [int(u - 12), int(v - 12), int(u + 12), int(v + 20)]})
    m = {"meta": {}, "image_size": [640, 480], "plate_polygon_px": [[250, 70], [530, 70], [530, 290], [250, 290]],
         "positions": pos}
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(m, f); f.close()
    return f.name, m


class Positions(unittest.TestCase):
    def setUp(self):
        self.path, self.raw = fake_map()
        self.pm = P.PositionMap(self.path, offset=(0, 0))
        self.img = np.full((480, 640, 3), 90, np.uint8)

    def test_thirty_crops_with_provenance(self):
        crops = P.extract_crops(self.img, self.pm, frame="f.jpg")
        self.assertEqual([c.position for c in crops], [f"G{i:02d}" for i in range(1, 31)])
        c = crops[0]
        self.assertEqual((c.index, c.frame, c.clipped), (1, "f.jpg", False))
        self.assertEqual(c.box, (500 - 12 - 6, 100 - 12 - 6, 500 + 12 + 6, 100 + 20 + 6))
        self.assertEqual(c.image.shape, (32 + 12, 24 + 12, 3))
        np.testing.assert_allclose(c.cap_center, (18, 18))
        np.testing.assert_allclose(c.hole_center, (18, 26))

    def test_offset_moves_boxes_and_centers(self):
        pm = P.PositionMap(self.path, offset=(2, -3))
        c = P.extract_crops(self.img, pm)[0]
        self.assertEqual(c.box, (500 - 18 + 2, 100 - 18 - 3, 500 + 18 + 2, 100 + 26 - 3))
        np.testing.assert_allclose(pm.cap_centers[0], (502, 97))

    def test_pad_is_configurable(self):
        c = P.extract_crops(self.img, self.pm, pad=0)[0]
        self.assertEqual(c.box, (488, 88, 512, 120))

    def test_clipped_at_border(self):
        path, _ = fake_map(edge=True)
        crops = P.extract_crops(self.img, P.PositionMap(path, offset=(0, 0)))
        self.assertTrue(crops[29].clipped and not crops[0].clipped)
        self.assertEqual(crops[29].box[3], 480)

    def test_wrong_image_size(self):
        with self.assertRaises(AssertionError):
            P.extract_crops(np.zeros((576, 768, 3), np.uint8), self.pm)

    def test_model_crop_is_square(self):
        self.assertEqual(P.crop_for_model(P.extract_crops(self.img, self.pm)[3]).shape, (P.MODEL_SIZE, P.MODEL_SIZE, 3))

    def test_fixture_layout(self):
        self.assertEqual([P.fixture_cell(i) for i in (1, 5, 6, 30)], [(0, 5), (4, 5), (0, 4), (4, 0)])
        self.assertEqual(P.layout_problems(self.raw), [])
        bad = json.loads(json.dumps(self.raw)); bad["positions"][0]["cap_center_px"][1] = 999
        self.assertTrue(P.layout_problems(bad))

    def test_contact_sheet_shape(self):
        crops = P.extract_crops(self.img, self.pm)
        sheet = P.contact_sheet(crops, scale=2)
        self.assertEqual(sheet.shape, (5 * (44 * 2 + 16), 6 * 36 * 2, 3))

    def test_registration_recovers_a_known_shift(self):
        img = self.img.copy()
        for k, (u, v) in enumerate(self.pm.cap_centers):
            if k % 3 == 0:                                   # some positions empty
                continue
            cv2.circle(img, (int(u) + 2, int(v) - 3), 9, (0, 220, 255), -1)   # yellow cap, shifted (+2, -3)
        res = P.residuals(img, self.pm)
        self.assertEqual(len(res), 20)
        np.testing.assert_allclose(np.median(np.array(list(res.values())), 0), (2, -3), atol=0.5)


if __name__ == "__main__":
    unittest.main()
