import json, os, sys, tempfile, unittest
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import positions as P
import fit_map_1080 as F
from rules import RuleInspector

M640 = json.load(open(P.MAP_PATH))
POS = sorted(M640["positions"], key=lambda p: p["index"])
A = np.array([p["plate_hole_center_px"] for p in POS])
S_TRUE, T_TRUE = 2.237, np.array([241.6, -5.1])            # about the CSI still's fit


def write_map(m):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(m, f); f.close()
    return f.name


def synthetic_still(s=S_TRUE, t=T_TRUE, r=19):
    """Grey frame, bright plate where the twin plate lands, 30 dark discs at the twin plate-hole centres (sub-pixel)."""
    img = np.full((1080, 1920, 3), 60, np.uint8)
    cv2.fillConvexPoly(img, np.round(s * np.array(M640["plate_polygon_px"]) + t).astype(np.int32), (200, 200, 200))
    for u, v in s * A + t:
        cv2.circle(img, (int(round(u * 16)), int(round(v * 16))), r * 16, (20, 20, 20), -1, cv2.LINE_AA, shift=4)
    return img


class Fit(unittest.TestCase):
    def test_nominal_guess_is_the_centre_4x3_crop(self):
        s, t = F.nominal_guess((640, 480), (1920, 1080))
        self.assertEqual(s, 2.25)
        np.testing.assert_allclose(t, (240, 0))

    def test_scale_offset_recovered(self):
        s, t = F.fit_scale_offset(A, S_TRUE * A + T_TRUE)
        self.assertAlmostEqual(s, S_TRUE, places=9)
        np.testing.assert_allclose(t, T_TRUE, atol=1e-7)
        noise = np.random.default_rng(0).normal(0, 1.5, A.shape)
        s, t = F.fit_scale_offset(A, S_TRUE * A + T_TRUE + noise)
        self.assertAlmostEqual(s, S_TRUE, delta=0.01)
        np.testing.assert_allclose(t, T_TRUE, atol=2.0)

    def test_holes_found_and_fit_from_synthetic_still(self):
        r = F.fit_still(synthetic_still(), M640)
        self.assertEqual(len(r["holes"]), 30)
        self.assertAlmostEqual(r["scale"], S_TRUE, delta=0.002)
        np.testing.assert_allclose(r["offset"], T_TRUE, atol=0.3)
        self.assertLess(r["err"].max(), 0.3)

    def test_no_map_unless_30_holes(self):
        with self.assertRaises(ValueError):
            F.fit_still(np.full((1080, 1920, 3), 200, np.uint8), M640)


class ScaleMap(unittest.TestCase):
    def setUp(self):
        self.before = json.dumps(M640)
        self.m, self.problems = F.scale_map(M640, S_TRUE, T_TRUE, F.SIZE, {"generated_by": "test"})

    def test_same_fields_and_size(self):
        self.assertEqual(set(self.m), set(M640))
        for p, q in zip(self.m["positions"], M640["positions"]):
            self.assertEqual(set(p), set(q))
        self.assertEqual(self.m["image_size"], [1920, 1080])
        self.assertEqual(self.problems, [])
        self.assertEqual(json.dumps(M640), self.before)                    # input not mutated

    def test_points_scaled_and_boxes_contain_scaled_640_boxes(self):
        np.testing.assert_allclose(self.m["plate_polygon_px"], S_TRUE * np.array(M640["plate_polygon_px"]) + T_TRUE, atol=0.01)
        for p, q in zip(self.m["positions"], M640["positions"]):
            for k in F.POINT_FIELDS:
                np.testing.assert_allclose(p[k], S_TRUE * np.array(q[k]) + T_TRUE, atol=0.01)
            x0, y0, x1, y1 = p["anchor_bbox_px"]
            (u0, v0), (u1, v1) = S_TRUE * np.array(q["anchor_bbox_px"]).reshape(2, 2) + T_TRUE
            self.assertTrue(x0 <= u0 < x0 + 1 and y0 <= v0 < y0 + 1 and x1 - 1 < u1 <= x1 and y1 - 1 < v1 <= y1)


class ResizedCrops(unittest.TestCase):
    def setUp(self):
        m, _ = F.scale_map(M640, S_TRUE, T_TRUE)
        self.pm, self.pm640 = P.PositionMap(write_map(m), offset=(0, 0)), P.PositionMap(offset=(0, 0))

    def test_crop_sizes_and_yellow_cap_found_at_its_seat(self):
        img = np.full((1080, 1920, 3), 120, np.uint8)
        u, v = self.pm.cap_centers[12]                                        # G13
        cv2.circle(img, (int(round(u * 16)), int(round(v * 16))), int(14 * S_TRUE * 16), (0, 220, 255), -1, cv2.LINE_AA, shift=4)
        crops, small, obs, states = F.colour_check(img, self.pm, self.pm640, S_TRUE)
        self.assertEqual([c.image.shape[1::-1] for c in small], F.crop_sizes(self.pm640))
        self.assertFalse(any(c.clipped for c in crops))
        f = obs[12].features
        self.assertGreater(f["cap_frac"], 0.9)
        self.assertLess(f["cap_offset"], 0.5)
        self.assertEqual(obs[0].visual_state, "no_vial")

    def test_empty_frame_reads_all_waiting(self):
        _, _, obs, states = F.colour_check(np.full((1080, 1920, 3), 120, np.uint8), self.pm, self.pm640, S_TRUE)
        self.assertEqual({o.visual_state for o in obs}, {"no_vial"})
        self.assertEqual({s for s, _ in states}, {"WAITING"})


class CommittedMap(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(F.OUT_MAP):
            self.skipTest("position_map_G_1080.json not written yet")
        self.m = json.load(open(F.OUT_MAP))

    def test_equals_position_map_G_scaled_by_its_own_fit(self):
        fit = self.m["meta"]["fit"]
        self.assertEqual(fit["model"], "scale+offset")
        self.assertEqual(self.m["meta"]["source_map_sha256"], F.sha256(P.MAP_PATH))
        m, _ = F.scale_map(M640, fit["scale"], np.array(fit["offset_px"]))
        for k in ("image_size", "plate_polygon_px", "positions"):
            self.assertEqual(self.m[k], m[k])

    def test_thirty_unclipped_crops_without_dev_offset(self):
        pm = P.PositionMap(F.OUT_MAP, offset=(0, 0))
        crops = P.extract_crops(np.zeros((1080, 1920, 3), np.uint8), pm, pad=int(round(P.CROP_PAD_PX * self.m["meta"]["fit"]["scale"])))
        self.assertEqual([c.position for c in crops], [f"G{i:02d}" for i in range(1, 31)])
        self.assertFalse(any(c.clipped for c in crops))


@unittest.skipUnless(os.path.exists(F.STILL), "CSI 1080 still not on this machine (never committed)")
class CsiStill(unittest.TestCase):
    def test_thirty_holes_and_all_empty(self):
        img = cv2.imread(F.STILL)
        r = F.fit_still(img, M640)
        self.assertEqual(len(r["holes"]), 30)
        self.assertLess(float(np.median(r["err"])), 3.0)
        m, problems = F.scale_map(M640, r["scale"], r["offset"])
        self.assertEqual(problems, [])
        _, _, obs, states = F.colour_check(img, P.PositionMap(write_map(m), offset=(0, 0)), P.PositionMap(), r["scale"], RuleInspector())
        self.assertEqual({o.visual_state for o in obs}, {"no_vial"})
        self.assertEqual({s for s, _ in states}, {"WAITING"})


if __name__ == "__main__":
    unittest.main()
