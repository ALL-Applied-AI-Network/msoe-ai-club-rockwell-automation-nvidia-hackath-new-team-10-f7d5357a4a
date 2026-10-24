import csv, hashlib, json, os, sys, tempfile, unittest
from datetime import datetime, timedelta
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import positions as P
import real_data as R
import frame_warp as FW
import live_inspect as L
import register_1080 as RG
import replay_1080 as RP
from fit_map_1080 import STILL
from register_dev import hole_check

FWARP = FW.FrameWarp()
PM = P.PositionMap()                                     # DEV offset, as at runtime
YELLOW = (0, 220, 230)
T0 = datetime(2026, 10, 26, 13, 0, 0)
DEV = os.path.exists(os.path.join(R.REAL_ROOT, "image_index.csv"))
EVENT_PNG = os.path.join(P.HERE, "results", "prototype", "events", "cap_seal_video_1.mp4@t1050.0.png")


def fixture(filled=()):
    img = np.full((480, 640, 3), 90, np.uint8)
    for i in filled:
        x, y = PM.cap_centers[i - 1]
        cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), 160, YELLOW, -1, cv2.LINE_AA, shift=4)
    return img


def blob_centroid(img, p, r=10):
    x0, y0 = int(round(p[0])), int(round(p[1])); w = img[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1].astype(float)
    Y, X = np.mgrid[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
    return (w * X).sum() / w.sum(), (w * Y).sum() / w.sum()


def name(k, dt=2.0):
    return f"syn_{(T0 + timedelta(seconds=k * dt)).strftime('%Y%m%d_%H%M%S_%f')}.png"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def session(**kw):
    out = tempfile.mkdtemp()
    return L.Session(out, quiet=True, ix=None, **kw), out


class TransformFile(unittest.TestCase):
    def test_committed_transform(self):
        d = json.load(open(FW.WARP_PATH))
        self.assertEqual((FWARP.size_in, FWARP.size_out), ((1920, 1080), (640, 480)))
        self.assertEqual(FWARP.sha256, hashlib.sha256(open(FW.WARP_PATH, "rb").read()).hexdigest())
        f = d["meta"]["forward_640_to_1080"]                                   # the readable (rounded) params = the matrix
        B = RG.apply(RG.invert(FWARP.M), PM.plate_hole_centers)
        M_meta = RG.invert(RG.similarity(f["scale"], f["rotation_deg"], f["offset_px"]))
        self.assertLess(np.abs(RG.apply(M_meta, B) - PM.plate_hole_centers).max(), 0.01)
        self.assertEqual(d["meta"]["reference"]["run"], "3")
        self.assertEqual(R.check_allowed(d["meta"]["reference"]["run"]), "3")  # registered on a DEV frame only
        self.assertTrue(d["meta"]["fit"]["converged"])

    def test_non_similarity_rejected_and_missing_file_is_none(self):
        d = json.load(open(FW.WARP_PATH)); d["matrix_in_to_out"][0][1] += 0.01
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(d, f); f.close()
        with self.assertRaises(ValueError):
            FW.FrameWarp(f.name)
        self.assertIsNone(FW.load(os.path.join(tempfile.mkdtemp(), "none.json")))

    def test_geometry_exact_and_round_trip(self):
        pts = [(150.3, 120.7), (320.0, 240.0), (300.6, 200.2), (500.25, 400.8)]
        F = np.linalg.inv(FW.h3(FWARP.M)); Y, X = np.mgrid[0:1080, 0:1920]; big = np.zeros((1080, 1920), np.float32)
        for x, y in pts:
            u, v, _ = F @ [x, y, 1]; big += 200 * np.exp(-((X - u) ** 2 + (Y - v) ** 2) / (2 * 6.7 ** 2))
        out = FWARP.apply(cv2.merge([np.clip(big, 0, 255).astype(np.uint8)] * 3))[:, :, 0]
        for p in pts:
            np.testing.assert_allclose(blob_centroid(out, p), p, atol=0.02)
        img = cv2.GaussianBlur(fixture(range(1, 31)), (0, 0), 2)
        back = FWARP.apply(RP.warp_up(img, FWARP))
        self.assertLess(np.abs(back[5:-5, 5:-5].astype(int) - img[5:-5, 5:-5]).mean(), 1.0)
        with self.assertRaises(ValueError):
            FWARP.apply(np.zeros((480, 640, 3), np.uint8))

    def test_register_recovers_a_known_similarity(self):
        rng = np.random.default_rng(1)
        T = cv2.GaussianBlur(rng.integers(0, 255, (480, 640)).astype(np.float32), (0, 0), 4)
        T = cv2.normalize(T, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        M_true = RG.invert(RG.similarity(2.2492, 0.3, (239.8, -5.4)))
        still = cv2.warpAffine(T, M_true, (1920, 1080), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        mask = np.zeros((480, 640), bool); mask[20:460, 20:620] = True
        M, info = RG.register(still, T, mask, RG.invert(RG.similarity(2.25, 0, (240, 0))))
        holes = np.array([PM.plate_hole_centers[i] for i in range(30)])
        e = np.linalg.norm(RG.apply(M, RG.apply(RG.invert(M_true), holes)) - holes, axis=1)
        self.assertLess(e.max(), 0.05)
        self.assertTrue(info["converged"])

    def test_masks_skip_the_run3_timestamp(self):
        m = RG.masks(PM)
        self.assertFalse(m["union"][440:, :280].any())
        self.assertTrue(all(m[k].any() for k in ("station", "G plate", "S plate")))


class SizeSwitch(unittest.TestCase):
    def test_paths_by_size_and_logged(self):
        s, out = session(gate="each")
        img = fixture([1, 2, 3])
        rows = [s.process_frame(name(0), img), s.process_frame(name(1), RP.warp_up(img, FWARP)),
                s.process_frame(name(2), np.zeros((700, 1000, 3), np.uint8)), s.process_frame(name(3), np.zeros((240, 320, 3), np.uint8))]
        s.close()
        self.assertEqual([r.get("path", "") for r in rows], ["640", "1080->640", "", ""])
        self.assertEqual([r["status"] for r in rows], ["scored", "scored", "wrong_size", "wrong_size"])
        self.assertIn("expected 640x480 or 1920x1080", rows[2]["note"])
        log = read_csv(os.path.join(out, "trial_log.csv"))
        self.assertEqual([r["path"] for r in log], ["640", "1080->640", "", ""])
        info = json.load(open(os.path.join(out, "session.json")))
        self.assertEqual(info["frame_warp"]["sha256"], FWARP.sha256)
        self.assertEqual(info["code_sha256"]["frame_warp_1080.json"], FWARP.sha256)

    def test_without_transform_1080_is_wrong_size(self):
        s, out = session(gate="each", warp=None)
        r = s.process_frame(name(0), RP.warp_up(fixture([1]), FWARP)); s.close()
        self.assertEqual((r["status"], r["note"]), ("wrong_size", "1920x1080, expected 640x480"))
        self.assertIsNone(json.load(open(os.path.join(out, "session.json")))["frame_warp"])

    def test_transform_must_output_the_map_grid(self):
        class Odd:
            size_in, size_out = (1920, 1080), (320, 240)
        with self.assertRaises(ValueError):
            session(warp=Odd())

    def test_same_dev_offset_and_states_on_both_paths(self):
        """The 1080 path uses position_map_G.json + the DEV offset, never an extra one: a DEV-grid frame and its 1080
        version give the same events and states, with the same offset logged."""
        res = {}
        for path in ("640", "1080"):
            s, out = session(gate="events")
            for k in range(12):                                                # fill G01-G03, one every 3 frames
                img = fixture(range(1, k // 3 + 1))
                s.process_frame(name(k), img if path == "640" else RP.warp_up(img, FWARP))
            s.close()
            res[path] = (read_csv(os.path.join(out, "predictions.csv")), json.load(open(os.path.join(out, "session.json"))))
        (p640, i640), (p1080, i1080) = res["640"], res["1080"]
        self.assertEqual(i640["position_map"]["offset_px"], i1080["position_map"]["offset_px"])
        self.assertEqual(i1080["position_map"]["offset_px"], list(json.load(open(P.REG_PATH))["offset_dx_dy_px"]))
        self.assertEqual([(r["frame"], r["position"], r["final_state"]) for r in p640],
                         [(r["frame"], r["position"], r["final_state"]) for r in p1080])
        self.assertIn("PASS", {r["final_state"] for r in p1080})
        for a, b in zip(p640, p1080):
            self.assertAlmostEqual(float(a["cap_frac"]), float(b["cap_frac"]), delta=0.05)
            if a["final_state"] == "PASS":                                     # a stray ~1.6 px offset would show here
                self.assertAlmostEqual(float(a["cap_offset"]), float(b["cap_offset"]), delta=0.3)


@unittest.skipUnless(DEV, "needs the DEV dataset")
class DevFrames(unittest.TestCase):
    def test_make_refuses_non_dev_frames(self):
        for n in ("cap_seal_video_4.mp4@t0001.0.png", "some_capture.png"):
            d = tempfile.mkdtemp(); cv2.imwrite(os.path.join(d, n), fixture())
            with self.assertRaises(R.ProtectedDataError):
                RP.make(d, tempfile.mkdtemp())

    @unittest.skipUnless(os.path.exists(EVENT_PNG), "cached run-1 event frame not on this machine")
    def test_real_caps_same_states_through_1080(self):
        img = cv2.imread(EVENT_PNG); res = {}
        for k, im in (("640", img), ("1080", RP.warp_up(img, FWARP))):
            s, out = session(gate="each"); s.process_frame(EVENT_PNG, im); s.close()
            res[k] = read_csv(os.path.join(out, "predictions.csv"))
        same = sum(a["final_state"] == b["final_state"] for a, b in zip(res["640"], res["1080"]))
        self.assertGreaterEqual(same, 29)                                      # threshold-edge holes may flip
        self.assertTrue(any(r["final_state"] == "PASS" for r in res["1080"]))

    @unittest.skipUnless(os.path.exists(STILL), "CSI 1080 still not on this machine (never committed)")
    def test_csi_still_warped_holes_and_rule(self):
        warped = FWARP.apply(cv2.imread(STILL))
        h = hole_check(warped, PM); e = np.linalg.norm(np.array(list(h.values())), axis=1)
        self.assertEqual(len(h), 30)
        self.assertLess(float(np.median(e)), 2.5)
        s, out = session(gate="each"); s.process_frame(STILL, cv2.imread(STILL)); s.close()
        p = read_csv(os.path.join(out, "predictions.csv"))
        self.assertEqual({r["final_state"] for r in p}, {"WAITING"})
        self.assertEqual(read_csv(os.path.join(out, "trial_log.csv"))[0]["path"], "1080->640")


if __name__ == "__main__":
    unittest.main()
