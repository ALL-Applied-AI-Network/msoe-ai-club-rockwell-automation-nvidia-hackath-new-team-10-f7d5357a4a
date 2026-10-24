"""D040 v2 pieces: work2 overlay (twin_fix2.py), the gen_scenes floor block, CameraAugV2, the no-gap threshold fallback
and the check-render judge (v2_check.py). Twin-dependent tests use IGNITE_TWIN / IGNITE_TWIN_FIX like test_gen_scenes."""
import hashlib, io, json, os, sys, tarfile, tempfile, unittest
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import positions as P
import train_resnet as T

TWIN = os.path.expanduser(os.environ.get("IGNITE_TWIN", "~/Downloads/csi_digital_twin/csi_digital_twin_v1"))
TWIN_FIX = os.path.expanduser(os.environ.get("IGNITE_TWIN_FIX", "~/Downloads/csi_digital_twin/twin_fix1"))
try:
    import pxr  # noqa: F401
    HAVE_USD = True
except ImportError:
    HAVE_USD = False
HAVE_TWINS = HAVE_USD and os.path.isdir(TWIN) and os.path.isfile(os.path.join(TWIN_FIX, "FIX.json"))
TRAIN_V1 = os.path.join(P.HERE, "results", "train_v1", "data_train")
RECIPE = json.load(open(os.path.join(P.HERE, "configs", "recipe_dev.json")))
V1_AUG_SHA = "7bf3b2085e5adac1a341bbad0f4d6bd76a91d3307641b07af3b55725f0c7b32f"   # CameraAug before D040


def sample_crop():
    return cv2.GaussianBlur((np.random.default_rng(7).random((34, 39, 3)) * 255).astype(np.uint8), (0, 0), 1.5)


def strip(lab):
    return {k: v for k, v in lab.items() if k not in ("scene_id", "station", "floor")}


@unittest.skipUnless(HAVE_TWINS, "needs usd-core, the twin and its work1 overlay")
class Work2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import twin_fix2 as F2, gen_scenes as G
        cls.F2, cls.G = F2, G
        cls.d = tempfile.mkdtemp()                                         # the overlays must sit next to each other
        for src, name in ((TWIN, os.path.basename(TWIN)), (TWIN_FIX, "twin_fix1")):
            os.symlink(src, os.path.join(cls.d, name))
        cls.w1, cls.w2 = os.path.join(cls.d, "twin_fix1"), os.path.join(cls.d, "twin_fix2")
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            F2.build(cls.w1, cls.w2)

    def test_check_ok_and_a_stray_change_fails(self):
        import contextlib
        from pxr import Usd, UsdShade
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.F2.check(self.w1, self.w2))
        bad = os.path.join(self.d, "twin_fix2b"); os.makedirs(os.path.dirname(bad), exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.F2.build(self.w1, bad)
        st = Usd.Stage.Open(os.path.join(bad, self.F2.STATION_REL))
        UsdShade.Shader(st.GetPrimAtPath(self.F2.SOURCE_MAT + "/Principled_BSDF")).GetInput("roughness").Set(0.9)   # posts too
        st.GetRootLayer().Save()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(self.F2.check(self.w1, bad))

    def test_chained_overlay_resolves_the_twin_station_and_selftest_passes(self):
        import contextlib, shutil
        tw = self.G.Twin(self.w2)
        self.assertEqual(tw.base_station_sha256, self.G.Twin(TWIN).station_sha256)
        self.assertEqual(tw.fix["fix"], "work2")
        with contextlib.redirect_stdout(io.StringIO()):
            fails, tmp = self.G.selftest(tw)
        self.assertEqual(fails, []); shutil.rmtree(tmp, ignore_errors=True)

    def test_floor_config_needs_work2(self):
        cfg = json.load(open(os.path.join(P.HERE, "configs", "train_v2.json")))
        with self.assertRaises(self.G.ConfigError):
            self.G.validate_config({**cfg, "twin_fix": "work1"}, self.G.Twin(self.w1))
        for bad in ({**cfg["floor"], "extra": 1}, {**cfg["floor"], "val_scale": [1.3, 0.7]}, {**cfg["floor"], "blue_scale": 0}):
            with self.assertRaises(self.G.ConfigError):
                self.G.validate_config({**cfg, "floor": bad}, self.G.Twin(self.w2))
        self.G.validate_config(cfg, self.G.Twin(self.w2))

    def test_same_scene_except_the_floor(self):
        """train_v1 scene i on work1 and train_v2 scene i on work2: identical labels except id, station and floor."""
        G = self.G; out = {}
        for name, twin, cfg in (("v1", self.w1, "train_v1.json"), ("v2", self.w2, "train_v2.json")):
            c, sha = G.load_config(os.path.join(P.HERE, "configs", cfg), G.Twin(twin))
            o = os.path.join(tempfile.mkdtemp(), name)
            G.generate(G.Twin(twin), c, sha, o, indices=[76, 3], log=lambda *a: None)
            out[name] = o
        for i in (76, 3):
            a = json.load(open(os.path.join(out["v1"], "scenes", f"train_v1_{i:05d}", "labels.json")))
            b = json.load(open(os.path.join(out["v2"], "scenes", f"train_v2_{i:05d}", "labels.json")))
            self.assertEqual(strip(a), strip(b))
            self.assertNotIn("floor", a)
            f = b["floor"]; base = self.G.Twin(self.w2).floor_base
            self.assertTrue(0.7 <= f["val_scale"] <= 1.3 and 0.6 <= f["roughness"] <= 0.9)
            np.testing.assert_allclose(f["rgb"], np.array(base) * f["val_scale"], atol=1e-5)

    def test_indices_equal_the_full_run(self):
        G = self.G; c, sha = G.load_config(os.path.join(P.HERE, "configs", "train_v2.json"), G.Twin(self.w2))
        a, b = os.path.join(tempfile.mkdtemp(), "a"), os.path.join(tempfile.mkdtemp(), "b")
        G.generate(G.Twin(self.w2), c, sha, a, indices=[2], log=lambda *x: None)
        G.generate(G.Twin(self.w2), c, sha, b, limit=3, log=lambda *x: None)
        la, lb = (json.load(open(os.path.join(d, "scenes", "train_v2_00002", "labels.json"))) for d in (a, b))
        self.assertEqual({k: v for k, v in la.items() if k != "station"}, {k: v for k, v in lb.items() if k != "station"})
        with self.assertRaises(G.ConfigError):
            G.generate(G.Twin(self.w2), c, sha, os.path.join(tempfile.mkdtemp(), "c"), indices=[520], log=lambda *x: None)


class FloorStream(unittest.TestCase):
    def test_floor_draw_is_its_own_stream(self):
        import gen_scenes as G
        fb = {"val_scale": [0.7, 1.3], "roughness": [0.6, 0.9], "blue_scale": 0.85}
        a, b = G.sample_floor(fb, (0.3, 0.24, 0.18), 0, 5), G.sample_floor(fb, (0.3, 0.24, 0.18), 0, 5)
        self.assertEqual(a, b)
        self.assertNotEqual(a, G.sample_floor(fb, (0.3, 0.24, 0.18), 0, 6))
        self.assertAlmostEqual(a["rgb"][2], round(0.18 * a["val_scale"] * 0.85, 6), places=5)


class AugV2(unittest.TestCase):
    def test_v1_aug_unchanged(self):
        aug, rng = T.make_aug(RECIPE), np.random.default_rng(123)
        self.assertIs(type(aug), T.CameraAug)
        self.assertEqual(hashlib.sha256(b"".join(aug(sample_crop(), rng).tobytes() for _ in range(5))).hexdigest(), V1_AUG_SHA)

    def test_v2_ranges_and_determinism(self):
        v2 = json.load(open(os.path.join(P.HERE, "configs", "aug_v2.json")))
        aug = T.make_aug(RECIPE, os.path.join(P.HERE, "configs", "aug_v2.json")); self.assertIsInstance(aug, T.CameraAugV2)
        img = sample_crop()
        a, b = aug(img, np.random.default_rng(5)), aug(img, np.random.default_rng(5))
        np.testing.assert_array_equal(a, b); self.assertEqual(a.shape, img.shape); self.assertEqual(a.dtype, np.uint8)
        L0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean() * 100 / 255
        Ls = [cv2.cvtColor(aug(img, np.random.default_rng(s)), cv2.COLOR_BGR2LAB)[..., 0].mean() * 100 / 255 for s in range(200)]
        self.assertLess(min(Ls), L0 - 6); self.assertGreater(max(Ls), L0)        # L* shift -12..+3 reaches both sides
        self.assertEqual((v2["noise_sigma_8bit"][0], v2["blur_sigma"][0]), (0.0, 0.0))

    def test_each_v2_step(self):
        aug = T.CameraAugV2(RECIPE, {"L_shift": [-10, -10], "L_contrast": [1, 1], "b_shift": [0, 0], "noise_sigma_8bit": [0, 0],
                                     "blur_sigma": [0, 0]})
        aug.gain = (1.0, 1.0); aug.q = 100
        img = np.full((40, 40, 3), 140, np.uint8)
        dL = (cv2.cvtColor(aug(img, np.random.default_rng(0)), cv2.COLOR_BGR2LAB)[..., 0].mean() - cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean()) * 100 / 255
        self.assertAlmostEqual(dL, -10, delta=1.0)
        aug.dL, aug.db = (0, 0), (-8, -8)
        db = cv2.cvtColor(aug(img, np.random.default_rng(0)), cv2.COLOR_BGR2LAB)[..., 2].astype(float).mean() - cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 2].mean()
        self.assertAlmostEqual(db, -8, delta=1.5)
        aug.db, aug.blur = (0, 0), (0.7, 0.7)
        edge = np.zeros((40, 40, 3), np.uint8); edge[:, 20:] = 255
        hf = lambda x: np.abs(np.diff(x.astype(float), axis=1)).max()
        self.assertLess(hf(aug(edge, np.random.default_rng(0))), hf(edge))


class Fallback(unittest.TestCase):
    def test_gap_and_no_gap(self):
        y = np.array([0, 3, 1, 2]); pre = 0.9
        gap = np.array([[0.9, 0.1, 0, 0], [0.1, 0, 0, 0.9], [0, 0.95, 0, 0.05], [0, 0, 0.97, 0.03]])
        r = T.gap_threshold(gap, y, pre, {}, fallback_preregistered=True)
        self.assertAlmostEqual(r["t_defect"], (0.1 + 0.95) / 2); self.assertNotIn("d040_fallback", r)
        over = np.array([[0.2, 0.8, 0, 0], [0.1, 0, 0, 0.9], [0, 0.5, 0, 0.5], [0, 0, 0.97, 0.03]])
        self.assertIsNone(T.gap_threshold(over, y, pre, {})["t_defect"])                      # v1 behaviour (D038)
        r = T.gap_threshold(over, y, pre, {}, fallback_preregistered=True)
        self.assertEqual(r["t_defect"], pre); self.assertIn("d040_fallback", r)


class CheckJudge(unittest.TestCase):
    def crit(self):
        return json.load(open(os.path.join(P.HERE, "configs", "v2_check.json")))

    def rows(self, n, **kw):
        return [dict(kw) for _ in range(n)]

    def test_rules_and_next_step(self):
        import v2_check as C
        crit = self.crit(); base = crit["train_v1_baseline_same_scenes"]
        good = {"empty": self.rows(10, hole_L=23.0, hole_Lp90=50.0, hole_b=15.0),
                "uncapped": self.rows(5, mouth_L=base["uncapped"]["mouth_L"] + 1, mouth_b=base["uncapped"]["mouth_b"] - 1, hole_Lp90=75.0)}
        m, checks = C.judge(good, crit); self.assertTrue(all(checks.values()), checks)
        dark = {**good, "empty": self.rows(10, hole_L=14.8, hole_Lp90=21.0, hole_b=7.4)}
        m, checks = C.judge(dark, crit)
        self.assertTrue(checks["E1 empty core L*"])                             # E1 holds train_v1's dark holes too
        self.assertFalse(checks["E2 empty lit floor L*"] or checks["E3 empty b*"])
        step = C.next_floor(m, crit)
        self.assertGreater(step["value_factor"], 1.0); self.assertEqual(step["blue_factor"], 0.85)
        worse = {**good, "uncapped": self.rows(5, mouth_L=base["uncapped"]["mouth_L"] + 5, mouth_b=base["uncapped"]["mouth_b"], hole_Lp90=75.0)}
        self.assertFalse(C.judge(worse, crit)[1]["U1 uncapped mouth L* no worse"])
        close = {**good, "uncapped": self.rows(5, mouth_L=base["uncapped"]["mouth_L"], mouth_b=base["uncapped"]["mouth_b"], hole_Lp90=45.0)}
        self.assertFalse(C.judge(close, crit)[1]["U3 uncapped apart from empty"])

    @unittest.skipUnless(os.path.isfile(os.path.join(TRAIN_V1, "scenes.tar.gz")), "train_v1 renders not on this machine")
    def test_train_v1_renders_fail_the_empty_rules_and_match_the_baseline(self):
        """The check scenes' own train_v1 renders: empty holes outside the real ranges, uncapped exactly the baseline."""
        import v2_check as C
        crit = self.crit(); d = tempfile.mkdtemp()
        with tarfile.open(os.path.join(TRAIN_V1, "scenes.tar.gz")) as t:
            for i in crit["indices"]:
                t.extract(f"scenes/train_v1_{i:05d}/labels.json", d)
        F = C.measure(os.path.join(TRAIN_V1, "renders"), os.path.join(d, "scenes"), crit["indices"])
        m, checks = C.judge(F, crit); base = crit["train_v1_baseline_same_scenes"]
        self.assertEqual((len(F["empty"]), len(F["uncapped"])), (base["empty"]["n"], base["uncapped"]["n"]))
        self.assertFalse(checks["E2 empty lit floor L*"] or checks["E3 empty b*"]); self.assertTrue(checks["U1 uncapped mouth L* no worse"])
        self.assertTrue(checks["U3 uncapped apart from empty"])
        self.assertGreater(C.next_floor(m, crit)["value_factor"], 1.5)
        self.assertAlmostEqual(m["uncapped"]["mouth_L"], base["uncapped"]["mouth_L"], delta=0.01)


if __name__ == "__main__":
    unittest.main()
