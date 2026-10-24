import os, sys, tempfile, unittest, warnings
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import real_data as R


class SplitGuard(unittest.TestCase):
    """Uses the real real_splits.json roles with a fake index, so the guard is tested as it is configured."""

    def setUp(self):
        self.root = tempfile.mkdtemp(); os.makedirs(os.path.join(self.root, "images"))
        rows = [("cap_seal_a.jpg", "2026-09-22 17:00:00", "1"), ("cap_seal_b.jpg", "2026-09-23 19:10:00", "4"),
                ("cap_seal_c.jpg", "2026-09-23 19:50:00", "5"), ("cap_seal_d.jpg", "2026-09-22 17:51:55", "unassigned")]
        with open(os.path.join(self.root, "image_index.csv"), "w") as f:
            f.write("filename,timestamp,video_id\n" + "".join(f"{a},{b},{c}\n" for a, b, c in rows))
        for name, *_ in rows:
            cv2.imwrite(os.path.join(self.root, "images", name), np.zeros((480, 640, 3), np.uint8))
        self.ix = R.Index(self.root)

    def test_roles_come_from_real_splits(self):
        r = R.roles()
        self.assertEqual({k: r[k] for k in ("1", "2", "3", "4", "5", "unassigned")},
                         {"1": "DEV", "2": "DEV", "3": "DEV", "4": "MODEL_HOLDOUT", "5": "MODEL_HOLDOUT", "unassigned": "IGNORE"})

    def test_dev_still_loads(self):
        self.assertEqual(self.ix.load("cap_seal_a.jpg").shape, (480, 640, 3))
        self.assertEqual(self.ix.frames("1"), ["cap_seal_a.jpg"])

    def test_holdout_and_unassigned_refused(self):
        for name in ("cap_seal_b.jpg", "cap_seal_c.jpg", "cap_seal_d.jpg"):
            with self.assertRaises(R.ProtectedDataError):
                self.ix.load(name)
        for run in ("4", "5"):
            with self.assertRaises(R.ProtectedDataError):
                self.ix.frames(run)
            with self.assertRaises(R.ProtectedDataError):
                self.ix.video_path(run)

    def test_cached_video_frame_keeps_its_run(self):
        cache = os.path.join(tempfile.mkdtemp(), "cap_seal_video_5.mp4@t0010.0.png")
        cv2.imwrite(cache, np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(self.ix.run_of_source(cache), "5")
        with self.assertRaises(R.ProtectedDataError):         # a copied holdout frame is still holdout
            self.ix.load(cache)

    def test_unindexed_file_inside_dataset_refused(self):
        stray = os.path.join(self.root, "images", "cap_seal_zz.jpg"); cv2.imwrite(stray, np.zeros((4, 4, 3), np.uint8))
        with self.assertRaises(R.ProtectedDataError):
            self.ix.load(stray)

    def test_external_frame_allowed_with_warning(self):
        ext = os.path.join(tempfile.mkdtemp(), "staged_session_a_001.png"); cv2.imwrite(ext, np.zeros((480, 640, 3), np.uint8))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertEqual(self.ix.load(ext).shape, (480, 640, 3))
        self.assertTrue(any("external" in str(x.message) for x in w))

    def test_explicit_holdout_flag(self):
        self.assertEqual(self.ix.load("cap_seal_b.jpg", allow_holdout=True).shape, (480, 640, 3))
        with self.assertRaises(R.ProtectedDataError):         # IGNORE stays refused even then
            self.ix.load("cap_seal_d.jpg", allow_holdout=True)


if __name__ == "__main__":
    unittest.main()
