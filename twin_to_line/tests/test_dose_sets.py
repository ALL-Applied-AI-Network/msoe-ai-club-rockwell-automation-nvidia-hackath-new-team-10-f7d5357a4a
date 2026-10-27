import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dose_sets import make_dose_sets, InsufficientDefects, unit


def pool(n_missing=10, n_crooked=10, n_pass=40, split="adapt"):
    rows = []
    for label, n in (("missing_cap", n_missing), ("crooked_cap", n_crooked), ("pass", n_pass)):
        rows += [dict(session="A", event=f"{label}-{i}", position=f"G{i % 30 + 1:02d}", label=label, split=split, path="")
                 for i in range(n)]
    return rows


ids = lambda rows: [unit(r) for r in rows]


class DoseSets(unittest.TestCase):
    def test_nested(self):
        s = make_dose_sets(pool())
        for a, b in ((0, 5), (5, 10), (10, 20)):
            self.assertEqual(ids(s[b]["defects"])[:a], ids(s[a]["defects"]))
            self.assertTrue(set(ids(s[a]["companions"]["pass"])) <= set(ids(s[b]["companions"]["pass"])))

    def test_deterministic_and_seeded(self):
        a, b, c = make_dose_sets(pool(), seed=0), make_dose_sets(pool(), seed=0), make_dose_sets(pool(), seed=1)
        self.assertEqual(ids(a[20]["defects"]), ids(b[20]["defects"]))
        self.assertNotEqual(ids(a[20]["defects"]), ids(c[20]["defects"]))

    def test_stratified_and_balanced(self):
        s = make_dose_sets(pool())
        self.assertEqual(s[0]["counts"], {})
        self.assertEqual(s[5]["counts"], {"missing_cap": 3, "crooked_cap": 2, "pass": 5})
        self.assertEqual(s[10]["counts"], {"missing_cap": 5, "crooked_cap": 5, "pass": 10})

    def test_short_class_is_filled_by_others(self):
        s = make_dose_sets(pool(n_missing=18, n_crooked=2))
        self.assertEqual(s[20]["counts"]["crooked_cap"], 2)
        self.assertEqual(s[20]["counts"]["missing_cap"], 18)

    def test_refuses_too_few_defects(self):
        with self.assertRaises(InsufficientDefects):          # today's DEV runs: ~2 real defect events
            make_dose_sets(pool(n_missing=1, n_crooked=1))
        with self.assertRaises(InsufficientDefects):
            make_dose_sets(pool(n_pass=5))

    def test_refuses_test_or_holdout_rows(self):
        p = pool(); p[0]["split"] = "holdout"
        with self.assertRaises(ValueError):
            make_dose_sets(p)

    def test_refuses_duplicate_units(self):
        p = pool(); p.append(dict(p[0]))
        with self.assertRaises(ValueError):
            make_dose_sets(p)


if __name__ == "__main__":
    unittest.main()
