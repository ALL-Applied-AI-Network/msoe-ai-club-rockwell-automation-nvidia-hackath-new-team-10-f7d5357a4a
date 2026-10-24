import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fill_state import (resolve_state, infer_fill_count, resolve_fixture, resolve_occupancy, SequenceTracker,
                        LIVE_FILL, FINAL_INSPECTION)

S = lambda i, K, v: resolve_state(i, K, v)[0]


def fx(occupied, other="no_vial", **special):
    """30 visual states: 'pass' at the occupied positions, `other` elsewhere; special={'G07': 'missing_cap'}."""
    v = ["pass" if i in occupied else other for i in range(1, 31)]
    for name, state in special.items():
        v[int(name[1:]) - 1] = state
    return v


def states(visuals, **kw):
    return [s for s, _ in resolve_occupancy(visuals, **kw)[1]]


class OfflineFillCount(unittest.TestCase):
    """K = optional offline metadata (synthetic scenes, staged ground truth); semantics unchanged."""

    def test_k0_everything_waits(self):
        self.assertEqual({S(i, 0, "no_vial") for i in range(1, 31)}, {"WAITING"})

    def test_partially_filled(self):
        K = 12
        self.assertEqual(S(1, K, "pass"), "PASS")
        self.assertEqual(S(7, K, "missing_cap"), "DEFECT")
        self.assertEqual(S(8, K, "crooked_cap"), "DEFECT")
        self.assertEqual(S(9, K, "damaged_vial"), "DEFECT")
        self.assertEqual(S(13, K, "no_vial"), "WAITING")
        self.assertEqual(S(30, K, "no_vial"), "WAITING")

    def test_skipped_position_inside_prefix_is_missed(self):
        self.assertEqual(S(5, 12, "no_vial"), "MISSED")
        self.assertEqual(S(12, 12, "no_vial"), "MISSED")       # a skipped LAST placement needs the count
        self.assertEqual(S(30, 30, "no_vial"), "MISSED")

    def test_positions_after_k(self):
        self.assertEqual(S(13, 12, "pass"), "UNSURE")           # a vial where none should be yet
        self.assertEqual(S(20, 12, "missing_cap"), "UNSURE")

    def test_unknown_visual_is_unsure(self):
        for K in (0, 12, 30):
            self.assertEqual(S(5, K, "unknown"), "UNSURE")

    def test_reason_text(self):
        self.assertEqual(resolve_state(7, 12, "missing_cap"), ("DEFECT", "missing_cap"))
        self.assertIn("K=12", resolve_state(5, 12, "no_vial")[1])

    def test_uncapped_vial_read_as_empty_at_frontier(self):
        # rule-baseline limit (DEV run 1, event 8): an uncapped vial read as no_vial looks exactly like waiting
        v = fx(set(range(1, 7)))
        self.assertEqual(resolve_fixture(v, fill_count=7)[2][6][0], "MISSED")      # a count still flags it
        self.assertEqual(states(v)[6], "WAITING")                                 # occupancy alone cannot
        self.assertEqual(states(v, frontier="unsure")[6], "UNSURE")

    def test_given_count_beats_occupancy(self):
        K, src, st = resolve_fixture(fx({1, 2}), fill_count=3)
        self.assertEqual((K, src, st[2][0]), (3, "given", "MISSED"))

    def test_bad_inputs(self):
        for K in (-1, 31, 2.0, "3", True, None):
            with self.assertRaises(ValueError):
                resolve_state(1, K, "pass")
        for i in (0, 31):
            with self.assertRaises(ValueError):
                resolve_state(i, 5, "pass")
        with self.assertRaises(ValueError):
            resolve_state(1, 5, "GOOD")
        with self.assertRaises(ValueError):
            resolve_fixture(["pass"] * 29)
        with self.assertRaises(ValueError):
            resolve_fixture(["GOOD"] * 30)
        with self.assertRaises(ValueError):
            resolve_occupancy(fx({1}), frontier="missed")
        for mode in ("complete", True, None):                         # no implicit / boolean completion
            with self.assertRaises(ValueError):
                resolve_occupancy(fx({1}), mode=mode)


class OccupancyOnly(unittest.TestCase):
    """Runtime default: no placement count from the line (D021)."""

    def test_frontier(self):
        self.assertEqual(infer_fill_count(fx(set())), 0)
        self.assertEqual(infer_fill_count(fx({1, 2, 3, 5, 6})), 6)
        self.assertEqual(infer_fill_count(fx({1, 2}, G03="missing_cap")), 3)   # a defect still occupies
        self.assertEqual(infer_fill_count(fx({1, 2}, G10="unknown")), 2)       # unknown never extends F

    def test_normal_partial_fill(self):
        F, src, st = resolve_fixture(fx(set(range(1, 9))))
        self.assertEqual((F, src), (8, "occupancy"))
        self.assertEqual([s for s, _ in st[:8]], ["PASS"] * 8)
        self.assertEqual([s for s, _ in st[8:]], ["WAITING"] * 22)

    def test_single_skipped_position(self):
        st = states(fx(set(range(1, 9)) | {10}))                      # G01-G08, G09 empty, G10
        self.assertEqual(st[8], "MISSED")
        self.assertEqual(st[9], "PASS")
        self.assertEqual(set(st[10:]), {"WAITING"})

    def test_two_skipped_positions(self):
        st = states(fx({1, 2, 3, 5, 7, 8}))
        self.assertEqual((st[3], st[5]), ("MISSED", "MISSED"))
        self.assertEqual(st.count("MISSED"), 2)

    def test_trailing_positions_wait_and_no_false_missed(self):
        st = states(fx(set(range(1, 13))))
        self.assertNotIn("MISSED", st)
        self.assertEqual(set(st[12:]), {"WAITING"})
        self.assertNotIn("MISSED", states(fx(set())))                 # empty fixture: nothing missed

    def test_frontier_ambiguity_policy(self):
        v = fx(set(range(1, 9)))
        default = resolve_occupancy(v)[1]
        self.assertEqual(default[8][0], "WAITING")
        self.assertIn("fill frontier", default[8][1])
        cautious = states(v, frontier="unsure")
        self.assertEqual((cautious[8], cautious[9]), ("UNSURE", "WAITING"))   # only F+1 is ambiguous

    def test_missing_cap_on_filled_position(self):
        st = resolve_occupancy(fx(set(range(1, 6)), G03="missing_cap", G04="crooked_cap"))[1]
        self.assertEqual(st[2], ("DEFECT", "missing_cap"))
        self.assertEqual(st[3], ("DEFECT", "crooked_cap"))

    def test_no_vial_before_later_defect_is_missed(self):
        st = states(fx({1, 2, 3}, G05="missing_cap"))                 # later position holds a defective vial
        self.assertEqual((st[3], st[4]), ("MISSED", "DEFECT"))

    def test_unreadable_position_is_flagged_alone(self):
        st = states(fx({1, 2, 3}, G06="unknown"))                     # glare / overlap ghost at G06
        self.assertEqual(st[3:6], ["WAITING", "WAITING", "UNSURE"])   # only G06 itself is flagged; F stays G03
        self.assertEqual(set(st[6:]), {"WAITING"})
        self.assertEqual(states(fx({1, 2}, G04="unknown", G06="pass"))[2:6], ["MISSED", "UNSURE", "MISSED", "PASS"])

    def test_live_fill_examples(self):
        st = states(fx(set(range(1, 9)) | {10}), mode=LIVE_FILL)     # G01-G08, G09 empty, G10 occupied
        self.assertEqual(st[8], "MISSED")
        st = states(fx(set(range(1, 9))), mode=LIVE_FILL)              # G01-G08, G09-G30 empty
        self.assertNotIn("MISSED", st)                                # nothing promoted while filling may continue
        self.assertEqual(set(st[8:]), {"WAITING"})

    def test_final_inspection_is_external_context(self):
        v = fx(set(range(1, 21)) - {7})
        st = states(v, mode=FINAL_INSPECTION)
        self.assertEqual(st[6], "MISSED")                             # internal empty
        self.assertEqual(set(st[20:]), {"MISSED"})                    # a complete good order has 30 vials
        self.assertNotIn("WAITING", st)
        self.assertIn("test protocol/operator", resolve_occupancy(v, mode=FINAL_INSPECTION)[1][25][1])
        self.assertEqual(states(fx(set(range(1, 31))), mode=FINAL_INSPECTION), ["PASS"] * 30)
        self.assertEqual(states(v)[20:], ["WAITING"] * 10)            # same frame, default LIVE_FILL


class Tracker(unittest.TestCase):
    def test_waiting_then_missed(self):
        t = SequenceTracker(persist=1)
        _, _, a = t.update(fx(set(range(1, 9))))                      # frame A: G09 empty, G10 not filled
        self.assertEqual(a[8][0], "WAITING")
        _, _, b = t.update(fx(set(range(1, 9)) | {10}))               # frame B: G10 filled, G09 still empty
        self.assertEqual(b[8][0], "MISSED")

    def test_unsure_frontier_then_missed(self):
        t = SequenceTracker(persist=1, frontier="unsure")
        self.assertEqual(t.update(fx(set(range(1, 9))))[2][8][0], "UNSURE")
        self.assertEqual(t.update(fx(set(range(1, 9)) | {10}))[2][8][0], "MISSED")

    def test_one_frame_flicker_does_not_latch(self):
        t = SequenceTracker(persist=2)
        base = set(range(1, 9))
        for _ in range(2):
            t.update(fx(base))
        C, F, st = t.update(fx(base | {10}))                          # G10 appears for one frame only
        self.assertEqual((C, F, st[8][0]), (8, 10, "UNSURE"))
        C, F, st = t.update(fx(base))                                 # flicker gone
        self.assertEqual((C, st[8][0], st[9][0]), (8, "WAITING", "WAITING"))
        t.update(fx(base | {10}))
        C, _, st = t.update(fx(base | {10}))                          # now persistent
        self.assertEqual((C, st[8][0]), (10, "MISSED"))

    def test_missed_stays_latched_and_unknown_frames_keep_history(self):
        t = SequenceTracker(persist=1)
        t.update(fx(set(range(1, 9)) | {10}))
        st = t.update(fx(set(range(1, 9)), G10="unknown"))[2]        # G10 occluded later
        self.assertEqual((st[8][0], st[9][0]), ("MISSED", "UNSURE"))
        self.assertEqual(t.frontier_confirmed, 10)

    def test_confirmed_vial_reads_empty(self):
        t = SequenceTracker(persist=1)
        t.update(fx({1, 2, 3}))
        st = t.update(fx({1, 3}))[2]
        self.assertEqual(st[1][0], "UNSURE")
        self.assertIn("was confirmed occupied", st[1][1])

    def test_persist1_matches_single_frame_on_monotone_fill(self):
        t = SequenceTracker(persist=1)
        seq = [set(range(1, k)) - {4} for k in range(2, 15)]
        for occ in seq:
            v = fx(occ)
            self.assertEqual([s for s, _ in t.update(v)[2]], states(v))

    def test_final_inspection_tracker(self):
        t = SequenceTracker(persist=1, mode=FINAL_INSPECTION)
        st = t.update(fx(set(range(1, 29))))[2]                       # protocol says complete; G29-G30 empty
        self.assertEqual((st[28][0], st[29][0]), ("MISSED", "MISSED"))

    def test_bad_args(self):
        for p in (0, -1, 1.5, None, True):
            with self.assertRaises(ValueError):
                SequenceTracker(persist=p)
        with self.assertRaises(ValueError):
            SequenceTracker(frontier="missed")
        with self.assertRaises(ValueError):
            SequenceTracker(mode="complete")


if __name__ == "__main__":
    unittest.main()
