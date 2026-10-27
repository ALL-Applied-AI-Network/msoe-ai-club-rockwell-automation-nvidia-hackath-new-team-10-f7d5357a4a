import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_metrics as em


def row(pos, gt, pred, score="", gseq="", pseq="", final="", run="2", event="1", frame="f1.jpg"):
    return dict(run=run, event=event, frame=frame, position=pos, ground_truth=gt, predicted_state=pred,
                defect_score=score, gt_sequence=gseq, pred_sequence=pseq, final_state=final)


class Eval(unittest.TestCase):
    def test_selftest(self):
        em.selftest()

    def test_events_from_two_frames_rejected(self):
        rows = [row("G01", "pass", "pass", frame="a.jpg"), row("G02", "pass", "pass", frame="b.jpg")]
        with self.assertRaises(ValueError):
            em.evaluate(rows)

    def test_process_confusion_and_waiting(self):
        rows = [row("G01", "pass", "pass", final="PASS"),
                row("G02", "no_vial", "no_vial", gseq="missed", pseq="missed", final="MISSED"),
                row("G03", "no_vial", "no_vial", gseq="waiting", pseq="waiting", final="WAITING"),
                row("G04", "no_vial", "unknown", gseq="waiting", final="UNSURE")]
        m = em.confusion(rows, em.gt_final, lambda r: r["final_state"])
        self.assertEqual(m, {"MISSED": {"MISSED": 1}, "PASS": {"PASS": 1}, "WAITING": {"WAITING": 1, "UNSURE": 1}})
        out = em.evaluate(rows)
        self.assertEqual(out["trailing_empty_false_alarms"], 1)     # UNSURE on a waiting hole goes to a person
        self.assertEqual(out["escalated_share"], 0.25)

    def test_sweep_ignores_other_classes(self):
        rows = [row("G01", "pass", "pass", 0.1), row("G02", "missing_cap", "missing_cap", 0.9),
                row("G03", "no_vial", "no_vial", 1.0, gseq="waiting")]   # high score, but not part of this task
        sw = em.sweep(rows, {"missing_cap"}, {"pass"})
        self.assertEqual((sw["n_pos"], sw["n_neg"], sw["far_at_full_recall"]), (1, 1, 0.0))
        self.assertIn("note", em.sweep(rows, {"crooked_cap"}, {"pass"}))  # no positives -> no fake number

    def test_join_labels_by_frame(self):
        preds = [row("G01", "", "pass"), row("G02", "", "no_vial")]
        for p in preds:
            del p["ground_truth"], p["gt_sequence"]
        labels = [dict(run="2", event="9", frame="f1.jpg", position="G01", ground_truth="pass", gt_sequence="")]
        joined, unlabeled, unmatched = em.join_labels(preds, labels)
        self.assertEqual((len(joined), unlabeled, unmatched), (1, 1, []))
        self.assertEqual(joined[0]["event"], "1")                   # event id comes from the predictions


if __name__ == "__main__":
    unittest.main()
