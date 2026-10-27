import csv, os, sys, tempfile, threading, time, unittest, unittest.mock
from datetime import datetime, timedelta
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import positions as P
import real_data as R
import eval_metrics as M
import live_inspect as L
from events import segment_events
import inspect_frame as IF
from inspect_frame import inspect
from rules import RuleInspector

PM = P.PositionMap()
YELLOW = (0, 220, 230)                                   # BGR, inside the cap HSV range
T0 = datetime(2026, 10, 26, 13, 0, 0)
DEV = os.path.exists(os.path.join(R.REAL_ROOT, "image_index.csv"))
MODEL_DIR = os.path.join(P.HERE, "results", "train", "resnet18_v1")
RENDERS = os.path.join(P.HERE, "results", "train_v1", "data_train", "renders")
G3_D038 = os.path.join(P.HERE, "results", "g3", "resnet18_v1", "predictions_resnet_d038.csv")
try:
    import torch  # noqa: F401
    HAVE_MODEL = os.path.exists(os.path.join(MODEL_DIR, "thresholds_d038.json"))
except ImportError:
    HAVE_MODEL = False


def fixture(filled=(), half=(), arm=False):
    """640x480 synthetic frame: painted caps at the twin's seated-cap centres (1-based positions), half caps
    shifted 9 px off the seat, optionally a large yellow 'arm' over the fixture."""
    img = np.full((480, 640, 3), 90, np.uint8)
    for i, dx in [(i, 0) for i in filled] + [(i, 9) for i in half]:
        x, y = PM.cap_centers[i - 1]
        cv2.circle(img, (int(round(x + dx)), int(round(y))), 10, YELLOW, -1)
    if arm:
        x, y = PM.cap_centers[14].astype(int)
        cv2.rectangle(img, (x - 60, y - 40), (x + 60, y + 40), YELLOW, -1)
    return img


def name(k, dt=2.0):
    return f"syn_{(T0 + timedelta(seconds=k * dt)).strftime('%Y%m%d_%H%M%S_%f')}.png"


def session(gate="each", ix=None, **kw):
    out = tempfile.mkdtemp()
    return L.Session(out, gate=gate, ix=ix, quiet=True, **kw), out


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def finals(out, event):
    return {r["position"]: r["final_state"] for r in read_csv(os.path.join(out, "predictions.csv")) if r["event"] == str(event)}


class OnlineGateMatchesBatch(unittest.TestCase):
    """EventGate (online) gives the same events as events.segment_events (batch) on the same samples (R3 logic)."""
    BOXES = [(10, 10, 30, 30), (40, 10, 60, 30), (70, 10, 90, 30)]
    IDS = ["G01", "G02", "G03"]

    def frame(self, state, rng):
        im = np.full((40, 100, 3), 100, np.uint8)
        for i, on in enumerate(state):
            if on:
                x0, y0, x1, y1 = self.BOXES[i]; im[y0:y1, x0:x1] = 200
        return np.clip(im.astype(int) + rng.integers(-4, 5, im.shape), 0, 255).astype(np.uint8)

    def test_random_sequences(self):
        rng = np.random.default_rng(0); n_events = 0
        for _ in range(300):
            state, seq, arm = [0, 0, 0], [], []
            for _ in range(int(rng.integers(5, 60))):
                r = rng.random()
                if r < 0.08: state[int(rng.integers(3))] ^= 1               # placement / change
                seq.append(self.frame(state, rng)); arm.append(bool(rng.random() < 0.1))
            t = np.cumsum(rng.choice([1.0, 2.0], len(seq)))
            samples = list(zip(t, seq))
            batch = [(e.t, e.changed) for e in segment_events(samples, self.BOXES, self.IDS, arm, 15, 4)]
            g = L.EventGate(self.BOXES, self.IDS, 15, 4); online = []
            for (tt, img), a in zip(samples, arm):
                s, ev, ch = g.update(str(tt), tt, img, a)
                if s == "event":
                    online.append((ev[1], ch))
            self.assertEqual(online, batch); n_events += len(batch)
        self.assertGreater(n_events, 100)


class Gating(unittest.TestCase):
    def test_arm_frames_never_scored(self):                                     # R3
        frames = [fixture()] * 3 + [fixture(arm=True)] * 2 + [fixture([1])] * 4 + [fixture([1], arm=True)] + [fixture([1, 2])] * 4
        arm_frames = {3, 4, 9}
        for gate in ("events", "each"):
            s, out = session(gate)
            rows = [s.process_frame(name(k), img) for k, img in enumerate(frames)]; s.close()
            self.assertEqual({k for k, r in enumerate(rows) if r["status"] == "arm"}, arm_frames)
            for k in arm_frames:
                self.assertGreater(rows[k]["arm_px"], 0)
                self.assertFalse(any(rows[k].get(f"pred_{p}") for p in L.IDS))
            scored = {r["event_frame"] for r in rows if r["status"] == "scored"}
            self.assertEqual({r["frame"] for r in read_csv(os.path.join(out, "predictions.csv"))}, scored)
            if gate == "events":                                                 # empty, G01 placed, G02 placed
                self.assertEqual([r["event_frame"] for r in rows if r["status"] == "scored"], [name(0), name(5), name(10)])
            else:
                self.assertEqual(len(scored), sum(not r["arm_px"] for r in rows))

    def test_event_scores_first_frame_of_segment(self):
        s, _ = session("events")
        rows = [s.process_frame(name(k), fixture([1])) for k in range(4)]; s.close()
        self.assertEqual([r["status"] for r in rows], ["settling", "settling", "scored", "unchanged"])
        self.assertEqual((rows[2]["event_frame"], rows[2]["event_delay_s"]), (name(0), 4.0))


class FillState(unittest.TestCase):
    def test_partial_fills_live_fill(self):                                     # R5
        for last in (1, 7, 12, 29):
            gaps = {g for g in (4, 9, 20) if g < last}
            s, out = session()
            s.process_frame(name(0), fixture([i for i in range(1, last + 1) if i not in gaps])); s.close()
            f = finals(out, 1)
            for i in range(1, 31):
                want = "MISSED" if i in gaps else "PASS" if i <= last else "WAITING"
                self.assertEqual(f[f"G{i:02d}"], want, (last, i))

    def test_unreadable_position_is_unsure_and_does_not_move_frontier(self):
        s, out = session()
        s.process_frame(name(0), fixture(range(1, 11), half=[14])); s.close()
        f = finals(out, 1)
        self.assertEqual(f["G14"], "UNSURE")
        self.assertEqual({f[f"G{i:02d}"] for i in (11, 12, 13, 15, 30)}, {"WAITING"})
        self.assertEqual(read_csv(os.path.join(out, "predictions.csv"))[0]["K"], "10")


class FrameNames(unittest.TestCase):
    def test_frame_id_and_time(self):
        self.assertEqual(L.frame_id("/x/cap_seal_video_1.mp4@t0612.0.png"), "cap_seal_video_1.mp4@t0612.0")   # = label id
        self.assertEqual(L.frame_id("/x/cap_seal_20260922_192843_993721.jpg"), "cap_seal_20260922_192843_993721.jpg")
        self.assertEqual(L.frame_time("cap_seal_video_1.mp4@t0612.0.png"), 612.0)
        a, b = L.frame_time("cap_seal_20260922_192843_993721.jpg"), L.frame_time("cap_seal_20260922_192845_993721.jpg")
        self.assertAlmostEqual(b - a, 2.0)


class ReplayAndOutputs(unittest.TestCase):
    def test_replay_in_time_order_writes_outputs_eval_reads_them(self):         # R1, R6
        d = tempfile.mkdtemp(); fills = [[], [1], [1, 2], [1, 2], [1, 3], [1, 3, 4], [1, 3, 4], [1, 3, 4]]
        for k, f in enumerate(fills):
            p = os.path.join(d, name(k)); cv2.imwrite(p, fixture(f))
            os.utime(p, (1e9 - k, 1e9 - k))                                     # mtimes reversed: order must come from names
        s, out = session("each")
        L.run(s, L.replay(d))
        trial = read_csv(os.path.join(out, "trial_log.csv"))
        self.assertEqual([r["input_file"] for r in trial], [name(k) for k in range(len(fills))])
        for f in ("predictions.csv", "trial_log.csv", "overlay_latest.png", "session.json"):
            self.assertTrue(os.path.exists(os.path.join(out, f)), f)
        self.assertEqual(len(os.listdir(os.path.join(out, "overlays"))), len(fills))
        labels = []
        for k, f in enumerate(fills):
            last = max(f, default=0)
            for i in range(1, 31):
                gt, seq = ("pass", "") if i in f else ("no_vial", "missed" if i < last else "waiting")
                labels.append({"frame": name(k), "position": f"G{i:02d}", "ground_truth": gt, "gt_sequence": seq})
        rows, unlabeled, unmatched = M.join_labels(M.load(os.path.join(out, "predictions.csv")), labels)
        self.assertEqual((len(rows), unlabeled, len(unmatched)), (30 * len(fills), 0, 0))
        m = M.evaluate(rows)
        self.assertEqual((m["defect_recall"], m["false_positive_rate"]), (1.0, 0.0))   # G02 missed in frames 4-7
        self.assertEqual(m["counts"]["defects"], 4)

    def test_expected_states_in_trial_log(self):
        d = tempfile.mkdtemp(); lab = os.path.join(d, "labels.csv")
        with open(lab, "w", newline="") as f:
            w = csv.DictWriter(f, ["frame", "position", "ground_truth", "gt_sequence", "usable"]); w.writeheader()
            w.writerow({"frame": name(0), "position": "G01", "ground_truth": "pass", "gt_sequence": "", "usable": "1"})
            w.writerow({"frame": name(0), "position": "G02", "ground_truth": "no_vial", "gt_sequence": "missed", "usable": "1"})
        s, _ = session(labels=lab)
        r = s.process_frame(name(0), fixture([1])); s.close()
        self.assertEqual((r["exp_G01"], r["exp_G02"], r["exp_G03"]), ("PASS", "MISSED", ""))
        self.assertEqual(r["n_differ_from_expected"], 1)                         # G02 reads WAITING, expected MISSED


class FolderSource(unittest.TestCase):
    def test_files_copied_in_are_picked_up_within_1s(self):                     # R1
        d = tempfile.mkdtemp(); src = tempfile.mkdtemp(); done = {}
        for k in range(5):
            cv2.imwrite(os.path.join(src, name(k)), fixture(range(1, k + 2)))
        half = os.path.join(src, name(5)); cv2.imwrite(half, fixture([1]))

        def writer():
            for k in range(5):
                time.sleep(0.4)
                data = open(os.path.join(src, name(k)), "rb").read()
                with open(os.path.join(d, name(k)), "wb") as f:
                    f.write(data)
                done[name(k)] = time.monotonic()
            data = open(half, "rb").read()                                       # half-written file, finished 0.6 s later
            with open(os.path.join(d, name(5)), "wb") as f:
                f.write(data[:len(data) // 2]); f.flush(); time.sleep(0.6); f.write(data[len(data) // 2:])
            done[name(5)] = time.monotonic()

        s, out = session(); picked = []
        frames = L.watch(d, poll=0.1, idle_exit=1.5)                            # empty folder at start
        th = threading.Thread(target=writer); th.start()
        for path, img, t in frames:
            picked.append((os.path.basename(path), time.monotonic(), s.process_frame(path, img, t)["status"]))
        th.join(); s.close()
        self.assertEqual([p[0] for p in picked], [name(k) for k in range(6)])
        self.assertEqual({p[2] for p in picked}, {"scored"})                    # the half-written file was not read early
        for n, t, _ in picked:
            self.assertLess(t - done[n], 1.0, n)

    def test_bad_inputs_are_logged_and_the_loop_continues(self):
        d = tempfile.mkdtemp()
        open(os.path.join(d, name(0)), "wb").write(os.urandom(5000))                       # never completes
        cv2.imwrite(os.path.join(d, name(1)), np.zeros((240, 320, 3), np.uint8))           # wrong size
        cv2.imwrite(os.path.join(d, name(2)), fixture([1]))
        s, out = session()
        L.run(s, L.watch(d, poll=0.05, idle_exit=0.6, give_up_s=0.3, skip=()))      # files written before the start
        st = {r["input_file"]: r["status"] for r in read_csv(os.path.join(out, "trial_log.csv"))}
        self.assertEqual(st, {name(0): "unreadable", name(1): "wrong_size", name(2): "scored"})


class DataGuard(unittest.TestCase):
    """Runs 4-5 (and unassigned frames) are refused; uses the real real_splits.json roles with a fake index."""

    def setUp(self):
        self.root = tempfile.mkdtemp(); os.makedirs(os.path.join(self.root, "images"))
        with open(os.path.join(self.root, "image_index.csv"), "w") as f:
            f.write("filename,timestamp,video_id\ncap_seal_a.jpg,2026-09-22 17:00:00,1\ncap_seal_b.jpg,2026-09-23 19:10:00,4\n")
        self.ix = R.Index(self.root)

    def test_replay_folder_with_a_holdout_frame_is_refused_before_reading(self):
        d = tempfile.mkdtemp()
        for n in ("cap_seal_a.jpg", "cap_seal_b.jpg"):
            cv2.imwrite(os.path.join(d, n), fixture())
        with self.assertRaises(R.ProtectedDataError):
            L.replay(d, self.ix)

    def test_folder_frames_refused_one_by_one(self):
        s, _ = session(ix=self.ix)
        st = [s.process_frame(n, fixture())["status"] for n in
              ("cap_seal_b.jpg", "cap_seal_video_5.mp4@t0010.0.png", "cap_seal_a.jpg", "new_csi_capture.jpg")]
        s.close()
        self.assertEqual(st[:2], ["refused", "refused"])
        self.assertNotIn("refused", st[2:])                                      # DEV and external frames pass


@unittest.skipUnless(HAVE_MODEL and os.path.isdir(RENDERS), "needs torch, results/train/resnet18_v1 and the train_v1 renders")
class ResNetSession(unittest.TestCase):
    """The ResNet inspector in a live session, on synthetic renders only (640x480, gate 'each')."""

    def test_session_scores_and_logs_model_provenance(self):
        import json
        from resnet_inspector import sha256 as file_sha
        insp = L.make_inspector("resnet", device="cpu", quiet=True)
        s, out = session("each", inspector=insp)
        renders = sorted(f for f in os.listdir(RENDERS) if f.endswith(".png") and "_pinhole" not in f)[:2]
        rows = [s.process_frame(name(k), cv2.imread(os.path.join(RENDERS, f))) for k, f in enumerate(renders)]
        s.close()
        self.assertEqual([r["status"] for r in rows], ["scored", "scored"])
        self.assertEqual({r["model_sha256"] for r in rows}, {file_sha(os.path.join(MODEL_DIR, "model.pt"))})
        self.assertEqual({r["inspector"] for r in rows}, {"resnet18_synth_v1[d038]"})
        info = json.load(open(os.path.join(out, "session.json")))["inspector"]
        self.assertEqual(info["t_defect"], insp.t_d038); self.assertEqual(info["t_defect_preregistered"], insp.t_pre)
        self.assertEqual(len(info["files"]), 3)
        preds = read_csv(os.path.join(out, "predictions.csv"))
        self.assertEqual(len(preds), 60); self.assertEqual({p["cap_frac"] for p in preds}, {""})
        self.assertTrue(all(0 <= float(p["defect_score"]) <= 1 for p in preds))
        M.evaluate([dict(p, ground_truth="pass", gt_sequence="") for p in preds])     # eval_metrics reads the schema

    def test_arm_frame_never_reaches_the_model(self):                            # R3 with the ResNet
        insp = L.make_inspector("resnet", device="cpu", quiet=True)
        s, _ = session("each", inspector=insp)
        with unittest.mock.patch.object(insp, "probabilities", side_effect=AssertionError("model called")):
            r = s.process_frame(name(0), fixture(filled=range(1, 6), arm=True))
        s.close()
        self.assertEqual(r["status"], "arm")


@unittest.skipUnless(DEV, "needs the DEV cap_seal dataset")
class DevReplay(unittest.TestCase):
    """DEV runs 1-3 replayed frame by frame (in memory; same Session.process_frame as the file sources)."""
    EVENTS = {"1": "video", "2": "stills", "3": "stills"}

    @classmethod
    def setUpClass(cls):
        ix = R.Index(); cls.res, cls.resnet = {}, {}
        insp = L.make_inspector("resnet", quiet=True) if HAVE_MODEL else None    # fed the same frames, after G3 only
        for run, src in cls.EVENTS.items():
            s, out = session("events", ix=ix)
            sr = session("events", ix=ix, inspector=insp) if insp else None
            if src == "video":
                stream = ((R.video_id(run, t), t, img) for t, img in R.iter_video(ix.video_path(run), 1.0))
            else:
                fr = ix.frames(run); t0 = ix.time_of[fr[0]]
                stream = ((f, (ix.time_of[f] - t0).total_seconds(), ix.load(f)) for f in fr)
            rows = []
            for n, t, img in stream:
                rows.append(s.process_frame(n, img, t))
                if sr:
                    sr[0].process_frame(n, img, t)
            s.close(); cls.res[run] = (rows, out)
            if sr:
                sr[0].close(); cls.resnet[run] = sr[1]

    def test_every_frame_in_order_events_match_the_labeled_events(self):         # R1, R3
        for run, src in self.EVENTS.items():
            rows, _ = self.res[run]
            t = [float(r["frame_t"]) for r in rows]
            self.assertEqual(t, sorted(t))
            stored = [r["frame"] for r in read_csv(os.path.join(P.HERE, "results", "prototype", "events", f"events_run{run}_{src}.csv"))]
            self.assertEqual([r["event_frame"] for r in rows if r["status"] == "scored"], stored, run)

    def test_no_arm_frame_produces_a_hole_result(self):                          # R3
        for run in self.EVENTS:
            rows, _ = self.res[run]
            arm = [r for r in rows if r["arm_px"]]
            self.assertTrue(arm or run != "1")                                  # run 1 has arm frames
            self.assertTrue(all(r["status"] == "arm" and not r.get("pred_G01") for r in arm))

    def test_run1_matches_offline_inspect_frame(self):
        rows, out = self.res["1"]; ix = R.Index(); pm, insp = P.PositionMap(), RuleInspector()
        preds = read_csv(os.path.join(out, "predictions.csv"))
        for r in (r for r in rows if r["status"] == "scored"):
            ref = inspect(IF.load(ix, r["event_frame"]), r["event_frame"], pm, insp)   # inspect_frame's cached event frames
            live = {p["position"]: p["final_state"] for p in preds if p["frame"] == r["event_frame"]}
            self.assertEqual(live, {p["position"]: p["final_state"] for p in ref["positions"]}, r["event_frame"])

    @unittest.skipUnless(HAVE_MODEL and os.path.exists(G3_D038), "needs the ResNet and the G3 result")
    def test_resnet_live_replay_equals_the_g3_run(self):
        """Live (online events, frames decoded in memory) == g3_check.py (stored event frames): same frames, states."""
        g3 = {(r["frame"], r["position"]): r for r in read_csv(G3_D038)}
        for run, out in self.resnet.items():
            live = read_csv(os.path.join(out, "predictions.csv"))
            self.assertEqual({p["frame"] for p in live}, {k[0] for k, r in g3.items() if r["run"] == run}, run)
            for p in live:
                ref = g3[(p["frame"], p["position"])]
                self.assertEqual((p["predicted_state"], p["final_state"]), (ref["predicted_state"], ref["final_state"]))
                self.assertAlmostEqual(float(p["defect_score"]), float(ref["defect_score"]), places=3)


if __name__ == "__main__":
    unittest.main()
