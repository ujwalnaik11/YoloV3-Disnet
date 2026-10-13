"""Tests for the temporal (GRU) DisNet: tracker helpers, TemporalDisNet behaviour and the KITTI-tracking trainer.

Run from the repository root:   python -m unittest discover -s tests -v

The trainer is exercised on synthetic KITTI-tracking label files made with a pinhole camera model, so no
dataset is needed.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch  # noqa: F401
    from utils.temporal import GRUBox, GRUDistance, TemporalDisNet, load_temporal, save_temporal, stride_for_fps
    import train_disnet_rnn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from utils.disnet import DisNet  # noqa: E402
from utils.tracker import greedy_match, iou_matrix  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "generated_files", "disnet_model_3.npz")
W, H, FOCAL = 1242, 375, 720.0


def car_box(x, z):
    """Pinhole-camera box (x1, y1, x2, y2) of a 1.8 m wide, 1.6 m tall car at lateral offset x, depth z."""
    cx, cy = W / 2.0, H / 2.0
    return (cx + FOCAL * (x - 0.9) / z, cy + FOCAL * (1.65 - 1.6) / z, cx + FOCAL * (x + 0.9) / z, cy + FOCAL * 1.65 / z)


def write_sequences(directory, n_seq=8, n_frames=40, seed=0):
    rng = np.random.RandomState(seed)
    for s in range(n_seq):
        lines = []
        for track in range(3):
            x, z0, v = rng.uniform(-4, 4), rng.uniform(10, 40), rng.uniform(-0.4, 0.4)
            for frame in range(n_frames):
                z = z0 + v * frame
                x1, y1, x2, y2 = car_box(x, z)
                lines.append("%d %d Car 0 0 0 %.2f %.2f %.2f %.2f 1.6 1.8 4.0 %.2f 0.85 %.2f 0" % (frame, track, x1, y1, x2, y2, x, z))
        lines.append("0 -1 DontCare -1 -1 -10 10 10 20 20 -1 -1 -1 -1000 -1000 -1000 -10")
        with open(os.path.join(directory, "%04d.txt" % s), "w") as f:
            f.write("\n".join(lines) + "\n")


class TrackerHelperTests(unittest.TestCase):
    def test_iou(self):
        m = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
        np.testing.assert_allclose(m[0], [1.0, 1.0 / 3.0, 0.0], atol=1e-9)

    def test_greedy_match_is_one_to_one_and_respects_mask(self):
        iou = np.array([[0.9, 0.8], [0.7, 0.1]])
        matches, lost, fresh = greedy_match(iou, 0.3)
        self.assertEqual(sorted(matches), [(0, 0)])  # (1, 0) is blocked because column 0 is taken, (1, 1) < thresh
        self.assertEqual((lost, fresh), ([1], [1]))
        matches, _, _ = greedy_match(iou, 0.3, allowed=np.array([[False, True], [True, True]]))
        self.assertEqual(sorted(matches), [(0, 1), (1, 0)])

    def test_stride_for_fps(self):
        self.assertEqual([stride_for_fps(f) for f in (10, 15, 24, 30, 60)], [1, 2, 2, 3, 6])


@unittest.skipUnless(HAVE_TORCH and os.path.exists(MODEL_PATH), "torch or the pretrained model is missing")
class TemporalDisNetTests(unittest.TestCase):
    def make(self, **kw):
        torch.manual_seed(0)
        return TemporalDisNet(GRUDistance(), GRUBox(), fallback=DisNet.load(MODEL_PATH), **kw)

    def test_fallback_then_gru_and_stable_ids(self):
        tdn = self.make()
        sources, ids = [], set()
        for frame in range(5):
            out = tdn.update([("car", car_box(0.0, 30.0 - frame))], W, H)
            self.assertEqual(len(out), 1)
            sources.append(out[0]["source"])
            ids.add(out[0]["id"])
        self.assertEqual(sources, ["fallback", "fallback", "gru", "gru", "gru"])  # a full window needs 3 frames
        self.assertEqual(len(ids), 1)

    def test_stride_needs_longer_history(self):
        tdn = self.make(stride=2)
        sources = [tdn.update([("car", car_box(0.0, 30.0))], W, H)[0]["source"] for _ in range(6)]
        self.assertEqual(sources, ["fallback"] * 4 + ["gru"] * 2)  # 2 * stride + 1 = 5 frames

    def test_missed_detections_are_coasted_then_dropped(self):
        tdn = self.make(max_age=3)
        for frame in range(4):
            tdn.update([("car", car_box(0.0, 30.0 - frame))], W, H)
        observed = []
        for _ in range(5):
            out = tdn.update([], W, H)
            observed.append((len(out), out[0]["observed"] if out else None))
        self.assertEqual(observed, [(1, False)] * 3 + [(0, None)] * 2)

    def test_track_resumes_after_a_gap(self):
        tdn = self.make(max_age=3)
        for frame in range(4):
            first = tdn.update([("car", car_box(0.0, 30.0 - 0.3 * frame))], W, H)[0]["id"]
        tdn.update([], W, H)
        tdn.update([], W, H)
        out = tdn.update([("car", car_box(0.0, 30.0 - 0.3 * 7))], W, H)
        self.assertEqual([o["id"] for o in out], [first])
        self.assertTrue(out[0]["observed"])

    def test_two_objects_and_class_without_size(self):
        tdn = self.make()
        for frame in range(4):
            out = tdn.update([("car", car_box(-3.0, 20.0)), ("car", car_box(3.0, 40.0)),
                              ("traffic light", (600, 20, 620, 60))], W, H)
        self.assertEqual(len(out), 3)
        by_name = sorted(out, key=lambda o: o["name"])
        self.assertIsNone(by_name[2]["distance"])  # no known size -> no distance, but still tracked
        self.assertTrue(all(o["distance"] is not None for o in by_name[:2]))
        self.assertEqual(len({o["id"] for o in out}), 3)

    def test_save_load_round_trip(self):
        torch.manual_seed(1)
        dist, box = GRUDistance(), GRUBox()
        x = torch.rand(4, 3, 6) * 50
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            save_temporal(path, dist, box, meta={"a": 1})
            dist2, box2, meta = load_temporal(path)
        np.testing.assert_allclose(dist(x).detach().numpy(), dist2(x).detach().numpy(), atol=1e-6)
        self.assertIsNotNone(box2)
        self.assertEqual(meta, {"a": 1})


@unittest.skipUnless(HAVE_TORCH and os.path.exists(MODEL_PATH), "torch or the pretrained model is missing")
class VideoIntegrationTests(unittest.TestCase):
    def test_annotate_tracked_draws_and_returns_tracker_results(self):
        try:
            import video
        except ImportError as e:
            self.skipTest("video.py dependencies missing: %s" % e)
        torch.manual_seed(0)
        tracker = TemporalDisNet(GRUDistance(), GRUBox(), fallback=DisNet.load(MODEL_PATH))
        det = torch.tensor([[150., 100., 200., 300., 0.9, 0.9, 0.]])
        for _ in range(4):
            frame = np.zeros((480, 640, 3), np.uint8)
            out = video.annotate_tracked(frame, det, ["person"], 416, tracker)
        self.assertEqual((len(out), out[0]["name"], out[0]["source"]), (1, "person", "gru"))
        self.assertGreater(frame.sum(), 0)
        frame = np.zeros((480, 640, 3), np.uint8)  # detector misses the object: still drawn, flagged as predicted
        out = video.annotate_tracked(frame, None, ["person"], 416, tracker)
        self.assertEqual((len(out), out[0]["observed"]), (1, False))
        self.assertGreater(frame.sum(), 0)


@unittest.skipUnless(HAVE_TORCH, "torch is missing")
class TrainerTests(unittest.TestCase):
    def test_trainer_learns_distance_on_synthetic_sequences(self):
        with tempfile.TemporaryDirectory() as d:
            labels = os.path.join(d, "label_02")
            os.makedirs(labels)
            write_sequences(labels)
            report = train_disnet_rnn.main(["--kitti_tracking_labels", labels, "--epochs", "30", "--batch_size", "64", "--n_val", "2",
                                            "--n_test", "2", "--out", os.path.join(d, "gru.pt")])
            self.assertTrue(os.path.exists(os.path.join(d, "gru.pt")))
            dist, box, meta = load_temporal(os.path.join(d, "gru.pt"))
            self.assertIsNotNone(box)
            self.assertEqual(meta["img_width"], 1242)
        # noise-free pinhole data: both models must learn it far better than guessing the mean (about 8 m)
        self.assertLess(report["gru"]["mae"], 2.5)
        self.assertLess(report["feedforward"]["mae"], 2.5)
        # constant-velocity motion: extrapolating beats leaving the box where it is
        self.assertLess(report["box"]["constant velocity"], report["box"]["box stays where it is"])

    def test_parser_skips_dontcare_and_unsupported_classes(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "0000.txt"), "w") as f:
                f.write("0 -1 DontCare -1 -1 -10 1 1 2 2 -1 -1 -1 -1000 -1000 -1000 -10\n"
                        "0 1 Van 0 0 0 1 1 20 20 2 2 4 0 1 20 0\n"
                        "0 2 Pedestrian 0 0 0 1 1 20 60 1.7 0.6 0.8 0 1 15 0\n")
            seqs = train_disnet_rnn.parse_tracking_labels(d)
        self.assertEqual(list(seqs["0000"]), [2])
        name, box, dist, _, _ = seqs["0000"][2][0]
        self.assertEqual(name, "Pedestrian")
        self.assertAlmostEqual(dist, float(np.sqrt(0 + 1 + 225)), places=5)


if __name__ == "__main__":
    unittest.main()
