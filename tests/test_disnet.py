"""Regression tests for the DisNet distance estimation.

Run from the repository root:   python -m unittest discover -s tests -v

tests/data/disnet_labelled_samples.csv holds 1,795 real labelled boxes (class, box size in pixels, laser-measured
distance in metres) taken from the authors' data-preparation notebook (github.com/guanjianyu/DisNet, MIT licence).
The images were 2592 x 1944 pixels.
"""
import csv
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.disnet import CLASS_SIZES_CM, DisNet, box_features, canonical_class  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "generated_files", "disnet_model_3.npz")
SAMPLES = os.path.join(ROOT, "tests", "data", "disnet_labelled_samples.csv")
IMG_W, IMG_H = 2592, 1944


def load_samples():
    with open(SAMPLES) as f:
        return list(csv.DictReader(f))


class FeatureTests(unittest.TestCase):
    def test_feature_vector_matches_paper_definition(self):
        # box of 10% x 10% of the image -> Bw = Bh = 0.1, Bd = sqrt(0.02) (diagonal of the *normalised* box)
        f = box_features(0.1 * IMG_W, 0.1 * IMG_H, IMG_W, IMG_H, "person")
        np.testing.assert_allclose(f, [10.0, 10.0, 1 / np.sqrt(0.02), 175, 55, 30], rtol=1e-5)

    def test_class_sizes_and_aliases(self):
        self.assertEqual(CLASS_SIZES_CM["car"], (160, 180, 400))
        self.assertEqual(canonical_class("Pedestrian"), "person")
        self.assertEqual(canonical_class("Cyclist"), "bicycle")
        self.assertIsNone(canonical_class("traffic light"))

    def test_unsupported_class_and_degenerate_box(self):
        self.assertIsNone(box_features(50, 100, 640, 480, "traffic light"))
        self.assertIsNone(box_features(0, 100, 640, 480, "person"))
        self.assertIsNone(box_features(50, -3, 640, 480, "person"))

    def test_features_are_resolution_independent(self):
        a = box_features(64, 192, 640, 480, "person")
        b = box_features(128, 384, 1280, 960, "person")  # same box on a 2x larger image
        np.testing.assert_allclose(a, b, rtol=1e-5)


@unittest.skipUnless(os.path.exists(MODEL_PATH), "pretrained model not found")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DisNet.load(MODEL_PATH)

    def test_architecture(self):
        shapes = [w.shape for w, _, _ in self.model.layers]
        self.assertEqual(shapes[0][0], 6)
        self.assertEqual(shapes[-1][1], 1)
        self.assertEqual(self.model.layers[-1][2], "selu")

    def test_accuracy_on_real_labelled_samples(self):
        """The pipeline must reproduce measured distances (observed: MAE 2.53 m, median 1.20 m, r = 0.95)."""
        rows = load_samples()
        feats = np.array([box_features(int(r["box_width_px"]), int(r["box_height_px"]), IMG_W, IMG_H, r["class"])
                          for r in rows])
        truth = np.array([float(r["distance_m"]) for r in rows])
        pred = self.model.predict(feats)
        err = np.abs(pred - truth)
        self.assertLess(err.mean(), 3.0)
        self.assertLess(np.median(err), 1.6)
        self.assertGreater(np.corrcoef(pred, truth)[0, 1], 0.93)
        self.assertGreater(np.mean(err / truth < 0.10), 0.65)

    def test_feature_variants_that_were_wrong_are_worse(self):
        """Guards the feature definition: swapping the width/height order or using a diagonal normalised by the
        image diagonal (as the paper text literally says) is clearly worse on the real data."""
        rows = load_samples()
        truth = np.array([float(r["distance_m"]) for r in rows])

        def mae(make):
            X = np.array([make(int(r["box_width_px"]), int(r["box_height_px"]), r["class"]) for r in rows])
            return np.abs(self.model.predict(X) - truth).mean()

        def good(w, h, c):
            return box_features(w, h, IMG_W, IMG_H, c)

        def swapped(w, h, c):
            f = good(w, h, c).copy()
            f[[0, 1]] = f[[1, 0]]
            return f

        def image_diag(w, h, c):
            f = good(w, h, c).copy()
            f[2] = 1.0 / (np.hypot(w, h) / np.hypot(IMG_W, IMG_H))
            return f

        base = mae(good)
        self.assertGreater(mae(swapped), 1.5 * base)
        self.assertGreater(mae(image_diag), 1.5 * base)

    def test_distance_decreases_as_box_grows(self):
        heights = np.linspace(40, 1400, 25)
        d = [self.model.estimate((0, 0, h * 0.32, h), "person", IMG_W, IMG_H) for h in heights]
        self.assertTrue(all(a > b for a, b in zip(d, d[1:])), "distance should fall monotonically as the box grows")
        self.assertTrue(all(x > 0 for x in d))

    def test_estimate_unsupported_class(self):
        self.assertIsNone(self.model.estimate((0, 0, 50, 100), "traffic light", 640, 480))

    def test_matches_independent_torch_forward_pass(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        rng = np.random.RandomState(0)
        X = np.column_stack([rng.uniform(3, 120, 200), rng.uniform(3, 120, 200), rng.uniform(3, 120, 200),
                             rng.choice([175, 160, 110], 200), rng.choice([55, 180, 50], 200),
                             rng.choice([30, 400, 180], 200)]).astype(np.float32)
        t = torch.from_numpy(X)
        for w, b, act in self.model.layers:
            t = t @ torch.from_numpy(w) + torch.from_numpy(b)
            t = torch.selu(t) if act == "selu" else t
        np.testing.assert_allclose(t[:, 0].numpy(), self.model.predict(X), atol=1e-3, rtol=1e-4)

    def test_npz_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            self.model.save_npz(path)
            again = DisNet.load(path)
        x = np.array([[12, 30, 32, 175, 55, 30]], dtype=np.float32)
        np.testing.assert_array_equal(self.model.predict(x), again.predict(x))


@unittest.skipUnless(os.path.exists(MODEL_PATH), "pretrained model not found")
class VideoPipelineTests(unittest.TestCase):
    """video.py geometry: boxes must map from the letterboxed network input back to the original frame."""

    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
            import torch  # noqa: F401
            import video
        except ImportError as e:
            raise unittest.SkipTest("video.py dependencies missing: %s" % e)
        cls.video = video
        cls.model = DisNet.load(MODEL_PATH)

    def test_letterbox_box_round_trip(self):
        import torch  # noqa: F401
        for frame_w, frame_h in [(640, 480), (480, 640), (1280, 720), (416, 416)]:
            frame = np.zeros((frame_h, frame_w, 3), np.uint8)
            x1, y1, x2, y2 = int(frame_w * 0.3), int(frame_h * 0.2), int(frame_w * 0.6), int(frame_h * 0.9)
            frame[y1:y2, x1:x2] = 255
            img = self.video.letterbox(frame, 416)[0, 0].numpy()  # one channel, (416, 416)
            ys, xs = np.where(img > 0.9)  # the grey letterbox padding is 128/255 = 0.50, the box is 1.0
            box = self.video.network_box_to_frame((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
                                                  frame_w, frame_h, 416)
            tol = max(frame_w, frame_h) / 416.0 * 2  # about two network pixels
            np.testing.assert_allclose(box, (x1, y1, x2, y2), atol=tol, err_msg="frame %dx%d" % (frame_w, frame_h))

    def test_annotate_returns_disnet_distance_for_supported_classes_only(self):
        import torch
        frame = np.zeros((480, 640, 3), np.uint8)
        classes = ["person", "traffic light"]
        # network-space boxes for a 640x480 frame letterboxed to 416 (pad 52 px top/bottom in network units)
        det = torch.tensor([[150., 100., 200., 300., 0.9, 0.9, 0.],
                            [300., 100., 320., 140., 0.9, 0.9, 1.]])
        out = self.video.annotate(frame, det, classes, 416, self.model)
        (name0, box0, dist0), (name1, _, dist1) = out
        self.assertEqual(name0, "person")
        self.assertIsNone(dist1)  # traffic light: no known size
        expected = self.model.estimate(box0, "person", 640, 480)
        self.assertAlmostEqual(dist0, expected, places=5)
        self.assertGreater(dist0, 0)
        self.assertGreater(frame.sum(), 0)  # something was drawn


if __name__ == "__main__":
    unittest.main()
