"""Temporal DisNet: GRU distance estimation over tracked objects (PyTorch).

The feed-forward DisNet in utils/disnet.py estimates a distance from a single box. The authors' repository
(github.com/guanjianyu/DisNet, DisNetRNN*.py) also has a recurrent variant, re-implemented here in PyTorch:

  * GRUDistance: the DisNet features of the last 3 frames of one object -> stacked GRUs (128, 128, 64) -> a
    distance per frame (the last one is used).
  * GRUBox: the last 3 boxes of one object -> the box in the next frame. When the detector misses an object,
    the tracker keeps it alive with the predicted box for up to `max_age` frames.
  * TemporalDisNet: IoU tracker + the two models + a fallback to the feed-forward DisNet while a track is
    still too young for a full window.

Models are trained by train_disnet_rnn.py on KITTI tracking labels, which are recorded at 10 frames per second.
For video at a higher frame rate pass `stride=stride_for_fps(fps)` so that the 3 frames fed to the GRU are
about 0.1 s apart, as in training.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from utils.disnet import box_features, canonical_class
from utils.tracker import greedy_match, iou_matrix

WINDOW = 3
TRAIN_FPS = 10.0  # KITTI tracking sequences


class GRUDistance(nn.Module):
    """(B, T, 6) raw DisNet features -> (B, T) distance in metres. Standardisation is built in."""

    def __init__(self, hidden=(128, 128, 64)):
        super().__init__()
        self.hidden = tuple(hidden)
        sizes = [6] + list(self.hidden)
        self.grus = nn.ModuleList([nn.GRU(sizes[i], sizes[i + 1], batch_first=True) for i in range(len(self.hidden))])
        self.head = nn.Linear(self.hidden[-1], 1)
        self.register_buffer("mean", torch.zeros(6))
        self.register_buffer("std", torch.ones(6))

    def forward(self, x):
        x = (x - self.mean) / self.std
        for gru in self.grus:
            x, _ = gru(x)
        return self.head(x).squeeze(-1)


class GRUBox(nn.Module):
    """(B, 3, 4) boxes normalised by the image size -> (B, 4) box in the next frame (last box + learned shift)."""

    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(4, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 4)
        nn.init.zeros_(self.head.weight)  # an untrained model predicts "the box stays where it is"
        nn.init.zeros_(self.head.bias)

    def forward(self, boxes):
        h, _ = self.gru(boxes)
        return boxes[:, -1] + self.head(h[:, -1])


def stride_for_fps(fps, train_fps=TRAIN_FPS):
    """Frame spacing that makes consecutive window frames about 1 / train_fps seconds apart."""
    return max(1, int(round(float(fps) / train_fps)))


def save_temporal(path, distance_model, box_model=None, meta=None):
    torch.save({
        "distance_hidden": list(distance_model.hidden),
        "distance": distance_model.state_dict(),
        "box_hidden": None if box_model is None else box_model.hidden,
        "box": None if box_model is None else box_model.state_dict(),
        "meta": meta or {},
    }, path)


def load_temporal(path, device="cpu"):
    """Returns (GRUDistance, GRUBox or None, meta) in eval mode."""
    ck = torch.load(path, map_location=device)
    dist = GRUDistance(ck["distance_hidden"])
    dist.load_state_dict(ck["distance"])
    box = None
    if ck.get("box") is not None:
        box = GRUBox(ck["box_hidden"])
        box.load_state_dict(ck["box"])
    dist.to(device).eval()
    if box is not None:
        box.to(device).eval()
    return dist, box, ck.get("meta", {})


class _Track(object):
    def __init__(self, track_id, name, box, maxlen):
        self.id = track_id
        self.name = name
        self.key = canonical_class(name) or str(name).lower()
        self.hist = deque([(tuple(box), True)], maxlen=maxlen)  # (box, observed by the detector)
        self.misses = 0


class TemporalDisNet(object):
    """Tracks detections over frames and estimates their distances with the GRU model.

    update(detections, img_w, img_h) -> list of dicts, one per live track:
        id, name, box (x1, y1, x2, y2 pixels), distance (m or None), observed (False while coasting on a
        predicted box), source ("gru", "fallback" or None).
    """

    def __init__(self, distance_model, box_model=None, fallback=None, stride=1, max_age=3, iou_thresh=0.3,
                 device="cpu"):
        self.device = torch.device(device)
        self.distance_model = distance_model.to(self.device).eval()
        self.box_model = None if box_model is None else box_model.to(self.device).eval()
        self.fallback = fallback  # a utils.disnet.DisNet used until a track has a full 3-frame window
        self.stride = int(stride)
        self.max_age = int(max_age)
        self.iou_thresh = float(iou_thresh)
        self.reset()

    def reset(self):
        self.tracks = []
        self._next_id = 1

    # ------------------------------------------------------------------ helpers
    def _span(self):
        return 2 * self.stride + 1  # frames of history needed for a 3-frame window

    def _window_boxes(self, track):
        s = self.stride
        return [track.hist[i][0] for i in (-1 - 2 * s, -1 - s, -1)]

    def _predict_boxes(self, size):
        """Expected box of every track in the current frame (pixels), shape (N, 4)."""
        preds = np.array([t.hist[-1][0] for t in self.tracks], dtype=np.float32).reshape(-1, 4)
        if not self.tracks:
            return preds
        full = [k for k, t in enumerate(self.tracks) if len(t.hist) >= self._span()]
        done = set()
        if self.box_model is not None and full:
            x = np.stack([np.array(self._window_boxes(self.tracks[k]), np.float32) / size for k in full])
            with torch.no_grad():
                out = self.box_model(torch.from_numpy(x).to(self.device)).cpu().numpy()
            for n, (row, k) in enumerate(zip(out, full)):
                last = x[n][-1]
                preds[k] = (last + (row - last) / self.stride) * size  # the model predicts one training step ahead
                done.add(k)
        for k, t in enumerate(self.tracks):
            if k not in done and len(t.hist) >= 2:  # constant velocity
                last, prev = np.array(t.hist[-1][0], np.float32), np.array(t.hist[-2][0], np.float32)
                preds[k] = last + (last - prev)
        for k, t in enumerate(self.tracks):  # reject degenerate predictions
            if preds[k][2] - preds[k][0] < 2 or preds[k][3] - preds[k][1] < 2:
                preds[k] = np.array(t.hist[-1][0], np.float32)
        return preds

    # ------------------------------------------------------------------ main entry point
    def update(self, detections, img_w, img_h):
        """detections: list of (class_name, (x1, y1, x2, y2)) in image pixels for the current frame."""
        size = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        maxlen = self._span()
        preds = self._predict_boxes(size)
        det_boxes = np.array([d[1] for d in detections], dtype=np.float32).reshape(-1, 4)
        det_keys = [canonical_class(d[0]) or str(d[0]).lower() for d in detections]

        if self.tracks and len(det_boxes):
            iou = iou_matrix(preds, det_boxes)
            allowed = np.array([[t.key == k for k in det_keys] for t in self.tracks])
            matches, lost, fresh = greedy_match(iou, self.iou_thresh, allowed)
        else:
            matches, lost, fresh = [], list(range(len(self.tracks))), list(range(len(det_boxes)))

        for i, j in matches:
            t = self.tracks[i]
            t.hist.append((tuple(det_boxes[j]), True))
            t.misses = 0
        survivors = [self.tracks[i] for i, _ in sorted(matches)]
        for i in lost:
            t = self.tracks[i]
            t.misses += 1
            if t.misses <= self.max_age:
                t.hist.append((tuple(float(v) for v in preds[i]), False))  # coast on the predicted box
                survivors.append(t)
        for j in fresh:
            survivors.append(_Track(self._next_id, detections[j][0], det_boxes[j], maxlen))
            self._next_id += 1
        self.tracks = sorted(survivors, key=lambda t: t.id)
        return self._distances(img_w, img_h)

    def _distances(self, img_w, img_h):
        results = [dict(id=t.id, name=t.name, box=tuple(float(v) for v in t.hist[-1][0]), distance=None,
                        observed=t.hist[-1][1], source=None) for t in self.tracks]
        windows, rows = [], []
        for k, t in enumerate(self.tracks):
            cls = canonical_class(t.name)
            if cls is None:
                continue
            if len(t.hist) >= self._span():
                feats = [box_features(b[2] - b[0], b[3] - b[1], img_w, img_h, cls) for b in self._window_boxes(t)]
                if all(f is not None for f in feats):
                    windows.append(np.stack(feats))
                    rows.append(k)
                    continue
            if self.fallback is not None:
                d = self.fallback.estimate(t.hist[-1][0], cls, img_w, img_h)
                if d is not None:
                    results[k].update(distance=d, source="fallback")
        if windows:
            with torch.no_grad():
                out = self.distance_model(torch.from_numpy(np.stack(windows)).float().to(self.device))[:, -1]
            for k, d in zip(rows, out.cpu().numpy()):
                results[k].update(distance=max(float(d), 0.0), source="gru")
        return results
