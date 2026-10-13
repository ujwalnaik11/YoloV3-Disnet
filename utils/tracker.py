"""Small IoU helpers for associating detections with tracked objects across video frames."""
import numpy as np


def iou_matrix(a, b):
    """Pairwise IoU between boxes `a` (N, 4) and `b` (M, 4), both (x1, y1, x2, y2). Returns (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def greedy_match(iou, thresh, allowed=None):
    """Greedy one-to-one matching, highest IoU first.

    iou: (N, M) matrix; `allowed`: optional boolean (N, M) mask of pairs that may be matched.
    Returns (matches [(i, j), ...], unmatched rows, unmatched columns).
    """
    n, m = iou.shape
    candidates = [(iou[i, j], i, j) for i in range(n) for j in range(m)
                  if iou[i, j] >= thresh and (allowed is None or allowed[i, j])]
    candidates.sort(key=lambda c: -c[0])
    used_i, used_j, matches = set(), set(), []
    for _, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matches.append((i, j))
    return matches, [i for i in range(n) if i not in used_i], [j for j in range(m) if j not in used_j]
