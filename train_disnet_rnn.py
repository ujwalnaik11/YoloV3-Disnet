"""Train the temporal (GRU) DisNet on KITTI tracking labels and compare it with a feed-forward DisNet.

    python train_disnet_rnn.py --kitti_tracking_labels /path/to/training/label_02

Needs only the label files of the KITTI tracking benchmark (data_tracking_label_2.zip, no images): every line
has a track id, a box and the 3D location, so the distance is the length of the location vector. For every
tracked object, windows of 3 consecutive frames are built:

  * distance model (utils.temporal.GRUDistance): DisNet features of the 3 frames -> distance of each frame
  * box model (utils.temporal.GRUBox): boxes of 3 frames -> box in the 4th frame

Sequences (not individual windows) are split into train / val / test, because frames of one drive are almost
identical. The report compares, on the same test windows,

  * the GRU distance model,
  * a feed-forward DisNet (6 -> 100 -> 100 -> 100 -> 1, SELU, L1) trained on the same train sequences, and
  * (optional, --ff_model) an existing DisNet .npz,

and for the box model against "box does not move" and "constant velocity".

Image size is assumed constant (KITTI tracking images are about 1242 x 375). The tracking sequences are recorded
at 10 frames per second; see utils.temporal.stride_for_fps for video at other frame rates.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

from utils.disnet import DisNet, box_features, canonical_class
from utils.temporal import GRUBox, GRUDistance, save_temporal


# ---------------------------------------------------------------------------------------------- data
def parse_tracking_labels(label_dir):
    """{sequence: {track_id: {frame: (class, (x1, y1, x2, y2), distance_m, truncated, occluded)}}}"""
    seqs = {}
    for path in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        tracks = {}
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) < 17:
                    continue
                track_id = int(p[1])
                if track_id < 0 or canonical_class(p[2]) is None:  # DontCare, Van, Tram, ...
                    continue
                loc = np.array([float(v) for v in p[13:16]])
                tracks.setdefault(track_id, {})[int(p[0])] = (
                    p[2], tuple(float(v) for v in p[6:10]), float(np.linalg.norm(loc)), float(p[3]), int(float(p[4])))
        seqs[os.path.splitext(os.path.basename(path))[0]] = tracks
    return seqs


def build_windows(tracks, img_w, img_h, max_truncated, max_occluded):
    """Distance windows (3 consecutive frames, all well visible) and box windows (4 consecutive frames)."""
    dist_x, dist_y, box_x, box_y = [], [], [], []
    for frames in tracks.values():
        for f in sorted(frames):
            if all(f + k in frames for k in range(3)):
                win = [frames[f + k] for k in range(3)]
                if all(w[3] <= max_truncated and w[4] <= max_occluded and w[2] > 0 for w in win):
                    feats = [box_features(w[1][2] - w[1][0], w[1][3] - w[1][1], img_w, img_h, w[0]) for w in win]
                    if all(x is not None for x in feats):
                        dist_x.append(np.stack(feats))
                        dist_y.append([w[2] for w in win])
            if all(f + k in frames for k in range(4)):
                boxes = np.array([frames[f + k][1] for k in range(4)], np.float32) / [img_w, img_h, img_w, img_h]
                if all(b[2] > b[0] and b[3] > b[1] for b in boxes):
                    box_x.append(boxes[:3])
                    box_y.append(boxes[3])
    return (np.array(dist_x, np.float32).reshape(-1, 3, 6), np.array(dist_y, np.float32).reshape(-1, 3),
            np.array(box_x, np.float32).reshape(-1, 3, 4), np.array(box_y, np.float32).reshape(-1, 4))


def collect(seqs, ids, args):
    parts = [build_windows(seqs[s], args.img_width, args.img_height, args.max_truncated, args.max_occluded)
             for s in ids]
    return tuple(np.concatenate([p[i] for p in parts]) if parts else np.zeros((0,)) for i in range(4))


# ---------------------------------------------------------------------------------------------- training
class FeedForward(nn.Module):
    """Baseline: the DisNet network (3 x 100 SELU) on single frames, with built-in standardisation."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 100), nn.SELU(), nn.Linear(100, 100), nn.SELU(),
                                 nn.Linear(100, 100), nn.SELU(), nn.Linear(100, 1))
        self.register_buffer("mean", torch.zeros(6))
        self.register_buffer("std", torch.ones(6))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)


def fit(model, Xtr, Ytr, Xva, Yva, epochs, lr, batch, device, label):
    """L1 loss + Adam; keeps the weights of the best validation epoch."""
    Xtr, Ytr = torch.from_numpy(Xtr).to(device), torch.from_numpy(Ytr).to(device)
    Xva, Yva = torch.from_numpy(Xva).to(device), torch.from_numpy(Yva).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state = float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            loss = (model(Xtr[idx]) - Ytr[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = (model(Xva) - Yva).abs().mean().item()
        if val < best:
            best, state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print("%s epoch %3d  val L1 %.4f  (best %.4f)" % (label, epoch, val, best))
    model.load_state_dict(state)
    return model.eval()


def distance_report(name, pred, true):
    err = np.abs(pred - true)
    rel = err / true
    row = dict(mae=float(err.mean()), mean_rel=float(rel.mean()), median_rel=float(np.median(rel)))
    print("  %-34s MAE %.2f m   mean rel. error %.1f %%   median rel. error %.1f %%"
          % (name, row["mae"], 100 * row["mean_rel"], 100 * row["median_rel"]))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti_tracking_labels", required=True, help="directory with the label_02/*.txt files")
    ap.add_argument("--img_width", type=int, default=1242)
    ap.add_argument("--img_height", type=int, default=375)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_val", type=int, default=3, help="number of sequences used for validation")
    ap.add_argument("--n_test", type=int, default=3, help="number of sequences used for the final test")
    ap.add_argument("--max_truncated", type=float, default=0.3)
    ap.add_argument("--max_occluded", type=int, default=1)
    ap.add_argument("--ff_model", default=None, help="optional feed-forward DisNet .npz to evaluate as well")
    ap.add_argument("--out", default="generated_files/disnet_gru.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seqs = parse_tracking_labels(args.kitti_tracking_labels)
    assert len(seqs) >= args.n_val + args.n_test + 1, "need more sequences than n_val + n_test"
    order = [list(seqs)[i] for i in np.random.permutation(len(seqs))]
    test_ids, val_ids = order[:args.n_test], order[args.n_test:args.n_test + args.n_val]
    train_ids = order[args.n_test + args.n_val:]
    print("sequences: %d train / %d val / %d test  (test: %s)" % (len(train_ids), len(val_ids), len(test_ids),
                                                                  ", ".join(sorted(test_ids))))
    tr, va, te = (collect(seqs, ids, args) for ids in (train_ids, val_ids, test_ids))
    print("distance windows: %d train / %d val / %d test; box windows: %d / %d / %d"
          % (len(tr[0]), len(va[0]), len(te[0]), len(tr[2]), len(va[2]), len(te[2])))
    assert len(tr[0]) and len(va[0]) and len(te[0]), "no usable windows (check the label directory and filters)"

    # ---- distance: GRU
    flat = tr[0].reshape(-1, 6)
    gru = GRUDistance()
    gru.mean.copy_(torch.from_numpy(flat.mean(0)))
    gru.std.copy_(torch.from_numpy(flat.std(0) + 1e-6))
    nn.init.constant_(gru.head.bias, float(tr[1].mean()))  # start from the average distance, not from 0 m
    gru = fit(gru.to(device), tr[0], tr[1], va[0], va[1], args.epochs, args.lr, args.batch_size, device, "GRU distance")

    # ---- distance: feed-forward baseline on the same frames
    ff = FeedForward()
    ff.mean.copy_(torch.from_numpy(flat.mean(0)))
    ff.std.copy_(torch.from_numpy(flat.std(0) + 1e-6))
    ff = fit(ff.to(device), flat, tr[1].reshape(-1), va[0].reshape(-1, 6), va[1].reshape(-1), args.epochs, args.lr,
             args.batch_size, device, "feed-forward")

    # ---- box model
    box = None
    if len(tr[2]) and len(va[2]) and len(te[2]):
        box = GRUBox().to(device)
        box = fit(box, tr[2], tr[3], va[2], va[3], args.epochs, args.lr, args.batch_size, device, "GRU box")

    # ---- report on the test sequences (last frame of every window)
    print("\nDistance on %d test windows (last frame of each window):" % len(te[0]))
    truth = te[1][:, -1]
    with torch.no_grad():
        gru_pred = gru(torch.from_numpy(te[0]).to(device))[:, -1].cpu().numpy()
        ff_pred = ff(torch.from_numpy(te[0][:, -1]).to(device)).cpu().numpy()
    report = {"gru": distance_report("GRU, 3-frame window", gru_pred, truth),
              "feedforward": distance_report("feed-forward, single frame", ff_pred, truth)}
    if args.ff_model:
        report["ff_model"] = distance_report("feed-forward (%s)" % os.path.basename(args.ff_model),
                                             DisNet.load(args.ff_model).predict(te[0][:, -1]), truth)

    if box is not None:
        scale = np.array([args.img_width, args.img_height] * 2, np.float32)
        with torch.no_grad():
            box_pred = box(torch.from_numpy(te[2]).to(device)).cpu().numpy()
        hold, velocity = te[2][:, -1], 2 * te[2][:, -1] - te[2][:, -2]
        print("\nNext-box prediction on %d test windows (mean abs. coordinate error, pixels):" % len(te[2]))
        report["box"] = {}
        for name, pred in (("box stays where it is", hold), ("constant velocity", velocity), ("GRU box model", box_pred)):
            report["box"][name] = float(np.abs((pred - te[3]) * scale).mean())
            print("  %-34s %.2f px" % (name, report["box"][name]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_temporal(args.out, gru.cpu(), None if box is None else box.cpu(),
                  meta=dict(img_width=args.img_width, img_height=args.img_height, train_fps=10.0,
                            train_sequences=sorted(train_ids), test_sequences=sorted(test_ids)))
    print("\nSaved %s" % args.out)
    return report


if __name__ == "__main__":
    main()
