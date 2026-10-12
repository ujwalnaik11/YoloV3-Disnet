"""Train a DisNet distance-estimation model (Haseeb et al., 2018) and export it for video.py.

The network maps the six features [1/Bw, 1/Bh, 1/Bd, Ch, Cw, Cb] (see utils/disnet.py) to a distance in metres:
Dense(100, selu) x N -> Dense(1), trained with the L1 loss and Adam, on an 80 / 10 / 10 train / val / test split.
Inputs are standardised for training and the standardisation is folded into the first layer afterwards, so the
exported model consumes the raw features, exactly like the authors' model.

Data sources (use one or both):

  KITTI object labels (distance comes from the 3D location in the label file):
      python train_disnet.py --kitti_labels /path/to/training/label_2 --kitti_images /path/to/training/image_2

  Authors' annotation json files (dict of {"width", "height", "size_h", "size_w", "size_d", "distance", ...},
  where width / height are the box size in pixels and distance is in metres):
      python train_disnet.py --annotations a.json b.json --image_width 2592 --image_height 1944

Because the features are relative to the image size but not to the focal length, a model is most accurate on the
camera it was trained on. Train on data from your own camera for best results.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils.disnet import CLASS_SIZES_CM, DisNet, box_features, canonical_class


# ---------------------------------------------------------------------------------------------- data
def load_kitti(label_dir, image_dir, default_size, distance_kind, max_truncated, max_occluded):
    """Features / distances from raw KITTI label files."""
    X, y, skipped = [], [], 0
    for path in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        img_w, img_h = default_size
        if image_dir:
            for ext in (".png", ".jpg", ".jpeg"):
                img_path = os.path.join(image_dir, stem + ext)
                if os.path.exists(img_path):
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                    break
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) < 15 or canonical_class(p[0]) is None:
                    continue
                truncated, occluded = float(p[1]), int(float(p[2]))
                if truncated > max_truncated or occluded > max_occluded:
                    skipped += 1
                    continue
                x1, y1, x2, y2 = (float(v) for v in p[4:8])
                loc = np.array([float(v) for v in p[11:14]])  # x, y, z in camera coordinates (m)
                dist = float(np.linalg.norm(loc)) if distance_kind == "range" else float(loc[2])
                feats = box_features(x2 - x1, y2 - y1, img_w, img_h, p[0])
                if feats is not None and dist > 0:
                    X.append(feats)
                    y.append(dist)
    print("KITTI: %d objects (%d skipped for truncation/occlusion)" % (len(X), skipped))
    return X, y


def load_annotations(paths, img_w, img_h):
    """Features / distances from the authors' annotation json format."""
    X, y = [], []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for d in data.values():
            bw, bh = float(d["width"]) / img_w, float(d["height"]) / img_h
            bd = float(np.sqrt(bw ** 2 + bh ** 2))
            X.append(np.array([1 / bw, 1 / bh, 1 / bd, float(d["size_h"]), float(d["size_w"]), float(d["size_d"])],
                              dtype=np.float32))
            y.append(float(d["distance"]))
    print("Annotations: %d objects" % len(X))
    return X, y


# -------------------------------------------------------------------------------------------- training
def build_net(hidden_layers, width):
    layers, n_in = [], 6
    for _ in range(hidden_layers):
        layers += [nn.Linear(n_in, width), nn.SELU()]
        n_in = width
    layers.append(nn.Linear(n_in, 1))
    return nn.Sequential(*layers)


def export(net, mean, std):
    """Fold the input standardisation into the first Linear layer and wrap the weights as a DisNet."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    out = []
    for i, lin in enumerate(linears):
        w = lin.weight.detach().numpy().T.astype(np.float64)  # (in, out)
        b = lin.bias.detach().numpy().astype(np.float64)
        if i == 0:
            b = b - (mean / std) @ w
            w = w / std[:, None]
        out.append((w, b, "selu" if i < len(linears) - 1 else "linear"))
    return DisNet(out)


def metrics(model, X, y):
    pred = model.predict(X)
    return float(np.mean(np.abs(pred - y))), float(np.mean(np.abs(pred - y) / y) * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti_labels", help="directory with raw KITTI label .txt files")
    ap.add_argument("--kitti_images", help="directory with the matching KITTI images (image sizes are read from them)")
    ap.add_argument("--kitti_distance", choices=["range", "depth"], default="range",
                    help="range = Euclidean camera-object distance, depth = z along the optical axis")
    ap.add_argument("--max_truncated", type=float, default=0.3, help="skip KITTI objects truncated more than this")
    ap.add_argument("--max_occluded", type=int, default=1, help="skip KITTI objects with occlusion level above this (0-3)")
    ap.add_argument("--annotations", nargs="*", default=[], help="annotation json files in the authors' format")
    ap.add_argument("--image_width", type=int, default=1242, help="image width for --annotations / KITTI without --kitti_images")
    ap.add_argument("--image_height", type=int, default=375, help="image height for --annotations / KITTI without --kitti_images")
    ap.add_argument("--hidden_layers", type=int, default=3, help="hidden layers (the paper uses 3)")
    ap.add_argument("--hidden_units", type=int, default=100, help="neurons per hidden layer (the paper uses 100)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="generated_files/disnet_trained.npz", help="where to save the model (.npz)")
    args = ap.parse_args()

    X, y = [], []
    if args.kitti_labels:
        xs, ys = load_kitti(args.kitti_labels, args.kitti_images, (args.image_width, args.image_height),
                            args.kitti_distance, args.max_truncated, args.max_occluded)
        X += xs
        y += ys
    if args.annotations:
        xs, ys = load_annotations(args.annotations, args.image_width, args.image_height)
        X += xs
        y += ys
    if len(X) < 20:
        raise SystemExit("Need at least 20 labelled objects to train, found %d" % len(X))
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # 80 / 10 / 10 split
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X))
    n_train, n_val = int(0.8 * len(X)), int(0.1 * len(X))
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    print("train / val / test: %d / %d / %d" % (len(tr), len(va), len(te)))

    mean = X[tr].mean(axis=0).astype(np.float64)
    std = X[tr].std(axis=0).astype(np.float64)
    std[std < 1e-8] = 1.0  # constant feature (e.g. a single class): leave unscaled

    torch.manual_seed(args.seed)
    net = build_net(args.hidden_layers, args.hidden_units)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    def tensor(a, rows):
        return torch.from_numpy(((a[rows] - mean) / std).astype(np.float32))

    Xtr, ytr = tensor(X, tr), torch.from_numpy(y[tr]).unsqueeze(1)
    Xva, yva = tensor(X, va), torch.from_numpy(y[va]).unsqueeze(1)

    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss_fn(net(Xtr[b]), ytr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xva), yva).item() if len(va) else loss_fn(net(Xtr), ytr).item()
        if val < best_val:
            best_val, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print("epoch %4d  val MAE %.3f m  (best %.3f)" % (epoch + 1, val, best_val))

    net.load_state_dict(best_state)
    model = export(net, mean, std)  # evaluated below with the exported NumPy model, i.e. what video.py will run

    for name, rows in (("train", tr), ("val", va), ("test", te)):
        if len(rows):
            mae, mape = metrics(model, X[rows], y[rows])
            print("%-5s MAE %.3f m, mean relative error %.1f %%" % (name, mae, mape))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.save_npz(args.out)
    print("Saved %s  ->  python video.py --disnet_model %s" % (args.out, args.out))


if __name__ == "__main__":
    main()
