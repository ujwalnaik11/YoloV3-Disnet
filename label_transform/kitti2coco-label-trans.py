"""Convert KITTI object-detection labels to YOLO / darknet format.

KITTI line:  <type> <trunc> <occ> <alpha> x1 y1 x2 y2 <h> <w> <l> <x> <y> <z> <ry> [score]
YOLO line:   <class_id> <cx> <cy> <w> <h>        (all normalised to 0..1)

Usage
-----
    python label_transform/kitti2coco-label-trans.py \
        --images  data/kitti/images/train \
        --labels  data/kitti/labels_raw/train \
        --out     data/kitti/labels/train \
        --names   data/kitti.names \
        --list    data/kitti/train.txt

Images and labels are paired by file name (e.g. 000000.png <-> 000000.txt),
objects of class 'DontCare' (or any class not listed in --names) are skipped.
`--list` writes the image list consumed by `config/*.data`; with `--relative`
(the default) the paths are written relative to the current directory so the
list works on any machine.
"""
import argparse
import os
import sys

from PIL import Image


def load_names(path):
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    return {name: i for i, name in enumerate(names)}


def convert_label_file(label_path, img_w, img_h, name_to_id):
    lines = []
    with open(label_path) as f:
        for raw in f:
            fields = raw.split()
            if len(fields) < 8:
                continue
            cls = fields[0]
            if cls not in name_to_id:  # 'DontCare' and unknown classes
                continue
            x1, y1, x2, y2 = (float(v) for v in fields[4:8])
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append("%d %.6f %.6f %.6f %.6f\n" % (name_to_id[cls], cx, cy, w, h))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory with KITTI images")
    ap.add_argument("--labels", required=True, help="directory with raw KITTI label .txt files")
    ap.add_argument("--out", required=True, help="directory to write YOLO label files to")
    ap.add_argument("--names", default="data/kitti.names", help="class names file (one per line)")
    ap.add_argument("--list", default=None, help="optional: write an image list file here")
    ap.add_argument("--absolute", action="store_true", help="write absolute paths in --list instead of relative ones")
    args = ap.parse_args()

    name_to_id = load_names(args.names)
    os.makedirs(args.out, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg")
    images = sorted(f for f in os.listdir(args.images) if f.lower().endswith(exts))
    if not images:
        sys.exit("No images found in %s" % args.images)

    written, missing = [], 0
    for fname in images:
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(args.labels, stem + ".txt")
        if not os.path.exists(label_path):
            missing += 1
            continue
        with Image.open(os.path.join(args.images, fname)) as im:
            img_w, img_h = im.size
        lines = convert_label_file(label_path, img_w, img_h, name_to_id)
        with open(os.path.join(args.out, stem + ".txt"), "w") as f:
            f.writelines(lines)
        written.append(os.path.join(args.images, fname))

    if args.list:
        with open(args.list, "w") as f:
            for p in written:
                f.write((os.path.abspath(p) if args.absolute else os.path.normpath(p)) + "\n")

    print("Converted %d label files (%d images had no label file)." % (len(written), missing))


if __name__ == "__main__":
    main()
