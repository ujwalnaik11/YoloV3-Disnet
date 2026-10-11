from __future__ import division

import argparse
import datetime
import os
import time

import matplotlib
matplotlib.use("Agg")  # we only write images to disk; no display needed
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullLocator
from PIL import Image
from torch.utils.data import DataLoader

from models import Darknet
from utils.datasets import ImageFolder
from utils.utils import load_classes, non_max_suppression


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv3 on a folder of images and save annotated copies")
    parser.add_argument("--image_folder", type=str, default="data/samples/", help="path to dataset")
    parser.add_argument("--output_folder", type=str, default="output", help="where annotated images are written")
    parser.add_argument("--config_path", type=str, default="config/yolov3-kitti.cfg", help="path to model config file")
    parser.add_argument("--weights_path", type=str, default="weights/kitti.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/kitti.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--use_cuda", type=str2bool, default=True, help="whether to use cuda if available")
    return parser.parse_args()


def main():
    opt = parse_args()
    print("Config:")
    print(opt)

    if not os.path.exists(opt.weights_path):
        raise SystemExit("Weights file %r not found. Train one with train.py or point --weights_path at a "
                         "darknet .weights file (see README)." % opt.weights_path)

    cuda = torch.cuda.is_available() and opt.use_cuda
    device = torch.device("cuda" if cuda else "cpu")
    os.makedirs(opt.output_folder, exist_ok=True)

    classes = load_classes(opt.class_path)  # extracts class labels from file
    num_classes = len(classes)

    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)
    print("model path: " + opt.weights_path)
    model = model.to(device)
    model.eval()  # set in evaluation mode

    dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    imgs = []            # image paths
    img_detections = []  # detections for each image index

    print("data size : %d" % len(dataloader))
    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = input_imgs.float().to(device)

        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, num_classes, opt.conf_thres, opt.nms_thres)

        current_time = time.time()
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, datetime.timedelta(seconds=current_time - prev_time)))
        prev_time = current_time

        imgs.extend(img_paths)
        img_detections.extend(detections)

    # One stable colour per class
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))

        img = np.array(Image.open(path).convert("RGB"))
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Amount of padding that was added to make the image square, then scaled to network size
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x

        if detections is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                cls_id = int(cls_pred)
                print("\t+ Label: %s, Conf: %.5f" % (classes[cls_id], cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                color = colors[cls_id % len(colors)]
                ax.add_patch(patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none"))
                ax.text(x1, y1, s="%s %.4f" % (classes[cls_id], cls_conf.item()), color="white",
                        verticalalignment="top", bbox={"color": color, "pad": 0})

        plt.axis("off")
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        fig.savefig(os.path.join(opt.output_folder, "%d.png" % img_i), bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


if __name__ == "__main__":
    main()
