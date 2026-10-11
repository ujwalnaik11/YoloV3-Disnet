from __future__ import division

import argparse

import torch
from torch.utils.data import DataLoader

from models import Darknet
from utils.datasets import ListDataset
from utils.evaluation import evaluate
from utils.parse_config import parse_data_config
from utils.utils import load_classes


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def main():
    parser = argparse.ArgumentParser(description="Compute per-class AP and mAP on the validation set")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--model_config_path", type=str, default="config/yolov3-kitti.cfg", help="path to model config file")
    parser.add_argument("--data_config_path", type=str, default="config/kitti.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/kitti.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/kitti.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.45, help="iou threshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--use_cuda", type=str2bool, default=True, help="whether to use cuda if available")
    parser.add_argument("--output", type=str, default="test_data.txt", help="file the per-class APs and mAP are written to")
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available() and opt.use_cuda
    device = torch.device("cuda" if cuda else "cpu")

    data_config = parse_data_config(opt.data_config_path)
    num_classes = int(data_config["classes"])
    classes = load_classes(opt.class_path)

    model = Darknet(opt.model_config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)
    model = model.to(device)
    model.eval()  # always (the original only did this when CUDA was available)

    dataset = ListDataset(data_config["valid"], img_size=opt.img_size)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    print("Compute mAP...")
    average_precisions, mAP = evaluate(
        model, dataloader, num_classes, img_size=opt.img_size,
        conf_thres=opt.conf_thres, nms_thres=opt.nms_thres, iou_thres=opt.iou_thres, device=device,
    )

    print("Average Precisions:")
    for c, ap in average_precisions.items():
        print("+ Class '%s' - AP: %s" % (classes[c] if c < len(classes) else c, ap))
    print("mAP: %s" % mAP)

    with open(opt.output, "w") as f:
        f.write(" ".join("%.5f" % ap for ap in average_precisions.values()) + " %.5f\n" % mAP)


if __name__ == "__main__":
    main()
