from __future__ import division

import argparse
import os

import torch
from torch.utils.data import DataLoader

from models import Darknet
from utils.datasets import ListDataset
from utils.evaluation import evaluate
from utils.parse_config import parse_data_config, parse_model_config
from utils.utils import Visualizer, load_classes, weights_init_normal


def str2bool(v):
    """argparse's `type=bool` treats any non-empty string (even 'False') as True; parse properly."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=10, help="size of each image batch")
    parser.add_argument("--model_config_path", type=str, default="config/yolov3-kitti.cfg", help="path to model config file")
    parser.add_argument("--data_config_path", type=str, default="config/kitti.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/darknet53.conv.74",
                        help="pretrained weights (e.g. darknet53.conv.74). If the file does not exist, training starts from random init")
    parser.add_argument("--class_path", type=str, default="data/kitti.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval (epochs) between saving model weights")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
    parser.add_argument("--use_cuda", type=str2bool, default=True, help="whether to use cuda if available")
    parser.add_argument("--freeze_epochs", type=int, default=20,
                        help="keep the darknet53 backbone (layers < 75) frozen for this many epochs; 0 disables freezing")
    parser.add_argument("--accumulated_batches", type=int, default=4, help="batches to accumulate gradients over before each optimizer step")
    parser.add_argument("--visdom", action="store_true", help="plot losses to a running visdom server (pip install visdom)")
    return parser.parse_args()


def set_backbone_trainable(model, trainable, n_backbone_layers=75):
    """(Un)freeze the first `n_backbone_layers` modules (the darknet53 backbone)."""
    for name, p in model.named_parameters():
        # parameter names look like "module_list.<layer index>.<...>"
        if int(name.split(".")[1]) < n_backbone_layers:
            p.requires_grad = trainable


def main():
    opt = parse_args()
    print(opt)

    cuda = torch.cuda.is_available() and opt.use_cuda
    device = torch.device("cuda" if cuda else "cpu")
    print("Using device: %s" % device)

    os.makedirs("output", exist_ok=True)
    os.makedirs(opt.checkpoint_dir, exist_ok=True)

    vis = Visualizer("yolo v3") if opt.visdom else None

    classes = load_classes(opt.class_path)

    # Data configuration
    data_config = parse_data_config(opt.data_config_path)
    train_path = data_config["train"]
    test_path = data_config["valid"]
    num_classes = int(data_config["classes"])
    assert num_classes == len(classes), "%s lists %d classes but %s says classes=%d" % (
        opt.class_path, len(classes), opt.data_config_path, num_classes)

    # Model
    model = Darknet(opt.model_config_path, img_size=opt.img_size)
    if os.path.exists(opt.weights_path):
        model.load_weights(opt.weights_path)
    else:
        print("WARNING: %s not found, training from random initialisation" % opt.weights_path)
        model.apply(weights_init_normal)
    model = model.to(device)
    model.train()

    dataloader = DataLoader(ListDataset(train_path, img_size=opt.img_size),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    test_dataloader = DataLoader(ListDataset(test_path, img_size=opt.img_size),
                                 batch_size=16, shuffle=False, num_workers=opt.n_cpu)

    # The optimizer is created while every parameter is trainable; frozen parameters
    # simply receive no gradient (Adam skips them) until they are unfrozen.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    best_mAP = 0.0
    names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    print("start training")
    with open("loss data.txt", "w") as loss_data_file, open("test_data.txt", "w") as test_data_file:
        window = dict.fromkeys(names + ["total"], 0.0)  # sums over the last <=50 batches, for visdom

        for epoch in range(opt.epochs):
            if opt.freeze_epochs > 0:
                set_backbone_trainable(model, trainable=epoch >= opt.freeze_epochs)

            model.train()
            optimizer.zero_grad()

            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                imgs = imgs.float().to(device)
                targets = targets.float().to(device)

                loss = model(imgs, targets)
                loss.backward()

                # accumulate gradients over several batches before stepping
                if ((batch_i + 1) % opt.accumulated_batches == 0) or (batch_i == len(dataloader) - 1):
                    optimizer.step()
                    optimizer.zero_grad()

                for n in names:
                    window[n] += model.losses[n]
                window["total"] += loss.item()

                if vis is not None and (batch_i + 1) % 50 == 0:
                    for n, label in zip(names, ["Losses:x", "Losses:y", "Losses:w", "Losses:h",
                                                "Losses:conf", "Losses:cls", "Recall", "Precision"]):
                        vis.plot(label, window[n])
                    vis.plot("Total Loss", window["total"])
                    window = dict.fromkeys(window, 0.0)

                loss_data_file.write("%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (
                    model.losses["x"], model.losses["y"], model.losses["w"], model.losses["h"],
                    model.losses["conf"], model.losses["cls"], loss.item(),
                    model.losses["recall"], model.losses["precision"]))
                print(
                    "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                    % (epoch, opt.epochs, batch_i, len(dataloader),
                       model.losses["x"], model.losses["y"], model.losses["w"], model.losses["h"],
                       model.losses["conf"], model.losses["cls"], loss.item(),
                       model.losses["recall"], model.losses["precision"])
                )

                model.seen += imgs.size(0)

            loss_data_file.flush()

            if epoch % opt.checkpoint_interval == 0:
                model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))

            # ---- validation mAP (evaluate() switches to eval mode and restores train mode)
            print("Compute %d Epoch mAP..." % epoch)
            average_precisions, mAP = evaluate(
                model, test_dataloader, num_classes, img_size=opt.img_size,
                conf_thres=opt.conf_thres, nms_thres=opt.nms_thres, iou_thres=opt.iou_thres, device=device,
            )
            print("Average Precisions:")
            for c, ap in average_precisions.items():
                print("+ Class '%s' - AP: %s" % (classes[c], ap))
                test_data_file.write("%.5f " % ap)
            print("mAP: %s" % mAP)
            test_data_file.write("%.5f\n" % mAP)
            test_data_file.flush()

            if mAP > best_mAP:
                best_mAP = mAP
                model.save_weights("%s/kitti_best.weights" % opt.checkpoint_dir)
                print("New Best AP appear !!! %f" % best_mAP)


if __name__ == "__main__":
    main()
