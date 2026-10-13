"""Real-time YOLOv3 detection + DisNet distance estimation on a webcam or a video file.

    python video.py                       # default webcam, COCO weights
    python video.py --source clip.mp4     # video file

DisNet (Haseeb et al., 2018) estimates the camera-to-object distance from the size of each YOLO bounding
box relative to the image and the typical real-world size of the detected class, see utils/disnet.py.
Distances are shown for the classes DisNet has a size for (person, car, truck, bus, bicycle, motorbike,
cat, dog, horse, sheep, cow); other detections are drawn without a distance.

Requires opencv: pip install -r requirements-extras.txt
"""
from __future__ import division

import argparse
import time

import cv2
import numpy as np
import torch
from skimage.transform import resize

from models import Darknet
from utils.disnet import DisNet, canonical_class
from utils.utils import load_classes, non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO v3 + DisNet video demo")
    parser.add_argument("--source", default="0", help="camera index (e.g. 0) or path to a video file")
    parser.add_argument("--cfg", default="config/yolov3.cfg", help="darknet cfg file")
    parser.add_argument("--weights", default="weights/yolov3.weights", help="darknet weights file")
    parser.add_argument("--names", default="data/coco.names", help="class names file")
    parser.add_argument("--confidence", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--reso", type=int, default=416,
                        help="network input resolution (multiple of 32). Larger = more accurate, smaller = faster")
    parser.add_argument("--disnet_model", default="generated_files/disnet_model_3.npz",
                        help="DisNet model (.npz, or a Keras 2.x .h5 model). See train_disnet.py to train your own")
    parser.add_argument("--temporal_model", default=None,
                        help="GRU DisNet (.pt from train_disnet_rnn.py): track objects over frames, smoother distances "
                             "and objects kept alive through short detector misses")
    parser.add_argument("--stride", type=int, default=0,
                        help="frame spacing of the GRU's 3-frame window (0 = from the video frame rate, so that frames "
                             "are about 0.1 s apart as in training)")
    parser.add_argument("--no_display", action="store_true", help="do not open a window (headless); use with --max_frames")
    parser.add_argument("--max_frames", type=int, default=0, help="stop after this many frames (0 = until the stream ends / 'q')")
    return parser.parse_args()


def letterbox(frame_bgr, size):
    """Pad a BGR frame to a square (grey), resize to `size` and return a (1, 3, size, size) RGB tensor."""
    h, w, _ = frame_bgr.shape
    diff = abs(h - w)
    pad1, pad2 = diff // 2, diff - diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    rgb = frame_bgr[:, :, ::-1]  # OpenCV is BGR, the network was trained on RGB
    img = np.pad(rgb, pad, "constant", constant_values=128) / 255.0
    img = resize(img, (size, size, 3), mode="reflect")
    return torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()).float().unsqueeze(0)


def network_box_to_frame(box, frame_w, frame_h, size):
    """Map (x1, y1, x2, y2) from letterboxed network coordinates back to original frame pixels."""
    side = max(frame_w, frame_h)
    factor = side / float(size)
    pad_left = (side - frame_w) // 2 if frame_w < side else 0
    pad_top = (side - frame_h) // 2 if frame_h < side else 0
    x1, y1, x2, y2 = box
    return x1 * factor - pad_left, y1 * factor - pad_top, x2 * factor - pad_left, y2 * factor - pad_top


def draw_label(frame, box, label, color=(255, 0, 0)):
    """Draw a box with a filled label onto `frame` in place. `box` is (x1, y1, x2, y2) in frame pixels."""
    frame_h, frame_w = frame.shape[:2]
    bx1, by1, bx2, by2 = (int(round(v)) for v in box)
    bx1, bx2 = np.clip([bx1, bx2], 0, frame_w - 1)
    by1, by2 = np.clip([by1, by2], 0, frame_h - 1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = by1 - 6 if by1 - th - 6 >= 0 else by1 + th + 6
    cv2.rectangle(frame, (bx1, ty - th - 4), (bx1 + tw + 4, ty + 4), color, -1)
    cv2.putText(frame, label, (bx1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def annotate(frame, detections, classes, size, disnet):
    """Draw boxes (and DisNet distances where available) onto `frame` in place.

    detections: tensor (N, 7) of x1, y1, x2, y2, obj_conf, cls_conf, cls_pred in network coordinates, or None.
    Returns a list of (class_name, (x1, y1, x2, y2) in frame pixels, distance_m or None).
    """
    frame_h, frame_w = frame.shape[:2]
    results = []
    if detections is None:
        return results
    for x1, y1, x2, y2, _, _, cls_pred in detections.cpu().numpy():
        name = classes[min(int(cls_pred), len(classes) - 1)]
        box = network_box_to_frame((x1, y1, x2, y2), frame_w, frame_h, size)

        distance = None
        if disnet is not None and canonical_class(name) is not None:
            # features use the full (unclipped) box and the original frame size, as in the paper
            distance = disnet.estimate(box, name, frame_w, frame_h)
        results.append((name, box, distance))

        draw_label(frame, box, name if distance is None else "%s %.1f m" % (name, distance))
    return results


def annotate_tracked(frame, detections, classes, size, tracker):
    """Like `annotate`, but distances come from a utils.temporal.TemporalDisNet that tracks objects over frames.

    Objects the detector missed for a few frames are still drawn (in orange) on the predicted box.
    Returns the tracker's result dicts (id, name, box, distance, observed, source).
    """
    frame_h, frame_w = frame.shape[:2]
    dets = []
    if detections is not None:
        for x1, y1, x2, y2, _, _, cls_pred in detections.cpu().numpy():
            name = classes[min(int(cls_pred), len(classes) - 1)]
            dets.append((name, network_box_to_frame((x1, y1, x2, y2), frame_w, frame_h, size)))
    results = tracker.update(dets, frame_w, frame_h)
    for r in results:
        label = "#%d %s" % (r["id"], r["name"]) + ("" if r["distance"] is None else " %.1f m" % r["distance"])
        draw_label(frame, r["box"], label, (255, 0, 0) if r["observed"] else (0, 140, 255))
    return results


def main():
    args = parse_args()
    assert args.reso % 32 == 0 and args.reso > 32, "--reso must be a multiple of 32 and > 32"

    classes = load_classes(args.names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Darknet(args.cfg, img_size=args.reso)
    model.load_weights(args.weights)
    model = model.to(device)
    model.eval()
    print("YOLO running on %s" % device)

    disnet = DisNet.load(args.disnet_model)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), "Cannot capture source %r" % args.source

    tracker = None
    if args.temporal_model:
        from utils.temporal import TemporalDisNet, load_temporal, stride_for_fps
        distance_model, box_model, _ = load_temporal(args.temporal_model, device)
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        stride = args.stride or (stride_for_fps(fps_in) if fps_in and fps_in > 1 else 1)
        tracker = TemporalDisNet(distance_model, box_model, fallback=disnet, stride=stride, device=device)
        print("Temporal DisNet: window of 3 frames, %d frame(s) apart" % stride)

    frames = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            start = time.time()

            with torch.no_grad():
                output = model(letterbox(frame, args.reso).to(device))
                detections = non_max_suppression(output, len(classes), args.confidence, args.nms_thresh)[0]

            if tracker is not None:
                for r in annotate_tracked(frame, detections, classes, args.reso, tracker):
                    print(r["id"], r["name"], "-" if r["distance"] is None else "%.2f m" % r["distance"],
                          "" if r["observed"] else "(predicted)")
            else:
                for name, _, distance in annotate(frame, detections, classes, args.reso, disnet):
                    print(name, "-" if distance is None else "%.2f m" % distance)

            fps = 1.0 / max(time.time() - start, 1e-6)
            cv2.putText(frame, "%.1f FPS" % fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frames += 1
            if not args.no_display:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
