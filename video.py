"""Real-time YOLOv3 detection + DisNet distance estimation on a webcam or video file.

    python video.py                       # default webcam
    python video.py --source clip.mp4     # video file
    python video.py --scaler scaler.pkl   # use the scaler fitted on your DisNet training data (recommended)

Requires the optional extras: pip install -r requirements-extras.txt
"""
from __future__ import division

import argparse
import os
import pickle
import time

import cv2
import numpy as np
import torch
from skimage.transform import resize

from models import Darknet
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
    parser.add_argument("--disnet_model", default="generated_files/model@1535470106",
                        help="path of the DisNet Keras model WITHOUT extension (<name>.json + <name>.h5)")
    parser.add_argument("--scaler", default=None,
                        help="pickle of the StandardScaler fitted on the DisNet training inputs. Without it the "
                             "legacy (incorrect) per-detection scaling is used, see README")
    parser.add_argument("--no_display", action="store_true", help="do not open a window (headless); use with --max_frames")
    parser.add_argument("--max_frames", type=int, default=0, help="stop after this many frames (0 = until the stream ends / 'q')")
    return parser.parse_args()


def load_disnet(path):
    """Load the DisNet regression model (Keras json architecture + h5 weights) and compile it once."""
    try:  # TensorFlow's bundled Keras (what most people have installed)
        from tensorflow.keras.models import model_from_json
    except ImportError:
        from keras.models import model_from_json
    with open(path + ".json") as f:
        model = model_from_json(f.read())
    model.load_weights(path + ".h5")
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def resize_img(img, img_size=416):
    """Pad the frame to a square, resize to the network resolution and return a (1, 3, H, W) tensor."""
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # OpenCV frames are BGR, the network was trained on RGB
    rgb = img[:, :, ::-1]
    input_img = np.pad(rgb, pad, "constant", constant_values=128) / 255.0
    input_img = resize(input_img, (img_size, img_size, 3), mode="reflect")
    input_img = np.transpose(input_img, (2, 0, 1))
    return torch.from_numpy(input_img.copy()).float().unsqueeze(0)


def estimate_distance(disnet, box, location, scaler=None):
    """Predict distance (metres) for one box given in network-input coordinates (x1, y1, x2, y2)."""
    from sklearn.preprocessing import StandardScaler

    x_test = np.array([box], dtype=np.float64)
    if scaler is None:
        # Legacy behaviour: fit on this single sample (this maps the input to zeros -- see README)
        scaler = StandardScaler()
        x_test = scaler.fit_transform(x_test)
    else:
        x_test = scaler.transform(x_test)

    y_pred = disnet.predict(x_test, verbose=0)

    # Put the prediction in the first column of a dataset-shaped array so the scaler can invert it
    dataset_like = np.zeros(shape=(len(y_pred), 4))
    dataset_like[:, 0] = y_pred[:, 0]
    y_pred = scaler.inverse_transform(dataset_like)[:, 0]
    y_pred = (y_pred - 25) / 100
    if location == "left":
        y_pred += 1.5
    elif location == "right":
        y_pred -= 1
    return float(y_pred[0])


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

    disnet = load_disnet(args.disnet_model)

    scaler = None
    if args.scaler:
        with open(args.scaler, "rb") as f:
            scaler = pickle.load(f)
    else:
        print("WARNING: no --scaler given; distances use the legacy per-detection scaling and are not reliable.")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), "Cannot capture source %r" % args.source

    frames = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            prev_time = time.time()
            frame_h, frame_w = frame.shape[:2]
            center = frame_w / 2

            input_img = resize_img(frame, args.reso).to(device)
            with torch.no_grad():
                detections = non_max_suppression(model(input_img), len(classes), args.confidence, args.nms_thresh)[0]

            # Geometry to map network-space boxes back onto the original frame
            pad_x = max(frame_h - frame_w, 0) * (args.reso / max(frame.shape[:2]))
            pad_y = max(frame_w - frame_h, 0) * (args.reso / max(frame.shape[:2]))
            unpad_h = args.reso - pad_y
            unpad_w = args.reso - pad_x

            if detections is not None:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                    centroid = (x1 + x2) / 2
                    loc = None
                    if centroid < (center - center / 2):
                        loc = "left"
                    elif centroid < center:
                        loc = "center"
                    elif centroid < (center + center / 3):
                        loc = "right"

                    # Box in original frame coordinates (for drawing)
                    x1b = int(((x1 - pad_x // 2) / unpad_w) * frame_w)
                    y1b = int(((y1 - pad_y // 2) / unpad_h) * frame_h)
                    x2b = int(((x2 - pad_x // 2) / unpad_w) * frame_w)
                    y2b = int(((y2 - pad_y // 2) / unpad_h) * frame_h)

                    distance = estimate_distance(disnet, [x1, y1, x2, y2], loc, scaler)
                    name = classes[min(int(cls_pred), len(classes) - 1)]
                    text = "{} m".format(round(distance, 2))
                    print(name, text, loc)

                    cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (255, 0, 0), 1)
                    cv2.putText(frame, name, (x1b, max(y1b - 5, 12)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
                    cv2.putText(frame, text, ((x1b + x2b) // 2, (y1b + y2b) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            fps = 1.0 / max(time.time() - prev_time, 1e-6)
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
