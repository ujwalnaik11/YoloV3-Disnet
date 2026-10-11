# YoloV3-Disnet

Object detection with **YOLOv3** (PyTorch) plus monocular **distance estimation** with a small
**DisNet**-style regression network (Keras). YOLOv3 finds objects (trained here on **KITTI**), and DisNet
turns each bounding box into an estimated distance in metres.

| | |
|---|---|
| ![tram and car](assets/0.png) | ![cyclist and pedestrians](assets/12.png) |

## Repository layout

```
detect.py                 run the detector on a folder of images, save annotated copies
train.py                  train / fine-tune on a dataset described by config/*.data
test.py                   per-class AP and mAP on the validation list
video.py                  webcam / video demo: YOLOv3 boxes + DisNet distances
models.py                 Darknet / YOLOv3 model, cfg parser, weight loading & saving
utils/
  datasets.py             image-folder and list-file datasets (letterbox padding, label loading)
  evaluation.py           mAP evaluation shared by train.py and test.py
  utils.py                NMS, IoU, target building, AP, optional visdom wrapper
config/                   yolov3, yolov3-tiny, yolov3-kitti cfgs + kitti.data / coco.data
data/                     class names, a 16+16 image KITTI sample, sample images for detect.py
label_transform/          KITTI -> YOLO label converter
generated_files/          the DisNet Keras models (.json architecture + .h5 weights) used by video.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # detect / train / test
pip install -r requirements-extras.txt     # only for video.py (opencv, tensorflow, scikit-learn) and --visdom
```

## Quick start

All commands run from the repository root and fall back to CPU when CUDA is unavailable
(`--use_cuda false` forces CPU).

**Detect** (needs a trained KITTI weights file, see below):

```bash
python detect.py --image_folder data/samples --weights_path weights/kitti.weights
# annotated images go to output/
```

**Train.** The repository ships a tiny KITTI sample (`data/kitti/`, 16 train / 16 val images) so you can check
that everything works end to end:

```bash
# from random init (smoke test)
python train.py --epochs 1 --batch_size 4 --weights_path none

# real training: start from the darknet53 backbone weights
wget https://pjreddie.com/media/files/darknet53.conv.74 -P checkpoints
python train.py --epochs 100 --batch_size 10 --weights_path checkpoints/darknet53.conv.74
```

Checkpoints are written to `checkpoints/` every `--checkpoint_interval` epochs, and the best validation mAP is
kept as `checkpoints/kitti_best.weights`. The backbone is frozen for the first `--freeze_epochs` epochs (default
20; `0` disables). Add `--visdom` for live loss plots (start `python -m visdom.server` first).

**Evaluate:**

```bash
python test.py --weights_path checkpoints/kitti_best.weights
```

**Video / webcam with distances:**

```bash
wget https://pjreddie.com/media/files/yolov3.weights -P weights   # COCO weights (default for video.py)
python video.py                       # default webcam, press q to quit
python video.py --source clip.mp4 --scaler scaler.pkl
```

## Using your own KITTI data

`train.py` expects YOLO-format labels (`class cx cy w h`, normalised) in a `labels/` directory that mirrors an
`images/` directory, and a text file listing one image path per line (paths relative to where you run the script
work fine). Convert raw KITTI labels with:

```bash
python label_transform/kitti2coco-label-trans.py \
    --images data/kitti/images/train --labels /path/to/kitti/label_2 \
    --out data/kitti/labels/train --names data/kitti.names --list data/kitti/train.txt
```

Then point `config/kitti.data` at your list files. `DontCare` boxes and classes missing from `--names` are skipped.

## DisNet distance estimation: read this before trusting `video.py`

`video.py` feeds each YOLO box `(x1, y1, x2, y2)` into the DisNet regression network in `generated_files/`.
That network was trained on **standardised** inputs, so at inference the inputs must be standardised with the
**same scaler that was fitted on the training data**.

The original script instead fitted a fresh `StandardScaler` on every single detection. A scaler fitted on one
sample maps the input to all zeros and turns the inverse transform into "add the box's x1 pixel coordinate", so the
printed distance was effectively `(x1 + c - 25) / 100` regardless of the box. It is kept as a fallback (with a
warning) so existing behaviour does not change silently, but for meaningful distances:

1. Save the scaler you fit on your DisNet training inputs: `pickle.dump(scaler, open("scaler.pkl", "wb"))`
2. Run `python video.py --scaler scaler.pkl`

The `left` / `right` offsets (+1.5 m / -1 m) and the `(y - 25) / 100` rescaling are hand-tuned constants from the
original project and have not been re-derived. The DisNet files in `generated_files/` were saved with Keras 2.1; with
recent TensorFlow see the note in `requirements-extras.txt`.

## Notes and known limitations

- The loss in `models.py` keeps the original class-loss formulation (cross-entropy applied to sigmoid outputs).
  It trains, but a plain BCE-on-logits or cross-entropy-on-logits classification loss is the usual choice.
- `mAP` counts classes with no ground truth in the validation set as AP = 0, which lowers the mean on small
  validation sets (the bundled sample has no `Person_sitting` or `Tram` boxes).
- Non-maximum suppression (`utils.utils.non_max_suppression`) is a plain Python loop; it is fine for evaluation
  but slow with a very low confidence threshold.

## Credits

Detector code is adapted from the open-source PyTorch-YOLOv3 implementation and its KITTI variant. DisNet follows
the monocular distance-estimation idea from Haseeb et al., *DisNet: A novel method for distance estimation from
monocular camera* (2018).
