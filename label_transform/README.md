# KITTI label transform

`kitti2coco-label-trans.py` converts KITTI object-detection labels to YOLO / darknet format
(`class cx cy w h`, all normalised to 0..1). Images and labels are paired by file name.

```bash
python label_transform/kitti2coco-label-trans.py \
    --images label_transform/kitti/images \
    --labels label_transform/kitti/labels \
    --out    /tmp/yolo_labels \
    --names  label_transform/kitti.names \
    --list   /tmp/train.txt
```

- `--list` writes the image list used by `config/*.data` (relative paths by default, `--absolute` for absolute ones).
- `DontCare` objects and classes not present in `--names` are skipped.
- `label_transform/kitti/` contains five raw KITTI samples to try it on.
