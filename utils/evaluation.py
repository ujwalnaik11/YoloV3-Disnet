"""Shared mAP evaluation used by both train.py (per-epoch) and test.py."""
import numpy as np
import torch
import tqdm

from utils.utils import bbox_iou_numpy, compute_ap, non_max_suppression


def _collect(model, dataloader, num_classes, img_size, conf_thres, nms_thres, device):
    """Run the model over `dataloader` and gather per-image, per-class boxes."""
    all_detections = []
    all_annotations = []

    for _, imgs, targets in tqdm.tqdm(dataloader, desc="Detecting objects"):
        imgs = imgs.float().to(device)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, num_classes, conf_thres=conf_thres, nms_thres=nms_thres)

        for output, annotations in zip(outputs, targets):
            # ---- detections: x1, y1, x2, y2, score per class
            all_detections.append([np.array([]) for _ in range(num_classes)])
            if output is not None:
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()

                order = np.argsort(scores)
                pred_labels = pred_labels[order]
                pred_boxes = pred_boxes[order]

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            # ---- ground truth: class, cx, cy, w, h (normalised) -> x1, y1, x2, y2 in pixels
            all_annotations.append([np.array([]) for _ in range(num_classes)])
            valid = annotations[:, -1] > 0
            if bool(valid.any()):
                annotation_labels = annotations[valid, 0].numpy()
                boxes = annotations[valid, 1:].numpy()

                xyxy = np.empty_like(boxes)
                xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
                xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
                xyxy *= img_size

                for label in range(num_classes):
                    all_annotations[-1][label] = xyxy[annotation_labels == label, :]

    return all_detections, all_annotations


def _average_precision(all_detections, all_annotations, label, iou_thres):
    true_positives = []
    scores = []
    num_annotations = 0

    for i in range(len(all_annotations)):
        detections = all_detections[i][label]
        annotations = all_annotations[i][label]

        num_annotations += annotations.shape[0]
        detected_annotations = []

        for *bbox, score in detections:
            scores.append(score)

            if annotations.shape[0] == 0:
                true_positives.append(0)
                continue

            overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
            assigned_annotation = int(np.argmax(overlaps, axis=1)[0])
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap >= iou_thres and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                true_positives.append(0)

    # No ground truth for this class -> AP is 0
    if num_annotations == 0:
        return 0.0

    true_positives = np.array(true_positives)
    false_positives = 1 - true_positives

    indices = np.argsort(-np.array(scores))
    true_positives = np.cumsum(true_positives[indices])
    false_positives = np.cumsum(false_positives[indices])

    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    return compute_ap(recall, precision)


def evaluate(model, dataloader, num_classes, img_size=416, conf_thres=0.5, nms_thres=0.45, iou_thres=0.5, device="cpu"):
    """Compute per-class AP and mAP.

    The model is switched to eval mode for the duration of the call and its
    previous train/eval state is restored afterwards, so this is safe to call
    from inside a training loop.

    Returns (average_precisions: dict[int, float], mAP: float)
    """
    was_training = model.training
    model.eval()
    try:
        all_detections, all_annotations = _collect(
            model, dataloader, num_classes, img_size, conf_thres, nms_thres, device
        )
    finally:
        model.train(was_training)

    average_precisions = {
        label: _average_precision(all_detections, all_annotations, label, iou_thres)
        for label in range(num_classes)
    }
    return average_precisions, float(np.mean(list(average_precisions.values())))
