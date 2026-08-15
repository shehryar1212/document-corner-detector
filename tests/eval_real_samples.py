"""
Runs the detector against real (not synthetic) document photos from the
SmartDoc 2015 - Challenge 1 dataset (CC BY 4.0, Burie et al., ICDAR 2015),
and scores against the dataset's real ground-truth corner coordinates.

This is a genuinely useful stand-in until real client samples arrive:
real cameras, real lighting, real clutter — not synthetic noise.

Source: https://github.com/jchazalon/smartdoc15-ch1-dataset
Citation: J-C. Burie et al., "ICDAR2015 Competition on Smartphone Document
Capture and OCR (SmartDoc)", ICDAR 2015.

Usage:
    python tests/eval_real_samples.py
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from corner_detector import DocumentCornerDetector
from corner_detector.corner_utils import order_points

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "real_samples")
GT_CSV = os.path.join(SAMPLE_DIR, "ground_truth.csv")


def load_ground_truth():
    """Map flattened filename -> ordered (4,2) corner array, from the
    dataset's own tl/bl/br/tr columns."""
    gt = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            flat_name = row["image_path"].replace("/", "_")
            corners = np.array([
                [float(row["tl_x"]), float(row["tl_y"])],
                [float(row["tr_x"]), float(row["tr_y"])],
                [float(row["br_x"]), float(row["br_y"])],
                [float(row["bl_x"]), float(row["bl_y"])],
            ], dtype=np.float32)
            gt[flat_name] = order_points(corners)
    return gt


def iou_quad(a: np.ndarray, b: np.ndarray, shape) -> float:
    """Rasterized IoU between two quadrilaterals — more forgiving and more
    standard for this task than raw corner distance."""
    mask_a = np.zeros(shape[:2], dtype=np.uint8)
    mask_b = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask_a, a.astype(np.int32), 1)
    cv2.fillConvexPoly(mask_b, b.astype(np.int32), 1)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union else 0.0


def run():
    gt = load_ground_truth()
    detector = DocumentCornerDetector()

    print(f"Evaluating against {len(gt)} real photos from SmartDoc 2015 (CC BY 4.0)\n")
    print(f"{'file':45s} {'method':10s} {'conf':6s} {'IoU':6s}")
    print("-" * 75)

    ious, methods = [], {"classical": 0, "hed": 0, "pagescan": 0, "none": 0}
    for fname, truth in sorted(gt.items()):
        path = os.path.join(SAMPLE_DIR, fname)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        result = detector.detect(img)
        methods[result.method] = methods.get(result.method, 0) + 1

        if result.corners is None:
            print(f"{fname:45s} {'none':10s} {'--':6s} {'0.000':6s}")
            ious.append(0.0)
            continue

        iou = iou_quad(order_points(result.corners), truth, img.shape)
        ious.append(iou)
        print(f"{fname:45s} {result.method:10s} {result.confidence:.3f}  {iou:.3f}")

    print("-" * 75)
    print(f"mean IoU: {np.mean(ious):.3f}   median IoU: {np.median(ious):.3f}")
    print(f"method breakdown: {methods}")
    print(f"frames with IoU < 0.5 (likely visible misdetection): {sum(1 for i in ious if i < 0.5)}/{len(ious)}")


if __name__ == "__main__":
    run()
