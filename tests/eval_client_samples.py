"""
Runs the detector against the two real client photos in tests/client_samples/
and scores against ground_truth.csv (Adobe Scan's own detected corners,
extracted from its adjust-corners UI — see tests/client_samples/README.md
for how). This is the primary benchmark: it reflects the client's actual
photos and the client's actual quality bar (Adobe Scan), not a public
dataset stand-in.

Also writes a green(Adobe)-vs-red(ours) overlay PNG per image to
tests/client_samples/_debug/ so a mismatch can be inspected visually,
not just read as a number.

Usage:
    python tests/eval_client_samples.py
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from corner_detector import DocumentCornerDetector
from corner_detector.corner_utils import order_points

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "client_samples")
GT_CSV = os.path.join(SAMPLE_DIR, "ground_truth.csv")
DEBUG_DIR = os.path.join(SAMPLE_DIR, "_debug")


def load_ground_truth():
    gt = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            corners = np.array([
                [float(row["tl_x"]), float(row["tl_y"])],
                [float(row["tr_x"]), float(row["tr_y"])],
                [float(row["br_x"]), float(row["br_y"])],
                [float(row["bl_x"]), float(row["bl_y"])],
            ], dtype=np.float32)
            gt[row["image_path"]] = order_points(corners)
    return gt


def iou_quad(a: np.ndarray, b: np.ndarray, shape) -> float:
    mask_a = np.zeros(shape[:2], dtype=np.uint8)
    mask_b = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask_a, a.astype(np.int32), 1)
    cv2.fillConvexPoly(mask_b, b.astype(np.int32), 1)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union else 0.0


def draw_overlay(img, ours, adobe, out_path):
    vis = img.copy()
    cv2.polylines(vis, [adobe.astype(np.int32)], True, (0, 255, 0), 4)   # green = Adobe (ground truth)
    if ours is not None:
        cv2.polylines(vis, [ours.astype(np.int32)], True, (0, 0, 255), 3)  # red = ours
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def run():
    gt = load_ground_truth()
    detector = DocumentCornerDetector()

    print(f"Evaluating against {len(gt)} real client photos\n")
    print(f"{'file':25s} {'method':18s} {'conf':6s} {'IoU':6s}")
    print("-" * 65)

    for fname, truth in sorted(gt.items()):
        path = os.path.join(SAMPLE_DIR, fname)
        if not os.path.isfile(path):
            print(f"{fname:25s} MISSING FILE")
            continue
        img = cv2.imread(path)
        result = detector.detect(img)

        if result.corners is None:
            print(f"{fname:25s} {'none':18s} {'--':6s} {'0.000':6s}   reason: {result.reason}")
            draw_overlay(img, None, truth, os.path.join(DEBUG_DIR, f"{fname}_overlay.png"))
            continue

        ours = order_points(result.corners)
        iou = iou_quad(ours, truth, img.shape)
        print(f"{fname:25s} {result.method:18s} {result.confidence:.3f}  {iou:.3f}   {result.reason[:80]}")
        draw_overlay(img, ours, truth, os.path.join(DEBUG_DIR, f"{fname}_overlay.png"))

    print("-" * 65)
    print(f"Overlays written to {DEBUG_DIR}")


if __name__ == "__main__":
    run()
