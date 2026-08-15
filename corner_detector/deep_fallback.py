"""
Hard-case path: Holistically-Nested Edge Detection (HED) via OpenCV's DNN
module, used when classical detection fails or scores low confidence
(heavy shadows, folded corners, low contrast, busy backgrounds).

The HED weights are ~57MB and are NOT bundled with this repo — run
`python download_weights.py` once to fetch them (see README). The
detector degrades gracefully (returns None with a clear reason) if the
weight files are missing, so the classical-only path still works without
them.
"""
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .classical_detection import score_quad
from .preprocessing import PreparedImage

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "hed_model")
PROTOTXT_PATH = os.path.join(_MODEL_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(_MODEL_DIR, "hed_pretrained_bsds.caffemodel")

# BGR mean subtraction values the HED BSDS model was trained with.
_MEAN = (104.00698793, 116.66876762, 122.67891434)

_net = None  # lazily loaded and cached across calls


@dataclass
class DeepResult:
    corners: Optional[np.ndarray]  # (4, 2) float32, in *resized* image coords, or None
    confidence: float
    reason: str


def weights_available() -> bool:
    return os.path.isfile(PROTOTXT_PATH) and os.path.isfile(CAFFEMODEL_PATH)


def _get_net():
    global _net
    if _net is None:
        if not weights_available():
            raise FileNotFoundError(
                "HED weights not found. Run `python download_weights.py` first "
                f"(expected {PROTOTXT_PATH} and {CAFFEMODEL_PATH})."
            )
        _net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    return _net


def _hed_edge_map(bgr_image: np.ndarray) -> np.ndarray:
    h, w = bgr_image.shape[:2]
    net = _get_net()
    blob = cv2.dnn.blobFromImage(
        bgr_image, scalefactor=1.0, size=(w, h),
        mean=_MEAN, swapRB=False, crop=False,
    )
    net.setInput(blob)
    out = net.forward()
    edge_map = out[0, 0]
    edge_map = cv2.resize(edge_map, (w, h))
    edge_map = (255 * edge_map).astype(np.uint8)
    return edge_map


def detect(prepared: PreparedImage) -> DeepResult:
    if not weights_available():
        return DeepResult(None, 0.0, "HED weights not found — run download_weights.py")

    edge_map = _hed_edge_map(prepared.resized)
    # HED outputs a soft probability map; threshold + close it the same
    # way we would a Canny map before contour extraction.
    _, binary = cv2.threshold(edge_map, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return DeepResult(None, 0.0, "HED produced no usable contours")

    frame_area = prepared.resized.shape[0] * prepared.resized.shape[1]
    # Same reasoning as the classical path: rank/approximate on the convex
    # hull of each contour, not the raw contour — robust to dilation
    # bridging scattered fragments into a sprawling, non-convex blob.
    hulls = [cv2.convexHull(c) for c in contours]
    hulls = sorted(hulls, key=cv2.contourArea, reverse=True)[:5]

    best_quad, best_score = None, 0.0
    for hull in hulls:
        perimeter = cv2.arcLength(hull, True)
        for eps_frac in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07):
            approx = cv2.approxPolyDP(hull, eps_frac * perimeter, True)
            if len(approx) != 4:
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            score = score_quad(quad, frame_area, prepared.resized.shape)
            if score > best_score:
                best_quad, best_score = quad, score

    if best_quad is None:
        # Fall back further: use the minAreaRect of the single largest
        # hull. This is a lower-confidence result but still often
        # better than nothing on very hard images (curved folds, torn
        # edges) where no clean quad exists in the edge map.
        largest_hull = hulls[0]
        rect = cv2.minAreaRect(largest_hull)
        box = cv2.boxPoints(rect).astype(np.float32)
        score = score_quad(box, frame_area, prepared.resized.shape) * 0.8  # penalize the coarser fit
        if score > 0:
            return DeepResult(box, score, "HED + minAreaRect fallback (coarse fit)")
        return DeepResult(None, 0.0, "HED found edges but no plausible quadrilateral")

    return DeepResult(best_quad, best_score, f"HED detection, confidence {best_score}")
