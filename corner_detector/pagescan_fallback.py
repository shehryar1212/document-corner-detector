"""
Third path: HU-PageScan, a small U-Net trained specifically for document
page segmentation (das Neves Junior et al., IET Image Processing, 2020,
trained on an extended SmartDoc dataset). Unlike HED (a generic 2015 edge
detector repurposed for this task), this model was actually trained to
answer "which pixels are the document" — it produces a filled region mask
rather than an edge map, which handles some low-contrast cases neither
classical detection nor HED can (see tests/eval_real_samples.py).

It is NOT uniformly better than HED, though — real testing showed each
catches document photos the other misses. Both are tried and scored the
same way (see pipeline.py) rather than one replacing the other.

Weights (~34MB) are NOT bundled — run `python download_weights.py` once
(see README). Degrades gracefully (returns None with a clear reason) if
missing, same as deep_fallback.py.

Source: https://github.com/ricardobnjunior/HU-PageScan (no explicit
license file at time of writing — used here for evaluation; confirm
licensing before any commercial redistribution of the weights themselves).
"""
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .classical_detection import score_quad
from .preprocessing import PreparedImage

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "hu_pagescan_model")
MODEL_PATH = os.path.join(_MODEL_DIR, "pre_trained_model.pb")

_INPUT_SIZE = 512
_BINARY_THRESHOLD = 127

_net = None  # lazily loaded and cached across calls


@dataclass
class PageScanResult:
    corners: Optional[np.ndarray]  # (4, 2) float32, in *resized* image coords, or None
    confidence: float
    reason: str


def weights_available() -> bool:
    return os.path.isfile(MODEL_PATH)


def _get_net():
    global _net
    if _net is None:
        if not weights_available():
            raise FileNotFoundError(
                "HU-PageScan weights not found. Run `python download_weights.py` "
                f"first (expected {MODEL_PATH})."
            )
        _net = cv2.dnn.readNetFromTensorflow(MODEL_PATH)
    return _net


def _segmentation_mask(bgr_image: np.ndarray) -> np.ndarray:
    """Returns a full-resolution uint8 mask (0-255) of predicted document
    pixels. Preprocessing here matches the model author's own reference
    script exactly (inferenceTestPB.py) — notably the input is *inverted*
    grayscale (1 - pixel/255), not plain normalized grayscale. Getting
    this wrong silently produces near-empty output rather than an error,
    so don't change it without re-validating against tests/real_samples/.
    """
    h, w = bgr_image.shape[:2]
    net = _get_net()

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inverted = 1.0 - (gray / 255.0)
    resized = cv2.resize(inverted, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    blob = resized.reshape(1, 1, _INPUT_SIZE, _INPUT_SIZE).astype(np.float32)

    net.setInput(blob, "input_1")
    out = net.forward("conv2d_19/Sigmoid")

    mask = (out[0, 0] * 255).astype(np.uint8)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)


def detect(prepared: PreparedImage) -> PageScanResult:
    if not weights_available():
        return PageScanResult(None, 0.0, "HU-PageScan weights not found — run download_weights.py")

    mask = _segmentation_mask(prepared.resized)
    _, binary = cv2.threshold(mask, _BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    # This is a filled-region mask, not an edge map — a wider close than
    # the edge-based paths, since small notches from objects resting on
    # the document (a pen, a marker) are holes in the region, not gaps in
    # a boundary line.
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return PageScanResult(None, 0.0, "HU-PageScan produced no usable region")

    frame_area = prepared.resized.shape[0] * prepared.resized.shape[1]
    hulls = sorted([cv2.convexHull(c) for c in contours], key=cv2.contourArea, reverse=True)[:5]

    best_quad, best_score = None, 0.0
    for hull in hulls:
        perimeter = cv2.arcLength(hull, True)
        for eps_frac in (0.01, 0.02, 0.03, 0.05, 0.07, 0.1):
            approx = cv2.approxPolyDP(hull, eps_frac * perimeter, True)
            if len(approx) != 4:
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            score = score_quad(quad, frame_area, prepared.resized.shape)
            if score > best_score:
                best_quad, best_score = quad, score

    if best_quad is None:
        # Same coarse fallback pattern as deep_fallback.py: a region mask
        # without a clean 4-point approximation is still often better
        # than nothing (irregular page edges, heavy occlusion notches).
        largest_hull = hulls[0]
        rect = cv2.minAreaRect(largest_hull)
        box = cv2.boxPoints(rect).astype(np.float32)
        score = score_quad(box, frame_area, prepared.resized.shape) * 0.8
        if score > 0:
            return PageScanResult(box, score, "HU-PageScan + minAreaRect fallback (coarse fit)")
        return PageScanResult(None, 0.0, "HU-PageScan found a region but no plausible quadrilateral")

    return PageScanResult(best_quad, best_score, f"HU-PageScan detection, confidence {best_score}")
