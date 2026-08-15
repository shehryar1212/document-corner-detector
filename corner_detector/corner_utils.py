"""Consistent corner ordering, independent of document rotation."""
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Given 4 (x, y) points in any order, return them as
    [top-left, top-right, bottom-right, bottom-left].
    """
    pts = np.asarray(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has the smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right has the largest x+y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has the smallest y-x
    rect[3] = pts[np.argmax(diff)]  # bottom-left has the largest y-x

    return rect


def scale_corners(corners: np.ndarray, scale: float) -> np.ndarray:
    """Map corners detected on a resized image back to original-image coords."""
    return (corners.astype(np.float64) * scale).astype(np.float32)
