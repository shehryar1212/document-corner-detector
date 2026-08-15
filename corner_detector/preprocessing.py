"""
Image preprocessing utilities shared by both the classical and deep
learning detection paths.

Everything here is intentionally lossless with respect to geometry: any
resizing is tracked with a scale factor so corner coordinates can always be
mapped back to the original, full-resolution image.
"""
from dataclasses import dataclass

import cv2
import numpy as np


# Working resolution for detection. Detection runs on a downscaled copy for
# speed and to normalize the effect of parameters (kernel sizes, Canny
# thresholds) across wildly different input resolutions. Corners are mapped
# back to the original image afterwards, so this does not cost accuracy.
DETECTION_MAX_DIM = 1000


@dataclass
class PreparedImage:
    original: np.ndarray        # full-resolution BGR image, untouched
    resized: np.ndarray         # BGR image resized for detection
    scale: float                # multiply resized-image coords by this to get original coords
    gray: np.ndarray            # grayscale, resized
    blurred: np.ndarray         # denoised grayscale, resized
    adaptive_thresh: np.ndarray # adaptive threshold binary image, resized


def prepare_image(img: np.ndarray, max_dim: int = DETECTION_MAX_DIM) -> PreparedImage:
    """Run the shared preprocessing pipeline on a loaded BGR image."""
    if img is None:
        raise ValueError("prepare_image received None — check that the image loaded correctly")

    h, w = img.shape[:2]
    longest_side = max(h, w)
    scale = 1.0
    resized = img
    if longest_side > max_dim:
        scale = longest_side / float(max_dim)
        new_w, new_h = int(round(w / scale)), int(round(h / scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )

    return PreparedImage(
        original=img,
        resized=resized,
        scale=scale,
        gray=gray,
        blurred=blurred,
        adaptive_thresh=adaptive,
    )


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path} (unsupported format or bad path)")
    return img
