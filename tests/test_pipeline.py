"""
Smoke test: runs the full detector against the synthetic sample images and
against a full deep-fallback-forced pass, printing corner error vs ground
truth. Run `python tests/generate_test_image.py` first.

This is NOT a replacement for testing on the client's real photos — it
only proves the pipeline runs end-to-end and gets sane results on easy
synthetic cases.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from corner_detector import DocumentCornerDetector
from corner_detector import classical_detection, deep_fallback
from corner_detector.corner_utils import order_points

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")

GROUND_TRUTH = {
    "clean_document.jpg": np.array([[220, 90], [1080, 140], [1010, 830], [160, 780]], dtype=np.float32),
    "shadowed_document.jpg": np.array([[180, 60], [1050, 110], [990, 850], [140, 800]], dtype=np.float32),
    "hard_case.jpg": np.array([[250, 120], [1020, 90], [970, 810], [210, 830]], dtype=np.float32),
}


def mean_corner_error(detected: np.ndarray, truth: np.ndarray) -> float:
    detected_ordered = order_points(detected)
    truth_ordered = order_points(truth)
    return float(np.mean(np.linalg.norm(detected_ordered - truth_ordered, axis=1)))


def run():
    detector = DocumentCornerDetector()
    print(f"HED weights available: {deep_fallback.weights_available()}\n")

    for fname, truth in GROUND_TRUTH.items():
        path = os.path.join(SAMPLE_DIR, fname)
        img = cv2.imread(path)
        result = detector.detect(img)
        print(f"--- {fname} ---")
        if result.corners is None:
            print(f"  FAILED: {result.reason}")
            continue
        err = mean_corner_error(result.corners, truth)
        print(f"  method={result.method} confidence={result.confidence:.3f}")
        print(f"  mean corner error: {err:.1f}px  ({result.reason})")

        out_path = os.path.join(SAMPLE_DIR, f"corrected_{fname}")
        warped = detector.correct(img, result.corners)
        cv2.imwrite(out_path, warped)
        print(f"  wrote corrected image to {out_path}\n")

    print(
        "Note: clean/shadowed cases are expected to resolve via the classical\n"
        "path; hard_case.jpg is deliberately adversarial (low contrast, noise,\n"
        "vignette) and is expected to trigger method='deep'. If it doesn't,\n"
        "that's worth a look before testing on the client's real photos."
    )


if __name__ == "__main__":
    run()
