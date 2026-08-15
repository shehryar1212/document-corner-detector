"""
Generates a few synthetic "document photo" test images so the pipeline can
be smoke-tested without real client samples. These are NOT a substitute
for testing on the client's actual hard cases (shadows, curled corners,
low-contrast backgrounds) — see README for that step.
"""
import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_images")


def make_clean_document(path):
    """A white page, mild perspective skew, on a plain dark background."""
    canvas = np.full((900, 1200, 3), (40, 40, 45), dtype=np.uint8)
    src_doc = np.array([[0, 0], [849, 0], [849, 1099], [0, 1099]], dtype=np.float32)
    dst_doc = np.array([[220, 90], [1080, 140], [1010, 830], [160, 780]], dtype=np.float32)

    page = np.full((1100, 850, 3), 250, dtype=np.uint8)
    cv2.putText(page, "INVOICE", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
    for i in range(8):
        y = 300 + i * 80
        cv2.line(page, (80, y), (770, y), (60, 60, 60), 2)

    m = cv2.getPerspectiveTransform(src_doc, dst_doc)
    warped = cv2.warpPerspective(page, m, (1200, 900), borderValue=(40, 40, 45))
    mask = cv2.warpPerspective(
        np.full((1100, 850), 255, dtype=np.uint8), m, (1200, 900), borderValue=0
    )
    canvas[mask > 0] = warped[mask > 0]
    cv2.imwrite(path, canvas)
    return dst_doc


def make_shadowed_document(path):
    """Same idea, but with a soft shadow gradient and a lower-contrast background."""
    canvas = np.full((900, 1200, 3), (120, 118, 110), dtype=np.uint8)
    src_doc = np.array([[0, 0], [849, 0], [849, 1099], [0, 1099]], dtype=np.float32)
    dst_doc = np.array([[180, 60], [1050, 110], [990, 850], [140, 800]], dtype=np.float32)

    page = np.full((1100, 850, 3), 235, dtype=np.uint8)
    cv2.putText(page, "CONTRACT", (220, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 30, 30), 4)

    m = cv2.getPerspectiveTransform(src_doc, dst_doc)
    warped = cv2.warpPerspective(page, m, (1200, 900), borderValue=(120, 118, 110))
    mask = cv2.warpPerspective(
        np.full((1100, 850), 255, dtype=np.uint8), m, (1200, 900), borderValue=0
    )
    canvas[mask > 0] = warped[mask > 0]

    # Soft diagonal shadow across a third of the frame.
    shadow = np.zeros((900, 1200), dtype=np.float32)
    yy, xx = np.mgrid[0:900, 0:1200]
    shadow_strength = np.clip(1.0 - ((xx + yy) - 600) / 500.0, 0, 1) * 0.45
    for c in range(3):
        canvas[:, :, c] = (canvas[:, :, c].astype(np.float32) * (1 - shadow_strength)).astype(np.uint8)

    cv2.imwrite(path, canvas)
    return dst_doc


def make_hard_case(path):
    """
    Deliberately adversarial: low-contrast document on a similarly-toned
    background, heavy sensor noise, and a strong vignette. This is the kind
    of shot that trips up adaptive-threshold + Canny alone and is meant to
    exercise the HED fallback path, not the classical path.
    """
    canvas = np.full((900, 1200, 3), (205, 200, 190), dtype=np.uint8)
    noise = np.random.normal(0, 12, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    src_doc = np.array([[0, 0], [849, 0], [849, 1099], [0, 1099]], dtype=np.float32)
    dst_doc = np.array([[250, 120], [1020, 90], [970, 810], [210, 830]], dtype=np.float32)

    page = np.full((1100, 850, 3), 222, dtype=np.uint8)
    page_noise = np.random.normal(0, 6, page.shape).astype(np.int16)
    page = np.clip(page.astype(np.int16) + page_noise, 0, 255).astype(np.uint8)

    m = cv2.getPerspectiveTransform(src_doc, dst_doc)
    warped = cv2.warpPerspective(page, m, (1200, 900), borderValue=(205, 200, 190))
    mask = cv2.warpPerspective(
        np.full((1100, 850), 255, dtype=np.uint8), m, (1200, 900), borderValue=0
    )
    canvas[mask > 0] = warped[mask > 0]

    yy, xx = np.mgrid[0:900, 0:1200]
    dist = np.sqrt((xx - 600) ** 2 + (yy - 450) ** 2) / 750.0
    shade = np.clip(1 - dist * 0.6, 0.4, 1.0)
    for c in range(3):
        canvas[:, :, c] = (canvas[:, :, c].astype(np.float32) * shade).astype(np.uint8)

    cv2.imwrite(path, canvas)
    return dst_doc


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    c1 = make_clean_document(os.path.join(OUT_DIR, "clean_document.jpg"))
    c2 = make_shadowed_document(os.path.join(OUT_DIR, "shadowed_document.jpg"))
    c3 = make_hard_case(os.path.join(OUT_DIR, "hard_case.jpg"))
    print("Generated sample_images/clean_document.jpg, ground truth corners:\n", c1)
    print("Generated sample_images/shadowed_document.jpg, ground truth corners:\n", c2)
    print("Generated sample_images/hard_case.jpg, ground truth corners:\n", c3)
