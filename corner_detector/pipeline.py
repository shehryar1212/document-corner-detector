"""
DocumentCornerDetector — the single entry point client code should use.

Runs classical detection first (fast). If confident (>= CONFIDENCE_THRESHOLD),
also runs HU-PageScan as a cheap sanity check and overrides classical only
if PageScan is both reasonably confident AND finds a notably larger region
— the specific signature of classical confidently cutting off real content
(see pipeline_notes below). HED is deliberately excluded from overriding
the shortcut: real testing showed it runs high-confidence even when wrong
more often than the other two paths, so letting it compete on raw
confidence alone made results worse, not better (validated: reduced mean
IoU on a real 18-photo benchmark from 0.663 to 0.601 — reverted).

If classical isn't confident, all three paths are compared and the
highest-scoring candidate wins, same as before.

KNOWN LIMITATION (found while testing against a real client photo): this
override rule uses total quad area to detect "classical cut off content",
but a case exists where classical missed real content (a document title)
while its total area was actually *smaller* than the correct answer by
only ~2% — nowhere near the area-ratio threshold below. Area alone can't
reliably detect this failure mode; it would need a check that specifically
looks for missed content near detected edges, which isn't implemented
here yet. Flagging honestly rather than claiming this is solved.

Always returns corners in the *original* image's coordinate space,
ordered [top-left, top-right, bottom-right, bottom-left].
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import classical_detection, content_region_fallback, deep_fallback, pagescan_fallback
from .corner_utils import order_points, scale_corners
from .perspective import four_point_transform
from .preprocessing import load_image, prepare_image

# How much larger (by area) PageScan's candidate must be than classical's
# to override the classical shortcut. See KNOWN LIMITATION above — this
# threshold does not catch every case where classical drops real content.
PAGESCAN_OVERRIDE_AREA_RATIO = 1.15
PAGESCAN_OVERRIDE_MIN_CONFIDENCE = 0.5


@dataclass
class DetectionResult:
    corners: Optional[np.ndarray]  # (4, 2) float32 in original-image coordinates, ordered TL/TR/BR/BL
    confidence: float
    method: str          # "classical", "hed", "pagescan", or "none"
    reason: str          # human-readable detail for logging


class DocumentCornerDetector:
    def __init__(self, confidence_threshold: float = classical_detection.CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> DetectionResult:
        prepared = prepare_image(image)

        classical = classical_detection.detect(prepared)

        if classical.corners is not None and classical.confidence >= self.confidence_threshold:
            snapped_corners = classical_detection.snap_edges_to_shadow_boundary(prepared.gray, classical.corners)
            frame_area = prepared.resized.shape[0] * prepared.resized.shape[1]
            snapped_score = classical_detection.score_quad(snapped_corners, frame_area, prepared.resized.shape)
            if snapped_score >= classical.confidence and cv2.contourArea(snapped_corners) > cv2.contourArea(classical.corners):
                classical = classical_detection.ClassicalResult(
                    snapped_corners, snapped_score,
                    f"{classical.reason} + edge-snapped to shadow boundary, confidence {snapped_score}",
                )

            pagescan = pagescan_fallback.detect(prepared)
            if pagescan.corners is not None and pagescan.confidence >= PAGESCAN_OVERRIDE_MIN_CONFIDENCE:
                classical_area = cv2.contourArea(classical.corners)
                pagescan_area = cv2.contourArea(pagescan.corners)
                if pagescan_area > classical_area * PAGESCAN_OVERRIDE_AREA_RATIO:
                    corners = order_points(scale_corners(pagescan.corners, prepared.scale))
                    reason = f"{pagescan.reason} (overrode classical={classical.confidence:.3f}, notably larger region)"
                    return DetectionResult(corners, pagescan.confidence, "pagescan-override", reason)
            corners = order_points(scale_corners(classical.corners, prepared.scale))
            return DetectionResult(corners, classical.confidence, "classical", classical.reason)

        # Classical wasn't confident enough — compare all three.
        hed = deep_fallback.detect(prepared)
        pagescan = pagescan_fallback.detect(prepared)
        candidates = [
            ("classical", classical.corners, classical.confidence, classical.reason),
            ("hed", hed.corners, hed.confidence, hed.reason),
            ("pagescan", pagescan.corners, pagescan.confidence, pagescan.reason),
        ]
        usable = [c for c in candidates if c[1] is not None and c[2] > 0]
        if not usable:
            reasons = "; ".join(f"{name}: {reason}" for name, _, _, reason in candidates)
            # All three photo-edge-based paths found nothing — possibly
            # because there's no physical page edge to find at all (a flat
            # render/screenshot rather than a camera photo; see
            # content_region_fallback.py). Only tried here, as a genuine
            # last resort: it can only replace a "none" with something
            # useful, never override or compete with a working edge-based
            # detection above.
            content = content_region_fallback.detect(prepared)
            if content.corners is not None:
                corners = order_points(scale_corners(content.corners, prepared.scale))
                return DetectionResult(
                    corners, content.confidence, "content-region",
                    f"{content.reason} (edge-based paths all failed: {reasons})",
                )
            return DetectionResult(None, 0.0, "none", reasons)

        method, corners, confidence, reason = max(usable, key=lambda c: c[2])
        corners = order_points(scale_corners(corners, prepared.scale))
        other_reasons = ", ".join(
            f"{name}={conf:.3f}" for name, c, conf, _ in candidates if c is not None and name != method
        )
        full_reason = f"{reason}" + (f" (also tried: {other_reasons})" if other_reasons else "")
        return DetectionResult(corners, confidence, method, full_reason)

    def detect_from_path(self, image_path: str) -> DetectionResult:
        return self.detect(load_image(image_path))

    def correct(self, image: np.ndarray, corners: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply perspective correction. Detects corners first if not given."""
        if corners is None:
            result = self.detect(image)
            if result.corners is None:
                raise ValueError(f"could not detect document corners: {result.reason}")
            corners = result.corners
        return four_point_transform(image, corners)
