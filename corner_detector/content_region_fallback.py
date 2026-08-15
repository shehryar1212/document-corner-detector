"""
Last-resort path: content-region detection for images with no physical page
edge to trace at all — flat/borderless renders, screenshots, already-flat
digital documents. See tests/client_samples/README.md (marks_statement.png)
for the real client case that motivated this.

This is a fundamentally different problem from the other three paths, which
all assume a photographed page against a background and look for that
edge (a gradient, for classical/HED, or a learned page-vs-not-page region,
for PageScan). A flat render often has *no such edge* — the page background
and surrounding canvas can be the same color, so there is nothing there to
find. What IS present is a whitespace-vs-content structure: title, tables,
and text form dense regions of non-background color, separated from empty
margin by real (if sometimes small) gaps.

Investigated but deliberately NOT attempted: reproducing exactly what a
document scanner's own auto-detect does when it excludes one part of the
content (e.g. a footnote paragraph) from another (e.g. a table) above it.
On the real client photo this was checked against, the whitespace gaps
between every line of text — inside the title, before the table, AND
between the table and the excluded footnote — were all the same size
(~10-13px, measured directly). There is no larger "section break" gap to
threshold on; that exclusion reflects real layout/paragraph understanding
(this is a table vs. this is body text), which needs OCR or a trained
layout model, not a generalization of gap-size clustering. Out of scope
here per the "no training from scratch" constraint.

What this DOES do, honestly: finds the bounding region of the page's
overall content (title + tables + body text merged into one region,
excluding only the blank margin around it), which recovered a real,
reasonable result on the motivating case (IoU 0.0 -> 0.709 against the
client's own Adobe Scan ground truth) even though it does not exclude the
footnote the way Adobe's own detector did.

Only invoked from pipeline.py when classical, HED, and PageScan all found
nothing — this can only turn an existing "no detection" into something
useful; it never competes with or overrides an edge-based detection, so it
carries no regression risk to the photo-based benchmark (tests/eval_real_
samples.py). Its confidence is deliberately not on the same 0-1 scale as
score_quad's edge-based confidence — it reflects a different, coarser
signal and should not be compared against the other methods' scores.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .classical_detection import MAX_AREA_RATIO, MIN_AREA_RATIO
from .preprocessing import PreparedImage

# A pixel counts as "content" if its color is at least this far (Euclidean,
# in 0-255 BGR space) from the estimated page background color. Loose
# enough to catch light table-cell fills and anti-aliased text edges, not
# just solid ink — those are legitimately part of "the content region" too.
_CONTENT_COLOR_DISTANCE = 20

# Closing kernel: sized to bridge ordinary inter-line/inter-row whitespace
# (measured at ~10-13px on the motivating real case) so a title, a table,
# and body text merge into one region, without being so large it swallows
# the actual margin around the page.
_CLOSE_KERNEL = 17

# Fixed, deliberately modest confidence — see module docstring on why this
# isn't compared against the other paths' scores.
_CONFIDENCE = 0.5


@dataclass
class ContentRegionResult:
    corners: Optional[np.ndarray]  # (4, 2) float32, in *resized* image coords, or None
    confidence: float
    reason: str


def _estimate_background_color(resized_bgr: np.ndarray) -> np.ndarray:
    """Median color of a thin border strip. A flat/borderless render's
    background typically extends to the image edge (unlike a photographed
    page, which usually has some framing margin of its own background)."""
    border = np.concatenate([
        resized_bgr[0:5, :].reshape(-1, 3),
        resized_bgr[-5:, :].reshape(-1, 3),
        resized_bgr[:, 0:5].reshape(-1, 3),
        resized_bgr[:, -5:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def detect(prepared: PreparedImage) -> ContentRegionResult:
    resized = prepared.resized
    h, w = resized.shape[:2]

    bg_color = _estimate_background_color(resized)
    diff = np.linalg.norm(resized.astype(np.float32) - bg_color, axis=2)
    content_mask = (diff > _CONTENT_COLOR_DISTANCE).astype(np.uint8) * 255

    kernel = np.ones((_CLOSE_KERNEL, _CLOSE_KERNEL), np.uint8)
    closed = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ContentRegionResult(None, 0.0, "no content region found")

    biggest = max(contours, key=cv2.contourArea)
    frame_area = h * w
    area_ratio = cv2.contourArea(biggest) / frame_area
    if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
        return ContentRegionResult(
            None, 0.0, f"content region area ratio {area_ratio:.3f} implausible"
        )

    rect = cv2.minAreaRect(biggest)
    quad = cv2.boxPoints(rect).astype(np.float32)
    return ContentRegionResult(
        quad, _CONFIDENCE,
        f"content-region fallback (no physical page edge found), area_ratio={area_ratio:.3f}",
    )
