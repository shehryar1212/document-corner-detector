"""
Fast path: classical CV document detection.

Per-channel Canny union -> dilate -> convex hull -> polygon approximation
-> confidence scoring. This handles clean, well-lit photos of documents on
a contrasting background. It is cheap, so we always try it first and only
fall back to the deep learning path (deep_fallback.py) when it fails or
scores low confidence.

Note on the per-channel + convex-hull combination below: this replaced an
earlier single-grayscale-Canny version after real-photo testing (SmartDoc
2015 samples) showed it fragmenting document boundaries into 100-300+
tiny contours on textured backgrounds (wood grain, desk clutter) — mean
IoU 0.161, 13/18 frames undetected. Root cause: converting to grayscale
before edge detection throws away color contrast that's often the only
real signal between white paper and a similarly-lit surface. Running
Canny separately on each B/G/R channel and taking the union recovers
that signal (mean IoU 0.650 on the same 18 photos — see
tests/eval_real_samples.py). The remaining piece — taking the convex hull
of each candidate *before* measuring area or running approxPolyDP — fixes
a separate issue where dilation bridges scattered edge fragments into one
sprawling, non-convex connected component whose raw polygon area
computes to near-zero via the shoelace formula even though its bounding
box matches the true document almost exactly. The hull recovers the
actual document-shaped region from that fragment cloud.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .corner_utils import order_points
from .preprocessing import PreparedImage

# Confidence below this triggers the HED fallback in the pipeline.
CONFIDENCE_THRESHOLD = 0.6

# A detected quadrilateral must cover at least this fraction of the frame
# to plausibly be "the document" rather than a shadow, a piece of furniture,
# or noise. Real-world document photos are rarely framed looser than this.
MIN_AREA_RATIO = 0.05
MAX_AREA_RATIO = 0.98

# Interior angles of a document corner should be close to 90 degrees.
# Real photos have perspective distortion, so we allow a wide tolerance
# rather than requiring a true rectangle.
ANGLE_TOLERANCE_DEG = 35

# Canny/adaptive-threshold both have border-handling artifacts — the
# convolution padding at the very edge of the frame can register as a
# spurious "edge", which approxPolyDP then happily turns into a
# deceptively clean, high-confidence quadrilateral that's actually just
# the picture frame, not the document. If 2+ corners of a candidate sit
# right on the image border, treat it as this artifact and reject it
# outright rather than trusting the angle/area score.
BORDER_MARGIN_FRAC = 0.012

# A real document edge is a genuine intensity step in the raw grayscale
# image. In principle this could double-check a candidate independently
# of how its edge map was produced — but on very noisy, low-contrast
# photos this signal is itself too noisy to threshold reliably (tested:
# it scored a genuine-but-slightly-offset detection lower than an actual
# artifact in stress testing). Kept as a diagnostic you can log or
# inspect manually, not as an automatic gate — see README known
# limitations for the real-world implication.


@dataclass
class ClassicalResult:
    corners: Optional[np.ndarray]  # (4, 2) float32, in *resized* image coords, or None
    confidence: float
    reason: str  # human-readable explanation, useful for logging/debugging


def _angle_between(p0, p1, p2) -> float:
    """Interior angle at p1, formed by rays p1->p0 and p1->p2, in degrees."""
    v1 = p0.astype(np.float64) - p1.astype(np.float64)
    v2 = p2.astype(np.float64) - p1.astype(np.float64)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1e-6
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def edge_gradient_support(gray: np.ndarray, quad: np.ndarray, samples_per_edge: int = 25) -> float:
    """
    Mean Sobel gradient magnitude sampled along the quad's edges, in the
    raw (unthresholded, un-Canny'd) grayscale image. A real document
    boundary is a genuine intensity step here; a border-padding artifact
    from Canny/adaptive-threshold is not — this is an independent check
    against the source pixels, not against whatever produced the contour.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    h, w = gray.shape[:2]

    samples = []
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        for t in np.linspace(0.1, 0.9, samples_per_edge):  # skip the corners themselves
            x, y = p0 + t * (p1 - p0)
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                samples.append(mag[yi, xi])
    return float(np.mean(samples)) if samples else 0.0


def _corners_hugging_border(quad: np.ndarray, frame_shape) -> bool:
    """True if 2+ corners sit within BORDER_MARGIN_FRAC of the frame edge —
    almost always a Canny/threshold border artifact, not a real document."""
    h, w = frame_shape[:2]
    margin_x, margin_y = w * BORDER_MARGIN_FRAC, h * BORDER_MARGIN_FRAC
    hugging = 0
    for x, y in quad:
        if x <= margin_x or x >= w - margin_x or y <= margin_y or y >= h - margin_y:
            hugging += 1
    return hugging >= 2


def score_quad(quad: np.ndarray, frame_area: float, frame_shape=None) -> float:
    """Return a 0-1 confidence that `quad` (4x2 points) is a real document."""
    area = cv2.contourArea(quad.astype(np.float32))
    area_ratio = area / frame_area if frame_area else 0.0

    if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
        return 0.0
    if not cv2.isContourConvex(quad.astype(np.int32)):
        return 0.0
    if frame_shape is not None and _corners_hugging_border(quad, frame_shape):
        return 0.0

    angles = [
        _angle_between(quad[(i - 1) % 4], quad[i], quad[(i + 1) % 4])
        for i in range(4)
    ]
    max_dev = max(abs(a - 90.0) for a in angles)
    angle_score = max(0.0, 1.0 - max_dev / ANGLE_TOLERANCE_DEG)
    if max_dev > ANGLE_TOLERANCE_DEG:
        return 0.0

    # Reward quads that fill a healthy portion of the frame without
    # penalizing area beyond a reasonable point.
    area_score = min(1.0, area_ratio / 0.5)

    return float(round(0.7 * angle_score + 0.3 * area_score, 3))


def detect(prepared: PreparedImage) -> ClassicalResult:
    # Blur first, then split channels and run Canny on each separately —
    # a document edge that's washed out in grayscale (paper vs. a
    # similarly-lit wood desk, for example) is often still clearly visible
    # in at least one color channel. Union recovers that signal.
    blurred = cv2.GaussianBlur(prepared.resized, (5, 5), 0)
    b, g, r = cv2.split(blurred)
    edges = cv2.Canny(b, 50, 150) | cv2.Canny(g, 50, 150) | cv2.Canny(r, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    # Closing small gaps left by shadows/folds before contour extraction.
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ClassicalResult(None, 0.0, "no contours found")

    frame_area = prepared.resized.shape[0] * prepared.resized.shape[1]
    # Dilation can bridge scattered edge fragments into one sprawling,
    # non-convex connected component. Its raw contour area is meaningless
    # (can compute near-zero via the shoelace formula on a branching
    # shape even when the fragments trace the real document boundary) —
    # take the convex hull first, which recovers the actual document-
    # shaped region, then rank and approximate on that.
    hulls = [cv2.convexHull(c) for c in contours]
    hulls = sorted(hulls, key=cv2.contourArea, reverse=True)[:5]

    best_quad, best_score = None, 0.0
    for hull in hulls:
        perimeter = cv2.arcLength(hull, True)
        # Try a small range of epsilon values — a single fixed epsilon is
        # the most common reason approxPolyDP misses documents with
        # slightly rounded corners or a bit of edge noise from a fold.
        for eps_frac in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07):
            approx = cv2.approxPolyDP(hull, eps_frac * perimeter, True)
            if len(approx) != 4:
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            score = score_quad(quad, frame_area, prepared.resized.shape)
            if score > best_score:
                best_quad, best_score = quad, score

    if best_quad is None:
        return ClassicalResult(None, 0.0, "no 4-point polygon passed shape/area/angle checks")

    return ClassicalResult(best_quad, best_score, f"classical detection, confidence {best_score}")


# GrabCut refinement radii, in resized-image pixels (prepared.resized is
# capped at DETECTION_MAX_DIM, so these don't need to scale with input size).
_GRABCUT_SURE_BG_DILATE = 161
_GRABCUT_CORE_EROSION = 25


def refine_with_grabcut(resized_bgr: np.ndarray, quad: np.ndarray) -> ClassicalResult:
    """
    Given an already-confident quad, try growing it to the true document
    boundary using GrabCut's color-region model rather than edge gradients.

    NOT currently called from pipeline.py — tested and found net-negative,
    keeping it here (not deleted) since the mechanism is real and may be
    worth revisiting with a better accept/reject signal. SUPERSEDED for the
    case that motivated it: see snap_edges_to_shadow_boundary below, which
    fixes the same undershoot with a bounded, local signal instead of
    GrabCut's global color model and is the one actually wired into
    pipeline.py. Kept for the historical record and in case a future case
    needs a genuinely global (not edge-local) region-growing signal. Full
    result from when this was tried:

    Targets a real failure mode found on a client photo (caste_certificate,
    see tests/client_samples/): a genuine document edge that's a real but
    *low-gradient* transition (white paper against a similarly light
    background/surface, softened by focus or a shadow) which Canny
    under-detects even though the two regions are separable by average
    color. The Canny-based path instead locks onto the dense text/table
    content just inside that edge, understating the true page extent on
    every side that borders a low-contrast surface. It also independently
    confirmed the same undershoot in HU-PageScan's own segmentation mask on
    that photo, not just the classical path.

    On that one photo this recovered real accuracy (IoU 0.694 -> 0.716,
    confidence 0.841 -> 0.911) without an obviously runaway result. But
    validated against the 18-photo real-world benchmark
    (tests/eval_real_samples.py), it's net negative: mean IoU 0.663 -> 0.641,
    median 0.803 -> 0.636. One frame
    (background03_datasheet001_frame_0108.jpeg) explains almost the whole
    drop — GrabCut bled slightly into a similarly-toned desk background and
    score_quad rated the result *higher* than the original near-perfect
    detection (0.622 vs 0.600) despite IoU collapsing from 0.978 to 0.583.

    The accept/reject gate tried was "adopt only if score_quad's confidence
    and area both increase" — bounded seeding (see below) prevented most bad
    growth, but score_quad's angle+area heuristic isn't a reliable proxy for
    *correctness* here: an over-grown region can still look sufficiently
    boxy to score as well or better than the true boundary. This is the same
    class of problem as the HED-confidence-comparison finding in
    pipeline.py's docstring — a method-specific confidence number isn't
    automatically comparable to "is this actually right." A margin
    threshold (require a bigger score gap before adopting) could plausibly
    separate this one regression from the one real win, but with only two
    data points that's curve-fitting, not validation — not done here.

    Bounded so a bad refinement can't be worse than a no-op when it IS used:
    - seeded from the already-confident quad, so it can't wander to an
      unrelated region
    - can't grow into the outer frame border strip, which avoids inheriting
      the same convolution-padding artifact that _corners_hugging_border
      already guards against elsewhere in this file (without this, GrabCut
      readily "confirms" a border-touching seed as real, since paper-white
      pixels there genuinely do match the foreground color model)
    - the grown region still has to pass score_quad's normal convexity/
      angle/area checks before being trusted
    """
    h, w = resized_bgr.shape[:2]
    quad_mask = cv2.fillConvexPoly(np.zeros((h, w), np.uint8), quad.astype(np.int32), 255)
    sure_bg_region = cv2.dilate(quad_mask, np.ones((_GRABCUT_SURE_BG_DILATE, _GRABCUT_SURE_BG_DILATE), np.uint8))

    mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    mask[sure_bg_region == 0] = cv2.GC_BGD
    inner = cv2.erode(quad_mask, np.ones((_GRABCUT_CORE_EROSION, _GRABCUT_CORE_EROSION), np.uint8))
    mask[inner > 0] = cv2.GC_FGD

    bm = int(round(min(h, w) * BORDER_MARGIN_FRAC)) + 2
    mask[:bm, :] = cv2.GC_BGD
    mask[-bm:, :] = cv2.GC_BGD
    mask[:, :bm] = cv2.GC_BGD
    mask[:, -bm:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(resized_bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        return ClassicalResult(None, 0.0, f"grabcut refinement failed: {e}")

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ClassicalResult(None, 0.0, "grabcut produced no region")

    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    frame_area = h * w
    perimeter = cv2.arcLength(hull, True)
    best_quad, best_score = None, 0.0
    for eps_frac in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1):
        approx = cv2.approxPolyDP(hull, eps_frac * perimeter, True)
        if len(approx) != 4:
            continue
        q = approx.reshape(4, 2).astype(np.float32)
        score = score_quad(q, frame_area, (h, w))
        if score > best_score:
            best_quad, best_score = q, score

    if best_quad is None:
        return ClassicalResult(None, 0.0, "grabcut region had no plausible quadrilateral")
    return ClassicalResult(best_quad, best_score, f"grabcut boundary refinement, confidence {best_score}")


# How far outward (as a fraction of the frame's larger dimension) each edge
# is allowed to search for a real boundary beyond the current detection.
# Bounded deliberately: this is a local refinement of an already-confident
# quad, not a search over the whole frame.
_EDGE_SNAP_MAX_SEARCH_FRAC = 0.2

# A candidate position only counts as a real edge if at least this fraction
# of the cross-section (the span between the two corners on that side)
# shows a band-pass response there — rejects single-point noise hits while
# still tolerating gaps from text/marks/torn edges along the line.
_EDGE_SNAP_SUSTAIN_FRAC = 0.35

# A qualifying position also has to be part of a run of at least this many
# *consecutive* qualifying steps to count as a real boundary line, not an
# isolated blip. Found necessary after testing on the SmartDoc benchmark:
# an unrelated desk seam produced a brief 2-3px response that scored above
# _EDGE_SNAP_SUSTAIN_FRAC and, being farther out than the true (already
# correctly detected) edge, got wrongly adopted as an "extension" — collapsing
# one near-perfect detection (IoU 0.977 -> 0.732). The genuine shadow-edge
# case this function targets was a run of ~20 consecutive rows, not 2-3, so
# this distinguishes them on the strength/width of the physical evidence
# itself rather than tuning against the outcome metric.
_EDGE_SNAP_MIN_RUN = 6

# Band-pass threshold: difference between a narrow (3px) and wide (21px)
# box blur of grayscale, in intensity levels. Tuned to the shadow-edge
# transition width measured on the motivating case (~15-25px on the
# original photo) — see snap_edges_to_shadow_boundary's docstring.
_EDGE_SNAP_BAND_THRESHOLD = 12


def _bandpass_edge_mask(gray: np.ndarray) -> np.ndarray:
    """
    A page edge that's a soft shadow rather than a hard printed line is a
    real, large intensity change (paper white to shadow to background) but
    spread over many pixels — its *per-pixel* gradient (what Canny/Sobel
    measure) can be small enough to fall below any reasonable global
    threshold even though the edge is clearly visible to the eye. This
    computes a band-pass response instead: the difference between a small
    and a large box blur is large wherever intensity changes over roughly
    the large blur's radius — i.e. it's tuned to a transition *width*
    rather than a per-pixel step size, which is exactly the signal a
    gradual shadow edge produces and a hard Canny threshold misses.
    """
    gray = gray.astype(np.float32)
    narrow = cv2.blur(gray, (3, 3))
    wide = cv2.blur(gray, (21, 21))
    return np.abs(narrow - wide) > _EDGE_SNAP_BAND_THRESHOLD


def _scan_for_edge(mask: np.ndarray, lo: float, hi: float, start: float,
                    axis: str, direction: int, max_search: int) -> Optional[int]:
    """Walk outward from `start` along `axis` ('row' sweeps y, 'col' sweeps
    x), checking the cross-section [lo, hi) at each step for a band-pass
    response. Only a *run* of at least _EDGE_SNAP_MIN_RUN consecutive
    qualifying steps counts as a real boundary line (see _EDGE_SNAP_MIN_RUN
    for why). Returns the far end of the outermost qualifying run, or None
    if no run beyond `start` qualifies."""
    h, w = mask.shape
    lo_i, hi_i = int(round(min(lo, hi))), int(round(max(lo, hi)))
    if hi_i <= lo_i:
        return None

    best = None
    run_start = None
    for step in range(1, max_search + 1):
        coord = int(round(start)) + direction * step
        if axis == "row":
            in_bounds = 0 <= coord < h
            frac = mask[coord, lo_i:hi_i].mean() if in_bounds else 0.0
        else:
            in_bounds = 0 <= coord < w
            frac = mask[lo_i:hi_i, coord].mean() if in_bounds else 0.0

        qualifies = in_bounds and frac > _EDGE_SNAP_SUSTAIN_FRAC
        if qualifies:
            if run_start is None:
                run_start = step
        else:
            if run_start is not None and (step - run_start) >= _EDGE_SNAP_MIN_RUN:
                best = int(round(start)) + direction * (step - 1)
            run_start = None
        if not in_bounds:
            break
    if run_start is not None and (max_search + 1 - run_start) >= _EDGE_SNAP_MIN_RUN:
        best = int(round(start)) + direction * max_search
    return best


def snap_edges_to_shadow_boundary(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Given an already-confident quad, extend each of its 4 sides outward if
    a real but low-gradient (shadow-based) page edge is found just beyond
    it. Fixes the same undershoot documented in refine_with_grabcut's
    docstring (see tests/client_samples/README.md, caste_certificate.png)
    with a different, more targeted signal: instead of GrabCut's global
    color-region model — which is what caused that attempt's regression,
    bleeding into an unrelated same-toned background elsewhere in the
    18-photo benchmark — this only ever looks in a bounded band directly
    outside each of the 4 sides already detected, restricted to that
    side's own span between its two corners. It cannot be pulled toward an
    unrelated region elsewhere in the frame the way a global color or
    region-growing model can.

    Assumes the quad's sides are close to axis-aligned (true for most
    handheld document photos, which are rarely rotated far from upright);
    each side is scanned as a straight row/column band rather than along
    its true edge-normal direction. This is intentionally simpler than
    full per-edge-normal geometry — cheap, and validated in this codebase's
    convention: if it's wrong for a given photo, the accept/reject
    happens via score_quad in the caller, same as every other candidate
    here, so a bad snap just gets discarded rather than trusted blindly.
    """
    h, w = gray.shape[:2]
    band_mask = _bandpass_edge_mask(gray)
    max_search = int(_EDGE_SNAP_MAX_SEARCH_FRAC * max(h, w))

    tl, tr, br, bl = order_points(quad)

    new_top = _scan_for_edge(band_mask, tl[0], tr[0], min(tl[1], tr[1]), "row", -1, max_search)
    new_bottom = _scan_for_edge(band_mask, bl[0], br[0], max(bl[1], br[1]), "row", +1, max_search)
    new_left = _scan_for_edge(band_mask, tl[1], bl[1], min(tl[0], bl[0]), "col", -1, max_search)
    new_right = _scan_for_edge(band_mask, tr[1], br[1], max(tr[0], br[0]), "col", +1, max_search)

    new_tl, new_tr, new_br, new_bl = tl.copy(), tr.copy(), br.copy(), bl.copy()
    if new_top is not None:
        new_tl[1] = new_top
        new_tr[1] = new_top
    if new_bottom is not None:
        new_bl[1] = new_bottom
        new_br[1] = new_bottom
    if new_left is not None:
        new_tl[0] = new_left
        new_bl[0] = new_left
    if new_right is not None:
        new_tr[0] = new_right
        new_br[0] = new_right

    return np.array([new_tl, new_tr, new_br, new_bl], dtype=np.float32)
