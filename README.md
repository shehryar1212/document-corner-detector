# Document Corner Detector

Hybrid document boundary detection: fast classical CV path for clean photos,
with an automatic deep-learning (HED) fallback for hard cases — shadows,
folded edges, low contrast, uneven backgrounds. Returns four ordered corner
coordinates and can apply perspective correction to produce a flat, cropped
output image.

This is the working implementation of the pipeline from the proposal,
ready to run and test against real sample images.

## Setup

```bash
pip install -r requirements.txt
python download_weights.py     # one-time, ~91MB (HED + HU-PageScan), needed for the fallback paths
```

`opencv-python-headless` is used instead of `opencv-python` since this runs
as a server with no GUI — smaller install, no libGL issues on a VPS.
Pinned to `<5`: OpenCV 5.0 removed its Caffe model importer entirely, which
HED's weights (`hed_pretrained_bsds.caffemodel`) depend on — installing an
unpinned `opencv-python-headless>=4.8` on a fresh setup today silently
breaks the HED fallback with an unhandled exception instead of the graceful
"weights not found" it's meant to degrade to.

## Quick check it works

```bash
python tests/generate_test_image.py   # builds two synthetic sample images
python tests/test_pipeline.py         # runs detection + correction on them
```

You should see confidence scores and corner error printed for both, plus
corrected images written to `tests/sample_images/corrected_*.jpg`. This
only proves the pipeline runs end to end — it does **not** replace testing
against the client's real photos (see "Testing on real samples" below).

## Using it directly in Python

```python
import cv2
from corner_detector import DocumentCornerDetector

detector = DocumentCornerDetector()
img = cv2.imread("photo.jpg")

result = detector.detect(img)
print(result.corners, result.confidence, result.method)

warped = detector.correct(img, result.corners)
cv2.imwrite("corrected.jpg", warped)
```

`result.method` is `"classical"` or `"deep"` so you always know which path
handled a given image — useful for tracking real-world failure rate once
this is running against live traffic.

## Demo API

```bash
uvicorn api:app --reload --port 8000
```

- `POST /detect` — upload an image, get back four corners + confidence + method as JSON
- `POST /correct` — upload an image, get back the perspective-corrected JPEG
- `GET /health` — liveness check, also reports whether the HED fallback is loaded

Example:
```bash
curl -X POST http://localhost:8000/detect -F "file=@sample.jpg"
```

## How the decision logic works

1. **Classical path** (always runs first): adaptive threshold → Canny →
   contour finding → `approxPolyDP` tried across a few epsilon values →
   each candidate quadrilateral is scored on convexity, corner angles
   (tolerant of perspective skew, rejects slivers), and how much of the
   frame it covers.
2. If the best classical score is **below 0.6 confidence**, or no clean
   4-point polygon was found at all, the **HED deep fallback** runs on the
   same image and is scored the same way.
3. Whichever result scored higher is returned, along with which method
   produced it and why (`result.reason`).
4. If neither path finds a confident result, the best available guess is
   still returned (never silently fails) — but `confidence` and `method`
   make it clear this needs review rather than treating it as reliable.

This mirrors what the proposal described: fast path first, deep learning
only when needed, so most images are processed quickly and the slower path
only kicks in for the genuinely hard ~1-in-4 photos.

## Testing on real samples

The synthetic images in `tests/generate_test_image.py` prove the code runs
correctly, but they don't represent real-world accuracy well. This repo
also includes 18 real photos from the SmartDoc 2015 dataset (ICDAR 2015,
CC BY 4.0 — see `tests/real_samples/`) with the dataset's own ground-truth
corner coordinates, spanning 6 document types across 3 lighting/background
conditions. Run:

```bash
python tests/eval_real_samples.py
```

**Current real-world result: mean IoU 0.671, median 0.803, 6/18 frames
below 0.5 IoU.** This started at mean IoU 0.161 (13/18 undetected) — the
root cause was grayscale Canny losing color contrast between white paper
and a similarly-lit textured background (wood grain, in most of these
photos). Fix: run Canny separately on each B/G/R channel and take the
union, then take the convex hull of each candidate contour before
measuring area or approximating a polygon (dilation was bridging
scattered edge fragments into a sprawling, non-convex shape whose raw
contour area computed to near-zero via the shoelace formula, even when
the fragments traced the real boundary almost exactly). Both changes are
in `classical_detection.py` and `deep_fallback.py`, with the reasoning
documented inline.

**Remaining known limitation:** most of the frames still below 0.5 IoU are
one specific scene — a document partially covered by markers, with a
cable physically crossing over it, sitting in a ragged multi-page stack.
That's occlusion, not contrast — no edge detector (classical or deep) can
trace a boundary that's physically blocked by another object on top of
it. This is worth flagging to the client as a UX consideration (ask users
to clear obstructions before photographing) rather than something to keep
chasing algorithmically; diminishing returns set in fast here.

## Testing against real client photos

`tests/client_samples/` holds real photos sent by the client, with ground
truth extracted directly from Adobe Scan's own corner-adjustment UI (see
`tests/client_samples/README.md` for exactly how) — this is the target
that actually matters most, since it's the client's own quality bar, not
a public dataset stand-in. Run:

```bash
python tests/eval_client_samples.py
```

This also writes a green (Adobe) vs red (ours) overlay PNG per image to
`tests/client_samples/_debug/` so a gap can be inspected visually, not
just read as a number. Add more client photos here as they come in and
re-run before any milestone payment — this, not the SmartDoc benchmark
above, is the bar that matters.

## Project layout

```
corner_detector/
  preprocessing.py           grayscale, blur, adaptive threshold, resize-for-detection
  classical_detection.py     Canny + contour + approxPolyDP + confidence scoring
  deep_fallback.py           HED edge detection + same scoring, for hard cases
  pagescan_fallback.py       HU-PageScan (trained segmentation model) + same scoring
  content_region_fallback.py last-resort path for images with no physical page edge
                              at all (flat renders/screenshots) — see its docstring
  corner_utils.py            consistent TL/TR/BR/BL ordering
  perspective.py             four-point perspective warp
  pipeline.py                DocumentCornerDetector — ties it all together
api.py                       FastAPI demo endpoint
download_weights.py          fetches the HED + HU-PageScan weights (not bundled)
tests/
  generate_test_image.py     synthetic sample images for a quick smoke test
  test_pipeline.py           runs detection + correction against synthetic images
  real_samples/              18 real photos + ground truth from SmartDoc 2015 (CC BY 4.0)
  eval_real_samples.py       scores the pipeline against real_samples with real IoU numbers
  client_samples/            real client photos + Adobe-Scan-derived ground truth
  eval_client_samples.py     scores the pipeline against client_samples, writes overlays
```

## Known limitations / next steps

- HED (2015-era model) is a reasonable, well-tested fallback but isn't the
  newest option — if the client's hard cases still fail after real-world
  testing, a more modern segmentation model could replace just
  `deep_fallback.py` without touching the rest of the pipeline.
- Confidence thresholds (`CONFIDENCE_THRESHOLD`, `MIN_AREA_RATIO`, angle
  tolerance in `classical_detection.py`) were set from general document-photo
  assumptions, not the client's actual data. These are the first things to
  tune once we see real accuracy numbers.
- No training required for either path — everything here runs off
  pretrained weights, matching the "no training required" note in the
  original plan.

## Attribution

`tests/real_samples/` contains a small subset of the SmartDoc 2015 -
Challenge 1 dataset, licensed CC BY 4.0. Citation: J-C. Burie, J. Chazalon,
M. Coustaty, S. Eskenazi, M. M. Luqman, M. Mehri, N. Nayef, J-M. Ogier,
S. Prum, M. Rusiñol, "ICDAR2015 Competition on Smartphone Document Capture
and OCR (SmartDoc)", ICDAR 2015. Original dataset:
http://smartdoc.univ-lr.fr/ — this subset via
https://github.com/jchazalon/smartdoc15-ch1-dataset.
