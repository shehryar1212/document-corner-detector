# Client sample ground truth — how this was extracted

These two images (caste_certificate.png, marks_statement.png) are real
photos from the actual client (Safroz), not synthetic or benchmark data.
ground_truth.csv corners were NOT hand-labeled — they were extracted
programmatically from Adobe Scan's own "adjust corners" screen, so they
represent a real, concrete target (what the client considers correct)
rather than a guess.

## Method

1. Client sent screenshots of Adobe Scan's corner-adjustment screen (the
   step with four draggable circle handles overlaid on the photo, shown
   after auto-detect but before confirming the crop).
2. Detected the four circle handle centers via cv2.HoughCircles on a
   color mask isolating Adobe's UI blue (HSV range ~(95,100,150) to
   (115,255,255)).
3. Found the displayed photo's bounding box within the screenshot by
   scanning for the transition from the UI's uniform dark-gray background
   ((57,57,57)) to actual photo content, checked across multiple
   rows/columns for robustness.
4. Mapped handle positions from screenshot pixel space to the original
   photo's pixel space via independent x/y linear scaling between the
   display bounding box and the original image dimensions.

Sanity check: for marks_statement.png this gave scale_x=1.8520 and
scale_y=1.8519 — matching almost exactly, which is a good sign the
mapping is correct. For caste_certificate.png the two scales differed by
~3% (1.846 vs 1.786), likely a few pixels of imprecision in the display
boundary detection — treat that ground truth as good but not
pixel-perfect.

## If more client screenshots come in

Same process: screenshot Adobe's adjust-corners screen (not the final
cropped result — that loses the coordinate mapping back to the original
photo), then reuse this extraction method. Worth turning into a proper
script (`extract_adobe_ground_truth.py`) if this happens more than once
or twice more.

## Current results against this ground truth (IoU)

Run `python tests/eval_client_samples.py` to reproduce; it also writes a
green (Adobe) vs red (ours) overlay per image to `tests/client_samples/_debug/`.

- **caste_certificate.png: IoU 0.873** (classical, confidence 0.850).
  Root-caused: the bottom (and part of the left) margin transitions from
  white paper to a *light gray* surface, sometimes via a soft shadow where
  the page lifts slightly off the surface — a genuinely low-gradient edge
  that Canny under-detects. Confirmed this isn't a code bug: HU-PageScan (a
  trained segmentation model, an independent signal from Canny) undershoots
  the same margin on its own. Verified against the actual raw Adobe-Scan
  screenshots the client sent (the corner-adjustment UI, not just the
  extracted coordinates) that this low-contrast margin is exactly where
  Adobe's own detector still finds the true edge — zoomed-in crops at the
  true corners show a real, visible shadow band, just spread over ~15-25px
  rather than a sharp step.
  First tried GrabCut color-region refinement: real improvement on this
  photo (IoU 0.694 -> 0.716) but net-negative on the SmartDoc benchmark
  (bled into a similarly-toned desk elsewhere). Reverted — see
  `classical_detection.refine_with_grabcut`'s docstring.
  Fixed properly with `classical_detection.snap_edges_to_shadow_boundary`:
  a band-pass filter (difference of a narrow and wide box-blur, tuned to
  the measured ~15-25px shadow width) finds the transition width-wise
  instead of step-wise, then a bounded outward scan — restricted to a
  narrow band directly outside each of the 4 already-detected sides, not
  the whole frame — extends the boundary only where a *sustained* run of
  response is found (>= 6 consecutive px), not a single noisy hit. First
  version used "farthest sustained hit" and wrongly extended one SmartDoc
  photo's top edge into an unrelated desk seam (IoU 0.977 -> 0.732,
  caught by full-benchmark validation); requiring a longer minimum run
  fixed it with zero regression (confirmed: that photo and all other 17
  SmartDoc frames returned to their exact original IoU). Net result: IoU
  0.694 -> 0.873 on this photo, SmartDoc benchmark unchanged (0.671 mean,
  0.803 median, identical per-frame to before this change).
- **marks_statement.png: IoU 0.709** (was a total miss — all three
  edge-based methods found nothing). Root cause, confirmed by pixel
  forensics (near-zero border noise, flat R=G=B regions, no true gradient
  at Adobe's own top edge): this image is not a camera photo at all, it's
  a flat rendered/screenshotted table with no physical page-vs-background
  edge to trace — a genuinely different problem from what
  classical/HED/PageScan are built to solve. Adobe's own box turns out to
  be whitespace-gap-based content segmentation (bounding the table+title,
  excluding a separate footnote paragraph below it) rather than edge
  detection; measuring the actual gaps in this image showed they're all a
  uniform ~10-13px throughout (title lines, table rows, and the
  table-to-footnote gap alike) — there's no larger "section break" gap to
  threshold on, so reproducing Adobe's specific footnote exclusion would
  need real OCR/layout understanding, not gap-size clustering. Built
  `content_region_fallback.py` instead: merges the page's content into one
  region via color-distance-from-background + closing sized to the
  measured line-gap, deliberately including the footnote rather than
  guessing where a "real" section break is. Only invoked as a last resort
  when classical/HED/PageScan all fail, so it carries zero regression risk
  to real camera photos — validated on the 18-photo SmartDoc benchmark too
  (mean IoU 0.663 -> 0.671, no regressions, one previously-undetected
  occluded frame also improved 0.000 -> 0.138 as a side effect).
