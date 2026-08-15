"""
FastAPI demo server for the document corner detector.

Run with:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /        -> upload-a-photo demo page (self-contained HTML)
    POST /detect   -> JSON: four corners + confidence + method used
    POST /correct  -> returns the perspective-corrected, cropped image (JPEG)
    POST /analyze  -> JSON: corners + confidence/method + a boundary
                       visualization and corrected crop, both as base64 JPEG
                       — everything the demo page needs from one upload
    GET  /health   -> basic liveness + whether HED fallback weights are loaded
"""
import base64
import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from corner_detector import DocumentCornerDetector
from corner_detector.deep_fallback import weights_available

app = FastAPI(title="Document Corner Detector", version="1.0")
detector = DocumentCornerDetector()

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "web")
_INDEX_PATH = os.path.join(_STATIC_DIR, "index.html")


def _read_upload_as_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image — is this a valid image file?")
    return img


@app.get("/", response_class=HTMLResponse)
def index():
    with open(_INDEX_PATH, encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "hed_fallback_available": weights_available()}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    img = _read_upload_as_image(data)
    result = detector.detect(img)

    if result.corners is None:
        raise HTTPException(status_code=422, detail=f"No document detected: {result.reason}")

    return {
        "corners": {
            "top_left": result.corners[0].tolist(),
            "top_right": result.corners[1].tolist(),
            "bottom_right": result.corners[2].tolist(),
            "bottom_left": result.corners[3].tolist(),
        },
        "confidence": result.confidence,
        "method": result.method,
        "reason": result.reason,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
    }


@app.post("/correct")
async def correct(file: UploadFile = File(...)):
    data = await file.read()
    img = _read_upload_as_image(data)
    result = detector.detect(img)

    if result.corners is None:
        raise HTTPException(status_code=422, detail=f"No document detected: {result.reason}")

    warped = detector.correct(img, result.corners)
    ok, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode corrected image")

    headers = {
        "X-Detection-Method": result.method,
        "X-Detection-Confidence": f"{result.confidence:.3f}",
    }
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg", headers=headers)


# Boundary color for /analyze's visualization, BGR — a confident amber/copper
# rather than plain red/green, since those are reserved elsewhere in this
# project (tests/eval_client_samples.py) for "Adobe's box" vs "ours".
_VIS_COLOR_BGR = (30, 110, 200)


def _encode_jpeg_b64(img: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Everything the upload demo page needs from a single request: the
    detected corners, a visualization with the boundary drawn on the
    original photo, and the perspective-corrected crop — all in one
    response so the page doesn't need three round trips."""
    data = await file.read()
    img = _read_upload_as_image(data)
    result = detector.detect(img)

    if result.corners is None:
        raise HTTPException(status_code=422, detail=f"No document boundary found: {result.reason}")

    pts = result.corners.astype(np.int32)
    thickness = max(2, img.shape[1] // 250)
    vis = img.copy()
    cv2.polylines(vis, [pts], True, _VIS_COLOR_BGR, thickness)
    for p in pts:
        cv2.circle(vis, tuple(p), max(5, thickness * 2), _VIS_COLOR_BGR, -1)

    warped = detector.correct(img, result.corners)

    return {
        "corners": {
            "top_left": result.corners[0].tolist(),
            "top_right": result.corners[1].tolist(),
            "bottom_right": result.corners[2].tolist(),
            "bottom_left": result.corners[3].tolist(),
        },
        "confidence": result.confidence,
        "method": result.method,
        "reason": result.reason,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "visualization_jpeg_base64": _encode_jpeg_b64(vis),
        "corrected_jpeg_base64": _encode_jpeg_b64(warped),
    }
