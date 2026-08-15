FROM python:3.11-slim

WORKDIR /app

# libglib2.0-0 is needed by opencv-python-headless even without GUI support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY corner_detector/ corner_detector/
COPY web/ web/
COPY api.py download_weights.py ./

# Bake the weights into the image at build time — avoids a slow/fragile
# download on first request and keeps the container self-contained.
RUN python download_weights.py

ENV PORT=8000
EXPOSE 8000
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
