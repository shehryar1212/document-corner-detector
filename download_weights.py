"""
Downloads pretrained weights for both deep-learning fallback paths:
- HED (Holistically-Nested Edge Detection): ~57MB, generic edge detector
- HU-PageScan: ~34MB, a U-Net trained specifically for document page
  segmentation (das Neves Junior et al., IET Image Processing, 2020)

Run once, from the project root:
    python download_weights.py

The classical-only path works fine without either — these are only
needed for the fallback paths on harder photos (heavy shadow, low
contrast, folded corners, cluttered backgrounds).
"""
import os
import urllib.request

HED_DIR = os.path.join(os.path.dirname(__file__), "hed_model")
PAGESCAN_DIR = os.path.join(os.path.dirname(__file__), "hu_pagescan_model")

HED_FILES = {
    "deploy.prototxt":
        "https://raw.githubusercontent.com/ashukid/hed-edge-detector/master/deploy.prototxt",
    "hed_pretrained_bsds.caffemodel":
        "https://raw.githubusercontent.com/ashukid/hed-edge-detector/master/hed_pretrained_bsds.caffemodel",
}

PAGESCAN_FILES = {
    "pre_trained_model.pb":
        "https://raw.githubusercontent.com/ricardobnjunior/HU-PageScan/master/pretrained/pre_trained_model.pb",
}


def download(url: str, dest: str):
    print(f"Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  saved to {dest} ({size_mb:.1f} MB)")


def fetch_set(name: str, target_dir: str, files: dict):
    print(f"\n-- {name} --")
    os.makedirs(target_dir, exist_ok=True)
    for filename, url in files.items():
        dest = os.path.join(target_dir, filename)
        if os.path.isfile(dest):
            print(f"{filename} already present, skipping.")
            continue
        download(url, dest)


if __name__ == "__main__":
    fetch_set("HED (edge detection fallback)", HED_DIR, HED_FILES)
    fetch_set("HU-PageScan (segmentation fallback)", PAGESCAN_DIR, PAGESCAN_FILES)
    print("\nDone. Both deep-learning fallback paths are now available.")
