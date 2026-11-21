"""Downloads both datasets into data/.

  Feline Behavior and Personality Survey (Figshare, CC BY 4.0)   ~2.7 MB
  Cat Emotion Dataset (Hugging Face, research use only)          ~2.1 GB, optional

Usage: python scripts/download_data.py [--skip-images]
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catsense import config as C
from catsense.data import download_feline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()

    print("feline survey:", download_feline())
    if args.skip_images:
        return
    C.DATA_IMAGES.mkdir(parents=True, exist_ok=True)
    for name in ["metadata.jsonl", "image.zip"]:
        target = C.DATA_IMAGES / name
        if not target.exists():
            print("downloading", name, "...")
            urllib.request.urlretrieve(f"{C.IMAGES_BASE_URL}/{name}", target)
    with zipfile.ZipFile(C.DATA_IMAGES / "image.zip") as zf:
        members = [m for m in zf.namelist() if m.startswith("images/") and not m.endswith("/")]
        zf.extractall(C.DATA_IMAGES, members)
    print("images:", len(members))


if __name__ == "__main__":
    main()
