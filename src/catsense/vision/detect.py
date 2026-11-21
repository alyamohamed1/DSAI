"""Cat detection with a COCO-pretrained YOLOv8.

Adapted from the DeepCat project (Cat-Emotional-Analysis, `image.py`): count the
cats in a frame and keep the highest-confidence box for analysis. DeepCat used
custom YOLOv5 weights for eyes/tail/mouth that are not published, so here the
detector only localises whole cats (COCO class 15) and the crop is passed to the
emotion classifier.
"""
from functools import lru_cache

import numpy as np
from PIL import Image

from .. import config as C


@lru_cache(maxsize=1)
def load_detector():
    from ultralytics import YOLO
    return YOLO(C.YOLO_WEIGHTS)


def detect_cats(image: Image.Image, conf: float = 0.25) -> list[dict]:
    """Returns cat boxes sorted by confidence: [{"box": (x1, y1, x2, y2), "conf": float}, ...]."""
    result = load_detector().predict(image, classes=[C.COCO_CAT_CLASS], conf=conf, verbose=False)[0]
    boxes = [
        {"box": tuple(float(v) for v in b.xyxy[0].tolist()), "conf": float(b.conf[0])}
        for b in result.boxes
    ]
    return sorted(boxes, key=lambda b: b["conf"], reverse=True)


def crop(image: Image.Image, box, pad: float = 0.10) -> Image.Image:
    """Crops a box with some padding so ears and tail are not cut off."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - pad * w), max(0, y1 - pad * h)
    x2, y2 = min(image.width, x2 + pad * w), min(image.height, y2 + pad * h)
    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def best_cat_crop(image: Image.Image, conf: float = 0.25):
    """Returns (crop or None, number of cats, best confidence)."""
    boxes = detect_cats(image, conf)
    if not boxes:
        return None, 0, 0.0
    return crop(image, boxes[0]["box"]), len(boxes), boxes[0]["conf"]


def draw_boxes(image: Image.Image, boxes: list[dict]) -> np.ndarray:
    import cv2
    arr = np.array(image.convert("RGB")).copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b["box"])
        colour = (46, 160, 67) if i == 0 else (200, 200, 200)
        cv2.rectangle(arr, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(arr, f"cat {b['conf']:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return arr
