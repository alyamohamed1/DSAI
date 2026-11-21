"""Inference helpers for the app: photo and video emotion analysis."""
from functools import lru_cache

import joblib
import numpy as np
import torch
from PIL import Image

from .. import config as C
from . import detect, emotion


@lru_cache(maxsize=1)
def load_emotion_model():
    bundle = joblib.load(C.MODELS / "emotion_model.joblib")
    if bundle["kind"] == "finetune":
        net = emotion.finetune_net(len(bundle["labels"]))
        net.load_state_dict(torch.load(C.MODELS / bundle["weights"], map_location="cpu"))
        bundle["net"] = net.eval().to(emotion.DEVICE)
    return bundle


def emotion_proba(images: list[Image.Image]) -> np.ndarray:
    bundle = load_emotion_model()
    if bundle["kind"] == "finetune":
        return emotion.predict_proba_net(bundle["net"], images)
    return bundle["model"].predict_proba(emotion.embed(images))


def analyse_image(image: Image.Image, fallback_full_image: bool = False) -> dict:
    """Detects the cat and classifies its emotion.

    With fallback_full_image=True the whole picture is classified when YOLO finds no cat
    (YOLO misses some close-up faces); the result is flagged as unverified.
    """
    image = image.convert("RGB")
    boxes = detect.detect_cats(image)
    if not boxes:
        if not fallback_full_image:
            return {"n_cats": 0, "boxes": [], "emotion": None, "proba": None}
        bundle = load_emotion_model()
        proba = emotion_proba([image])[0]
        return {"n_cats": 0, "boxes": [], "crop": image, "unverified": True,
                "emotion": bundle["labels"][int(proba.argmax())],
                "proba": dict(zip(bundle["labels"], proba.round(4).tolist()))}
    bundle = load_emotion_model()
    use_crop = bundle["kind"] == "finetune" or bundle.get("uses_crop", True)
    target = detect.crop(image, boxes[0]["box"]) if use_crop else image
    proba = emotion_proba([target])[0]
    return {
        "n_cats": len(boxes), "boxes": boxes, "crop": target,
        "emotion": bundle["labels"][int(proba.argmax())],
        "proba": dict(zip(bundle["labels"], proba.round(4).tolist())),
    }


def analyse_video(path: str, every_n_frames: int = 15, max_frames: int = 60) -> dict:
    """Classifies sampled frames and returns the most frequent emotion (DeepCat video.py approach)."""
    import cv2
    cap = cv2.VideoCapture(path)
    labels, frames_seen, analysed = [], 0, 0
    while cap.isOpened() and analysed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frames_seen % every_n_frames == 0:
            result = analyse_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            analysed += 1
            if result["emotion"]:
                labels.append(result["emotion"])
        frames_seen += 1
    cap.release()
    label, share = emotion.majority_vote(labels)
    counts = {e: labels.count(e) for e in load_emotion_model()["labels"]}
    return {"emotion": label, "agreement": share, "frames_analysed": analysed,
            "frames_with_cat": len(labels), "counts": counts}
