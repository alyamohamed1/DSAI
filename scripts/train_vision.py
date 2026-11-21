"""Image module: cat emotion classification (supporting system for shelter staff).

Steps
  1. Sample a class-capped subset of the Cat Emotion Dataset (Hugging Face, KAITANG2003).
  2. Data quality: run YOLOv8 (COCO) and drop images where no cat is found.
  3. Split 68/12/20 (train/validation/test), stratified.
  4. Baseline (majority class), LogReg on ResNet-50 embeddings of the full image,
     LogReg on embeddings of the YOLO cat crop (detection ablation), fine-tuned ResNet-18 on crops.
  5. Select on validation macro-F1, report everything on the test split.

Usage: python scripts/train_vision.py [--per-class 1500] [--epochs 6]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Silence a scikit-learn/joblib UserWarning repeated by every parallel worker
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from catsense import config as C
from catsense import evaluate
from catsense.vision import detect, emotion


def load_metadata(per_class: int) -> pd.DataFrame:
    meta = pd.read_json(C.DATA_IMAGES / "metadata.jsonl", lines=True)
    meta = meta.rename(columns={"final_emotion_label": "label"})
    meta = meta[meta["label"].isin(C.EMOTIONS)]
    meta = meta[meta["image"].map(lambda p: (C.DATA_IMAGES / p).exists())]
    sample = pd.concat(g.sample(min(len(g), per_class), random_state=C.RANDOM_STATE)
                       for _, g in meta.groupby("label"))
    return sample.reset_index(drop=True)


def run_detection(df: pd.DataFrame, cache: Path):
    if cache.exists():
        det = pd.read_csv(cache)
        if det["image"].tolist() == df["image"].tolist():
            return det
    rows = []
    for i, path in enumerate(df["image"]):
        boxes = detect.detect_cats(Image.open(C.DATA_IMAGES / path).convert("RGB"))
        rows.append({"image": path, "n_cats": len(boxes),
                     "conf": boxes[0]["conf"] if boxes else 0.0,
                     "box": json.dumps(boxes[0]["box"]) if boxes else ""})
        if i % 500 == 0:
            print(f"  detected {i}/{len(df)}")
    det = pd.DataFrame(rows)
    det.to_csv(cache, index=False)
    return det


def lr_search(X, y):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    search = GridSearchCV(pipe, {"logisticregression__C": [0.001, 0.01, 0.1, 1]}, scoring="f1_macro", cv=cv, n_jobs=-1)
    start = time.perf_counter()
    search.fit(X, y)
    return search, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(C.RANDOM_STATE)

    df = load_metadata(args.per_class)
    print("sampled:", df["label"].value_counts().to_dict())

    det = run_detection(df, C.RESULTS / f"vision_detections_{Path(C.YOLO_WEIGHTS).stem}.csv")
    df = df.merge(det, on="image")
    quality = df.assign(cat_found=df.n_cats > 0).groupby("label").agg(
        images=("image", "size"), cat_found_rate=("cat_found", "mean"), multi_cat_rate=("n_cats", lambda s: (s > 1).mean()))
    quality.to_csv(C.RESULTS / "vision_data_quality.csv")
    print(quality)
    df = df[df.n_cats > 0].reset_index(drop=True)

    label_ids = {e: i for i, e in enumerate(C.EMOTIONS)}
    y = df["label"].map(label_ids).to_numpy()
    idx_trval, idx_test = train_test_split(np.arange(len(df)), test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    idx_train, idx_val = train_test_split(idx_trval, test_size=0.15, stratify=y[idx_trval], random_state=C.RANDOM_STATE)
    print(f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    full = [Image.open(C.DATA_IMAGES / p).convert("RGB") for p in df["image"]]
    crops = [detect.crop(im, json.loads(b)) for im, b in zip(full, df["box"])]

    emb_cache = C.RESULTS / f"vision_embeddings_{Path(C.YOLO_WEIGHTS).stem}.npz"
    if emb_cache.exists() and np.load(emb_cache)["n"] == len(df):
        z = np.load(emb_cache)
        emb_full, emb_crop = z["full"], z["crop"]
    else:
        start = time.perf_counter()
        emb_full, emb_crop = emotion.embed(full), emotion.embed(crops)
        print(f"embedded {2 * len(df)} images in {time.perf_counter() - start:.1f}s")
        np.savez(emb_cache, full=emb_full, crop=emb_crop, n=len(df))

    rows, val_scores, cms = [], {}, {}

    def record(name, pred_val, pred_test, fit_s, ms):
        val_scores[name] = evaluate.multiclass_metrics(y[idx_val], pred_val)["macro_f1"]
        rows.append({"model": name, "val_macro_f1": val_scores[name],
                     **evaluate.multiclass_metrics(y[idx_test], pred_test),
                     "train_time_s": fit_s, "predict_ms_per_image": ms})
        cms[name] = pred_test
        print(f"{name:28s} val_f1={val_scores[name]:.3f} test_f1={rows[-1]['macro_f1']:.3f}")

    base = DummyClassifier(strategy="most_frequent").fit(emb_crop[idx_train], y[idx_train])
    record("baseline (majority)", base.predict(emb_crop[idx_val]), base.predict(emb_crop[idx_test]), 0.0, 0.0)

    lr_models = {}
    for name, emb in [("logreg_full_image", emb_full), ("logreg_yolo_crop", emb_crop)]:
        search, fit_s = lr_search(emb[idx_train], y[idx_train])
        start = time.perf_counter()
        pred_test = search.predict(emb[idx_test])
        ms = (time.perf_counter() - start) / len(idx_test) * 1000
        record(name, search.predict(emb[idx_val]), pred_test, fit_s, ms)
        lr_models[name] = search.best_estimator_

    start = time.perf_counter()
    net = emotion.train_finetune([crops[i] for i in idx_train], y[idx_train], epochs=args.epochs)
    fit_s = time.perf_counter() - start
    proba_val = emotion.predict_proba_net(net, [crops[i] for i in idx_val])
    start = time.perf_counter()
    proba_test = emotion.predict_proba_net(net, [crops[i] for i in idx_test])
    ms = (time.perf_counter() - start) / len(idx_test) * 1000
    record("finetuned_resnet18_crop", proba_val.argmax(1), proba_test.argmax(1), fit_s, ms)

    results = pd.DataFrame(rows)
    results.to_csv(C.RESULTS / "vision_results.csv", index=False)
    for name, pred in cms.items():
        evaluate.plot_confusion(y[idx_test], pred, C.EMOTIONS, f"{name} (row-normalised)",
                                C.FIGURES / f"cm_vision_{name.split(' ')[0]}.png", normalize="true")
    evaluate.plot_metric_bars(results.assign(split="test"), "macro_f1", C.FIGURES / "vision_macro_f1.png",
                              hue="model", x="split")

    best = max((k for k in val_scores if not k.startswith("baseline")), key=val_scores.get)
    print(f"selected by validation: {best}")
    if best == "finetuned_resnet18_crop":
        torch.save(net.state_dict(), C.MODELS / "emotion_resnet18.pt")
        bundle = {"kind": "finetune", "weights": "emotion_resnet18.pt"}
    else:
        bundle = {"kind": "embed_logreg", "model": lr_models[best], "uses_crop": best.endswith("crop")}
    bundle.update({"labels": C.EMOTIONS, "name": best,
                   "test_metrics": results.set_index("model").loc[best].to_dict()})
    joblib.dump(bundle, C.MODELS / "emotion_model.joblib")
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
