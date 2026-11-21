"""Metrics and plots shared by the tabular and image experiments."""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    out["roc_auc"] = roc_auc_score(y_true, y_score) if y_score is not None and len(np.unique(y_score)) > 1 else 0.5
    return out


def multiclass_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def timed_predict(model, X):
    start = time.perf_counter()
    pred = model.predict(X)
    score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return pred, score, (time.perf_counter() - start) / len(X) * 1000  # ms per sample


def plot_confusion(y_true, y_pred, labels, title, path, normalize=None):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, normalize=normalize, cmap="Blues", ax=ax, colorbar=False,
        values_format=".2f" if normalize else "d",
    )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_bars(results: pd.DataFrame, metric: str, path, hue="model", x="feature_set"):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=results, x=x, y=metric, hue=hue, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Held-out test {metric}")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
