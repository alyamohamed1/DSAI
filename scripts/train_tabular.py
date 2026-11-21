"""Fearfulness classification experiments (Experiments 1-5 of the progress report).

Runs baseline / Logistic Regression / Random Forest on three feature sets,
tunes with 5-fold CV on the training split, evaluates once on the held-out
20% test split, and saves tables, figures and the selected model.

Usage: python scripts/train_tabular.py
"""
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from catsense import config as C
from catsense import data, evaluate
from catsense.features import FEATURE_SETS, feature_columns, usable_items
from catsense.models import MODELS, build

LABELS = ["Lower fear", "Higher fear"]


def run_grid(model_name, numeric, categorical, X, y):
    pipe, grid = build(model_name, numeric, categorical)
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    search = GridSearchCV(pipe, grid or {"clf__strategy": ["most_frequent"]}, scoring="f1_macro", cv=cv, n_jobs=-1)
    start = time.perf_counter()
    search.fit(X, y)
    return search, time.perf_counter() - start


def subgroup_errors(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    df = test.assign(pred=pred, age_band=data.age_band(test["AGE_BEHAVIOUR"]).astype("string"))
    top_breeds = df["BREEDGROUP"].value_counts().index[:8]
    df["breed_top8"] = df["BREEDGROUP"].where(df["BREEDGROUP"].isin(top_breeds), "Other breeds")
    rows = []
    for col in ["SEX", "age_band", "OTHER_CATS", "breed_top8"]:
        for value, g in df.groupby(col, dropna=False):
            pos, neg = g[C.TARGET] == 1, g[C.TARGET] == 0
            rows.append({
                "attribute": col, "group": value, "n": len(g),
                "high_fear_rate": g[C.TARGET].mean(),
                "accuracy": (g["pred"] == g[C.TARGET]).mean(),
                "false_negative_rate": ((g["pred"] == 0) & pos).sum() / max(pos.sum(), 1),
                "false_positive_rate": ((g["pred"] == 1) & neg).sum() / max(neg.sum(), 1),
            })
    return pd.DataFrame(rows)


def main():
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    C.MODELS.mkdir(parents=True, exist_ok=True)

    df = data.load_clean()
    train, test, threshold = data.split(df)
    y_train, y_test = train[C.TARGET], test[C.TARGET]
    print(f"train={len(train)} test={len(test)} median threshold={threshold:.4f} "
          f"high-fear share train={y_train.mean():.3f} test={y_test.mean():.3f}")

    rows, fitted = [], {}
    for fs in FEATURE_SETS:
        numeric, categorical = feature_columns(fs, train)
        for m in MODELS:
            search, fit_s = run_grid(m, numeric, categorical, train, y_train)
            pred, score, ms = evaluate.timed_predict(search.best_estimator_, test)
            metrics = evaluate.binary_metrics(y_test, pred, score)
            rows.append({
                "feature_set": fs, "model": m, "n_features": len(numeric) + len(categorical),
                "cv_macro_f1": search.best_score_, **metrics,
                "tuning_time_s": fit_s, "predict_ms_per_cat": ms,
                "best_params": json.dumps({k.replace("clf__", ""): v for k, v in search.best_params_.items()}),
            })
            fitted[(fs, m)] = search.best_estimator_
            evaluate.plot_confusion(y_test, pred, LABELS, f"{m} | {fs}", C.FIGURES / f"cm_{fs}_{m}.png")
            print(f"{fs:12s} {m:14s} cv_f1={search.best_score_:.3f} test_f1={metrics['macro_f1']:.3f} "
                  f"auc={metrics['roc_auc']:.3f} ({fit_s:.1f}s)")

    # Leakage demonstration: what happens if the 19 items that define the target are used
    leak_items = [c for c in data.item_columns(train) if train[c].isna().mean() <= C.MAX_MISSING_FRACTION]
    search, fit_s = run_grid("logreg", leak_items, [], train, y_train)
    pred, score, ms = evaluate.timed_predict(search.best_estimator_, test)
    rows.append({"feature_set": "LEAKY (incl. fear items)", "model": "logreg", "n_features": len(leak_items),
                 "cv_macro_f1": search.best_score_, **evaluate.binary_metrics(y_test, pred, score),
                 "tuning_time_s": fit_s, "predict_ms_per_cat": ms, "best_params": json.dumps(search.best_params_)})

    results = pd.DataFrame(rows)
    results.to_csv(C.RESULTS / "tabular_results.csv", index=False)
    clean_results = results[~results.feature_set.str.startswith("LEAKY")]
    evaluate.plot_metric_bars(clean_results, "macro_f1", C.FIGURES / "tabular_macro_f1.png")
    evaluate.plot_metric_bars(clean_results, "roc_auc", C.FIGURES / "tabular_roc_auc.png")

    # Model selection uses CV score only (test set is not used to choose)
    best = clean_results[clean_results.model != "baseline"].sort_values("cv_macro_f1", ascending=False).iloc[0]
    best_fs, best_m = best.feature_set, best.model
    best_model = fitted[(best_fs, best_m)]
    print(f"selected by CV: {best_m} on {best_fs}")

    # Feature importance: permutation importance on the test split for the selected model,
    # plus RF impurity importance and LR coefficients on the combined set
    numeric, categorical = feature_columns(best_fs, train)
    perm = permutation_importance(best_model, test[numeric + categorical], y_test, scoring="f1_macro",
                                  n_repeats=10, random_state=C.RANDOM_STATE, n_jobs=-1)
    perm_df = (pd.DataFrame({"feature": numeric + categorical, "importance": perm.importances_mean,
                             "std": perm.importances_std})
               .sort_values("importance", ascending=False))
    perm_df.to_csv(C.RESULTS / "permutation_importance.csv", index=False)
    top = perm_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.feature, top.importance, xerr=top["std"], color="#4C72B0")
    ax.set_xlabel("Mean drop in macro-F1 when shuffled")
    ax.set_title(f"Permutation importance ({best_m}, {best_fs})")
    fig.tight_layout(); fig.savefig(C.FIGURES / "permutation_importance.png", dpi=150); plt.close(fig)

    rf = fitted[("combined", "random_forest")]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    imp.rename("importance").to_csv(C.RESULTS / "rf_impurity_importance.csv")
    lr = fitted[("combined", "logreg")]
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=lr.named_steps["prep"].get_feature_names_out())
    coef.sort_values(key=np.abs, ascending=False).rename("coefficient").to_csv(C.RESULTS / "logreg_coefficients.csv")
    top_coef = coef.sort_values(key=np.abs, ascending=False).head(15).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_coef.index, top_coef.values, color=np.where(top_coef.values > 0, "#C44E52", "#4C72B0"))
    ax.set_xlabel("Standardised coefficient (+ = more likely higher fear)")
    ax.set_title("Logistic Regression coefficients (combined features)")
    fig.tight_layout(); fig.savefig(C.FIGURES / "logreg_coefficients.png", dpi=150); plt.close(fig)

    # Error analysis by group (bias check for the ethics section)
    pred = best_model.predict(test)
    subgroup_errors(test, pred).to_csv(C.RESULTS / "subgroup_errors.csv", index=False)

    joblib.dump({
        "model": best_model, "feature_set": best_fs, "model_name": best_m, "threshold": threshold,
        "numeric": numeric, "categorical": categorical, "items": usable_items(train),
        "train_medians": train[numeric].median().to_dict(),
    }, C.MODELS / "fearfulness_model.joblib")
    print(results.drop(columns="best_params").round(3).to_string(index=False))

    # Compact intake form: how many survey questions does a shelter really need?
    # Items ranked by |coefficient| of the combined Logistic Regression (fitted on training data only).
    item_rank = [f for f in coef.abs().sort_values(ascending=False).index if f in usable_items(train)]
    compact_rows, compact_fitted = [], {}
    for k in [3, 5, 10, 15, 20, 30, len(item_rank)]:
        items_k = item_rank[:k]
        num_k = list(C.DEMOGRAPHIC_NUMERIC) + items_k
        search, fit_s = run_grid("logreg", num_k, list(C.DEMOGRAPHIC_CATEGORICAL), train, y_train)
        pred, score, ms = evaluate.timed_predict(search.best_estimator_, test)
        compact_rows.append({"k_items": k, "cv_macro_f1": search.best_score_,
                             **evaluate.binary_metrics(y_test, pred, score)})
        compact_fitted[k] = (search.best_estimator_, items_k)
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(C.RESULTS / "compact_form_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(compact.k_items, compact.cv_macro_f1, "o-", label="5-fold CV (train)")
    ax.plot(compact.k_items, compact.macro_f1, "s--", label="Held-out test")
    ax.set_xscale("log"); ax.set_xticks(compact.k_items); ax.set_xticklabels(compact.k_items)
    ax.set_xlabel("Number of survey items (plus demographics)"); ax.set_ylabel("Macro-F1")
    ax.set_title("Accuracy vs. intake-form length (Logistic Regression)"); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIGURES / "compact_form_tradeoff.png", dpi=150); plt.close(fig)
    # Smallest form within 0.02 macro-F1 of the best CV score (chosen without the test set)
    k_best = int(compact[compact.cv_macro_f1 >= compact.cv_macro_f1.max() - 0.02].k_items.min())
    model_k, items_k = compact_fitted[k_best]
    joblib.dump({
        "model": model_k, "threshold": threshold, "items": items_k,
        "numeric": list(C.DEMOGRAPHIC_NUMERIC) + items_k, "categorical": list(C.DEMOGRAPHIC_CATEGORICAL),
        "breed_groups": sorted(train["BREEDGROUP"].dropna().unique()),
        # Used by the app for default answers and to decide where "I don't know" is offered
        "item_median": train[items_k].median().to_dict(),
        "item_missing_rate": train[items_k].isna().mean().to_dict(),
        "test_metrics": compact.set_index("k_items").loc[k_best].to_dict(),
    }, C.MODELS / "intake_model.joblib")
    print(compact.round(3).to_string(index=False))
    print(f"intake form uses {k_best} items: {items_k}")


if __name__ == "__main__":
    main()
