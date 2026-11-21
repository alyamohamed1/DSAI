"""Baseline and classifiers with their hyperparameter grids."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config as C
from .features import preprocessor


def build(model: str, numeric: list[str], categorical: list[str]) -> tuple[Pipeline, dict]:
    """Returns (pipeline, param_grid) for GridSearchCV."""
    if model == "baseline":
        clf, grid, scale = DummyClassifier(strategy="most_frequent"), {}, False
    elif model == "logreg":
        clf = LogisticRegression(max_iter=5000, solver="liblinear", random_state=C.RANDOM_STATE)
        grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10], "clf__l1_ratio": [0.0, 1.0]}  # 0 = L2 (ridge), 1 = L1 (lasso)
        scale = True
    elif model == "random_forest":
        clf = RandomForestClassifier(random_state=C.RANDOM_STATE, n_jobs=-1)
        grid = {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 10],
            "clf__min_samples_leaf": [1, 5],
        }
        scale = False  # trees are scale-invariant
    else:
        raise ValueError(f"unknown model {model!r}")
    return Pipeline([("prep", preprocessor(numeric, categorical, scale)), ("clf", clf)]), grid


MODELS = ("baseline", "logreg", "random_forest")
