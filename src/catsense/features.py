"""Feature sets and preprocessing pipelines (fitted inside CV folds, never on test data)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .data import item_columns

FEATURE_SETS = ("demographic", "behaviour", "combined")


def usable_items(train: pd.DataFrame) -> list[str]:
    """Non-fear survey items with acceptable missingness, decided on the training split."""
    items = [c for c in item_columns(train) if c not in C.FEAR_ITEMS]
    missing = train[items].isna().mean()
    return [c for c in items if missing[c] <= C.MAX_MISSING_FRACTION]


def feature_columns(feature_set: str, train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Returns (numeric, categorical) columns for a feature set."""
    if feature_set == "demographic":
        return list(C.DEMOGRAPHIC_NUMERIC), list(C.DEMOGRAPHIC_CATEGORICAL)
    if feature_set == "behaviour":
        return usable_items(train), []
    if feature_set == "combined":
        return list(C.DEMOGRAPHIC_NUMERIC) + usable_items(train), list(C.DEMOGRAPHIC_CATEGORICAL)
    raise ValueError(f"unknown feature set {feature_set!r}")


def preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)]
    if categorical:
        cat = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            # Rare breed groups pooled to avoid near-empty one-hot columns
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)),
        ])
        transformers.append(("cat", cat, categorical))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)
