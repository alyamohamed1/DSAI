"""Loading, cleaning and splitting the Feline Behavior and Personality data."""
import re
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def download_feline(force: bool = False):
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if force or not C.FELINE_XLSX.exists():
        urllib.request.urlretrieve(C.FELINE_URL, C.FELINE_XLSX)
    return C.FELINE_XLSX


def clean_column(name: str) -> str:
    # Raw headers contain stray spaces, e.g. "Runs_ hides_noise_inside", "easily_scared_things_sounds "
    if name == "excessive_ grooming":
        return "excessive_grooming_item"  # survey item; would otherwise collide with the trait score
    name = re.sub(r"\s*_\s*", "_", name.strip())
    return re.sub(r"\s+", "_", name)


def load_raw() -> pd.DataFrame:
    download_feline()
    return pd.read_excel(C.FELINE_XLSX, sheet_name="Dataset")


def load_dictionary() -> pd.DataFrame:
    info = pd.read_excel(C.FELINE_XLSX, sheet_name="Information")
    info.columns = ["section", "column", "question", "values"]
    info["section"] = info["section"].ffill()
    info = info.dropna(subset=["column"])
    info["column"] = info["column"].map(clean_column)
    return info


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix headers and types. Does not impute: imputation is fitted on training data only."""
    df = df.rename(columns=clean_column).copy()
    df["AGE_BEHAVIOUR"] = pd.to_numeric(df["AGE_BEHAVIOUR"], errors="coerce")
    for col in C.DEMOGRAPHIC_CATEGORICAL:
        df[col] = df[col].astype("string").str.strip()
    for col in item_columns(df):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset=[c for c in df.columns if c != "CATID"])


def item_columns(df: pd.DataFrame) -> list[str]:
    meta = {"CATID", C.TARGET, *C.DEMOGRAPHIC_CATEGORICAL, *C.DEMOGRAPHIC_NUMERIC, *C.TRAIT_SCORES}
    return [c for c in df.columns if c not in meta]


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


def split(df: pd.DataFrame):
    """80/20 split stratified on fearfulness quintiles, then median threshold from train only."""
    bins = pd.qcut(df[C.TARGET_SCORE], 5, labels=False)
    train, test = train_test_split(df, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=bins)
    threshold = float(train[C.TARGET_SCORE].median())
    train, test = train.copy(), test.copy()
    for part in (train, test):
        part[C.TARGET] = (part[C.TARGET_SCORE] > threshold).astype(int)
    return train, test, threshold


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().rename("missing_fraction").to_frame()
    miss["missing_count"] = df.isna().sum()
    return miss.sort_values("missing_fraction", ascending=False)


def unknown_report(df: pd.DataFrame) -> pd.Series:
    return pd.Series({c: int((df[c] == "unknown").sum()) for c in C.DEMOGRAPHIC_CATEGORICAL})


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 1, 3, 7, 11, np.inf], labels=["<1", "1-3", "3-7", "7-11", "11+"], right=False)
