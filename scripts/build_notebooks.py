"""Generates the project notebooks (run once, then execute with nbconvert).

Usage: python scripts/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from catsense import config as C, data
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40, "display.width", 160)"""


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    return nb


EDA = [
    "md:# 01 · Exploratory Data Analysis and Data Quality\n"
    "**Dataset:** Feline Behavior and Personality Survey Data, Mikkola et al. (2021), Figshare, CC BY 4.0.  \n"
    "**Target:** `fearfulness` factor score, binarised at the *training-set* median (lower vs. higher fear).",
    SETUP,
    "md:## 1. Loading and structure",
    "raw = data.load_raw()\nprint(raw.shape)\nraw.iloc[:5, :12]",
    "dictionary = data.load_dictionary()\ndictionary.groupby('section').size()",
    "df = data.load_clean()\nitems = data.item_columns(df)\n"
    "print(f'{len(df)} cats, {len(items)} survey items, {len(C.TRAIT_SCORES)} trait scores, '\n"
    "      f'{len(C.DEMOGRAPHIC_CATEGORICAL) + 1} demographic columns')\n"
    "df[items].stack().value_counts().sort_index()  # Likert answers 1-5",
    "md:## 2. Data quality\n### 2.1 Duplicates and inconsistent headers",
    "print('Duplicate CATIDs:', raw['CATID'].duplicated().sum())\n"
    "print('Duplicate rows (ignoring CATID):', raw.drop(columns='CATID').duplicated().sum())\n"
    "print('Headers with stray spaces:', [c for c in raw.columns if c != c.strip() or ' ' in c])",
    "md:Thirteen headers contain stray spaces, and the survey item `excessive_ grooming` collides with the trait score "
    "`excessive_grooming` once spaces are removed. `data.clean_column` normalises the names and renames the item to "
    "`excessive_grooming_item`.",
    "md:### 2.2 Missing values and 'unknown' codes",
    "miss = data.missing_report(df[items])\nprint((miss.missing_fraction > C.MAX_MISSING_FRACTION).sum(), "
    "'items exceed', C.MAX_MISSING_FRACTION)\nmiss.head(15)",
    "fig, ax = plt.subplots(figsize=(10, 3.5))\nax.hist(miss.missing_fraction, bins=30, color='#4C72B0')\n"
    "ax.axvline(C.MAX_MISSING_FRACTION, color='red', ls='--', label='drop threshold')\n"
    "ax.set(xlabel='Fraction missing', ylabel='Number of survey items', title='Missingness per survey item')\nax.legend()\n"
    "fig.savefig(C.FIGURES / 'eda_missingness.png', dpi=150, bbox_inches='tight')",
    "print(data.unknown_report(df))\nprint('AGE missing (\"unknown\"):', df.AGE_BEHAVIOUR.isna().sum())\n"
    "both = ((df.OTHER_CATS == 'unknown') & (df.problematic_behavior == 'unknown')).sum()\n"
    "print('Rows where both OTHER_CATS and problematic_behavior are unknown:', both)",
    "md:Missing item answers are mostly questions that do not apply to every household (e.g. behaviour towards dogs, "
    "kittens or babies). Items with more than 40% missing are dropped; remaining gaps are median-imputed **inside the "
    "pipeline** with a missing-indicator column, fitted on training folds only. The 997 rows with `unknown` for both "
    "`OTHER_CATS` and `problematic_behavior` appear to come from an earlier survey version; `unknown` is kept as its own "
    "category instead of being dropped.",
    "md:### 2.3 Outliers",
    "fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))\n"
    "sns.histplot(df.AGE_BEHAVIOUR, bins=40, ax=axes[0]); axes[0].set_title('Age (years)')\n"
    "sns.boxplot(data=df[C.TRAIT_SCORES], orient='h', ax=axes[1]); axes[1].set_title('Trait factor scores')\n"
    "fig.tight_layout(); fig.savefig(C.FIGURES / 'eda_outliers.png', dpi=150)\n"
    "q1, q3 = df.AGE_BEHAVIOUR.quantile([.25, .75]); iqr = q3 - q1\n"
    "print('Age IQR outliers:', (df.AGE_BEHAVIOUR > q3 + 1.5 * iqr).sum(), '| max age', df.AGE_BEHAVIOUR.max().round(1))",
    "md:Age outliers (cats up to ~22.7 years) are biologically plausible, so they are kept. Aggression, litterbox and "
    "grooming scores are strongly right-skewed (few cats with problems), but they are *not* used as features.",
    "md:## 3. Target: fearfulness",
    "train, test, threshold = data.split(df)\nprint(f'Training median threshold = {threshold:.4f}')\n"
    "print(pd.DataFrame({'train': train[C.TARGET].value_counts(), 'test': test[C.TARGET].value_counts()}))",
    "fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))\n"
    "sns.histplot(train.fearfulness, bins=40, ax=axes[0]); axes[0].axvline(threshold, color='red', ls='--', label='train median')\n"
    "axes[0].set_title('Fearfulness score (training split)'); axes[0].legend()\n"
    "train[C.TARGET].map({0: 'Lower', 1: 'Higher'}).value_counts().plot.bar(ax=axes[1], color=['#4C72B0', '#C44E52'], rot=0)\n"
    "axes[1].set_title('Class balance after median split')\n"
    "fig.tight_layout(); fig.savefig(C.FIGURES / 'eda_target.png', dpi=150)",
    "md:The median split yields balanced classes (50/50), so accuracy is meaningful, but macro-F1 is still reported.",
    "md:## 4. Demographics vs. fearfulness (training split only)",
    "t = train.assign(age_band=data.age_band(train.AGE_BEHAVIOUR))\n"
    "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n"
    "for ax, col in zip(axes.flat[:3], ['SEX', 'OTHER_CATS', 'age_band']):\n"
    "    sns.barplot(data=t, x=col, y=C.TARGET, ax=ax, errorbar=('ci', 95)); ax.set_ylabel('share higher fear')\n"
    "order = t.groupby('BREEDGROUP')[C.TARGET].mean().sort_values().index\n"
    "sns.barplot(data=t, y='BREEDGROUP', x=C.TARGET, order=order, ax=axes[1, 1], errorbar=None)\n"
    "axes[1, 1].tick_params(axis='y', labelsize=6); axes[1, 1].set_xlabel('share higher fear')\n"
    "fig.tight_layout(); fig.savefig(C.FIGURES / 'eda_demographics.png', dpi=150)",
    "md:## 5. Behaviour items vs. fearfulness and leakage check",
    "corr = train[items].corrwith(train.fearfulness).sort_values(key=abs, ascending=False)\n"
    "leak = pd.DataFrame({'corr_with_fearfulness': corr, 'fear_item (excluded)': corr.index.isin(C.FEAR_ITEMS)})\n"
    "leak.head(25)",
    "fig, ax = plt.subplots(figsize=(8, 6))\ntop = leak.head(30).iloc[::-1]\n"
    "ax.barh(top.index, top.corr_with_fearfulness, color=np.where(top['fear_item (excluded)'], '#C44E52', '#4C72B0'))\n"
    "ax.set_title('Top-30 item correlations with fearfulness (red = defines the target, excluded)')\n"
    "ax.tick_params(axis='y', labelsize=7); fig.tight_layout(); fig.savefig(C.FIGURES / 'eda_item_correlations.png', dpi=150)",
    "md:The 19 items that load on the Fearfulness factor in Mikkola et al. (2021, Table 1) dominate the correlations "
    "(|r| up to 0.83). They are removed from all feature sets. The other six trait scores are also excluded because "
    "they are computed with the tenBerge method from the same item responses.",
    "keep = [c for c in items if c not in C.FEAR_ITEMS]\ntop_keep = corr[keep].abs().sort_values(ascending=False).index[:6]\n"
    "fig, axes = plt.subplots(2, 3, figsize=(12, 6))\n"
    "for ax, col in zip(axes.flat, top_keep):\n"
    "    sns.boxplot(data=train, x=C.TARGET, y=col, ax=ax); ax.set_xticklabels(['Lower', 'Higher']); ax.set_xlabel('')\n"
    "fig.suptitle('Strongest non-target items by fear class'); fig.tight_layout()\n"
    "fig.savefig(C.FIGURES / 'eda_boxplots.png', dpi=150)",
    "md:### Correlation between candidate predictors (multicollinearity)",
    "sel = corr[keep].abs().sort_values(ascending=False).index[:20]\n"
    "fig, ax = plt.subplots(figsize=(10, 8))\n"
    "sns.heatmap(train[list(sel)].corr(), cmap='vlag', center=0, ax=ax, xticklabels=True, yticklabels=True)\n"
    "ax.tick_params(labelsize=7); fig.tight_layout(); fig.savefig(C.FIGURES / 'eda_heatmap.png', dpi=150)",
    "md:## 6. Summary of decisions\n"
    "| Finding | Decision |\n|---|---|\n"
    "| 0 duplicate rows / IDs | nothing removed |\n"
    "| 13 messy headers, 1 name collision | normalised, item renamed |\n"
    "| 13 items > 40% missing (not applicable questions) | dropped |\n"
    "| remaining missing answers | median + missing indicator, fitted on train folds |\n"
    "| 997 `unknown` in two demographics, 123 unknown ages | `unknown` kept as category; age median-imputed |\n"
    "| plausible age outliers | kept |\n"
    "| 19 target-defining items + 6 other factor scores | excluded (leakage) |\n"
    "| balanced target after median split | stratified split, macro-F1 + AUC reported |\n"
    "| Likert scales differ in spread | standardised for Logistic Regression only |",
]

MODELING = [
    "md:# 02 · Modelling: fearfulness classification\n"
    "Runs `scripts/train_tabular.py` (baseline, Logistic Regression, Random Forest × three feature sets, "
    "5-fold CV tuning on the 80% training split, one evaluation on the 20% held-out split).",
    SETUP + "\nfrom IPython.display import Image, display",
    "!cd .. && {sys.executable} scripts/train_tabular.py 2>/dev/null | tail -30",
    "res = pd.read_csv(C.RESULTS / 'tabular_results.csv')\n"
    "res.drop(columns='best_params').round(3)",
    "res[['feature_set', 'model', 'best_params']]",
    "md:## Comparison",
    "display(Image(C.FIGURES / 'tabular_macro_f1.png'))\ndisplay(Image(C.FIGURES / 'tabular_roc_auc.png'))",
    "md:## Confusion matrices (combined features)",
    "for m in ['baseline', 'logreg', 'random_forest']:\n    display(Image(C.FIGURES / f'cm_combined_{m}.png', width=330))",
    "md:## What drives the predictions?",
    "display(Image(C.FIGURES / 'permutation_importance.png'))\ndisplay(Image(C.FIGURES / 'logreg_coefficients.png'))",
    "pd.read_csv(C.RESULTS / 'rf_impurity_importance.csv').head(15)",
    "md:## How short can a shelter intake form be?",
    "display(Image(C.FIGURES / 'compact_form_tradeoff.png'))\npd.read_csv(C.RESULTS / 'compact_form_results.csv').round(3)",
    "md:## Error analysis by group (bias check)",
    "sub = pd.read_csv(C.RESULTS / 'subgroup_errors.csv')\nsub.round(3)",
    "md:## Leakage demonstration\nIncluding the 19 target-defining items gives near-perfect scores, which shows why they "
    "must be excluded: such a model would only rebuild the questionnaire score, not learn anything new.",
    "res[res.feature_set.str.startswith('LEAKY')].drop(columns='best_params').round(3)",
]

VISION = [
    "md:# 03 · Image module: cat emotion from photos\n"
    "Supporting component inspired by DeepCat (Cat-Emotional-Analysis). YOLOv8 finds the cat, then a classifier "
    "predicts one of four emotions. Data: Cat Emotion Dataset (Hugging Face `KAITANG2003/cat-emotion-dataset`, "
    "research use only). Labels in that dataset appear to be model-generated from visual descriptions, so they are "
    "noisy; results are treated as indicative only.",
    SETUP + "\nfrom IPython.display import Image, display",
    "q = pd.read_csv(C.RESULTS / 'vision_data_quality.csv')\nq",
    "md:Images where YOLO finds no cat (including non-cat images such as dogs) are removed before training.",
    "vr = pd.read_csv(C.RESULTS / 'vision_results.csv')\nvr.round(3)",
    "display(Image(C.FIGURES / 'vision_macro_f1.png'))",
    "import glob\nfor f in sorted(glob.glob(str(C.FIGURES / 'cm_vision_*.png'))):\n    display(Image(f, width=380))",
]

if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in [("01_eda.ipynb", EDA), ("02_modeling.ipynb", MODELING), ("03_vision.ipynb", VISION)]:
        nbf.write(notebook(cells), NB_DIR / name)
        print("wrote", NB_DIR / name)
