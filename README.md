# CatGPT: Predicting Cat Fearfulness for Shelter Decision Support

DSAI 310 Final Project, Fall 2026, American University of Bahrain
Alya Altawash (F2300202), Wajd Alballal (F2300050), Shadow Alghamdi (F2300047)

CatGPT predicts whether a cat has **lower or higher fearfulness** from owner-reported behaviour and
demographic data. A shelter could use the prediction as one extra input when matching cats with adopters.
A supporting image module, adapted from the
[DeepCat / Cat-Emotional-Analysis](https://ieeexplore.ieee.org/document/10479630) project, detects a cat in
a photo or video and estimates its momentary emotional state.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py            # survey (2.7 MB) + emotion images (2.1 GB)
python scripts/download_data.py --skip-images   # survey only

python scripts/train_tabular.py            # main project: ~1.5 min on a laptop
python scripts/train_vision.py             # image module: YOLO + embeddings + fine-tuning
python scripts/build_notebooks.py          # (re)generate notebooks
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

streamlit run app/app.py                   # demo app
```

Tested with Python 3.14 on macOS (Apple M3, PyTorch MPS). Exact package versions are in `requirements.txt`.
All randomness uses seed 42.

## Project structure

```
src/catsense/
  config.py          paths, constants, the 19 target-defining items (leakage list)
  data.py            loading, header cleanup, type fixes, train/test split + median threshold
  features.py        feature sets and preprocessing pipelines
  models.py          baseline, Logistic Regression, Random Forest + hyperparameter grids
  evaluate.py        metrics, confusion matrices, comparison plots
  vision/detect.py   YOLOv8 cat detection and cropping (adapted from DeepCat image.py)
  vision/emotion.py  ResNet-50 embeddings, fine-tuned ResNet-18, frame majority vote
  vision/predict.py  photo / video inference used by the app (adapted from DeepCat video.py)
scripts/             download, training and notebook-building entry points
notebooks/           01_eda, 02_modeling, 03_vision (executed, with outputs)
app/app.py           Streamlit app: intake form, photo/video analysis, model card
reports/figures/     all generated figures
reports/results/     result tables (CSV) and run logs
models/              saved models used by the app
```

## Data

| Dataset | Use | Size | Licence |
|---|---|---|---|
| [Feline Behavior and Personality Survey](https://figshare.com/articles/dataset/Feline_behavior_and_personality_survey_data/14899077) (Mikkola et al., 2021) | main project | 4,316 cats, 138 survey items, 7 trait scores | CC BY 4.0 |
| [Cat Emotion Dataset](https://huggingface.co/datasets/KAITANG2003/cat-emotion-dataset) (Hugging Face) | image module | 13,592 images, 4 emotions | research use only |

## Method (main project)

1. **Cleaning.** Fix 13 headers with stray spaces and one name collision (`excessive_ grooming` item vs. trait
   score). Parse `"unknown"` ages as missing. No duplicate rows were found.
2. **Split.** 80/20, stratified on fearfulness quintiles, seed 42. The median fearfulness of the **training**
   split (−0.1996) is the class threshold for both splits, which gives balanced classes (50/50).
3. **Leakage control.** The 19 items that load on the Fearfulness factor in Mikkola et al. (2021, Table 1) are
   removed, together with all 7 factor scores (they are computed from the same answers).
4. **Preprocessing inside the pipeline** (fitted on training folds only): drop items with more than 40% missing
   answers, median imputation with missing indicators, one-hot encoding (rare breed groups pooled), and
   standard scaling for Logistic Regression only.
5. **Models.** Majority-class baseline; Logistic Regression (C, L1 vs. L2); Random Forest (trees, depth, split
   and leaf sizes). Hyperparameters are tuned by 5-fold stratified CV on macro-F1.
6. **Feature sets (ablation).** `demographic` (breed group, sex, age, other cats, problem behaviour),
   `behaviour` (106 non-fear items), `combined` (both).
7. **Extras.** Permutation importance, LR coefficients, RF impurity importance, error rates by sex, age, breed
   and household (bias check), an intake-form length study, and a leakage demonstration.

## Results (held-out test set, n = 864)

| Feature set | Model | CV macro-F1 | Test accuracy | Test macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| demographic | baseline | 0.333 | 0.497 | 0.332 | 0.500 |
| demographic | Logistic Regression | 0.589 | 0.571 | 0.570 | 0.592 |
| demographic | Random Forest | 0.591 | 0.594 | 0.594 | 0.629 |
| behaviour | Logistic Regression | 0.758 | 0.751 | 0.751 | 0.844 |
| behaviour | Random Forest | 0.750 | 0.764 | 0.763 | 0.843 |
| combined | Logistic Regression | **0.759** | 0.758 | 0.758 | 0.844 |
| combined | Random Forest | 0.752 | 0.758 | 0.758 | 0.844 |
| *leaky (fear items included)* | *Logistic Regression* | *0.977* | *0.981* | *0.981* | *0.999* |

* Demographics alone carry weak signal (macro-F1 about 0.59). Behaviour that does not define the target lifts
  macro-F1 to about 0.76 (AUC 0.84).
* Logistic Regression and Random Forest perform the same, so the relationship is mostly linear. Logistic
  Regression is selected by CV score and is simpler to explain to shelter staff.
* **Intake-form study:** a 10-question form plus demographics reaches a test macro-F1 of 0.757, matching the
  full 106-item model (`reports/figures/compact_form_tradeoff.png`). The app uses this form.
* Including the 19 target items gives a near-perfect score, which confirms that they had to be excluded.

Full tables: `reports/results/tabular_results.csv`, `compact_form_results.csv`, `subgroup_errors.csv`.

**Error analysis by group** (`subgroup_errors.csv`, selected model): accuracy is similar for females (0.76)
and males (0.75). It drops for cats aged 11 and over (0.68). Kittens under 1 year have a false-negative rate
of 0.50, so half of the fearful kittens are missed. Breed groups with fewer than 40 test cats vary widely
(accuracy 0.64 to 0.86), so those estimates are unreliable.

## Image module

Data: 4,172 images sampled from the Cat Emotion Dataset, capped at 1,500 per class (Relaxed, Happy, Aggressive,
Sad). YOLOv8m found a cat in 98% of Relaxed, 94% of Happy, 89% of Aggressive and only 66% of Sad images. The
dataset contains non-cat images, such as dogs and horses, and these are removed. The remaining images are split
68/12/20 into train, validation and test. The model is selected on validation macro-F1.

| Model | Test accuracy | Test macro-F1 | Balanced accuracy | ms per image |
|---|---|---|---|---|
| Majority baseline | 0.386 | 0.139 | 0.250 | – |
| ResNet-50 embeddings + LR, full image | 0.433 | 0.399 | 0.417 | 0.01 (+ embedding) |
| ResNet-50 embeddings + LR, YOLO crop | 0.457 | 0.434 | 0.456 | 0.01 (+ embedding) |
| **Fine-tuned ResNet-18, YOLO crop** | **0.542** | **0.515** | 0.514 | 4.3 |

* Cropping the cat with YOLO adds +0.035 macro-F1 over the full image (detection ablation).
* Relaxed and Happy are confused most often (about 30% each way). Sad has the lowest recall (0.41).
* Fine-tuning reaches a training loss of 0.23 but a test macro-F1 of only 0.52. This overfitting points to noisy
  labels.
* YOLOv8n missed many close-up faces. YOLOv8m recovered 41% of those misses, so v8m is used. In the app, a
  photo with no detected cat is still classified on the whole image, and the result is flagged as unverified.

## What we took from DeepCat and what we changed

| DeepCat | CatGPT |
|---|---|
| YOLOv5 custom weights for eyes, ears, tail and mouth (weights not published) | COCO-pretrained YOLOv8m detects whole cats; the highest-confidence box is kept (`list_spilter_Images` logic) |
| Hand-written OpenCV rules (eye white ratio, tail angle) | Learned classifiers (ResNet-50 embeddings + LR, fine-tuned ResNet-18) evaluated on a held-out split |
| Video: per-frame counters, most frequent case wins | Kept: frame sampling + majority vote (`analyse_video`) |
| Flask endpoint, Pygame UI, Flutter app | Replaced by one Streamlit app |
| No quantitative evaluation | Baseline, two approaches, a detection ablation, macro-F1 and confusion matrices |
| — | Main contribution: a tabular fearfulness model for adoption decisions |

## Limitations and responsible use

* Survey answers are owner-reported, and most respondents are in Finland with many pedigree cats. Shelter cats
  with unknown histories may differ.
* The fearfulness label is a binarised factor score, not a clinical assessment.
* The emotion labels appear to be model-generated from image descriptions and are noisy. Some images are not
  cats; the YOLO filter removes them.
* Predictions should support staff judgement, never replace it, and should never be used to reject a cat
  for adoption.
