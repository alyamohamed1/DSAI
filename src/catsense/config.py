"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_IMAGES = ROOT / "data" / "images"
FELINE_XLSX = DATA_RAW / "Feline_dataset.xlsx"
FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "reports" / "results"
MODELS = ROOT / "models"

FELINE_URL = "https://ndownloader.figshare.com/files/33014249"
IMAGES_BASE_URL = "https://huggingface.co/datasets/KAITANG2003/cat-emotion-dataset/resolve/main"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_SCORE = "fearfulness"
TARGET = "high_fear"  # 1 = score above the training median

DEMOGRAPHIC_CATEGORICAL = ["BREEDGROUP", "SEX", "OTHER_CATS", "problematic_behavior"]
DEMOGRAPHIC_NUMERIC = ["AGE_BEHAVIOUR"]

# Factor scores published with the dataset. They are computed from the item
# responses (tenBerge method), so none of them may be used as predictors.
TRAIT_SCORES = [
    "fearfulness", "human_aggression", "activity_playfulness", "cat_sociability",
    "human_sociability", "litterbox_issues", "excessive_grooming",
]

# The 19 items that load (|loading| >= 0.30) on the Fearfulness factor in
# Mikkola et al. (2021), Table 1, mapped to the dataset's column names
# (after whitespace cleanup). These form the target and are excluded.
FEAR_ITEMS = [
    "Growls_hisses_unfamiliar_adults",
    "Growls_hisses_unfamiliar_person_pet",
    "not_approach_people_right_away",          # "Suspicious"
    "Escapes_unfamiliar_people",
    "Freezes_examined_veterinarian",
    "Hides_household_member_leave",
    "Runs_hide_noise_outside",
    "restlessness_home_modified",
    "restlessness_unfamiliar_objects",
    "Stares_unfamiliar_people",
    "easily_scared_things_sounds",
    "Runs_hides_noise_inside",
    "greets_unfamiliar_adults",
    "greets_unfamiliar_children",
    "comfortable_social_gatherings",
    "comfortable_petted_unfamiliar_people",
    "comfortable_unfamiliar_places",
    "Seems_confident",                         # "Confident at home"
    "comfortable_playing_unfamiliar_people",
]

# Items with more missing answers than this are dropped (mostly questions
# that only apply to some households, e.g. cats living with dogs).
MAX_MISSING_FRACTION = 0.40

# Image module
EMOTIONS = ["Relaxed", "Happy", "Aggressive", "Sad"]
YOLO_WEIGHTS = "yolov8m.pt"  # v8m recovered 41% of the cats v8n missed (close-ups, dark fur)
COCO_CAT_CLASS = 15
