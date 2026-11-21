"""CatSense: shelter decision-support demo.

Run from the project root:  streamlit run app/app.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from catsense import config as C
from catsense import data

st.set_page_config(page_title="CatSense", page_icon="🐱", layout="wide")

LIKERT = {
    "1 - Strongly disagree": 1, "2 - Somewhat disagree": 2, "3 - Neither agree nor disagree": 3,
    "4 - Somewhat agree": 4, "5 - Strongly agree": 5, "I don't know": np.nan,
}
DISCLAIMER = ("Decision support only. Predictions come from owner-reported survey data and are not a "
              "behavioural assessment. Always confirm with trained shelter staff.")


@st.cache_resource
def intake_bundle():
    return joblib.load(C.MODELS / "intake_model.joblib")


@st.cache_resource
def questions():
    return data.load_dictionary().set_index("column")["question"].to_dict()


def contributions(bundle, row: pd.DataFrame) -> pd.Series:
    """Per-feature contribution to the log-odds for the Logistic Regression intake model."""
    pipe = bundle["model"]
    z = pipe.named_steps["prep"].transform(row)[0]
    names = pipe.named_steps["prep"].get_feature_names_out()
    contrib = pd.Series(z * pipe.named_steps["clf"].coef_[0], index=names)
    return contrib[[n for n in names if n in bundle["items"]]].sort_values(key=np.abs, ascending=False)


def guidance(p_high: float) -> tuple[str, str]:
    if p_high >= 0.70:
        return "warning", ("Likely **higher fearfulness**. Consider a quiet adult home, a slow introduction plan, "
                           "a safe hiding space, and staff follow-up before adoption.")
    if p_high <= 0.30:
        return "success", ("Likely **lower fearfulness**. The cat may cope well with visitors and new places; "
                           "still match on the adopter's household and lifestyle.")
    return "info", "Uncertain prediction. Rely on direct observation by staff for this cat."


def intake_tab():
    bundle = intake_bundle()
    q = questions()
    st.subheader("Fearfulness intake form")
    st.caption(f"{len(bundle['items'])} questions plus demographics. "
               f"Held-out test macro-F1 {bundle['test_metrics']['macro_f1']:.2f}, "
               f"ROC-AUC {bundle['test_metrics']['roc_auc']:.2f}.")
    with st.form("intake"):
        c1, c2, c3 = st.columns(3)
        breed = c1.selectbox("Breed group", bundle["breed_groups"],
                             index=bundle["breed_groups"].index("House_cat") if "House_cat" in bundle["breed_groups"] else 0)
        sex = c2.selectbox("Sex", ["Female", "Male"])
        age = c3.number_input("Age (years, 0 if unknown)", 0.0, 25.0, 3.0, 0.5)
        c4, c5 = st.columns(2)
        other = c4.selectbox("Other cats in the household?", ["YES", "NO", "unknown"])
        problem = c5.selectbox("Owner-reported problem behaviour", ["No", "A_little", "Some", "A_lot", "unknown"])
        answers = {}
        for item in bundle["items"]:
            # "I don't know" is rare for most items and then signals fear in the training data,
            # so it is only offered where owners commonly used it (e.g. no children at home).
            options = [o for o in LIKERT if o != "I don't know" or bundle["item_missing_rate"][item] >= 0.05]
            default = options[int(bundle["item_median"][item]) - 1]
            answers[item] = LIKERT[st.select_slider(q.get(item, item), options, value=default, key=item)]
        submitted = st.form_submit_button("Predict", type="primary")
    if not submitted:
        return
    row = pd.DataFrame([{"BREEDGROUP": breed, "SEX": sex, "AGE_BEHAVIOUR": age or np.nan,
                         "OTHER_CATS": other, "problematic_behavior": problem, **answers}])
    p_high = float(bundle["model"].predict_proba(row[bundle["numeric"] + bundle["categorical"]])[0, 1])
    m1, m2 = st.columns([1, 2])
    m1.metric("Probability of higher fearfulness", f"{p_high:.0%}")
    kind, text = guidance(p_high)
    getattr(m2, kind)(text)
    contrib = contributions(bundle, row[bundle["numeric"] + bundle["categorical"]])
    st.markdown("**Answers that pushed the prediction most** (positive = towards higher fear)")
    st.bar_chart(contrib.rename(index=lambda n: q.get(n, n)[:70]).head(8), horizontal=True)
    st.caption(DISCLAIMER)


def image_tab():
    from catsense.vision import detect, predict
    st.subheader("Photo or video: emotional state")
    st.caption("YOLOv8 finds the cat, then the emotion model classifies the crop. Labels: "
               + ", ".join(C.EMOTIONS) + ". The training labels are noisy, so treat the result as a hint.")
    upload = st.file_uploader("Upload a cat photo or a short video", type=["jpg", "jpeg", "png", "webp", "mp4", "mov"])
    if upload is None:
        return
    if upload.type.startswith("video"):
        with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix) as tmp:
            tmp.write(upload.read())
            tmp.flush()
            with st.spinner("Analysing frames..."):
                result = predict.analyse_video(tmp.name)
        if result["emotion"] is None:
            st.error("No cat detected in the sampled frames.")
            return
        st.metric("Most frequent emotion", result["emotion"], f"{result['agreement']:.0%} of frames with a cat")
        st.bar_chart(pd.Series(result["counts"], name="frames"))
        st.caption(f"{result['frames_analysed']} frames analysed, cat found in {result['frames_with_cat']}.")
        return
    image = Image.open(upload).convert("RGB")
    with st.spinner("Detecting cat..."):
        result = predict.analyse_image(image, fallback_full_image=True)
    c1, c2 = st.columns(2)
    if result["n_cats"] == 0:
        c1.image(image, width="stretch")
        c2.warning("No cat detected. This can be a close-up the detector misses, or the photo may not show "
                   "a cat. The result below uses the whole picture and is less reliable.")
        c2.metric("Predicted emotion (unverified)", result["emotion"])
        c2.bar_chart(pd.Series(result["proba"], name="probability"))
        return
    c1.image(detect.draw_boxes(image, result["boxes"]), caption=f"{result['n_cats']} cat(s) found",
             width="stretch")
    c2.metric("Predicted emotion", result["emotion"])
    c2.bar_chart(pd.Series(result["proba"], name="probability"))
    c2.image(result["crop"], caption="Region analysed", width=200)


def about_tab():
    st.subheader("Models and evaluation")
    for title, file in [("Fearfulness models (held-out test, 20%)", "tabular_results.csv"),
                        ("Intake form length", "compact_form_results.csv"),
                        ("Emotion models (held-out test, 20%)", "vision_results.csv")]:
        path = C.RESULTS / file
        if path.exists():
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(path).drop(columns=["best_params"], errors="ignore").round(3), hide_index=True)
    st.markdown(
        "**Data.** Feline Behavior and Personality Survey (Mikkola et al., 2021, CC BY 4.0, 4,316 cats). "
        "Cat Emotion Dataset (Hugging Face, research use).  \n"
        "**Limitations.** Owner-reported answers; mostly Finnish cats and pedigree breeds; emotion labels are noisy. "
        f"{DISCLAIMER}"
    )


st.title("🐱 CatSense: shelter cat assessment")
tabs = st.tabs(["Fearfulness intake", "Photo / video", "About the models"])
with tabs[0]:
    intake_tab()
with tabs[1]:
    image_tab()
with tabs[2]:
    about_tab()
