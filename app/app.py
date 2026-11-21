"""CatGPT: shelter decision-support demo.

Run from the project root:  streamlit run app/app.py
"""
import base64
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

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="CatGPT", page_icon=str(ASSETS / "silhouette.png"), layout="wide")

LIKERT = {
    "1 - Strongly disagree": 1, "2 - Somewhat disagree": 2, "3 - Neither agree nor disagree": 3,
    "4 - Somewhat agree": 4, "5 - Strongly agree": 5, "I don't know": np.nan,
}
DISCLAIMER = ("Decision support only. Predictions come from owner-reported survey data and are not a "
              "behavioural assessment. Always confirm with trained shelter staff.")
GREEN = "#b8e0b0"

# Body-language cues: (image file, label, meaning, signal). Signal is +1 for a fearful or defensive
# sign, -1 for a calm or confident sign and 0 when the cue is ambiguous on its own.
CUES = [
    ("00_rubbing", "Rubbing against you", "Seeking contact", -1),
    ("01_purring", "Purring", "Often content or self-soothing", -1),
    ("02_stretch", "A big stretch", "Loosening up", -1),
    ("03_kneading", "Kneading", "Getting comfortable", -1),
    ("04_sitting_upright", "Sitting upright", "Calm and attentive", -1),
    ("05_splooting", "Splooting", "Stretching out to relax", -1),
    ("06_legs_out", "Legs-out sitting", "Relaxing or grooming", -1),
    ("07_belly_up", "Belly up", "Often feeling at ease", -1),
    ("08_lounging", "Lounging sideways", "Looking relaxed", -1),
    ("09_belly_down", "Belly-down resting", "Resting but watchful", 0),
    ("10_tail_up", "Tail straight up", "A friendly approach", -1),
    ("11_scent_face", "The scent face", "Analysing a scent", 0),
    ("12_rolling", "Rolling around", "Playful or seeking attention", -1),
    ("13_scratching", "Self-scratching", "Feeling itchy", 0),
    ("14_low_crouch", "Low crouch", "Getting ready to pounce", 0),
    ("15_paws_tucked", "Paws tucked in", "Often resting comfortably", -1),
    ("16_tense_hunch", "Tense hunch", "May be uncomfortable", 1),
    ("17_nipping", "Nipping", "Play or overstimulation", 0),
    ("18_wide_pupils", "Wide pupils", "Arousal or low light", 0),
    ("19_tail_flicks", "Tail-tip flicks", "Focused or irritated", 0),
    ("20_tucked_tail", "Tucked tail", "Feeling unsure", 1),
    ("21_airplane_ears", "Airplane ears", "Uneasy or on alert", 1),
    ("22_lashing_tail", "Lashing tail", "Often overstimulated", 0),
    ("23_arched", "Arched and puffed up", "Startled or defensive", 1),
    ("24_hissing", "Hissing", "Needs more space", 1),
]


@st.cache_data
def img_uri(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()


def inject_css():
    # Markdown ends an HTML block at a blank line, so blank lines are stripped before rendering.
    css = f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=VT323&display=swap");
:root {{
  --bg: #17221c; --panel: #1d2b23; --line: #8fbf8a; --text: {GREEN}; --dim: #7fa37a; --ink: #17221c;
}}
html, body, .stApp, .stMarkdown, p, label, input, textarea, button, li, td, th {{
  font-family: 'VT323', ui-monospace, monospace !important;
}}
.stApp {{ background: var(--bg); color: var(--text); }}
.stApp p, .stApp label, .stApp li, .stMarkdown, .stCaption {{ font-size: 1.25rem !important; letter-spacing: .02em; }}
.stApp::after {{
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 9999;
  background: repeating-linear-gradient(to bottom, rgba(0,0,0,0) 0, rgba(0,0,0,0) 2px, rgba(0,0,0,.18) 3px);
}}
h1, h2, h3 {{ font-family: 'VT323', monospace !important; color: var(--text) !important;
  text-shadow: 0 0 6px rgba(184,224,176,.45); letter-spacing: .04em; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.cg-hero {{ display: flex; gap: 28px; align-items: center; margin: 4px 0 18px; flex-wrap: wrap; }}
.cg-hero .frame {{ border: 2px solid var(--line); background: #22332a; padding: 14px; width: 190px; height: 190px;
  display: flex; align-items: center; justify-content: center; }}
.cg-hero .frame img {{ max-width: 100%; max-height: 100%; image-rendering: pixelated; }}
.cg-title {{ font-size: 4.2rem; line-height: 1; margin: 0 0 12px; text-shadow: 0 0 8px rgba(184,224,176,.55); }}
.cg-quote {{ font-size: 1.5rem; margin: 4px 0; }}
.cg-online {{ font-size: 1.6rem; color: var(--dim); margin-top: 10px; letter-spacing: .12em; }}
.cg-online span {{ display: inline-block; width: 12px; height: 22px; background: var(--dim); margin-right: 10px;
  vertical-align: -3px; animation: blink 1.2s steps(1) infinite; }}
@keyframes blink {{ 50% {{ opacity: 0; }} }}
.cg-pets {{ margin-left: auto; display: flex; gap: 10px; align-items: flex-end; }}
.cg-pets img {{ height: 92px; image-rendering: pixelated; }}

.cg-bar {{ background: var(--text); color: var(--ink); font-size: 1.7rem; padding: 2px 14px; margin: 18px 0 0;
  border: 2px solid var(--line); letter-spacing: .05em; }}
.cg-box {{ border: 2px solid var(--line); border-top: none; padding: 12px 16px; margin-bottom: 14px; font-size: 1.3rem; }}
.cg-box table {{ width: 100%; border-collapse: collapse; }}
.cg-box td {{ padding: 4px 8px 4px 0; vertical-align: top; border: none; }}
.cg-box td:first-child {{ width: 40%; color: var(--text); }}
.cg-big {{ font-size: 3rem; line-height: 1.1; }}

.cg-nocat {{ border: 3px solid var(--line); text-align: center; padding: 48px 16px; margin: 8px 0;
  font-size: clamp(3rem, 9vw, 6.5rem); line-height: 1; letter-spacing: .08em;
  text-shadow: 0 0 12px rgba(184,224,176,.6); }}
.cg-nocat img {{ height: 110px; display: block; margin: 0 auto 18px; image-rendering: pixelated; }}

[role="tablist"] {{ gap: 6px; border-bottom: 2px solid var(--line); }}
[data-testid="stTab"] {{ border: 2px solid var(--line); border-bottom: none; padding: 4px 16px; background: var(--panel); }}
[data-testid="stTab"] p {{ font-size: 1.45rem !important; }}
[data-testid="stTab"][aria-selected="true"] {{ background: var(--text); }}
[data-testid="stTab"][aria-selected="true"] p {{ color: var(--ink) !important; }}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display: none; }}
[data-testid="stForm"] {{ border: 2px solid var(--line); border-radius: 0; background: var(--panel); }}
.stButton button, [data-testid="stFormSubmitButton"] button {{ border-radius: 0 !important; border: 2px solid var(--line) !important;
  background: var(--text) !important; color: var(--ink) !important; font-size: 1.5rem !important; padding: 2px 28px !important; }}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{ background: #d4f0cc !important; }}
[data-baseweb="select"] > div, [data-baseweb="input"], input {{ border-radius: 0 !important; }}
[data-testid="stFileUploaderDropzone"] {{ border: 2px dashed var(--line); border-radius: 0; background: var(--panel); }}
[data-testid="stImage"] img {{ border-radius: 0; }}
[data-testid="stDataFrame"] {{ border: 2px solid var(--line); }}

.cg-cue img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: #dcefd6; border: 2px solid var(--line);
  padding: 4px; filter: saturate(.85); }}
[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"], [data-testid="stCheckbox"] p {{
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}
[data-testid="stColumn"]:has(input[type="checkbox"]:checked) .cg-cue img {{ background: var(--text); outline: 3px solid var(--text);
  outline-offset: 2px; }}
.cg-cue .m {{ color: var(--dim); font-size: 1.1rem; line-height: 1.1; min-height: 2.3em; margin: 2px 0 0; }}
</style>"""
    st.markdown("\n".join(line for line in css.splitlines() if line.strip()), unsafe_allow_html=True)


def bar(title: str):
    st.markdown(f'<div class="cg-bar">{title}</div>', unsafe_allow_html=True)


def box(html: str):
    st.markdown(f'<div class="cg-box">{html}</div>', unsafe_allow_html=True)


def hero():
    st.markdown(f"""
<div class="cg-hero">
  <div class="frame"><img src="{img_uri('silhouette.png')}" alt=""></div>
  <div>
    <div class="cg-title">~CatGPT[ ]</div>
    <div class="cg-quote">"reading your cat"</div>
    <div class="cg-quote">"shelter decision support"</div>
    <div class="cg-online"><span></span>ONLINE!</div>
  </div>
  <div class="cg-pets">
    <img src="{img_uri('tuxedo.png')}" alt=""><img src="{img_uri('tabby.png')}" alt=""><img src="{img_uri('black.png')}" alt="">
  </div>
</div>""", unsafe_allow_html=True)


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


def guidance(p_high: float) -> str:
    if p_high >= 0.70:
        return ("Likely <b>higher fearfulness</b>. Consider a quiet adult home, a slow introduction plan, "
                "a safe hiding space, and staff follow-up before adoption.")
    if p_high <= 0.30:
        return ("Likely <b>lower fearfulness</b>. The cat may cope well with visitors and new places; "
                "still match on the adopter's household and lifestyle.")
    return "Uncertain prediction. Rely on direct observation by staff for this cat."


def cue_picker() -> list[tuple]:
    """Grid of body-language cards; returns the cues the user ticked."""
    picked = []
    for start in range(0, len(CUES), 5):
        cols = st.columns(5)
        for col, cue in zip(cols, CUES[start:start + 5]):
            file, label, meaning, _ = cue
            col.markdown(f'<div class="cg-cue"><img src="{img_uri(f"cues/{file}.png")}" alt="{label}">'
                         f'</div>', unsafe_allow_html=True)
            if col.checkbox(label, key=f"cue_{file}"):
                picked.append(cue)
            col.markdown(f'<div class="cg-cue"><p class="m">{meaning}</p></div>', unsafe_allow_html=True)
    return picked


def cue_summary(picked: list[tuple], p_high: float):
    bar("Observed body language")
    if not picked:
        box("No cues selected. Tick the poses you saw during the shelter visit to compare them with the prediction.")
        return
    fear = sum(1 for c in picked if c[3] > 0)
    calm = sum(1 for c in picked if c[3] < 0)
    if fear > calm:
        lean, agrees = "fearful / defensive", p_high >= 0.5
    elif calm > fear:
        lean, agrees = "calm / confident", p_high < 0.5
    else:
        lean, agrees = "mixed", None
    check = {True: "Matches the survey-based prediction.",
             False: "Differs from the survey-based prediction. Ask staff to observe the cat again.",
             None: "No clear lean; staff observation decides."}[agrees]
    rows = "".join(f"<tr><td>{c[1]}</td><td>{c[2]}</td></tr>" for c in picked)
    box(f'<div class="cg-big">{calm} calm · {fear} fearful</div>'
        f"<p>Overall the cues lean <b>{lean}</b>. {check}</p><table>{rows}</table>"
        "<p style='opacity:.75'>One sign alone does not tell the full story. Read the whole body and the "
        "situation.</p>")


def intake_tab():
    bundle = intake_bundle()
    q = questions()
    bar("Fearfulness intake form")
    box(f"{len(bundle['items'])} questions plus demographics. "
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
        st.markdown("### Body language you observed (select one or more)")
        picked = cue_picker()
        submitted = st.form_submit_button("Predict", type="primary")
    if not submitted:
        return
    row = pd.DataFrame([{"BREEDGROUP": breed, "SEX": sex, "AGE_BEHAVIOUR": age or np.nan,
                         "OTHER_CATS": other, "problematic_behavior": problem, **answers}])
    p_high = float(bundle["model"].predict_proba(row[bundle["numeric"] + bundle["categorical"]])[0, 1])
    bar("Prediction")
    box(f'<div class="cg-big">{p_high:.0%}</div><p>Probability of higher fearfulness.</p><p>{guidance(p_high)}</p>')
    cue_summary(picked, p_high)
    contrib = contributions(bundle, row[bundle["numeric"] + bundle["categorical"]])
    bar("Answers that pushed the prediction most")
    st.caption("Positive = towards higher fear")
    st.bar_chart(contrib.rename(index=lambda n: q.get(n, n)[:70]).head(8), horizontal=True, color=GREEN)
    st.caption(DISCLAIMER)


def no_cat():
    st.markdown(f'<div class="cg-nocat"><img src="{img_uri("silhouette.png")}" alt="">NO CAT DETECTED</div>',
                unsafe_allow_html=True)


def image_tab():
    from catsense.vision import detect, predict
    bar("Photo or video: emotional state")
    box("YOLOv8 finds the cat, then the emotion model classifies the crop. Labels: "
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
            no_cat()
            return
        bar("Most frequent emotion")
        box(f'<div class="cg-big">{result["emotion"]}</div>'
            f"<p>{result['agreement']:.0%} of frames with a cat. {result['frames_analysed']} frames analysed, "
            f"cat found in {result['frames_with_cat']}.</p>")
        st.bar_chart(pd.Series(result["counts"], name="frames"), color=GREEN)
        return
    image = Image.open(upload).convert("RGB")
    with st.spinner("Detecting cat..."):
        result = predict.analyse_image(image)
    if result["n_cats"] == 0:
        no_cat()
        st.image(image, width=320)
        return
    c1, c2 = st.columns(2)
    c1.image(detect.draw_boxes(image, result["boxes"]), caption=f"{result['n_cats']} cat(s) found",
             width="stretch")
    with c2:
        bar("Predicted emotion")
        box(f'<div class="cg-big">{result["emotion"]}</div>')
        st.bar_chart(pd.Series(result["proba"], name="probability"), color=GREEN)
        st.image(result["crop"], caption="Region analysed", width=200)


def about_tab():
    bar("Models and evaluation")
    for title, file in [("Fearfulness models (held-out test, 20%)", "tabular_results.csv"),
                        ("Intake form length", "compact_form_results.csv"),
                        ("Emotion models (held-out test, 20%)", "vision_results.csv")]:
        path = C.RESULTS / file
        if path.exists():
            st.markdown(f"### {title}")
            st.dataframe(pd.read_csv(path).drop(columns=["best_params"], errors="ignore").round(3), hide_index=True)
    bar("Data and limitations")
    box("<table>"
        "<tr><td>Survey</td><td>Feline Behavior and Personality Survey (Mikkola et al., 2021, CC BY 4.0, 4,316 cats)</td></tr>"
        "<tr><td>Images</td><td>Cat Emotion Dataset (Hugging Face, research use)</td></tr>"
        "<tr><td>Limitations</td><td>Owner-reported answers; mostly Finnish cats and pedigree breeds; emotion labels "
        "are noisy.</td></tr>"
        f"<tr><td>Use</td><td>{DISCLAIMER}</td></tr></table>")


inject_css()
hero()
tabs = st.tabs(["Fearfulness intake", "Photo / video", "About the models"])
with tabs[0]:
    intake_tab()
with tabs[1]:
    image_tab()
with tabs[2]:
    about_tab()
