"""
Breast Cancer Survival Risk Screening Tool
==========================================
Streamlit application for the MLDP Term 2 project.

Loads the Logistic Regression pipeline trained in the notebook and returns a
calibrated risk of death within the follow-up window, together with a referral
recommendation at the operating threshold chosen in Section 5.

Run locally with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

MODEL_FILE = "breast_cancer_model.joblib"

st.set_page_config(
    page_title="Breast Cancer Survival Risk Tool",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling. Kept deliberately restrained: a clinical decision aid should read as
# a medical instrument, not a consumer app.
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .app-title {font-size: 2.05rem; font-weight: 700; margin-bottom: 0.15rem;}
      .app-sub {color: #5b6b7c; font-size: 1.02rem; margin-bottom: 1.4rem;}
      .panel {background: #f7f9fb; border: 1px solid #e3e9ef; border-radius: 10px;
              padding: 1.1rem 1.25rem; margin-bottom: 1rem;}
      .panel-title {font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
                    text-transform: uppercase; color: #5b6b7c; margin-bottom: 0.7rem;}
      .riskvalue {font-size: 3.4rem; font-weight: 700; line-height: 1; margin: 0;}
      .risklabel {font-size: 0.95rem; color: #5b6b7c; margin-top: 0.35rem;}
      .banner {border-radius: 10px; padding: 1rem 1.2rem; font-size: 1.02rem;
               margin: 0.4rem 0 1.1rem 0; border-left: 5px solid;}
      .banner-flag {background: #fdf0ee; border-color: #c0392b; color: #7d2018;}
      .banner-ok {background: #eef5fb; border-color: #2f6ea5; color: #1d4266;}
      .track {position: relative; height: 26px; background: linear-gradient(90deg,
              #cfe3f2 0%, #86b6da 35%, #3d7fb5 70%, #08519c 100%);
              border-radius: 13px; margin: 0.9rem 0 0.2rem 0;}
      .marker {position: absolute; top: -7px; width: 3px; height: 40px;
               background: #12212f; border-radius: 2px;}
      .dot {position: absolute; top: 3px; width: 20px; height: 20px;
            background: #fff; border: 4px solid #12212f; border-radius: 50%;
            transform: translateX(-10px);}
      .scale {display: flex; justify-content: space-between; color: #7c8b99;
              font-size: 0.76rem;}
      .contrib-row {display: flex; align-items: center; margin-bottom: 6px;
                    font-size: 0.86rem;}
      .contrib-name {width: 210px; color: #33414f;}
      .contrib-bar {height: 15px; border-radius: 3px;}
      .contrib-val {margin-left: 8px; color: #6b7a89; font-size: 0.79rem;}
      .footnote {color: #7c8b99; font-size: 0.82rem; line-height: 1.55;}
      div[data-testid="stMetricValue"] {font-size: 1.45rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artefact(path: str):
    """Load the saved pipeline and its metadata. Returns (artefact, error)."""
    if joblib is None:
        return None, "joblib is not installed. Add joblib to requirements.txt."
    try:
        return joblib.load(path), None
    except FileNotFoundError:
        return None, (
            f"Could not find `{path}`. Run the Deployment section of the notebook "
            "to create it, and keep it in the same folder as this app."
        )
    except Exception as exc:  # covers version mismatch on unpickling
        return None, (
            f"The model file could not be loaded ({type(exc).__name__}). This is "
            "usually a scikit-learn version mismatch between the machine that "
            "trained the model and this one. Pin the version printed by the "
            "notebook's Deployment cell in requirements.txt."
        )


artefact, load_error = load_artefact(MODEL_FILE)

st.markdown('<div class="app-title">Breast Cancer Survival Risk Screening Tool</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Prioritising newly diagnosed node-positive patients '
    'for genomic testing and closer follow-up, using pathology already on file.</div>',
    unsafe_allow_html=True,
)

if load_error:
    st.error(load_error)
    st.stop()

pipeline = artefact["pipeline"]
FEATURES = artefact["feature_order"]
MAPS = artefact["ordinal_maps"]
THRESHOLD = artefact["threshold"]
RANGES = artefact["feature_ranges"]
METRICS = artefact["test_metrics"]

# Human-readable labels for the ordinal levels, built from the notebook's own
# mappings so the app can never drift out of step with how the model was trained.
T_HELP = {
    "T1": "T1  (up to 20 mm)",
    "T2": "T2  (20 to 50 mm)",
    "T3": "T3  (over 50 mm)",
    "T4": "T4  (chest wall or skin involvement)",
}
N_HELP = {
    "N1": "N1  (1 to 3 positive nodes)",
    "N2": "N2  (4 to 9 positive nodes)",
    "N3": "N3  (10 or more positive nodes)",
}


# ----------------------------------------------------------------------------
# Sidebar: model provenance, measured performance, and limits on use
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("About this model")
    st.markdown(
        "Logistic Regression trained on **4,023 patients** from the US National "
        "Cancer Institute SEER programme, diagnosed 2006 to 2010, all with "
        "node-positive breast cancer."
    )

    st.markdown("**Measured on 805 unseen patients**")
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{METRICS['recall']:.0%}",
              help="Share of patients who died that the tool correctly flagged.")
    c2.metric("ROC-AUC", f"{METRICS['roc_auc']:.2f}",
              help="Ability to rank patients by risk. 0.5 is random.")
    c3, c4 = st.columns(2)
    c3.metric("PR-AUC", f"{METRICS['pr_auc']:.2f}",
              help=f"Against a base rate of {METRICS['base_rate']:.2f}.")
    c4.metric("Precision", f"{METRICS['precision']:.0%}",
              help="Share of flagged patients who died.")

    st.divider()
    st.subheader("Read this before using it")
    st.markdown(
        f"""
<div class="footnote">
<b>It ranks, it does not diagnose.</b> The output supports a treatment
discussion. It is not a prognosis to be given to a patient and does not replace
clinical judgement.<br><br>
<b>Node-positive patients only.</b> Every patient in the training data had at
least one positive lymph node. Nothing here applies to node-negative disease.<br><br>
<b>All-cause mortality.</b> The registry records that a patient died, not what
of, so some deaths in the training data were unrelated to the cancer.<br><br>
<b>Nine-year window.</b> Follow-up ended at the 2017 registry update, so the
estimate covers survival within that window, not indefinitely.<br><br>
<b>US data, 2006 to 2010.</b> Treatment has improved since, so estimates are
likely conservative. Use in Singapore would require recalibration on local data.<br><br>
<b>Race is deliberately excluded</b> so the tool cannot recommend harsher
treatment on the basis of a protected characteristic.
</div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
st.markdown("### Patient details")

col_a, col_b, col_c = st.columns(3, gap="large")

with col_a:
    st.markdown('<div class="panel-title">Patient</div>', unsafe_allow_html=True)
    age = st.number_input(
        "Age at diagnosis (years)",
        min_value=18, max_value=100, value=int(RANGES["Age"]["median"]), step=1,
        help=f"Training data covers {RANGES['Age']['min']:.0f} to "
             f"{RANGES['Age']['max']:.0f} years.",
    )
    grade = st.selectbox(
        "Tumour differentiation (grade)",
        list(MAPS["differentiate"].keys()), index=1,
        help="The strongest single predictor in this model.",
    )

with col_b:
    st.markdown('<div class="panel-title">Tumour</div>', unsafe_allow_html=True)
    t_stage = st.selectbox(
        "T stage", list(MAPS["T Stage"].keys()), index=1,
        format_func=lambda k: T_HELP.get(k, k),
    )
    tumour_size = st.number_input(
        "Tumour size (mm)",
        min_value=1, max_value=250, value=int(RANGES["Tumor Size"]["median"]), step=1,
        help=f"Training data covers {RANGES['Tumor Size']['min']:.0f} to "
             f"{RANGES['Tumor Size']['max']:.0f} mm.",
    )
    a_stage = st.radio(
        "Extent of spread", list(MAPS["A Stage"].keys()), index=0, horizontal=True,
    )

with col_c:
    st.markdown('<div class="panel-title">Nodes and receptors</div>',
                unsafe_allow_html=True)
    n_stage = st.selectbox(
        "N stage", list(MAPS["N Stage"].keys()), index=0,
        format_func=lambda k: N_HELP.get(k, k),
    )
    nodes_examined = st.number_input(
        "Regional nodes examined", min_value=1, max_value=90,
        value=int(RANGES["Regional Node Examined"]["median"]), step=1,
    )
    nodes_positive = st.number_input(
        "Regional nodes positive", min_value=1, max_value=90,
        value=int(RANGES["Regional Node Positive"]["median"]), step=1,
        help="Must be at least 1. This tool applies to node-positive disease only.",
    )

col_d, col_e = st.columns(2, gap="large")
with col_d:
    er = st.radio("Oestrogen receptor (ER)", list(MAPS["Estrogen Status"].keys()),
                  index=0, horizontal=True)
with col_e:
    pr = st.radio("Progesterone receptor (PR)",
                  list(MAPS["Progesterone Status"].keys()), index=0, horizontal=True)

st.write("")
calculate = st.button("Calculate risk", type="primary", width="stretch")


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate() -> tuple[list[str], list[str]]:
    """Return (blocking errors, non-blocking warnings)."""
    errors, warnings = [], []

    if nodes_positive > nodes_examined:
        errors.append(
            f"Positive nodes ({nodes_positive}) cannot exceed nodes examined "
            f"({nodes_examined}). A node can only be positive if it was examined."
        )

    expected_n = {"N1": (1, 3), "N2": (4, 9), "N3": (10, 90)}[n_stage]
    if not expected_n[0] <= nodes_positive <= expected_n[1]:
        warnings.append(
            f"{n_stage} normally corresponds to {expected_n[0]} to "
            f"{min(expected_n[1], 90)} positive nodes, but {nodes_positive} was "
            "entered. Check the staging and the node count agree."
        )

    expected_size = {"T1": (1, 20), "T2": (21, 50), "T3": (51, 250), "T4": (1, 250)}
    lo, hi = expected_size[t_stage]
    if not lo <= tumour_size <= hi:
        warnings.append(
            f"{t_stage} normally corresponds to a tumour of {lo} to {hi} mm, but "
            f"{tumour_size} mm was entered. T4 is defined by local invasion rather "
            "than size, so a mismatch is only expected for T4."
        )

    for field, value in [("Age", age), ("Tumor Size", tumour_size),
                         ("Regional Node Examined", nodes_examined),
                         ("Regional Node Positive", nodes_positive)]:
        r = RANGES[field]
        if not r["min"] <= value <= r["max"]:
            warnings.append(
                f"{field} of {value} falls outside the training range of "
                f"{r['min']:.0f} to {r['max']:.0f}. The estimate is an "
                "extrapolation and should be treated with extra caution."
            )

    return errors, warnings


# ----------------------------------------------------------------------------
# Prediction and results
# ----------------------------------------------------------------------------
def build_row() -> pd.DataFrame:
    """Assemble one patient in the exact column order the pipeline expects."""
    values = {
        "Age": age,
        "T Stage": MAPS["T Stage"][t_stage],
        "N Stage": MAPS["N Stage"][n_stage],
        "differentiate": MAPS["differentiate"][grade],
        "A Stage": MAPS["A Stage"][a_stage],
        "Tumor Size": tumour_size,
        "Estrogen Status": MAPS["Estrogen Status"][er],
        "Progesterone Status": MAPS["Progesterone Status"][pr],
        "Regional Node Examined": nodes_examined,
        "Regional Node Positive": nodes_positive,
    }
    return pd.DataFrame([values])[FEATURES]


def contributions(row: pd.DataFrame) -> pd.Series:
    """Per-feature push on the log-odds, for explaining a single prediction."""
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    standardised = (row.to_numpy(dtype=float)[0] - scaler.mean_) / scaler.scale_
    return pd.Series(standardised * model.coef_[0], index=FEATURES)


if not calculate:
    st.markdown(
        '<div class="panel"><div class="panel-title">No assessment yet</div>'
        "Enter the patient's details above and select <b>Calculate risk</b>. "
        "The tool will return an estimated risk of death within the follow-up "
        "window, a referral recommendation, and a breakdown of which findings "
        "drove the result.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

errors, warnings = validate()

if errors:
    for message in errors:
        st.error(message)
    st.warning("Correct the errors above, then select Calculate risk again.")
    st.stop()

for message in warnings:
    st.warning(message)

try:
    row = build_row()
    risk = float(pipeline.predict_proba(row)[0, 1])
except Exception as exc:
    st.error(
        f"The prediction could not be completed ({type(exc).__name__}). Please "
        "check the inputs and try again."
    )
    st.stop()

flagged = risk >= THRESHOLD

st.divider()
st.markdown("### Assessment")

res_a, res_b = st.columns([1, 2], gap="large")

with res_a:
    st.markdown(
        f'<div class="panel" style="text-align:center;">'
        f'<div class="panel-title">Estimated risk</div>'
        f'<p class="riskvalue">{risk * 100:.1f}%</p>'
        f'<div class="risklabel">of death within the follow-up window</div></div>',
        unsafe_allow_html=True,
    )
    baseline = METRICS["base_rate"]
    st.metric(
        "Compared with a typical patient in this cohort",
        f"{risk / baseline:.1f}x",
        delta=f"{(risk - baseline) * 100:+.1f} percentage points",
        help=f"The average risk across the cohort is {baseline * 100:.1f}%.",
    )

with res_b:
    if flagged:
        st.markdown(
            '<div class="banner banner-flag"><b>Prioritise for genomic testing '
            'and a closer treatment discussion.</b><br>This patient scores at or '
            'above the referral threshold.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner banner-ok"><b>Standard follow-up.</b><br>'
            'This patient scores below the referral threshold. Clinical judgement '
            'still takes precedence.</div>',
            unsafe_allow_html=True,
        )

    pos = min(max(risk, 0.0), 1.0) * 100
    thr = THRESHOLD * 100
    st.markdown(
        f"""
<div class="track">
  <div class="marker" style="left:{thr}%;"></div>
  <div class="dot" style="left:{pos}%;"></div>
</div>
<div class="scale"><span>0%</span><span>50%</span><span>100%</span></div>
<div class="footnote" style="margin-top:0.6rem;">
The vertical line marks the referral threshold of {thr:.1f}%. It is set
deliberately low: missing a high-risk patient is far more costly than an
unnecessary test, so the tool is tuned to catch
{METRICS['recall']:.0%} of deaths and accepts that {1 - METRICS['precision']:.0%}
of the patients it flags will not have died within the window.
</div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("#### What drove this result")

contrib = contributions(row).sort_values(key=abs, ascending=False)
largest = max(abs(contrib).max(), 1e-9)
pretty = {
    "differentiate": "Tumour grade",
    "Regional Node Positive": "Positive nodes",
    "Regional Node Examined": "Nodes examined",
    "Estrogen Status": "ER status",
    "Progesterone Status": "PR status",
    "A Stage": "Extent of spread",
    "Tumor Size": "Tumour size",
    "T Stage": "T stage",
    "N Stage": "N stage",
    "Age": "Age",
}

rows_html = []
for name, value in contrib.items():
    width = abs(value) / largest * 46
    colour = "#c0392b" if value > 0 else "#2f6ea5"
    offset = 50 if value > 0 else 50 - width
    rows_html.append(
        f'<div class="contrib-row"><div class="contrib-name">{pretty.get(name, name)}'
        f'</div><div style="flex:1; position:relative; height:15px;">'
        f'<div style="position:absolute; left:50%; top:-3px; width:1px; '
        f'height:21px; background:#c8d2dc;"></div>'
        f'<div class="contrib-bar" style="position:absolute; left:{offset}%; '
        f'width:{width}%; background:{colour};"></div></div>'
        f'<div class="contrib-val">{"increases" if value > 0 else "reduces"} risk</div>'
        f"</div>"
    )

st.markdown("".join(rows_html), unsafe_allow_html=True)
st.markdown(
    '<div class="footnote" style="margin-top:0.7rem;">'
    "Bars show how far each finding pushed this patient away from the cohort "
    "average, on the model's log-odds scale. Red pushes risk up, blue pushes it "
    "down. Note that examining <i>more</i> nodes reduces predicted risk once the "
    "number of positive nodes is held constant, because it reflects more thorough "
    "surgical staging and a lower proportion of involved nodes."
    "</div>",
    unsafe_allow_html=True,
)

with st.expander("Values sent to the model"):
    # Everything is cast to text so the two columns share one type, which keeps
    # the table serialisable and avoids a rendering fallback.
    display = pd.DataFrame({
        "Entered as": [str(v) for v in (
            age, t_stage, n_stage, grade, a_stage, tumour_size, er, pr,
            nodes_examined, nodes_positive)],
        "Encoded value": [str(v) for v in row.iloc[0].tolist()],
    }, index=FEATURES)
    display.index.name = "Feature"
    st.dataframe(display, width="stretch")

st.caption(
    "Research and educational use only. Not a medical device and not validated "
    "for clinical decision-making. MLDP Term 2 project, Temasek Polytechnic."
)
