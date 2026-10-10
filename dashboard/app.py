"""
HEALIX-FL Research Dashboard
============================

Streamlit dashboard for the HEALIX-FL research project
(Heterogeneity-Aware and Explainable Federated Learning Framework).

This dashboard is a READ-ONLY presentation and inference layer:

* It uses the EXISTING validated prediction module
  (src/prediction/predict.py) for preprocessing, inference and the
  local SHAP explanation. No prediction logic is duplicated or replaced.
* It reads the EXISTING experiment result files (Phases 12-14) directly
  from results/ - no experiment is re-run and no research file is modified.
* The trained checkpoint, the preprocessor and feature_names.csv are
  never changed.

Launch:
    streamlit run dashboard/app.py
    (or: python -m streamlit run dashboard/app.py)

Footer:
    HEALIX-FL | MSc AIML Research Project
"""

import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make the project root importable regardless of the working directory
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="HEALIX-FL Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "HEALIX-FL: Heterogeneity-Aware and Explainable Federated Learning Framework for Hospital Readmission Prediction"
    }
)

# Existing validated prediction module (never modified by the dashboard).
from src.prediction import predict as P  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MISSING_LABEL = "(not recorded)"

PAGE_HOME = "Home / Overview"
PAGE_PREDICT = "Patient Prediction"
PAGE_SHAP = "SHAP Explanation"
PAGE_PERFORMANCE = "Model Performance"
PAGE_HETEROGENEITY = "Heterogeneity"
PAGE_ABLATION = "Ablation Study"
PAGE_EXPLAINABILITY = "Explainability"

PAGES = [
    PAGE_HOME,
    PAGE_PREDICT,
    PAGE_SHAP,
    PAGE_PERFORMANCE,
    PAGE_HETEROGENEITY,
    PAGE_ABLATION,
    PAGE_EXPLAINABILITY,
]

# The 47 raw model input fields, grouped into dashboard input sections.
# The flat list itself is taken at runtime from the fitted preprocessor
# (preprocessor.feature_names_in_) - the SAME schema validated by
# src/prediction/predict.py - so the dashboard can never drift from the model.
INPUT_SECTIONS = [
    ("1. Demographics", ["race", "gender", "age", "weight"]),
    ("2. Admission Information", [
        "admission_type_id", "discharge_disposition_id",
        "admission_source_id", "time_in_hospital",
    ]),
    ("3. Payer / Medical Specialty", ["payer_code", "medical_specialty"]),
    ("4. Utilization and Counts", [
        "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient",
        "number_diagnoses",
    ]),
    ("5. Diagnosis Information", ["diag_1", "diag_2", "diag_3"]),
    ("6. Glucose / A1C", ["max_glu_serum", "A1Cresult"]),
    ("7. Medication Information", [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
        "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone", "change", "diabetesMed",
    ]),
]

# Existing Phase 14 ablation configurations (names exactly as in the CSV).
ABLATION_ORDER = [
    "fedavg_baseline",
    "fedprox_only",
    "adaptive_only",
    "fedprox_adaptive",
    "personalized_only",
    "fedprox_personalized",
    "adaptive_personalized",
    "full_healix",
]

ABLATION_LABELS = {
    "fedavg_baseline": "1. FedAvg baseline",
    "fedprox_only": "2. FedProx only",
    "adaptive_only": "3. Adaptive only",
    "fedprox_adaptive": "4. FedProx + Adaptive",
    "personalized_only": "5. Personalization only",
    "fedprox_personalized": "6. FedProx + Personalization",
    "adaptive_personalized": "7. Adaptive + Personalization",
    "full_healix": "8. Full HEALIX-FL",
}

# Existing Phase 14 heterogeneity conditions (names exactly as in the CSV).
HETEROGENEITY_ORDER = [
    "iid",
    "mild_non_iid",
    "moderate_non_iid",
    "strong_non_iid",
    "extreme_non_iid",
]

HETEROGENEITY_LABELS = {
    "iid": "IID (identically distributed)",
    "mild_non_iid": "Mild non-IID",
    "moderate_non_iid": "Moderate non-IID",
    "strong_non_iid": "Strong non-IID",
    "extreme_non_iid": "Extreme non-IID",
}

# Research result files (read-only, loaded from disk, never re-generated).
PHASE14_SUMMARY_CSV = "results/phase14_ablation/phase14_ablation_summary.csv"
PHASE12_GLOBAL_CSV = "results/shap/phase12_global_feature_importance.csv"
PHASE12_TOP_CSV = "results/shap/phase12_top_features.csv"
PHASE12_COMPARISON_CSV = "results/shap/phase12_model_comparison.csv"
PHASE12_OVERLAP_CSV = "results/shap/phase12_feature_overlap.csv"
PHASE13_EFFICIENCY_CSV = "results/shap/phase13_efficiency_comparison.csv"

# ---------------------------------------------------------------------------
# Cached loaders - the model, preprocessor, feature names and static result
# CSVs are loaded ONCE, not on every widget interaction.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_preprocessor():
    """Load the EXISTING preprocessor (never refit)."""
    return P.load_preprocessor()


@st.cache_resource
def get_feature_names():
    """Load the EXISTING transformed feature names (2370)."""
    return P.load_feature_names()


@st.cache_resource
def get_model():
    """Load the EXISTING trained HEALIX-FL checkpoint (never retrained)."""
    return P.load_model()


@st.cache_resource
def get_input_schema():
    """Return (columns, numeric_columns, categorical_columns): the exact
    47-column raw input schema expected by the fitted preprocessor."""
    preprocessor = get_preprocessor()
    columns = [str(c) for c in getattr(preprocessor, "feature_names_in_", [])]
    if len(columns) != P.RAW_INPUT_COLUMNS:
        raise ValueError(
            "Preprocessor exposes " + str(len(columns))
            + " input columns, expected " + str(P.RAW_INPUT_COLUMNS)
        )
    numeric_columns = []
    categorical_columns = []
    for name, _transformer, cols in preprocessor.transformers_:
        if name == "num":
            numeric_columns = [str(c) for c in cols]
        elif name == "cat":
            categorical_columns = [str(c) for c in cols]
    return columns, numeric_columns, categorical_columns


@st.cache_data
def load_raw_dataset():
    """Read the EXISTING raw dataset (read-only), exactly like predict.py."""
    return pd.read_csv(
        P.RAW_DATA_PATH, na_values=["?"], keep_default_na=True, low_memory=False
    )


@st.cache_data
def get_demo_defaults():
    """Default widget values = the EXISTING demonstration patient used by
    src/prediction/predict.py (row DEMO_ROW_INDEX of the raw dataset)."""
    columns, numeric_columns, _categorical = get_input_schema()
    row = load_raw_dataset().iloc[P.DEMO_ROW_INDEX]
    defaults = {}
    for col in columns:
        value = row[col]
        if pd.isna(value):
            defaults[col] = MISSING_LABEL
        elif col in numeric_columns:
            defaults[col] = int(value)
        else:
            defaults[col] = str(value)
    return defaults


@st.cache_data
def get_category_options():
    """Allowed values for every categorical field, taken from the EXISTING
    dataset (plus a '(not recorded)' option where data are missing)."""
    _columns, _numeric, categorical = get_input_schema()
    raw = load_raw_dataset()
    options = {}
    for col in categorical:
        values = sorted({str(v) for v in raw[col].dropna().tolist()})
        if raw[col].isna().any():
            options[col] = [MISSING_LABEL] + values
        else:
            options[col] = values
    return options


@st.cache_data
def get_numeric_ranges():
    """Observed [min, max] range for every numeric field (existing dataset)."""
    _columns, numeric, _categorical = get_input_schema()
    raw = load_raw_dataset()
    ranges = {}
    for col in numeric:
        series = pd.to_numeric(raw[col], errors="coerce").dropna()
        ranges[col] = (int(series.min()), int(series.max()))
    return ranges


@st.cache_data
def load_csv(relative_path):
    """Load a static research CSV (read-only). Returns None if missing."""
    path = os.path.join(P.PROJECT_ROOT, relative_path)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def build_patient_dataframe(values):
    """Build the 47-column raw patient DataFrame expected by predict.py.

    The row starts from the EXISTING raw dataset row used by the demo
    patient, so column order and dtypes are identical to the validated
    pipeline; each of the 47 fields is then overwritten with the value
    entered in the dashboard widgets.
    """
    columns, numeric_columns, _categorical = get_input_schema()
    base = load_raw_dataset().iloc[[P.DEMO_ROW_INDEX]][columns].copy()
    index = base.index[0]
    for col in columns:
        value = values[col]
        if isinstance(value, str) and value == MISSING_LABEL:
            base.at[index, col] = np.nan
        elif col in numeric_columns:
            base.at[index, col] = int(value)
        else:
            base.at[index, col] = str(value)
    return base


def readable_feature_name(feature):
    """Make a transformed feature name human readable for display."""
    name = str(feature)
    if name.startswith("num__"):
        name = name[len("num__"):]
    elif name.startswith("cat__"):
        name = name[len("cat__"):]
    return name.replace("_", " ")


def require_csv(relative_path, dataframe):
    """Show a clear error when an expected research file is missing."""
    if dataframe is None:
        st.error("Result file not found: " + relative_path)
        return False
    return True

# ---------------------------------------------------------------------------
# Sidebar navigation and footer
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 1.5rem 0 1rem 0;">
                <h1 style="background: linear-gradient(135deg, #3B82F6 0%, #818CF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 2.25rem; font-weight: 800; margin: 0; letter-spacing: -0.025em;">HEALIX-FL</h1>
                <p style="color: #94A3B8; font-size: 0.8rem; margin: 0.5rem 0 0 0; font-weight: 500; letter-spacing: 0.025em; text-transform: uppercase;">Heterogeneity-Aware & Explainable FL</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            '<p style="font-size: 0.875rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.75rem;">🧭 Navigation</p>',
            unsafe_allow_html=True,
        )
        st.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
        st.divider()
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(129, 140, 248, 0.1) 100%); padding: 1rem; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3); backdrop-filter: blur(10px);">
                <p style="font-size: 0.8125rem; color: #E2E8F0; margin: 0; line-height: 1.5;">
                    <span style="font-size: 1.25rem; vertical-align: middle;">🛡️</span> 
                    <span style="margin-left: 0.5rem;">Read-only dashboard using validated models & research results</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            '<p style="text-align: center; color: #64748B; font-size: 0.75rem; font-weight: 500;">HEALIX-FL | MSc AIML Research Project</p>',
            unsafe_allow_html=True,
        )


def render_footer():
    st.divider()
    st.markdown(
        '<p style="text-align: center; color: #64748B; font-size: 0.8rem;">🧬 HEALIX-FL | Heterogeneity-Aware and Explainable Federated Learning Framework | MSc AIML Research Project</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 1 - Home / Overview
# ---------------------------------------------------------------------------
def page_home():
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 3rem 0;">
            <h1 class="hero-title">HEALIX-FL</h1>
            <p class="hero-subtitle">Heterogeneity-Aware and Explainable Federated Learning for Hospital Readmission Prediction</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-header">Key Features</div>
        """,
        unsafe_allow_html=True,
    )
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Patient Risk Prediction</div>
                <div class="feature-desc">Enter patient data and get instant readmission risk assessment with confidence scores.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Explainable AI with SHAP</div>
                <div class="feature-desc">Understand which features drive predictions with transparent, interpretable explanations.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Research Analytics</div>
                <div class="feature-desc">Explore comprehensive research results across phases, ablations, and heterogeneity settings.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Project Overview")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Problem Statement")
            st.markdown(
                "Hospital readmission prediction using sensitive patient data. "
                "Federated learning enables collaboration while preserving privacy."
            )
    with col2:
        with st.container(border=True):
            st.markdown("#### Our Approach")
            st.markdown(
                "HEALIX-FL combines heterogeneity-aware federated learning with "
                "SHAP explainability to ensure transparency and effectiveness."
            )

    st.subheader("Dataset")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Dataset", "Diabetes 130-US Hospitals")
    d2.metric("Dataset size", "101,766 records")
    d3.metric("Raw input fields", "47")
    d4.metric("Transformed features", "2,370")
    st.caption(
        "Diabetes 130-US Hospitals for years 1999-2008 "
        "(data/raw/diabetic_data.csv, 101,766 records)."
    )

    st.subheader("Key Concepts")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔐 Privacy-Preserving** - Raw data never leaves hospital clients")
            st.markdown("**🌐 Federated Learning** - Collaborative training without data sharing")
            st.markdown("**⚖️ Heterogeneity-Aware** - Handles non-IID data distributions")
        with col2:
            st.markdown("**🧠 Explainable AI** - SHAP-based feature importance")
            st.markdown("**🔧 Adaptive Aggregation** - Quality-weighted model updates")
            st.markdown("**🎯 Personalized Models** - Client-specific fine-tuning")

    st.markdown("### How to use this dashboard")
    with st.container(border=True):
        st.markdown(
            "1. **Patient Prediction** - Enter the 47 patient parameters to predict readmission risk.\n"
            "2. **SHAP Explanation** - Understand which factors influenced the prediction.\n"
            "3. **Model Performance** - Compare model performance across configurations.\n"
            "4. **Heterogeneity** - Analyze impact of non-IID data distributions.\n"
            "5. **Ablation Study** - Explore contributions of each component.\n"
            "6. **Explainability** - Review global SHAP insights from research phases."
        )

    with st.container(border=True):
        st.markdown(
            "ℹ️ **Note:** This dashboard displays existing research results. "
            "No model retraining or experiment re-execution occurs."
        )

# ---------------------------------------------------------------------------
# Page 2 - Patient Prediction
# ---------------------------------------------------------------------------
def render_prediction_result(prediction):
    probability = float(prediction["probability"])
    label = str(prediction["label"])

    st.subheader("Prediction Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.metric("Predicted Probability", "{:.1f}%".format(probability * 100.0))
    with c2:
        with st.container(border=True):
            st.metric("Threshold", "{:.1f}".format(P.THRESHOLD))
    with c3:
        with st.container(border=True):
            if label == "HIGH RISK":
                st.metric("Risk Level", "HIGH ⚠️", delta_color="inverse")
            else:
                st.metric("Risk Level", "LOW ✅")

    st.markdown(
        "The HEALIX-FL model assigns this input patient a "
        "**{:.2f}%** predicted probability of the positive readmission class."
        .format(probability * 100.0)
    )
    st.markdown(
        "Classification rule (existing project threshold):\n"
        "- probability **< 0.5** \u2192 **LOW RISK**\n"
        "- probability **\u2265 0.5** \u2192 **HIGH RISK**"
    )
    if label == "HIGH RISK":
        st.error("⚠️ **HIGH RISK** - Predicted probability is at or above the threshold.")
    else:
        st.success("✅ **LOW RISK** - Predicted probability is below the threshold.")

    st.caption("⚠️ This is a model prediction for research purposes, not a medical diagnosis.")

    with st.expander("Model internals"):
        st.write("Raw logit before sigmoid:", prediction.get("logit"))
        st.write("Threshold used:", P.THRESHOLD)
    st.caption(
        "Result reflects the patient values at the time *Predict Readmission* "
        "was clicked. Preprocessing: saved preprocessor \u2192 2,370 transformed "
        "features \u2192 existing HEALIX-FL MLP."
    )


def page_predict():
    st.title("Patient Prediction")
    st.markdown(
        "Get a readmission risk prediction for a patient. The model uses the existing "
        "HEALIX-FL pipeline with validated preprocessing and training."
    )
    
    with st.container(border=True):
        st.markdown(
            "**How it works:** Input patient data → Preprocessing (2,370 features) → "
            "Trained HEALIX-FL model → Risk classification (threshold: 0.5)"
        )

    columns, numeric_columns, _categorical = get_input_schema()
    defaults = get_demo_defaults()
    category_options = get_category_options()
    numeric_ranges = get_numeric_ranges()

    # Initialise every widget with the EXISTING demo patient defaults.
    for col in columns:
        key = "patient_" + col
        if key not in st.session_state:
            st.session_state[key] = defaults[col]

    # Reset button with better UX
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Reset to demo patient", key="reset_demo_button", use_container_width=True):
            for col in columns:
                st.session_state["patient_" + col] = defaults[col]
            st.success("Reset to demo values")

    with st.form("patient_input_form"):
        for title, fields in INPUT_SECTIONS:
            st.markdown("#### " + title)
            widget_columns = st.columns(3)
            for position, field in enumerate(fields):
                with widget_columns[position % 3]:
                    key = "patient_" + field
                    if field in numeric_ranges:
                        low, high = numeric_ranges[field]
                        st.number_input(
                            field,
                            min_value=low,
                            max_value=high,
                            step=1,
                            key=key,
                        )
                    else:
                        st.selectbox(
                            field,
                            options=category_options[field],
                            key=key,
                        )
            st.divider()
        submitted = st.form_submit_button("🔍 Predict Readmission Risk", type="primary", use_container_width=True)

    if submitted:
        values = {col: st.session_state["patient_" + col] for col in columns}
        try:
            patient_df = build_patient_dataframe(values)
            with st.spinner(
                "Running the existing HEALIX-FL preprocessing and model..."
            ):
                prediction = P.predict_patient(patient_df)
        except Exception as exc:  # noqa: BLE001 - surface pipeline errors in the UI
            st.error("Prediction failed: " + repr(exc))
            st.session_state["prediction"] = None
            st.session_state["patient_df"] = None
            st.session_state["explanation"] = None
            st.session_state["shap_plot_path"] = None
            return
        st.session_state["prediction"] = prediction
        st.session_state["patient_df"] = patient_df
        # Any previous SHAP explanation belongs to the previous prediction.
        st.session_state["explanation"] = None
        st.session_state["shap_plot_path"] = None

    prediction = st.session_state.get("prediction")
    if prediction is not None:
        render_prediction_result(prediction)
    else:
        st.info(
            "No prediction yet - review the patient values and click "
            "*Predict Readmission*."
        )

# ---------------------------------------------------------------------------
# Page 3 - SHAP Explanation (uses the EXISTING explain_patient() only)
# ---------------------------------------------------------------------------
def page_shap():
    st.title("SHAP Explanation")
    st.markdown(
        "Understand which features contributed to the prediction. SHAP values show "
        "how each feature impacts the model's decision."
    )
    with st.container(border=True):
        st.markdown(
            "**Interpretation:** Positive SHAP values push toward HIGH RISK, "
            "negative values push toward LOW RISK."
        )

    prediction = st.session_state.get("prediction")
    patient_df = st.session_state.get("patient_df")
    if prediction is None or patient_df is None:
        st.info("💡 Run a patient prediction first on the **Patient Prediction** page to enable SHAP explanations.")
        return

    p1, p2 = st.columns(2)
    with p1:
        with st.container(border=True):
            st.metric("Predicted Probability", "{:.1f}%".format(float(prediction["probability"]) * 100.0))
    with p2:
        with st.container(border=True):
            st.metric("Risk Level", str(prediction["label"]))

    if st.button(
        "🔍 Generate SHAP Explanation (Top 10 Features)",
        type="primary",
        key="run_shap_button",
        use_container_width=True,
    ):
        try:
            with st.spinner("Computing the local SHAP explanation..."):
                explanation = P.explain_patient(patient_df, top_k=10)
        except Exception as exc:  # noqa: BLE001 - surface errors in the UI
            st.error("SHAP explanation failed: " + repr(exc))
            return
        st.session_state["explanation"] = explanation
        # Dashboard-specific plot ONLY under results/prediction/ - the existing
        # research SHAP outputs under results/shap/ are never touched.
        try:
            plot_path = P.save_shap_plot(
                explanation["shap_values"],
                get_feature_names(),
                os.path.join(P.PREDICTION_RESULTS_DIR, "dashboard_shap.png"),
            )
            st.session_state["shap_plot_path"] = plot_path
        except Exception as exc:  # noqa: BLE001
            st.session_state["shap_plot_path"] = None
            st.caption("Dashboard SHAP plot not saved: " + repr(exc))

    explanation = st.session_state.get("explanation")
    if explanation is None:
        st.caption(
            "No explanation generated yet for this prediction - click the button "
            "above to run the existing SHAP explainer (SHAP runs only on "
            "request, never on every dashboard rerun)."
        )
        return

    rows = []
    for item in explanation["top_features"]:
        value = float(item["shap_value"])
        if value > 0:
            direction = "positive contribution (toward the positive class)"
        elif value < 0:
            direction = "negative contribution (away from the positive class)"
        else:
            direction = "zero contribution"
        rows.append({
            "rank": int(item["rank"]),
            "feature": readable_feature_name(item["feature"]),
            "transformed_feature": str(item["feature"]),
            "shap_value": value,
            "contribution": direction,
        })
    top_df = pd.DataFrame(rows)

    st.subheader("Top 10 Contributing Features")
    st.dataframe(top_df, hide_index=True, use_container_width=True)

    with c1:
        with st.container(border=True):
            st.markdown("**🟢 Positive Impact**")
            st.markdown("Increases probability of high readmission risk")
    with c2:
        with st.container(border=True):
            st.markdown("**🔵 Negative Impact**")
            st.markdown("Decreases probability of high readmission risk")

    st.markdown("#### Contribution chart (top 10 features)")
    chart_data = top_df.set_index("feature")["shap_value"]
    st.bar_chart(chart_data, horizontal=True, x_label="SHAP value (logit contribution)")

    plot_path = st.session_state.get("shap_plot_path")
    if plot_path and os.path.exists(plot_path):
        st.markdown("#### Saved matplotlib plot")
        st.image(
            plot_path,
            caption=(
                "results/prediction/dashboard_shap.png - top 15 features by |SHAP| "
                "(dashboard-specific output; research SHAP files are untouched)"
            ),
        )

    st.caption(
        "SHAP shows which features contributed to the model's prediction; it does "
        "NOT prove causation. SHAP values explain the model only."
    )

# ---------------------------------------------------------------------------
# Page 4 - Model Performance (existing Phase 14 summary, never re-run)
# ---------------------------------------------------------------------------
def page_performance():
    st.title("Model Performance")
    st.caption(
        "Source: " + PHASE14_SUMMARY_CSV
        + " - existing Phase 14 ablation results (read-only, nothing is re-run)."
    )
    summary = load_csv(PHASE14_SUMMARY_CSV)
    if not require_csv(PHASE14_SUMMARY_CSV, summary):
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ablation configurations", int(summary["ablation"].nunique()))
    m2.metric("Heterogeneity settings", int(summary["heterogeneity"].nunique()))
    best_roc = summary.loc[summary["test_roc_auc"].idxmax()]
    m3.metric(
        "Best test ROC-AUC (existing)",
        "{:.4f}".format(best_roc["test_roc_auc"]),
        delta=str(best_roc["ablation"]) + " / " + str(best_roc["heterogeneity"]),
    )
    best_pr = summary.loc[summary["test_pr_auc"].idxmax()]
    m4.metric(
        "Best test PR-AUC (existing)",
        "{:.4f}".format(best_pr["test_pr_auc"]),
        delta=str(best_pr["ablation"]) + " / " + str(best_pr["heterogeneity"]),
    )

    st.subheader("Configuration comparison")
    metric = st.radio(
        "Metric",
        ["test_roc_auc", "test_pr_auc"],
        horizontal=True,
        key="perf_metric",
    )

    st.markdown("**Across heterogeneity settings**")
    pivot = summary.pivot(
        index="heterogeneity", columns="ablation", values=metric
    ).reindex(HETEROGENEITY_ORDER)
    pivot = pivot.reindex(
        columns=[a for a in ABLATION_ORDER if a in pivot.columns]
    )
    st.bar_chart(pivot)

    st.markdown("**Within one heterogeneity setting**")
    het = st.selectbox(
        "Heterogeneity setting",
        HETEROGENEITY_ORDER,
        format_func=lambda h: HETEROGENEITY_LABELS[h],
        key="perf_het",
    )
    subset = summary[summary["heterogeneity"] == het].set_index("ablation")[metric]
    subset = subset.reindex([a for a in ABLATION_ORDER if a in subset.index])
    st.bar_chart(subset)

    with st.expander("Ablation configuration names"):
        st.markdown(
            "\n".join(
                "- **" + ABLATION_LABELS[a] + "** (`" + a + "`)"
                for a in ABLATION_ORDER
            )
        )

    st.subheader("Full Phase 14 summary table")
    st.dataframe(summary, hide_index=True)
    st.caption(
        "All values are read directly from the existing CSV; no experiment was "
        "re-run and no value was modified."
    )


# ---------------------------------------------------------------------------
# Page 5 - Heterogeneity (IID vs non-IID, existing Phase 14 results)
# ---------------------------------------------------------------------------
def page_heterogeneity():
    st.title("Heterogeneity")
    st.caption(
        "Source: " + PHASE14_SUMMARY_CSV
        + " - existing Phase 14 results across the five heterogeneity conditions "
        "(read-only, nothing is re-run)."
    )
    summary = load_csv(PHASE14_SUMMARY_CSV)
    if not require_csv(PHASE14_SUMMARY_CSV, summary):
        return

    info_left, info_right = st.columns(2)
    with info_left:
        st.markdown("#### IID")
        st.markdown(
            "IID (independent and identically distributed): every client's local "
            "data is drawn from the same distribution. This is the reference "
            "setting (`iid`)."
        )
    with info_right:
        st.markdown("#### Non-IID / heterogeneous")
        st.markdown(
            "Non-IID: client data distributions differ from each other, which is "
            "the realistic hospital setting. Four heterogeneous conditions are "
            "reported: `mild_non_iid`, `moderate_non_iid`, `strong_non_iid` and "
            "`extreme_non_iid`."
        )

    st.subheader("Test ROC-AUC comparison")
    pivot = summary.pivot(
        index="heterogeneity", columns="ablation", values="test_roc_auc"
    ).reindex(HETEROGENEITY_ORDER)
    pivot = pivot.reindex(columns=[a for a in ABLATION_ORDER if a in pivot.columns])
    st.bar_chart(pivot)
    st.caption(
        "First category (IID) is the identically distributed reference; the "
        "remaining four settings are non-IID / heterogeneous."
    )

    st.subheader("Trend from IID to extreme non-IID")
    st.line_chart(pivot)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Test ROC-AUC values**")
        st.dataframe(pivot)
    with t2:
        st.markdown("**Test PR-AUC values**")
        pivot_pr = summary.pivot(
            index="heterogeneity", columns="ablation", values="test_pr_auc"
        ).reindex(HETEROGENEITY_ORDER)
        pivot_pr = pivot_pr.reindex(
            columns=[a for a in ABLATION_ORDER if a in pivot_pr.columns]
        )
        st.dataframe(pivot_pr)

    st.caption(
        "Values are read directly from the existing Phase 14 summary CSV; no "
        "experiment was re-run."
    )

# ---------------------------------------------------------------------------
# Page 6 - Ablation Study (existing Phase 14 configurations)
# ---------------------------------------------------------------------------
def page_ablation():
    st.title("Ablation Study")
    st.caption(
        "Source: " + PHASE14_SUMMARY_CSV
        + " - the eight existing Phase 14 ablation configurations across the five "
        "heterogeneity conditions (read-only, nothing is re-run)."
    )
    summary = load_csv(PHASE14_SUMMARY_CSV)
    if not require_csv(PHASE14_SUMMARY_CSV, summary):
        return

    st.markdown(
        "The eight existing ablation configurations isolate the contribution of "
        "each HEALIX-FL component:"
    )
    known = set(summary["ablation"])
    flags = (
        summary[["ablation", "proximal", "adaptive", "personalized"]]
        .drop_duplicates("ablation")
        .set_index("ablation")
        .reindex([a for a in ABLATION_ORDER if a in known])
    )
    flags.index = [ABLATION_LABELS.get(a, a) for a in flags.index]
    st.dataframe(flags, hide_index=True)

    st.subheader("Test ROC-AUC across the five heterogeneity conditions")
    wide = summary.pivot(
        index="ablation", columns="heterogeneity", values="test_roc_auc"
    ).reindex([a for a in ABLATION_ORDER if a in known])
    wide = wide.reindex(columns=HETEROGENEITY_ORDER)
    wide.index = [ABLATION_LABELS.get(a, a) for a in wide.index]
    st.dataframe(wide)

    with st.expander("Test PR-AUC across the five heterogeneity conditions"):
        wide_pr = summary.pivot(
            index="ablation", columns="heterogeneity", values="test_pr_auc"
        ).reindex([a for a in ABLATION_ORDER if a in known])
        wide_pr = wide_pr.reindex(columns=HETEROGENEITY_ORDER)
        wide_pr.index = [ABLATION_LABELS.get(a, a) for a in wide_pr.index]
        st.dataframe(wide_pr)

    st.subheader("Best existing configuration per heterogeneity level")
    st.caption("Ranked by the existing test ROC-AUC values in the Phase 14 summary.")
    best_columns = st.columns(5)
    for position, het in enumerate(HETEROGENEITY_ORDER):
        subset = summary[summary["heterogeneity"] == het]
        best = subset.loc[subset["test_roc_auc"].idxmax()]
        with best_columns[position]:
            st.metric(
                label=HETEROGENEITY_LABELS[het],
                value=ABLATION_LABELS.get(best["ablation"], best["ablation"]),
                delta="test ROC-AUC {:.4f}".format(best["test_roc_auc"]),
            )
    st.caption(
        "The highlighted configuration is simply the row with the highest "
        "existing test ROC-AUC for that condition - no result was recalculated "
        "or modified."
    )

    st.subheader("Configuration comparison in one condition")
    het = st.selectbox(
        "Heterogeneity setting",
        HETEROGENEITY_ORDER,
        format_func=lambda h: HETEROGENEITY_LABELS[h],
        key="ablation_het",
    )
    subset = summary[summary["heterogeneity"] == het].set_index("ablation")[
        "test_roc_auc"
    ]
    subset = subset.reindex([a for a in ABLATION_ORDER if a in subset.index])
    subset.index = [ABLATION_LABELS.get(a, a) for a in subset.index]
    st.bar_chart(subset)

# ---------------------------------------------------------------------------
# Page 7 - Explainability (existing Phase 12 and Phase 13 results)
# ---------------------------------------------------------------------------
def page_explainability():
    st.title("Explainability")
    st.caption(
        "Existing Phase 12 and Phase 13 SHAP results - read directly from "
        "results/shap/, never re-run or regenerated."
    )

    phase12_global = load_csv(PHASE12_GLOBAL_CSV)
    phase12_top = load_csv(PHASE12_TOP_CSV)
    phase12_comparison = load_csv(PHASE12_COMPARISON_CSV)
    phase12_overlap = load_csv(PHASE12_OVERLAP_CSV)
    phase13 = load_csv(PHASE13_EFFICIENCY_CSV)

    # ------------------------- Phase 12 -----------------------------------
    st.subheader("Phase 12 - Global SHAP feature importance")
    if not require_csv(PHASE12_TOP_CSV, phase12_top):
        phase12_top = None
    if phase12_top is not None:
        models = list(dict.fromkeys(phase12_top["model"].tolist()))
        model_col, topn_col = st.columns(2)
        with model_col:
            model = st.selectbox("Model (Phase 12)", models, key="xai_model")
        with topn_col:
            top_n = st.slider("Number of top features", 5, 20, 10, key="xai_top_n")

        subset = (
            phase12_top[phase12_top["model"] == model]
            .sort_values("rank")
            .head(top_n)
        )
        st.markdown("**Top features by mean |SHAP| (SHAP importance)**")
        st.dataframe(subset, hide_index=True)
        if not subset.empty:
            st.bar_chart(
                subset.set_index("feature")["mean_abs_shap"],
                horizontal=True,
                x_label="mean |SHAP|",
            )
        st.caption(
            "mean_abs_shap = mean absolute SHAP value (SHAP importance); "
            "mean_shap = mean signed SHAP value. Values are the existing "
            "Phase 12 results for the selected model."
        )

        if phase12_global is not None:
            with st.expander(
                "Full global feature importance (" + PHASE12_GLOBAL_CSV
                + ", selected model, first 100 ranks)"
            ):
                full_subset = (
                    phase12_global[phase12_global["model"] == model]
                    .sort_values("rank")
                    .head(100)
                )
                st.dataframe(full_subset, hide_index=True)
                st.caption(
                    "The full file contains every transformed feature for every "
                    "model; showing the first 100 ranks of the selected model."
                )
        else:
            st.error("Result file not found: " + PHASE12_GLOBAL_CSV)

        if phase12_comparison is not None:
            with st.expander("Model comparison - top-10 features per model"):
                matrix_src = phase12_comparison[
                    (phase12_comparison["model"].isin(models))
                    & (phase12_comparison["rank"] <= 10)
                ]
                matrix = matrix_src.pivot(
                    index="rank", columns="model", values="feature"
                )
                st.dataframe(matrix)
                st.caption(
                    "Top-10 SHAP features of every Phase 12 model ("
                    + PHASE12_COMPARISON_CSV + ")."
                )
        else:
            st.error("Result file not found: " + PHASE12_COMPARISON_CSV)

        if phase12_overlap is not None:
            with st.expander("Top-k feature overlap between model pairs"):
                st.dataframe(phase12_overlap, hide_index=True)
                st.caption(
                    "Shared top-k features between model pairs ("
                    + PHASE12_OVERLAP_CSV + ")."
                )
        else:
            st.error("Result file not found: " + PHASE12_OVERLAP_CSV)

    # ------------------------- Phase 13 -----------------------------------
    st.subheader("Phase 13 - Efficient explainability")
    if not require_csv(PHASE13_EFFICIENCY_CSV, phase13):
        return

    baseline_rows = phase13[phase13["condition"] == "baseline"]
    efficient_rows = phase13[phase13["condition"] != "baseline"]
    if baseline_rows.empty:
        baseline_runtime = float("nan")
    else:
        baseline_runtime = float(baseline_rows["runtime_seconds"].iloc[0])

    r1, r2, r3 = st.columns(3)
    r1.metric("Baseline runtime (s)", "{:.2f}".format(baseline_runtime))
    if not efficient_rows.empty:
        fastest = efficient_rows.loc[efficient_rows["runtime_seconds"].idxmin()]
        best_overlap = efficient_rows.loc[
            efficient_rows["top10_overlap_percentage"].idxmax()
        ]
        best_spearman = efficient_rows.loc[
            efficient_rows["spearman_rank_correlation"].idxmax()
        ]
        r2.metric(
            "Fastest efficient runtime (s)",
            "{:.2f}".format(fastest["runtime_seconds"]),
            delta=str(fastest["condition"]),
        )
        r3.metric(
            "Best top-10 overlap (%)",
            "{:.1f}".format(best_overlap["top10_overlap_percentage"]),
            delta=str(best_overlap["condition"]),
        )
        r4, r5, r6 = st.columns(3)
        r4.metric(
            "Best Spearman correlation",
            "{:.4f}".format(best_spearman["spearman_rank_correlation"]),
            delta=str(best_spearman["condition"]),
        )
    else:
        r4, r5, r6 = st.columns(3)
    r5.metric(
        "Max explanation reduction (%)",
        "{:.1f}".format(phase13["explanation_reduction_percentage"].max()),
    )
    r6.metric(
        "Max background reduction (%)",
        "{:.1f}".format(phase13["background_reduction_percentage"].max()),
    )

    st.markdown("**Runtime comparison (seconds)**")
    st.bar_chart(
        phase13.set_index("condition")["runtime_seconds"],
        horizontal=True,
        x_label="runtime (seconds)",
    )

    st.markdown("**Full efficiency comparison table**")
    st.dataframe(phase13, hide_index=True)
    st.caption(
        "top10_overlap = features shared with the baseline top-10; "
        "spearman_rank_correlation = rank agreement with the baseline; "
        "explanation/background reduction = sample reduction vs baseline. "
        "All values are the existing Phase 13 results."
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
PAGE_FUNCTIONS = {
    PAGE_HOME: page_home,
    PAGE_PREDICT: page_predict,
    PAGE_SHAP: page_shap,
    PAGE_PERFORMANCE: page_performance,
    PAGE_HETEROGENEITY: page_heterogeneity,
    PAGE_ABLATION: page_ablation,
    PAGE_EXPLAINABILITY: page_explainability,
}


def main():
    render_sidebar()
    page = st.session_state.get("nav_page", PAGE_HOME)
    if page not in PAGE_FUNCTIONS:
        page = PAGE_HOME
    PAGE_FUNCTIONS[page]()
    render_footer()


main()

