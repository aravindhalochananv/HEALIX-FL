"""
HEALIX-FL Prediction / Inference Module

Patient-level inference using artifacts produced by the COMPLETED HEALIX-FL
experiments (Phases 1-14). This module never trains a model and never
modifies any existing experiment result, checkpoint, or preprocessing step.

Artifacts used (read-only):
    - Saved preprocessing pipeline : data/processed/preprocessor.joblib
    - Transformed feature names    : data/processed/feature_names.csv
    - SHAP background sample       : data/processed/X_train.npy
    - Raw demonstration dataset    : data/raw/diabetic_data.csv
    - Preferred federated checkpoint (RAW state_dict):
          results/phase14_ablation/full_healix/iid/best_global_model.pt
    - Compatible fallback checkpoint:
          results/adaptive_aggregation/iid/best_global_model.pt

Model (src.models.mlp.MLP, unchanged):
    2370 -> 128 -> ReLU -> Dropout(0.2)
         -> 64  -> ReLU -> Dropout(0.2)
         -> 1 output logit

Inference flow:
    raw 47-column patient DataFrame
        -> SAVED preprocessor (2370 transformed features)
        -> MLP forward pass (logit)
        -> sigmoid
        -> probability
        -> threshold 0.5 (project default, src/evaluation/metrics.py)
        -> LOW RISK / HIGH RISK

Optional SHAP explanation uses shap.GradientExplainer with a small background
sample from data/processed/X_train.npy - the same approach as
src/explainability/shap_explainability.py (Phase 12). SHAP values explain the
MODEL'S PREDICTION (logit contributions); they do NOT prove medical causation.

Usage:
    python -m src.prediction.predict --demo
    python -m src.prediction.predict --demo --explain
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import joblib
import shap

from src.models.mlp import MLP

# ---------------------------------------------------------------------------
# Paths (anchored to the project root so the module works from any CWD)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "preprocessor.joblib")
FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "feature_names.csv")
X_TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "X_train.npy")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "diabetic_data.csv")
PREFERRED_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "results", "phase14_ablation", "full_healix", "iid", "best_global_model.pt"
)
FALLBACK_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "results", "adaptive_aggregation", "iid", "best_global_model.pt"
)
PREDICTION_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "prediction")

# ---------------------------------------------------------------------------
# Fixed settings aligned with the completed experiments
# ---------------------------------------------------------------------------
N_FEATURES = 2370          # config.yaml phase5 input_dim
HIDDEN_DIM1 = 128
HIDDEN_DIM2 = 64
DROPOUT = 0.2
THRESHOLD = 0.5            # default classification threshold (src/evaluation/metrics.py)
BACKGROUND_SIZE = 50        # same SHAP background size as Phase 12
RANDOM_SEED = 42
RAW_INPUT_COLUMNS = 47      # original model input columns (50 raw - 3 removed)
DEMO_ROW_INDEX = 0
ID_COLUMNS = ["encounter_id", "patient_nbr"]
TARGET_COLUMN = "readmitted"

# Module-level caches: the SAME loaded model/preprocessor are reused by
# prediction and SHAP explanation.
_preprocessor_cache = None
_feature_names_cache = None
_model_cache = None
_model_checkpoint_path = None


def get_model_checkpoint_path():
    """Return the preferred checkpoint, or an explicit compatible fallback."""
    if os.path.exists(PREFERRED_CHECKPOINT_PATH):
        return PREFERRED_CHECKPOINT_PATH
    if os.path.exists(FALLBACK_CHECKPOINT_PATH):
        return FALLBACK_CHECKPOINT_PATH
    raise FileNotFoundError(
        "No compatible inference checkpoint found. Expected either: "
        + PREFERRED_CHECKPOINT_PATH
        + " or "
        + FALLBACK_CHECKPOINT_PATH
    )


def load_preprocessor():
    """Load the already-fitted preprocessing pipeline. NEVER refit."""
    global _preprocessor_cache
    if _preprocessor_cache is None:
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError("Saved preprocessor missing: " + PREPROCESSOR_PATH)
        _preprocessor_cache = joblib.load(PREPROCESSOR_PATH)
    return _preprocessor_cache


def load_feature_names():
    """Load the transformed feature names (expecting 2370)."""
    global _feature_names_cache
    if _feature_names_cache is None:
        if not os.path.exists(FEATURE_NAMES_PATH):
            raise FileNotFoundError("Feature names file missing: " + FEATURE_NAMES_PATH)
        names = pd.read_csv(FEATURE_NAMES_PATH)["feature_name"].tolist()
        if len(names) != N_FEATURES:
            raise ValueError(
                "Expected " + str(N_FEATURES) + " feature names, got " + str(len(names))
            )
        _feature_names_cache = names
    return _feature_names_cache


def build_model():
    """Construct the project MLP architecture (untrained)."""
    return MLP(
        input_dim=N_FEATURES,
        hidden_dim1=HIDDEN_DIM1,
        hidden_dim2=HIDDEN_DIM2,
        dropout=DROPOUT,
    )


def load_model():
    """Load the RAW PyTorch state_dict checkpoint into the project MLP."""
    global _model_cache, _model_checkpoint_path
    if _model_cache is None:
        checkpoint_path = get_model_checkpoint_path()
        model = build_model()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # The Phase 14 checkpoint is a RAW state_dict (no metadata dict).
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint)
        model.eval()
        if model.network[0].in_features != N_FEATURES:
            raise ValueError(
                "Loaded model expects "
                + str(model.network[0].in_features)
                + " input features, expected "
                + str(N_FEATURES)
            )
        _model_cache = model
        _model_checkpoint_path = checkpoint_path
    return _model_cache


def transform_raw_patient(patient_dataframe):
    """Apply the SAVED preprocessor to a raw patient DataFrame.

    Args:
        patient_dataframe: DataFrame containing the original 47 model input
            columns (identifier/target columns are dropped if present).

    Returns:
        np.ndarray of shape (n_patients, 2370), dtype float32.
    """
    if not isinstance(patient_dataframe, pd.DataFrame):
        raise TypeError("patient_dataframe must be a pandas DataFrame")

    df = patient_dataframe.copy()
    drop_cols = [c for c in ID_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    preprocessor = load_preprocessor()

    # The fitted ColumnTransformer remembers the exact 47 input columns.
    expected_cols = list(getattr(preprocessor, "feature_names_in_", []))
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(
                "Patient DataFrame is missing expected columns: " + ", ".join(missing)
            )
        if len(expected_cols) != RAW_INPUT_COLUMNS:
            raise ValueError(
                "Preprocessor expects "
                + str(len(expected_cols))
                + " raw input columns, expected "
                + str(RAW_INPUT_COLUMNS)
            )
        df = df[expected_cols]

    X = preprocessor.transform(df)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != N_FEATURES:
        raise ValueError(
            "Preprocessor produced "
            + str(X.shape[1])
            + " features, expected "
            + str(N_FEATURES)
        )
    return X


def predict_patient(patient_dataframe, threshold=THRESHOLD):
    """Run inference on raw patient row(s).

    Pipeline: saved preprocessor -> MLP (logit) -> sigmoid -> threshold 0.5.

    Returns:
        dict with keys:
            probability: sigmoid output (float for a single patient)
            prediction: 0 or 1 (threshold THRESHOLD)
            label: "LOW RISK" or "HIGH RISK"
            logit: raw model logit
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    X = transform_raw_patient(patient_dataframe)
    model = load_model()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).numpy().flatten()
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= threshold).astype(int)

    if len(probabilities) == 1:
        pred = int(predictions[0])
        return {
            "probability": float(probabilities[0]),
            "prediction": pred,
            "label": "HIGH RISK" if pred == 1 else "LOW RISK",
            "logit": float(logits[0]),
        }
    return {
        "probability": probabilities,
        "prediction": predictions,
        "label": ["HIGH RISK" if p == 1 else "LOW RISK" for p in predictions],
        "logit": logits,
    }


def explain_patient(patient_dataframe, background_size=BACKGROUND_SIZE, top_k=10):
    """Explain the model prediction with shap.GradientExplainer.

    Uses a small background sample from data/processed/X_train.npy and the
    SAME loaded prediction model used by predict_patient().

    Returns:
        dict with keys:
            shap_values: np.ndarray of shape (n_patients, 2370)
            top_features: list of dicts {rank, feature, shap_value} ranked by
                |SHAP| for the FIRST patient
    """
    X = transform_raw_patient(patient_dataframe)
    model = load_model()
    model.eval()  # keep dropout disabled during explanation
    feature_names = load_feature_names()

    # Small background sample (memory-mapped read; first rows, same as Phase 12).
    X_train = np.load(X_TRAIN_PATH, mmap_mode="r")
    n_background = int(min(background_size, X_train.shape[0]))
    background = np.array(X_train[:n_background], dtype=np.float32)

    explainer = shap.GradientExplainer(model, torch.from_numpy(background))
    shap_raw = explainer.shap_values(torch.from_numpy(X))
    sv = np.asarray(shap_raw)
    if sv.ndim == 3:
        sv = sv.reshape(sv.shape[0], sv.shape[1])
    if sv.shape != (X.shape[0], N_FEATURES):
        raise ValueError("SHAP values shape mismatch: " + str(sv.shape))

    order = np.argsort(-np.abs(sv[0]))
    top_features = [
        {"rank": i + 1, "feature": feature_names[idx], "shap_value": float(sv[0][idx])}
        for i, idx in enumerate(order[:top_k])
    ]
    return {"shap_values": sv, "top_features": top_features}


def format_prediction(prediction, top_features=None, top_k=10, is_demo=False,
                      top_header="Top features:"):
    """Format a prediction (and optional SHAP top features) as a report string.

    Args:
        prediction: dict returned by predict_patient().
        top_features: optional list returned by explain_patient()["top_features"].
        top_k: maximum number of features to list.
        is_demo: label the patient as a DEMONSTRATION patient.
        top_header: header line for the feature list.

    Returns:
        str: printable report.
    """
    lines = []
    lines.append("=" * 40)
    lines.append("HEALIX-FL PREDICTION")
    lines.append("=" * 40)
    lines.append("")
    if is_demo:
        lines.append("NOTE: DEMONSTRATION patient taken from the EXISTING dataset")
        lines.append("      (data/raw/diabetic_data.csv) - NOT a new unseen patient.")
        lines.append("")
    lines.append("Prediction probability : {:.2f}%".format(prediction["probability"] * 100.0))
    lines.append("Prediction              : " + prediction["label"])
    lines.append("")
    if top_features:
        lines.append(top_header)
        for tf in top_features[:top_k]:
            lines.append(
                str(tf["rank"]) + ". " + tf["feature"] + " : "
                + "{:+.4f}".format(tf["shap_value"])
            )
        lines.append("")
        lines.append(
            "Note: SHAP values explain the model prediction only;"
            " they do NOT prove medical causation."
        )
    return "\n".join(lines)


def save_shap_plot(shap_values, feature_names, path, top_k=15):
    """Save a horizontal bar plot of the first patient's local SHAP values.

    Args:
        shap_values: np.ndarray of shape (n_patients, 2370).
        feature_names: list of 2370 transformed feature names.
        path: output PNG path.
        top_k: number of highest-|SHAP| features to plot.

    Returns:
        str: the saved path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sv = np.asarray(shap_values)[0]
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    features = [feature_names[i] for i in order][::-1]
    values = [float(sv[i]) for i in order][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(features, values, color=["red" if v > 0 else "blue" for v in values])
    ax.set_xlabel("SHAP value (logit contribution)")
    ax.set_title("HEALIX-FL demo patient - local SHAP explanation")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_demo(explain=False):
    """Run the demonstration pipeline on ONE patient from the existing dataset."""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError("Raw dataset missing: " + RAW_DATA_PATH)

    # Load one EXISTING patient (demonstration only, NOT a new unseen patient).
    raw = pd.read_csv(RAW_DATA_PATH, na_values=["?"], keep_default_na=True)
    patient_row = raw.iloc[[DEMO_ROW_INDEX]]
    input_columns = [c for c in patient_row.columns if c not in ID_COLUMNS + [TARGET_COLUMN]]
    patient = patient_row[input_columns]

    # Load artifacts (checkpoint verified inside load_model()).
    model = load_model()
    load_preprocessor()
    feature_names = load_feature_names()

    X = transform_raw_patient(patient)
    prediction = predict_patient(patient)
    explanation = explain_patient(patient)

    if explain:
        report = format_prediction(
            prediction, explanation["top_features"], top_k=10, is_demo=True,
            top_header="Top 10 contributing features:",
        )
    else:
        report = format_prediction(
            prediction, explanation["top_features"], top_k=5, is_demo=True,
        )
    print(report)
    print("")

    # Optional SHAP visualization (--explain)
    plot_path = None
    if explain:
        plot_path = os.path.join(PREDICTION_RESULTS_DIR, "demo_shap.png")
        save_shap_plot(explanation["shap_values"], feature_names, plot_path)
        print("SHAP plot saved to: " + os.path.relpath(plot_path, PROJECT_ROOT))
        print("")

    # ------------------------- verification ------------------------------
    prob = float(prediction["probability"])
    shap_shape = explanation["shap_values"].shape
    print("-" * 40)
    print("VERIFICATION")
    print("-" * 40)
    print("[VERIFY] Checkpoint loads successfully       : OK")
    print("[VERIFY] Model input features (expect 2370) : " + str(model.network[0].in_features))
    print("[VERIFY] Raw input columns (expect 47)      : " + str(len(input_columns)))
    print("[VERIFY] Preprocessor output features       : " + str(X.shape[1]))
    print("[VERIFY] Feature names length (expect 2370) : " + str(len(feature_names)))
    print("[VERIFY] Prediction probability in [0, 1]   : " + str(prob)
          + " -> " + str(0.0 <= prob <= 1.0))
    print("[VERIFY] SHAP values feature dimension      : " + str(shap_shape)
          + " -> " + str(shap_shape[1] == N_FEATURES))
    if explain and plot_path:
        created = os.path.exists(plot_path) and os.path.getsize(plot_path) > 0
        print("[VERIFY] SHAP plot created                  : " + str(created)
              + " -> " + os.path.relpath(plot_path, PROJECT_ROOT))
    else:
        print("[VERIFY] SHAP plot created                  : not requested"
              " (use --demo --explain)")


def main():
    parser = argparse.ArgumentParser(
        description="HEALIX-FL patient prediction / inference module"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a demonstration prediction using one patient from the existing dataset",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="With --demo: print the top 10 SHAP features and save "
             "results/prediction/demo_shap.png",
    )
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        print("\nRun the demonstration with:   python -m src.prediction.predict --demo")
        print("Add SHAP visualization with:  python -m src.prediction.predict --demo --explain")
        return

    run_demo(explain=args.explain)


if __name__ == "__main__":
    main()