"""Phase 15: reproducible final evaluation of available trained artifacts."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import calculate_metrics
from src.models.advanced_mlp import AdvancedMLP
from src.models.mlp import MLP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "phase15_final_evaluation"
N_FEATURES = 2370

MODEL_PATHS = {
    "centralized": (RESULTS_DIR / "centralized_model.pt", "metadata"),
    "advanced_centralized": (
        RESULTS_DIR / "advanced_centralized" / "advanced_model.pt",
        "advanced_metadata",
    ),
    "fedavg_iid": (RESULTS_DIR / "fedavg" / "iid" / "best_global_model.pt", "state"),
    "fedprox_iid": (RESULTS_DIR / "fedprox" / "iid" / "best_global_model.pt", "state"),
    "personalized_iid": (RESULTS_DIR / "personalized" / "iid" / "best_global_model.pt", "state"),
    "adaptive_iid": (RESULTS_DIR / "adaptive_aggregation" / "iid" / "best_global_model.pt", "state"),
}


def load_test_data():
    """Load and validate the frozen Phase 2 test artifacts."""
    x_path = PROCESSED_DIR / "X_test.npy"
    y_path = PROCESSED_DIR / "y_test.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Phase 2 test artifacts are missing.")
    x_test = np.asarray(np.load(x_path), dtype=np.float32)
    y_test = np.asarray(np.load(y_path)).astype(int)
    if x_test.ndim != 2 or x_test.shape[1] != N_FEATURES:
        raise ValueError(f"Expected X_test shape (*, {N_FEATURES}), got {x_test.shape}")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("X_test and y_test contain different numbers of rows.")
    return x_test, y_test


def load_model(path, model_format):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if model_format == "advanced_metadata":
        config = checkpoint.get("model_config", {})
        model = AdvancedMLP(
            input_dim=config.get("input_dim", N_FEATURES),
            width=config.get("width", 256),
            num_blocks=config.get("num_blocks", 3),
            dropout=config.get("dropout", 0.25),
        )
        checkpoint = checkpoint["model_state_dict"]
    else:
        model = MLP(input_dim=N_FEATURES, hidden_dim1=128, hidden_dim2=64, dropout=0.2)
    if model_format == "metadata":
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def evaluate_model(path, model_format, x_test, y_test):
    model = load_model(path, model_format)
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test)).squeeze(1).numpy()
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return calculate_metrics(y_test, probabilities)


def run_phase15():
    """Evaluate every available baseline/federated checkpoint without training."""
    x_test, y_test = load_test_data()
    rows = []
    missing = []
    for name, (path, model_format) in MODEL_PATHS.items():
        if not path.exists():
            missing.append({"model": name, "path": str(path)})
            continue
        metrics = evaluate_model(path, model_format, x_test, y_test)
        rows.append({"model": name, **metrics})

    if not rows:
        raise FileNotFoundError("No trained checkpoints were found for Phase 15.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows).sort_values(
        ["pr_auc", "roc_auc"], ascending=False
    )
    metrics_df.to_csv(OUTPUT_DIR / "final_model_comparison.csv", index=False)
    (OUTPUT_DIR / "missing_artifacts.json").write_text(
        json.dumps(missing, indent=2), encoding="utf-8"
    )
    summary = {
        "phase": 15,
        "test_rows": int(len(y_test)),
        "feature_count": int(x_test.shape[1]),
        "models_evaluated": int(len(metrics_df)),
        "best_by_pr_auc": str(metrics_df.iloc[0]["model"]),
        "best_pr_auc": float(metrics_df.iloc[0]["pr_auc"]),
        "best_roc_auc": float(metrics_df.iloc[0]["roc_auc"]),
        "missing_artifacts": int(len(missing)),
    }
    (OUTPUT_DIR / "final_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics_df.to_string(index=False))
    print(f"Phase 15 results written to: {OUTPUT_DIR}")
    return metrics_df


if __name__ == "__main__":
    run_phase15()
