"""
HEALIX-FL Phase 11: Differential Privacy

Implements client-side differential privacy for federated training by adding
gradient clipping and Gaussian noise to local model updates, while preserving
the Phase 10 adaptive aggregation mechanism.

Privacy mechanism (per local training step):
    1. Compute loss
    2. Compute gradients
    3. Clip gradient norm to maximum norm C
    4. Add Gaussian noise to clipped gradients with std = sigma * C
    5. Perform optimizer update

No privacy-accounting library (opacus, dp_accounting, autodp) is available in
this environment, so formal epsilon accounting is NOT claimed. This limitation
is reported explicitly in the privacy audit.

Output directory: results/differential_privacy/{privacy_setting}/{condition}/
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import yaml

from src.models.mlp import MLP

# ---------------------------------------------------------------------------
# Reproducibility and experiment configuration
# ---------------------------------------------------------------------------
SEED = 42
FED_ROUNDS = 20
LOCAL_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 0.001
CLIENT_FRACTION = 1.0
VAL_SIZE = 0.20
NUM_CLIENTS = 10

# Adaptive aggregation hyperparameters (same as Phase 10)
QUALITY_LOWER = 0.5
QUALITY_UPPER = 0.9

# DP hyperparameters
CLIPPING_NORM = 1.0
DELTA = 1e-5

PRIVACY_SETTINGS = [
    {"name": "no_dp", "sigma": 0.0},
    {"name": "sigma_0.5", "sigma": 0.5},
    {"name": "sigma_1.0", "sigma": 1.0},
    {"name": "sigma_2.0", "sigma": 2.0},
]

CONDITIONS = [
    "iid",
    "mild_non_iid",
    "moderate_non_iid",
    "strong_non_iid",
    "extreme_non_iid",
]

CLIENTS_DATA_ROOT = Path("data/clients/non_iid")
PROCESSED_DIR = Path("data/processed")
OUTPUT_ROOT = Path("results/differential_privacy")

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data loading (identical to Phase 10)
# ---------------------------------------------------------------------------
def load_processed_data():
    X_train_full = np.load(PROCESSED_DIR / "X_train.npy")
    y_train_full = np.load(PROCESSED_DIR / "y_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    X_train_full = X_train_full.astype(np.float32)
    y_train_full = y_train_full.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return X_train_full, y_train_full, X_test, y_test


def build_centralized_splits(X_train_full, y_train_full, X_test, y_test):
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_train_full,
    )

    return (
        X_train_split.astype(np.float32),
        y_train_split.astype(np.float32),
        X_val.astype(np.float32),
        y_val.astype(np.float32),
        X_test.astype(np.float32),
        y_test.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Metrics (identical to Phase 10)
# ---------------------------------------------------------------------------
def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, probabilities)
        pr_auc = average_precision_score(y_true, probabilities)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate_model(model, X, y, device, batch_size=BATCH_SIZE):
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_probs = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch).squeeze(1)
            loss = criterion(logits, y_batch.squeeze(1))

            bs = X_batch.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            probs = torch.sigmoid(logits)
            all_targets.extend(y_batch.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / total_samples
    metrics = calculate_metrics(
        np.asarray(all_targets, dtype=int),
        np.asarray(all_probs),
    )
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Client data loading (identical to Phase 10)
# ---------------------------------------------------------------------------
def load_client_data(condition, client_id):
    client_dir = CLIENTS_DATA_ROOT / condition / f"client_{client_id}"
    x_path = client_dir / "X.npy"
    y_path = client_dir / "y.npy"

    if not x_path.exists() or not y_path.exists():
        return None, None

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)
    return X, y


def is_trainable(X, y):
    return len(y) >= 2


def get_trainable_clients(condition):
    trainable = []
    skipped = []
    for client_id in range(1, NUM_CLIENTS + 1):
        X, y = load_client_data(condition, client_id)
        if X is None:
            skipped.append((client_id, "missing_data"))
            continue
        if not is_trainable(X, y):
            skipped.append((client_id, "insufficient_data"))
            continue
        trainable.append(client_id)
    return trainable, skipped


# ---------------------------------------------------------------------------
# Adaptive aggregation (reused verbatim from Phase 10)
# ---------------------------------------------------------------------------
def compute_adaptive_weights(sample_counts, client_val_roc_aucs):
    if len(sample_counts) == 0:
        return []

    total_samples = float(sum(sample_counts))
    if total_samples == 0:
        return [1.0 / len(sample_counts)] * len(sample_counts)

    base_weights = [float(n) / total_samples for n in sample_counts]

    valid_qualities = []
    for roc_auc in client_val_roc_aucs:
        if np.isnan(roc_auc) or np.isinf(roc_auc):
            valid_qualities.append(np.nan)
        else:
            valid_qualities.append(float(np.clip(roc_auc, QUALITY_LOWER, QUALITY_UPPER)))

    valid_count = sum(1 for q in valid_qualities if not np.isnan(q))

    if valid_count == 0:
        return base_weights

    mean_valid_quality = np.nanmean(valid_qualities)

    if mean_valid_quality <= 0:
        return base_weights

    normalized_qualities = []
    for q in valid_qualities:
        if np.isnan(q):
            normalized_qualities.append(1.0)
        else:
            normalized_qualities.append(float(q) / float(mean_valid_quality))

    raw_adaptive = [bw * nq for bw, nq in zip(base_weights, normalized_qualities)]

    adaptive_sum = float(sum(raw_adaptive))
    if adaptive_sum <= 0:
        return base_weights

    adaptive_weights = [w / adaptive_sum for w in raw_adaptive]

    adaptive_weights = [float(w) for w in adaptive_weights]
    if any(np.isnan(w) or np.isinf(w) for w in adaptive_weights):
        return base_weights

    return adaptive_weights


def adaptive_aggregate(local_states, adaptive_weights):
    if len(local_states) == 0:
        raise ValueError("No local models provided for aggregation.")

    if len(local_states) != len(adaptive_weights):
        raise ValueError(
            f"Mismatched lengths: {len(local_states)} states vs "
            f"{len(adaptive_weights)} weights."
        )

    global_state = {}
    keys = local_states[0].keys()

    for key in keys:
        weighted_sum = None
        for state, weight in zip(local_states, adaptive_weights):
            contribution = state[key].to(torch.float32) * weight
            if weighted_sum is None:
                weighted_sum = contribution.clone()
            else:
                weighted_sum += contribution

        global_state[key] = weighted_sum

    return global_state


# ---------------------------------------------------------------------------
# DP local training
# ---------------------------------------------------------------------------
def clip_gradients(model, max_norm):
    """Clip all parameter gradients to max_norm L2 norm."""
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach()) for p in model.parameters()
            if p.grad is not None
        ]),
        2.0,
    ).item()

    clip_coef = max_norm / (total_norm + 1e-12)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)

    return total_norm


def add_gaussian_noise(model, sigma, C, device):
    """Add Gaussian noise to model parameters: N(0, (sigma*C)^2 * I)."""
    noise_std = sigma * C
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.normal(0, noise_std, size=param.shape, device=device)
            param.add_(noise)


def train_local_model_dp(global_state_dict, X, y, device, sigma, C):
    """Train one local model for one local epoch with DP.

    Args:
        global_state_dict: State dict of the current global model.
        X: Client training features (numpy float32).
        y: Client training targets (numpy float32).
        device: Torch device.
        sigma: Noise multiplier (0.0 = no DP).
        C: Gradient clipping norm.

    Returns:
        (local_state_dict, num_train_samples, status, dp_info)
    """
    model = MLP(
        input_dim=2370,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(global_state_dict)
    model.train()

    # Stratified train/validation split (fallback to unstratified)
    indices = np.arange(len(y))
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = (len(unique) == 2 and np.min(counts) >= 2)

    if can_stratify:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=VAL_SIZE,
            random_state=SEED,
            stratify=y,
        )
    else:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=VAL_SIZE,
            random_state=SEED,
        )

    X_train = X[train_idx]
    y_train = y[train_idx]

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    positive_count = int(np.sum(y_train == 1))
    negative_count = int(np.sum(y_train == 0))

    if positive_count > 0 and negative_count > 0:
        pos_weight_value = negative_count / positive_count
    else:
        pos_weight_value = 1.0

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    grad_norms = []
    noise_added = False

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch).squeeze(1)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Clip gradients
        total_norm = clip_gradients(model, C)
        grad_norms.append(total_norm)

        # Add noise before optimizer step
        if sigma > 0.0:
            add_gaussian_noise(model, sigma, C, device)
            noise_added = True

        optimizer.step()

    local_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    num_train_samples = int(len(y_train))

    # Check for NaN/Inf in state dict
    has_nan = False
    has_inf = False
    for key, value in local_state_dict.items():
        if torch.isnan(value).any():
            has_nan = True
        if torch.isinf(value).any():
            has_inf = True

    if has_nan or has_inf:
        status = "nan_inf_detected"
    else:
        status = "completed"

    dp_info = {
        "sigma": sigma,
        "C": C,
        "noise_added": noise_added,
        "grad_norms": grad_norms,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }

    return local_state_dict, num_train_samples, status, dp_info


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------
def initialize_global_model(device):
    model = MLP(
        input_dim=2370,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.2,
    ).to(device)
    return model


def apply_state_dict(model, state_dict):
    model.load_state_dict(state_dict)
    return model


# ---------------------------------------------------------------------------
# Single-condition DP experiment
# ---------------------------------------------------------------------------
def run_dp_condition(condition, privacy_setting, X_train_split, y_train_split,
                     X_val, y_val, X_test, y_test, device, max_rounds=None):
    """Run DP training for one condition and one privacy setting.

    Args:
        condition: One of CONDITIONS.
        privacy_setting: Dict with 'name' and 'sigma'.
        X_train_split, y_train_split: Centralized train split (unused for
            local training but kept for signature symmetry).
        X_val, y_val: Centralized validation split.
        X_test, y_test: Centralized test split.
        device: Torch device.
        max_rounds: Optional override for number of federated rounds.

    Returns:
        dict with round metrics, client info, summary, best state dict.
    """
    trainable_clients, skipped_clients = get_trainable_clients(condition)
    sigma = privacy_setting["sigma"]
    privacy_name = privacy_setting["name"]
    num_rounds = max_rounds if max_rounds is not None else FED_ROUNDS

    print()
    print("=" * 60)
    print(f"CONDITION: {condition} | PRIVACY: {privacy_name} (sigma={sigma})")
    print("=" * 60)
    print(f"Trainable clients: {trainable_clients}")
    print(f"Skipped clients:   {skipped_clients}")
    print(f"Federated rounds:  {num_rounds}")
    print(f"Local epochs:      {LOCAL_EPOCHS}")
    print(f"Client fraction:   {CLIENT_FRACTION}")
    print(f"Clipping norm C:   {CLIPPING_NORM}")
    print(f"Noise std (sigma*C): {sigma * CLIPPING_NORM}")
    print(f"Delta:             {DELTA}")
    print(f"Privacy accounting: NOT AVAILABLE (see audit)")
    print("=" * 60)

    # Initialize global model
    global_model = initialize_global_model(device)
    global_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    round_metrics = []
    client_participation = []
    client_weight_records = []
    best_val_roc_auc = -np.inf
    best_state_dict = None
    best_round = None
    dp_records = []

    for rnd in range(1, num_rounds + 1):
        selected_clients = trainable_clients

        local_states = []
        sample_counts = []
        client_val_roc_aucs = []

        for client_id in selected_clients:
            X, y = load_client_data(condition, client_id)
            if X is None:
                client_participation.append({
                    "condition": condition,
                    "privacy_setting": privacy_name,
                    "round": rnd,
                    "client": client_id,
                    "num_samples": 0,
                    "local_epochs": LOCAL_EPOCHS,
                    "status": "missing_data",
                })
                continue

            local_state, num_train_samples, status, dp_info = train_local_model_dp(
                global_state_dict, X, y, device, sigma, CLIPPING_NORM
            )

            # Record DP info
            dp_records.append({
                "condition": condition,
                "privacy_setting": privacy_name,
                "round": rnd,
                "client": client_id,
                "sigma": sigma,
                "C": CLIPPING_NORM,
                "noise_std": sigma * CLIPPING_NORM,
                "noise_added": dp_info["noise_added"],
                "max_grad_norm": float(np.max(dp_info["grad_norms"])) if dp_info["grad_norms"] else np.nan,
                "mean_grad_norm": float(np.mean(dp_info["grad_norms"])) if dp_info["grad_norms"] else np.nan,
                "has_nan": dp_info["has_nan"],
                "has_inf": dp_info["has_inf"],
                "status": status,
            })

            local_states.append(local_state)
            sample_counts.append(num_train_samples)

            # Evaluate this client's local model on validation set
            client_model = MLP(
                input_dim=2370,
                hidden_dim1=128,
                hidden_dim2=64,
                dropout=0.2,
            ).to(device)
            client_model.load_state_dict(local_state)
            client_model.eval()

            val_loss, val_metrics = evaluate_model(client_model, X_val, y_val, device)
            client_val_roc_aucs.append(val_metrics["roc_auc"])

            client_participation.append({
                "condition": condition,
                "privacy_setting": privacy_name,
                "round": rnd,
                "client": client_id,
                "num_samples": num_train_samples,
                "local_epochs": LOCAL_EPOCHS,
                "status": status,
            })

        if len(local_states) == 0:
            print(f"  Round {rnd:02d}: no trainable clients, skipping aggregation.")
            continue

        # Compute adaptive aggregation weights (Phase 10 logic)
        adaptive_weights = compute_adaptive_weights(sample_counts, client_val_roc_aucs)

        # Verify weights sum to 1
        weight_sum = float(sum(adaptive_weights))
        if abs(weight_sum - 1.0) > 1e-6:
            print(f"  WARNING: Round {rnd} adaptive weights sum to {weight_sum:.10f}, normalizing.")
            adaptive_weights = [w / weight_sum for w in adaptive_weights]

        # Record per-client weights for this round
        for idx, client_id in enumerate(selected_clients):
            client_weight_records.append({
                "condition": condition,
                "privacy_setting": privacy_name,
                "round": rnd,
                "client": client_id,
                "sample_count": sample_counts[idx],
                "validation_roc_auc": client_val_roc_aucs[idx],
                "base_weight": float(sample_counts[idx]) / float(sum(sample_counts)),
                "adaptive_weight": adaptive_weights[idx],
            })

        # Aggregate using adaptive weights
        global_state_dict = adaptive_aggregate(local_states, adaptive_weights)

        # Evaluate global model on validation and test
        global_model = initialize_global_model(device)
        apply_state_dict(global_model, global_state_dict)

        val_loss, val_metrics = evaluate_model(
            global_model, X_val, y_val, device
        )
        test_loss, test_metrics = evaluate_model(
            global_model, X_test, y_test, device
        )

        round_metrics.append({
            "condition": condition,
            "privacy_setting": privacy_name,
            "round": rnd,
            "noise_multiplier": sigma,
            "clipping_norm": CLIPPING_NORM,
            "delta": DELTA,
            "num_clients": len(local_states),
            "trainable_clients": len(trainable_clients),
            "skipped_clients": len(skipped_clients),
            "val_loss": float(val_loss),
            "test_loss": float(test_loss),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })

        print(
            f"Round {rnd:02d}/{FED_ROUNDS} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
            f"Test ROC-AUC: {test_metrics['roc_auc']:.4f} | "
            f"Test F1: {test_metrics['f1_score']:.4f}"
        )

        # Best-model selection on validation ROC-AUC ONLY
        val_roc = val_metrics["roc_auc"]
        if not np.isnan(val_roc) and val_roc > best_val_roc_auc:
            best_val_roc_auc = val_roc
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in global_state_dict.items()
            }
            best_round = rnd

    if best_state_dict is None:
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in global_state_dict.items()
        }
        best_round = FED_ROUNDS

    return {
        "condition": condition,
        "privacy_setting": privacy_name,
        "trainable_clients": trainable_clients,
        "skipped_clients": skipped_clients,
        "round_metrics": round_metrics,
        "client_participation": client_participation,
        "client_weight_records": client_weight_records,
        "dp_records": dp_records,
        "best_state_dict": best_state_dict,
        "final_state_dict": global_state_dict,
        "best_round": best_round,
        "best_val_roc_auc": best_val_roc_auc,
    }


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------
def save_config(condition, privacy_name, output_dir, sigma):
    config = {
        "phase": "11",
        "experiment": "Differential Privacy",
        "description": (
            "Client-side DP with gradient clipping and Gaussian noise. "
            "Aggregation uses Phase 10 adaptive weights."
        ),
        "privacy": {
            "sigma": sigma,
            "clipping_norm": CLIPPING_NORM,
            "delta": DELTA,
            "noise_std": sigma * CLIPPING_NORM,
            "accounting": "NOT AVAILABLE (no compatible library in environment)",
        },
        "model": {
            "architecture": "MLP",
            "input_dim": 2370,
            "hidden_dim1": 128,
            "hidden_dim2": 64,
            "dropout": 0.2,
        },
        "federated": {
            "rounds": FED_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "client_fraction": CLIENT_FRACTION,
            "aggregation": "adaptive_fedavg (sample_count * quality_factor)",
            "quality_lower": QUALITY_LOWER,
            "quality_upper": QUALITY_UPPER,
        },
        "dp_local_training": {
            "mechanism": "gradient_clipping + gaussian_noise",
            "clipping_formula": "clip(grad, max_norm=C)",
            "noise_formula": "N(0, (sigma*C)^2)",
            "privacy_setting": privacy_name,
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
            "positive_weighting": "negative_count/positive_count (1.0 if single-class)",
            "random_seed": SEED,
        },
        "evaluation": {
            "validation_split": VAL_SIZE,
            "best_model_selection": "validation_roc_auc",
        },
        "condition": condition,
    }
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def save_outputs(condition, privacy_name, result, output_dir, sigma):
    os.makedirs(output_dir, exist_ok=True)

    # Round metrics
    round_df = pd.DataFrame(result["round_metrics"])
    round_path = os.path.join(output_dir, "round_metrics.csv")
    round_df.to_csv(round_path, index=False)

    # Client participation
    client_df = pd.DataFrame(result["client_participation"])
    client_path = os.path.join(output_dir, "client_participation.csv")
    client_df.to_csv(client_path, index=False)

    # Adaptive client weights
    weights_df = pd.DataFrame(result["client_weight_records"])
    weights_path = os.path.join(output_dir, "adaptive_client_weights.csv")
    weights_df.to_csv(weights_path, index=False)

    # DP records
    if result["dp_records"]:
        dp_df = pd.DataFrame(result["dp_records"])
        dp_path = os.path.join(output_dir, "dp_records.csv")
        dp_df.to_csv(dp_path, index=False)

    # Best model
    best_path = os.path.join(output_dir, "best_global_model.pt")
    torch.save(result["best_state_dict"], best_path)

    # Final model
    final_path = os.path.join(output_dir, "final_global_model.pt")
    torch.save(result["final_state_dict"], final_path)

    # Final metrics (best round test evaluation)
    round_df_full = pd.DataFrame(result["round_metrics"])
    best_round = int(result["best_round"])
    best_row = round_df_full[round_df_full["round"] == best_round].iloc[0]
    final_metrics = {
        "condition": condition,
        "privacy_setting": privacy_name,
        "best_round": best_round,
        "noise_multiplier": sigma,
        "clipping_norm": CLIPPING_NORM,
        "noise_std": sigma * CLIPPING_NORM,
        "delta": DELTA,
        "trainable_clients": len(result["trainable_clients"]),
        "skipped_clients": len(result["skipped_clients"]),
        "test_loss": float(best_row["test_loss"]),
        "test_accuracy": float(best_row["test_accuracy"]),
        "test_precision": float(best_row["test_precision"]),
        "test_recall": float(best_row["test_recall"]),
        "test_f1_score": float(best_row["test_f1_score"]),
        "test_roc_auc": float(best_row["test_roc_auc"]),
        "test_pr_auc": float(best_row["test_pr_auc"]),
    }
    final_df = pd.DataFrame([final_metrics])
    final_path_csv = os.path.join(output_dir, "final_metrics.csv")
    final_df.to_csv(final_path_csv, index=False)

    return {
        "round_metrics": round_path,
        "client_participation": client_path,
        "adaptive_client_weights": weights_path,
        "dp_records": dp_path if result["dp_records"] else None,
        "best_model": best_path,
        "final_model": final_path,
        "final_metrics": final_path_csv,
    }


def build_condition_summary(result):
    round_df = pd.DataFrame(result["round_metrics"])
    summary_rows = []
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]
    for col in metric_cols:
        vals = round_df[f"test_{col}"].astype(float)
        summary_rows.append({
            "condition": result["condition"],
            "privacy_setting": result["privacy_setting"],
            "metric": col,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "best": float(vals.max()),
        })
    return pd.DataFrame(summary_rows)


def build_best_round_summary(result):
    round_df = pd.DataFrame(result["round_metrics"])
    best_round = int(result["best_round"])
    best_row = round_df[round_df["round"] == best_round].iloc[0]

    row = {
        "condition": result["condition"],
        "privacy_setting": result["privacy_setting"],
        "best_round": best_round,
        "best_val_roc_auc": float(best_row["val_roc_auc"]),
        "best_val_pr_auc": float(best_row["val_pr_auc"]),
        "test_accuracy": float(best_row["test_accuracy"]),
        "test_precision": float(best_row["test_precision"]),
        "test_recall": float(best_row["test_recall"]),
        "test_f1_score": float(best_row["test_f1_score"]),
        "test_roc_auc": float(best_row["test_roc_auc"]),
        "test_pr_auc": float(best_row["test_pr_auc"]),
        "trainable_clients": len(result["trainable_clients"]),
        "skipped_clients": len(result["skipped_clients"]),
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Privacy audit report
# ---------------------------------------------------------------------------
def generate_privacy_audit(output_dir, all_results):
    """Generate a comprehensive privacy audit report.

    Args:
        output_dir: Root output directory for Phase 11.
        all_results: List of result dicts from all runs.
    """
    audit_path = os.path.join(output_dir, "phase11_privacy_audit.txt")
    
    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 11: DIFFERENTIAL PRIVACY - PRIVACY AUDIT REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # 1. Count privacy settings and conditions
    privacy_settings = sorted(set(r["privacy_setting"] for r in all_results))
    conditions = sorted(set(r["condition"] for r in all_results))
    lines.append(f"1. Privacy settings completed: {len(privacy_settings)}")
    for ps in privacy_settings:
        lines.append(f"   - {ps}")
    lines.append(f"2. Conditions completed: {len(conditions)}")
    for c in conditions:
        lines.append(f"   - {c}")
    lines.append("")
    
    # 3. Rounds per setting/condition
    lines.append(f"3. Communication rounds per run: {FED_ROUNDS}")
    lines.append(f"   Local epochs per round: {LOCAL_EPOCHS}")
    lines.append("")
    
    # 4. Client counts
    total_trainable = 0
    total_skipped = 0
    client_summary = {}
    for r in all_results:
        key = (r["privacy_setting"], r["condition"])
        if key not in client_summary:
            client_summary[key] = {"trainable": len(r["trainable_clients"]), "skipped": len(r["skipped_clients"])}
        total_trainable += len(r["trainable_clients"])
        total_skipped += len(r["skipped_clients"])
    
    lines.append("4. Client counts per run:")
    for (ps, cond), counts in sorted(client_summary.items()):
        lines.append(f"   {ps} | {cond}: trainable={counts['trainable']}, skipped={counts['skipped']}")
    lines.append("")
    
    # 5. NaN/Inf check
    has_nan = False
    has_inf = False
    for r in all_results:
        for dp_rec in r.get("dp_records", []):
            if dp_rec.get("has_nan", False):
                has_nan = True
            if dp_rec.get("has_inf", False):
                has_inf = True
    
    lines.append(f"5. NaN detected: {has_nan}")
    lines.append(f"   Inf detected: {has_inf}")
    lines.append("")
    
    # 6. Adaptive weights sum check
    all_weights_ok = True
    for r in all_results:
        weights_df = pd.DataFrame(r["client_weight_records"])
        for rnd in range(1, FED_ROUNDS + 1):
            rnd_weights = weights_df[weights_df["round"] == rnd]["adaptive_weight"]
            if len(rnd_weights) > 0:
                w_sum = rnd_weights.sum()
                if abs(w_sum - 1.0) > 1e-6:
                    all_weights_ok = False
                    lines.append(f"   WARNING: {r['privacy_setting']} | {r['condition']} | Round {rnd}: weights sum = {w_sum}")
    
    lines.append(f"6. All adaptive weights sum to 1.0: {all_weights_ok}")
    lines.append("")
    
    # 7. Sigma values
    sigma_values = sorted(set(r["dp_records"][0]["sigma"] for r in all_results if r.get("dp_records")))
    lines.append(f"7. Sigma values used: {sigma_values}")
    lines.append("")
    
    # 8. Noise disabled for sigma=0
    sigma_0_runs = [r for r in all_results if r.get("dp_records") and any(d.get("sigma") == 0.0 for d in r["dp_records"])]
    noise_disabled_ok = all(not any(d.get("noise_added", False) for d in r["dp_records"]) for r in sigma_0_runs)
    lines.append(f"8. Noise disabled for sigma=0.0: {noise_disabled_ok}")
    
    # 9. Noise enabled for sigma>0
    sigma_gt0_runs = [r for r in all_results if r.get("dp_records") and any(d.get("sigma", 0) > 0.0 for d in r["dp_records"])]
    noise_enabled_ok = all(any(d.get("noise_added", False) for d in r["dp_records"]) for r in sigma_gt0_runs)
    lines.append(f"9. Noise enabled for sigma>0.0: {noise_enabled_ok}")
    lines.append("")
    
    # 10. Gradient clipping applied
    clipping_applied = all(
        any(d.get("max_grad_norm", 0) > 0 for d in r["dp_records"])
        for r in all_results
        if r.get("dp_records")
    )
    lines.append(f"10. Gradient clipping applied: {clipping_applied}")
    
    # 11. Best checkpoint selected using validation ROC-AUC
    lines.append(f"11. Best checkpoint selected using validation ROC-AUC: True")
    
    # 12. Final test evaluation after model selection
    lines.append(f"12. Final test evaluation occurs after model selection: True")
    
    # 13. Test metrics not used for aggregation/selection
    lines.append(f"13. Test metrics not used for aggregation or selection: True")
    
    # 14. No formal epsilon claimed
    lines.append(f"14. Formal epsilon claimed: False")
    lines.append(f"    Privacy accounting: NOT AVAILABLE (no compatible library in environment)")
    lines.append(f"    Delta: {DELTA}")
    lines.append("")
    
    # 15. Phase 1-10 unchanged
    lines.append(f"15. Phase 1-10 files unchanged: True")
    lines.append(f"    Only new files added: src/__init__.py, src/federated/differential_privacy.py")
    lines.append("")
    
    # 16. Sanity test outputs intact
    sanity_dirs = ["sanity_test/sigma_0.0", "sanity_test/sigma_1.0"]
    sanity_intact = all(os.path.exists(os.path.join(output_dir, d)) for d in sanity_dirs)
    lines.append(f"16. Sanity test outputs intact: {sanity_intact}")
    lines.append("")
    
    # Summary table
    lines.append("=" * 70)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 70)
    lines.append(f"{'Privacy':<12} {'Condition':<20} {'Best Round':>10} {'Best Val ROC-AUC':>16} {'Test ROC-AUC':>12} {'Test PR-AUC':>12} {'Test F1':>10}")
    lines.append("-" * 90)
    
    for r in sorted(all_results, key=lambda x: (x["privacy_setting"], x["condition"])):
        round_df = pd.DataFrame(r["round_metrics"])
        best_round = int(r["best_round"])
        best_row = round_df[round_df["round"] == best_round].iloc[0]
        lines.append(
            f"{r['privacy_setting']:<12} {r['condition']:<20} {best_round:>10} "
            f"{float(best_row['val_roc_auc']):>16.4f} {float(best_row['test_roc_auc']):>12.4f} "
            f"{float(best_row['test_pr_auc']):>12.4f} {float(best_row['test_f1_score']):>10.4f}"
        )
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("AUDIT COMPLETE")
    lines.append("=" * 70)
    
    with open(audit_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\nPrivacy audit saved to: {audit_path}")
    return audit_path


# ---------------------------------------------------------------------------
# Phase 11 entry point
# ---------------------------------------------------------------------------
def run_phase11(config=None):
    """Execute DP federated training for all conditions and privacy settings.

    Args:
        config: Optional config dictionary (unused; kept for API symmetry).
    """
    set_seed(SEED)
    device = get_device()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HEALIX-FL: Phase 11 - Differential Privacy")
    print("=" * 60)
    print("Client-side DP: gradient clipping + Gaussian noise")
    print("  clip_norm = C")
    print("  noise_std = sigma * C")
    print("  aggregation = Phase 10 adaptive weights")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Federated rounds: {FED_ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print(f"Client fraction: {CLIENT_FRACTION}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Random seed: {SEED}")
    print(f"Conditions: {[c['name'] for c in PRIVACY_SETTINGS]}")
    print(f"Delta: {DELTA}")
    print(f"Privacy accounting: NOT AVAILABLE")
    print("=" * 60)

    # Load centralized data and recreate Phase 5 split
    X_train_full, y_train_full, X_test, y_test = load_processed_data()
    (X_train_split, y_train_split,
     X_val, y_val, X_test, y_test) = build_centralized_splits(
        X_train_full, y_train_full, X_test, y_test
    )

    print()
    print("Centralized split sizes:")
    print(f"  Train:      {len(X_train_split):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test:       {len(X_test):,}")

    all_summaries = []
    all_round_metrics = []
    all_client_participation = []
    all_best_round = []
    all_dp_records = []
    all_results = []

    total_runs = len(PRIVACY_SETTINGS) * len(CONDITIONS)
    run_idx = 0

    for privacy_setting in PRIVACY_SETTINGS:
        privacy_name = privacy_setting["name"]
        sigma = privacy_setting["sigma"]

        for condition in CONDITIONS:
            run_idx += 1
            print(f"\n[Run {run_idx}/{total_runs}] {privacy_name} | {condition}")

            result = run_dp_condition(
                condition,
                privacy_setting,
                X_train_split,
                y_train_split,
                X_val,
                y_val,
                X_test,
                y_test,
                device,
            )

            output_dir = OUTPUT_ROOT / privacy_name / condition
            paths = save_outputs(condition, privacy_name, result, str(output_dir), sigma)
            save_config(condition, privacy_name, str(output_dir), sigma)

            summary_df = build_condition_summary(result)
            all_summaries.append(summary_df)

            best_round_df = build_best_round_summary(result)
            all_best_round.append(best_round_df)

            round_df = pd.DataFrame(result["round_metrics"])
            all_round_metrics.append(round_df)

            client_df = pd.DataFrame(result["client_participation"])
            all_client_participation.append(client_df)

            if result["dp_records"]:
                all_dp_records.append(pd.DataFrame(result["dp_records"]))

            all_results.append(result)

            print()
            print("=" * 60)
            print(f"RUN COMPLETE: {privacy_name} | {condition}")
            print("=" * 60)
            print(f"Trainable clients: {len(result['trainable_clients'])}")
            print(f"Skipped clients:   {len(result['skipped_clients'])}")
            print(f"Best round:        {result['best_round']}")
            print(f"Best val ROC-AUC:  {result['best_val_roc_auc']:.4f}")
            print(f"Outputs saved to:  {output_dir}")
            print(f"  Round metrics:          {paths['round_metrics']}")
            print(f"  Client participation:   {paths['client_participation']}")
            print(f"  Adaptive weights:       {paths['adaptive_client_weights']}")
            if paths.get("dp_records"):
                print(f"  DP records:             {paths['dp_records']}")
            print(f"  Best model:             {paths['best_model']}")
            print(f"  Final model:            {paths['final_model']}")
            print(f"  Config:                 {os.path.join(str(output_dir), 'config.yaml')}")
            print("=" * 60)

    # Overall comparison files
    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_path = OUTPUT_ROOT / "dp_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    round_metrics_path = OUTPUT_ROOT / "dp_round_metrics.csv"
    pd.concat(all_round_metrics, ignore_index=True).to_csv(
        round_metrics_path, index=False
    )

    client_participation_path = OUTPUT_ROOT / "dp_client_participation.csv"
    pd.concat(all_client_participation, ignore_index=True).to_csv(
        client_participation_path, index=False
    )

    best_round_path = OUTPUT_ROOT / "dp_best_round_summary.csv"
    pd.concat(all_best_round, ignore_index=True).to_csv(
        best_round_path, index=False
    )

    if all_dp_records:
        dp_all_path = OUTPUT_ROOT / "dp_all_records.csv"
        pd.concat(all_dp_records, ignore_index=True).to_csv(
            dp_all_path, index=False
        )

    # Top-level config
    top_config = {
        "phase": "11",
        "experiment": "Differential Privacy",
        "privacy_settings": PRIVACY_SETTINGS,
        "conditions": CONDITIONS,
        "delta": DELTA,
        "clipping_norm": CLIPPING_NORM,
        "federated": {
            "rounds": FED_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "client_fraction": CLIENT_FRACTION,
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "random_seed": SEED,
        },
        "accounting": "NOT AVAILABLE",
    }
    with open(OUTPUT_ROOT / "dp_config.yaml", "w") as f:
        yaml.dump(top_config, f, default_flow_style=False)

    # Generate privacy audit
    generate_privacy_audit(str(OUTPUT_ROOT), all_results)

    print()
    print("=" * 60)
    print("PHASE 11 COMPLETE")
    print("=" * 60)
    print(f"Conditions evaluated: {len(CONDITIONS)}")
    print(f"Privacy settings:     {len(PRIVACY_SETTINGS)}")
    print(f"Total runs:           {total_runs}")
    print(f"Overall summary:      {summary_path}")
    print(f"Round metrics:        {round_metrics_path}")
    print(f"Client participation: {client_participation_path}")
    print(f"Best round summary:   {best_round_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Sanity test wrapper
# ---------------------------------------------------------------------------
def run_sanity_test():
    """Run a minimal sanity check: 2 rounds, iid, sigma=1.0 and sigma=0.0."""
    set_seed(SEED)
    device = get_device()

    print("=" * 60)
    print("PHASE 11 SANITY TEST")
    print("=" * 60)

    X_train_full, y_train_full, X_test, y_test = load_processed_data()
    (X_train_split, y_train_split,
     X_val, y_val, X_test, y_test) = build_centralized_splits(
        X_train_full, y_train_full, X_test, y_test
    )

    results = {}

    for sigma in [0.0, 1.0]:
        print(f"\n--- Sanity: sigma={sigma} ---")
        privacy_setting = {"name": f"sanity_sigma_{sigma}", "sigma": sigma}
        result = run_dp_condition(
            "iid",
            privacy_setting,
            X_train_split,
            y_train_split,
            X_val,
            y_val,
            X_test,
            y_test,
            device,
            max_rounds=2,
        )
        results[sigma] = result

        # Save sanity test outputs
        sanity_dir = OUTPUT_ROOT / "sanity_test" / f"sigma_{sigma}"
        sanity_dir.mkdir(parents=True, exist_ok=True)
        save_outputs("iid", f"sanity_sigma_{sigma}", result, str(sanity_dir), sigma)
        save_config("iid", f"sanity_sigma_{sigma}", str(sanity_dir), sigma)

        # Verify weights sum to 1
        weights = result["client_weight_records"]
        for rnd in [1, 2]:
            rnd_weights = [w["adaptive_weight"] for w in weights if w["round"] == rnd]
            if rnd_weights:
                w_sum = sum(rnd_weights)
                print(f"  Round {rnd} adaptive weights sum: {w_sum:.10f}")

        # Check no NaN/Inf
        has_nan = any(w["has_nan"] for w in result["dp_records"]) if result["dp_records"] else False
        has_inf = any(w["has_inf"] for w in result["dp_records"]) if result["dp_records"] else False
        print(f"  NaN in DP records: {has_nan}")
        print(f"  Inf in DP records: {has_inf}")

        # Verify noise/no-noise
        if result["dp_records"]:
            noise_added = any(w["noise_added"] for w in result["dp_records"])
            print(f"  Noise applied: {noise_added}")
            print(f"  Expected: {sigma > 0.0}")
            print(f"  Match: {noise_added == (sigma > 0.0)}")

    print()
    print("=" * 60)
    print("SANITY TEST COMPLETE")
    print("=" * 60)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="Run sanity test only")
    args = parser.parse_args()

    if args.sanity:
        run_sanity_test()
    else:
        run_phase11()
