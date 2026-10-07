"""
HEALIX-FL Phase 8: Federated Proximal (FedProx)

Implements FedProx by adding a proximal regularization term to the
local training objective:

    L_FedProx(w) = L_local(w) + (mu / 2) * ||w - w_global||^2

where:
- L_local(w) = BCEWithLogitsLoss on the client local training data
- mu       = FedProx proximal coefficient (0.01)
- w_global = frozen global parameters at the start of the round
- w        = current local model parameters

The global reference is cloned and detached so the proximal term does
not create gradients through the global model. The proximal term is
added to the local loss at every local training step. Local models are
then aggregated using the same sample-count-weighted FedAvg as Phase 7.

Best global model is selected on validation ROC-AUC only (test is
evaluation-only). All other settings mirror Phase 7 for a controlled
comparison.
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

# FedProx-specific coefficient (single value; no hyperparameter sweep)
MU = 0.01

CONDITIONS = [
    "iid",
    "mild_non_iid",
    "moderate_non_iid",
    "strong_non_iid",
    "extreme_non_iid",
]

CLIENTS_DATA_ROOT = Path("data/clients/non_iid")
PROCESSED_DIR = Path("data/processed")
OUTPUT_ROOT = Path("results/fedprox")


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
# Data loading (identical to Phase 7)
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
# Metrics (identical to Phase 7; NaN-safe for single-class evaluation)
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
# Client data loading (identical to Phase 7)
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
# FedProx local training
# ---------------------------------------------------------------------------
def train_local_model_fedprox(global_state_dict, X, y, device, mu=MU):
    """Train one local model for one local epoch with a FedProx proximal term.

    The local objective is:

        L_FedProx(w) = L_local(w) + (mu / 2) * ||w - w_global||^2

    The global reference parameters are cloned and detached so the proximal
    term does not create gradients through the global model.

    Args:
        global_state_dict: State dict of the current global model (frozen).
        X: Client training features (numpy float32).
        y: Client training targets (numpy float32).
        device: Torch device.
        mu: FedProx proximal coefficient.

    Returns:
        (local_state_dict, num_train_samples, status)
    """
    # Create local model and load global weights
    model = MLP(
        input_dim=2370,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(global_state_dict)
    model.train()

    # Freeze a copy of the global reference parameters for the proximal term.
    # These are detached so gradients never flow back into the global model.
    global_reference = {
        key: value.detach().to(device)
        for key, value in global_state_dict.items()
    }

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

    # Positive weighting with safe fallbacks (same as Phase 7):
    # - both classes present: pos_weight = negative_count / positive_count
    # - only negatives: pos_weight = 1.0
    # - only positives: pos_weight = 1.0 (NEVER 0)
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

    # One local epoch with the FedProx proximal term
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch).squeeze(1)
        local_loss = criterion(logits, y_batch)

        # Proximal term: (mu / 2) * sum(||w - w_global||^2)
        proximal_term = None
        for key, param in model.named_parameters():
            if param.requires_grad and key in global_reference:
                diff = param - global_reference[key]
                sq = (diff * diff).sum()
                proximal_term = sq if proximal_term is None \
                                else proximal_term + sq

        if proximal_term is None:
            total_loss = local_loss
        else:
            total_loss = local_loss + (mu / 2.0) * proximal_term

        total_loss.backward()
        optimizer.step()

    local_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    num_train_samples = int(len(y_train))

    return local_state_dict, num_train_samples, "completed"

# ---------------------------------------------------------------------------
# FedAvg aggregation (identical to Phase 7)
# ---------------------------------------------------------------------------
def federated_average(local_states, sample_counts):
    if len(local_states) == 0:
        raise ValueError("No local models provided for aggregation.")

    total_samples = sum(sample_counts)
    if total_samples == 0:
        raise ValueError("Total sample count is zero.")

    global_state = {}
    keys = local_states[0].keys()

    for key in keys:
        weighted_sum = None
        for state, count in zip(local_states, sample_counts):
            contribution = state[key].to(torch.float32) * count
            if weighted_sum is None:
                weighted_sum = contribution.clone()
            else:
                weighted_sum += contribution

        global_state[key] = weighted_sum / total_samples

    return global_state


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
# Single-condition FedProx experiment
# ---------------------------------------------------------------------------
def run_fedprox_condition(condition, X_train_split, y_train_split, X_val, y_val, X_test, y_test, device, mu=MU):
    """Run FedProx for one condition.

    Args:
        condition: One of CONDITIONS.
        X_train_split, y_train_split: Centralized train split (unused for
            local training; kept for signature symmetry with Phase 7).
        X_val, y_val: Centralized validation split.
        X_test, y_test: Centralized test split.
        device: Torch device.
        mu: FedProx proximal coefficient.

    Returns:
        dict with round metrics, client info, summary, best state dict.
    """
    trainable_clients, skipped_clients = get_trainable_clients(condition)

    print()
    print("=" * 60)
    print(f"CONDITION: {condition}")
    print("=" * 60)
    print(f"Trainable clients: {trainable_clients}")
    print(f"Skipped clients:   {skipped_clients}")
    print(f"Federated rounds:  {FED_ROUNDS}")
    print(f"Local epochs:      {LOCAL_EPOCHS}")
    print(f"Client fraction:   {CLIENT_FRACTION}")
    print(f"mu (FedProx):      {mu}")
    print("=" * 60)

    # Initialize global model
    global_model = initialize_global_model(device)
    global_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    round_metrics = []
    client_participation = []
    best_val_roc_auc = -np.inf
    best_state_dict = None
    best_round = None

    for rnd in range(1, FED_ROUNDS + 1):
        # Select all trainable clients (client_fraction = 1.0)
        selected_clients = trainable_clients

        local_states = []
        sample_counts = []

        for client_id in selected_clients:
            X, y = load_client_data(condition, client_id)
            if X is None:
                client_participation.append({
                    "condition": condition,
                    "round": rnd,
                    "client": client_id,
                    "num_samples": 0,
                    "local_epochs": LOCAL_EPOCHS,
                    "mu": mu,
                    "status": "missing_data",
                })
                continue

            local_state, num_train_samples, status = train_local_model_fedprox(
                global_state_dict, X, y, device, mu=mu
            )

            local_states.append(local_state)
            sample_counts.append(num_train_samples)

            client_participation.append({
                "condition": condition,
                "round": rnd,
                "client": client_id,
                "num_samples": num_train_samples,
                "local_epochs": LOCAL_EPOCHS,
                "mu": mu,
                "status": status,
            })

        # Aggregate using sample-count-weighted FedAvg
        if len(local_states) == 0:
            print(f"  Round {rnd:02d}: no trainable clients, skipping aggregation.")
            continue

        global_state_dict = federated_average(local_states, sample_counts)

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
            "round": rnd,
            "mu": mu,
            "participating_clients": len(local_states),
            "skipped_clients": len(skipped_clients),
            "trainable_clients": len(trainable_clients),
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
        "mu": mu,
        "trainable_clients": trainable_clients,
        "skipped_clients": skipped_clients,
        "round_metrics": round_metrics,
        "client_participation": client_participation,
        "best_state_dict": best_state_dict,
        "final_state_dict": global_state_dict,
        "best_round": best_round,
        "best_val_roc_auc": best_val_roc_auc,
    }

# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------
def save_config(condition, output_dir, mu=MU):
    config = {
        "phase": "8",
        "experiment": "FedProx",
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
            "aggregation": "sample_count_weighted_fedavg",
            "proximal_term": "mu/2 * ||w - w_global||^2",
            "mu": mu,
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


def save_outputs(condition, result, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    round_df = pd.DataFrame(result["round_metrics"])
    round_path = os.path.join(output_dir, "round_metrics.csv")
    round_df.to_csv(round_path, index=False)

    client_df = pd.DataFrame(result["client_participation"])
    client_path = os.path.join(output_dir, "client_participation.csv")
    client_df.to_csv(client_path, index=False)

    best_path = os.path.join(output_dir, "best_global_model.pt")
    torch.save(result["best_state_dict"], best_path)

    final_path = os.path.join(output_dir, "final_global_model.pt")
    torch.save(result["final_state_dict"], final_path)

    return {
        "round_metrics": round_path,
        "client_participation": client_path,
        "best_model": best_path,
        "final_model": final_path,
    }


def build_condition_summary(result):
    """Build a per-condition summary with BEST and FINAL round test metrics.

    BEST MODEL: selected using validation ROC-AUC.
    FINAL MODEL: round 20 (may or may not coincide with best).

    Returns one row per condition with all metrics as columns.
    """
    round_df = pd.DataFrame(result["round_metrics"])
    best_round = int(result["best_round"])

    best_row = round_df[round_df["round"] == best_round].iloc[0]
    final_row = round_df[round_df["round"] == FED_ROUNDS].iloc[0]

    row = {
        "condition": result["condition"],
        "best_round": best_round,
        "best_validation_roc_auc": float(best_row["val_roc_auc"]),
        "best_validation_pr_auc": float(best_row["val_pr_auc"]),
        "mu": result["mu"],
        "trainable_client_count": len(result["trainable_clients"]),
        "skipped_client_count": len(result["skipped_clients"]),
    }
    for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
        row[f"best_model_{col}"] = float(best_row[f"test_{col}"])
    for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
        row[f"final_model_{col}"] = float(final_row[f"test_{col}"])
    return pd.DataFrame([row])

# ---------------------------------------------------------------------------
# Phase 8 entry point
# ---------------------------------------------------------------------------
def run_phase8(config=None):
    """Execute FedProx for all five conditions.

    Runs FedProx separately for:
        iid, mild_non_iid, moderate_non_iid, strong_non_iid, extreme_non_iid

    Args:
        config: Optional config dictionary (unused; kept for API symmetry
                with other phases).
    """
    set_seed(SEED)
    device = get_device()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HEALIX-FL: Phase 8 - Federated Proximal (FedProx)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Federated rounds: {FED_ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print(f"Client fraction: {CLIENT_FRACTION}")
    print(f"mu (FedProx): {MU}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Random seed: {SEED}")
    print(f"Conditions: {CONDITIONS}")
    print("=" * 60)

    # Load centralized data and recreate Phase 5 split
    X_train_full, y_train_full, X_test, y_test = load_processed_data()
    (X_train_split, y_train_split,
     X_val, y_val, X_test, y_test) = build_centralized_splits(
        X_train_full, y_train_full, X_test, y_test
    )

    print()
    print(f"Centralized split sizes:")
    print(f"  Train:      {len(X_train_split):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test:       {len(X_test):,}")

    all_summaries = []
    all_round_metrics = []
    all_client_participation = []

    for condition in CONDITIONS:
        result = run_fedprox_condition(
            condition,
            X_train_split,
            y_train_split,
            X_val,
            y_val,
            X_test,
            y_test,
            device,
            mu=MU,
        )

        output_dir = OUTPUT_ROOT / condition
        paths = save_outputs(condition, result, str(output_dir))
        save_config(condition, str(output_dir), mu=MU)

        summary_df = build_condition_summary(result)
        all_summaries.append(summary_df)

        round_df = pd.DataFrame(result["round_metrics"])
        round_df["condition"] = condition
        all_round_metrics.append(round_df)

        client_df = pd.DataFrame(result["client_participation"])
        all_client_participation.append(client_df)

        print()
        print("=" * 60)
        print(f"CONDITION {condition} COMPLETE")
        print("=" * 60)
        print(f"Trainable clients: {len(result['trainable_clients'])}")
        print(f"Skipped clients:   {len(result['skipped_clients'])}")
        print(f"Best round:        {result['best_round']}")
        print(f"Best val ROC-AUC:  {result['best_val_roc_auc']:.4f}")
        print(f"Outputs saved to:  {output_dir}")
        print(f"  Round metrics:       {paths['round_metrics']}")
        print(f"  Client participation: {paths['client_participation']}")
        print(f"  Best model:          {paths['best_model']}")
        print(f"  Final model:         {paths['final_model']}")
        print(f"  Config:              {os.path.join(str(output_dir), 'config.yaml')}")
        print("=" * 60)

    # Overall summary (BEST and FINAL per condition per metric)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_path = OUTPUT_ROOT / "fedprox_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    round_metrics_path = OUTPUT_ROOT / "fedprox_round_metrics.csv"
    pd.concat(all_round_metrics, ignore_index=True).to_csv(
        round_metrics_path, index=False
    )

    client_participation_path = OUTPUT_ROOT / "fedprox_client_participation.csv"
    pd.concat(all_client_participation, ignore_index=True).to_csv(
        client_participation_path, index=False
    )

    print()
    print("=" * 60)
    print("PHASE 8 COMPLETE")
    print("=" * 60)
    print(f"Conditions evaluated: {len(CONDITIONS)}")
    print(f"Overall summary:      {summary_path}")
    print(f"Round metrics:        {round_metrics_path}")
    print(f"Client participation: {client_participation_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_phase8()
