"""
HEALIX-FL Phase 9: Personalized Federated Learning Baseline.

Implements a simple personalized federated learning baseline:
1. Maintain one global MLP (standard architecture from Phase 5).
2. Perform federated global training (FedAvg-style).
3. Each participating client starts from the current global model.
4. Train the client locally on its own data for 1 epoch.
5. Keep a personalized client-specific model after local training.
6. Evaluate each client using its personalized model.
7. Aggregate the client models to produce the next global model using sample-count weighted averaging.
8. Repeat for 20 rounds.

This is a BASELINE for comparison - NOT a novel method.
Uses the standard MLP: Linear(2370->128) -> ReLU -> Dropout(0.2) -> Linear(128->64) -> ReLU -> Dropout(0.2) -> Linear(64->1)
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
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

from src.models.mlp import MLP

# Reproducibility and experiment configuration
SEED = 42
FED_ROUNDS = 20
LOCAL_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 0.001
CLIENT_FRACTION = 1.0
VAL_SIZE = 0.20
NUM_CLIENTS = 10

CONDITIONS = [
    "iid",
    "mild_non_iid",
    "moderate_non_iid",
    "strong_non_iid",
    "extreme_non_iid",
]

CLIENTS_DATA_ROOT = Path("data/clients/non_iid")
PROCESSED_DIR = Path("data/processed")
OUTPUT_ROOT = Path("results/personalized")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_processed_data():
    """Load the existing processed train/test split from Phase 2/5."""
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
    """Recreate the Phase 5 centralized train/validation/test split."""
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
# Metrics
# ---------------------------------------------------------------------------
def calculate_metrics(y_true, probabilities):
    """Compute binary classification metrics.

    ROC-AUC / PR-AUC are NaN when the evaluation target contains only one class.
    """
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
    """Evaluate a model on a fixed numpy array (validation or test)."""
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
# Client data loading
# ---------------------------------------------------------------------------
def load_client_data(condition, client_id):
    """Load a single client's X.npy and y.npy."""
    client_dir = CLIENTS_DATA_ROOT / condition / f"client_{client_id}"
    x_path = client_dir / "X.npy"
    y_path = client_dir / "y.npy"

    if not x_path.exists() or not y_path.exists():
        return None, None

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)
    return X, y


def is_trainable(X, y):
    """A client is trainable only if it has at least 2 samples."""
    return len(y) >= 2


def get_trainable_clients(condition):
    """Return the list of trainable client IDs for a condition."""
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
# Local personalized training
# ---------------------------------------------------------------------------
def train_local_personalized(global_state_dict, X, y, device):
    """Train one local personalized model for one local epoch.

    Args:
        global_state_dict: Current global model parameters.
        X: Client training features (numpy float32).
        y: Client training targets (numpy float32).

    Returns:
        (local_state_dict, num_train_samples, status)
    """
    model = MLP(
        input_dim=2370,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(global_state_dict)
    model.train()

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

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch).squeeze(1)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

    local_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    num_train_samples = int(len(y_train))

    return local_state_dict, num_train_samples, "completed"


# ---------------------------------------------------------------------------
# Aggregation (full model)
# ---------------------------------------------------------------------------
def federated_average(local_states, sample_counts):
    """Aggregate local model parameters using sample-count-weighted averaging.

    global_param = sum(n_k * param_k) / sum(n_k)

    Args:
        local_states: List of state dicts from participating clients.
        sample_counts: List of training sample counts, aligned with local_states.

    Returns:
        Aggregated state dict.
    """
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
    """Create a fresh global MLP with the Phase 5 architecture."""
    model = MLP(
        input_dim=2370,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.2,
    ).to(device)
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return state


# ---------------------------------------------------------------------------
# Single-condition personalized experiment
# ---------------------------------------------------------------------------
def run_personalized_condition(condition, X_train_split, y_train_split,
                                X_val, y_val, X_test, y_test, device):
    """Run personalized federated learning for one condition.

    Returns a dict with round metrics, client info, per-client metrics,
    best global state, best round, and final global state.
    """
    trainable_clients, skipped_clients = get_trainable_clients(condition)

    print()
    print("=" * 60)
    print("CONDITION: " + condition)
    print("=" * 60)
    print("Trainable clients: " + str(trainable_clients))
    print("Skipped clients:   " + str(skipped_clients))
    print("Federated rounds:  " + str(FED_ROUNDS))
    print("Local epochs:      " + str(LOCAL_EPOCHS))
    print("Client fraction:   " + str(CLIENT_FRACTION))
    print("=" * 60)

    # Initialize global model
    global_state = initialize_global_model(device)

    # Each trainable client keeps its own personalized model across all rounds
    personalized_states = {
        client_id: None for client_id in trainable_clients
    }

    round_metrics = []
    client_participation = []
    client_round_metrics = []
    best_val_roc_auc = -np.inf
    best_global_state = None
    best_round = None
    best_personalized_states = None

    # Track best personalized models for checkpointing
    best_personalized_states = None

    for rnd in range(1, FED_ROUNDS + 1):
        print(f"[Phase 9] {condition} | Round {rnd}/{FED_ROUNDS} | Trainable clients: {len(trainable_clients)}", flush=True)
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
                    "status": "missing_data",
                })
                continue

            local_state, num_train_samples, status = train_local_personalized(
                global_state, X, y, device,
            )

            # Update this client's personalized model (kept locally)
            personalized_states[client_id] = local_state

            # Return full model for aggregation
            local_states.append(local_state)
            sample_counts.append(num_train_samples)

            client_participation.append({
                "condition": condition,
                "round": rnd,
                "client": client_id,
                "num_samples": num_train_samples,
                "local_epochs": LOCAL_EPOCHS,
                "status": status,
            })

        if len(local_states) == 0:
            print(f"[Phase 9] {condition} | Round {rnd}: no trainable clients, skipping aggregation.", flush=True)
            continue

        # Aggregate full model parameters using sample-count weighted averaging
        global_state = federated_average(local_states, sample_counts)

        # Evaluate each client's personalized model on validation set only (for model selection)
        client_val_roc_aucs = []
        client_val_pr_aucs = []
        client_val_f1s = []
        client_val_accs = []
        client_val_precs = []
        client_val_recs = []

        for client_id in selected_clients:
            model = MLP(
                input_dim=2370,
                hidden_dim1=128,
                hidden_dim2=64,
                dropout=0.2,
            ).to(device)
            model.load_state_dict(personalized_states[client_id])
            model.eval()

            val_loss, val_metrics = evaluate_model(model, X_val, y_val, device)

            client_round_metrics.append({
                "condition": condition,
                "round": rnd,
                "client": client_id,
                "num_samples": load_client_data(condition, client_id)[0].shape[0],
                "val_loss": float(val_loss),
                **{"val_" + k: v for k, v in val_metrics.items()},
            })

            if not np.isnan(val_metrics["roc_auc"]):
                client_val_roc_aucs.append(val_metrics["roc_auc"])
            if not np.isnan(val_metrics["pr_auc"]):
                client_val_pr_aucs.append(val_metrics["pr_auc"])
            if not np.isnan(val_metrics["f1_score"]):
                client_val_f1s.append(val_metrics["f1_score"])
            if not np.isnan(val_metrics["accuracy"]):
                client_val_accs.append(val_metrics["accuracy"])
            if not np.isnan(val_metrics["precision"]):
                client_val_precs.append(val_metrics["precision"])
            if not np.isnan(val_metrics["recall"]):
                client_val_recs.append(val_metrics["recall"])

        # Average personalized-client validation metrics (primary model selection metric)
        if len(client_val_roc_aucs) > 0:
            personalized_val_roc_auc = float(np.mean(client_val_roc_aucs))
            personalized_val_pr_auc = float(np.mean(client_val_pr_aucs)) if client_val_pr_aucs else np.nan
            personalized_val_f1 = float(np.mean(client_val_f1s)) if client_val_f1s else np.nan
            personalized_val_acc = float(np.mean(client_val_accs)) if client_val_accs else np.nan
            personalized_val_prec = float(np.mean(client_val_precs)) if client_val_precs else np.nan
            personalized_val_rec = float(np.mean(client_val_recs)) if client_val_recs else np.nan
            personalized_val_roc_std = float(np.std(client_val_roc_aucs)) if len(client_val_roc_aucs) > 1 else 0.0
        else:
            personalized_val_roc_auc = np.nan
            personalized_val_pr_auc = np.nan
            personalized_val_f1 = np.nan
            personalized_val_acc = np.nan
            personalized_val_prec = np.nan
            personalized_val_rec = np.nan
            personalized_val_roc_std = np.nan

        # Create round metrics with only validation (no test evaluation during rounds)
        # Use the last client's val_loss as representative
        round_metrics.append({
            "condition": condition,
            "round": rnd,
            "val_loss": float(val_loss),
            "val_accuracy": personalized_val_acc,
            "val_precision": personalized_val_prec,
            "val_recall": personalized_val_rec,
            "val_f1_score": personalized_val_f1,
            "val_roc_auc": personalized_val_roc_auc,
            "val_pr_auc": personalized_val_pr_auc,
            "val_roc_auc_std": personalized_val_roc_std,
        })

        print(f"[Phase 9] {condition} | Round {rnd}/{FED_ROUNDS} | Personalized Val ROC-AUC: {personalized_val_roc_auc:.4f} (+/- {personalized_val_roc_std:.4f})", flush=True)

        # Best model selection using personalized validation ROC-AUC
        if not np.isnan(personalized_val_roc_auc) and personalized_val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = personalized_val_roc_auc
            best_global_state = {
                key: value.detach().cpu().clone()
                for key, value in global_state.items()
            }
            best_personalized_states = {
                cid: {
                    key: value.detach().cpu().clone()
                    for key, value in state.items()
                }
                for cid, state in personalized_states.items()
            }
            best_round = rnd
            print(f"[Phase 9] {condition} | Round {rnd} -> NEW BEST (Val ROC-AUC: {best_val_roc_auc:.4f})", flush=True)

    if best_global_state is None:
        best_global_state = {
            key: value.detach().cpu().clone()
            for key, value in global_state.items()
        }
        best_personalized_states = {
            cid: {
                key: value.detach().cpu().clone()
                for key, value in state.items()
            }
            for cid, state in personalized_states.items()
        }
        best_round = FED_ROUNDS

    # Final test evaluation using best-round personalized models
    print(f"[Phase 9] {condition} | Final test evaluation using best round {best_round}", flush=True)
    test_client_metrics = []
    test_roc_aucs = []
    test_pr_aucs = []
    test_f1s = []
    test_accs = []
    test_precs = []
    test_recs = []

    for client_id in trainable_clients:
        model = MLP(
            input_dim=2370,
            hidden_dim1=128,
            hidden_dim2=64,
            dropout=0.2,
        ).to(device)
        model.load_state_dict(best_personalized_states[client_id])
        model.eval()

        test_loss, test_metrics = evaluate_model(model, X_test, y_test, device)

        test_client_metrics.append({
            "condition": condition,
            "client": client_id,
            "best_round": best_round,
            "test_loss": float(test_loss),
            **{"test_" + k: v for k, v in test_metrics.items()},
        })

        if not np.isnan(test_metrics["roc_auc"]):
            test_roc_aucs.append(test_metrics["roc_auc"])
        if not np.isnan(test_metrics["pr_auc"]):
            test_pr_aucs.append(test_metrics["pr_auc"])
        if not np.isnan(test_metrics["f1_score"]):
            test_f1s.append(test_metrics["f1_score"])
        if not np.isnan(test_metrics["accuracy"]):
            test_accs.append(test_metrics["accuracy"])
        if not np.isnan(test_metrics["precision"]):
            test_precs.append(test_metrics["precision"])
        if not np.isnan(test_metrics["recall"]):
            test_recs.append(test_metrics["recall"])

    avg_test_roc_auc = float(np.mean(test_roc_aucs)) if test_roc_aucs else np.nan
    avg_test_pr_auc = float(np.mean(test_pr_aucs)) if test_pr_aucs else np.nan
    avg_test_f1 = float(np.mean(test_f1s)) if test_f1s else np.nan
    avg_test_acc = float(np.mean(test_accs)) if test_accs else np.nan
    avg_test_prec = float(np.mean(test_precs)) if test_precs else np.nan
    avg_test_rec = float(np.mean(test_recs)) if test_recs else np.nan
    std_test_roc_auc = float(np.std(test_roc_aucs)) if len(test_roc_aucs) > 1 else 0.0
    std_test_pr_auc = float(np.std(test_pr_aucs)) if len(test_pr_aucs) > 1 else 0.0
    std_test_f1 = float(np.std(test_f1s)) if len(test_f1s) > 1 else 0.0

    print(f"[Phase 9] {condition} | Final Test ROC-AUC: {avg_test_roc_auc:.4f} (+/- {std_test_roc_auc:.4f})", flush=True)
    print(f"[Phase 9] {condition} | Final Test F1: {avg_test_f1:.4f} (+/- {std_test_f1:.4f})", flush=True)
    print(f"[Phase 9] Completed condition: {condition}", flush=True)

    return {
        "condition": condition,
        "trainable_clients": trainable_clients,
        "skipped_clients": skipped_clients,
        "round_metrics": round_metrics,
        "client_participation": client_participation,
        "client_round_metrics": client_round_metrics,
        "test_client_metrics": test_client_metrics,
        "personalized_states": best_personalized_states,
        "best_global_state": best_global_state,
        "final_global_state": global_state,
        "best_round": best_round,
        "best_val_roc_auc": best_val_roc_auc,
        "avg_test_roc_auc": avg_test_roc_auc,
        "avg_test_pr_auc": avg_test_pr_auc,
        "avg_test_f1": avg_test_f1,
        "avg_test_accuracy": avg_test_acc,
        "avg_test_precision": avg_test_prec,
        "avg_test_recall": avg_test_rec,
        "std_test_roc_auc": std_test_roc_auc,
        "std_test_pr_auc": std_test_pr_auc,
        "std_test_f1": std_test_f1,
    }


# ---------------------------------------------------------------------------
# Ensemble evaluation across personalized models
# ---------------------------------------------------------------------------
def evaluate_personalized_ensemble(personalized_states, client_ids, X, y, device,
                                    batch_size=BATCH_SIZE):
    """Evaluate a personalized ensemble on a centralized numpy array.

    For each client, build their personalized model once, then predict
    over all batches. Client-level probabilities are averaged
    (macro-averaged ensemble) and evaluated against the centralized targets.

    Returns:
        (avg_loss, metrics_dict)
    """
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()

    # Build each client's complete model ONCE (not per batch).
    models = []
    for client_id in client_ids:
        model = MLP(
            input_dim=2370,
            hidden_dim1=128,
            hidden_dim2=64,
            dropout=0.2,
        ).to(device)
        model.load_state_dict(personalized_states[client_id])
        model.eval()
        models.append(model)

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            batch_probs = None
            batch_loss_sum = 0.0

            for model in models:
                logits = model(X_batch).squeeze(1)
                probs = torch.sigmoid(logits)

                loss = criterion(logits, y_batch.squeeze(1))
                batch_loss_sum += loss.item()

                if batch_probs is None:
                    batch_probs = probs.clone()
                else:
                    batch_probs += probs

            batch_probs = batch_probs / len(models)
            batch_loss = batch_loss_sum / len(models)

            bs = X_batch.size(0)
            total_loss += batch_loss * bs
            total_samples += bs

            all_targets.extend(y_batch.cpu().numpy().flatten().tolist())
            all_probs.extend(batch_probs.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / total_samples
    metrics = calculate_metrics(
        np.asarray(all_targets, dtype=int),
        np.asarray(all_probs),
    )
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------
def save_config(condition, output_dir):
    """Save the personalized federated learning configuration for a condition."""
    config = {
        "phase": "9",
        "experiment": "Personalized Federated Learning Baseline",
        "description": "Simple personalization baseline: each client trains full MLP locally, "
                       "full models aggregated via FedAvg, personalized models kept for evaluation.",
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
            "aggregation": "sample_count_weighted_fedavg (full model)",
            "personalization": "each client keeps its locally trained full model",
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
            "personalized_evaluation": "each client's personalized model evaluated on test set",
        },
        "condition": condition,
    }
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def save_outputs(condition, result, output_dir):
    """Save round metrics, client participation, models for a condition."""
    os.makedirs(output_dir, exist_ok=True)

    round_df = pd.DataFrame(result["round_metrics"])
    round_path = os.path.join(output_dir, "round_metrics.csv")
    round_df.to_csv(round_path, index=False)

    client_df = pd.DataFrame(result["client_participation"])
    client_path = os.path.join(output_dir, "client_participation.csv")
    client_df.to_csv(client_path, index=False)

    client_round_df = pd.DataFrame(result["client_round_metrics"])
    client_round_path = os.path.join(output_dir, "client_round_metrics.csv")
    client_round_df.to_csv(client_round_path, index=False)

    # Save test client metrics
    if "test_client_metrics" in result:
        test_client_df = pd.DataFrame(result["test_client_metrics"])
        test_client_path = os.path.join(output_dir, "client_metrics.csv")
        test_client_df.to_csv(test_client_path, index=False)

    best_path = os.path.join(output_dir, "best_global_model.pt")
    torch.save(result["best_global_state"], best_path)

    final_path = os.path.join(output_dir, "final_global_model.pt")
    torch.save(result["final_global_state"], final_path)

    # Save personalized models
    personalized_dir = os.path.join(output_dir, "personalized_models")
    os.makedirs(personalized_dir, exist_ok=True)
    for cid, state in result["personalized_states"].items():
        torch.save(state, os.path.join(personalized_dir, "client_" + str(cid) + "_model.pt"))

    return {
        "round_metrics": round_path,
        "client_participation": client_path,
        "client_round_metrics": client_round_path,
        "client_metrics": test_client_path if "test_client_metrics" in result else None,
        "best_model": best_path,
        "final_model": final_path,
        "personalized_models": personalized_dir,
    }


def build_condition_summary(result):
    """Build a per-condition summary DataFrame from round metrics."""
    round_df = pd.DataFrame(result["round_metrics"])
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]
    rows = []
    for col in metric_cols:
        vals = round_df["val_" + col].astype(float)
        rows.append({
            "condition": result["condition"],
            "metric": col,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "best": float(vals.max()),
        })
    return pd.DataFrame(rows)


def build_best_round_summary(result):
    """Build a per-condition best-round summary DataFrame.

    Uses the final test evaluation of the best-round personalized models.
    """
    best_round = int(result["best_round"])

    row = {
        "condition": result["condition"],
        "best_round": best_round,
        "best_val_roc_auc": float(result["best_val_roc_auc"]),
        "test_accuracy": float(result["avg_test_accuracy"]),
        "test_precision": float(result["avg_test_precision"]),
        "test_recall": float(result["avg_test_recall"]),
        "test_f1_score": float(result["avg_test_f1"]),
        "test_roc_auc": float(result["avg_test_roc_auc"]),
        "test_pr_auc": float(result["avg_test_pr_auc"]),
        "avg_personalized_test_roc_auc": float(result["avg_test_roc_auc"]),
        "avg_personalized_test_pr_auc": float(result["avg_test_pr_auc"]),
        "avg_personalized_test_f1": float(result["avg_test_f1"]),
        "std_personalized_test_roc_auc": float(result["std_test_roc_auc"]),
        "std_personalized_test_pr_auc": float(result["std_test_pr_auc"]),
        "std_personalized_test_f1": float(result["std_test_f1"]),
        "trainable_clients": len(result["trainable_clients"]),
        "skipped_clients": len(result["skipped_clients"]),
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Phase 9 entry point
# ---------------------------------------------------------------------------
def run_phase9(config=None):
    """Execute personalized federated learning for all five conditions.

    Runs personalized federated learning separately for:
        iid, mild_non_iid, moderate_non_iid, strong_non_iid, extreme_non_iid

    Args:
        config: Optional config dictionary (unused; kept for API symmetry
                with other phases).
    """
    set_seed(SEED)
    device = get_device()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HEALIX-FL: Phase 9 - Personalized Federated Learning")
    print("=" * 60)
    print("Personalization design: Simple baseline (full model per client)")
    print("  GLOBAL MODEL: standard MLP aggregated via FedAvg")
    print("  PERSONALIZED MODELS: each client's locally trained full MLP")
    print("Device: " + str(device))
    print("Federated rounds: " + str(FED_ROUNDS))
    print("Local epochs: " + str(LOCAL_EPOCHS))
    print("Client fraction: " + str(CLIENT_FRACTION))
    print("Learning rate: " + str(LEARNING_RATE))
    print("Batch size: " + str(BATCH_SIZE))
    print("Random seed: " + str(SEED))
    print("Conditions: " + str(CONDITIONS))
    print("=" * 60)

    X_train_full, y_train_full, X_test, y_test = load_processed_data()
    (X_train_split, y_train_split,
     X_val, y_val, X_test, y_test) = build_centralized_splits(
        X_train_full, y_train_full, X_test, y_test
    )

    print()
    print("Centralized split sizes:")
    print("  Train:      " + format(len(X_train_split), ","))
    print("  Validation: " + format(len(X_val), ","))
    print("  Test:       " + format(len(X_test), ","))

    all_summaries = []
    all_round_metrics = []
    all_client_participation = []
    all_client_round_metrics = []
    all_test_client_metrics = []
    all_best_round = []

    for condition in CONDITIONS:
        print(f"[Phase 9] Starting condition: {condition}", flush=True)
        result = run_personalized_condition(
            condition,
            X_train_split,
            y_train_split,
            X_val,
            y_val,
            X_test,
            y_test,
            device,
        )

        output_dir = OUTPUT_ROOT / condition
        paths = save_outputs(condition, result, str(output_dir))
        save_config(condition, str(output_dir))

        summary_df = build_condition_summary(result)
        all_summaries.append(summary_df)

        best_round_df = build_best_round_summary(result)
        all_best_round.append(best_round_df)

        round_df = pd.DataFrame(result["round_metrics"])
        round_df["condition"] = condition
        all_round_metrics.append(round_df)

        client_df = pd.DataFrame(result["client_participation"])
        all_client_participation.append(client_df)

        client_round_df = pd.DataFrame(result["client_round_metrics"])
        client_round_df["condition"] = condition
        all_client_round_metrics.append(client_round_df)

        if "test_client_metrics" in result:
            test_client_df = pd.DataFrame(result["test_client_metrics"])
            test_client_df["condition"] = condition
            all_test_client_metrics.append(test_client_df)

        print()
        print("=" * 60)
        print("CONDITION " + condition + " COMPLETE")
        print("=" * 60)
        print("Trainable clients: " + str(len(result["trainable_clients"])))
        print("Skipped clients:   " + str(len(result["skipped_clients"])))
        print("Best round:        " + str(result["best_round"]))
        print("Best val ROC-AUC:  " + str(round(result["best_val_roc_auc"], 4)))
        print("Final Test ROC-AUC: " + str(round(result["avg_test_roc_auc"], 4)))
        print("Final Test F1:     " + str(round(result["avg_test_f1"], 4)))
        print("Outputs saved to:  " + str(output_dir))
        print("  Round metrics:       " + str(paths["round_metrics"]))
        print("  Client participation: " + str(paths["client_participation"]))
        print("  Client round metrics: " + str(paths["client_round_metrics"]))
        print("  Client test metrics: " + str(paths.get("client_metrics", "N/A")))
        print("  Best global model:   " + str(paths["best_model"]))
        print("  Final global model:  " + str(paths["final_model"]))
        print("  Personalized models: " + str(paths["personalized_models"]))
        print("  Config:              " + os.path.join(str(output_dir), "config.yaml"))
        print("=" * 60)

    # Overall comparison files
    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_path = OUTPUT_ROOT / "personalized_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    round_metrics_path = OUTPUT_ROOT / "personalized_round_metrics.csv"
    pd.concat(all_round_metrics, ignore_index=True).to_csv(
        round_metrics_path, index=False
    )

    client_participation_path = OUTPUT_ROOT / "personalized_client_participation.csv"
    pd.concat(all_client_participation, ignore_index=True).to_csv(
        client_participation_path, index=False
    )

    client_metrics_path = OUTPUT_ROOT / "personalized_client_metrics.csv"
    pd.concat(all_client_round_metrics, ignore_index=True).to_csv(
        client_metrics_path, index=False
    )

    if all_test_client_metrics:
        test_client_metrics_path = OUTPUT_ROOT / "personalized_test_client_metrics.csv"
        pd.concat(all_test_client_metrics, ignore_index=True).to_csv(
            test_client_metrics_path, index=False
        )

    best_round_path = OUTPUT_ROOT / "personalized_best_round_summary.csv"
    pd.concat(all_best_round, ignore_index=True).to_csv(
        best_round_path, index=False
    )

    print()
    print("=" * 60)
    print("PHASE 9 COMPLETE")
    print("=" * 60)
    print("Conditions evaluated: " + str(len(CONDITIONS)))
    print("Overall summary:      " + str(summary_path))
    print("Round metrics:        " + str(round_metrics_path))
    print("Client participation: " + str(client_participation_path))
    print("Client metrics:       " + str(client_metrics_path))
    print("Best round summary:   " + str(best_round_path))
    print("=" * 60)


if __name__ == "__main__":
    run_phase9()