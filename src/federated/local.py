"""
HEALIX-FL Phase 6B: IID Local-Only Client Training

Trains 10 IID clients independently using the same MLP architecture
and training configuration as the Phase 5 centralized baseline.
"""

from pathlib import Path
import random

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

from src.models.mlp import MLP


SEED = 42
NUM_CLIENTS = 10
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001
VAL_SIZE = 0.20

CLIENT_ROOT = Path("data/clients/non_iid/iid")
OUTPUT_ROOT = Path("results/local")
MODEL_ROOT = OUTPUT_ROOT / "models"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_client(client_id):
    client_dir = CLIENT_ROOT / f"client_{client_id}"

    x_path = client_dir / "X.npy"
    y_path = client_dir / "y.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim != 2:
        raise ValueError(
            f"client_{client_id}: X must be 2D, got shape {X.shape}"
        )

    if X.shape[1] != 2370:
        raise ValueError(
            f"client_{client_id}: expected 2370 features, got {X.shape[1]}"
        )

    unique_targets = np.unique(y)

    if not np.all(np.isin(unique_targets, [0, 1])):
        raise ValueError(
            f"client_{client_id}: target must be binary, got {unique_targets}"
        )

    return X.astype(np.float32), y.astype(np.float32)


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, probabilities)
        pr_auc = average_precision_score(y_true, probabilities)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch).squeeze(1)

            loss = criterion(logits, y_batch)

            batch_size = X_batch.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities = torch.sigmoid(logits)

            all_targets.extend(
                y_batch.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    avg_loss = total_loss / total_samples

    metrics = calculate_metrics(
        np.asarray(all_targets, dtype=int),
        np.asarray(all_probabilities),
    )

    return avg_loss, metrics


def is_trainable(X, y):
    """Check if a client has sufficient data for training.

    Returns True if the client has at least 2 samples.
    Clients with only 1 sample cannot be split into train/validation
    and are marked as insufficient_data.
    """
    return len(y) >= 2


def train_client(client_id, X, y, device, model_root=None):
    print()
    print("=" * 60)
    print(f"CLIENT {client_id}")
    print("=" * 60)

    positive_rate = float(np.mean(y))

    print(f"Samples: {len(y)}")
    print(f"Features: {X.shape[1]}")
    print(f"Positive rate: {positive_rate:.4%}")
    print(
        f"Positive samples: {int(np.sum(y == 1))}"
    )
    print(
        f"Negative samples: {int(np.sum(y == 0))}"
    )

    indices = np.arange(len(y))

    unique, counts = np.unique(y, return_counts=True)

    can_stratify = (
        len(unique) == 2
        and np.min(counts) >= 2
    )

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

    X_val = X[val_idx]
    y_val = y[val_idx]

    print(
        f"Train samples: {len(y_train)}"
    )
    print(
        f"Validation samples: {len(y_val)}"
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MLP(
        input_dim=2370,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.2,
    ).to(device)

    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)

    if positive_count > 0:
        pos_weight_value = (
            negative_count / positive_count
        )
    else:
        pos_weight_value = 1.0

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    best_state = None
    best_epoch = None
    best_val_loss = np.inf
    best_roc_auc = -np.inf
    best_pr_auc = -np.inf

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch).squeeze(1)

            loss = criterion(logits, y_batch)

            loss.backward()

            optimizer.step()

            batch_size = X_batch.size(0)

            total_train_loss += (
                loss.item() * batch_size
            )

            total_train_samples += batch_size

        train_loss = (
            total_train_loss /
            total_train_samples
        )

        val_loss, val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )

        roc_auc = val_metrics["roc_auc"]
        pr_auc = val_metrics["pr_auc"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": roc_auc,
            "val_pr_auc": pr_auc,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val ROC-AUC: "
            f"{roc_auc:.4f} | "
            f"Val PR-AUC: "
            f"{pr_auc:.4f}"
        )

        is_better = False

        if not np.isnan(roc_auc):
            if roc_auc > best_roc_auc:
                is_better = True
            elif (
                roc_auc == best_roc_auc
                and not np.isnan(pr_auc)
                and pr_auc > best_pr_auc
            ):
                is_better = True
        elif not np.isnan(pr_auc):
            if (
                np.isnan(best_pr_auc)
                or pr_auc > best_pr_auc
            ):
                is_better = True
        else:
            if val_loss < best_val_loss:
                is_better = True

        if is_better:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            best_epoch = epoch
            best_val_loss = val_loss
            best_roc_auc = roc_auc
            best_pr_auc = pr_auc

    if best_state is None:
        raise RuntimeError(
            f"Could not select best model for client_{client_id}"
        )

    model.load_state_dict(best_state)

    final_val_loss, final_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        device,
    )

    model_path = (
        model_root /
        f"client_{client_id}_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path,
    )

    result = {
        "client": client_id,
        "num_samples": len(y),
        "positive_rate": positive_rate,
        "accuracy": final_metrics["accuracy"],
        "precision": final_metrics["precision"],
        "recall": final_metrics["recall"],
        "f1_score": final_metrics["f1_score"],
        "roc_auc": final_metrics["roc_auc"],
        "pr_auc": final_metrics["pr_auc"],
        "tp": final_metrics["tp"],
        "tn": final_metrics["tn"],
        "fp": final_metrics["fp"],
        "fn": final_metrics["fn"],
        "best_epoch": best_epoch,
        "best_val_loss": final_val_loss,
        "best_val_roc_auc": final_metrics["roc_auc"],
        "best_val_pr_auc": final_metrics["pr_auc"],
        "status": "completed",
    }

    return result, history


def skip_client(client_id, X, y, reason="insufficient_data"):
    """Create a result record for a client that cannot be trained.

    Used when a client has only 1 sample and cannot be split into
    train/validation sets. The client is recorded with NaN metrics
    and a status of 'insufficient_data'. No model is created.
    """
    positive_rate = float(np.mean(y)) if len(y) > 0 else 0.0

    print()
    print("=" * 60)
    print(f"CLIENT {client_id} - SKIPPED")
    print("=" * 60)
    print(f"Samples: {len(y)}")
    print(f"Reason: {reason}")
    print("This client has insufficient data for training.")
    print("No model will be created for this client.")

    result = {
        "client": client_id,
        "num_samples": len(y),
        "positive_rate": positive_rate,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "best_epoch": np.nan,
        "best_val_loss": np.nan,
        "best_val_roc_auc": np.nan,
        "best_val_pr_auc": np.nan,
        "status": reason,
    }

    return result, []


def main():
    set_seed(SEED)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print("=" * 60)
    print("HEALIX-FL: Phase 6B - IID Local-Only Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Random seed: {SEED}")
    print("=" * 60)

    results = []
    all_history = []

    for client_id in range(1, NUM_CLIENTS + 1):

        print(
            f"\nLoading client_{client_id}..."
        )

        X, y = load_client(client_id)

        result, history = train_client(
            client_id,
            X,
            y,
            device,
        )

        results.append(result)

        for row in history:
            row["client"] = client_id
            all_history.append(row)

        print(
            f"Client {client_id} completed."
        )

    metrics_df = pd.DataFrame(results)

    history_df = pd.DataFrame(all_history)

    metrics_path = (
        OUTPUT_ROOT /
        "iid_client_metrics.csv"
    )

    summary_path = (
        OUTPUT_ROOT /
        "iid_summary.csv"
    )

    history_path = (
        OUTPUT_ROOT /
        "iid_training_history.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]

    summary_rows = []

    for metric in metric_columns:

        summary_rows.append({
            "metric": metric,
            "mean": metrics_df[metric].mean(),
            "std": metrics_df[metric].std(),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("PHASE 6B COMPLETE")
    print("=" * 60)

    print(
        f"Clients completed: "
        f"{len(metrics_df)}/{NUM_CLIENTS}"
    )

    print()
    print("IID CLIENT SUMMARY")
    print("-" * 60)

    for _, row in summary_df.iterrows():
        print(
            f"{row['metric']:12s} | "
            f"Mean: {row['mean']:.4f} | "
            f"Std: {row['std']:.4f}"
        )

    print()
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"History: {history_path}")
    print(f"Models: {MODEL_ROOT}")
    print("=" * 60)


def run_phase6c(config=None):
    """Phase 6C: Non-IID Local-Only Client Training.

    Trains 10 clients independently for each of the 4 non-IID conditions:
    mild_non_iid, moderate_non_iid, strong_non_iid, extreme_non_iid.

    Clients with only 1 sample are skipped (status='insufficient_data').
    """
    set_seed(SEED)

    conditions = [
        "mild_non_iid",
        "moderate_non_iid",
        "strong_non_iid",
        "extreme_non_iid",
    ]

    output_root = Path("results/local/non_iid")
    output_root.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 60)
    print("HEALIX-FL: Phase 6C - Non-IID Local-Only Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Conditions: {conditions}")
    print(f"Clients per condition: {NUM_CLIENTS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Random seed: {SEED}")
    print("=" * 60)

    all_results = []
    all_history = []
    skipped_clients = []

    for condition in conditions:
        condition_root = Path("data/clients/non_iid") / condition
        condition_model_root = output_root / "models" / condition
        condition_model_root.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 60)
        print(f"CONDITION: {condition}")
        print("=" * 60)

        for client_id in range(1, NUM_CLIENTS + 1):
            client_dir = condition_root / f"client_{client_id}"
            x_path = client_dir / "X.npy"
            y_path = client_dir / "y.npy"

            if not x_path.exists():
                print(f"WARNING: Missing {x_path}, skipping client_{client_id}")
                skipped_clients.append(f"{condition}/client_{client_id}")
                continue

            X = np.load(x_path).astype(np.float32)
            y = np.load(y_path).astype(np.float32)

            if not is_trainable(X, y):
                print(f"\nSkipping client_{client_id}: only {len(y)} sample(s)")
                result, history = skip_client(
                    client_id, X, y, reason="insufficient_data"
                )
                result["condition"] = condition
                all_results.append(result)
                skipped_clients.append(f"{condition}/client_{client_id}")
                continue

            result, history = train_client(
                client_id,
                X,
                y,
                device,
                model_root=condition_model_root,
            )
            result["condition"] = condition
            all_results.append(result)

            for row in history:
                row["condition"] = condition
                row["client"] = client_id
                all_history.append(row)

    metrics_df = pd.DataFrame(all_results)
    history_df = pd.DataFrame(all_history)

    metrics_path = output_root / "non_iid_client_metrics.csv"
    summary_path = output_root / "non_iid_summary.csv"
    history_path = output_root / "non_iid_training_history.csv"

    metrics_df.to_csv(metrics_path, index=False)
    history_df.to_csv(history_path, index=False)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]

    summary_rows = []
    for condition in conditions:
        condition_df = metrics_df[metrics_df["condition"] == condition]
        for metric in metric_columns:
            summary_rows.append({
                "condition": condition,
                "metric": metric,
                "mean": condition_df[metric].mean(),
                "std": condition_df[metric].std(),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print()
    print("=" * 60)
    print("PHASE 6C COMPLETE")
    print("=" * 60)

    trainable = metrics_df[metrics_df["status"] == "completed"]
    print(f"Clients trained: {len(trainable)}")
    print(f"Clients skipped: {len(skipped_clients)}")
    if skipped_clients:
        print(f"Skipped: {skipped_clients}")

    print()
    print("NON-IID CLIENT SUMMARY")
    print("-" * 60)
    for condition in conditions:
        cond_df = metrics_df[metrics_df["condition"] == condition]
        cond_trainable = cond_df[cond_df["status"] == "completed"]
        print(f"\n{condition}: {len(cond_trainable)} trainable, "
              f"{len(cond_df) - len(cond_trainable)} skipped")
        for _, row in summary_df[
            summary_df["condition"] == condition
        ].iterrows():
            print(f"  {row['metric']:12s} | "
                  f"Mean: {row['mean']:.4f} | "
                  f"Std: {row['std']:.4f}")

    print()
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"History: {history_path}")
    print(f"Models: {output_root / 'models'}")
    print("=" * 60)


def run_phase6(config=None):
    """Wrapper for Phase 6B IID local-only client training.

    This function provides the entry point expected by main.py.
    It delegates to the existing main() function which contains
    the complete Phase 6B implementation.
    """
    main()


if __name__ == "__main__":
    main()
