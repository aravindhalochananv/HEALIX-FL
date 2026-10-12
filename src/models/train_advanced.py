"""Phase 16: Advanced centralized tabular model training.

This phase keeps the original Phase 5 MLP baseline intact and adds a stronger
centralized baseline based on residual fully connected blocks.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.evaluation.metrics import print_metrics, save_metrics, plot_confusion_matrix
from src.models.advanced_mlp import AdvancedMLP, count_parameters
from src.models.train import (
    check_data_leakage,
    create_data_loaders,
    evaluate_model,
    get_device,
    load_processed_data,
    save_training_history,
    set_seed,
    train_model,
)


def run_phase16(config):
    """Train the advanced centralized residual MLP."""
    print("=" * 60)
    print("HEALIX-FL: Phase 16 - Advanced Centralized Residual MLP")
    print("=" * 60)
    print()

    processed_dir = config["paths"]["processed_data"]
    results_dir = config["paths"]["results"]
    seed = config["project"]["random_seed"]

    phase16_config = config["phase16"]["advanced_centralized"]
    model_config = phase16_config["model"]
    training_config = phase16_config["training"]

    input_dim = phase16_config["input_dim"]
    width = model_config["width"]
    num_blocks = model_config["num_blocks"]
    dropout = model_config["dropout"]

    lr = training_config["learning_rate"]
    weight_decay = training_config["weight_decay"]
    batch_size = training_config["batch_size"]
    epochs = training_config["epochs"]
    model_output_path = phase16_config["model_output"]
    metrics_output_path = phase16_config["metrics_output"]
    config_output_path = phase16_config["config_output"]

    set_seed(seed)
    device = get_device()

    X_train, y_train, X_test, y_test = load_processed_data(processed_dir)
    check_data_leakage(X_train, y_train, X_test, y_test)

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    pos_weight = neg_count / pos_count

    print("\nClass distribution in training data:")
    print(f"  Positive class (readmitted <30): {pos_count:,} ({pos_count/len(y_train)*100:.2f}%)")
    print(f"  Negative class (NO or >30):      {neg_count:,} ({neg_count/len(y_train)*100:.2f}%)")
    print(f"  Calculated pos_weight: {pos_weight:.4f}")

    train_loader, val_loader, test_loader, X_train_split, y_train_split, X_val, y_val = (
        create_data_loaders(X_train, y_train, X_test, y_test, batch_size=batch_size, seed=seed)
    )

    model = AdvancedMLP(
        input_dim=input_dim,
        width=width,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    print("\nModel Architecture:")
    print(model)
    print(f"Total trainable parameters: {count_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("\nTraining Configuration:")
    print("  Optimizer: AdamW")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_weight:.4f})")
    print(f"  Random seed: {seed}")

    print("\n" + "=" * 60)
    print("ADVANCED TRAINING STARTED")
    print("=" * 60)

    history, best_model_state = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=epochs,
        device=device,
        patience=5,
    )

    print("\n" + "=" * 60)
    print("ADVANCED TRAINING COMPLETED")
    print("=" * 60)

    model.load_state_dict(best_model_state)
    model = model.to(device)

    print("\nEvaluating best advanced model on test set...")
    test_metrics, test_preds, test_labels = evaluate_model(model, test_loader, device)
    print_metrics(test_metrics, title="Advanced Model Final Test Metrics")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "model_type": "AdvancedMLP",
            "model_config": {
                "input_dim": input_dim,
                "width": width,
                "num_blocks": num_blocks,
                "dropout": dropout,
            },
            "training_config": {
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "pos_weight": pos_weight,
                "seed": seed,
            },
            "best_val_roc_auc": max(history["val_roc_auc"]),
            "best_val_pr_auc": max(history["val_pr_auc"]),
        },
        model_output_path,
    )
    print(f"Best advanced model saved to: {model_output_path}")

    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    save_metrics(test_metrics, metrics_output_path)

    import yaml

    os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
    with open(config_output_path, "w") as f:
        yaml.dump(phase16_config, f, default_flow_style=False)
    print(f"Config saved to: {config_output_path}")

    advanced_results_dir = os.path.join(results_dir, "advanced_centralized")
    os.makedirs(advanced_results_dir, exist_ok=True)
    save_training_history(history, advanced_results_dir)
    plot_confusion_matrix(
        test_metrics,
        os.path.join(advanced_results_dir, "advanced_confusion_matrix.png"),
        title="Advanced Centralized Model: Test Confusion Matrix",
    )

    print("\n" + "=" * 60)
    print("PHASE 16 COMPLETE")
    print("=" * 60)
    print(f"Model Architecture: AdvancedMLP({input_dim} -> width {width}, blocks {num_blocks} -> 1)")
    print("Dataset Sizes:")
    print(f"  Training:   {len(X_train_split):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test:       {len(X_test):,}")
    print("Final Test Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print("\nSaved Artifacts:")
    print(f"  Model:     {model_output_path}")
    print(f"  Metrics:   {metrics_output_path}")
    print(f"  Config:    {config_output_path}")
    print(f"  Plots:     {advanced_results_dir}/centralized_*.png")
    print("=" * 60)

    return {
        "model": model,
        "history": history,
        "best_model_state": best_model_state,
        "test_metrics": test_metrics,
        "pos_weight": pos_weight,
        "train_size": len(X_train_split),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }
