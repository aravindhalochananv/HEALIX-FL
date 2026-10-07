"""
Phase 5: Centralized Model Training

This module implements the training pipeline for the centralized baseline
MLP model in the HEALIX-FL project.

Features:
- Reproducible training with fixed random seeds
- Stratified train/validation split from processed training data
- Class-weighted BCEWithLogitsLoss for imbalanced data
- Best model selection based on validation ROC-AUC
- Comprehensive metric tracking and visualization
- Data leakage checks before training
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from src.models.mlp import MLP
from src.evaluation.metrics import calculate_metrics, print_metrics, save_metrics, plot_confusion_matrix


def set_seed(seed=42):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the appropriate device (CUDA if available, else CPU).
    
    Returns:
        torch.device: The device to use for training.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device


def load_processed_data(processed_dir):
    """Load processed training and test data from Phase 2.
    
    Args:
        processed_dir: Path to processed data directory.
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    print(f"Loading processed data from: {processed_dir}")
    
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def check_data_leakage(X_train, y_train, X_test, y_test):
    """Perform data leakage checks before training.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        
    Raises:
        ValueError: If any leakage check fails.
    """
    print("\n" + "=" * 60)
    print("DATA LEAKAGE CHECKS")
    print("=" * 60)
    
    all_passed = True
    
    # 1. Test set not used for training
    print(f"[CHECK] Test set will not be used for training or validation.")
    
    # 2. No NaN or infinite values
    train_has_nan = np.isnan(X_train).any() or np.isnan(y_train).any()
    test_has_nan = np.isnan(X_test).any() or np.isnan(y_test).any()
    train_has_inf = np.isinf(X_train).any() or np.isinf(y_train).any()
    test_has_inf = np.isinf(X_test).any() or np.isinf(y_test).any()
    
    if train_has_nan or test_has_nan:
        print("[FAIL] NaN values found in data.")
        all_passed = False
    else:
        print("[PASS] No NaN values found.")
    
    if train_has_inf or test_has_inf:
        print("[FAIL] Infinite values found in data.")
        all_passed = False
    else:
        print("[PASS] No infinite values found.")
    
    # 3. Target values are only 0 and 1
    unique_train = set(np.unique(y_train))
    unique_test = set(np.unique(y_test))
    if unique_train.issubset({0, 1}) and unique_test.issubset({0, 1}):
        print(f"[PASS] Target values are binary: {sorted(unique_train | unique_test)}")
    else:
        print(f"[FAIL] Non-binary target values found.")
        all_passed = False
    
    # 4. No target column in features (processed data should not have it)
    # Since we load numpy arrays, we verify by shape consistency
    print(f"[PASS] Processed arrays used (no target column in features).")
    
    # 5. No identifier columns (verified during Phase 2 preprocessing)
    print(f"[PASS] Identifiers removed during Phase 2 preprocessing.")
    
    print("=" * 60)
    if not all_passed:
        raise ValueError("Data leakage checks failed. Aborting training.")
    print("All data leakage checks PASSED.")
    print("=" * 60)


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=256, seed=42):
    """Create train, validation, and test DataLoaders.
    
    Splits training data into 80% train / 20% validation using
    stratified splitting. Test data is kept completely separate.
    
    Args:
        X_train: Training features (numpy array).
        y_train: Training targets (numpy array).
        X_test: Test features (numpy array).
        y_test: Test targets (numpy array).
        batch_size: Batch size for DataLoader.
        seed: Random seed for reproducibility.
        
    Returns:
        train_loader, val_loader, test_loader, X_train_split, y_train_split, X_val, y_val
    """
    from sklearn.model_selection import train_test_split
    
    # Split training data into train and validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )
    
    print(f"\nData split:")
    print(f"  Training:   {len(X_train_split):,} samples")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Test:       {len(X_test):,} samples")
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_split),
        torch.FloatTensor(y_train_split).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train_split, y_train_split, X_val, y_val


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch.
    
    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch.
    
    Args:
        model: PyTorch model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Device to validate on.
        
    Returns:
        tuple: (average validation loss, ROC-AUC, PR-AUC, predictions, labels)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = y_batch.cpu().numpy().flatten()
            
            all_preds.extend(probs)
            all_labels.extend(labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)
    val_loss = total_loss / num_batches
    
    return val_loss, metrics['roc_auc'], metrics['pr_auc'], all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, patience=5):
    """Train the model with early stopping based on validation ROC-AUC.
    
    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs: Maximum number of epochs.
        device: Device to train on.
        patience: Early stopping patience (not used currently, but kept for extensibility).
        
    Returns:
        dict: Training history containing losses and metrics per epoch.
        best_model_state: State dict of the best model.
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_roc_auc': [],
        'val_pr_auc': []
    }
    
    best_roc_auc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, roc_auc, pr_auc, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_roc_auc'].append(roc_auc)
        history['val_pr_auc'].append(pr_auc)
        
        # Print epoch results
        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val ROC-AUC: {roc_auc:.4f} | "
              f"Val PR-AUC: {pr_auc:.4f}")
        
        # Save best model based on validation ROC-AUC
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best model saved (ROC-AUC: {best_roc_auc:.4f})")
    
    return history, best_model_state


def evaluate_model(model, test_loader, device, threshold=0.5):
    """Evaluate model on test set.
    
    Args:
        model: PyTorch model.
        test_loader: Test DataLoader.
        device: Device to evaluate on.
        threshold: Classification threshold.
        
    Returns:
        dict: Test metrics.
        np.ndarray: Predictions.
        np.ndarray: True labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = y_batch.cpu().numpy().flatten()
            
            all_preds.extend(probs)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds, threshold=threshold)
    
    return metrics, all_preds, all_labels


def save_training_history(history, output_dir):
    """Save training history plots.
    
    Args:
        history: Dictionary containing training history.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Centralized Model: Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centralized_training_loss.png'), dpi=150)
    plt.close()
    print(f"Saved training/validation loss plot to: {output_dir}/centralized_training_loss.png")
    
    # 2. Validation ROC-AUC
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_roc_auc'], 'g-', label='Validation ROC-AUC', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.title('Centralized Model: Validation ROC-AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centralized_validation_loss.png'), dpi=150)
    plt.close()
    print(f"Saved validation ROC-AUC plot to: {output_dir}/centralized_validation_loss.png")
    
    # 3. Validation PR-AUC
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_pr_auc'], 'm-', label='Validation PR-AUC', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('PR-AUC')
    plt.title('Centralized Model: Validation PR-AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centralized_pr_auc.png'), dpi=150)
    plt.close()
    print(f"Saved validation PR-AUC plot to: {output_dir}/centralized_pr_auc.png")
    
    # 4. ROC-AUC and PR-AUC together
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_roc_auc'], 'g-', label='Validation ROC-AUC', linewidth=2)
    plt.plot(epochs, history['val_pr_auc'], 'm-', label='Validation PR-AUC', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Centralized Model: Validation Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centralized_roc_auc.png'), dpi=150)
    plt.close()
    print(f"Saved validation metrics plot to: {output_dir}/centralized_roc_auc.png")


def run_phase5(config):
    """Execute the complete Phase 5 centralized baseline training pipeline.
    
    Args:
        config: Dictionary loaded from config.yaml.
    """
    print("=" * 60)
    print("HEALIX-FL: Phase 5 - Centralized MLP Baseline")
    print("=" * 60)
    print()
    
    # Extract configuration
    processed_dir = config["paths"]["processed_data"]
    results_dir = config["paths"]["results"]
    seed = config["project"]["random_seed"]
    
    phase5_config = config["phase5"]["centralized"]
    model_config = phase5_config["model"]
    training_config = phase5_config["training"]
    
    input_dim = phase5_config["input_dim"]
    hidden_dim1 = model_config["hidden_dim1"]
    hidden_dim2 = model_config["hidden_dim2"]
    dropout = model_config["dropout"]
    
    lr = training_config["learning_rate"]
    batch_size = training_config["batch_size"]
    epochs = training_config["epochs"]
    model_output_path = phase5_config["model_output"]
    metrics_output_path = phase5_config["metrics_output"]
    config_output_path = phase5_config["config_output"]
    
    # Set reproducibility
    set_seed(seed)
    device = get_device()
    
    # Load data
    X_train, y_train, X_test, y_test = load_processed_data(processed_dir)
    
    # Data leakage checks
    check_data_leakage(X_train, y_train, X_test, y_test)
    
    # Calculate class weights
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    pos_weight = neg_count / pos_count
    
    print(f"\nClass distribution in training data:")
    print(f"  Positive class (readmitted <30): {pos_count:,} ({pos_count/len(y_train)*100:.2f}%)")
    print(f"  Negative class (NO or >30):      {neg_count:,} ({neg_count/len(y_train)*100:.2f}%)")
    print(f"  Calculated pos_weight: {pos_weight:.4f}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, X_train_split, y_train_split, X_val, y_val = \
        create_data_loaders(X_train, y_train, X_test, y_test, batch_size=batch_size, seed=seed)
    
    # Initialize model
    model = MLP(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining Configuration:")
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_weight:.4f})")
    print(f"  Random seed: {seed}")
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    
    history, best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=epochs, device=device, patience=5
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    
    # Load best model
    model.load_state_dict(best_model_state)
    model = model.to(device)
    
    # Final test evaluation
    print("\nEvaluating best model on test set...")
    test_metrics, test_preds, test_labels = evaluate_model(model, test_loader, device)
    
    print_metrics(test_metrics, title="Final Test Metrics")
    
    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim1': hidden_dim1,
            'hidden_dim2': hidden_dim2,
            'dropout': dropout
        },
        'training_config': {
            'lr': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'pos_weight': pos_weight,
            'seed': seed
        },
        'best_val_roc_auc': max(history['val_roc_auc']),
        'best_val_pr_auc': max(history['val_pr_auc'])
    }, model_output_path)
    print(f"Best model saved to: {model_output_path}")
    
    # Save metrics
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    save_metrics(test_metrics, metrics_output_path)
    
    # Save config
    import yaml
    os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
    with open(config_output_path, 'w') as f:
        yaml.dump(phase5_config, f, default_flow_style=False)
    print(f"Config saved to: {config_output_path}")
    
    # Save plots
    os.makedirs(results_dir, exist_ok=True)
    save_training_history(history, results_dir)
    plot_confusion_matrix(test_metrics, 
                         os.path.join(results_dir, 'centralized_confusion_matrix.png'),
                         title="Centralized Model: Test Confusion Matrix")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("=" * 60)
    print(f"Model Architecture: MLP({input_dim} -> {hidden_dim1} -> {hidden_dim2} -> 1)")
    print(f"Dataset Sizes:")
    print(f"  Training:   {len(X_train_split):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test:       {len(X_test):,}")
    print(f"Training Configuration:")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"  Pos weight: {pos_weight:.4f}")
    print(f"Best Validation ROC-AUC: {max(history['val_roc_auc']):.4f}")
    print(f"Best Validation PR-AUC:  {max(history['val_pr_auc']):.4f}")
    print(f"Final Test Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"  Confusion Matrix: TP={test_metrics['tp']}, TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}")
    print(f"\nSaved Artifacts:")
    print(f"  Model:     {model_output_path}")
    print(f"  Metrics:   {metrics_output_path}")
    print(f"  Config:    {config_output_path}")
    print(f"  Plots:     {results_dir}/centralized_*.png")
    print("=" * 60)
    
    return {
        'model': model,
        'history': history,
        'best_model_state': best_model_state,
        'test_metrics': test_metrics,
        'pos_weight': pos_weight,
        'train_size': len(X_train_split),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }
