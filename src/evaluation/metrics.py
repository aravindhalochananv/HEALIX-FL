"""
Phase 5: Evaluation Metrics Module

This module provides reusable evaluation functions for binary classification
in the HEALIX-FL centralized baseline experiment.

Metrics include:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion matrix
- TP, TN, FP, FN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate all binary classification metrics.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_pred_proba: Predicted probabilities (sigmoid output).
        threshold: Classification threshold (default: 0.5).
        
    Returns:
        dict: Dictionary containing all metrics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC metrics
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }
    
    return metrics


def print_metrics(metrics, title="Test Metrics"):
    """Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics().
        title: Title for the metrics section.
    """
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1 Score:      {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:        {metrics['pr_auc']:.4f}")
    print(f"Confusion Matrix:")
    print(f"  TP: {metrics['tp']}  FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}  TN: {metrics['tn']}")
    print("=" * 60)


def save_metrics(metrics, output_path):
    """Save metrics to a CSV file.
    
    Args:
        metrics: Dictionary of metrics.
        output_path: Path to save the CSV file.
    """
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to: {output_path}")


def plot_confusion_matrix(metrics, output_path, title="Test Confusion Matrix"):
    """Plot and save confusion matrix.
    
    Args:
        metrics: Dictionary containing tp, tn, fp, fn.
        output_path: Path to save the PNG file.
        title: Title for the plot.
    """
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")
