"""
Phase 4: Heterogeneity Metrics Module

This module implements metrics to quantify distribution differences
between federated clients in the HEALIX-FL project.

Metrics include:
1. Jensen-Shannon divergence for label distributions
2. Feature distribution distance (mean feature vector distance)
3. Client-to-client similarity matrix
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy


def calculate_label_divergence(client_y, global_y, eps=1e-10):
    """Calculate Jensen-Shannon divergence between client and global label distributions.
    
    Jensen-Shannon divergence is a symmetric and smoothed version of
    Kullback-Leibler divergence. It measures the similarity between two
    probability distributions.
    
    Args:
        client_y: numpy.ndarray of client target values (0s and 1s).
        global_y: numpy.ndarray of global target values (0s and 1s).
        eps: Small constant to avoid log(0).
        
    Returns:
        float: Jensen-Shannon divergence value.
    """
    # Compute label probabilities for binary case
    client_counts = np.bincount(client_y.astype(int), minlength=2)
    global_counts = np.bincount(global_y.astype(int), minlength=2)
    
    client_p = client_counts / client_counts.sum()
    global_p = global_counts / global_counts.sum()
    
    # Add epsilon to avoid zeros in log
    client_p = np.clip(client_p, eps, 1)
    global_p = np.clip(global_p, eps, 1)
    
    # Jensen-Shannon divergence
    m = 0.5 * (client_p + global_p)
    jsd = 0.5 * (entropy(client_p, m) + entropy(global_p, m))
    
    return float(jsd)


def calculate_feature_distance(client_X, global_X):
    """Calculate feature distribution distance between client and global data.
    
    Uses the Euclidean distance between mean feature vectors as a proxy
    for distribution distance. This is efficient for high-dimensional data
    and captures meaningful differences in feature distributions.
    
    Args:
        client_X: numpy.ndarray of client features.
        global_X: numpy.ndarray of global features.
        
    Returns:
        float: Feature distribution distance (Euclidean norm of mean difference).
    """
    client_mean = np.mean(client_X, axis=0)
    global_mean = np.mean(global_X, axis=0)
    
    # Euclidean distance between mean vectors
    distance = np.linalg.norm(client_mean - global_mean)
    
    return float(distance)


def calculate_client_similarity(clients, metric="euclidean"):
    """Calculate pairwise client similarity/distance matrix.
    
    Computes distances between client mean feature vectors.
    
    Args:
        clients: dict mapping client_id -> {"X": ndarray, ...}.
        metric: Distance metric to use (currently supports euclidean).
        
    Returns:
        similarity_matrix: 2D numpy array of pairwise distances.
        client_ids: List of client IDs in sorted order.
    """
    client_ids = sorted(clients.keys())
    n_clients = len(client_ids)
    
    # Compute mean feature vector for each client
    mean_vectors = []
    for cid in client_ids:
        mean_vec = np.mean(clients[cid]["X"], axis=0)
        mean_vectors.append(mean_vec)
    
    mean_vectors = np.array(mean_vectors)
    
    # Compute pairwise distance matrix
    similarity_matrix = np.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(n_clients):
            if metric == "euclidean":
                similarity_matrix[i, j] = np.linalg.norm(mean_vectors[i] - mean_vectors[j])
            else:
                similarity_matrix[i, j] = np.linalg.norm(mean_vectors[i] - mean_vectors[j])
    
    return similarity_matrix, client_ids


def generate_heterogeneity_report(clients, X_global, y_global, heterogeneity_level, output_dir):
    """Generate a comprehensive heterogeneity report for a set of clients.
    
    Args:
        clients: dict mapping client_id -> {"X": ndarray, "y": ndarray, ...}.
        X_global: numpy.ndarray of global features.
        y_global: numpy.ndarray of global targets.
        heterogeneity_level: String identifier for the heterogeneity level.
        output_dir: Directory to save reports.
        
    Returns:
        stats_df: pandas.DataFrame with client statistics including heterogeneity metrics.
        summary_dict: Dictionary with summary heterogeneity metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    client_ids = sorted(clients.keys())
    
    records = []
    label_divergences = []
    feature_distances = []
    
    for cid in client_ids:
        client_data = clients[cid]
        Xc = client_data["X"]
        yc = client_data["y"]
        
        num_samples = len(yc)
        positive_samples = int(np.sum(yc == 1))
        negative_samples = int(np.sum(yc == 0))
        positive_rate = positive_samples / num_samples * 100 if num_samples > 0 else 0.0
        negative_rate = negative_samples / num_samples * 100 if num_samples > 0 else 0.0
        
        # Heterogeneity metrics
        label_div = calculate_label_divergence(yc, y_global)
        feature_dist = calculate_feature_distance(Xc, X_global)
        
        label_divergences.append(label_div)
        feature_distances.append(feature_dist)
        
        records.append({
            "client_id": cid,
            "heterogeneity_level": heterogeneity_level,
            "num_samples": num_samples,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "positive_rate": round(positive_rate, 2),
            "negative_rate": round(negative_rate, 2),
            "label_divergence": round(label_div, 6),
            "feature_distance": round(feature_dist, 6)
        })
    
    stats_df = pd.DataFrame(records)
    
    # Summary statistics across all clients
    summary_dict = {
        "heterogeneity_level": heterogeneity_level,
        "average_label_divergence": round(float(np.mean(label_divergences)), 6),
        "average_feature_distance": round(float(np.mean(feature_distances)), 6),
        "minimum_client_positive_rate": round(float(stats_df["positive_rate"].min()), 2),
        "maximum_client_positive_rate": round(float(stats_df["positive_rate"].max()), 2),
        "mean_client_samples": round(float(stats_df["num_samples"].mean()), 2),
        "std_client_samples": round(float(stats_df["num_samples"].std()), 2)
    }
    
    return stats_df, summary_dict


def plot_client_label_divergence(stats_df, output_path, heterogeneity_level):
    """Create a bar chart of label divergence for each client."""
    plt.figure(figsize=(10, 6))
    plt.bar(stats_df["client_id"], stats_df["label_divergence"], color="coral")
    plt.xlabel("Client ID")
    plt.ylabel("Label Divergence (JSD)")
    plt.title(f"Client Label Divergence - {heterogeneity_level}")
    plt.axhline(y=stats_df["label_divergence"].mean(), color="red", linestyle="--",
                label=f"Mean: {stats_df['label_divergence'].mean():.4f}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved label divergence plot to: {output_path}")


def plot_client_feature_distance(stats_df, output_path, heterogeneity_level):
    """Create a bar chart of feature distance for each client."""
    plt.figure(figsize=(10, 6))
    plt.bar(stats_df["client_id"], stats_df["feature_distance"], color="mediumseagreen")
    plt.xlabel("Client ID")
    plt.ylabel("Feature Distance (Euclidean)")
    plt.title(f"Client Feature Distance - {heterogeneity_level}")
    plt.axhline(y=stats_df["feature_distance"].mean(), color="red", linestyle="--",
                label=f"Mean: {stats_df['feature_distance'].mean():.4f}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved feature distance plot to: {output_path}")


def plot_client_similarity_heatmap(similarity_matrix, client_ids, output_path, heterogeneity_level):
    """Create a heatmap of client-to-client similarity/distance."""
    plt.figure(figsize=(10, 8))
    im = plt.imshow(similarity_matrix, cmap="viridis")
    plt.colorbar(im, label="Euclidean Distance")
    plt.xticks(range(len(client_ids)), client_ids, rotation=45)
    plt.yticks(range(len(client_ids)), client_ids)
    plt.title(f"Client Similarity Matrix - {heterogeneity_level}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved similarity heatmap to: {output_path}")
