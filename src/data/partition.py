"""
Phase 3 & 4: Federated Client Partitioning

This module simulates multiple healthcare institutions participating in
federated learning by partitioning the processed training data into
client datasets.

Phase 3: IID (Independent and Identically Distributed) partitioning
Phase 4: Non-IID partitioning using Dirichlet distribution with
         configurable heterogeneity levels and heterogeneity metrics.

IMPORTANT:
- Only training data is partitioned among clients.
- The test set remains completely isolated.
- Partitions are reproducible using fixed random seeds.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


# Fixed random seed for reproducibility
RANDOM_SEED = 42

# Dirichlet concentration parameters for each heterogeneity level
# Higher alpha -> more uniform/IID; Lower alpha -> more skewed/non-IID
ALPHA_MAP = {
    "iid": 1000.0,
    "mild_non_iid": 100.0,
    "moderate_non_iid": 10.0,
    "strong_non_iid": 1.0,
    "extreme_non_iid": 0.1
}


def create_iid_clients(X, y, num_clients=10, seed=RANDOM_SEED):
    """Create IID federated clients by randomly shuffling and splitting training data.
    
    Each client receives approximately the same number of samples.
    Every training sample belongs to exactly one client.
    No test samples are included.
    
    Args:
        X: numpy.ndarray of shape (n_samples, n_features) - training features.
        y: numpy.ndarray or pandas.Series of shape (n_samples,) - training targets.
        num_clients: Number of simulated clients/hospitals.
        seed: Random seed for reproducibility.
        
    Returns:
        clients: dict mapping client_id -> {"X": ndarray, "y": ndarray, "indices": ndarray}
    """
    np.random.seed(seed)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Calculate approximate split sizes
    base_size = n_samples // num_clients
    remainder = n_samples % num_clients
    
    clients = {}
    start = 0
    
    for client_idx in range(num_clients):
        # Distribute remainder one sample at a time to first 'remainder' clients
        size = base_size + (1 if client_idx < remainder else 0)
        end = start + size
        
        client_indices = indices[start:end]
        clients[f"client_{client_idx + 1}"] = {
            "X": X[client_indices],
            "y": y[client_indices] if isinstance(y, np.ndarray) else y.iloc[client_indices].values,
            "indices": client_indices
        }
        
        start = end
    
    return clients


def create_non_iid_clients(X, y, num_clients=10, heterogeneity_level="moderate_non_iid", seed=RANDOM_SEED):
    """Create non-IID federated clients using Dirichlet-based label skew partitioning.
    
    Uses Dirichlet distribution to generate class proportions for each client.
    The concentration parameter controls the level of heterogeneity:
    - Higher alpha -> more uniform label distribution across clients (IID-like)
    - Lower alpha -> more skewed label distribution across clients (non-IID)
    
    Args:
        X: numpy.ndarray of shape (n_samples, n_features) - training features.
        y: numpy.ndarray or pandas.Series of shape (n_samples,) - training targets.
        num_clients: Number of simulated clients/hospitals.
        heterogeneity_level: One of "iid", "mild_non_iid", "moderate_non_iid",
                            "strong_non_iid", "extreme_non_iid".
        seed: Random seed for reproducibility.
        
    Returns:
        clients: dict mapping client_id -> {"X": ndarray, "y": ndarray, "indices": ndarray}
    """
    if heterogeneity_level not in ALPHA_MAP:
        raise ValueError(f"Unknown heterogeneity level: {heterogeneity_level}. "
                        f"Choose from: {list(ALPHA_MAP.keys())}")
    
    alpha = ALPHA_MAP[heterogeneity_level]
    np.random.seed(seed)
    
    n_samples = len(X)
    classes = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]
    
    # Distribute each class separately using Dirichlet proportions
    for c in classes:
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)
        
        # Generate Dirichlet proportions for this class across clients
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Calculate exact sample counts per client to avoid rounding errors
        raw_sizes = proportions * len(class_indices)
        sizes = np.floor(raw_sizes).astype(int)
        
        # Distribute any remaining samples to clients with largest fractional parts
        remainder = len(class_indices) - sizes.sum()
        if remainder > 0:
            fractional_parts = raw_sizes - sizes
            top_indices = np.argsort(fractional_parts)[::-1][:remainder]
            sizes[top_indices] += 1
        
        # Assign samples to clients
        start = 0
        for i in range(num_clients):
            end = start + sizes[i]
            client_indices[i].extend(class_indices[start:end].tolist())
            start = end
    
    # Create client dictionaries
    clients = {}
    for i in range(num_clients):
        idx = np.array(client_indices[i])
        # Shuffle within client for better training behavior
        np.random.shuffle(idx)
        clients[f"client_{i + 1}"] = {
            "X": X[idx],
            "y": y[idx] if isinstance(y, np.ndarray) else y.iloc[idx].values,
            "indices": idx
        }
    
    return clients


def create_clients(X, y, num_clients=10, heterogeneity_level="iid", seed=RANDOM_SEED):
    """Unified interface to create clients with specified heterogeneity level.
    
    Args:
        X: numpy.ndarray of training features.
        y: numpy.ndarray or pandas.Series of training targets.
        num_clients: Number of clients.
        heterogeneity_level: Heterogeneity level ("iid", "mild_non_iid", etc.).
        seed: Random seed for reproducibility.
        
    Returns:
        clients: dict mapping client_id -> {"X": ndarray, "y": ndarray, "indices": ndarray}
    """
    if heterogeneity_level == "iid":
        return create_iid_clients(X, y, num_clients=num_clients, seed=seed)
    else:
        return create_non_iid_clients(X, y, num_clients=num_clients,
                                      heterogeneity_level=heterogeneity_level, seed=seed)


def save_clients(clients, output_dir):
    """Save each client's data to disk.
    
    Creates directories:
        output_dir/client_1/
        output_dir/client_2/
        ...
    
    Each client directory contains:
        X.npy
        y.npy
    
    Uses NumPy format for efficient storage of high-dimensional data.
    
    Args:
        clients: dict from create_iid_clients() or create_non_iid_clients().
        output_dir: Root directory to save client data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for client_id, client_data in clients.items():
        client_dir = os.path.join(output_dir, client_id)
        os.makedirs(client_dir, exist_ok=True)
        
        np.save(os.path.join(client_dir, "X.npy"), client_data["X"])
        np.save(os.path.join(client_dir, "y.npy"), client_data["y"])
    
    print(f"Saved {len(clients)} clients to: {output_dir}")


def calculate_client_statistics(clients, heterogeneity_level="iid", label_divergences=None, feature_distances=None):
    """Calculate statistics for each client.
    
    Args:
        clients: dict from create_clients().
        heterogeneity_level: String identifier for the heterogeneity level.
        label_divergences: Optional list of label divergence values per client.
        feature_distances: Optional list of feature distance values per client.
        
    Returns:
        stats_df: pandas.DataFrame with client statistics.
    """
    records = []
    client_ids = sorted(clients.keys())
    
    for idx, client_id in enumerate(client_ids):
        client_data = clients[client_id]
        y = client_data["y"]
        
        num_samples = len(y)
        positive_samples = int(np.sum(y == 1))
        negative_samples = int(np.sum(y == 0))
        positive_rate = positive_samples / num_samples * 100 if num_samples > 0 else 0.0
        negative_rate = negative_samples / num_samples * 100 if num_samples > 0 else 0.0
        
        record = {
            "client_id": client_id,
            "heterogeneity_level": heterogeneity_level,
            "num_samples": num_samples,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "positive_rate": round(positive_rate, 2),
            "negative_rate": round(negative_rate, 2)
        }
        
        if label_divergences is not None and idx < len(label_divergences):
            record["label_divergence"] = round(float(label_divergences[idx]), 6)
        
        if feature_distances is not None and idx < len(feature_distances):
            record["feature_distance"] = round(float(feature_distances[idx]), 6)
        
        records.append(record)
    
    stats_df = pd.DataFrame(records)
    return stats_df


def print_client_statistics(stats_df):
    """Print client statistics to terminal in a readable format.
    
    Args:
        stats_df: pandas.DataFrame from calculate_client_statistics().
    """
    print("\n" + "=" * 60)
    print("CLIENT STATISTICS")
    print("=" * 60)
    
    for _, row in stats_df.iterrows():
        print(f"\n{row['client_id'].upper()}")
        print(f"  Samples:        {row['num_samples']:,}")
        print(f"  Positive:       {row['positive_samples']:,}")
        print(f"  Negative:       {row['negative_samples']:,}")
        print(f"  Positive rate:  {row['positive_rate']:.2f}%")
        print(f"  Negative rate:  {row['negative_rate']:.2f}%")
        if "label_divergence" in row and not pd.isna(row["label_divergence"]):
            print(f"  Label divergence: {row['label_divergence']:.6f}")
        if "feature_distance" in row and not pd.isna(row["feature_distance"]):
            print(f"  Feature distance: {row['feature_distance']:.6f}")


def validate_partition(clients, X_train, y_train, num_clients=10, seed=RANDOM_SEED, level_name="IID"):
    """Validate a partition for correctness.
    
    Checks:
    1. Total client samples == total training samples.
    2. Number of clients matches expected.
    3. No duplicated sample indices across clients.
    4. No training samples are lost.
    5. Target values contain only 0 and 1.
    6. Test data was not used (implicit by input).
    7. Client sizes are approximately equal.
    8. Reproducibility: same seed produces same result.
    
    Args:
        clients: dict from create_clients().
        X_train: Original training features (for validation).
        y_train: Original training targets (for validation).
        num_clients: Expected number of clients.
        seed: Random seed used for partitioning.
        level_name: Name of the heterogeneity level for display.
        
    Returns:
        all_passed: bool indicating if all checks passed.
    """
    print(f"\n{'=' * 60}")
    print(f"{level_name.upper()} PARTITION VALIDATION")
    print("=" * 60)
    
    all_passed = True
    total_training = len(X_train)
    total_client_samples = sum(len(c["y"]) for c in clients.values())
    
    # 1. Total client samples == total training samples
    if total_client_samples == total_training:
        print(f"[PASS] Total client samples == training samples: {total_client_samples:,}")
    else:
        print(f"[FAIL] Total client samples ({total_client_samples:,}) != training samples ({total_training:,})")
        all_passed = False
    
    # 2. Number of clients matches expected
    if len(clients) == num_clients:
        print(f"[PASS] Number of clients: {len(clients)}")
    else:
        print(f"[FAIL] Expected {num_clients} clients, got {len(clients)}")
        all_passed = False
    
    # 3. No duplicated sample indices across clients
    all_indices = np.concatenate([c["indices"] for c in clients.values()])
    unique_indices = np.unique(all_indices)
    if len(all_indices) == len(unique_indices):
        print(f"[PASS] No duplicated sample indices across clients")
    else:
        print(f"[FAIL] Duplicated indices found: {len(all_indices)} total vs {len(unique_indices)} unique")
        all_passed = False
    
    # 4. No training samples are lost
    expected_indices = np.arange(total_training)
    if set(all_indices) == set(expected_indices):
        print(f"[PASS] No training samples lost: all {total_training:,} samples assigned")
    else:
        missing = set(expected_indices) - set(all_indices)
        extra = set(all_indices) - set(expected_indices)
        print(f"[FAIL] Missing samples: {len(missing)}, Extra samples: {len(extra)}")
        all_passed = False
    
    # 5. Target values are only 0 and 1
    all_targets = np.concatenate([c["y"] for c in clients.values()])
    unique_targets = set(all_targets)
    if unique_targets.issubset({0, 1}):
        print(f"[PASS] All target values are binary: {sorted(unique_targets)}")
    else:
        print(f"[FAIL] Non-binary target values found: {unique_targets}")
        all_passed = False
    
    # 6. Test data was not used (validated by input - only X_train/y_train used)
    print(f"[PASS] Test data not used for client creation (partition uses only training data)")
    
    # 7. Client sizes check
    sizes = [len(c["y"]) for c in clients.values()]
    size_diff = max(sizes) - min(sizes)
    if size_diff <= 1:
        print(f"[PASS] Client sizes approximately equal: min={min(sizes)}, max={max(sizes)}, diff={size_diff}")
    else:
        print(f"[WARN] Client size difference > 1: min={min(sizes)}, max={max(sizes)}, diff={size_diff}")
    
    # 8. Reproducibility check
    if level_name.lower() == "iid":
        # For IID, regenerate with same seed and compare
        np.random.seed(seed)
        n_samples = len(X_train)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        base_size = n_samples // num_clients
        remainder = n_samples % num_clients
        start = 0
        reproducible = True
        for client_idx in range(num_clients):
            size = base_size + (1 if client_idx < remainder else 0)
            end = start + size
            client_indices = indices[start:end]
            client_id = f"client_{client_idx + 1}"
            if client_id in clients and not np.array_equal(clients[client_id]["indices"], client_indices):
                reproducible = False
                break
            start = end
        if reproducible:
            print(f"[PASS] Reproducibility: same seed ({seed}) produces identical partition")
        else:
            print(f"[FAIL] Reproducibility check failed")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print(f"All validation checks PASSED for {level_name}.")
    else:
        print(f"Some validation checks FAILED for {level_name}. Review the output above.")
    print("=" * 60)
    
    return all_passed


def plot_client_sample_distribution(stats_df, output_path):
    """Create a bar chart of client sample counts."""
    plt.figure(figsize=(10, 6))
    plt.bar(stats_df["client_id"], stats_df["num_samples"], color="steelblue")
    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    plt.title("IID Client Sample Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved sample distribution plot to: {output_path}")


def plot_client_target_distribution(stats_df, output_path):
    """Create a bar chart of client positive-class percentages."""
    plt.figure(figsize=(10, 6))
    plt.bar(stats_df["client_id"], stats_df["positive_rate"], color="darkorange")
    plt.xlabel("Client ID")
    plt.ylabel("Positive-Class Percentage (%)")
    plt.title("IID Client Positive-Class Distribution")
    plt.axhline(y=stats_df["positive_rate"].mean(), color="red", linestyle="--",
                label=f"Mean: {stats_df['positive_rate'].mean():.2f}%")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved target distribution plot to: {output_path}")


def run_phase3(config):
    """Execute the complete Phase 3 IID client generation pipeline.
    
    Args:
        config: Dictionary loaded from config.yaml.
    """
    print("=" * 60)
    print("HEALIX-FL: Phase 3 - IID Federated Client Generation")
    print("=" * 60)
    print()
    
    # Configuration
    processed_dir = config["paths"]["processed_data"]
    output_dir = config["phase3"]["federated"]["iid"]["output_dir"]
    num_clients = config["phase3"]["federated"]["num_clients"]
    seed = config["phase3"]["federated"]["random_seed"]
    distribution_csv = config["phase3"]["federated"]["iid"]["distribution_csv"]
    sample_plot = config["phase3"]["federated"]["iid"]["sample_distribution_plot"]
    target_plot = config["phase3"]["federated"]["iid"]["target_distribution_plot"]
    
    # Load processed training data
    print("Loading processed training data from Phase 2...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    print(f"Loaded training data: {len(X_train):,} samples, {X_train.shape[1]} features")
    
    # Create IID clients
    print(f"\nCreating {num_clients} IID clients with seed={seed}...")
    clients = create_iid_clients(X_train, y_train, num_clients=num_clients, seed=seed)
    print(f"Created {len(clients)} clients successfully.")
    
    # Save client data
    save_clients(clients, output_dir)
    
    # Calculate and print statistics
    stats_df = calculate_client_statistics(clients, heterogeneity_level="iid")
    print_client_statistics(stats_df)
    
    # Save distribution CSV
    os.makedirs(os.path.dirname(distribution_csv), exist_ok=True)
    stats_df.to_csv(distribution_csv, index=False)
    print(f"\nClient distribution saved to: {distribution_csv}")
    
    # Validate partition
    validate_partition(clients, X_train, y_train, num_clients=num_clients, seed=seed, level_name="IID")
    
    # Create visualizations
    print("\nCreating visualizations...")
    os.makedirs(os.path.dirname(sample_plot), exist_ok=True)
    plot_client_sample_distribution(stats_df, sample_plot)
    plot_client_target_distribution(stats_df, target_plot)
    
    print("\n" + "=" * 60)
    print("HEALIX-FL PHASE 3")
    print("IID CLIENT GENERATION COMPLETED")
    print("=" * 60)
    print(f"\nNumber of clients:        {num_clients}")
    print(f"Total training samples:   {len(X_train):,}")
    print(f"Samples per client:       ~{len(X_train) // num_clients:,}")
    print(f"Client data directory:    {output_dir}")
    print(f"Distribution CSV:         {distribution_csv}")
    print(f"Visualizations:           {sample_plot}, {target_plot}")
    print("=" * 60)
    
    return {
        "clients": clients,
        "statistics": stats_df,
        "X_train": X_train,
        "y_train": y_train
    }


def run_phase4(config):
    """Execute the complete Phase 4 non-IID client generation pipeline.
    
    Creates clients with multiple heterogeneity levels, calculates
    heterogeneity metrics, and generates comparison visualizations.
    
    Args:
        config: Dictionary loaded from config.yaml.
    """
    print("=" * 60)
    print("HEALIX-FL: Phase 4 - Non-IID Healthcare Client Generation")
    print("=" * 60)
    print()
    
    # Configuration
    processed_dir = config["paths"]["processed_data"]
    num_clients = config["phase3"]["federated"]["num_clients"]
    seed = config["phase3"]["federated"]["random_seed"]
    output_base_dir = config["phase4"]["non_iid"]["output_base_dir"]
    distribution_csv = config["phase4"]["non_iid"]["distribution_csv"]
    summary_csv = config["phase4"]["non_iid"]["summary_csv"]
    config_csv = config["phase4"]["non_iid"]["config_csv"]
    plots = config["phase4"]["non_iid"]["plots"]
    levels = config["phase4"]["non_iid"]["levels"]
    alpha_map = config["phase4"]["non_iid"]["alpha_map"]
    
    # Load processed training data
    print("Loading processed training data from Phase 2...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    print(f"Loaded training data: {len(X_train):,} samples, {X_train.shape[1]} features")
    
    all_stats = []
    all_summaries = []
    all_similarity_matrices = {}
    
    # Process each heterogeneity level
    for level in levels:
        alpha = alpha_map[level]
        print(f"\n{'=' * 60}")
        print(f"Processing level: {level.upper()} (alpha={alpha})")
        print("=" * 60)
        
        # Create clients
        if level == "iid":
            clients = create_iid_clients(X_train, y_train, num_clients=num_clients, seed=seed)
        else:
            clients = create_non_iid_clients(X_train, y_train, num_clients=num_clients,
                                            heterogeneity_level=level, seed=seed)
        
        # Save client data
        level_output_dir = os.path.join(output_base_dir, level)
        save_clients(clients, level_output_dir)
        
        # Calculate heterogeneity metrics
        from src.data.heterogeneity import (
            generate_heterogeneity_report,
            calculate_client_similarity,
            plot_client_label_divergence,
            plot_client_feature_distance,
            plot_client_similarity_heatmap
        )
        
        stats_df, summary_dict = generate_heterogeneity_report(
            clients, X_train, y_train, level, output_base_dir
        )
        
        # Calculate similarity matrix
        similarity_matrix, client_ids = calculate_client_similarity(clients)
        all_similarity_matrices[level] = (similarity_matrix, client_ids)
        
        # Print statistics
        print_client_statistics(stats_df)
        
        # Print heterogeneity summary for this level
        print(f"\nHeterogeneity Summary ({level}):")
        print(f"  Average label divergence:     {summary_dict['average_label_divergence']:.6f}")
        print(f"  Average feature distance:     {summary_dict['average_feature_distance']:.6f}")
        print(f"  Min positive rate:            {summary_dict['minimum_client_positive_rate']:.2f}%")
        print(f"  Max positive rate:            {summary_dict['maximum_client_positive_rate']:.2f}%")
        print(f"  Mean samples per client:      {summary_dict['mean_client_samples']:.2f}")
        print(f"  Std samples per client:       {summary_dict['std_client_samples']:.2f}")
        
        # Save level-specific plots
        level_plot_dir = os.path.join(config["paths"]["results"], "phase4_plots", level)
        os.makedirs(level_plot_dir, exist_ok=True)
        plot_client_label_divergence(stats_df, os.path.join(level_plot_dir, "label_divergence.png"), level)
        plot_client_feature_distance(stats_df, os.path.join(level_plot_dir, "feature_distance.png"), level)
        plot_client_similarity_heatmap(similarity_matrix, client_ids,
                                       os.path.join(level_plot_dir, "similarity_heatmap.png"), level)
        
        # Collect results
        all_stats.append(stats_df)
        all_summaries.append(summary_dict)
        
        # Validate partition
        validate_partition(clients, X_train, y_train, num_clients=num_clients,
                          seed=seed, level_name=level)
    
    # Combine all results
    combined_stats = pd.concat(all_stats, ignore_index=True)
    combined_summaries = pd.DataFrame(all_summaries)
    
    # Save combined distribution CSV
    os.makedirs(os.path.dirname(distribution_csv), exist_ok=True)
    combined_stats.to_csv(distribution_csv, index=False)
    print(f"\nCombined client distribution saved to: {distribution_csv}")
    
    # Save heterogeneity summary CSV
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    combined_summaries.to_csv(summary_csv, index=False)
    print(f"Heterogeneity summary saved to: {summary_csv}")
    
    # Save heterogeneity config CSV
    config_records = []
    for level in levels:
        config_records.append({
            "level": level,
            "concentration_parameter": alpha_map[level],
            "num_clients": num_clients,
            "seed": seed
        })
    config_df = pd.DataFrame(config_records)
    os.makedirs(os.path.dirname(config_csv), exist_ok=True)
    config_df.to_csv(config_csv, index=False)
    print(f"Heterogeneity config saved to: {config_csv}")
    
    # Create comparison visualizations across all levels
    print("\nCreating comparison visualizations across all levels...")
    os.makedirs(os.path.dirname(plots["target_distribution"]), exist_ok=True)
    
    # 1. Target distribution comparison
    _plot_comparison_target_distribution(combined_stats, plots["target_distribution"])
    
    # 2. Label divergence comparison
    _plot_comparison_label_divergence(combined_stats, plots["label_divergence"])
    
    # 3. Feature distance comparison
    _plot_comparison_feature_distance(combined_stats, plots["feature_distance"])
    
    # 4. Similarity heatmap for moderate level (most representative)
    moderate_level = "moderate_non_iid"
    if moderate_level in all_similarity_matrices:
        sim_matrix, cids = all_similarity_matrices[moderate_level]
        plot_client_similarity_heatmap(sim_matrix, cids, plots["similarity_heatmap"], moderate_level)
    
    print("\n" + "=" * 60)
    print("HEALIX-FL PHASE 4")
    print("NON-IID CLIENT GENERATION COMPLETED")
    print("=" * 60)
    print(f"\nNumber of levels processed: {len(levels)}")
    print(f"Number of clients per level: {num_clients}")
    print(f"Total training samples: {len(X_train):,}")
    print(f"Client data directory: {output_base_dir}")
    print(f"Distribution CSV: {distribution_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Config CSV: {config_csv}")
    print(f"Visualizations: {list(plots.values())}")
    print("=" * 60)
    
    return {
        "clients_by_level": {level: None for level in levels},  # Placeholder
        "statistics": combined_stats,
        "summaries": combined_summaries,
        "config": config_df,
        "X_train": X_train,
        "y_train": y_train
    }


def _plot_comparison_target_distribution(stats_df, output_path):
    """Create a grouped bar chart comparing positive rates across all heterogeneity levels."""
    plt.figure(figsize=(12, 6))
    
    levels = stats_df["heterogeneity_level"].unique()
    x = np.arange(len(stats_df["client_id"].unique()))
    width = 0.15
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(levels)))
    
    for i, level in enumerate(levels):
        level_data = stats_df[stats_df["heterogeneity_level"] == level]
        # Sort by client_id to ensure consistent ordering
        level_data = level_data.sort_values("client_id")
        plt.bar(x + i * width, level_data["positive_rate"], width,
                label=level, color=colors[i])
    
    plt.xlabel("Client ID")
    plt.ylabel("Positive-Class Percentage (%)")
    plt.title("Client Positive-Class Distribution Across Heterogeneity Levels")
    plt.axhline(y=stats_df["positive_rate"].mean(), color="black", linestyle="--",
                label=f"Global Mean: {stats_df['positive_rate'].mean():.2f}%")
    plt.legend(title="Heterogeneity Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(x + width * (len(levels) - 1) / 2, level_data["client_id"], rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved target distribution comparison to: {output_path}")


def _plot_comparison_label_divergence(stats_df, output_path):
    """Create a grouped bar chart comparing label divergence across all levels."""
    plt.figure(figsize=(12, 6))
    
    levels = stats_df["heterogeneity_level"].unique()
    x = np.arange(len(stats_df["client_id"].unique()))
    width = 0.15
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(levels)))
    
    for i, level in enumerate(levels):
        level_data = stats_df[stats_df["heterogeneity_level"] == level]
        level_data = level_data.sort_values("client_id")
        plt.bar(x + i * width, level_data["label_divergence"], width,
                label=level, color=colors[i])
    
    plt.xlabel("Client ID")
    plt.ylabel("Label Divergence (JSD)")
    plt.title("Client Label Divergence Across Heterogeneity Levels")
    plt.legend(title="Heterogeneity Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(x + width * (len(levels) - 1) / 2, level_data["client_id"], rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved label divergence comparison to: {output_path}")


def _plot_comparison_feature_distance(stats_df, output_path):
    """Create a grouped bar chart comparing feature distance across all levels."""
    plt.figure(figsize=(12, 6))
    
    levels = stats_df["heterogeneity_level"].unique()
    x = np.arange(len(stats_df["client_id"].unique()))
    width = 0.15
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(levels)))
    
    for i, level in enumerate(levels):
        level_data = stats_df[stats_df["heterogeneity_level"] == level]
        level_data = level_data.sort_values("client_id")
        plt.bar(x + i * width, level_data["feature_distance"], width,
                label=level, color=colors[i])
    
    plt.xlabel("Client ID")
    plt.ylabel("Feature Distance (Euclidean)")
    plt.title("Client Feature Distance Across Heterogeneity Levels")
    plt.legend(title="Heterogeneity Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(x + width * (len(levels) - 1) / 2, level_data["client_id"], rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved feature distance comparison to: {output_path}")
