"""
Phase 13: Efficient SHAP Explainability for HEALIX-FL

This module evaluates whether SHAP explanations remain stable
when the number of background and explanation samples is reduced.

Phase 12 remains frozen and is used only as the reference baseline.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src.explainability.shap_explainability import (
    set_seed,
    load_processed_data,
    select_stratified_subset,
    load_model,
    compute_shap,
    global_feature_importance,
    feature_overlap,
    save_metadata,
    RANDOM_SEED,
    N_FEATURES,
    PRIMARY_MODEL,
)

RESULTS_DIR = "results/shap"
REPORTS_DIR = "reports"

PHASE12_BASELINE_CSV = os.path.join(
    RESULTS_DIR,
    "phase12_global_feature_importance.csv",
)

PHASE13_COMPARISON_CSV = os.path.join(
    RESULTS_DIR,
    "phase13_efficiency_comparison.csv",
)

PHASE13_GLOBAL_CSV = os.path.join(
    RESULTS_DIR,
    "phase13_global_feature_importance.csv",
)

PHASE13_METADATA_JSON = os.path.join(
    RESULTS_DIR,
    "phase13_run_metadata.json",
)

PHASE13_REPORT_PATH = os.path.join(
    REPORTS_DIR,
    "PHASE_13_EFFICIENT_EXPLAINABILITY_REPORT.md",
)


# ---------------------------------------------------------
# Phase 13 configurations
# ---------------------------------------------------------

CONFIGURATIONS = [
    {
        "condition": "baseline",
        "background_samples": 50,
        "explanation_samples": 100,
    },
    {
        "condition": "efficient_1",
        "background_samples": 25,
        "explanation_samples": 50,
    },
    {
        "condition": "efficient_2",
        "background_samples": 10,
        "explanation_samples": 50,
    },
    {
        "condition": "efficient_3",
        "background_samples": 10,
        "explanation_samples": 25,
    },
]

BASELINE_EXPLANATION = 100
BASELINE_BACKGROUND = 50


# ---------------------------------------------------------
# Load frozen Phase 12 baseline
# ---------------------------------------------------------

def load_phase12_baseline():
    """Load the frozen Phase 12 personalized feature-importance baseline."""
    if not os.path.exists(PHASE12_BASELINE_CSV):
        raise FileNotFoundError(
            "Phase 12 baseline missing: " + PHASE12_BASELINE_CSV
        )

    df = pd.read_csv(PHASE12_BASELINE_CSV)

    df = df[df["model"] == PRIMARY_MODEL].copy()

    if len(df) != N_FEATURES:
        raise ValueError(
            "Phase 12 baseline feature count mismatch: "
            + str(len(df))
        )

    df = df.sort_values('rank').reset_index(drop=True)

    return df


# ---------------------------------------------------------
# Calculate efficiency metrics
# ---------------------------------------------------------

def compute_efficiency_metrics(
    phase13_df,
    phase12_df,
    config,
    runtime,
):
    merged = phase13_df.merge(
        phase12_df,
        on="feature",
        suffixes=("_p13", "_p12"),
    )

    if len(merged) != N_FEATURES:
        raise ValueError(
            "Feature merge mismatch: " + str(len(merged))
        )

    # Top-10 feature overlap
    overlap_result = feature_overlap(
        phase13_df,
        phase12_df,
        top_k=10,
    )

    overlap_count = int(overlap_result["overlap_count"])
    overlap_percentage = float(overlap_result["overlap_percentage"])

    # Spearman rank correlation
    correlation, _ = spearmanr(
        merged["rank_p13"],
        merged["rank_p12"],
    )

    if np.isnan(correlation):
        spearman = None
    else:
        spearman = round(float(correlation), 6)

    # Mean absolute difference in global SHAP magnitude
    mean_abs_difference = float(
        np.mean(
            np.abs(
                merged["mean_abs_shap_p13"]
                - merged["mean_abs_shap_p12"]
            )
        )
    )

    mean_abs_difference = round(mean_abs_difference, 6)

    # Efficiency ratios
    explanation_fraction = (
        config["explanation_samples"]
        / BASELINE_EXPLANATION
    )

    background_fraction = (
        config["background_samples"]
        / BASELINE_BACKGROUND
    )

    explanation_reduction = (
        1.0 - explanation_fraction
    ) * 100.0

    background_reduction = (
        1.0 - background_fraction
    ) * 100.0

    return {
        "condition": config["condition"],
        "model": PRIMARY_MODEL,
        "background_samples": config["background_samples"],
        "explanation_samples": config["explanation_samples"],
        "runtime_seconds": round(runtime, 2),
        "top10_overlap_count": overlap_count,
        "top10_overlap_percentage": overlap_percentage,
        "spearman_rank_correlation": spearman,
        "mean_absolute_shap_difference": mean_abs_difference,
        "explanation_fraction_of_baseline": round(explanation_fraction, 4),
        "background_fraction_of_baseline": round(background_fraction, 4),
        "explanation_reduction_percentage": round(explanation_reduction, 2),
        "background_reduction_percentage": round(background_reduction, 2),
    }

# ---------------------------------------------------------
# Phase 12 reproducibility check
# ---------------------------------------------------------

def validate_baseline_reproducibility(comparison_row, phase12_df):
    """Verify Phase 13 baseline reproduces the frozen Phase 12 ranking."""
    if comparison_row['top10_overlap_count'] != 10:
        raise RuntimeError(
            "Phase 13 baseline failed reproducibility: "
            + str(comparison_row['top10_overlap_count'])
            + '/10 top-10 overlap, expected 10.'
        )

    if comparison_row['spearman_rank_correlation'] < 0.999:
        raise RuntimeError(
            "Phase 13 baseline failed reproducibility: "
            + 'Spearman correlation = '
            + str(comparison_row['spearman_rank_correlation'])
            + ', expected >= 0.999.'
        )

    print(
        "Baseline reproducibility check PASSED: "
        + str(comparison_row['top10_overlap_count'])
        + '/10 top-10 overlap, '
        + 'Spearman = '
        + str(comparison_row['spearman_rank_correlation'])
        + ', mean abs SHAP diff = '
        + str(comparison_row['mean_absolute_shap_difference'])
    )


# ---------------------------------------------------------
# Report generation
# ---------------------------------------------------------

def write_report(comparison_df, metadata):
    """Write the Phase 13 efficient explainability report."""
    report_path = Path('reports/PHASE_13_EFFICIENT_EXPLAINABILITY_REPORT.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = comparison_df[comparison_df['condition'] == 'baseline'].iloc[0]

    lines = [
        '# Phase 13 - Efficient Explainability',
        '',
        '## Objective',
        '',
        'Phase 13 evaluates whether SHAP explanations for the personalized IID HEALIX-FL model can be obtained with substantially fewer background and explanation samples while preserving agreement with the frozen Phase 12 global feature-importance ranking.',
        '',
        '## Experimental Setup',
        '',
        f'- Primary model: `{PRIMARY_MODEL}`',
        f'- Explainer: `shap.GradientExplainer`',
        f"- Features: {metadata['n_features']}",
        f"- Random seed: {metadata['random_seed']}",
        f"- Phase 12 baseline explanation samples: {metadata['phase12_baseline']['explanation_samples']}",
        f"- Phase 12 baseline background samples: {metadata['phase12_baseline']['background_samples']}",
        '',
        '## Configurations',
        '',
        '| Condition | Background | Explanation samples |',
        '|---|---:|---:|',
    ]

    for _, row in comparison_df.iterrows():
        lines.append(
            f"| {row['condition']} | "
            f"{int(row['background_samples'])} | "
            f"{int(row['explanation_samples'])} |"
        )

    lines.extend([
        '',
        '## Reproducibility',
        '',
        'The Phase 13 baseline uses the same SHAP configuration as Phase 12 and is validated before efficient configurations are interpreted.',
        '',
        f"- Baseline top-10 overlap: {int(baseline['top10_overlap_count'])}/10",
        f"- Baseline Spearman correlation: {baseline['spearman_rank_correlation']:.6f}",
        f"- Baseline mean absolute SHAP difference: {baseline['mean_absolute_shap_difference']:.8f}",
        '',
        'The baseline reproducibility check passed.',
        '',
        '## Efficiency Results',
        '',
        '| Condition | Runtime (s) | Top-10 overlap | Spearman | Mean abs SHAP difference | Explanation reduction | Background reduction |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])

    for _, row in comparison_df.iterrows():
        lines.append(
            f"| {row['condition']} | "
            f"{row['runtime_seconds']:.4f} | "
            f"{int(row['top10_overlap_count'])}/10 | "
            f"{row['spearman_rank_correlation']:.6f} | "
            f"{row['mean_absolute_shap_difference']:.8f} | "
            f"{row['explanation_reduction_percentage']:.2f}% | "
            f"{row['background_reduction_percentage']:.2f}% |"
        )

    lines.extend([
        '',
        '## Key Observations',
        '',
        'Efficiency should be interpreted jointly with explanation agreement. A smaller SHAP sample is useful only when the result remains sufficiently consistent with the Phase 12 reference.',
        '',
        '## Conclusion',
        '',
        'Phase 13 provides an empirical comparison between the original Phase 12 SHAP configuration and reduced-sample configurations. The results quantify the trade-off between SHAP computation cost and preservation of global feature importance.',
        '',
        'No model was retrained during this phase, and the Phase 12 outputs were treated as frozen reference results.',
        '',
    ])

    report_path.write_text(
        chr(10).join(lines),
        encoding='utf-8'
    )

    print(f'Report written to: {report_path}')


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():
    set_seed(RANDOM_SEED)

    print('=' * 70)
    print('PHASE 13 - EFFICIENT SHAP EXPLAINABILITY')
    print('=' * 70)

    X_train, X_test, y_test, feature_names = load_processed_data()

    phase12_df = load_phase12_baseline()

    model, model_meta = load_model(PRIMARY_MODEL)

    comparison_rows = []
    global_rows = []

    for config in CONFIGURATIONS:
        condition = config['condition']
        background_size = config['background_samples']
        explanation_size = config['explanation_samples']

        print(
            f'\nRunning {condition}: '
            f'background={background_size}, '
            f'explanation={explanation_size}'
        )

        background = X_train[:background_size]

        X_explain, y_explain, explain_idx = select_stratified_subset(
            X_test,
            y_test,
            explanation_size,
            RANDOM_SEED
        )

        start_time = time.perf_counter()

        shap_values = compute_shap(
            model,
            background,
            X_explain
        )

        runtime_seconds = time.perf_counter() - start_time

        importance_df = global_feature_importance(
            PRIMARY_MODEL,
            shap_values,
            feature_names
        )

        metrics = compute_efficiency_metrics(
            importance_df,
            phase12_df,
            config,
            runtime_seconds
        )

        comparison_rows.append(metrics)

        importance_df.insert(0, 'condition', condition)
        global_rows.append(importance_df)

        print(
            f'{condition}: runtime={runtime_seconds:.4f}s, '
            f'top10={metrics["top10_overlap_count"]}/10, '
            f'spearman={metrics["spearman_rank_correlation"]:.6f}'
        )

        if condition == 'baseline':
            validate_baseline_reproducibility(
                metrics,
                phase12_df
            )

    comparison_df = pd.DataFrame(comparison_rows)

    global_df = pd.concat(
        global_rows,
        ignore_index=True
    )

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    comparison_path = Path(RESULTS_DIR) / 'phase13_efficiency_comparison.csv'
    global_path = Path(RESULTS_DIR) / 'phase13_global_feature_importance.csv'
    metadata_path = Path(RESULTS_DIR) / 'phase13_run_metadata.json'

    comparison_df.to_csv(comparison_path, index=False)
    global_df.to_csv(global_path, index=False)

    metadata = {
        'phase': 13,
        'description': 'Efficient SHAP explainability',
        'primary_model': PRIMARY_MODEL,
        'random_seed': RANDOM_SEED,
        'n_features': int(X_train.shape[1]),
        'n_test_samples': int(X_test.shape[0]),
        'explainer': 'shap.GradientExplainer',
        'shap_version': shap.__version__,
        'phase12_baseline': {
            'background_samples': 50,
            'explanation_samples': 100,
        },
        'configurations': CONFIGURATIONS,
        'model_metadata': model_meta,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding='utf-8'
    )

    write_report(comparison_df, metadata)

    print('\n' + '=' * 70)
    print('PHASE 13 COMPLETED SUCCESSFULLY')
    print('=' * 70)
    print(f'Comparison CSV: {comparison_path}')
    print(f'Global CSV:     {global_path}')
    print(f'Metadata JSON:  {metadata_path}')


if __name__ == '__main__':
    main()
