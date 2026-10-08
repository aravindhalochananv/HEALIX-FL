"""
Phase 14: Ablation Study for HEALIX-FL

Controlled 2x2x2 factorial ablation over three federated learning components:

1. Proximal term
2. Adaptive aggregation
3. Personalized client models

8 ablation configurations x 5 heterogeneity levels = 40 conditions.

Test data are NEVER used during the federated training/selection loop.
The test set is evaluated only after the best validation round is frozen.
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.federated.fedavg import (
    set_seed,
    load_processed_data,
    build_centralized_splits,
    evaluate_model,
    load_client_data,
    get_trainable_clients,
    train_local_model,
    federated_average,
    initialize_global_model,
    get_device,
    SEED,
    FED_ROUNDS,
    LOCAL_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    CLIENT_FRACTION,
    VAL_SIZE,
    NUM_CLIENTS,
    CONDITIONS,
    CLIENTS_DATA_ROOT,
    PROCESSED_DIR,
)

from src.federated.fedprox import (
    train_local_model_fedprox,
    MU,
)

from src.federated.adaptive_aggregation import (
    compute_adaptive_weights,
    adaptive_aggregate,
    QUALITY_LOWER,
    QUALITY_UPPER,
)

from src.models.mlp import MLP


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("results/phase14_ablation")
REPORTS_DIR = "reports"


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    {
        "name": "fedavg_baseline",
        "proximal": False,
        "adaptive": False,
        "personalized": False,
    },
    {
        "name": "fedprox_only",
        "proximal": True,
        "adaptive": False,
        "personalized": False,
    },
    {
        "name": "adaptive_only",
        "proximal": False,
        "adaptive": True,
        "personalized": False,
    },
    {
        "name": "fedprox_adaptive",
        "proximal": True,
        "adaptive": True,
        "personalized": False,
    },
    {
        "name": "personalized_only",
        "proximal": False,
        "adaptive": False,
        "personalized": True,
    },
    {
        "name": "fedprox_personalized",
        "proximal": True,
        "adaptive": False,
        "personalized": True,
    },
    {
        "name": "adaptive_personalized",
        "proximal": False,
        "adaptive": True,
        "personalized": True,
    },
    {
        "name": "full_healix",
        "proximal": True,
        "adaptive": True,
        "personalized": True,
    },
]

HETEROGENEITY_CONDITIONS = CONDITIONS


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _clone_state_dict(state_dict):
    """Deep-clone a state dictionary onto CPU."""
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def _clone_personalized_states(personalized_states):
    """Deep-clone all personalized client states."""
    cloned = {}

    for client_id, state in personalized_states.items():
        if state is None:
            cloned[client_id] = None
        else:
            cloned[client_id] = _clone_state_dict(state)

    return cloned


def _build_model_from_state(state_dict, device):
    """Create a fresh MLP and load a state dictionary."""
    model = MLP(
        input_dim=2370,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ---------------------------------------------------------------------------
# Trainer selection
# ---------------------------------------------------------------------------

def _select_trainer(ablation_config):
    """
    Select FedAvg-style or FedProx-style local training.
    """

    if ablation_config["proximal"]:
        return lambda global_state, X, y, device: (
            train_local_model_fedprox(
                global_state,
                X,
                y,
                device,
                mu=MU,
            )
        )

    return lambda global_state, X, y, device: (
        train_local_model(
            global_state,
            X,
            y,
            device,
        )
    )


# ---------------------------------------------------------------------------
# Personalized validation
# ---------------------------------------------------------------------------

def _mean_personalized_val_roc(
    personalized_states,
    client_ids,
    X_val,
    y_val,
    device,
):
    """
    Evaluate personalized client models on centralized validation data.

    Returns:
        mean ROC-AUC,
        list of client ROC-AUC values
    """

    roc_aucs = []

    for client_id in client_ids:

        state = personalized_states.get(client_id)

        if state is None:
            continue

        model = _build_model_from_state(
            state,
            device,
        )

        _, val_metrics = evaluate_model(
            model,
            X_val,
            y_val,
            device,
        )

        roc = val_metrics["roc_auc"]

        if np.isfinite(roc):
            roc_aucs.append(float(roc))

    if not roc_aucs:
        return np.nan, []

    return float(np.mean(roc_aucs)), roc_aucs


# ---------------------------------------------------------------------------
# One ablation-condition experiment
# ---------------------------------------------------------------------------

def run_ablation_condition(
    ablation_config,
    condition,
    X_train_split,
    y_train_split,
    X_val,
    y_val,
    X_test,
    y_test,
    device,
):
    """
    Run one ablation configuration for one heterogeneity condition.

    Important:
        - Training uses client data.
        - Validation is used for model selection.
        - Test is used only after the best validation round is frozen.
    """

    # These centralized training arrays are supplied for API consistency.
    # The federated experiment trains directly from client files.
    del X_train_split
    del y_train_split

    start_time = time.perf_counter()

    set_seed(SEED)

    name = ablation_config["name"]

    # -----------------------------------------------------------------------
    # Client availability
    # -----------------------------------------------------------------------

    trainable_clients, skipped_clients = get_trainable_clients(
        condition
    )

    if not trainable_clients:
        raise RuntimeError(
            f"No trainable clients for condition '{condition}'."
        )

    # -----------------------------------------------------------------------
    # Fresh global model
    # -----------------------------------------------------------------------

    global_model = initialize_global_model(device)

    global_state_dict = _clone_state_dict(
        global_model.state_dict()
    )

    # -----------------------------------------------------------------------
    # Best-model tracking
    # -----------------------------------------------------------------------

    personalized_states = {}

    best_personalized_states = None
    best_global_state = None

    best_round = None

    best_global_val_roc_auc = -np.inf
    best_personalized_val_roc_auc = -np.inf

    # -----------------------------------------------------------------------
    # Experiment records
    # -----------------------------------------------------------------------

    round_metrics = []
    client_participation = []
    adaptive_weight_records = []

    trainer = _select_trainer(
        ablation_config
    )

    # =======================================================================
    # FEDERATED TRAINING LOOP
    # =======================================================================

    for rnd in range(
        1,
        FED_ROUNDS + 1,
    ):

        selected_clients = list(
            trainable_clients
        )

        trained_client_ids = []

        local_states = []
        sample_counts = []

        adaptive_val_roc_aucs = []

        # -------------------------------------------------------------------
        # Local client training
        # -------------------------------------------------------------------

        for client_id in selected_clients:

            X, y = load_client_data(
                condition,
                client_id,
            )

            if X is None or len(y) < 2:

                client_participation.append(
                    {
                        "ablation": name,
                        "heterogeneity": condition,
                        "round": rnd,
                        "client": client_id,
                        "num_samples": 0,
                        "local_epochs": LOCAL_EPOCHS,
                        "status": "missing_or_insufficient_data",
                    }
                )

                continue

            local_state, num_train_samples, status = trainer(
                global_state_dict,
                X,
                y,
                device,
            )

            trained_client_ids.append(
                client_id
            )

            local_states.append(
                local_state
            )

            sample_counts.append(
                num_train_samples
            )

            client_participation.append(
                {
                    "ablation": name,
                    "heterogeneity": condition,
                    "round": rnd,
                    "client": client_id,
                    "num_samples": num_train_samples,
                    "local_epochs": LOCAL_EPOCHS,
                    "status": status,
                }
            )

            # ---------------------------------------------------------------
            # Adaptive aggregation quality
            # ---------------------------------------------------------------

            if ablation_config["adaptive"]:

                local_model = _build_model_from_state(
                    local_state,
                    device,
                )

                _, local_val_metrics = evaluate_model(
                    local_model,
                    X_val,
                    y_val,
                    device,
                )

                adaptive_val_roc_aucs.append(
                    float(
                        local_val_metrics["roc_auc"]
                    )
                )

            # ---------------------------------------------------------------
            # Personalized state
            # ---------------------------------------------------------------

            if ablation_config["personalized"]:

                personalized_states[client_id] = (
                    _clone_state_dict(local_state)
                )

        # -------------------------------------------------------------------
        # Check local training
        # -------------------------------------------------------------------

        if not local_states:

            raise RuntimeError(
                f"Round {rnd} produced no trainable local models "
                f"for {name}/{condition}."
            )

        # -------------------------------------------------------------------
        # Aggregation
        # -------------------------------------------------------------------

        if ablation_config["adaptive"]:

            adaptive_weights = compute_adaptive_weights(
                sample_counts,
                adaptive_val_roc_aucs,
            )

            weight_sum = float(
                sum(adaptive_weights)
            )

            if (
                weight_sum <= 0
                or not np.isfinite(weight_sum)
            ):
                raise RuntimeError(
                    f"Invalid adaptive weights in round {rnd}: "
                    f"{adaptive_weights}"
                )

            if abs(weight_sum - 1.0) > 1e-6:

                adaptive_weights = [
                    weight / weight_sum
                    for weight in adaptive_weights
                ]

            total_samples = float(
                sum(sample_counts)
            )

            for idx, client_id in enumerate(
                trained_client_ids
            ):

                adaptive_weight_records.append(
                    {
                        "ablation": name,
                        "heterogeneity": condition,
                        "round": rnd,
                        "client": client_id,
                        "sample_count": int(
                            sample_counts[idx]
                        ),
                        "validation_roc_auc": (
                            adaptive_val_roc_aucs[idx]
                        ),
                        "base_weight": (
                            float(sample_counts[idx])
                            / total_samples
                        ),
                        "adaptive_weight": (
                            float(adaptive_weights[idx])
                        ),
                    }
                )

            global_state_dict = adaptive_aggregate(
                local_states,
                adaptive_weights,
            )

        else:

            global_state_dict = federated_average(
                local_states,
                sample_counts,
            )

        # -------------------------------------------------------------------
        # Global validation
        # -------------------------------------------------------------------

        global_eval_model = _build_model_from_state(
            global_state_dict,
            device,
        )

        global_val_loss, global_val_metrics = evaluate_model(
            global_eval_model,
            X_val,
            y_val,
            device,
        )

        global_val_roc = float(
            global_val_metrics["roc_auc"]
        )

        # -------------------------------------------------------------------
        # Personalized validation
        # -------------------------------------------------------------------

        personalized_val_roc = np.nan
        personalized_client_val_roc_aucs = []

        if ablation_config["personalized"]:

            (
                personalized_val_roc,
                personalized_client_val_roc_aucs,
            ) = _mean_personalized_val_roc(
                personalized_states,
                trained_client_ids,
                X_val,
                y_val,
                device,
            )

        # -------------------------------------------------------------------
        # Selection metric
        # -------------------------------------------------------------------

        if ablation_config["personalized"]:

            selection_metric = personalized_val_roc

        else:

            selection_metric = global_val_roc

        # -------------------------------------------------------------------
        # Round metrics
        # -------------------------------------------------------------------

        round_metrics.append(
            {
                "ablation": name,
                "heterogeneity": condition,
                "round": rnd,
                "global_val_loss": float(
                    global_val_loss
                ),
                "global_val_roc_auc": global_val_roc,
                "global_val_pr_auc": float(
                    global_val_metrics["pr_auc"]
                ),
                "global_val_accuracy": float(
                    global_val_metrics["accuracy"]
                ),
                "global_val_precision": float(
                    global_val_metrics["precision"]
                ),
                "global_val_recall": float(
                    global_val_metrics["recall"]
                ),
                "global_val_f1_score": float(
                    global_val_metrics["f1_score"]
                ),
                "personalized_val_roc_auc": (
                    float(personalized_val_roc)
                    if np.isfinite(
                        personalized_val_roc
                    )
                    else np.nan
                ),
                "personalized_val_pr_auc": np.nan,
                "personalized_val_clients": len(
                    personalized_client_val_roc_aucs
                ),
                "selection_metric": (
                    float(selection_metric)
                    if np.isfinite(
                        selection_metric
                    )
                    else np.nan
                ),
                "num_trained_clients": len(
                    trained_client_ids
                ),
            }
        )

        # -------------------------------------------------------------------
        # BEST-ROUND SELECTION
        #
        # IMPORTANT:
        # This block is INSIDE the federated round loop.
        # Test data are not used here.
        # -------------------------------------------------------------------

        if ablation_config["personalized"]:

            # Track global validation performance separately.
            if (
                np.isfinite(global_val_roc)
                and global_val_roc
                > best_global_val_roc_auc
            ):

                best_global_val_roc_auc = (
                    global_val_roc
                )

            # Personalized configurations select the best round
            # using mean personalized validation ROC-AUC.
            if (
                np.isfinite(personalized_val_roc)
                and personalized_val_roc
                > best_personalized_val_roc_auc
            ):

                best_personalized_val_roc_auc = (
                    personalized_val_roc
                )

                best_round = rnd

                best_global_state = (
                    _clone_state_dict(
                        global_state_dict
                    )
                )

                best_personalized_states = (
                    _clone_personalized_states(
                        personalized_states
                    )
                )

        else:

            # Non-personalized configurations select the best
            # global model using global validation ROC-AUC.
            if (
                np.isfinite(global_val_roc)
                and global_val_roc
                > best_global_val_roc_auc
            ):

                best_global_val_roc_auc = (
                    global_val_roc
                )

                best_round = rnd

                best_global_state = (
                    _clone_state_dict(
                        global_state_dict
                    )
                )

    # =======================================================================
    # END OF FEDERATED TRAINING LOOP
    # =======================================================================

    # -----------------------------------------------------------------------
    # Safety fallback
    # -----------------------------------------------------------------------

    if best_global_state is None:

        best_global_state = (
            _clone_state_dict(
                global_state_dict
            )
        )

        best_round = FED_ROUNDS

    if (
        ablation_config["personalized"]
        and best_personalized_states is None
    ):

        best_personalized_states = (
            _clone_personalized_states(
                personalized_states
            )
        )

        best_round = FED_ROUNDS

    # =======================================================================
    # FINAL TEST EVALUATION
    #
    # IMPORTANT:
    # This is OUTSIDE the federated training loop.
    # =======================================================================

    final_test_client_metrics = []

    if ablation_config["personalized"]:

        test_rocs = []
        test_prs = []

        for client_id in trainable_clients:

            state = best_personalized_states.get(
                client_id
            )

            if state is None:
                continue

            model = _build_model_from_state(
                state,
                device,
            )

            test_loss, test_metrics = evaluate_model(
                model,
                X_test,
                y_test,
                device,
            )

            row = {
                "ablation": name,
                "heterogeneity": condition,
                "client": client_id,
                "test_loss": float(
                    test_loss
                ),
            }

            row.update(
                {
                    f"test_{key}": value
                    for key, value
                    in test_metrics.items()
                }
            )

            final_test_client_metrics.append(
                row
            )

            if np.isfinite(
                test_metrics["roc_auc"]
            ):

                test_rocs.append(
                    float(
                        test_metrics["roc_auc"]
                    )
                )

            if np.isfinite(
                test_metrics["pr_auc"]
            ):

                test_prs.append(
                    float(
                        test_metrics["pr_auc"]
                    )
                )

        test_roc_auc = (
            float(np.mean(test_rocs))
            if test_rocs
            else np.nan
        )

        test_pr_auc = (
            float(np.mean(test_prs))
            if test_prs
            else np.nan
        )

    else:

        best_model = _build_model_from_state(
            best_global_state,
            device,
        )

        test_loss, test_metrics = evaluate_model(
            best_model,
            X_test,
            y_test,
            device,
        )

        test_roc_auc = float(
            test_metrics["roc_auc"]
        )

        test_pr_auc = float(
            test_metrics["pr_auc"]
        )

        row = {
            "ablation": name,
            "heterogeneity": condition,
            "client": "global",
            "test_loss": float(
                test_loss
            ),
        }

        row.update(
            {
                f"test_{key}": value
                for key, value
                in test_metrics.items()
            }
        )

        final_test_client_metrics.append(
            row
        )

    # -----------------------------------------------------------------------
    # Runtime and final result
    # -----------------------------------------------------------------------

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    if ablation_config["personalized"]:

        best_validation_metric = (
            best_personalized_val_roc_auc
        )

    else:

        best_validation_metric = (
            best_global_val_roc_auc
        )

    return {
        "ablation": name,
        "heterogeneity": condition,
        "config": dict(ablation_config),
        "trainable_clients": trainable_clients,
        "skipped_clients": skipped_clients,
        "round_metrics": round_metrics,
        "client_participation": client_participation,
        "adaptive_weight_records": adaptive_weight_records,
        "best_global_state": best_global_state,
        "best_personalized_states": best_personalized_states,
        "best_round": int(best_round),
        "global_best_val_roc_auc": float(
            best_global_val_roc_auc
        ),
        "best_personalized_val_roc_auc": (
            float(
                best_personalized_val_roc_auc
            )
            if np.isfinite(
                best_personalized_val_roc_auc
            )
            else np.nan
        ),
        "best_validation_metric": float(
            best_validation_metric
        ),
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "final_test_client_metrics": (
            final_test_client_metrics
        ),
        "runtime_seconds": float(
            runtime_seconds
        ),
    }


# ---------------------------------------------------------------------------
# Configuration output
# ---------------------------------------------------------------------------

def _config_payload(
    ablation_config,
    condition,
):
    """Create YAML-compatible configuration metadata."""

    return {
        "phase": 14,
        "experiment": "2x2x2 factorial ablation study",
        "ablation": ablation_config["name"],
        "heterogeneity": condition,

        "proximal": bool(
            ablation_config["proximal"]
        ),

        "adaptive": bool(
            ablation_config["adaptive"]
        ),

        "personalized": bool(
            ablation_config["personalized"]
        ),

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
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "client_fraction": CLIENT_FRACTION,
            "num_clients": NUM_CLIENTS,
            "fedprox_mu": MU,
        },

        "adaptive": {
            "quality_lower": QUALITY_LOWER,
            "quality_upper": QUALITY_UPPER,
            "quality_source": (
                "local_state_centralized_validation_roc_auc"
            ),
        },

        "selection": (
            "mean_personalized_validation_roc_auc"
            if ablation_config["personalized"]
            else "global_validation_roc_auc"
        ),

        "test_policy": (
            "test evaluation only after "
            "best round is frozen"
        ),

        "random_seed": SEED,
    }


# ---------------------------------------------------------------------------
# Save one condition
# ---------------------------------------------------------------------------

def save_condition_outputs(
    result,
    output_dir,
):
    """Save outputs for one ablation-condition pair."""

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Configuration
    with open(
        output_dir / "config.yaml",
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            _config_payload(
                result["config"],
                result["heterogeneity"],
            ),
            f,
            sort_keys=False,
        )

    # Round metrics
    pd.DataFrame(
        result["round_metrics"]
    ).to_csv(
        output_dir / "round_metrics.csv",
        index=False,
    )

    # Client participation
    pd.DataFrame(
        result["client_participation"]
    ).to_csv(
        output_dir / "client_participation.csv",
        index=False,
    )

    # Adaptive weights
    if result["adaptive_weight_records"]:

        pd.DataFrame(
            result["adaptive_weight_records"]
        ).to_csv(
            output_dir
            / "adaptive_client_weights.csv",
            index=False,
        )

    # Best global model
    torch.save(
        result["best_global_state"],
        output_dir
        / "best_global_model.pt",
    )

    # Personalized models
    if (
        result["best_personalized_states"]
        is not None
    ):

        personal_dir = (
            output_dir
            / "best_personalized_models"
        )

        personal_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for (
            client_id,
            state,
        ) in result[
            "best_personalized_states"
        ].items():

            if state is not None:

                torch.save(
                    state,
                    personal_dir
                    / f"client_{client_id}.pt",
                )

    # Final test metrics
    pd.DataFrame(
        result["final_test_client_metrics"]
    ).to_csv(
        output_dir
        / "test_client_metrics.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Summary row
# ---------------------------------------------------------------------------

def build_ablation_summary_row(
    result,
):
    """Build one row for the Phase 14 summary."""

    cfg = result["config"]

    return {
        "ablation": result["ablation"],
        "heterogeneity": result[
            "heterogeneity"
        ],

        "proximal": bool(
            cfg["proximal"]
        ),

        "adaptive": bool(
            cfg["adaptive"]
        ),

        "personalized": bool(
            cfg["personalized"]
        ),

        "best_round": int(
            result["best_round"]
        ),

        "best_validation_metric": float(
            result[
                "best_validation_metric"
            ]
        ),

        "global_best_val_roc_auc": float(
            result[
                "global_best_val_roc_auc"
            ]
        ),

        "best_personalized_val_roc_auc": (
            result[
                "best_personalized_val_roc_auc"
            ]
        ),

        "test_roc_auc": float(
            result["test_roc_auc"]
        ),

        "test_pr_auc": float(
            result["test_pr_auc"]
        ),

        "num_trainable_clients": len(
            result["trainable_clients"]
        ),

        "num_skipped_clients": len(
            result["skipped_clients"]
        ),

        "runtime_seconds": float(
            result["runtime_seconds"]
        ),
    }


# ---------------------------------------------------------------------------
# Full Phase 14 experiment
# ---------------------------------------------------------------------------

def run_ablation_study(
    config=None,
):
    """
    Run all 8 x 5 = 40 Phase 14 conditions.

    IMPORTANT:
    This function performs the complete experiment.
    Do not call it until the smoke test has passed.
    """

    del config

    study_start = time.perf_counter()

    set_seed(SEED)

    device = get_device()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Load existing processed data
    # -----------------------------------------------------------------------

    (
        X_train_full,
        y_train_full,
        X_test,
        y_test,
    ) = load_processed_data()

    (
        X_train_split,
        y_train_split,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = build_centralized_splits(
        X_train_full,
        y_train_full,
        X_test,
        y_test,
    )

    # -----------------------------------------------------------------------
    # Storage
    # -----------------------------------------------------------------------

    summary_rows = []
    all_round_metrics = []
    all_client_participation = []
    all_test_client_metrics = []

    # -----------------------------------------------------------------------
    # Create 40 planned conditions
    # -----------------------------------------------------------------------

    planned_conditions = []

    for ablation_config in ABLATION_CONFIGS:

        for condition in (
            HETEROGENEITY_CONDITIONS
        ):

            planned_conditions.append(
                (
                    ablation_config,
                    condition,
                )
            )

    if len(planned_conditions) != 40:

        raise AssertionError(
            "Expected 40 conditions, "
            f"found {len(planned_conditions)}"
        )

    # -----------------------------------------------------------------------
    # Execute all conditions
    # -----------------------------------------------------------------------

    for index, (
        ablation_config,
        condition,
    ) in enumerate(
        planned_conditions,
        start=1,
    ):

        print("=" * 72)

        print(
            f"PHASE 14 CONDITION "
            f"{index}/{len(planned_conditions)}: "
            f"{ablation_config['name']} / "
            f"{condition}"
        )

        print("=" * 72)

        result = run_ablation_condition(
            ablation_config,
            condition,
            X_train_split,
            y_train_split,
            X_val,
            y_val,
            X_test,
            y_test,
            device,
        )

        condition_dir = (
            OUTPUT_ROOT
            / ablation_config["name"]
            / condition
        )

        save_condition_outputs(
            result,
            condition_dir,
        )

        summary_rows.append(
            build_ablation_summary_row(
                result
            )
        )

        all_round_metrics.extend(
            result["round_metrics"]
        )

        all_client_participation.extend(
            result[
                "client_participation"
            ]
        )

        all_test_client_metrics.extend(
            result[
                "final_test_client_metrics"
            ]
        )

    # -----------------------------------------------------------------------
    # Root summary files
    # -----------------------------------------------------------------------

    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUTPUT_ROOT
        / "phase14_ablation_summary.csv",
        index=False,
    )

    pd.DataFrame(
        all_round_metrics
    ).to_csv(
        OUTPUT_ROOT
        / "phase14_ablation_round_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        all_client_participation
    ).to_csv(
        OUTPUT_ROOT
        / "phase14_ablation_client_participation.csv",
        index=False,
    )

    pd.DataFrame(
        all_test_client_metrics
    ).to_csv(
        OUTPUT_ROOT
        / "phase14_ablation_test_client_metrics.csv",
        index=False,
    )

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    metadata = {
        "phase": 14,
        "experiment": (
            "2x2x2 factorial ablation study"
        ),
        "seed": SEED,
        "num_ablation_configurations": len(
            ABLATION_CONFIGS
        ),
        "num_heterogeneity_levels": len(
            HETEROGENEITY_CONDITIONS
        ),
        "planned_conditions": len(
            planned_conditions
        ),
        "ablation_configurations": (
            ABLATION_CONFIGS
        ),
        "heterogeneity_conditions": list(
            HETEROGENEITY_CONDITIONS
        ),
        "test_used_inside_training_loop": False,
        "test_evaluation_policy": (
            "after best validation round "
            "is frozen"
        ),
        "total_runtime_seconds": float(
            time.perf_counter()
            - study_start
        ),
    }

    with open(
        OUTPUT_ROOT
        / "phase14_run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    return pd.DataFrame(
        summary_rows
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    run_ablation_study()