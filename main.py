import yaml
import os
import sys
import argparse
import subprocess
from src.data.audit import run_audit
from src.data.preprocess import run_preprocessing
from src.data.partition import run_phase3, run_phase4
from src.models.train import run_phase5
from src.models.train_advanced import run_phase16
from src.federated.local import run_phase6, run_phase6c
from src.federated.fedavg import run_phase7
from src.federated.fedprox import run_phase8
from src.federated.personalized import run_phase9
from src.federated.adaptive_aggregation import run_phase10
from src.federated.differential_privacy import run_phase11
from src.federated.phase14_ablation import run_ablation_study
from src.evaluation.phase15_final_evaluation import run_phase15



def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_phase1(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 1 - Data Audit")
    print("=" * 60)
    print(f"Dataset: {config['paths']['raw_data']}")
    print(f"Target column: {config['dataset']['target_column']}")
    print(f"Output: {config['phase1']['audit']['output_csv']}")
    print("=" * 60)
    print()

    if not os.path.exists(config["paths"]["raw_data"]):
        raise FileNotFoundError(
            f"Dataset not found at {config['paths']['raw_data']}"
        )

    run_audit(config)

    print()
    print("=" * 60)
    print("Phase 1 completed successfully.")
    print(f"Audit summary saved to: {config['phase1']['audit']['output_csv']}")
    print("=" * 60)


def run_phase2(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 2 - Data Preprocessing")
    print("=" * 60)
    print()

    if not os.path.exists(config["paths"]["raw_data"]):
        raise FileNotFoundError(
            f"Dataset not found at {config['paths']['raw_data']}"
        )

    run_preprocessing(config)

    print()
    print("=" * 60)
    print("Phase 2 preprocessing completed successfully.")
    print(f"Processed data saved to: {config['paths']['processed_data']}")
    print("=" * 60)


def run_phase3_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 3 - IID Federated Client Generation")
    print("=" * 60)
    print()

    processed_dir = config["paths"]["processed_data"]
    if not os.path.exists(os.path.join(processed_dir, "X_train.npy")):
        raise FileNotFoundError(
            "Processed training data not found. Run Phase 2 first: python main.py 2"
        )

    run_phase3(config)

    print()
    print("=" * 60)
    print("Phase 3 IID client generation completed successfully.")
    print(f"Client data saved to: {config['phase3']['federated']['iid']['output_dir']}")
    print("=" * 60)


def run_phase4_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 4 - Non-IID Healthcare Client Generation")
    print("=" * 60)
    print()

    processed_dir = config["paths"]["processed_data"]
    if not os.path.exists(os.path.join(processed_dir, "X_train.npy")):
        raise FileNotFoundError(
            "Processed training data not found. Run Phase 2 first: python main.py 2"
        )

    run_phase4(config)

    print()
    print("=" * 60)
    print("Phase 4 non-IID client generation completed successfully.")
    print(f"Client data saved to: {config['phase4']['non_iid']['output_base_dir']}")
    print("=" * 60)


def run_phase5_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 5 - Centralized MLP Baseline")
    print("=" * 60)
    print()

    processed_dir = config["paths"]["processed_data"]
    if not os.path.exists(os.path.join(processed_dir, "X_train.npy")):
        raise FileNotFoundError(
            "Processed training data not found. Run Phase 2 first: python main.py 2"
        )

    run_phase5(config)

    print()
    print("=" * 60)
    print("Phase 5 centralized baseline completed successfully.")
    print("=" * 60)


def run_phase6_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 6 - Local-Only Client Training")
    print("=" * 60)
    print()

    run_phase6(config)

    print()
    print("=" * 60)
    print("Phase 6 local-only client training completed successfully.")
    print("=" * 60)


def run_phase6c_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 6C - Non-IID Local-Only Client Training")
    print("=" * 60)
    print()

    run_phase6c(config)

    print()
    print("=" * 60)
    print("Phase 6C non-IID local-only client training completed successfully.")
    print("=" * 60)


def run_phase7_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 7 - Federated Averaging (FedAvg)")
    print("=" * 60)
    print()

    run_phase7(config)

    print()
    print("=" * 60)
    print("Phase 7 FedAvg completed successfully.")
    print("=" * 60)


def run_phase8_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 8 - Federated Proximal (FedProx)")
    print("=" * 60)
    print()

    run_phase8(config)

    print()
    print("=" * 60)
    print("Phase 8 FedProx completed successfully.")
    print("=" * 60)


def run_phase9_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 9 - Personalized Federated Learning")
    print("=" * 60)
    print()

    run_phase9(config)

    print()
    print("=" * 60)
    print("Phase 9 personalized federated learning completed successfully.")
    print("=" * 60)


def run_phase10_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 10 - Adaptive Aggregation")
    print("=" * 60)
    run_phase10(config)


def run_phase11_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 11 - Differential Privacy")
    print("=" * 60)
    run_phase11(config)


def run_explainability_module(module_name, phase_name):
    """Run a script-style explainability phase with the active interpreter."""
    print("=" * 60)
    print("HEALIX-FL: " + phase_name)
    print("=" * 60)
    subprocess.run(
        [sys.executable, "-m", module_name],
        check=True,
    )


def run_phase14_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 14 - Ablation Study")
    print("=" * 60)
    run_ablation_study(config)


def run_phase15_wrapper(config):
    del config
    print("=" * 60)
    print("HEALIX-FL: Phase 15 - Final Evaluation and Reproducibility")
    print("=" * 60)
    run_phase15()


def run_phase16_wrapper(config):
    print("=" * 60)
    print("HEALIX-FL: Phase 16 - Advanced Centralized Residual MLP")
    print("=" * 60)
    print()

    processed_dir = config["paths"]["processed_data"]
    if not os.path.exists(os.path.join(processed_dir, "X_train.npy")):
        raise FileNotFoundError(
            "Processed training data not found. Run Phase 2 first: python main.py 2"
        )

    run_phase16(config)

    print()
    print("=" * 60)
    print("Phase 16 advanced centralized model completed successfully.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="HEALIX-FL: Heterogeneity-Aware and Explainable Federated Learning Framework"
    )
    parser.add_argument(
        "phase",
        nargs="?",
        default="5",
        choices=["1", "2", "3", "4", "5", "6", "6c", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"],
        help="Phase to run: '1' audit, '2' preprocessing, '3' IID clients, '4' non-IID clients, '5' centralized baseline, '6' local-only, '6c' non-IID local-only, '7' FedAvg, '8' FedProx, '9' personalization, '10' adaptive aggregation, '11' differential privacy, '12' SHAP, '13' efficient SHAP, '14' ablation, '15' final evaluation, '16' advanced centralized model (default: 5)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.phase == "1":
        run_phase1(config)
    elif args.phase == "2":
        run_phase2(config)
    elif args.phase == "3":
        run_phase3_wrapper(config)
    elif args.phase == "4":
        run_phase4_wrapper(config)
    elif args.phase == "5":
        run_phase5_wrapper(config)
    elif args.phase == "6":
        run_phase6_wrapper(config)
    elif args.phase == "6c":
        run_phase6c_wrapper(config)
    elif args.phase == "7":
        run_phase7_wrapper(config)
    elif args.phase == "8":
        run_phase8_wrapper(config)
    elif args.phase == "9":
        run_phase9_wrapper(config)
    elif args.phase == "10":
        run_phase10_wrapper(config)
    elif args.phase == "11":
        run_phase11_wrapper(config)
    elif args.phase == "12":
        run_explainability_module("src.explainability.shap_explainability", "Phase 12 - SHAP Explainability")
    elif args.phase == "13":
        run_explainability_module("src.explainability.phase13_efficient_shap", "Phase 13 - Efficient Explainability")
    elif args.phase == "14":
        run_phase14_wrapper(config)
    elif args.phase == "15":
        run_phase15_wrapper(config)
    elif args.phase == "16":
        run_phase16_wrapper(config)
    


if __name__ == "__main__":
    main()
