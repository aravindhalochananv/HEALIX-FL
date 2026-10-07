import yaml
import os
import sys
import argparse
from src.data.audit import run_audit
from src.data.preprocess import run_preprocessing
from src.data.partition import run_phase3, run_phase4
from src.models.train import run_phase5
from src.federated.local import run_phase6, run_phase6c
from src.federated.fedavg import run_phase7
from src.federated.fedprox import run_phase8
from src.federated.personalized import run_phase9



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




    


def main():
    parser = argparse.ArgumentParser(
        description="HEALIX-FL: Heterogeneity-Aware and Explainable Federated Learning Framework"
    )
    parser.add_argument(
        "phase",
        nargs="?",
        default="5",
        choices=["1", "2", "3", "4", "5", "6", "6c", "7", "8", "9"],
        help="Phase to run: '1' for data audit, '2' for preprocessing, '3' for IID clients, '4' for non-IID clients, '5' for centralized baseline, '6' for IID local-only training, '6c' for non-IID local-only training, '7' for FedAvg, '8' for FedProx, '9' for personalized federated learning (default: 5)"
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
    


if __name__ == "__main__":
    main()
