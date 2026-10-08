# HEALIX-FL

**HEALIX-FL: A Heterogeneity-Aware and Explainable Federated Learning Framework for Privacy-Preserving Healthcare Prediction**

---

## Overview

HEALIX-FL is a research framework for federated learning in healthcare settings. It addresses three core challenges:

1. **Data Heterogeneity:** Real healthcare institutions have highly non-IID patient populations. HEALIX-FL evaluates multiple heterogeneity conditions—from near-IID to extreme label skew—to understand how federated algorithms behave under realistic distribution shifts.

2. **Privacy Preservation:** The framework implements client-side differential privacy with gradient clipping and Gaussian noise, allowing measurement of the privacy–utility trade-off without sharing raw patient data.

3. **Explainability Requirement:** Healthcare models must be interpretable. HEALIX-FL integrates SHAP explainability (global and local feature attribution), efficient explainability, and provides a Streamlit prediction dashboard for manual patient-level predictions with SHAP explanations.

The project is implemented as a phased experimental pipeline. Each phase builds on the previous ones, with clear separation between data preparation, baseline training, federated algorithms, personalization, adaptive aggregation, and differential privacy.

---

## Research Problem

Healthcare prediction tasks—such as hospital readmission forecasting—require training models on sensitive patient data. Centralized training is often infeasible due to privacy regulations (e.g., HIPAA). Federated learning offers a solution by training models locally across institutions without exchanging raw data.

However, standard FL faces critical challenges in healthcare:

- **Heterogeneity:** Hospitals serve different populations, leading to highly non-IID data.
- **Class Imbalance:** Readmission events are rare, creating imbalanced learning problems.
- **Privacy Risks:** Model updates themselves can leak information, necessitating differential privacy.
- **Explainability:** Clinicians need to understand model decisions for trust and regulatory compliance.

HEALIX-FL systematically investigates these challenges through controlled experiments on a real-world diabetes readmission dataset.

---

## Research Objectives

1. Establish a reproducible preprocessing and client-partitioning pipeline for healthcare FL.
2. Evaluate standard FL baselines (FedAvg, FedProx) under varying heterogeneity.
3. Investigate adaptive aggregation that weights clients by sample count and local validation quality.
4. Implement and evaluate client-side differential privacy with gradient clipping and Gaussian noise.
5. Analyze the privacy–utility trade-off across multiple noise levels.
6. Provide a comprehensive empirical comparison with clear methodology and limitations.
7. Lay the groundwork for explainability and ablation studies in subsequent phases.

---

## Key Research Components

| Component | Description |
|-----------|-------------|
| **Data Preprocessing** | Stratified train/validation/test splits, leakage prevention, 2,370-feature representation |
| **IID Client Generation** | 10 randomly partitioned clients with near-uniform distributions |
| **Non-IID Generation** | Dirichlet-based partitioning across 5 heterogeneity conditions |
| **Heterogeneity Metrics** | Label divergence and feature distance measured per condition |
| **Centralized Baseline** | MLP upper-bound trained on pooled data |
| **Local-Only Training** | Per-client baselines without federation |
| **FedAvg** | Standard sample-count weighted federated averaging |
| **FedProx** | FedAvg with proximal term (μ = 0.01) for client-drift mitigation |
| **Personalization** | Full-model local fine-tuning on top of global model |
| **Adaptive Aggregation** | Quality-weighted aggregation using validation ROC-AUC |
| **Differential Privacy** | Client-side gradient clipping + Gaussian noise mechanism |
| **SHAP Explainability** | Phase 12: global and local SHAP feature attribution across trained models |
| **Efficient Explainability** | Phase 13: reduced-sample explanations evaluated against the baseline explainer (top-10 overlap, Spearman correlation) |
| **Phase 14 Ablation Study** | 2×2×2 factorial ablation of FedProx, adaptive aggregation and personalization across 5 heterogeneity conditions |
| **Streamlit Prediction Dashboard** | Manual patient input → model-predicted readmission probability, LOW/HIGH risk classification (threshold 0.5), and local SHAP explanation |

---

## Current Status

**COMPLETED:**

- Phase 1 — Data Audit
- Phase 2 — Data Preprocessing
- Phase 3 — IID Client Generation
- Phase 4 — Non-IID/Heterogeneity
- Phase 5 — Centralized Baseline
- Phase 6 — Local Training (IID and Non-IID)
- Phase 7 — FedAvg
- Phase 8 — FedProx
- Phase 9 — Personalization
- Phase 10 — Adaptive Aggregation
- Phase 11 — Differential Privacy
- Phase 12 — SHAP Explainability
- Phase 13 — Efficient Explainability
- Phase 14 — Ablation Study
- Streamlit Prediction Dashboard (validated)

**PLANNED:**

- Phase 15 — Final Evaluation and Reproducibility

---

## Dataset

**Source:** 10-Year Diabetes Readmission Dataset (UCI Machine Learning Repository)

**Records:** 101,766 inpatient encounters

**Features:** 50 raw columns → 2,370 features after preprocessing

**Target Variable:** `readmitted`
- Original classes: NO (53.91%), >30 (34.93%), <30 (11.16%)
- Binary formulation used in Phases 5–11: <30 → 1 (positive), NO/>30 → 0 (negative)

**Missing Values:**
- `weight`: 96.86% missing
- `max_glu_serum`: 94.75% missing
- `A1Cresult`: 83.28% missing
- `medical_specialty`: 49.08% missing
- `payer_code`: 39.56% missing

**Important Notes:**
- Two identifier columns (`encounter_id`, `patient_nbr`) are excluded from features to prevent leakage.
- Two constant columns (`examide`, `citoglipton`) are dropped during preprocessing.
- The dataset is highly imbalanced; accuracy alone is not a reliable metric.

**Dataset Availability:** The raw dataset (`diabetic_data.csv`) is NOT included in this repository due to size and data-sharing considerations. To reproduce the experiments, obtain the dataset from the UCI Machine Learning Repository and place it at `data/raw/diabetic_data.csv`.

---

## Methodology

The experimental pipeline follows this sequence:

```
Raw healthcare data
    ↓
Phase 1: Data audit
    ↓
Phase 2: Preprocessing (stratified split, encoding, scaling)
    ↓
Phase 3: IID client generation
    ↓
Phase 4: Non-IID client generation (5 heterogeneity conditions)
    ↓
Phase 5: Centralized MLP baseline
    ↓
Phase 6: Local-only training baselines
    ↓
Phase 7: FedAvg
    ↓
Phase 8: FedProx (μ = 0.01)
    ↓
Phase 9: Personalization (global + local fine-tuning)
    ↓
Phase 10: Adaptive aggregation (sample-count × validation quality)
    ↓
Phase 11: Differential privacy (gradient clipping + Gaussian noise)
    ↓
Phase 12: SHAP explainability (global + local)
    ↓
Phase 13: Efficient explainability
    ↓
Phase 14: Ablation study (2×2×2 factorial, 5 heterogeneity conditions)
    ↓
Phase 15: Final evaluation (planned)
```

---

## Federated Settings

- **Number of clients:** 10
- **Heterogeneity conditions:** 5
  - IID (Dirichlet α ≈ 1000)
  - Mild Non-IID (α = 100)
  - Moderate Non-IID (α = 10)
  - Strong Non-IID (α = 1)
  - Extreme Non-IID (α = 0.1)
- **Communication rounds:** 20
- **Local epochs:** 1
- **Client fraction:** 1.0
- **Random seed:** 42

**Extreme condition caveat:** In `extreme_non_iid`, clients 5, 8, and 9 contain insufficient or single-class data and are automatically skipped across all relevant phases.

---

## Model

All phases use the same MLP architecture:

- **Input dimension:** 2,370
- **Hidden layer 1:** 128 units
- **Hidden layer 2:** 64 units
- **Dropout:** 0.2
- **Optimizer:** Adam
- **Loss:** BCEWithLogitsLoss with positive class weighting (`pos_weight = negative_count / positive_count`)
- **Learning rate:** 0.001
- **Batch size:** 256

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Device | CPU |
| Random seed | 42 |
| Learning rate | 0.001 |
| Batch size | 256 |
| Local epochs | 1 |
| Federated rounds | 20 |
| Client fraction | 1.0 |
| Validation split | 20% of training data |
| Model selection | Validation ROC-AUC |
| Test evaluation | Once, after best round selection |

---

## Differential Privacy

Phase 11 adds client-side differential privacy to the federated pipeline.

**Mechanism (per local training step):**
1. Forward pass → loss
2. Backward pass
3. Gradient clipping to maximum norm C = 1.0
4. Add Gaussian noise with standard deviation `noise_std = sigma × C`
5. Optimizer step

**Privacy settings evaluated:**
- `no_dp` (sigma = 0.0): No-DP baseline
- `sigma_0.5` (sigma = 0.5): Weak noise
- `sigma_1.0` (sigma = 1.0): Moderate noise
- `sigma_2.0` (sigma = 2.0): Strong noise

**Fixed parameters:**
- Clipping norm C = 1.0
- Delta = 1e-5

**Important Limitation:** The current implementation does NOT provide a formally verified differential privacy guarantee. No privacy-accounting library (e.g., Opacus) is currently integrated, and no formal epsilon value is computed or claimed. The noise mechanism is a standard heuristic approach. Formal DP accounting remains future work.

---

## Results

Detailed results are available in `results/` and summarized in `reports/PHASE_1_11_RESEARCH_PROGRESS_REPORT.md`.

**Centralized Baseline (Phase 5):**
- ROC-AUC: 0.6627
- PR-AUC: 0.2131
- F1: 0.2619

**FedAvg (Phase 7, validation-selected):**
- Best ROC-AUC: 0.6540 (IID) to 0.6234 (extreme_non_iid)

**FedProx (Phase 8, μ = 0.01, validation-selected):**
- Best ROC-AUC: 0.6584 (IID) to 0.6589 (extreme_non_iid)

**Adaptive Aggregation (Phase 10, validation-selected):**
- Best ROC-AUC: 0.6640 (IID) to 0.6554 (extreme_non_iid)

**Differential Privacy (Phase 11, validation-selected):**
- No-DP: ROC-AUC 0.6335–0.6665
- sigma=0.5: ROC-AUC 0.5184–0.5570
- sigma=1.0: ROC-AUC 0.5326–0.5398
- sigma=2.0: ROC-AUC 0.5192–0.5517

**Privacy–Utility Observation:** Increasing noise consistently degrades predictive utility. At sigma = 2.0, some conditions show near-random performance. This expected trade-off demonstrates the cost of noise injection but does not constitute a formal privacy guarantee.

---

## Repository Structure

```
PYTHON PROJECT/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config.yaml
├── main.py
│
├── dashboard/
│   └── app.py                             # Streamlit prediction dashboard
│
├── data/
│   ├── raw/
│   │   └── diabetic_data.csv          # NOT in Git (see .gitignore)
│   ├── processed/
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   ├── X_test.npy
│   │   └── y_test.npy
│   └── clients/
│       ├── iid/
│       │   └── client_1/ ... client_10/
│       └── non_iid/
│           ├── iid/
│           ├── mild_non_iid/
│           ├── moderate_non_iid/
│           ├── strong_non_iid/
│           └── extreme_non_iid/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── audit.py
│   │   ├── preprocess.py
│   │   └── partition.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py
│   │   └── train.py
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── local.py
│   │   ├── fedavg.py
│   │   ├── fedprox.py
│   │   ├── personalized.py
│   │   ├── adaptive_aggregation.py
│   │   └── differential_privacy.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── explainability/
│       └── __init__.py
│
├── results/
│   ├── centralized/
│   ├── local/
│   ├── fedavg/
│   ├── fedprox/
│   ├── personalized/
│   ├── adaptive_aggregation/
│   ├── differential_privacy/
│   ├── shap/                              # Phase 12–13 SHAP results
│   ├── phase14_ablation/                  # Phase 14 experiment outputs
│   ├── phase14_analysis/                  # Phase 14 report CSVs and figures
│   ├── prediction/                        # dashboard SHAP plots
│   ├── phase4_plots/
│   ├── data_audit.csv
│   ├── heterogeneity_summary.csv
│   └── ... (experiment outputs)
│
├── reports/
│   ├── PHASE_1_11_RESEARCH_PROGRESS_REPORT.md
│   ├── PHASE_12_SHAP_EXPLAINABILITY_REPORT.md
│   ├── PHASE_13_EFFICIENT_EXPLAINABILITY_REPORT.md
│   ├── phase14/
│   │   └── PHASE_14_ABLATION_REPORT.md
│   ├── dataset_audit/
│   ├── figures/
│   └── tables/
│
├── experiments/                     # Reserved for future experiment configs
│   ├── ablation/
│   ├── baseline/
│   ├── explainability/
│   ├── heterogeneity/
│   ├── personalization/
│   └── privacy/
│
├── config.yaml
├── main.py
└── requirements.txt
```

---

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

The project is run via `main.py` with phase selection:

```bash
# Phase 1: Data audit
python main.py 1

# Phase 2: Preprocessing
python main.py 2

# Phase 3: IID client generation
python main.py 3

# Phase 4: Non-IID client generation
python main.py 4

# Phase 5: Centralized baseline
python main.py 5

# Phase 6: IID local-only training
python main.py 6

# Phase 6C: Non-IID local-only training
python main.py 6c

# Phase 7: FedAvg
python main.py 7

# Phase 8: FedProx
python main.py 8

# Phase 9: Personalization
python main.py 9
```

**Note:** Phases 10–11 are currently run as standalone modules (e.g., `python -m src.federated.adaptive_aggregation` and `python -m src.federated.differential_privacy`). Integration into `main.py` is planned.

---

## Streamlit Prediction Dashboard

The validated dashboard in `dashboard/app.py` allows entering a patient's values manually and sending them through the EXISTING HEALIX-FL preprocessing pipeline and trained model. It produces:

- a **model-predicted readmission probability** (displayed as a percentage),
- a **LOW RISK / HIGH RISK classification** using the project threshold of 0.5,
- a **local SHAP explanation** showing which features contributed to the prediction.

The dashboard also displays the existing Phase 12, Phase 13 and Phase 14 research results. It does not retrain the model and does not re-run any experiment.

```bash
# Launch the dashboard
python -m streamlit run dashboard/app.py
```

---

## Reproducibility

- **Random seed:** 42 (fixed across all phases)
- **Configuration:** `config.yaml` contains all hyperparameters and paths
- **Model selection:** Best checkpoint selected using validation ROC-AUC only
- **Test evaluation:** Performed once after best round identification; test metrics never influence training or selection
- **Results:** All experiment outputs saved to `results/` as CSV, YAML, and PyTorch model files
- **Reports:** Phase 1–11 in `reports/PHASE_1_11_RESEARCH_PROGRESS_REPORT.md`; Phase 12 in `reports/PHASE_12_SHAP_EXPLAINABILITY_REPORT.md`; Phase 13 in `reports/PHASE_13_EFFICIENT_EXPLAINABILITY_REPORT.md`; Phase 14 in `reports/phase14/PHASE_14_ABLATION_REPORT.md`

---

## Limitations

- **Class Imbalance:** The readmission target is highly imbalanced (~11% positive), limiting maximum F1 and precision.
- **Extreme Non-IID:** Some clients in the extreme condition have insufficient or single-class data and must be skipped.
- **Single Dataset:** Results are based on one dataset; generalization to other healthcare tasks is not established.
- **MLP Baseline:** A single architecture is used; more advanced models are not explored.
- **No Formal Privacy Accountant:** Phase 11 implements a heuristic DP mechanism but does not provide a provable (ε, δ)-DP guarantee.
- **Personalization Metrics:** Phase 9 uses client-level macro averaging, which is not directly comparable with pooled global metrics from other phases.
- **Single Seed:** All experiments use one random seed; variance across initializations is not quantified.

---

## Disclaimer

This is a research and academic framework. It is not intended for clinical diagnosis or patient care decisions. All data should be handled in accordance with applicable privacy regulations and institutional review board requirements.

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Audit | ✅ Completed |
| 2 | Data Preprocessing | ✅ Completed |
| 3 | IID Client Generation | ✅ Completed |
| 4 | Non-IID/Heterogeneity | ✅ Completed |
| 5 | Centralized Baseline | ✅ Completed |
| 6 | Local Training | ✅ Completed |
| 7 | FedAvg | ✅ Completed |
| 8 | FedProx | ✅ Completed |
| 9 | Personalization | ✅ Completed |
| 10 | Adaptive Aggregation | ✅ Completed |
| 11 | Differential Privacy | ✅ Completed |
| 12 | SHAP Explainability | ✅ Completed |
| 13 | Efficient Explainability | ✅ Completed |
| 14 | Ablation Study | ✅ Completed |
| 15 | Final Evaluation | 🔄 Planned |
| — | Streamlit Prediction Dashboard | ✅ Completed |

---

## License

[Add appropriate license here]
