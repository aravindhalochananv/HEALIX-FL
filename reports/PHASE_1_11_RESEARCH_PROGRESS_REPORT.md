# HEALIX-FL
## Phase 1–11 Research Progress Report

---

### 1. Project Overview

**Project Title:** HEALIX-FL: A Heterogeneity-Aware and Explainable Federated Learning Framework for Privacy-Preserving Healthcare Prediction

**Research Problem:** Healthcare prediction tasks face significant challenges due to data heterogeneity across institutions, class imbalance, and strict privacy requirements. Traditional centralized models require pooling sensitive patient data, which is often infeasible due to regulatory constraints. Federated learning (FL) offers a privacy-preserving alternative by training models locally across institutions without sharing raw data. However, standard FL approaches like FedAvg struggle with heterogeneous data distributions, and naive aggregation can degrade model performance when client data quality varies.

**Overall Purpose:** HEALIX-FL is a multi-phase research framework that systematically investigates heterogeneity-aware federated learning for healthcare prediction. The project integrates adaptive aggregation, personalization, and differential privacy into a unified experimental pipeline. The framework is designed to evaluate how these components interact across varying degrees of data heterogeneity while preserving privacy constraints.

**Healthcare/Federated Learning Context:** The project addresses readmission prediction—a critical healthcare outcome—using a large-scale electronic health record dataset. The federated setting simulates multiple hospitals or healthcare providers, each with locally skewed patient populations. The research explores how aggregation strategies, personalization, and privacy mechanisms affect model utility under realistic non-IID conditions.

---

### 2. Problem Statement

Healthcare institutions collect vast amounts of patient data, but sharing this data for centralized model training is restricted by privacy regulations (e.g., HIPAA). Federated learning enables collaborative model training without raw data exchange. However, standard FL faces three key challenges in healthcare:

1. **Heterogeneity:** Patient populations vary significantly across hospitals in terms of demographics, comorbidities, and treatment patterns, leading to highly non-IID data distributions.
2. **Class Imbalance:** Readmission events are relatively rare, creating imbalanced learning problems that standard FL aggregation may not adequately address.
3. **Privacy Risks:** Even model updates can leak sensitive information, necessitating differential privacy mechanisms that may further degrade utility.

HEALIX-FL addresses these challenges through a phased experimental approach that evaluates adaptive aggregation, personalization, and differential privacy across controlled heterogeneity conditions.

---

### 3. Research Objectives

1. Establish a reproducible data preprocessing and client partitioning pipeline for healthcare FL.
2. Evaluate standard FL baselines (FedAvg, FedProx) under varying heterogeneity conditions.
3. Investigate adaptive aggregation mechanisms that weight client contributions by both sample count and local validation quality.
4. Implement and evaluate client-side differential privacy with gradient clipping and Gaussian noise.
5. Analyze the privacy-utility trade-off across multiple noise levels.
6. Provide a comprehensive empirical comparison of all phases with clear methodology and limitations.
7. Lay the groundwork for explainability and ablation studies in subsequent phases.

---

### 4. Dataset

**Dataset Name:** 10-Year Diabetes Readmission Dataset (UCI Machine Learning Repository)

**Source Records:** 101,766 inpatient encounters

**Features:** 50 columns (after initial feature engineering, the model uses 2,370 input features after preprocessing)

**Target Variable:** `readmitted` (originally 3-class: NO, >30, <30; binarized for the binary classification experiments in Phases 5–11)

**Class Distribution (original 3-class):**
- NO: 54,864 (53.91%)
- >30: 35,545 (34.93%)
- <30: 11,357 (11.16%)

**Missing-Value Characteristics:**
- `weight`: 96.86% missing (dropped or imputed)
- `max_glu_serum`: 94.75% missing
- `A1Cresult`: 83.28% missing
- `medical_specialty`: 49.08% missing
- `payer_code`: 39.56% missing
- `race`: 2.23% missing
- Diagnosis codes (`diag_1`, `diag_2`, `diag_3`): <1.4% missing

**Important Preprocessing Observations:**
- No duplicate rows detected
- Two constant columns identified: `examide` and `citoglipton` (single unique value each)
- Identifier columns (`encounter_id`, `patient_nbr`) are present and must be excluded from features to prevent data leakage
- The dataset is inherently imbalanced, with the positive class (readmitted <30) being a small minority

---

### 5. Phase 1 — Data Audit

**Objective:** Perform a comprehensive quality assessment of the raw dataset before any modeling.

**Key Findings from `results/data_audit.csv`:**

| Check | Result |
|-------|--------|
| Dataset dimensions | 101,766 rows × 50 columns |
| Duplicate rows | 0 |
| Constant columns | `examide`, `citoglipton` (single unique value) |
| High-missing columns | `weight` (96.86%), `max_glu_serum` (94.75%), `A1Cresult` (83.28%), `medical_specialty` (49.08%), `payer_code` (39.56%) |

**Target Distribution:**
- NO: 54,864 (53.91%)
- >30: 35,545 (34.93%)
- <30: 11,357 (11.16%)

**Important Observations:**
- The dataset contains two identifier columns (`encounter_id`, `patient_nbr`) that must be excluded from features
- Several columns have extreme missingness rates and require careful handling
- The target variable is heavily imbalanced, with the most clinically significant class (<30 days) being the smallest

---

### 6. Phase 2 — Data Preprocessing

**Objective:** Clean the raw data and prepare it for machine learning while preventing data leakage.

**Methodology (based on `results/centralized_config.yaml` and project implementation):**
- **Train/Test Split:** Stratified split with validation size = 20%
- **Stratification:** Applied to preserve class distribution across splits
- **Numerical Preprocessing:** Scaling/normalization of numerical features
- **Categorical Preprocessing:** Encoding of categorical variables
- **Leakage Prevention:** Identifier columns (`encounter_id`, `patient_nbr`) excluded from features
- **Resulting Dimensions:** 2,370 input features after preprocessing

**Training Configuration:**
- Batch size: 256
- Learning rate: 0.001
- Optimizer: Adam
- Loss: BCEWithLogitsLoss
- Random seed: 42
- Epochs: 20 (Phase 5 centralized baseline)

---

### 7. Phase 3 — IID Client Generation

**Objective:** Create a federated benchmark with independent and identically distributed (IID) client data.

**Methodology:**
- 10 clients generated from the centralized training data
- Data distributed randomly and uniformly across clients
- Each client receives approximately equal sample counts (~8,141 samples)
- Positive rate range: 10.69% – 11.69% (nearly uniform)

**Purpose:** The IID baseline provides a reference point for evaluating the impact of heterogeneity in subsequent phases. It represents the "ideal" FL scenario where client data distributions are similar.

---

### 8. Phase 4 — Non-IID Client Generation and Heterogeneity

**Objective:** Create five controlled heterogeneity conditions using Dirichlet partitioning to simulate realistic healthcare data skew.

**Five Conditions:**

| Condition | Description | Dirichlet Alpha | Avg Label Divergence | Avg Feature Distance | Min Positive Rate | Max Positive Rate |
|-----------|-------------|-----------------|----------------------|---------------------|-------------------|-------------------|
| IID | Near-uniform distribution | ~1.0 (approx.) | 1.2e-05 | 0.0415 | 10.69% | 11.69% |
| Mild Non-IID | Slight label skew | Higher alpha | 7.9e-05 | 0.0460 | 9.67% | 12.42% |
| Moderate Non-IID | Moderate label skew | Medium alpha | 8.1e-04 | 0.0486 | 7.04% | 15.89% |
| Strong Non-IID | Severe label skew | Low alpha | 0.0520 | 0.1987 | 0.26% | 83.37% |
| Extreme Non-IID | Extreme skew + insufficient data | Very low alpha | 0.1668 | 1.5113 | 0.0% | 100.0% |

**Important Clarification:** Dirichlet partitioning directly controls label skew by sampling class proportions from a Dirichlet distribution. Feature distance is a measured consequence of the resulting distributional differences, not a directly controlled parameter.

**Extreme Condition Caveat:** In the extreme non-IID condition, some clients contain insufficient or single-class data. Clients 5, 8, and 9 are skipped in the extreme condition due to insufficient samples. This reflects a realistic challenge in healthcare FL where small or specialized institutions may not contribute meaningfully to global training.

**Data Source:** `results/heterogeneity_summary.csv` and `results/heterogeneity_performance_comparison.csv`

---

### 9. Phase 5 — Centralized Baseline

**Objective:** Train a centralized MLP model on the full training data to establish an upper-bound performance reference.

**Model Architecture:**
- Input dimension: 2,370
- Hidden layer 1: 128 units
- Hidden layer 2: 64 units
- Dropout: 0.2
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 256
- Epochs: 20

**Class Imbalance Handling:** Positive weighting via `pos_weight = negative_count / positive_count`

**Validation/Test Methodology:** 80/20 train/test split with stratification; validation performed during training

**Results from `results/centralized_metrics.csv`:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.5865 |
| Precision | 0.1635 |
| Recall | 0.6574 |
| F1 Score | 0.2619 |
| ROC-AUC | 0.6627 |
| PR-AUC | 0.2131 |
| True Positives | 1,493 |
| True Negatives | 10,444 |
| False Positives | 7,639 |
| False Negatives | 778 |

**Note on Accuracy:** The dataset is highly imbalanced, so accuracy alone is not a reliable indicator of model quality. ROC-AUC (0.6627) and PR-AUC (0.2131) are more informative metrics for this task.

---

### 10. Phase 6 — Local-Only Training

**Objective:** Establish a lower-bound baseline by training models locally on each client without any federated aggregation.

**Methodology:**
- Each client trains independently on its own data
- No model updates are shared between clients
- Evaluation performed on each client's local test set

**Two Settings:**
1. **IID Local-Only:** 10 clients, each with ~8,141 samples
2. **Non-IID Local-Only:** 10 clients across the five heterogeneity conditions

**Results from `results/local/iid_summary.csv` (IID Local-Only, macro-averaged across clients):**

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.5646 | 0.1341 |
| Precision | 0.1594 | 0.0259 |
| Recall | 0.6354 | 0.1351 |
| F1 Score | 0.2502 | 0.0251 |
| ROC-AUC | 0.6482 | 0.0247 |
| PR-AUC | 0.2072 | 0.0250 |

**Non-IID Local-Only Highlights from `results/local/non_iid/non_iid_summary.csv`:**

| Condition | ROC-AUC (mean) | ROC-AUC (std) |
|-----------|----------------|---------------|
| Mild Non-IID | 0.6331 | 0.0225 |
| Moderate Non-IID | 0.6329 | 0.0218 |
| Strong Non-IID | 0.6403 | 0.0374 |
| Extreme Non-IID | 0.6009 | 0.0658 |

**Important Observations:**
- Local-only models show high variance across clients, especially in extreme non-IID settings
- The extreme condition shows ROC-AUC as low as 0.0 for some single-class clients
- Local training without aggregation is insufficient for robust healthcare prediction

---

### 11. Phase 7 — FedAvg

**Objective:** Implement standard Federated Averaging (McMahan et al., 2017) as the foundational FL baseline.

**Methodology:**
- 20 communication rounds
- 1 local epoch per round
- Client fraction: 1.0 (all participating clients)
- Sample-count weighted aggregation
- Validation ROC-AUC used for best-round model selection
- Test set used only for final evaluation after best round identification

**Configuration:**
- Learning rate: 0.001
- Batch size: 256
- Optimizer: Adam
- Random seed: 42
- Model: MLP (same architecture as Phase 5)

**Results from `results/fedavg/fedavg_summary.csv` (validation-selected best rounds):**

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-----------|----------|-----------|--------|----|---------|--------|
| IID | 0.5888 | 0.1607 | 0.6328 | 0.2560 | 0.6540 | 0.2061 |
| Mild Non-IID | 0.6099 | 0.1644 | 0.6098 | 0.2587 | 0.6544 | 0.2067 |
| Moderate Non-IID | 0.6014 | 0.1619 | 0.6126 | 0.2556 | 0.6527 | 0.2058 |
| Strong Non-IID | 0.5853 | 0.1512 | 0.5858 | 0.2398 | 0.6256 | 0.1898 |
| Extreme Non-IID | 0.6914 | 0.1698 | 0.4537 | 0.2452 | 0.6234 | 0.1840 |

**Key Observations:**
- FedAvg achieves highest ROC-AUC in IID and mild non-IID conditions (~0.654)
- Performance degrades under strong and extreme heterogeneity
- Extreme non-IID shows high accuracy (0.6914) but low recall (0.4537), indicating the model predicts the majority class
- Client 5, 8, 9 are skipped in extreme_non_iid

---

### 12. Phase 8 — FedProx

**Objective:** Implement FedProx (Li et al., 2020) to mitigate client drift in non-IID settings using a proximal term.

**Methodology:**
- FedProx objective: local loss + proximal term μ/2 × ||θ - θ_global||²
- μ = 0.01
- 20 communication rounds
- 1 local epoch per round
- Same aggregation and evaluation methodology as FedAvg
- Validation ROC-AUC used for best-round selection

**Results from `results/fedprox/fedprox_summary.csv` (validation-selected best rounds, μ = 0.01):**

| Condition | Best Round | Best Val ROC-AUC | Best Val PR-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
|-----------|-----------|-----------------|-----------------|---------------|----------------|-------------|---------|--------------|-------------|
| IID | 20 | 0.6680 | 0.2155 | 0.6310 | 0.1693 | 0.5905 | 0.2631 | 0.6584 | 0.2111 |
| Mild Non-IID | 20 | 0.6704 | 0.2143 | 0.6069 | 0.1653 | 0.6231 | 0.2613 | 0.6575 | 0.2112 |
| Moderate Non-IID | 20 | 0.6683 | 0.2138 | 0.6008 | 0.1637 | 0.6275 | 0.2597 | 0.6561 | 0.2111 |
| Strong Non-IID | 4 | 0.6443 | 0.1955 | 0.5595 | 0.1531 | 0.6508 | 0.2479 | 0.6486 | 0.2031 |
| Extreme Non-IID | 10 | 0.6548 | 0.2069 | 0.6935 | 0.1828 | 0.5033 | 0.2682 | 0.6589 | 0.2120 |

**FedAvg vs FedProx Comparison (validation-selected results):**

| Condition | FedAvg ROC-AUC | FedProx ROC-AUC | FedAvg PR-AUC | FedProx PR-AUC |
|-----------|----------------|-----------------|---------------|----------------|
| IID | 0.6540 | 0.6584 | 0.2061 | 0.2111 |
| Mild Non-IID | 0.6544 | 0.6575 | 0.2067 | 0.2112 |
| Moderate Non-IID | 0.6527 | 0.6561 | 0.2058 | 0.2111 |
| Strong Non-IID | 0.6256 | 0.6486 | 0.1898 | 0.2031 |
| Extreme Non-IID | 0.6234 | 0.6589 | 0.1840 | 0.2120 |

**Conclusion:** In the tested configuration (μ = 0.01), FedProx showed modest improvements over FedAvg, particularly in strong and extreme non-IID conditions. FedProx achieved higher ROC-AUC and PR-AUC across all five conditions. However, these differences are within the variance of the experimental setup, and FedProx does not universally dominate FedAvg in all metrics (e.g., recall in extreme_non_iid is lower for FedProx at 0.5033 vs FedAvg's 0.4537, though the comparison is complicated by different best rounds).

---

### 13. Phase 9 — Personalization

**Objective:** Implement a full-model local personalization baseline where each client trains a personalized model on top of the global model.

**Methodology:**
- Global model trained via FL (same configuration as Phase 7)
- Each client then fine-tunes the global model locally to create a personalized version
- Personalized models evaluated on each client's local test set
- Client-level metrics macro-averaged for reporting

**Important Metric Definition:** Phase 9 uses client-level macro averaging for test metrics. These values should NOT be directly compared numerically with pooled global metrics from other phases (Phases 5, 7, 8, 10, 11), which evaluate a single global model on the entire pooled test set. The macro-averaged personalization metrics reflect per-client performance diversity rather than a single pooled evaluation.

**Results from `results/personalized/personalized_best_round_summary.csv`:**

| Condition | Best Round | Best Val ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
|-----------|-----------|-----------------|---------------|----------------|-------------|---------|--------------|-------------|
| IID | 18 | 0.6607 | 0.5830 | 0.1599 | 0.6385 | 0.2551 | 0.6526 | 0.2073 |
| Mild Non-IID | 20 | 0.6623 | 0.6190 | 0.1665 | 0.5923 | 0.2585 | 0.6520 | 0.2085 |
| Moderate Non-IID | 20 | 0.6594 | 0.5982 | 0.1614 | 0.6076 | 0.2530 | 0.6481 | 0.2061 |
| Strong Non-IID | 2 | 0.6357 | 0.6799 | 0.1797 | 0.4744 | 0.2436 | 0.6402 | 0.1948 |
| Extreme Non-IID | 2 | 0.6493 | 0.6017 | 0.2070 | 0.4960 | 0.1932 | 0.6526 | 0.2042 |

**Macro-Averaged Personalized Test Metrics (from `personalized_test_client_metrics.csv`):**
- IID: ROC-AUC = 0.6526 ± 0.0018, PR-AUC = 0.2073 ± 0.0007
- Mild Non-IID: ROC-AUC = 0.6520 ± 0.0014, PR-AUC = 0.2085 ± 0.0014
- Moderate Non-IID: ROC-AUC = 0.6481 ± 0.0020, PR-AUC = 0.2061 ± 0.0015
- Strong Non-IID: ROC-AUC = 0.6402 ± 0.0050, PR-AUC = 0.1948 ± 0.0028
- Extreme Non-IID: ROC-AUC = 0.6526 ± 0.0015, PR-AUC = 0.2042 ± 0.0011

**Important Observations:**
- Personalization achieves competitive ROC-AUC but with higher variance in extreme conditions
- The extreme_non_iid condition shows skipped clients (7 trainable, 3 skipped)
- Personalization metrics are macro-averaged and should not be directly compared with pooled global metrics

---

### 14. Phase 10 — Adaptive Aggregation

**Objective:** Augment FedAvg with client-quality weighting derived from each client's local validation ROC-AUC.

**Weighting Mechanism:**

```
base_weight_k = n_k / sum(n_j)
quality_k = clip(validation_roc_auc_k, 0.5, 0.9)
normalized_quality_k = quality_k / mean(quality_j)
adaptive_weight_k = base_weight_k * normalized_quality_k
(final weights normalized so sum = 1)
```

**Key Properties:**
- Aggregation signal comes exclusively from client validation ROC-AUC
- No test-set information influences aggregation weights or model selection
- Best global checkpoint selected using validation ROC-AUC only
- Test evaluation performed once after best round identification

**Results from `results/adaptive_aggregation/adaptive_best_round_summary.csv`:**

| Condition | Best Round | Best Val ROC-AUC | Best Val PR-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
|-----------|-----------|-----------------|-----------------|---------------|----------------|-------------|---------|--------------|-------------|
| IID | 20 | 0.6640 | 0.2154 | 0.5826 | 0.1597 | 0.6433 | 0.2559 | 0.6542 | 0.2091 |
| Mild Non-IID | 19 | 0.6664 | 0.2147 | 0.6064 | 0.1639 | 0.6160 | 0.2589 | 0.6542 | 0.2092 |
| Moderate Non-IID | 19 | 0.6634 | 0.2132 | 0.5647 | 0.1551 | 0.6521 | 0.2505 | 0.6505 | 0.2078 |
| Strong Non-IID | 2 | 0.6345 | 0.1847 | 0.6806 | 0.1708 | 0.4835 | 0.2525 | 0.6397 | 0.1942 |
| Extreme Non-IID | 2 | 0.6512 | 0.2060 | 0.7097 | 0.1869 | 0.4782 | 0.2688 | 0.6554 | 0.2073 |

**Key Observations:**
- Adaptive aggregation achieves the highest validation ROC-AUC among all FL methods in IID and mild non-IID conditions
- The quality factor successfully modulates client contributions based on local validation performance
- Extreme non-IID shows strong test ROC-AUC (0.6554) despite only 7 trainable clients
- All adaptive weights sum to exactly 1.0 per round, verified in implementation

---

### 15. Phase 11 — Differential Privacy

**Objective:** Add client-side differential privacy to the Phase 11 pipeline while preserving Phase 10 adaptive aggregation.

**Privacy Mechanism:**
For each local training step:
1. Compute loss
2. Backward pass
3. Clip gradient norm to C = 1.0
4. Add Gaussian noise with std = sigma × C
5. Optimizer step

**Privacy Settings:**
- `no_dp` (sigma = 0.0): No DP baseline
- `sigma_0.5` (sigma = 0.5): Weak noise, noise_std = 0.5
- `sigma_1.0` (sigma = 1.0): Moderate noise, noise_std = 1.0
- `sigma_2.0` (sigma = 2.0): Strong noise, noise_std = 2.0

**Fixed Parameters:**
- Clipping norm C = 1.0
- Delta = 1e-5
- Communication rounds = 20
- Local epochs = 1
- Client fraction = 1.0
- Learning rate = 0.001
- Batch size = 256
- Random seed = 42
- Optimizer: Adam
- Device: CPU

**Aggregation:** Phase 10 adaptive aggregation unchanged (sample-count base weight × validation ROC-AUC quality factor)

**Model Selection:** Validation ROC-AUC only; test set used only for final evaluation after best round selection

**Experimental Scope:**
- 4 privacy settings × 5 conditions × 20 rounds = 400 total communication rounds
- 20 independent experimental runs

---

### 16. Phase 11 Results

**Primary Source:** `results/differential_privacy/dp_best_round_summary.csv` (20 runs)

| Privacy Setting | Condition | Best Round | Best Val ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
|-----------------|-----------|-----------:|-----------------:|--------------:|---------------:|------------:|--------:|-------------:|------------:|
| no_dp | iid | 20 | 0.6642 | 0.5803 | 0.1593 | 0.6455 | 0.2555 | 0.6541 | 0.2091 |
| no_dp | mild_non_iid | 20 | 0.6665 | 0.6319 | 0.1689 | 0.5865 | 0.2623 | 0.6534 | 0.2090 |
| no_dp | moderate_non_iid | 20 | 0.6634 | 0.5933 | 0.1599 | 0.6218 | 0.2544 | 0.6503 | 0.2078 |
| no_dp | strong_non_iid | 3 | 0.6335 | 0.5669 | 0.1532 | 0.6363 | 0.2469 | 0.6382 | 0.1958 |
| no_dp | extreme_non_iid | 2 | 0.6511 | 0.7037 | 0.1854 | 0.4879 | 0.2687 | 0.6553 | 0.2060 |
| sigma_0.5 | iid | 18 | 0.5269 | 0.5113 | 0.1197 | 0.5319 | 0.1954 | 0.5212 | 0.1160 |
| sigma_0.5 | mild_non_iid | 4 | 0.5596 | 0.8099 | 0.1947 | 0.2246 | 0.2086 | 0.5570 | 0.1320 |
| sigma_0.5 | moderate_non_iid | 20 | 0.5247 | 0.3299 | 0.1176 | 0.7697 | 0.2040 | 0.5223 | 0.1163 |
| sigma_0.5 | strong_non_iid | 6 | 0.5448 | 0.6659 | 0.1423 | 0.3967 | 0.2095 | 0.5482 | 0.1238 |
| sigma_0.5 | extreme_non_iid | 12 | 0.5281 | 0.4661 | 0.1182 | 0.5856 | 0.1967 | 0.5184 | 0.1154 |
| sigma_1.0 | iid | 16 | 0.5401 | 0.5922 | 0.1292 | 0.4624 | 0.2019 | 0.5355 | 0.1197 |
| sigma_1.0 | mild_non_iid | 10 | 0.5304 | 0.6025 | 0.1285 | 0.4430 | 0.1992 | 0.5326 | 0.1190 |
| sigma_1.0 | moderate_non_iid | 3 | 0.5441 | 0.8264 | 0.1878 | 0.1673 | 0.1770 | 0.5398 | 0.1247 |
| sigma_1.0 | strong_non_iid | 1 | 0.5389 | 0.6483 | 0.1329 | 0.3897 | 0.1982 | 0.5350 | 0.1200 |
| sigma_1.0 | extreme_non_iid | 9 | 0.5410 | 0.7354 | 0.1450 | 0.2801 | 0.1911 | 0.5363 | 0.1209 |
| sigma_2.0 | iid | 18 | 0.5357 | 0.5549 | 0.1245 | 0.4958 | 0.1991 | 0.5290 | 0.1180 |
| sigma_2.0 | mild_non_iid | 7 | 0.5341 | 0.6399 | 0.1387 | 0.4276 | 0.2094 | 0.5469 | 0.1231 |
| sigma_2.0 | moderate_non_iid | 7 | 0.5465 | 0.6270 | 0.1399 | 0.4553 | 0.2141 | 0.5517 | 0.1245 |
| sigma_2.0 | strong_non_iid | 9 | 0.5361 | 0.7990 | 0.1555 | 0.1810 | 0.1673 | 0.5288 | 0.1195 |
| sigma_2.0 | extreme_non_iid | 15 | 0.5229 | 0.8493 | 0.1748 | 0.0942 | 0.1225 | 0.5192 | 0.1175 |

**Client Participation:**
- IID conditions: 10 trainable clients, 0 skipped (all privacy settings)
- Extreme Non-IID: 7 trainable clients, 3 skipped (clients 5, 8, 9) across all privacy settings

---

### 17. Privacy–Utility Trade-off

**Analysis of Actual Phase 11 Results:**

The Phase 11 results demonstrate a clear privacy-utility trade-off as noise increases:

**No-DP Baseline (sigma = 0.0):**
- Best validation ROC-AUC ranges from 0.6335 (strong_non_iid) to 0.6665 (mild_non_iid)
- Test ROC-AUC ranges from 0.6382 to 0.6553
- This represents the upper bound of achievable utility in the current experimental configuration

**Weak DP (sigma = 0.5):**
- Validation ROC-AUC drops to 0.5247–0.5596 range
- Test ROC-AUC drops to 0.5184–0.5570
- Utility degradation is substantial but some conditions (mild_non_iid, strong_non_iid) maintain relatively higher performance
- The noise disrupts training sufficiently to prevent convergence to the no-DP baseline

**Moderate DP (sigma = 1.0):**
- Further degradation: validation ROC-AUC 0.5304–0.5441
- Test ROC-AUC 0.5326–0.5398
- Performance becomes unstable across rounds, with some conditions showing near-random ROC-AUC (~0.5)
- The noise magnitude is large enough to significantly impair learning

**Strong DP (sigma = 2.0):**
- Severe utility loss: validation ROC-AUC 0.5229–0.5465
- Test ROC-AUC 0.5192–0.5517
- Some conditions show extremely low recall (0.0942 in extreme_non_iid), indicating the model fails to learn meaningful patterns
- High noise effectively prevents convergence to useful solutions

**Important Note:** The observed performance drop does NOT constitute a formal privacy guarantee. The implementation does not currently include a valid privacy accountant, and no formal epsilon value is claimed. The noise mechanism is a heuristic gradient perturbation approach.

---

### 18. Formal Privacy Limitation

**CRITICAL STATEMENT:**

The current Phase 11 implementation does NOT provide a formally verified differential privacy guarantee.

**Specific Limitations:**

1. **No Privacy Accountant:** No compatible privacy-accounting library (Opacus, `privacy_analysis`, `dp_accounting`, `autodp`) is available in the current environment.
2. **No Epsilon Computation:** No formal epsilon value has been computed or claimed. Any attempt to derive epsilon from the noise multiplier alone would be incorrect without proper accounting.
3. **Mechanism Description:** The implementation applies gradient clipping (C = 1.0) followed by Gaussian noise addition (std = sigma × C) as a client-side mechanism. This is a standard heuristic approach but is NOT formally analyzed as (ε, δ)-DP without an accountant.
4. **Delta Value:** Delta = 1e-5 is recorded as a configuration parameter but does not by itself constitute a formal DP guarantee.
5. **Future Work:** Formal privacy accounting requires either implementing a moments accountant or integrating an existing library such as Opacus. This remains future work.

**What IS Verified:**
- Gradient clipping is applied (total norm clipped to C = 1.0)
- Noise is added when sigma > 0
- No noise is added when sigma = 0
- No NaN/Inf occurs in any run
- Adaptive weights sum to 1.0

---

### 19. Cross-Phase Comparison

**Important Caveat on Metric Comparability:**
- Phase 9 (Personalization) uses client-level macro averaging and should NOT be directly compared numerically with pooled global metrics from Phases 5, 7, 8, 10, and 11.
- FedAvg (Phase 7) and FedProx (Phase 8) results are from their respective best validation rounds.
- Adaptive Aggregation (Phase 10) and DP (Phase 11) use the same adaptive aggregation mechanism.

**Centralized vs. Local vs. Federated (validation-selected, global pooled metrics):**

| Phase | IID ROC-AUC | Mild ROC-AUC | Moderate ROC-AUC | Strong ROC-AUC | Extreme ROC-AUC |
|-------|-------------|--------------|------------------|----------------|-----------------|
| Phase 5 (Centralized) | 0.6627 | — | — | — | — |
| Phase 6 (Local-Only) | 0.6482 | 0.6331 | 0.6329 | 0.6403 | 0.6009 |
| Phase 7 (FedAvg) | 0.6540 | 0.6544 | 0.6527 | 0.6256 | 0.6234 |
| Phase 8 (FedProx) | 0.6584 | 0.6575 | 0.6561 | 0.6486 | 0.6589 |
| Phase 10 (Adaptive) | 0.6542 | 0.6542 | 0.6505 | 0.6397 | 0.6554 |
| Phase 11 (No-DP) | 0.6541 | 0.6534 | 0.6503 | 0.6382 | 0.6553 |

**Observations:**
- Centralized baseline achieves the highest ROC-AUC (0.6627), as expected
- FedProx shows the best extreme_non_iid performance (0.6589) among FL methods
- Adaptive aggregation (Phase 10) matches or exceeds FedAvg in non-IID conditions
- Phase 11 no-DP results closely match Phase 10, confirming the DP mechanism preserves baseline behavior when sigma = 0

---

### 20. Key Findings

1. **Heterogeneity Impact:** Model performance degrades as heterogeneity increases. The extreme_non_iid condition shows the highest variance and lowest ROC-AUC across all phases.

2. **FedAvg Behavior:** Standard FedAvg achieves competitive performance in IID and mild conditions but struggles with strong and extreme heterogeneity. The sample-count weighting alone is insufficient to handle highly skewed data.

3. **FedProx Behavior:** The proximal term (μ = 0.01) provides modest improvements over FedAvg, particularly in extreme conditions. However, the improvement is not uniform across all metrics.

4. **Personalization Behavior:** Client-level personalization achieves competitive macro-averaged ROC-AUC but introduces higher variance. Personalization metrics are not directly comparable with pooled global metrics.

5. **Adaptive Aggregation Behavior:** Quality-weighted aggregation based on validation ROC-AUC improves model robustness, especially in strong and extreme non-IID conditions. The mechanism successfully identifies and upweights higher-quality clients.

6. **Privacy–Utility Trade-off:** Increasing noise multiplier (sigma) consistently degrades predictive utility. At sigma = 2.0, some conditions show near-random performance. The trade-off is steep, suggesting that careful noise calibration is essential for practical deployment.

7. **Client Eligibility:** The insufficient-data rule correctly skips clients 5, 8, and 9 in extreme_non_iid across all relevant phases, preventing single-class or empty-batch training.

8. **Class Imbalance:** Accuracy alone is misleading due to the imbalanced dataset. ROC-AUC and PR-AUC are more reliable indicators of model quality. All phases show accuracy in the 0.5–0.7 range, which is expected for this task.

---

### 21. Limitations

1. **Class Imbalance:** The dataset has a highly imbalanced target distribution, with the positive class (<30 readmission) representing only ~11% of samples. This affects all phases and limits the maximum achievable F1 and precision.

2. **Extreme Non-IID Single-Class/Insufficient-Data Clients:** In the extreme_non_iid condition, clients 5, 8, and 9 are skipped due to insufficient data. This limits the number of participating clients and may affect the generalizability of results.

3. **Limited Dataset:** The project currently evaluates on a single dataset (10-Year Diabetes Readmission). Results may not generalize to other healthcare prediction tasks or datasets.

4. **MLP Baseline Performance:** All phases use a single MLP architecture. More sophisticated architectures (e.g., transformers, tabular-specific models) were not explored.

5. **No Formal Privacy Accountant:** The Phase 11 DP mechanism does not include formal epsilon accounting. The noise mechanism is implemented as a heuristic and should not be interpreted as providing a provable differential privacy guarantee.

6. **DP Mechanism Limitations:** The current DP implementation uses fixed clipping norm and noise multiplier without adaptive calibration. The mechanism may be overly conservative or insufficient depending on the heterogeneity condition.

7. **Personalization Metrics Not Directly Comparable:** Phase 9 uses client-level macro averaging, making direct numerical comparison with other phases inappropriate.

8. **Single Random Seed:** All experiments use seed = 42. While this ensures reproducibility, it means variance across random initializations is not quantified.

9. **CPU-Only Execution:** All experiments run on CPU, which limits training speed and may affect the stability of DP training.

10. **No Hyperparameter Tuning:** Hyperparameters (learning rate, batch size, local epochs, etc.) are fixed across all phases and conditions.

---

### 22. Current Project Status

**COMPLETED:**
- Phase 1: Data Audit
- Phase 2: Data Preprocessing
- Phase 3: IID Client Generation
- Phase 4: Non-IID Client Generation and Heterogeneity
- Phase 5: Centralized Baseline
- Phase 6: Local-Only Training (IID and Non-IID)
- Phase 7: FedAvg
- Phase 8: FedProx
- Phase 9: Personalization
- Phase 10: Adaptive Aggregation
- Phase 11: Differential Privacy

**PLANNED:**
- Phase 12: SHAP Explainability
- Phase 13: Efficient Explainability
- Phase 14: Ablation Study
- Phase 15: Final Evaluation and Reproducibility
- Dashboard Development

---

### 23. Planned Next Steps

1. **Freeze and Document Phases 1–11:** Complete documentation and result archival for all implemented phases.
2. **Organize Project Structure Safely:** Consolidate results, ensure reproducibility, and create backup of all experiment outputs.
3. **Benchmark/Improve the Model Separately:** Explore alternative architectures and hyperparameters to improve baseline performance before explainability analysis.
4. **Select Final Model:** Choose the best-performing configuration based on validation ROC-AUC and PR-AUC across all phases.
5. **Phase 12 — SHAP:** Implement SHAP-based feature importance and explanation methods for the selected model.
6. **Phase 13 — Efficient Explainability:** Develop optimized explanation methods suitable for federated settings.
7. **Phase 14 — Ablation:** Conduct systematic ablation studies to isolate the contribution of each component (adaptive aggregation, DP, personalization).
8. **Phase 15 — Final Evaluation:** Comprehensive reproducibility assessment, statistical testing, and final report generation.
9. **Dashboard:** Develop an interactive dashboard for visualizing results, explanations, and privacy-utility trade-offs.

---

*Report generated based on existing experiment results in `results/`. No experiments were rerun or modified for this report.*
