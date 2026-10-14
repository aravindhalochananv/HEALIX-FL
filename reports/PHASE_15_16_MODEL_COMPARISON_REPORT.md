# HEALIX-FL
## Phase 15–16 Model Comparison Report

**Project:** HEALIX-FL: A Heterogeneity-Aware and Explainable Federated Learning Framework for Privacy-Preserving Healthcare Prediction
**Task type:** Binary classification
**Prediction target:** `readmitted` (readmitted within 30 days = positive class; `NO` or `>30` = negative class)
**Report purpose:** To compare the five models carried forward from Phase 15 with the new Advanced Centralized Residual MLP trained in Phase 16, identify the best model for each evaluation metric, and honestly assess whether the new architecture improved on the original centralized baseline.
**Status:** Documentation-only. No experiments were rerun, no code or checkpoints were modified.

---

### Conventions used in this report

- Numeric metric values are reported here to **6 decimal places**, matching the precision of the source artifacts. Full-precision values are preserved in the saved CSV files.
- All metrics reported below are **test-set** metrics, unless a value is explicitly labelled as a validation metric.
- All test metrics in the comparison table are computed on the **same held-out test set** of 20,354 rows, so the models are directly comparable.
- Relative paths are used throughout; paths are relative to the project root (`C:\Users\smallcase\PYTHON PROJECT`).

---

## 1. Executive summary

This report compares six models evaluated on the HEALIX-FL readmission-prediction task: the original Centralized MLP, the new Advanced Centralized Residual MLP, and four federated learning (FL) models trained under the IID condition (FedAvg, FedProx, Adaptive Aggregation, and Personalized). All six models are evaluated once, on the same 20,354-sample test set.

**Main findings**

1. **Test metrics cluster tightly.** Across all six models, ROC-AUC ranges only from 0.652836 to 0.662686 and PR-AUC from 0.208612 to 0.213104. No model dominates across every metric simultaneously.
2. **No single model is best on every metric.** The original Centralized MLP holds the top rank on the ranking metrics (ROC-AUC and PR-AUC). FedProx (IID) leads accuracy, precision, and F1. Recall is highest for the original Centralized MLP.
3. **The advanced architecture changed the decision boundary, not the ranking quality.** Compared with the original Centralized MLP, the Advanced Centralized Residual MLP improved accuracy (+0.032181) and precision (+0.003439) but reduced recall (-0.051520). Its F1 score (-0.000115), ROC-AUC (-0.003963), and PR-AUC (-0.001661) are marginally lower. In other words, the architectural changes did **not** clearly improve the original centralized baseline on the ranking or balanced-F1 metrics that matter most under class imbalance.
4. **The federated methods are not separated from centralized baselines on the metrics reported here.** The IID federated results are competitive with, but do not exceed, the centralized baselines on ROC-AUC/PR-AUC, which is expected for the IID condition because all clients draw from the same pooled distribution.

Because the dataset is imbalanced (11.16% positives), accuracy and precision improvements can coincide with *lower* recall and essentially unchanged F1. This report therefore treats ROC-AUC and PR-AUC as the more reliable ranking metrics, and flags that "best" depends on the operational objective.

**Best model by each metric** (test set, 6 decimal places)

| Metric | Best model | Value |
|--------|------------|------:|
| Accuracy | FedProx (IID) | 0.630982 |
| Precision | FedProx (IID) | 0.169275 |
| Recall | Centralized MLP | 0.657420 |
| F1 score | FedProx (IID) | 0.263122 |
| ROC-AUC | Centralized MLP | 0.662686 |
| PR-AUC | Centralized MLP | 0.213104 |

The `final_evaluation_summary.json` artifact confirms the same ranking for the two ranking metrics (`best_by_pr_auc: "centralized"`, `best_roc_auc: 0.6626860248329459`).

---

## 2. Dataset and evaluation setup

**Source dataset:** The 10-Year Diabetes Readmission Dataset (UCI Machine Learning Repository), 101,766 inpatient encounters.

**Target variable:** `readmitted`, binarized to `readmitted <30` (positive) versus `NO` or `>30` (negative), consistent with the Phase 5–11 pipeline.

**Preprocessing (Phase 2):** The raw 50-column table is encoded and scaled to a fixed-width numeric feature matrix. After preprocessing, each row is represented by **2,370 input features**. The identifier columns `encounter_id` and `patient_nbr` are excluded, and constant columns (`examide`, `citoglipton`) are removed to prevent leakage.

**Splits (verified from `data/processed/*.npy` and `src/models/train.py`):**

| Split | Rows | Origin |
|-------|-----:|--------|
| Full training array | 81,412 | Phase 2 train/test split (80/20 of 101,766) |
| Test set | 20,354 | Separated in Phase 2; never used for training or validation |
| Phase 16 training subset | 65,129 | 80% stratified split of the 81,412 training array (`validation_split = 0.2`) |
| Phase 16 validation subset | 16,283 | 20% stratified split of the 81,412 training array |

The Phase 16 training subset (65,129) and validation subset (16,283) together form the original 81,412-row training array; the 20,354-row test set remains completely separate and is used only for final evaluation. This split is recreated identically by `build_centralized_splits` in `src/federated/fedavg.py` so that every model is scored on the same test set.

**Class imbalance and its implications:** In the full training array, the positive class (`readmitted <30`) accounts for **11.16%** of rows (verified: `pos_rate_train = 0.111605...`). The test positive rate is 11.16% as well. Because positives are a minority, accuracy is inflated by predicting the majority class. All models therefore train with a class-weighted loss, `BCEWithLogitsLoss` using `pos_weight = negative_count / positive_count` (= 7.96 on the full array). For the federated models, `pos_weight` is computed per client on each client's local label distribution (with a safe fallback of 1.0 for single-class clients), so the exact weighting differs client to client.

Because of this imbalance, **ROC-AUC and PR-AUC are more informative than accuracy**, and F1 (the harmonic mean of precision and recall) is the more honest single-number summary than accuracy alone.

---

## 3. Model architectures

All architectures in this report are **tabulated multi-layer perceptrons (MLPs)** for binary classification. An MLP is a stack of fully connected layers where each neuron in one layer is connected to every neuron in the next; activations are applied after each linear transform to introduce non-linearity. The network ends in a single logit (a raw score converted to a probability by the sigmoid function).

### 3.1 Original Centralized MLP (Phase 5)

A small, deliberately simple MLP used as the baseline throughout HEALIX-FL.

- **Architecture** (`src/models/mlp.py`): `Linear(2370 → 128) → ReLU → Dropout(0.2) → Linear(128 → 64) → ReLU → Dropout(0.2) → Linear(64 → 1)`.
- **Activation:** ReLU (Rectified Linear Unit) — returns 0 for negative inputs and the input unchanged for positive inputs.
- **Dropout:** 0.2 — randomly zeroes 20% of activations during training as a regularizer.
- **Parameters:** 311,809 trainable parameters (verified by constructing the model).
- **Training:** Centralised training on the pooled 65,129 training subset, 20 epochs, Adam optimizer, learning rate 0.001, batch size 256, `BCEWithLogitsLoss` with `pos_weight = 7.96`. Best checkpoint selected on validation ROC-AUC.

### 3.2 Advanced Centralized Residual MLP (Phase 16)

A stronger centralized MLP that adds three kinds of modern tabular-MLP components.

- **Architecture** (`src/models/advanced_mlp.py`): `Linear(2370 → 256) → BatchNorm1d(256) → GELU → Dropout(0.25) → [ResidualBlock × 3] → BatchNorm1d(256) → Linear(256 → 1)`.
- **ResidualBlock** (`src/models/advanced_mlp.py:12`): `BatchNorm1d → Linear(256 → 256) → GELU → Dropout → BatchNorm1d → Linear(256 → 256) → GELU → Dropout`, combined with a **skip connection** that adds the block's input to its output (`x + block(x)`). Skip connections let gradients flow directly through the network and make deeper models easier to train.
- **BatchNorm1d:** normalizes layer activations to stabilize and accelerate training.
- **GELU:** a smoother activation than ReLU (Gaussian Error Linear Unit), commonly used in deeper networks.
- **Width:** 256 hidden units; **3** residual blocks; **dropout:** 0.25.
- **Parameters:** 1,006,081 trainable parameters (verified by constructing the model) — approximately 3.2× the parameter count of the original MLP.
- **Training:** Centralised training on the pooled 65,129 training subset, **30 epochs** (full run), `AdamW` optimizer (Adam with decoupled weight decay), learning rate 0.0005, weight decay 0.0001, batch size 256, `BCEWithLogitsLoss` with `pos_weight = 7.96`. Best checkpoint selected on validation ROC-AUC (stored `best_val_roc_auc = 0.649670`).

**Architectural changes versus the original:** wider hidden layers (256 vs 128/64), three added residual blocks with skip connections, BatchNorm after each linear layer, and the GELU activation. The parameter count grew from 311,809 to 1,006,081.

### 3.3 FedAvg (IID) — Phase 7

Standard federated averaging. Each communication round, all 10 IID clients train the **same Phase 5 MLP** for one local epoch, then client parameters are averaged weighted by each client's sample count.

- **Aggregation:** `global_param = Σ(n_k · param_k) / Σ(n_k)`.
- **Configuration:** 20 rounds, 1 local epoch per round, client fraction 1.0, batch size 256, learning rate 0.001, Adam, seed 42, 10 IID clients (~8,141 samples each). Local `pos_weight` computed per client.
- **Difference from centralized:** identical **architecture** and **per-step optimizer/settings**, but training is distributed (partitioned data, one epoch per round) rather than centralised.

### 3.4 FedProx (IID) — Phase 8

FedAvg plus a **proximal term** that discourages each client's model from drifting too far from the global model.

- **Local objective:** `L_local(w) + (μ / 2) · ||w − w_global||²`, with `μ = 0.01`.
- The global reference parameters are detached so no gradient flows back into the global model.
- **Otherwise identical** to FedAvg (same MLP, 20 rounds, 1 local epoch, Adam, lr 0.001, etc.). The FedProx source states explicitly that "all other settings mirror Phase 7 for a controlled comparison."

### 3.5 Adaptive Aggregation (IID) — Phase 10

Sample-count-weighted FedAvg augmented with a per-client quality factor derived from each client's local validation ROC-AUC for the current round.

- **Weights:** `base_weight_k = n_k / Σ(n_j)`, `quality_k = clip(val_roc_auc_k, 0.5, 0.9)`, `normalized_quality_k = quality_k / mean(quality)`, then `adaptive_weight_k = base_weight_k · normalized_quality_k` (renormalized to sum to 1).
- **Otherwise identical** to FedAvg (same MLP, 20 rounds, 1 local epoch, Adam, lr 0.001).
- Validation ROC-AUC is used only for client weighting and best-round selection; the test set influences nothing.

### 3.6 Personalized (IID) — Phase 9

A personalization baseline on which each client keeps its own locally trained full model (in addition to contributing to a global model via FedAvg averaging).

- **Per-client model:** the same Phase 5 MLP, started from the global weights and trained one local epoch on the client's own data.
- **Evaluation:** each client's personalized model is scored on the test set, and the per-client metrics are macro-averaged across the 10 clients. In the Phase 15 comparison file, personalization is reported as a pooled ensemble (per-client probabilities averaged across the 10 client models on the 20,354 test rows).
- **Configuration:** same as FedAvg (20 rounds, 1 local epoch, Adam, lr 0.001, etc.).

**Training-procedure caveat (verified from source):** all four federated methods use the identical Phase 5 MLP architecture and near-identical per-step settings. They differ only in the aggregation rule (FedProx adds the proximal term; Adaptive multiplies by a quality factor; Personalized keeps per-client models). However, the federated procedure as a whole is **not** identical to the centralized procedure: centralized models train on the pooled 65,129 samples for 20–30 epochs in a single optimizer, while federated models train on partitioned client subsets for one epoch per round over 20 rounds. Model selection also differs: centralized best checkpoint is the best epoch by validation ROC-AUC, whereas federated best checkpoint is the best communication round by validation ROC-AUC.

---

## 4. Quantitative comparison

All values are **test-set** metrics on the 20,354-sample test set, taken from `results/phase15_final_evaluation/final_model_comparison.csv`. The Advanced Centralized Residual MLP row is additionally cross-checked against `results/advanced_centralized/advanced_metrics.csv`. All six models are evaluated on the same test set, so the rows are directly comparable.

**Test-set metrics (6 decimal places)**

| Model | Accuracy | Precision | Recall | F1 score | ROC-AUC | PR-AUC |
|-------|---------:|----------:|-------:|---------:|--------:|-------:|
| Centralized MLP | 0.586469 | 0.163491 | 0.657420 | 0.261861 | 0.662686 | 0.213104 |
| Advanced Centralized Residual MLP | 0.618650 | 0.166930 | 0.605900 | 0.261746 | 0.658723 | 0.211443 |
| FedAvg (IID) | 0.622679 | 0.166646 | 0.595332 | 0.260401 | 0.652836 | 0.208678 |
| FedProx (IID) | 0.630982 | 0.169275 | 0.590489 | 0.263122 | 0.658386 | 0.211066 |
| Adaptive Aggregation (IID) | 0.582588 | 0.159724 | 0.643329 | 0.255912 | 0.654199 | 0.209095 |
| Personalized (IID) | 0.587010 | 0.160412 | 0.638045 | 0.256369 | 0.655083 | 0.208612 |

**Confusion-matrix counts (test set)** — provided for completeness; these are the TP/TN/FP/FN underlying the metrics above and are read directly from the same CSV.

| Model | TP | TN | FP | FN | (sum) |
|-------|---:|---:|---:|---:|---:|
| Centralized MLP | 1,493 | 10,444 | 7,639 | 778 | 20,354 |
| Advanced Centralized Residual MLP | 1,376 | 11,216 | 6,867 | 895 | 20,354 |
| FedAvg (IID) | 1,352 | 11,322 | 6,761 | 919 | 20,354 |
| FedProx (IID) | 1,341 | 11,502 | 6,581 | 930 | 20,354 |
| Adaptive Aggregation (IID) | 1,461 | 10,397 | 7,686 | 810 | 20,354 |
| Personalized (IID) | 1,449 | 10,499 | 7,584 | 822 | 20,354 |

**Notes on sources and precision**

- The Phase 15 comparison CSV already contains the Advanced Centralized Residual MLP row (`model = advanced_centralized`), so the advanced model is **not** duplicated in the table above. The row is retained from the Phase 15 artifact and cross-checked against the Phase 16 dedicated artifact.
- All centralized and federated IID rows come from `final_model_comparison.csv` (Phase 15).
- The Advanced Centralized Residual MLP row is verified against `results/advanced_centralized/advanced_metrics.csv` (Phase 16).

---

## 5. Best model by metric

Selecting a single "best overall" model is not meaningful without declaring which metric the task optimizes. Under class imbalance, the metrics disagree:

- **Accuracy and precision** are highest for **FedProx (IID)**. FedProx raises accuracy by predicting the majority class more often (its confusion matrix has the most true negatives but also the most false negatives among the high-accuracy models).
- **Recall** is highest for the **Centralized MLP**, which catches the most true positive readmissions (1,493) at the cost of more false positives.
- **F1** (precision-recall balance at the 0.5 threshold) is highest for **FedProx (IID)**, but all F1 values are within 0.007 of each other — a near-tie.
- **ROC-AUC** and **PR-AUC** (threshold-independent ranking quality) are highest for the **Centralized MLP**. ROC-AUC measures ranking ability across all thresholds; PR-AUC is especially informative under class imbalance because it ignores true negatives, which dominate when negatives outnumber positives.

**Interpretation:** because the positive class is rare (11.16%), a model can post higher accuracy simply by being more conservative (fewer positive predictions). That is exactly what FedProx (IID) does: higher accuracy/precision/low-F1-balance, but lower recall and lower ROC-AUC/PR-AUC than the original centralized baseline. Therefore "best" depends on the evaluation objective:

- A screening utility prioritizing **recall** (catch as many at-risk patients as possible) would favor the **Centralized MLP**.
- A program tolerating **lower recall for higher precision** (fewer false alarms) may prefer **FedProx (IID)**.
- A neutral **ranking** objective (e.g., feeding a downstream threshold-tuned decision rule) would favor the **Centralized MLP** on ROC-AUC/PR-AUC.

No model is a clear, universal winner.

---

## 6. Advanced model versus original centralized MLP

This is the comparison that motivated Phase 16.

### 6.1 Architectural changes and parameter count

| Property | Original Centralized MLP | Advanced Centralized Residual MLP |
|----------|--------------------------|-----------------------------------|
| Hidden widths | 128, 64 | 256 (×3 residual blocks) |
| Activation | ReLU | GELU |
| Normalization | none | BatchNorm1d after each linear layer |
| Residual connections | none | 3 residual blocks with skip connections (`x + block(x)`) |
| Dropout | 0.2 | 0.25 |
| Trainable parameters | 311,809 | 1,006,081 (~3.2× larger) |
| Optimizer | Adam | AdamW (decoupled weight decay) |
| Learning rate | 0.001 | 0.0005 |
| Weight decay | none | 0.0001 |
| Epochs | 20 | 30 |

The advanced model is a materially larger and deeper network: ~3.2× the parameters, normalization, smoother activations, and residual pathways, trained for more epochs with a dedicated optimizer and regularizer.

### 6.2 Metric-by-metric differences (Advanced − Original)

| Metric | Original MLP | Advanced MLP | Difference (Advanced − Original) | Direction |
|--------|-------------:|-------------:|----------------------------------|-----------|
| Accuracy | 0.586469 | 0.618650 | +0.032181 | Improved |
| Precision | 0.163491 | 0.166930 | +0.003439 | Improved (marginal) |
| Recall | 0.657420 | 0.605900 | −0.051520 | Reduced |
| F1 score | 0.261861 | 0.261746 | −0.000115 | Unchanged (marginal decrease) |
| ROC-AUC | 0.662686 | 0.658723 | −0.0003963 | Reduced (marginal) |
| PR-AUC | 0.213104 | 0.211443 | −0.0001661 | Reduced (marginal) |

### 6.3 Did accuracy improve?

**Yes.** Accuracy rose from 0.586469 to 0.618650 (+0.032181), and precision rose slightly. Confusion-matrix counts confirm the trade-off: the advanced model makes fewer false positives (6,867 vs 7,639) but more false negatives (895 vs 778) — it is more conservative, predicting "readmitted" less often.

### 6.4 Did F1, ROC-AUC, and PR-AUC improve?

**No.** F1 fell by 0.000115, ROC-AUC by 0.003963, and PR-AUC by 0.001661. These are small absolute changes, but they are in the wrong direction, and ROC-AUC/PR-AUC are the metrics that matter most under this level of class imbalance. The apparent contradiction — higher accuracy and precision but lower recall, F1, ROC-AUC, and PR-AUC — is a direct consequence of threshold-dependent operating points under imbalance: shifting the decision boundary to predict the positive class less often raises accuracy/precision in a majority-dominated set while hurting recall and ranking quality.

### 6.5 Training/validation loss behavior and overfitting

The following is read from saved artifacts, not inferred:

- The Phase 16 training loop is `train_model` in `src/models/train.py:271`. It runs for the full `num_epochs` and stores the best checkpoint by validation ROC-AUC. The `patience=5` argument is documented in the source as **"not used currently, but kept for extensibility"** — that is, **no early stopping was actually applied**, so all 30 epochs ran.
- The saved checkpoint (`results/advanced_centralized/advanced_model.pt`) records the selected best metrics: **validation ROC-AUC = 0.649670**, **validation PR-AUC = 0.204644**.
- Per-epoch loss values are **not** saved as a CSV; only plots are saved: `centralized_training_loss.png` and `centralized_validation_loss.png` (plus `centralized_roc_auc.png` and `centralized_pr_auc.png`), which must be inspected visually to quantify the train/validation loss gap.
- The advanced model has 1,006,081 parameters trained on 65,129 samples (~48 parameters per training row), so its capacity is substantially higher than the original MLP (311,809 parameters) and it has a **greater capacity to overfit** the training split. Regularization is applied via dropout (0.25) and weight decay (1e-4), but there is no early stopping and no explicit validation-loss-based stopping.
- One cautionary observation: the best **validation** ROC-AUC (0.6497) is **lower** than the **test** ROC-AUC (0.658723). This indicates that the validation-selected checkpoint slightly under-tracks the test ranking. Rather than evidence of clean overfitting, this points to split-to-split variability for a 16,283-sample validation set and underscores why the test set must remain evaluation-only.

In short: the artifacts do not contain a numeric train/validation-loss table, so a precise overfitting gap cannot be quoted from a file. The saved plots are the source for that, and the larger parameter count relative to data is the structural reason overfitting is a plausible concern.

---

## 7. Federated-learning interpretation

**How the federated models compare with the centralized baselines**

- All four IID federated models (FedAvg, FedProx, Adaptive, Personalized) use the same Phase 5 MLP architecture as the centralized baseline, evaluated on the same 20,354-sample test set.
- On the ranking metrics that matter under imbalance, the centralized baselines are **at or slightly above** the IID federated results. The original Centralized MLP leads ROC-AUC (0.662686) and PR-AUC (0.213104); the advanced centralized model leads accuracy/precision within the centralized pair.
- FedProx (IID) posts the best accuracy (0.630982) and F1 (0.263122) of all six models, which is noteworthy but must not be over-read: it achieves this by being more conservative (lower recall, 0.590489).

**What these results do and do not establish about federated learning**

These results **do** establish that, on the IID condition with the full test set pooled across all clients, the federated methods can approach centralized performance when the data is evenly distributed. They **do not** establish that federated learning is superior to centralized training on this task. In fact, on the ranking metrics (ROC-AUC, PR-AUC), the centralized baseline is marginally ahead.

**Why that is expected, not a failure:** the IID condition is the most favourable FL setting — all 10 clients draw from the same distribution, so the global average of client models is close to a model trained on the pooled data. Federated learning's value proposition is privacy (no raw data is shared), not a blanket accuracy gain.

**Important scope caveat:** all four federated results shown here are **IID only**. The HEALIX-FL pipeline also defines four non-IID conditions (mild, moderate, strong, extreme), and the Phase 1–11 records show that under strong/extreme heterogeneity, the gap to centralized training can widen and client-skipping begins. Because this comparison table includes only the IID federated rows, **it does not establish performance across non-IID settings**, and it should not be read as a general claim about federated learning on this task.

---

## 8. Discrepancies found during verification

During artifact inspection, three items were cross-checked:

1. **Advanced model ROC-AUC and PR-AUC differ only at floating-point precision between the two saved artifacts.**
   - `results/phase15_final_evaluation/final_model_comparison.csv` (`advanced_centralized`): ROC-AUC = 0.6587232564514335, PR-AUC = 0.21144269290772716.
   - `results/advanced_centralized/advanced_metrics.csv` (Phase 16): ROC-AUC = 0.6587231833991765, PR-AUC = 0.2114426421044775.
   - The difference is at the 7th decimal and is floating-point noise; both round to **0.658723** and **0.211443**, matching the cross-check values supplied for Phase 16. No data integrity issue. The other four metrics (accuracy, precision, recall, F1) match exactly between the two files.

2. **Phase 15 already contains the advanced model.** The `final_model_comparison.csv` includes an `advanced_centralized` row, so the comparison table above lists the advanced model once (sourced from Phase 15, cross-checked with Phase 16). There is no duplicate row.

3. **The cross-check values supplied for Phase 16 match the saved artifacts to 6 decimal places.** Accuracy 0.618650, precision 0.166930, recall 0.605900, F1 0.261746, ROC-AUC 0.658723, PR-AUC 0.211443 — all consistent with `advanced_metrics.csv` and the Phase 15 CSV.

**Best-model consistency check:** the Phase 15 summary (`final_evaluation_summary.json`) reports `best_by_pr_auc: "centralized"` and `best_roc_auc: 0.6626860248329459`, which matches the Centralized MLP row in the table. The `models_evaluated` field is 6.

---

## 9. Limitations

1. **Single random seed.** All experiments use seed 42 (confirmed in `src/models/train.py` and every `src/federated/*.py`). Because there is only one seed, variance across random initializations and data shuffles is **not** quantified, and no statistical test of significance is performed. Reported differences on the order of 0.003–0.005 (the spread in ROC-AUC across models) are within plausible single-seed noise.

2. **Class imbalance.** With an 11.16% positive rate, precision and F1 are low across all models (~0.16–0.17 and ~0.26 respectively). Accuracy is inflated by the majority class and is not a reliable quality signal for this task.

3. **Modest precision and PR-AUC.** All models produce precision ≈ 0.16 and PR-AUC ≈ 0.21. PR-AUC is only slightly above the 0.1116 positive rate, indicating weak positive-class separation at the default 0.5 threshold.

4. **Threshold fixed at 0.5.** All metrics use a 0.5 probability threshold. No threshold tuning was performed; tuning to the validation set could change the accuracy/precision/recall balance materially without changing ROC-AUC/PR-AUC.

5. **Potential overfitting in Phase 16.** The advanced model trains for the full 30 epochs with no early stopping (the `patience` parameter is not wired into `train_model`), has ~1.0M parameters for ~65K training rows, and only dropout (0.25) plus weight decay (1e-4) regularize it. Per-epoch numeric loss history is not saved as a CSV — only plots — so the train/validation loss gap cannot be numerically quantified from the repository alone.

6. **No statistical-significance claims.** The reported metrics are point estimates from one test-set evaluation. Without repeated runs (multiple seeds) and an appropriate statistical test, **no claim of statistically significant improvement should be made**.

7. **IID-only federated comparison.** Four of the six rows in the comparison table are IID federated results. The non-IID conditions (mild/moderate/strong/extreme) are not included here, so the table does not represent federated performance under data heterogeneity.

8. **Centralised vs federated training procedure differs.** Although the federated methods share the centralized MLP architecture and per-step optimizer settings, the federated training procedure (20 rounds × 1 local epoch on partitioned clients) differs from the centralized procedure (20–30 epochs on pooled data). They are therefore not strictly apples-to-apples in optimization budget, even when the architecture is identical.

9. **Single dataset.** All results come from one dataset (10-Year Diabetes Readmission). Generalization to other clinical prediction tasks is not established.

10. **Not a clinical decision system.** This is an academic experiment. None of these models are validated for clinical use, and no clinical efficacy is claimed.

---

## 10. Conclusion and future work

**Conclusion.** On the IID condition, the six models perform within a narrow band on the ranking metrics that matter under class imbalance. The original Centralized MLP and the Advanced Centralized Residual MLP lead on ROC-AUC and PR-AUC respectively of the two centralized models; the advanced architecture improved accuracy (+0.032181) and precision (+0.003439) but **did not improve** F1, ROC-AUC, or PR-AUC versus the original centralized MLP — its recall fell and its ranking metrics were marginally lower. FedProx (IID) led accuracy, precision, and F1. No model is a universal winner, and the centralized baselines marginally edge the IID federated models on ROC-AUC/PR-AUC, which is the expected outcome for the most favourable FL scenario. The advanced architecture is therefore **not** a clear upgrade over the original centralized MLP on this task as currently configured.

**Future work** (not yet executed; proposed for a subsequent phase):

- **Repeated seeds.** Re-run Phases 5 and 16 (and the IID federated methods) with multiple seeds (e.g., 5–10) and report mean ± standard deviation, so that metric differences can be assessed for significance rather than treated as deterministic.
- **Validation-based early stopping.** Wire the existing `patience` parameter into `train_model` so the advanced model stops when validation ROC-AUC stops improving, and save a per-epoch loss CSV so the train/validation gap can be measured directly.
- **Threshold tuning.** Select the 0.5 decision threshold on the validation set (by optimizing F1 or a cost-weighted objective) rather than using the default 0.5; this can materially change precision/recall/accuracy without altering ROC-AUC/PR-AUC.
- **Hyperparameter search for the advanced model.** Sweep learning rate, weight decay, dropout, number of residual blocks, and width; the current advanced configuration (lr 0.0005, 30 epochs) was a single hand-set choice.
- **Non-IID federated comparison.** Include the mild/moderate/strong/extreme non-IID conditions so the federated methods are judged across the heterogeneity spectrum the project was built to study.
- **Loss-history logging.** Persist per-epoch train/validation loss and metrics as a CSV for every training run, so overfitting can be quantified numerically rather than read from plots.

---

## 11. Reproducibility and artifact paths

All listed files are actual saved artifacts used for this report. Paths are relative to the project root.

**Metrics and comparison tables**

- `results/phase15_final_evaluation/final_model_comparison.csv` — all 6 models, full-precision test metrics (source of the comparison table; includes the `advanced_centralized` row).
- `results/phase15_final_evaluation/final_evaluation_summary.json` — Phase 15 summary (best metrics, row count, missing-artifact check).
- `results/phase15_final_evaluation/missing_artifacts.json` — `[]`, i.e. no missing artifacts reported.
- `results/advanced_centralized/advanced_metrics.csv` — Phase 16 Advanced Centralized Residual MLP test metrics (cross-check of the advanced row).

**Model checkpoint**

- `results/advanced_centralized/advanced_model.pt` — Phase 16 saved checkpoint (state dict, model config, training config, `best_val_roc_auc`, `best_val_pr_auc`).
  - **Git note:** `advanced_model.pt` is present locally but is **ignored by Git**. The repository's `.gitignore` contains the rule `*.pt` (with a comment that future generated checkpoints should not be staged), so this file is not tracked and must be regenerated by re-running Phase 16 to be restored from version control.

**Configuration**

- `results/advanced_centralized/advanced_config.yaml` — Phase 16 configuration (model + training). Verified: `input_dim: 2370`, `model: {type: advanced_residual_mlp, width: 256, num_blocks: 3, dropout: 0.25}`, `training: {batch_size: 256, epochs: 30, learning_rate: 0.0005, weight_decay: 0.0001, validation_split: 0.2}`.

**Plots**

- `results/advanced_centralized/centralized_training_loss.png` — training and validation loss curves (used for visual overfitting inspection; no numeric CSV counterpart exists).
- `results/advanced_centralized/centralized_validation_loss.png` — validation ROC-AUC curve (file is named `centralized_validation_loss.png` but plots validation ROC-AUC per `src/models/train.py:392-402`).
- `results/advanced_centralized/centralized_roc_auc.png` — validation ROC-AUC over epochs.
- `results/advanced_centralized/centralized_pr_auc.png` — validation PR-AUC over epochs.
- `results/advanced_centralized/advanced_confusion_matrix.png` — test-set confusion matrix for the advanced model.

**Source code**

- `src/models/advanced_mlp.py` — `ResidualBlock` and `AdvancedMLP` definitions and parameter counting.
- `src/models/train_advanced.py` — Phase 16 training driver.
- `src/models/mlp.py` — original Phase 5 MLP.
- `src/models/train.py` — shared training/evaluation utilities (`set_seed`, `create_data_loaders`, `train_model`, `evaluate_model`); note that `train_model` documents `patience` as not currently used.
- `src/federated/fedavg.py`, `src/federated/fedprox.py`, `src/federated/adaptive_aggregation.py`, `src/federated/personalized.py` — federated methods (identical MLP architecture; FedProx adds the proximal term; Adaptive adds the quality factor; Personalized keeps per-client models).
- `data/processed/X_train.npy`, `data/processed/y_train.npy`, `data/processed/X_test.npy`, `data/processed/y_test.npy` — processed feature/target arrays (Git-ignored via `*.npy`; shapes verified: `X_train (81412, 2370)`, `X_test (20354, 2370)`).

**Verification commands used (read-only inspection only)**

- Constructed `MLP` and `AdvancedMLP` from source and called `count_parameters`: 311,809 and 1,006,081.
- Loaded `data/processed/*.npy` only to read shapes and the positive-class rate (no training performed).
- Loaded `results/advanced_centralized/advanced_model.pt` only to read stored metadata (`model_type`, `best_val_roc_auc`, `best_val_pr_auc`, `model_config`, `training_config`).

---

*Report generated from existing saved artifacts in `results/` and source in `src/`. No models were retrained, no code or existing reports were modified, and no Git history was altered.*
