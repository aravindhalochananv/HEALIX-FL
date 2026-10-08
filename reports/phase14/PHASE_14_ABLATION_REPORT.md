# HEALIX-FL Phase 14 — Ablation Study Report

## 1. Objective

Phase 14 evaluates the contribution of three HEALIX-FL components using a controlled 2×2×2 factorial ablation:

- FedProx proximal regularization
- Adaptive aggregation
- Personalized client models

There are 8 component configurations evaluated across 5 heterogeneity conditions, giving 40 experiment conditions.

## 2. Experimental design

The eight configurations were:

1. FedAvg baseline
2. FedProx only
3. Adaptive aggregation only
4. FedProx + Adaptive aggregation
5. Personalization only
6. FedProx + Personalization
7. Adaptive aggregation + Personalization
8. Full HEALIX-FL: FedProx + Adaptive aggregation + Personalization

Heterogeneity conditions:

- IID
- Mild Non-IID
- Moderate Non-IID
- Strong Non-IID
- Extreme Non-IID

The test set was evaluated only after the best validation round was selected.

## 3. Main results

| Heterogeneity | Best configuration | Test ROC-AUC | Test PR-AUC |
|---|---|---:|---:|
| Iid | fedprox_adaptive | 0.658984 | 0.211675 |
| Mild Non Iid | fedprox_adaptive | 0.658647 | 0.211295 |
| Moderate Non Iid | fedprox_only | 0.656799 | 0.210325 |
| Strong Non Iid | fedprox_personalized | 0.648231 | 0.201796 |
| Extreme Non Iid | fedprox_only | 0.656024 | 0.209981 |

FedProx appears in the best-performing configuration at all five heterogeneity levels. FedProx-only is best under moderate and extreme non-IID conditions, while FedProx + Adaptive is best under IID and mild non-IID. FedProx + Personalization is best under strong non-IID.

## 4. FedAvg versus Full HEALIX-FL

| Heterogeneity | FedAvg ROC-AUC | Full HEALIX-FL ROC-AUC | Difference |
|---|---:|---:|---:|
| Iid | 0.653774 | 0.656287 | +0.002513 |
| Mild Non Iid | 0.654215 | 0.655877 | +0.001662 |
| Moderate Non Iid | 0.652237 | 0.654515 | +0.002278 |
| Strong Non Iid | 0.639844 | 0.647885 | +0.008041 |
| Extreme Non Iid | 0.653230 | 0.650732 | -0.002498 |

The largest improvement of Full HEALIX-FL over FedAvg occurs under strong non-IID conditions (+0.008041 ROC-AUC). Under extreme non-IID, Full HEALIX-FL is slightly below the FedAvg baseline (-0.002498 ROC-AUC).

## 5. Component interpretation

### FedProx

FedProx is the most consistently beneficial component. Compared with FedAvg, FedProx-only improves ROC-AUC in all five heterogeneity conditions. The largest improvement is under strong non-IID conditions (+0.008340).

### Adaptive aggregation

Adaptive aggregation alone produces small gains under IID, mild non-IID, strong non-IID and extreme non-IID, but decreases ROC-AUC under moderate non-IID. Its benefit is therefore condition-dependent rather than uniformly dominant.

### Personalization

Personalization alone does not consistently improve the reported global test ROC-AUC. It is below FedAvg under IID, mild, moderate and extreme non-IID, while giving a small improvement under strong non-IID. Therefore, the present ablation does not support claiming that personalization alone improves global predictive performance.

## 6. Full HEALIX-FL

Full HEALIX-FL achieves:

- IID: ROC-AUC 0.656287
- Mild Non-IID: ROC-AUC 0.655877
- Moderate Non-IID: ROC-AUC 0.654515
- Strong Non-IID: ROC-AUC 0.647885
- Extreme Non-IID: ROC-AUC 0.650732

The strongest relative advantage over FedAvg occurs under strong non-IID conditions.

## 7. Research interpretation

The ablation results indicate that FedProx is the most consistently useful component for maintaining predictive performance under heterogeneous client distributions. Adaptive aggregation can provide additional benefit in selected heterogeneity regimes, while personalization alone does not consistently improve the global predictive metrics used here.

The full HEALIX-FL combination is competitive across all conditions and improves over FedAvg under IID, mild, moderate and strong non-IID settings. However, it does not dominate every simpler configuration, and it is slightly below FedAvg under extreme non-IID. Therefore, the evidence supports a nuanced conclusion: the contribution of the individual components depends on the degree of heterogeneity, with FedProx providing the most robust benefit in this experiment.

## 8. Limitations of this ablation

This analysis is based on one experimental seed/configuration and the reported test metrics. It does not establish statistical significance across repeated independent runs. Personalization may also provide client-specific benefits that are not fully represented by the aggregate global test ROC-AUC/PR-AUC summary.

## 9. Conclusion

Phase 14 successfully evaluated all 40 planned conditions. The results provide evidence that the proximal regularization component is the strongest and most consistently beneficial element of the tested HEALIX-FL design, particularly under strong heterogeneity. Adaptive aggregation contributes condition-dependent gains, while personalization alone does not consistently improve global predictive performance. The full framework remains competitive but should not be described as universally optimal across all heterogeneity levels.
