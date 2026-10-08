# Phase 13 - Efficient Explainability

## Objective

Phase 13 evaluates whether SHAP explanations for the personalized IID HEALIX-FL model can be obtained with substantially fewer background and explanation samples while preserving agreement with the frozen Phase 12 global feature-importance ranking.

## Experimental Setup

- Primary model: `personalized`
- Explainer: `shap.GradientExplainer`
- Features: 2370
- Random seed: 42
- Phase 12 baseline explanation samples: 100
- Phase 12 baseline background samples: 50

## Configurations

| Condition | Background | Explanation samples |
|---|---:|---:|
| baseline | 50 | 100 |
| efficient_1 | 25 | 50 |
| efficient_2 | 10 | 50 |
| efficient_3 | 10 | 25 |

## Reproducibility

The Phase 13 baseline uses the same SHAP configuration as Phase 12 and is validated before efficient configurations are interpreted.

- Baseline top-10 overlap: 10/10
- Baseline Spearman correlation: 0.999997
- Baseline mean absolute SHAP difference: 0.00002900

The baseline reproducibility check passed.

## Efficiency Results

| Condition | Runtime (s) | Top-10 overlap | Spearman | Mean abs SHAP difference | Explanation reduction | Background reduction |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3.4000 | 10/10 | 0.999997 | 0.00002900 | 0.00% | 0.00% |
| efficient_1 | 1.7100 | 7/10 | 0.854650 | 0.00047400 | 50.00% | 50.00% |
| efficient_2 | 1.7300 | 8/10 | 0.818048 | 0.00069600 | 50.00% | 80.00% |
| efficient_3 | 0.8300 | 6/10 | 0.764320 | 0.00082100 | 75.00% | 80.00% |

## Key Observations

Efficiency should be interpreted jointly with explanation agreement. A smaller SHAP sample is useful only when the result remains sufficiently consistent with the Phase 12 reference.

## Conclusion

Phase 13 provides an empirical comparison between the original Phase 12 SHAP configuration and reduced-sample configurations. The results quantify the trade-off between SHAP computation cost and preservation of global feature importance.

No model was retrained during this phase, and the Phase 12 outputs were treated as frozen reference results.
