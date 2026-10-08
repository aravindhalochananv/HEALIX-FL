# Phase 12 - SHAP Explainability

## 1. Objective

SHAP was used to quantify the contribution of input features to the HEALIX-FL readmission prediction models.
SHAP explains model output contributions and does NOT establish causal relationships.

## 2. Models Explained

Models successfully explained: centralized, fedavg, fedprox, personalized, adaptive, dp_no_dp, dp_sigma_0.5, dp_sigma_1.0, dp_sigma_2.0

Primary model: personalized

## 3. SHAP Method

- Explainer: shap.GradientExplainer
- SHAP version: 0.52.0
- Background size: 50
- Explanation subset size: 100
- Random seed: 42
- Feature count: 2370
- Output explained: raw model logit (not sigmoid probability)

## 4. Data

- Total test samples: 20354
- Explanation subset: 100
- Background subset: 50
- Feature count: 2370

## 5. Global Explainability - Primary Model

Top 10 features (mean |SHAP|):

| Rank | Feature | mean_abs_shap |
| ---- | ------- | ------------- |
| 1 | num__number_inpatient | 0.233555 |
| 2 | cat__diag_3_250 | 0.076685 |
| 3 | num__discharge_disposition_id | 0.071558 |
| 4 | cat__diag_3_401 | 0.055294 |
| 5 | num__number_diagnoses | 0.052837 |
| 6 | num__number_emergency | 0.048407 |
| 7 | num__time_in_hospital | 0.046550 |
| 8 | cat__diag_1_786 | 0.044366 |
| 9 | cat__diag_2_401 | 0.041358 |
| 10 | cat__diag_1_682 | 0.036601 |

## 6. Local Explainability

     case_type  test_index  actual_class  predicted_class  predicted_probability            positive_feature_1  positive_shap_1            positive_feature_2  positive_shap_2            positive_feature_3  positive_shap_3 positive_feature_4  positive_shap_4 positive_feature_5  positive_shap_5    negative_feature_1  negative_shap_1 negative_feature_2  negative_shap_2    negative_feature_3  negative_shap_3       negative_feature_4  negative_shap_4  negative_feature_5  negative_shap_5
 true_negative       13702             0                0               0.311640         num__number_diagnoses         0.106827               cat__diag_3_250         0.070452               cat__diag_3_401         0.040600    cat__diag_1_682         0.035195    cat__diag_2_278         0.031185 cat__weight_[125-150)        -0.440800    cat__diag_1_722        -0.430751 num__number_inpatient        -0.220369       cat__payer_code_UN        -0.197867     cat__diag_3_585        -0.163037
 true_positive        6153             1                1               0.771124         num__number_inpatient         0.694036 num__discharge_disposition_id         0.275202          num__num_medications         0.051419    cat__diag_3_250         0.030135    cat__diag_3_401         0.028282      cat__diag_3_E888        -0.203372    cat__diag_2_428        -0.096414    cat__payer_code_SP        -0.043582 cat__pioglitazone_Steady        -0.032338 cat__insulin_Steady        -0.028533
false_positive       12802             0                1               0.501200 num__discharge_disposition_id         0.501490               cat__diag_3_250         0.063660               cat__diag_3_401         0.051411    cat__diag_1_786         0.049858    cat__diag_2_401         0.048559       cat__diag_3_714        -0.371336    cat__diag_2_250        -0.318775    cat__payer_code_CM        -0.278473    num__number_inpatient        -0.211003     cat__diag_1_715        -0.111055
false_negative         831             1                0               0.402043         num__number_diagnoses         0.096220         num__time_in_hospital         0.074065 num__discharge_disposition_id         0.055901    cat__diag_3_401         0.040946    cat__diag_3_250         0.038994       cat__diag_3_410        -0.436419    cat__diag_1_453        -0.390169 num__number_inpatient        -0.227536          cat__diag_2_496        -0.196900 cat__diabetesMed_No        -0.090349

## 7. Cross-Model Comparison

Top-10 feature overlap between models:

| Model A | Model B | Overlap count | Overlap % |
| ------- | ------- | ------------- | --------- |
| centralized | fedavg | 6 | 60.0% |
| centralized | fedprox | 7 | 70.0% |
| centralized | personalized | 6 | 60.0% |
| centralized | adaptive | 6 | 60.0% |
| centralized | dp_no_dp | 6 | 60.0% |
| centralized | dp_sigma_0.5 | 3 | 30.0% |
| centralized | dp_sigma_1.0 | 4 | 40.0% |
| centralized | dp_sigma_2.0 | 3 | 30.0% |
| fedavg | fedprox | 9 | 90.0% |
| fedavg | personalized | 10 | 100.0% |
| fedavg | adaptive | 10 | 100.0% |
| fedavg | dp_no_dp | 10 | 100.0% |
| fedavg | dp_sigma_0.5 | 3 | 30.0% |
| fedavg | dp_sigma_1.0 | 4 | 40.0% |
| fedavg | dp_sigma_2.0 | 3 | 30.0% |
| fedprox | personalized | 9 | 90.0% |
| fedprox | adaptive | 9 | 90.0% |
| fedprox | dp_no_dp | 9 | 90.0% |
| fedprox | dp_sigma_0.5 | 3 | 30.0% |
| fedprox | dp_sigma_1.0 | 4 | 40.0% |
| fedprox | dp_sigma_2.0 | 3 | 30.0% |
| personalized | adaptive | 10 | 100.0% |
| personalized | dp_no_dp | 10 | 100.0% |
| personalized | dp_sigma_0.5 | 3 | 30.0% |
| personalized | dp_sigma_1.0 | 4 | 40.0% |
| personalized | dp_sigma_2.0 | 3 | 30.0% |
| adaptive | dp_no_dp | 10 | 100.0% |
| adaptive | dp_sigma_0.5 | 3 | 30.0% |
| adaptive | dp_sigma_1.0 | 4 | 40.0% |
| adaptive | dp_sigma_2.0 | 3 | 30.0% |
| dp_no_dp | dp_sigma_0.5 | 3 | 30.0% |
| dp_no_dp | dp_sigma_1.0 | 4 | 40.0% |
| dp_no_dp | dp_sigma_2.0 | 3 | 30.0% |
| dp_sigma_0.5 | dp_sigma_1.0 | 4 | 40.0% |
| dp_sigma_0.5 | dp_sigma_2.0 | 6 | 60.0% |
| dp_sigma_1.0 | dp_sigma_2.0 | 5 | 50.0% |

## 8. Differential Privacy Explainability

DP models explained: dp_no_dp, dp_sigma_0.5, dp_sigma_1.0, dp_sigma_2.0

## 9. Important Interpretation Limitation

SHAP quantifies the contribution of input features to the model predictions.
It does NOT establish causal relationships between features and readmission outcomes.

## 10. Limitations

- SHAP was computed on a representative subset (100 samples) rather than all 20354 test samples.
- The input space has 2370 features, making global visualization challenging.
- One-hot encoded categorical features may fragment the interpretation of an original categorical variable.
- Explanations depend on the trained model and do not transfer automatically across models.

## 11. Phase 13 Motivation

Phase 12 establishes a baseline SHAP explainability pipeline.
Phase 13 will investigate more computationally efficient explainability methods.
