"""
Phase 12: SHAP Explainability for HEALIX-FL
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import shap

from src.models.mlp import MLP


RANDOM_SEED = 42
BACKGROUND_SIZE = 50
EXPLANATION_SIZE = 100
N_FEATURES = 2370

PROCESSED_DIR = "data/processed"
FEATURE_NAMES_CSV = os.path.join(PROCESSED_DIR, "feature_names.csv")
X_TRAIN_NPY = os.path.join(PROCESSED_DIR, "X_train.npy")
X_TEST_NPY = os.path.join(PROCESSED_DIR, "X_test.npy")
Y_TEST_NPY = os.path.join(PROCESSED_DIR, "y_test.npy")

RESULTS_DIR = "results/shap"
LOCAL_PLOTS_DIR = os.path.join(RESULTS_DIR, "local")
REPORT_PATH = "reports/PHASE_12_SHAP_EXPLAINABILITY_REPORT.md"

MODELS = {
    "centralized": {"path": "results/centralized_model.pt", "format": "metadata_dict"},
    "fedavg": {"path": "results/fedavg/iid/best_global_model.pt", "format": "state_dict"},
    "fedprox": {"path": "results/fedprox/iid/best_global_model.pt", "format": "state_dict"},
    "personalized": {"path": "results/personalized/iid/best_global_model.pt", "format": "state_dict"},
    "adaptive": {"path": "results/adaptive_aggregation/iid/best_global_model.pt", "format": "state_dict"},
    "dp_no_dp": {"path": "results/differential_privacy/no_dp/iid/best_global_model.pt", "format": "state_dict"},
    "dp_sigma_0.5": {"path": "results/differential_privacy/sigma_0.5/iid/best_global_model.pt", "format": "state_dict"},
    "dp_sigma_1.0": {"path": "results/differential_privacy/sigma_1.0/iid/best_global_model.pt", "format": "state_dict"},
    "dp_sigma_2.0": {"path": "results/differential_privacy/sigma_2.0/iid/best_global_model.pt", "format": "state_dict"},
}

PRIMARY_MODEL = "personalized"

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_processed_data():
    for p in [X_TRAIN_NPY, X_TEST_NPY, Y_TEST_NPY, FEATURE_NAMES_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError('Required Phase 2 artifact missing: ' + p)
    X_train = np.load(X_TRAIN_NPY)
    X_test = np.load(X_TEST_NPY)
    y_test = np.load(Y_TEST_NPY)
    feature_names = pd.read_csv(FEATURE_NAMES_CSV)['feature_name'].tolist()
    assert X_train.shape[1] == N_FEATURES
    assert X_test.shape[1] == N_FEATURES
    assert len(feature_names) == N_FEATURES
    assert X_test.shape[0] == y_test.shape[0]
    return X_train, X_test, y_test, feature_names


def select_stratified_subset(X, y, size, seed=RANDOM_SEED):
    from sklearn.model_selection import train_test_split
    n = len(X)
    if size >= n:
        idx = np.arange(n)
    else:
        _, idx = train_test_split(np.arange(n), test_size=size, random_state=seed, stratify=y)
        idx = np.sort(idx)
    return X[idx], y[idx], idx


def load_model(model_name):
    if model_name not in MODELS:
        raise ValueError('Unknown model: ' + model_name)
    spec = MODELS[model_name]
    path = spec['path']
    if not os.path.exists(path):
        raise FileNotFoundError('Checkpoint missing: ' + path)
    model = MLP(input_dim=N_FEATURES, hidden_dim1=128, hidden_dim2=64, dropout=0.2)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if spec['format'] == 'metadata_dict':
        model.load_state_dict(ck['model_state_dict'])
        meta = ck
    else:
        model.load_state_dict(ck)
        meta = {}
    model.eval()
    return model, meta


def predict_logits(model, X):
    X_t = torch.FloatTensor(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        logits = model(X_t).numpy().flatten()
    return logits


def predict_probs(model, X):
    return 1.0 / (1.0 + np.exp(-predict_logits(model, X)))


def compute_shap(model, background, X_explain):
    bg = torch.FloatTensor(np.asarray(background, dtype=np.float32))
    explainer = shap.GradientExplainer(model, bg)
    X_t = torch.FloatTensor(np.asarray(X_explain, dtype=np.float32))
    shap_values = explainer.shap_values(X_t)
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        sv = sv.reshape(sv.shape[0], sv.shape[1])
    assert sv.shape == (len(X_explain), N_FEATURES), 'SHAP shape mismatch: ' + str(sv.shape)
    return sv


def global_feature_importance(model_name, shap_values, feature_names):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)
    df = pd.DataFrame({
        'model': model_name,
        'feature': feature_names,
        'mean_abs_shap': mean_abs,
        'mean_shap': mean_shap,
    })
    df = df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    return df[['model', 'rank', 'feature', 'mean_abs_shap', 'mean_shap']]


def save_global_csv(all_rows, path):
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(path, index=False)
    return path


def save_top_features_csv(all_rows, top_k=20, path=None):
    if path is None:
        path = os.path.join(RESULTS_DIR, 'phase12_top_features.csv')
    df = pd.concat(all_rows, ignore_index=True)
    top = df[df['rank'] <= top_k].copy()
    top.to_csv(path, index=False)
    return path


def make_bar_plot(df, model_name, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    top = df.nlargest(20, 'mean_abs_shap')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top['feature'][::-1], top['mean_abs_shap'][::-1])
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title('Top 20 SHAP features - ' + model_name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_summary_plot(df, X_explain, shap_values, feature_names, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    top20 = df.nlargest(20, 'mean_abs_shap')
    col_idx = [feature_names.index(f) for f in top20['feature']]
    sub = shap_values[:, col_idx]
    order = np.argsort(-np.abs(sub).mean(axis=0))
    sub = sub[:, order]
    ordered_features = [top20['feature'].iloc[i] for i in order]
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(sub, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Explanation sample')
    ax.set_xticks(range(len(ordered_features)))
    ax.set_xticklabels(ordered_features, rotation=90, fontsize=8)
    fig.colorbar(im, ax=ax, label='SHAP value (logit contribution)')
    ax.set_title('SHAP values (explanation subset, top 20 features) - ' + df['model'].iloc[0])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def find_local_cases(y_true, y_prob, y_pred, n=5):
    cases = {}
    tn = np.where((y_true == 0) & (y_pred == 0))[0]
    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]
    rng = np.random.RandomState(RANDOM_SEED)
    if len(tn) > 0:
        cases['true_negative'] = int(rng.choice(tn))
    if len(tp) > 0:
        cases['true_positive'] = int(rng.choice(tp))
    if len(fp) > 0:
        cases['false_positive'] = int(rng.choice(fp))
    if len(fn) > 0:
        cases['false_negative'] = int(rng.choice(fn))
    return cases


def local_explanations(shap_values, feature_names, y_true, y_prob, y_pred, test_indices):
    cases = find_local_cases(y_true, y_prob, y_pred)
    rows = []
    for case_type, local_idx in cases.items():
        sv = shap_values[local_idx]
        order = np.argsort(-sv)
        pos_feats = [feature_names[i] for i in order[:5]]
        pos_vals = [float(sv[i]) for i in order[:5]]
        neg_feats = [feature_names[i] for i in order[-5:][::-1]]
        neg_vals = [float(sv[i]) for i in order[-5:][::-1]]
        rows.append({
            'case_type': case_type,
            'test_index': int(test_indices[local_idx]),
            'actual_class': int(y_true[local_idx]),
            'predicted_class': int(y_pred[local_idx]),
            'predicted_probability': float(y_prob[local_idx]),
            'positive_feature_1': pos_feats[0], 'positive_shap_1': pos_vals[0],
            'positive_feature_2': pos_feats[1], 'positive_shap_2': pos_vals[1],
            'positive_feature_3': pos_feats[2], 'positive_shap_3': pos_vals[2],
            'positive_feature_4': pos_feats[3], 'positive_shap_4': pos_vals[3],
            'positive_feature_5': pos_feats[4], 'positive_shap_5': pos_vals[4],
            'negative_feature_1': neg_feats[0], 'negative_shap_1': neg_vals[0],
            'negative_feature_2': neg_feats[1], 'negative_shap_2': neg_vals[1],
            'negative_feature_3': neg_feats[2], 'negative_shap_3': neg_vals[2],
            'negative_feature_4': neg_feats[3], 'negative_shap_4': neg_vals[3],
            'negative_feature_5': neg_feats[4], 'negative_shap_5': neg_vals[4],
        })
    return pd.DataFrame(rows)


def save_local_csv(df, path):
    df.to_csv(path, index=False)
    return path


def make_local_plot(shap_values, feature_names, y_true, y_prob, y_pred, test_indices, case_type, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = find_local_cases(y_true, y_prob, y_pred)
    if case_type not in cases:
        return None
    local_idx = cases[case_type]
    sv = shap_values[local_idx]
    order = np.argsort(np.abs(sv))[::-1][:15]
    feats = [feature_names[i] for i in order][::-1]
    vals = [sv[i] for i in order][::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feats, vals, color=['red' if v > 0 else 'blue' for v in vals])
    ax.set_xlabel('SHAP value (logit contribution)')
    ax.set_title('Local explanation - ' + case_type + ' (test index ' + str(int(test_indices[local_idx])) + ')')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def feature_overlap(df_a, df_b, top_k=10):
    set_a = set(df_a[df_a['rank'] <= top_k]['feature'])
    set_b = set(df_b[df_b['rank'] <= top_k]['feature'])
    shared = set_a & set_b
    return {
        'top_k': top_k,
        'shared_features': '|'.join(sorted(shared)),
        'overlap_count': len(shared),
        'overlap_percentage': round(len(shared) / top_k * 100, 2),
    }


def run_shap_analysis(model_names, X_train, X_test, y_test, feature_names):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOCAL_PLOTS_DIR, exist_ok=True)

    X_bg = X_train[:BACKGROUND_SIZE]
    X_ex, y_ex, test_idx = select_stratified_subset(X_test, y_test, EXPLANATION_SIZE, RANDOM_SEED)

    y_ex = np.asarray(y_ex)
    test_idx = np.asarray(test_idx)

    global_rows = []
    top_rows = []
    local_df = None
    comparison_rows = []
    dp_rows = []
    overlap_rows = []
    primary_df = None
    primary_model_name = None
    primary_shap = None
    primary_y_pred = None
    primary_y_prob = None
    primary_y_true = None
    primary_test_idx = None

    for model_name in model_names:
        try:
            model, meta = load_model(model_name)
        except Exception as e:
            print('SKIP ' + model_name + ': ' + str(e))
            continue

        t0 = time.time()
        sv = compute_shap(model, X_bg, X_ex)
        runtime = round(time.time() - t0, 2)

        logits = predict_logits(model, X_ex)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

        df = global_feature_importance(model_name, sv, feature_names)
        global_rows.append(df)
        top_rows.append(df[df['rank'] <= 20])

        if model_name == PRIMARY_MODEL:
            primary_df = df
            primary_model_name = model_name
            primary_shap = sv
            primary_y_pred = preds
            primary_y_prob = probs
            primary_y_true = y_ex
            primary_test_idx = test_idx

        if local_df is None and model_name == PRIMARY_MODEL:
            local_df = local_explanations(sv, feature_names, y_ex, probs, preds, test_idx)

        for _, r in df.iterrows():
            comparison_rows.append({
                'model': model_name,
                'rank': int(r['rank']),
                'feature': r['feature'],
                'mean_abs_shap': float(r['mean_abs_shap']),
            })

        if model_name.startswith('dp_'):
            for _, r in df.iterrows():
                dp_rows.append({
                    'dp_condition': model_name,
                    'rank': int(r['rank']),
                    'feature': r['feature'],
                    'mean_abs_shap': float(r['mean_abs_shap']),
                })
            print('  ' + model_name + ' runtime=' + str(runtime) + 's')
        else:
            print('  ' + model_name + ' runtime=' + str(runtime) + 's')

    return {
        'global_rows': global_rows,
        'top_rows': top_rows,
        'local_df': local_df,
        'comparison_rows': comparison_rows,
        'dp_rows': dp_rows,
        'primary_df': primary_df,
        'primary_model_name': primary_model_name,
        'primary_shap': primary_shap,
        'primary_y_pred': primary_y_pred,
        'primary_y_prob': primary_y_prob,
        'primary_y_true': primary_y_true,
        'primary_test_idx': primary_test_idx,
    }


def compute_overlaps(global_rows, top_k=10):
    rows = []
    names = [df['model'].iloc[0] for df in global_rows]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = feature_overlap(global_rows[i], global_rows[j], top_k)
            rows.append({
                'model_a': names[i],
                'model_b': names[j],
                'top_k': ov['top_k'],
                'shared_features': ov['shared_features'],
                'overlap_count': ov['overlap_count'],
                'overlap_percentage': ov['overlap_percentage'],
            })
    return pd.DataFrame(rows)


def save_metadata(metadata, path):
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


def write_report(metadata, primary_top10, local_summary, overlap_df, dp_summary, path):
    lines = []
    lines.append('# Phase 12 - SHAP Explainability')
    lines.append('')
    lines.append('## 1. Objective')
    lines.append('')
    lines.append('SHAP was used to quantify the contribution of input features to the HEALIX-FL readmission prediction models.')
    lines.append('SHAP explains model output contributions and does NOT establish causal relationships.')
    lines.append('')
    lines.append('## 2. Models Explained')
    lines.append('')
    lines.append('Models successfully explained: ' + ', '.join(metadata['models_explained']))
    lines.append('')
    lines.append('Primary model: ' + metadata['primary_model'])
    lines.append('')
    lines.append('## 3. SHAP Method')
    lines.append('')
    lines.append('- Explainer: shap.GradientExplainer')
    lines.append('- SHAP version: ' + str(metadata['shap_version']))
    lines.append('- Background size: ' + str(metadata['background_size']))
    lines.append('- Explanation subset size: ' + str(metadata['explanation_size']))
    lines.append('- Random seed: ' + str(metadata['random_seed']))
    lines.append('- Feature count: ' + str(metadata['n_features']))
    lines.append('- Output explained: raw model logit (not sigmoid probability)')
    lines.append('')
    lines.append('## 4. Data')
    lines.append('')
    lines.append('- Total test samples: ' + str(metadata['total_test_samples']))
    lines.append('- Explanation subset: ' + str(metadata['explanation_size']))
    lines.append('- Background subset: ' + str(metadata['background_size']))
    lines.append('- Feature count: ' + str(metadata['n_features']))
    lines.append('')
    lines.append('## 5. Global Explainability - Primary Model')
    lines.append('')
    lines.append('Top 10 features (mean |SHAP|):')
    lines.append('')
    lines.append('| Rank | Feature | mean_abs_shap |')
    lines.append('| ---- | ------- | ------------- |')
    for _, r in primary_top10.iterrows():
        lines.append('| ' + str(int(r['rank'])) + ' | ' + r['feature'] + ' | ' + format(float(r['mean_abs_shap']), '.6f') + ' |')
    lines.append('')
    lines.append('## 6. Local Explainability')
    lines.append('')
    lines.append(local_summary)
    lines.append('')
    lines.append('## 7. Cross-Model Comparison')
    lines.append('')
    lines.append('Top-10 feature overlap between models:')
    lines.append('')
    if len(overlap_df) > 0:
        lines.append('| Model A | Model B | Overlap count | Overlap % |')
        lines.append('| ------- | ------- | ------------- | --------- |')
        for _, r in overlap_df.iterrows():
            lines.append('| ' + str(r['model_a']) + ' | ' + str(r['model_b']) + ' | ' + str(int(r['overlap_count'])) + ' | ' + str(r['overlap_percentage']) + '% |')
    lines.append('')
    lines.append('## 8. Differential Privacy Explainability')
    lines.append('')
    lines.append(dp_summary)
    lines.append('')
    lines.append('## 9. Important Interpretation Limitation')
    lines.append('')
    lines.append('SHAP quantifies the contribution of input features to the model predictions.')
    lines.append('It does NOT establish causal relationships between features and readmission outcomes.')
    lines.append('')
    lines.append('## 10. Limitations')
    lines.append('')
    lines.append('- SHAP was computed on a representative subset (' + str(metadata['explanation_size']) + ' samples) rather than all ' + str(metadata['total_test_samples']) + ' test samples.')
    lines.append('- The input space has ' + str(metadata['n_features']) + ' features, making global visualization challenging.')
    lines.append('- One-hot encoded categorical features may fragment the interpretation of an original categorical variable.')
    lines.append('- Explanations depend on the trained model and do not transfer automatically across models.')
    lines.append('')
    lines.append('## 11. Phase 13 Motivation')
    lines.append('')
    lines.append('Phase 12 establishes a baseline SHAP explainability pipeline.')
    lines.append('Phase 13 will investigate more computationally efficient explainability methods.')
    lines.append('')
    with open(path, 'w') as f:
        f.write(chr(10).join(lines))


def main():
    parser = argparse.ArgumentParser(description='HEALIX-FL Phase 12: SHAP Explainability')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset of model names to explain (default: all available)')
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    X_train, X_test, y_test, feature_names = load_processed_data()
    print('Loaded X_train=' + str(X_train.shape) + ' X_test=' + str(X_test.shape) +
          ' y_test=' + str(y_test.shape) + ' features=' + str(len(feature_names)))

    model_names = args.models if args.models else list(MODELS.keys())
    results = run_shap_analysis(model_names, X_train, X_test, y_test, feature_names)

    # Global importance
    global_path = os.path.join(RESULTS_DIR, 'phase12_global_feature_importance.csv')
    save_global_csv(results['global_rows'], global_path)
    print('Saved ' + global_path)

    # Top features
    top_path = save_top_features_csv(results['top_rows'], top_k=20)
    print('Saved ' + top_path)

    # Local explanations
    if results['local_df'] is not None:
        local_path = os.path.join(RESULTS_DIR, 'phase12_local_explanations.csv')
        save_local_csv(results['local_df'], local_path)
        print('Saved ' + local_path)
        for _, row in results['local_df'].iterrows():
            plot_path = make_local_plot(
                results['primary_shap'], feature_names,
                results['primary_y_true'], results['primary_y_prob'],
                results['primary_y_pred'], results['primary_test_idx'],
                row['case_type'],
                os.path.join(LOCAL_PLOTS_DIR, 'patient_' + str(row['test_index']) + '_' + row['case_type'] + '.png'),
            )
            if plot_path:
                print('Saved ' + plot_path)

    # Plots for primary model
    if results['primary_df'] is not None:
        bar_path = os.path.join(RESULTS_DIR, 'phase12_bar_' + results['primary_model_name'] + '.png')
        make_bar_plot(results['primary_df'], results['primary_model_name'], bar_path)
        print('Saved ' + bar_path)
        summ_path = os.path.join(RESULTS_DIR, 'phase12_summary_' + results['primary_model_name'] + '.png')
        make_summary_plot(results['primary_df'], X_test[:EXPLANATION_SIZE], results['primary_shap'],
                          feature_names, summ_path)
        print('Saved ' + summ_path)

    # Model comparison
    comp_path = os.path.join(RESULTS_DIR, 'phase12_model_comparison.csv')
    comp_df = pd.DataFrame(results['comparison_rows'],
                           columns=['model', 'rank', 'feature', 'mean_abs_shap'])
    comp_df.to_csv(comp_path, index=False)
    print('Saved ' + comp_path)

    # Feature overlap
    overlap_df = compute_overlaps(results['global_rows'], top_k=10)
    overlap_path = os.path.join(RESULTS_DIR, 'phase12_feature_overlap.csv')
    overlap_df.to_csv(overlap_path, index=False)
    print('Saved ' + overlap_path)

    # DP comparison
    dp_df = pd.DataFrame(results['dp_rows'],
                         columns=['dp_condition', 'rank', 'feature', 'mean_abs_shap'])
    dp_path = os.path.join(RESULTS_DIR, 'phase12_dp_comparison.csv')
    dp_df.to_csv(dp_path, index=False)
    print('Saved ' + dp_path)

    # Metadata
    metadata = {
        'phase': 12,
        'shap_version': shap.__version__,
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'random_seed': RANDOM_SEED,
        'background_size': BACKGROUND_SIZE,
        'explanation_size': EXPLANATION_SIZE,
        'n_features': N_FEATURES,
        'total_test_samples': int(len(y_test)),
        'explainer': 'shap.GradientExplainer',
        'models_explained': [df['model'].iloc[0] for df in results['global_rows']],
        'primary_model': results['primary_model_name'],
    }
    meta_path = os.path.join(RESULTS_DIR, 'phase12_run_metadata.json')
    save_metadata(metadata, meta_path)
    print('Saved ' + meta_path)

    # Report
    primary_top10 = results['primary_df'].head(10) if results['primary_df'] is not None else pd.DataFrame()
    local_summary = 'No local explanations generated.' if results['local_df'] is None else results['local_df'].to_string(index=False)
    dp_summary = 'No DP models explained.' if len(dp_df) == 0 else 'DP models explained: ' + ', '.join(dp_df['dp_condition'].unique().tolist())
    write_report(metadata, primary_top10, local_summary, overlap_df, dp_summary, REPORT_PATH)
    print('Saved ' + REPORT_PATH)

    print('Phase 12 SHAP explainability completed.')


if __name__ == '__main__':
    main()
