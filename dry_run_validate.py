"""
Simulated dry-run of the Award B pipeline on fabricated datasets.
This script follows the CLAUDE.md playbook step-by-step, proving the
methodology works on arbitrary tabular data without domain assumptions.

Runs on both regression and classification datasets.
"""
import numpy as np, pandas as pd, lightgbm as lgb, os, sys, shutil, time
from sklearn.model_selection import KFold, GroupKFold
np.random.seed(42)

def run_pipeline(data_dir, task_label):
    print(f'\n{"="*70}')
    print(f'DRY-RUN: {task_label}')
    print(f'Data dir: {data_dir}')
    print(f'{"="*70}')

    # Phase 1: Data Inventory
    print('\n--- Phase 1: Data Inventory ---')
    files = sorted(os.listdir(data_dir))
    print(f'Files: {files}')
    dfs = {}
    for f in files:
        if f.endswith('.csv'):
            dfs[f] = pd.read_csv(f'{data_dir}/{f}')
            print(f'  {f}: shape={dfs[f].shape}, columns={list(dfs[f].columns)}')

    # Identify sample_submission (target name source)
    sub_file = [f for f in dfs if 'submission' in f.lower()][0]
    sub = dfs[sub_file]
    target_col = [c for c in sub.columns if c != sub.columns[0]][0]
    id_col = sub.columns[0]
    print(f'Target: {target_col}, ID: {id_col}')

    # Identify train/test
    train_file = [f for f in dfs if 'train' in f.lower()][0]
    test_file = [f for f in dfs if 'test' in f.lower() and 'submission' not in f.lower()][0]
    train = dfs[train_file]
    test = dfs[test_file]
    print(f'Train: {train.shape}, Test: {test.shape}')

    # Phase 2: Quick EDA
    print('\n--- Phase 2: EDA ---')
    y = train[target_col]
    if y.dtype in ['float64', 'float64', 'int64'] and y.nunique() > 10:
        task_type = 'regression'
        print(f'Task: REGRESSION (target has {y.nunique()} unique values)')
        print(f'  Target stats: mean={y.mean():.2f} std={y.std():.2f} min={y.min():.2f} max={y.max():.2f}')
    else:
        task_type = 'classification'
        print(f'Task: CLASSIFICATION (target has {y.nunique()} unique values)')
        print(f'  Class distribution: {dict(y.value_counts())}')

    # Missing values
    for fname, df in dfs.items():
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f'  {fname} missing: {dict(missing[missing > 0])}')
        else:
            print(f'  {fname}: no missing values')

    # Phase 3: Baseline Submission
    print('\n--- Phase 3: Baseline Submission ---')
    baseline_sub = sub.copy()
    if task_type == 'regression':
        baseline_sub[target_col] = y.mean()
    else:
        baseline_sub[target_col] = y.mode().iloc[0]
    print(f'Baseline: predict {baseline_sub[target_col].iloc[0]:.2f} for all rows')
    baseline_sub.to_csv(f'{data_dir}/../baseline_submission.csv', index=False)
    print('Baseline submission written (safety net)')

    # Phase 4: Model Training
    print('\n--- Phase 4: Model Training ---')

    # Prepare features
    feature_cols = [c for c in train.columns if c not in {target_col, id_col}
                    and train[c].dtype in ['float64', 'int64', 'object']]
    cat_cols = [c for c in feature_cols if train[c].dtype == 'object']
    for c in cat_cols:
        train[c] = train[c].astype('category')
        test[c] = test[c].astype('category')

    X = train[feature_cols]; X_test = test[feature_cols]

    if task_type == 'regression':
        lgb_params = {
            'objective': 'regression_l1', 'metric': 'mae', 'verbose': -1,
            'learning_rate': 0.05, 'num_leaves': 31, 'seed': 42,
        }
    else:
        lgb_params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'learning_rate': 0.05, 'num_leaves': 31, 'seed': 42,
        }

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold_i, (tr, va) in enumerate(kf.split(X, y)):
        dtrain = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dval = lgb.Dataset(X.iloc[va], label=y.iloc[va])
        model = lgb.train(lgb_params, dtrain, 500, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va] = model.predict(X.iloc[va])
        test_preds += model.predict(X_test) / 5

        if task_type == 'regression':
            fold_mae = np.mean(np.abs(y.iloc[va] - oof[va]))
            fold_scores.append(fold_mae)
            print(f'  Fold {fold_i}: MAE={fold_mae:.4f}')
        else:
            fold_acc = np.mean((oof[va] > 0.5).astype(int) == y.iloc[va])
            fold_scores.append(fold_acc)
            print(f'  Fold {fold_i}: Accuracy={fold_acc:.4f}')

    if task_type == 'regression':
        overall = np.mean(np.abs(y - oof))
        print(f'Overall MAE: {overall:.4f} (std={np.std(fold_scores):.4f})')
    else:
        overall = np.mean((oof > 0.5).astype(int) == y)
        print(f'Overall Accuracy: {overall:.4f} (std={np.std(fold_scores):.4f})')

    # Phase 5: Post-process & Submit
    print('\n--- Phase 5: Final Submission ---')
    final_sub = sub.copy()
    if task_type == 'regression':
        final_sub[target_col] = np.clip(test_preds, y.min(), y.max())
    else:
        final_sub[target_col] = (test_preds > 0.5).astype(int)

    # Verify
    assert final_sub.shape[1] == 2, f"Expected 2 columns, got {final_sub.shape[1]}"
    assert set(final_sub[id_col]) == set(sub[id_col]), "Row ID mismatch"
    assert len(final_sub) == len(sub), f"Length mismatch"
    assert np.isfinite(final_sub[target_col]).all(), "Non-finite predictions"
    assert list(final_sub.columns) == list(sub.columns), "Column name mismatch"

    out_path = f'{data_dir}/../submission.csv'
    final_sub.to_csv(out_path, index=False)
    print(f'Wrote {out_path} — shape {final_sub.shape}')
    print(f'Predictions: mean={final_sub[target_col].mean():.2f} range=[{final_sub[target_col].min():.2f}, {final_sub[target_col].max():.2f}]')

    # Phase 6: Report (minimal PDF)
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, f'Data Analysis Report — {task_label}', ln=True)
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, f'Task type: {task_type}', ln=True)
        pdf.cell(0, 8, f'Target: {target_col}', ln=True)
        pdf.cell(0, 8, f'Train: {train.shape}, Test: {test.shape}', ln=True)
        pdf.cell(0, 8, f'Model: LightGBM ({lgb_params["objective"]})', ln=True)
        if task_type == 'regression':
            pdf.cell(0, 8, f'CV MAE: {overall:.4f} (std={np.std(fold_scores):.4f})', ln=True)
        else:
            pdf.cell(0, 8, f'CV Accuracy: {overall:.4f} (std={np.std(fold_scores):.4f})', ln=True)
        pdf.cell(0, 8, f'Features: {len(feature_cols)}', ln=True)
        pdf.cell(0, 8, f'Submission shape: {final_sub.shape}', ln=True)
        pdf_report_path = f'{data_dir}/../report.pdf'
        pdf.output(pdf_report_path)
        print(f'Wrote {pdf_report_path}')
    except ImportError:
        print('fpdf not installed, skipping PDF generation')

    return True

# Run on both datasets
base = os.path.dirname(os.path.abspath(__file__))
success_a = run_pipeline(f'{base}/dry_run_regression/data', 'Regression (Housing Prices)')
success_b = run_pipeline(f'{base}/dry_run_classification/data', 'Classification (Customer Churn)')

print(f'\n{"="*70}')
print('DRY-RUN SUMMARY')
print(f'{"="*70}')
print(f'  Regression: {"PASSED" if success_a else "FAILED"}')
print(f'  Classification: {"PASSED" if success_b else "FAILED"}')
print(f'  Both domains handled without any domain-specific logic.')
