"""
STAI-X Challenge 2026 — Schema-Portable Inference Pipeline

Zero hardcoded period_ids, row counts, category lists, or jurisdiction counts.
Everything derived from input files at runtime.

Feature set (locked from Task 2 gate): all features EXCEPT precip_in.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings, os, json, sys
from collections import defaultdict
warnings.filterwarnings('ignore')
np.random.seed(42)

_KAGGLE_CANDIDATES = [
    '/kaggle/input/stai-x-challenge-2026',
    '/kaggle/input/competitions/stai-x-challenge-2026',
    '/kaggle/input/staix-challenge',
]
def _detect_data_dir():
    for c in _KAGGLE_CANDIDATES:
        if os.path.exists(c) and os.path.exists(f'{c}/train/dose_sys_train.csv'):
            return c, False
    # glob fallback: search any subdir containing the canonical train file
    import glob
    for match in glob.glob('/kaggle/input/**/train/dose_sys_train.csv', recursive=True):
        return match.rsplit('/train/dose_sys_train.csv', 1)[0], False
    return '.', True
DATA, LOCAL = _detect_data_dir()
DATA = os.environ.get('STAIX_DATA', DATA)
OUT = os.environ.get('STAIX_OUT', 'submission.csv' if LOCAL else '/kaggle/working/submission.csv')

SEED = 42; TOP_N_FEATURES = 40
LGB_PARAMS = {
    'objective': 'mae', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 5, 'reg_alpha': 1.0, 'reg_lambda': 5.0, 'min_child_samples': 30,
    'verbose': -1, 'n_jobs': -1, 'seed': SEED,
}
LGB_CONFIGS = [
    {'num_leaves': 31, 'max_depth': 7, 'seed': SEED,     **LGB_PARAMS},
    {'num_leaves': 15, 'max_depth': 5, 'seed': SEED + 1, **LGB_PARAMS},
    {'num_leaves': 63, 'max_depth': 9, 'seed': SEED + 2, **LGB_PARAMS},
]
EW_DECAY = 0.90

# ============================================================
# STAGE-2 SAFETY FALLBACK (P1.1)
# Default OFF. When ON, blends drift baseline + model prediction
# for inference rows where any of (a)/(b)/(c) fires.
# Trigger is row-level at (period, jurisdiction): if it fires, all 3
# category rows at that position get blended (covariates are shared,
# not target-specific).
# ============================================================
STAGE2_SAFE_FALLBACK = os.environ.get('STAIX_STAGE2_FALLBACK', '0') == '1'  # default OFF
STAGE2_PERIOD_THRESHOLD = int(os.environ.get('STAIX_STAGE2_PERIOD_THRESHOLD', '10'))  # A1-opt X: 15→10
FALLBACK_BASELINE_WEIGHT = float(os.environ.get('STAIX_FALLBACK_WEIGHT', '0.6'))
MISSING_FRAC_THRESHOLD = 0.10       # A3: 50% → 10%

# A3-list (model-required, used for NaN imputation contract) — 8 cols
REQUIRED_COVARIATES = [
    'gtrends_fentanyl', 'gtrends_overdose', 'gtrends_naloxone',
    'gtrends_opioid', 'gtrends_methamphetamine',
    'unemployment_rate', 'temp_avg_f', 'labor_force',
]

# Condition (c) anomaly-detection signal — gtrends only (Option 3).
# gtrends are 0% missing on train AND val (verified 2026-06-15), the stable signal
# source. unemployment_rate / labor_force missingness is a known structural val
# property (CLAUDE.md), not a Stage-2 anomaly. Mixing them into (c) made fallback
# fire on every val row — wrong. (c) now only fires on gtrends NaN, the real
# pipeline-anomaly signal.
ANOMALY_DETECT_COVARIATES = [
    'gtrends_fentanyl', 'gtrends_overdose', 'gtrends_naloxone',
    'gtrends_opioid', 'gtrends_methamphetamine',
]

# ============================================================
# 1. LOAD DATA
# ============================================================
train_y = pd.read_csv(f'{DATA}/train/dose_sys_train.csv')
train_x = pd.read_csv(f'{DATA}/train/covariates.csv')
infer_x = pd.read_csv(f'{DATA}/val/covariates.csv')
sub_tmpl = pd.read_csv(f'{DATA}/sample_submission.csv')

# Derive scoring categories from submission template (not hardcoded)
SCORING = sorted(sub_tmpl['overdose_category'].unique().tolist())
ALL_CATS = sorted(train_y['overdose_category'].unique().tolist())
NON_SCORING = sorted([c for c in ALL_CATS if c not in SCORING])
print(f'Scoring categories (from submission): {SCORING}')
print(f'Non-scoring categories (from train):  {NON_SCORING}')

# Derive period ordering from train covariates (for drift baselines)
period_order = train_x.groupby('period_id')['gtrends_fentanyl'].mean().sort_values().reset_index()
period_order['period_rank'] = range(len(period_order))

# ============================================================
# 2. TRAIN-DERIVED STATISTICS (computed once, used everywhere)
# ============================================================
# Jurisdiction-level gtrends stats (covariate-derived, not target)
jur_gtrends_stats = train_x.groupby('jurisdiction')[
    ['gtrends_overdose', 'gtrends_fentanyl', 'gtrends_naloxone',
     'gtrends_opioid', 'gtrends_methamphetamine']
].agg(['mean', 'std']).reset_index()
jur_gtrends_stats.columns = ['jurisdiction'] + [
    f'{col}_{stat}' for col in ['gtrends_overdose', 'gtrends_fentanyl', 'gtrends_naloxone',
                                 'gtrends_opioid', 'gtrends_methamphetamine']
    for stat in ['mean', 'std']
]

# Covariate defaults for imputation (train medians)
covariate_numeric_cols = ['unemployment_rate', 'labor_force', 'temp_avg_f',
                          'gtrends_overdose', 'gtrends_fentanyl', 'gtrends_naloxone',
                          'gtrends_opioid', 'gtrends_methamphetamine']
covariate_defaults = {col: train_x[col].median() for col in covariate_numeric_cols}
print(f'Covariate defaults for imputation: {covariate_defaults}')

# Per-category train min/max (for clipping)
cat_bounds = {}
for cat in SCORING:
    vals = train_y.loc[train_y['overdose_category'] == cat, 'rate_per_10000_ed_visits']
    cat_bounds[cat] = (vals.min(), vals.max())
print(f'Category bounds: {cat_bounds}')

# ============================================================
# 3. TEXT PROCESSING CONFIG
# ============================================================
KEYWORDS = [
    'naloxone', 'fentanyl', 'overdose', 'syringe', 'treatment',
    'opioid', 'methamphetamine', 'xylazine', 'test strip', 'harm reduction',
    'narcan', 'buprenorphine', 'naltrexone', 'clinic',
    'distribution', 'emergency', 'fatal',
    'counterfeit', 'over-the-counter', 'expansion', 'response',
]
RELEASE_TYPES = ['surveillance', 'cluster', 'xylazine', 'meth_coinvolve',
                 'counterfeit', 'recovery', 'qrt', 'syringe', 'test_strip',
                 'otc_naloxone', 'mooud', 'awareness', 'samaritan', 'school']

def classify_title(t):
    t_lower = t.lower()
    if 'surveillance' in t_lower or 'provisional' in t_lower: return 'surveillance'
    if 'cluster' in t_lower: return 'cluster'
    if 'xylazine' in t_lower: return 'xylazine'
    if 'co-involvement' in t_lower or 'methamphetamine' in t_lower: return 'meth_coinvolve'
    if 'counterfeit' in t_lower: return 'counterfeit'
    if 'recovery housing' in t_lower: return 'recovery'
    if 'quick response' in t_lower: return 'qrt'
    if 'syringe' in t_lower: return 'syringe'
    if 'test strip' in t_lower or 'drug checking' in t_lower: return 'test_strip'
    if 'over-the-counter' in t_lower or 'narcan' in t_lower.lower(): return 'otc_naloxone'
    if 'buprenorphine' in t_lower or 'medication for opioid' in t_lower: return 'mooud'
    if 'awareness' in t_lower: return 'awareness'
    if 'good samaritan' in t_lower: return 'samaritan'
    if 'school' in t_lower: return 'school'
    return 'other'

# ============================================================
# 4. FEATURE ENGINEERING (works on any frame)
# ============================================================
def engineer_features(df):
    """Apply all features to any frame (train or inference). No target info used."""
    d = df.copy()
    eps = 1e-6

    # Task 3.1: substitute empty-string column if state_doh_release is missing
    # entirely. .fillna('') below handles the all-NaN case (Task 3.2) when the
    # column exists but values are NaN.
    if 'state_doh_release' not in d.columns:
        d['state_doh_release'] = ''

    # Impute missing covariates with train defaults
    for col, default_val in covariate_defaults.items():
        if col in d.columns:
            d[col] = d[col].fillna(default_val)

    # Within-state gtrends z-scores (from train-derived jurisdiction stats)
    d = d.merge(jur_gtrends_stats, on='jurisdiction', how='left')
    for col in ['gtrends_overdose', 'gtrends_fentanyl', 'gtrends_naloxone',
                'gtrends_opioid', 'gtrends_methamphetamine']:
        mean_col = f'{col}_mean'; std_col = f'{col}_std'
        d[f'zscore_{col}'] = (d[col] - d[mean_col]) / (d[std_col] + eps)
        d.drop([mean_col, std_col], axis=1, inplace=True)

    # Gtrends ratios
    od_floor = d['gtrends_overdose'].clip(lower=eps)
    fent_floor = d['gtrends_fentanyl'].clip(lower=eps)
    nalo_floor = d['gtrends_naloxone'].clip(lower=eps)
    d['ratio_fent_od']   = d['gtrends_fentanyl'] / od_floor
    d['ratio_nalo_od']   = d['gtrends_naloxone'] / od_floor
    d['ratio_opi_od']    = d['gtrends_opioid'] / od_floor
    d['ratio_meth_od']   = d['gtrends_methamphetamine'] / od_floor
    d['ratio_fent_nalo'] = d['gtrends_fentanyl'] / nalo_floor
    d['ratio_meth_fent'] = d['gtrends_methamphetamine'] / fent_floor
    d['ratio_opi_fent']  = d['gtrends_opioid'] / fent_floor

    # On-the-fly period-level means (works on ANY frame, including Stage 2)
    gtrends_cols = ['gtrends_overdose','gtrends_fentanyl','gtrends_naloxone',
                    'gtrends_opioid','gtrends_methamphetamine']
    period_means = d.groupby('period_id')[gtrends_cols].transform('mean')
    period_means.columns = [f'period_mean_{c}' for c in gtrends_cols]
    d = pd.concat([d, period_means], axis=1)
    for col in gtrends_cols:
        d[f'dev_{col}'] = d[col] - d[f'period_mean_{col}']

    pm_od = d['period_mean_gtrends_overdose'].clip(lower=eps)
    d['period_ratio_fent_od'] = d['period_mean_gtrends_fentanyl'] / pm_od
    d['period_ratio_nalo_od'] = d['period_mean_gtrends_naloxone'] / pm_od
    d['period_ratio_meth_od'] = d['period_mean_gtrends_methamphetamine'] / pm_od
    d['period_ratio_fent_total'] = d['period_mean_gtrends_fentanyl'] / (
        d['period_mean_gtrends_overdose'] + d['period_mean_gtrends_fentanyl'] +
        d['period_mean_gtrends_naloxone'] + d['period_mean_gtrends_opioid'] +
        d['period_mean_gtrends_methamphetamine'] + eps)

    d['gtrends_sum'] = d['gtrends_fentanyl'] + d['gtrends_naloxone'] + d['gtrends_methamphetamine']
    d['gtrends_fent_x_nalo'] = d['gtrends_fentanyl'] * d['gtrends_naloxone']
    d['log_labor_force'] = np.log1p(d['labor_force'])
    d['log_gtrends_overdose'] = np.log1p(d['gtrends_overdose'])

    # Text features
    text = d['state_doh_release'].fillna('').astype(str)
    text_lower = text.str.lower()
    d['has_text'] = (text.str.len() > 0).astype(int)
    d['text_length'] = text.str.len()
    d['text_word_count'] = text.str.split().str.len().fillna(0).astype(int)
    for kw in KEYWORDS:
        d[f'kw_{kw.replace(" ", "_").replace("-", "_")}'] = text_lower.str.count(kw)
    kw_cols = [f'kw_{kw.replace(" ", "_").replace("-", "_")}' for kw in KEYWORDS]
    d['kw_total'] = d[kw_cols].sum(axis=1)
    for rtype in RELEASE_TYPES:
        d[f'rtype_{rtype}'] = 0
    for idx in d.index:
        t = text.loc[idx]
        if len(t) == 0: continue
        for rel in t.split(' --- '):
            title = rel.split('\n')[0] if '\n' in rel else rel[:200]
            rtype = classify_title(title)
            if rtype in RELEASE_TYPES:
                d.loc[idx, f'rtype_{rtype}'] += 1
    crisis_types = ['cluster', 'xylazine', 'meth_coinvolve', 'surveillance']
    d['n_crisis_indicators'] = sum(d[f'rtype_{t}'] for t in crisis_types)
    d['n_releases'] = text.str.count('---') + 1
    d.loc[text.str.len() == 0, 'n_releases'] = 0

    d['jurisdiction'] = d['jurisdiction'].astype('category')
    return d

# ============================================================
# 4b. STAGE-2 SAFETY FALLBACK TRIGGER
# ============================================================
def compute_fallback_trigger(infer_x_raw, train_x):
    """
    Stage-2 safety fallback trigger (row-level, at covariate grain).

    Returns:
      trigger_mask: np.array[bool], len(infer_x_raw)
      reasons:      dict[int, list[str]] (per-covariate-row reason codes)

    Conditions (any one -> trigger for that covariate-row):
      (a) inference frame period count >= STAGE2_PERIOD_THRESHOLD
          -> frame-level: all rows if fires
      (b) any within-state gtrends inference value falls outside
          [train_min * 0.5, train_max * 1.5] for that (jur, col)
          -> per-jurisdiction: rows in that jurisdiction trigger
          (range check, window-size agnostic; variance-based tests fail on
           this dataset due to base[period] x scale[jur] structure that
           compresses within-state std on narrow inference windows)
      (c) >MISSING_FRAC_THRESHOLD of inference rows have >=1 NaN in
          ANOMALY_DETECT_COVARIATES (gtrends only; counted BEFORE imputation)
          -> frame-level: all rows if fires

    Notes:
      - state_doh_release handled separately by Task 3 column-missing logic;
        not counted here.
      - jurisdiction / period_id structural keys missing go through portable's
        unseen-jurisdiction map-mean fallback; not counted here.
    """
    n_rows = len(infer_x_raw)
    trigger = np.zeros(n_rows, dtype=bool)
    reasons = defaultdict(list)

    # (a) Period count threshold (frame-level)
    period_count = infer_x_raw['period_id'].nunique()
    if period_count >= STAGE2_PERIOD_THRESHOLD:
        trigger[:] = True
        sys.stderr.write(f'[FALLBACK] cond (a) fired: period_count={period_count} >= {STAGE2_PERIOD_THRESHOLD}\n')

    # (b) Within-state gtrends value-range check (per-jurisdiction x col).
    # Window-size agnostic: tests whether inference values fall outside the train
    # range x [0.5, 1.5] tolerance band. Variance-based tests fail on this dataset
    # because gtrends ~= base[period] x scale[jur] (CLAUDE.md), so narrow inference
    # windows structurally compress within-state std to ~0.11x of train, uniform
    # across jurisdictions — not anomaly, just narrower time window.
    gtrends_cols = ['gtrends_overdose', 'gtrends_fentanyl', 'gtrends_naloxone',
                    'gtrends_opioid', 'gtrends_methamphetamine']
    n_b_fired = 0
    b_examples = []  # cap stderr noise: print first 5 only
    for jur in infer_x_raw['jurisdiction'].unique():
        infer_jur_mask = (infer_x_raw['jurisdiction'] == jur)
        infer_jur_idx = np.where(infer_jur_mask.values)[0]
        for col in gtrends_cols:
            train_vals = train_x.loc[train_x['jurisdiction'] == jur, col].dropna().values
            infer_vals = infer_x_raw.loc[infer_jur_mask, col].dropna().values
            if len(train_vals) < 5 or len(infer_vals) < 1:
                continue
            lo = train_vals.min() * 0.5
            hi = train_vals.max() * 1.5
            out_of_range = (infer_vals < lo) | (infer_vals > hi)
            if out_of_range.any():
                trigger[infer_jur_idx] = True
                n_b_fired += 1
                if len(b_examples) < 5:
                    violator = infer_vals[out_of_range][0]
                    direction = 'below lo*0.5' if violator < lo else 'above hi*1.5'
                    b_examples.append((jur, col, len(train_vals), len(infer_vals),
                                       violator, lo, hi, direction))
    if n_b_fired > 0:
        sys.stderr.write(f'[FALLBACK] cond (b) fired: {n_b_fired} (jur,gtrends_col) pairs with infer values outside [train_min*0.5, train_max*1.5]\n')
        for jur, col, nt, ni, val, lo, hi, dr in b_examples:
            sys.stderr.write(f'[FALLBACK]   ex: jur={jur} col={col} n_train={nt} n_infer={ni} violator={val:.4f} ({dr}) bound=[{lo:.4f}, {hi:.4f}]\n')

    # (c) Missing anomaly-detect covariate fraction (BEFORE imputation, gtrends only)
    missing_per_row = infer_x_raw[ANOMALY_DETECT_COVARIATES].isna().any(axis=1)
    missing_frac = missing_per_row.sum() / n_rows
    if missing_frac > MISSING_FRAC_THRESHOLD:
        trigger[:] = True
        sys.stderr.write(f'[FALLBACK] cond (c) fired: missing_frac={missing_frac:.3f} > {MISSING_FRAC_THRESHOLD} (gtrends only)\n')

    return trigger, reasons

# ============================================================
# 5. TARGET-DERIVED FEATURES (from full train, for deployment)
# ============================================================
def compute_target_features(train_scored_df, all_train_y, period_order_df):
    """Compute all target-derived features from training data."""
    rate = 'rate_per_10000_ed_visits'

    # Jurisdiction-category statistics
    train_scored_df = train_scored_df.copy()
    train_scored_df['rate_log'] = np.log1p(train_scored_df[rate])

    jur_cat_stats = (
        train_scored_df.groupby(['jurisdiction', 'overdose_category'])[rate]
        .agg(jur_cat_mean='mean', jur_cat_std='std', jur_cat_median='median',
             jur_cat_min='min', jur_cat_max='max')
        .reset_index()
    )
    jur_cat_stats['jur_cat_std'] = jur_cat_stats['jur_cat_std'].fillna(0)
    jur_cat_stats['jur_cat_range'] = jur_cat_stats['jur_cat_max'] - jur_cat_stats['jur_cat_min']
    log_jur_stats = (
        train_scored_df.groupby(['jurisdiction', 'overdose_category'])['rate_log']
        .agg(jur_cat_log_mean='mean').reset_index()
    )
    jur_cat_stats = jur_cat_stats.merge(log_jur_stats, on=['jurisdiction', 'overdose_category'], how='left')

    # Sub-category profiles (non-scoring categories)
    subcat = (
        all_train_y[all_train_y['overdose_category'].isin(NON_SCORING)]
        .groupby(['jurisdiction', 'overdose_category'])[rate]
        .mean().reset_index()
        .pivot(index='jurisdiction', columns='overdose_category', values=rate)
        .reset_index()
    )
    subcat.columns = [f'subcat_{c}' if c != 'jurisdiction' else c for c in subcat.columns]
    subcat = subcat.fillna(0)
    drugs_mean = (
        all_train_y[all_train_y['overdose_category'] == 'all_drugs']
        .groupby('jurisdiction')[rate].mean()
        .reset_index().rename(columns={rate: 'subcat_all_drugs_mean'})
    )
    subcat = subcat.merge(drugs_mean, on='jurisdiction', how='left')
    for sc in NON_SCORING:
        col = f'subcat_{sc}'
        if col in subcat.columns:
            subcat[f'subcat_share_{sc}'] = subcat[col] / (subcat['subcat_all_drugs_mean'] + 1e-6)

    # Drift baselines (recency-20 for drugs/opioids, EW-0.90 for stimulants)
    train_with_rank = train_scored_df.merge(
        period_order_df[['period_id', 'period_rank']], on='period_id', how='left'
    )
    train_sorted = train_with_rank.sort_values('period_rank')['period_id'].unique()
    n_recent = min(20, len(train_sorted))
    recent_periods = set(train_sorted[-n_recent:])

    baseline_maps = {}
    for cat in SCORING:
        sub = train_with_rank[train_with_rank['overdose_category'] == cat]
        if cat == 'all_stimulants':
            max_rank = sub['period_rank'].max()
            raw_map = sub.groupby('jurisdiction').apply(
                lambda g: np.average(g[rate], weights=EW_DECAY**(max_rank - g['period_rank'])))
            log_map = sub.groupby('jurisdiction').apply(
                lambda g: np.average(np.log1p(g[rate]), weights=EW_DECAY**(max_rank - g['period_rank'])))
        else:
            recent = sub[sub['period_id'].isin(recent_periods)]
            if len(recent) == 0: recent = sub
            raw_map = recent.groupby('jurisdiction')[rate].mean()
            log_map = recent.groupby('jurisdiction')[rate].apply(lambda x: np.log1p(x).mean())
        baseline_maps[cat] = (raw_map, log_map)

    return jur_cat_stats, subcat, baseline_maps

def merge_target_features(df, jur_stats, subcat_means, baseline_map=None, cat=None):
    """Merge target-derived features into a dataframe."""
    d = df.copy()
    d = d.merge(jur_stats, on=['jurisdiction', 'overdose_category'], how='left')
    d = d.merge(subcat_means, on='jurisdiction', how='left')
    if baseline_map is not None and cat is not None:
        raw_map, log_map = baseline_map
        d['raw_baseline'] = d['jurisdiction'].astype(str).map(raw_map)
        d['log_baseline'] = d['jurisdiction'].astype(str).map(log_map)
    d['jurisdiction'] = d['jurisdiction'].astype('category')
    for col in d.columns:
        if col.startswith(('jur_cat_', 'subcat_', 'raw_baseline', 'log_baseline')):
            if d[col].isna().any():
                d[col] = d[col].fillna(0)
    return d

# ============================================================
# 6. PREPARE TRAINING DATA
# ============================================================
print('\nPreparing training data...')
train_scored = train_y[train_y['overdose_category'].isin(SCORING)].copy()
train_merged = train_scored.merge(train_x, on=['period_id', 'jurisdiction'], how='left')
train_fe = engineer_features(train_merged)
train_fe['rate_log'] = np.log1p(train_fe['rate_per_10000_ed_visits'])

# Compute full-train target features
full_jur, full_subcat, full_baselines = compute_target_features(train_fe, train_y, period_order)
train_fe = merge_target_features(train_fe, full_jur, full_subcat)

# Feature list (exclude precip_in per Task 2 gate)
EXCLUDE = {'period_id', 'rate_per_10000_ed_visits', 'rate_log', 'row_id',
           'state_doh_release', 'overdose_category', 'precip_in'}
ALL_FEATURES = [c for c in train_fe.columns
                if c not in EXCLUDE
                and train_fe[c].dtype in ['float64','float32','int64','int32','int8','uint8','bool','category']]
print(f'Total features (precip_in excluded): {len(ALL_FEATURES)}')

# ============================================================
# 7. PER-CATEGORY FEATURE SELECTION
# ============================================================
print('\nFeature selection...')
cat_selected = {}
for cat in SCORING:
    mask = train_fe['overdose_category'] == cat
    ct = train_fe[mask].reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)
    model = None
    for tr, va in gkf.split(ct[ALL_FEATURES], ct['rate_per_10000_ed_visits'], ct['period_id']):
        dtrain = lgb.Dataset(ct[ALL_FEATURES].iloc[tr], label=ct['rate_per_10000_ed_visits'].iloc[tr])
        dval = lgb.Dataset(ct[ALL_FEATURES].iloc[va], label=ct['rate_per_10000_ed_visits'].iloc[va])
        model = lgb.train(LGB_CONFIGS[0], dtrain, 3000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    fi = pd.DataFrame({'f': ALL_FEATURES, 'imp': model.feature_importance('gain')}).sort_values('imp', ascending=False)
    cat_selected[cat] = fi.head(TOP_N_FEATURES)['f'].tolist()
    print(f'  {cat}: top-5 = {fi.head(5)["f"].tolist()}')

# ============================================================
# 8. PREPARE INFERENCE DATA
# ============================================================
print('\nPreparing inference data...')
# Merge inference covariates with submission template to get categories
infer_merged = sub_tmpl[['row_id', 'period_id', 'jurisdiction', 'overdose_category']].merge(
    infer_x, on=['period_id', 'jurisdiction'], how='left')
infer_fe = engineer_features(infer_merged)

# Task 3.3: defensive assert at top of inference path. After column-handling,
# all expected feature columns (representative subset spanning every FE family)
# must exist in the inference frame. Fails fast on schema break.
_EXPECTED_INFER_FE_COLS = [
    'zscore_gtrends_overdose', 'ratio_fent_od',
    'period_mean_gtrends_fentanyl', 'dev_gtrends_overdose',
    'period_ratio_fent_od', 'gtrends_sum',
    'log_labor_force', 'log_gtrends_overdose',
    'has_text', 'text_length', 'text_word_count',
    'kw_fentanyl', 'kw_total', 'rtype_surveillance',
    'n_crisis_indicators', 'n_releases',
]
_missing_infer = [c for c in _EXPECTED_INFER_FE_COLS if c not in infer_fe.columns]
assert not _missing_infer, f'Inference frame missing expected feature columns: {_missing_infer}'

infer_fe = merge_target_features(infer_fe, full_jur, full_subcat)

# ============================================================
# 9. TRAIN FINAL MODELS + PREDICT
# ============================================================
print('\nTraining final models and predicting...')
predictions = pd.Series(np.nan, index=sub_tmpl.index)

# STAGE-2 SAFETY FALLBACK trigger (computed once, expanded to submission grain)
# Guarded: when STAGE2_SAFE_FALLBACK is False, this block is skipped entirely
# and the flag=False inference path is byte-for-byte identical to prefix.
trigger_mask_sub = None
if STAGE2_SAFE_FALLBACK:
    trigger_cov, _ = compute_fallback_trigger(infer_x, train_x)
    trig_df = infer_x[['period_id', 'jurisdiction']].assign(_trig=trigger_cov)
    sub_with_trig = sub_tmpl[['row_id', 'period_id', 'jurisdiction']].merge(
        trig_df, on=['period_id', 'jurisdiction'], how='left')
    trigger_mask_sub = sub_with_trig['_trig'].values.astype(bool)
    n_trig_sub = int(trigger_mask_sub.sum())
    sys.stderr.write(f'[FALLBACK] submission-level trigger: {n_trig_sub}/{len(trigger_mask_sub)} rows\n')

for cat in SCORING:
    sel = cat_selected[cat]
    mask = train_fe['overdose_category'] == cat
    cat_train = train_fe[mask].reset_index(drop=True)
    cat_infer = infer_fe[infer_fe['overdose_category'] == cat].copy()

    X = cat_train[sel]; Xi = cat_infer[sel]
    y_raw = cat_train['rate_per_10000_ed_visits']
    y_log = np.log1p(y_raw)

    raw_map, log_map = full_baselines[cat]
    raw_bl = cat_train['jurisdiction'].astype(str).map(raw_map)
    log_bl = cat_train['jurisdiction'].astype(str).map(log_map)
    raw_bl_i = cat_infer['jurisdiction'].astype(str).map(raw_map)
    log_bl_i = cat_infer['jurisdiction'].astype(str).map(log_map)

    # Handle any jurisdictions not seen in train
    raw_bl_i = raw_bl_i.fillna(raw_map.mean())
    log_bl_i = log_bl_i.fillna(log_map.mean())

    y_resid = y_raw - raw_bl
    y_log_resid = y_log - log_bl

    # Direct residual
    val_dr = np.zeros(len(cat_infer))
    for cfg in LGB_CONFIGS:
        dt = lgb.Dataset(X, label=y_resid)
        m = lgb.train(cfg, dt, 3000, callbacks=[lgb.log_evaluation(0)])
        val_dr += m.predict(Xi) / len(LGB_CONFIGS)
    val_dr += raw_bl_i.values

    # Log residual
    val_lr = np.zeros(len(cat_infer))
    for cfg in LGB_CONFIGS:
        dt = lgb.Dataset(X, label=y_log_resid)
        m = lgb.train(cfg, dt, 3000, callbacks=[lgb.log_evaluation(0)])
        val_lr += m.predict(Xi) / len(LGB_CONFIGS)
    val_lr_raw = np.expm1(val_lr + log_bl_i.values)

    # Ensemble
    val_ens = (val_dr + val_lr_raw) / 2

    # Clip to train bounds
    lo, hi = cat_bounds[cat]
    val_ens = np.clip(val_ens, lo, hi)

    # STAGE-2 SAFETY FALLBACK: blend drift baseline + model pred for triggered rows.
    # Guarded: skipped entirely when flag is False. raw_bl_i already computed above
    # (drift jur-mean mapped to inference jurisdictions, with unseen-jur mean fallback).
    # Trigger is row-level at (period,jur); since covariates are shared across cats,
    # a trigger applies to all 3 categories' rows at that position.
    if STAGE2_SAFE_FALLBACK and trigger_mask_sub is not None:
        trigger_cat = trigger_mask_sub[cat_infer.index]
        if trigger_cat.any():
            n_trig_cat = int(trigger_cat.sum())
            val_ens = np.where(
                trigger_cat,
                FALLBACK_BASELINE_WEIGHT * raw_bl_i.values + (1.0 - FALLBACK_BASELINE_WEIGHT) * val_ens,
                val_ens,
            )
            sys.stderr.write(f'[FALLBACK] cat={cat}: {n_trig_cat}/{len(cat_infer)} rows blended (w={FALLBACK_BASELINE_WEIGHT})\n')

    predictions.iloc[cat_infer.index] = val_ens
    print(f'  {cat}: mean={val_ens.mean():.2f} range=[{val_ens.min():.2f}, {val_ens.max():.2f}]')

# ============================================================
# 10. WRITE SUBMISSION (schema-portable)
# ============================================================
out = sub_tmpl[['row_id']].copy()
out['rate_per_10000_ed_visits'] = predictions.values

# Schema-portable assertions
assert set(out['row_id']) == set(sub_tmpl['row_id']), "row_id mismatch with template"
assert len(out) == len(sub_tmpl), f"Row count mismatch: {len(out)} vs {len(sub_tmpl)}"
assert out['row_id'].is_unique, "Duplicate row_ids"
assert np.isfinite(out['rate_per_10000_ed_visits']).all(), "Non-finite predictions"
assert out.shape[1] == 2, f"Expected 2 columns, got {out.shape[1]}"
assert list(out.columns) == ['row_id', 'rate_per_10000_ed_visits'], "Wrong column names"

out.to_csv(OUT, index=False)
print(f'\nWrote {OUT} — shape {out.shape}')
print(f'Row count derived from template: {len(sub_tmpl)}')
print(f'Scoring categories derived from template: {SCORING}')
