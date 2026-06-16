"""
Fabricate two test datasets for Award B dry-run validation:
(A) Regression: predict house prices from features
(B) Classification: predict customer churn (binary)
"""
import numpy as np, pandas as pd, os
np.random.seed(42)

out_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# (A) REGRESSION: California-style housing
# ============================================================
n_train = 500; n_test = 100
n_loc = 20
locations = [f'LOC_{i:03d}' for i in range(n_loc)]

def make_housing(n, locations, is_train=True):
    loc = np.random.choice(locations, n)
    rooms = np.random.uniform(2, 8, n)
    income = np.random.uniform(1, 15, n)
    age = np.random.uniform(1, 50, n)
    noise = np.random.normal(0, 20, n)
    price = 50 * income + 15 * rooms - 0.5 * age + noise + 100
    price = np.clip(price, 50, 900)
    df = pd.DataFrame({
        'item_id': range(n),
        'location': loc,
        'avg_rooms': np.round(rooms, 1),
        'median_income': np.round(income, 1),
        'house_age': np.round(age, 1),
    })
    if is_train:
        df['sale_price'] = np.round(price, 0)
    return df

train_a = make_housing(n_train, locations, True)
test_a = make_housing(n_test, locations, False)
sub_a = test_a[['item_id']].copy()
sub_a['sale_price'] = 0.0

reg_dir = f'{out_dir}/dry_run_regression'
os.makedirs(f'{reg_dir}/data', exist_ok=True)
train_a.to_csv(f'{reg_dir}/data/train.csv', index=False)
test_a.to_csv(f'{reg_dir}/data/test.csv', index=False)
sub_a.to_csv(f'{reg_dir}/data/sample_submission.csv', index=False)
print(f'Regression dataset: train={train_a.shape} test={test_a.shape} sub={sub_a.shape}')

# ============================================================
# (B) CLASSIFICATION: Customer churn
# ============================================================
n_train_b = 600; n_test_b = 150

def make_churn(n, is_train=True):
    tenure = np.random.randint(1, 72, n)
    monthly = np.random.uniform(20, 120, n)
    support_calls = np.random.poisson(2, n)
    contract = np.random.choice(['monthly', 'annual', 'two_year'], n, p=[0.5, 0.3, 0.2])
    log_odds = -2.0 + 0.02 * support_calls + 0.01 * monthly - 0.03 * tenure
    if contract is not None:
        log_odds += np.where(np.array(contract) == 'monthly', 1.0,
                    np.where(np.array(contract) == 'annual', 0.0, -0.5))
    prob = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.uniform(0, 1, n) < prob).astype(int)
    df = pd.DataFrame({
        'customer_id': range(n),
        'tenure_months': tenure,
        'monthly_charges': np.round(monthly, 2),
        'support_calls': support_calls,
        'contract_type': contract,
    })
    if is_train:
        df['churn'] = churn
    return df

train_b = make_churn(n_train_b, True)
test_b = make_churn(n_test_b, False)
sub_b = test_b[['customer_id']].copy()
sub_b['churn'] = 0

clf_dir = f'{out_dir}/dry_run_classification'
os.makedirs(f'{clf_dir}/data', exist_ok=True)
train_b.to_csv(f'{clf_dir}/data/train.csv', index=False)
test_b.to_csv(f'{clf_dir}/data/test.csv', index=False)
sub_b.to_csv(f'{clf_dir}/data/sample_submission.csv', index=False)
print(f'Classification dataset: train={train_b.shape} test={test_b.shape} sub={sub_b.shape}')

print(f'\nDatasets written to:')
print(f'  {reg_dir}/data/')
print(f'  {clf_dir}/data/')
