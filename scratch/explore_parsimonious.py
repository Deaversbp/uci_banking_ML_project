import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.data.load_data import load_bank_marketing_data
from src.features.build_features import split_data, PreCallFeatureEngineer
from src.features.feature_contract import select_canonical_pre_campaign_features

X, y = load_bank_marketing_data()
X_dev, _, y_dev, _ = split_data(X, y)
X_clean = select_canonical_pre_campaign_features(X_dev)
fe = PreCallFeatureEngineer(enforce_contract=True, drop_leakage=True)
X_fe = fe.fit_transform(X_clean)

num_cols = ['age', 'balance_log', 'previous', 'pdays_recency']
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
for c in cat_cols:
    X_fe[c] = X_fe[c].fillna('unknown').astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None), cat_cols),
    ]
)
X_prep = preprocessor.fit_transform(X_fe)

# Evaluate K-means k=2..8
eval_records = []
rng = np.random.RandomState(42)
sample_indices = rng.choice(len(X_prep), size=10000, replace=False)
X_sample = X_prep[sample_indices]

for k in range(2, 9):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_prep)
    inertia = km.inertia_
    sil = silhouette_score(X_sample, labels[sample_indices])
    counts = pd.Series(labels).value_counts().sort_index()
    props = counts / len(labels) * 100
    eval_records.append({
        'k': k,
        'inertia': float(inertia),
        'silhouette': float(sil),
        'min_prop': float(props.min()),
        'max_prop': float(props.max()),
        'smallest_cluster_size': int(counts.min()),
        'largest_cluster_size': int(counts.max()),
        'counts': counts.to_dict(),
    })

eval_df = pd.DataFrame(eval_records)
print("K-Means Candidate Evaluation:")
print(eval_df[['k', 'inertia', 'silhouette', 'min_prop', 'max_prop', 'smallest_cluster_size', 'largest_cluster_size']])
for r in eval_records:
    print(f"k={r['k']}: {r['counts']}")

# Stability evaluation
stability_records = []
n_seeds = 5
subsample_ratio = 0.8
for k in range(2, 9):
    # Seed stability
    seed_models = []
    for s in [42, 123, 456, 789, 999]:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=s)
        seed_models.append(km.fit_predict(X_prep))
    seed_aris = []
    for i in range(len(seed_models)):
        for j in range(i + 1, len(seed_models)):
            seed_aris.append(adjusted_rand_score(seed_models[i], seed_models[j]))

    # Subsample stability
    sub_aris = []
    for s in [42, 123, 456, 789, 999]:
        sub_rng = np.random.RandomState(s)
        sub_idx1 = sub_rng.choice(len(X_prep), size=int(len(X_prep) * subsample_ratio), replace=False)
        sub_idx2 = sub_rng.choice(len(X_prep), size=int(len(X_prep) * subsample_ratio), replace=False)
        overlap = np.intersect1d(sub_idx1, sub_idx2)
        
        km1 = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km2 = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=123)
        l1 = km1.fit_predict(X_prep[sub_idx1])
        l2 = km2.fit_predict(X_prep[sub_idx2])
        
        map1 = {idx: l1[i] for i, idx in enumerate(sub_idx1)}
        map2 = {idx: l2[i] for i, idx in enumerate(sub_idx2)}
        l1_over = [map1[idx] for idx in overlap]
        l2_over = [map2[idx] for idx in overlap]
        sub_aris.append(adjusted_rand_score(l1_over, l2_over))

    stability_records.append({
        'k': k,
        'seed_ari_mean': float(np.mean(seed_aris)),
        'seed_ari_std': float(np.std(seed_aris)),
        'seed_ari_min': float(np.min(seed_aris)),
        'subsample_ari_mean': float(np.mean(sub_aris)),
        'subsample_ari_std': float(np.std(sub_aris)),
        'subsample_ari_min': float(np.min(sub_aris)),
    })

stab_df = pd.DataFrame(stability_records)
print("\nStability Analysis:")
print(stab_df)
