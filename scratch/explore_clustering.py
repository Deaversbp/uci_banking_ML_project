import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from src.data.load_data import load_bank_marketing_data
from src.features.build_features import split_data, PreCallFeatureEngineer
from src.features.feature_contract import validate_pre_campaign_feature_contract

# 1. Load data
X, y = load_bank_marketing_data()
X_dev, _, y_dev, _ = split_data(X, y)
fe = PreCallFeatureEngineer(enforce_contract=True)
X_fe = fe.fit_transform(X_dev)

# Verify no forbidden features in X_fe
validate_pre_campaign_feature_contract(X_fe.columns)

num_cols = ["age", "balance_log", "previous", "pdays_recency", "was_previously_contacted"]
cat_cols = ["job", "marital", "education", "default", "housing", "loan", "poutcome"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
    ]
)
X_prep = preprocessor.fit_transform(X_fe)
feat_names = list(preprocessor.get_feature_names_out())
print(f"Matrix shape: {X_prep.shape} ({len(num_cols)} numeric, {len(feat_names)-len(num_cols)} categorical)")

# 2. PCA
pca = PCA(random_state=42).fit(X_prep)
cum_var = np.cumsum(pca.explained_variance_ratio_)
print("\n=== PCA Variance ===")
for p in [0.5, 0.7, 0.8, 0.9]:
    n_c = int(np.argmax(cum_var >= p) + 1)
    print(f"Components for {int(p*100)}% variance: {n_c}/{len(cum_var)} (exact: {cum_var[n_c-1]:.4f})")

print("Top 5 PC ratios:", np.round(pca.explained_variance_ratio_[:5], 4))

# Inspect major loadings for PC1 and PC2
loadings = pd.DataFrame(pca.components_[:2, :].T, index=feat_names, columns=["PC1", "PC2"])
print("\nTop 5 absolute loadings for PC1:")
print(loadings["PC1"].abs().sort_values(ascending=False).head(7))
print("\nTop 5 absolute loadings for PC2:")
print(loadings["PC2"].abs().sort_values(ascending=False).head(7))

# 3. K-Means evaluation for k = 2..8
print("\n=== Candidate k Evaluation (k=2..8) ===")
# We compute silhouette on a 10,000 stratified subsample for deterministic high speed
np.random.seed(42)
sample_idx = np.random.choice(len(X_prep), size=10000, replace=False)

k_results = []
seeds = [42, 100, 2024, 7, 99]

for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_prep)
    inertia = km.inertia_
    sil = float(silhouette_score(X_prep[sample_idx], km.labels_[sample_idx]))
    counts = np.bincount(km.labels_)
    props = counts / len(X_prep)
    min_size = int(counts.min())
    max_size = int(counts.max())
    
    # Seed stability (ARI across pairs of seeds)
    seed_labels = [KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(X_prep) for s in seeds]
    seed_aris = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            seed_aris.append(adjusted_rand_score(seed_labels[i], seed_labels[j]))
    mean_seed_ari = float(np.mean(seed_aris))
    std_seed_ari = float(np.std(seed_aris))
    min_seed_ari = float(np.min(seed_aris))
    
    # Resampling stability (subsample ARI)
    resamp_aris = []
    for s_idx, s in enumerate(seeds):
        rng = np.random.RandomState(s)
        sub_idx = rng.choice(len(X_prep), size=int(len(X_prep) * 0.8), replace=False)
        km_sub = KMeans(n_clusters=k, random_state=s, n_init=10).fit(X_prep[sub_idx])
        pred_full = km.predict(X_prep[sub_idx])
        resamp_aris.append(adjusted_rand_score(km_sub.labels_, pred_full))
    mean_resamp_ari = float(np.mean(resamp_aris))
    std_resamp_ari = float(np.std(resamp_aris))
    
    k_results.append({
        "k": k,
        "inertia": float(inertia),
        "silhouette": sil,
        "counts": [int(c) for c in counts],
        "proportions": [float(p) for p in props],
        "min_size": min_size,
        "max_size": max_size,
        "mean_seed_ari": mean_seed_ari,
        "std_seed_ari": std_seed_ari,
        "min_seed_ari": min_seed_ari,
        "mean_resamp_ari": mean_resamp_ari,
        "std_resamp_ari": std_resamp_ari,
    })
    print(f"k={k}: Inertia={inertia:.1f}, Sil={sil:.4f}, MinSize={min_size}, MaxSize={max_size}, "
          f"SeedARI={mean_seed_ari:.4f} (+/- {std_seed_ari:.4f}), ResampARI={mean_resamp_ari:.4f}")

# 4. Profile k=2, 3, 4, 5
y_bin = (y_dev.values == "yes").astype(int)

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_prep)
    df_prof = X_fe.copy()
    df_prof["cluster"] = km.labels_
    df_prof["actual"] = y_bin
    
    print(f"\n==================== PROFILE FOR K={k} ====================")
    for c in range(k):
        sub = df_prof[df_prof["cluster"] == c]
        conv_rate = float(sub["actual"].mean())
        prev_cont = float(sub["was_previously_contacted"].mean())
        housing_rate = float((sub["housing"] == "yes").mean())
        loan_rate = float((sub["loan"] == "yes").mean())
        mean_age = float(sub["age"].mean())
        med_bal = float(sub["balance"].median())
        top_job = sub["job"].value_counts().index[0]
        top_edu = sub["education"].value_counts().index[0]
        top_mar = sub["marital"].value_counts().index[0]
        
        print(f"Cluster {c} (N={len(sub):,}, {len(sub)/len(df_prof):.1%}): "
              f"ConvRate={conv_rate:.2%}, PrevContact={prev_cont:.1%}, "
              f"Housing={housing_rate:.1%}, Loan={loan_rate:.1%}, "
              f"Age={mean_age:.1f}, MedBal={med_bal:.0f}, "
              f"Job={top_job}, Edu={top_edu}, Marital={top_mar}")
