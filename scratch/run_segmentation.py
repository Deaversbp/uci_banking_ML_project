import sys
import json
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.data.load_data import load_bank_marketing_data
from src.features.build_features import split_data
from src.models.segmentation import (
    build_clustering_feature_matrix,
    compute_pca_summary,
    evaluate_kmeans_candidates,
    evaluate_cluster_stability,
    fit_final_segmentation,
    profile_clusters,
    generate_segmentation_figures,
)

print("Loading data...")
X, y = load_bank_marketing_data()
X_dev, _, y_dev, _ = split_data(X, y)

print("Building compliant clustering matrix...")
X_prep, feature_names, preprocessor, X_fe = build_clustering_feature_matrix(X_dev)
print(f"Matrix shape: {X_prep.shape} ({len(feature_names)} features)")

print("Computing PCA summary...")
pca, pca_summary = compute_pca_summary(X_prep, feature_names, random_state=42)

print("Evaluating K-Means candidates k=2..8...")
km_eval_df = evaluate_kmeans_candidates(X_prep, k_range=range(2, 9), random_state=42, silhouette_sample_size=10000)

print("Evaluating cluster stability...")
stability_df = evaluate_cluster_stability(X_prep, k_values=range(2, 9), n_seeds=5, subsample_ratio=0.8, random_state=42)

print("Fitting final segmentation k=3...")
km_final, cluster_labels = fit_final_segmentation(X_prep, k=3, random_state=42)

# Load compliant OOF RF scores
scratch_dir = Path("C:/Users/usasl/.gemini/antigravity-ide/brain/1d659fec-07ea-4468-bd08-0f99d30458b8/scratch")
oof_df = pd.read_csv(scratch_dir / "oof_calibrated_predictions.csv")
rf_scores = oof_df["raw_score"].values

print("Profiling final clusters...")
profile_summary = profile_clusters(
    X_fe=X_fe,
    cluster_labels=cluster_labels,
    y_dev=y_dev,
    rf_scores=rf_scores,
    k_capacity=4000,
)

print("Generating figures 13-18...")
figures = generate_segmentation_figures(
    X_prep=X_prep,
    pca=pca,
    km_eval_df=km_eval_df,
    stability_df=stability_df,
    cluster_labels=cluster_labels,
    profile_summary=profile_summary,
    output_dir=Path("reports/figures"),
)

for k_name, fig_path in figures.items():
    print(f"Generated {k_name}: {fig_path} ({fig_path.stat().st_size} bytes)")

# Save JSON results to scratch
results = {
    "matrix_shape": list(X_prep.shape),
    "feature_names": feature_names,
    "pca_summary": pca_summary,
    "km_eval": km_eval_df.to_dict(orient="records"),
    "stability": stability_df.to_dict(orient="records"),
    "profile_summary": profile_summary,
}

with open(scratch_dir / "segmentation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("All segmentation computations and figures complete!")
