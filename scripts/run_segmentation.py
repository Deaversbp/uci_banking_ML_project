"""Orchestrate parsimonious unsupervised segmentation, PCA, stability, and figure generation."""
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load_data import load_bank_marketing_data
from src.features.build_features import split_data
from src.features.feature_contract import select_canonical_pre_campaign_features
from src.models.diagnostics import generate_oof_predictions
from src.models.segmentation import (
    build_clustering_feature_matrix,
    compute_pca_summary,
    evaluate_kmeans_candidates,
    evaluate_cluster_stability,
    fit_final_segmentation,
    profile_clusters,
    generate_segmentation_figures,
)

def run_segmentation():
    print("Loading data for unsupervised customer segmentation...")
    X, y = load_bank_marketing_data()
    X_dev, _, y_dev, _ = split_data(X, y, test_size=0.2, random_state=42)
    X_clean = select_canonical_pre_campaign_features(X_dev)

    print("Building compliant parsimonious clustering matrix...")
    X_prep, feature_names, preprocessor, X_fe = build_clustering_feature_matrix(X_clean)
    print(f"Matrix shape: {X_prep.shape} ({len(feature_names)} features)")

    print("Computing PCA summary...")
    pca, pca_summary = compute_pca_summary(X_prep, feature_names, random_state=42)

    print("Evaluating K-Means candidates k=2..8...")
    km_eval_df = evaluate_kmeans_candidates(X_prep, k_range=range(2, 9), random_state=42, silhouette_sample_size=10000)

    print("Evaluating cluster stability across seeds and bootstrap subsamples...")
    stability_df = evaluate_cluster_stability(X_prep, k_values=range(2, 9), n_seeds=5, subsample_ratio=0.8, random_state=42)

    print("Fitting final segmentation (k=3)...")
    km_final, cluster_labels = fit_final_segmentation(X_prep, k=3, random_state=42)

    print("Generating compliant OOF RF scores for post-hoc ranking cross-tabulation...")
    df_oof = generate_oof_predictions(X_clean, y_dev, n_splits=5, random_state=42)
    rf_scores = df_oof["score"].values

    print("Profiling final clusters...")
    profile_summary = profile_clusters(
        X_fe=X_fe,
        cluster_labels=cluster_labels,
        y_dev=y_dev,
        rf_scores=rf_scores,
        k_capacity=4000,
    )

    print("Generating segmentation figures (13-18)...")
    fig_paths = generate_segmentation_figures(
        X_prep=X_prep,
        pca=pca,
        km_eval_df=km_eval_df,
        stability_df=stability_df,
        cluster_labels=cluster_labels,
        profile_summary=profile_summary,
        output_dir=project_root / "reports" / "figures",
    )
    for name, path in fig_paths.items():
        print(f"Generated {name}: {path}")

    print("Unsupervised customer segmentation complete.")
    return {
        "pca_summary": pca_summary,
        "km_eval_df": km_eval_df,
        "stability_df": stability_df,
        "profile_summary": profile_summary,
        "fig_paths": fig_paths,
    }

if __name__ == "__main__":
    run_segmentation()
