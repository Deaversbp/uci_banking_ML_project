"""Unsupervised Customer Segmentation and Dimensionality Analysis.

Methodological Guarantees:
- Strictly Unsupervised: Target 'y' is NEVER accessed during feature selection, preprocessing,
  PCA fitting, cluster count evaluation, stability testing, or cluster assignment.
- Strict Pre-Campaign Contract: Enforces canonical 11 raw pre-campaign features.
  Purges all post-decision execution variables (duration, contact, month, day, campaign, y).
- Non-Redundant Feature Contract: Avoids double-counting constructs (uses balance_log over raw balance,
  pdays_recency + was_previously_contacted over sentinel pdays, raw loans over has_debt_burden,
  and multi-level poutcome over prior_success).
- Partition Isolation: Operates strictly on the 80% development partition (36,168 records);
  the held-out test partition is never accessed.
- Post-Hoc Analysis: Target 'y' and supervised Random Forest scores are reintroduced strictly
  after cluster assignments are frozen to provide descriptive profiling and ranking comparison.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.features.build_features import PreCallFeatureEngineer
from src.features.feature_contract import (
    select_canonical_pre_campaign_features,
    validate_pre_campaign_feature_contract,
    CANONICAL_FORBIDDEN_FEATURES,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Centralized clustering feature specification
# Note: 'was_previously_contacted' is excluded from clustering coordinates because it is exactly
# identical to (pdays_recency > 0) on this dataset (pdays == 0 never occurs), which would give
# the prior-contact construct duplicate Euclidean weight. It is retained for descriptive profiling.
CLUSTERING_NUMERIC_FEATURES: List[str] = [
    "age",
    "balance_log",
    "previous",
    "pdays_recency",
]

CLUSTERING_CATEGORICAL_FEATURES: List[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "poutcome",
]


def build_clustering_feature_matrix(
    X: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], ColumnTransformer, pd.DataFrame]:
    """Prepare a standardized, leak-free, parsimonious clustering feature matrix.

    Clustering Feature Contract:
    - Enforces canonical pre-campaign contract; strips all forbidden operational fields.
    - Non-Redundant Concept Representation:
      - 'balance_log' over raw 'balance' and 'negative_balance_flag': Preserves signed magnitude
        while taming extreme financial outliers that distort Euclidean distance.
      - 'pdays_recency' over raw 'pdays' and 'was_previously_contacted': Since pdays == 0 never occurs,
        'was_previously_contacted' == (pdays_recency > 0) exactly for every row. Including both would
        double-count the prior-contact construct in Euclidean distance.
      - 'housing' and 'loan' over 'has_debt_burden': Preserves distinct liability types without double counting.
      - Multi-level 'poutcome' over 'prior_success': Preserves full outcome structure without duplicate coordinates.
    - Categorical Encoding:
      - Full one-hot encoding (drop=None) is explicitly used: while it introduces linearly dependent dummy
        columns, it provides symmetric distance treatment across all categorical levels, which is
        desirable for distance-based K-Means geometry.
    - Explicitly Excluded:
      - 'duration', 'contact', 'month', 'day', 'day_of_week', 'contact_day_of_month', 'campaign', 'y'.

    Args:
        X: Raw feature DataFrame.

    Returns:
        Tuple containing:
            - X_prep: Standardized, one-hot encoded numpy array.
            - feature_names: List of column names in X_prep.
            - preprocessor: Fitted ColumnTransformer.
            - X_fe: Feature-engineered DataFrame.
    """
    X_clean = select_canonical_pre_campaign_features(X)
    validate_pre_campaign_feature_contract(X_clean, raise_on_violation=True)

    fe = PreCallFeatureEngineer(enforce_contract=True, drop_leakage=True)
    X_fe = fe.fit_transform(X_clean)

    # Handle categorical missing values consistently
    for col in CLUSTERING_CATEGORICAL_FEATURES:
        X_fe[col] = X_fe[col].fillna("unknown").astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CLUSTERING_NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                CLUSTERING_CATEGORICAL_FEATURES,
            ),
        ]
    )

    X_prep = preprocessor.fit_transform(X_fe)
    feature_names = list(preprocessor.get_feature_names_out())

    logger.info(
        f"Built compliant clustering matrix with {X_prep.shape[0]:,} rows and {X_prep.shape[1]} features "
        f"({len(CLUSTERING_NUMERIC_FEATURES)} numeric, {len(feature_names)-len(CLUSTERING_NUMERIC_FEATURES)} one-hot categorical)."
    )
    return X_prep, feature_names, preprocessor, X_fe


def compute_pca_summary(
    X_prep: np.ndarray,
    feature_names: List[str],
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[PCA, Dict[str, Any]]:
    """Perform PCA on preprocessed clustering matrix and compute explained variance summary.

    Args:
        X_prep: Preprocessed feature matrix.
        feature_names: List of feature names.
        n_components: Number of components (default: all).
        random_state: Random state for deterministic PCA.

    Returns:
        Tuple of (fitted PCA model, summary dictionary).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_prep)

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    thresholds = [0.50, 0.70, 0.80, 0.90]
    variance_threshold_components: Dict[str, int] = {}
    for th in thresholds:
        n_comp = int(np.argmax(cum_evr >= th) + 1)
        variance_threshold_components[f"{int(th*100)}%"] = n_comp

    # Top loadings for PC1 and PC2
    pc1_series = pd.Series(pca.components_[0], index=feature_names)
    pc2_series = pd.Series(pca.components_[1], index=feature_names)

    top_pc1_loadings = pc1_series.abs().sort_values(ascending=False).head(10).to_dict()
    top_pc2_loadings = pc2_series.abs().sort_values(ascending=False).head(10).to_dict()

    summary: Dict[str, Any] = {
        "n_features": len(feature_names),
        "total_components": len(evr),
        "pc1_explained_variance": float(evr[0]),
        "pc2_explained_variance": float(evr[1]),
        "pc1_pc2_cumulative_variance": float(cum_evr[1]),
        "variance_threshold_components": variance_threshold_components,
        "explained_variance_ratio": evr.tolist(),
        "cumulative_explained_variance": cum_evr.tolist(),
        "top_pc1_loadings": top_pc1_loadings,
        "top_pc2_loadings": top_pc2_loadings,
    }

    return pca, summary


def evaluate_kmeans_candidates(
    X_prep: np.ndarray,
    k_range: Sequence[int] = range(2, 9),
    random_state: int = 42,
    silhouette_sample_size: Optional[int] = 10000,
) -> pd.DataFrame:
    """Evaluate K-Means across candidate cluster counts k=2..8.

    Args:
        X_prep: Preprocessed feature matrix.
        k_range: Range of k values to evaluate.
        random_state: Random state for K-Means initialization.
        silhouette_sample_size: Subsample size for silhouette score computation (for speed/determinism).

    Returns:
        pd.DataFrame: Table containing k, inertia, silhouette score, and cluster size distributions.
    """
    n_samples = len(X_prep)
    if silhouette_sample_size is not None and silhouette_sample_size < n_samples:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(n_samples, size=silhouette_sample_size, replace=False)
    else:
        sample_idx = np.arange(n_samples)

    rows: List[Dict[str, Any]] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_prep)
        inertia = float(km.inertia_)

        # Silhouette score on representative sample
        sil = float(silhouette_score(X_prep[sample_idx], labels[sample_idx]))

        counts = pd.Series(labels).value_counts().sort_index()
        min_cluster_prop = float(counts.min() / n_samples * 100)
        max_cluster_prop = float(counts.max() / n_samples * 100)

        rows.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": sil,
            "min_cluster_prop": min_cluster_prop,
            "max_cluster_prop": max_cluster_prop,
            "smallest_cluster_size": int(counts.min()),
            "largest_cluster_size": int(counts.max()),
            "cluster_sizes": counts.to_dict(),
        })

    return pd.DataFrame(rows)


def evaluate_cluster_stability(
    X_prep: np.ndarray,
    k_values: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    n_seeds: int = 5,
    subsample_ratio: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate cluster stability using Adjusted Rand Index (ARI) across random seeds and subsamples.

    Args:
        X_prep: Preprocessed feature matrix.
        k_values: Sequence of candidate k values.
        n_seeds: Number of distinct random seeds to compare.
        subsample_ratio: Fraction of data for bootstrap subsampling.
        random_state: Base random state.

    Returns:
        pd.DataFrame: Stability summary table with seed ARI and subsample ARI statistics.
    """
    n_samples = len(X_prep)
    n_sub = int(n_samples * subsample_ratio)
    rows: List[Dict[str, Any]] = []

    seed_list = [42, 100, 2024, 7, 99][:n_seeds]

    for k in k_values:
        # 1. Random seed stability across pairs of seeds
        seed_labels = [KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(X_prep) for s in seed_list]
        seed_aris: List[float] = []
        for i in range(len(seed_list)):
            for j in range(i + 1, len(seed_list)):
                seed_aris.append(float(adjusted_rand_score(seed_labels[i], seed_labels[j])))

        # 2. Subsample stability
        ref_km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X_prep)
        sub_aris: List[float] = []
        for s in seed_list:
            rng = np.random.RandomState(s)
            sub_idx = rng.choice(n_samples, size=n_sub, replace=False)
            km_sub = KMeans(n_clusters=k, random_state=s, n_init=10).fit(X_prep[sub_idx])
            pred_full = ref_km.predict(X_prep[sub_idx])
            sub_aris.append(float(adjusted_rand_score(km_sub.labels_, pred_full)))

        rows.append({
            "k": k,
            "seed_ari_mean": float(np.mean(seed_aris)) if seed_aris else 1.0,
            "seed_ari_std": float(np.std(seed_aris)) if seed_aris else 0.0,
            "seed_ari_min": float(np.min(seed_aris)) if seed_aris else 1.0,
            "subsample_ari_mean": float(np.mean(sub_aris)),
            "subsample_ari_std": float(np.std(sub_aris)),
            "subsample_ari_min": float(np.min(sub_aris)),
        })

    return pd.DataFrame(rows)


def fit_final_segmentation(
    X_prep: np.ndarray,
    k: int = 3,
    random_state: int = 42,
) -> Tuple[KMeans, np.ndarray]:
    """Fit final K-Means segmentation model and return cluster assignments.

    Args:
        X_prep: Preprocessed feature matrix.
        k: Chosen cluster count (default: 3).
        random_state: Random state for deterministic fitting.

    Returns:
        Tuple of (fitted KMeans model, cluster labels array).
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_prep)
    return km, labels


def profile_clusters(
    X_fe: pd.DataFrame,
    cluster_labels: np.ndarray,
    y_dev: Optional[pd.Series] = None,
    rf_scores: Optional[np.ndarray] = None,
    k_capacity: int = 4000,
) -> Dict[str, Any]:
    """Profile clusters on pre-call attributes, post-hoc conversions, and RF ranking overlap.

    Args:
        X_fe: Feature-engineered pre-call DataFrame.
        cluster_labels: Integer cluster labels (0..k-1).
        y_dev: Optional development ground-truth target Series ('yes'/'no').
        rf_scores: Optional out-of-fold supervised Random Forest continuous scores.
        k_capacity: Capacity constraint for outreach ranking (default: 4,000).

    Returns:
        Dict[str, Any]: Comprehensive profiling summary per cluster.
    """
    df_prof = X_fe.copy()
    df_prof["cluster"] = cluster_labels
    n_total = len(df_prof)

    has_target = y_dev is not None
    has_rf = rf_scores is not None

    if has_target:
        df_prof["actual"] = (y_dev.values == "yes").astype(int)
        total_converters = df_prof["actual"].sum()
        base_rate = float(total_converters / n_total * 100)
    else:
        base_rate = np.nan

    if has_rf:
        df_prof["rf_score"] = rf_scores
        # Deterministic top-k cutoff
        ranked_idx = np.lexsort((np.arange(len(rf_scores)), -rf_scores))
        top_k_indices = set(ranked_idx[:k_capacity])
        df_prof["rf_top_k"] = [i in top_k_indices for i in range(len(df_prof))]

    unique_clusters = np.sort(np.unique(cluster_labels))
    cluster_profiles: List[Dict[str, Any]] = []

    for c in unique_clusters:
        sub = df_prof[df_prof["cluster"] == c]
        n_c = len(sub)
        pct_pop = float(n_c / n_total * 100)

        # Pre-call demographics and finances
        mean_age = float(sub["age"].mean())
        med_balance = float(sub["balance"].median()) if "balance" in sub.columns else np.nan
        mean_balance = float(sub["balance"].mean()) if "balance" in sub.columns else np.nan
        pct_housing = float((sub["housing"] == "yes").mean() * 100)
        pct_loan = float((sub["loan"] == "yes").mean() * 100)
        pct_debt_free = float(((sub["housing"] == "no") & (sub["loan"] == "no")).mean() * 100)
        pct_pos_balance = float((sub["balance"] > 0).mean() * 100) if "balance" in sub.columns else np.nan
        pct_neg_balance = float((sub["balance"] < 0).mean() * 100) if "balance" in sub.columns else np.nan

        # Prior contact history
        pct_contacted = float(sub["was_previously_contacted"].mean() * 100)
        pct_prior_success = float((sub["poutcome"] == "success").mean() * 100)
        pct_prior_failure = float((sub["poutcome"] == "failure").mean() * 100)
        pct_prior_other = float((sub["poutcome"] == "other").mean() * 100)
        pct_prior_unknown = float((sub["poutcome"] == "unknown").mean() * 100)
        mean_previous = float(sub["previous"].mean())

        # Dominant jobs
        top_jobs = sub["job"].value_counts(normalize=True).head(3).to_dict()
        top_jobs_fmt = {k: round(v * 100, 1) for k, v in top_jobs.items()}

        # Dominant education and marital
        top_edu = sub["education"].value_counts(normalize=True).head(2).to_dict()
        top_edu_fmt = {k: round(v * 100, 1) for k, v in top_edu.items()}
        top_marital = sub["marital"].value_counts(normalize=True).head(2).to_dict()
        top_marital_fmt = {k: round(v * 100, 1) for k, v in top_marital.items()}

        prof_row: Dict[str, Any] = {
            "cluster_id": int(c),
            "count": n_c,
            "pct_of_population": pct_pop,
            "mean_age": mean_age,
            "median_balance": med_balance,
            "mean_balance": mean_balance,
            "pct_positive_balance": pct_pos_balance,
            "pct_negative_balance": pct_neg_balance,
            "pct_housing_loan": pct_housing,
            "pct_personal_loan": pct_loan,
            "pct_debt_free": pct_debt_free,
            "pct_previously_contacted": pct_contacted,
            "pct_prior_success": pct_prior_success,
            "pct_prior_failure": pct_prior_failure,
            "pct_prior_other": pct_prior_other,
            "pct_prior_unknown": pct_prior_unknown,
            "mean_previous_contacts": mean_previous,
            "top_jobs": top_jobs_fmt,
            "top_education": top_edu_fmt,
            "top_marital": top_marital_fmt,
        }

        # Post-hoc conversion outcome
        if has_target:
            converters = int(sub["actual"].sum())
            conv_rate = float(converters / n_c * 100)
            diff_from_base = float(conv_rate - base_rate)
            prof_row.update({
                "converters": converters,
                "conversion_rate": conv_rate,
                "diff_from_base_rate": diff_from_base,
            })

        # Post-hoc Random Forest ranking connection
        if has_rf:
            mean_rf = float(sub["rf_score"].mean())
            rf_selected = int(sub["rf_top_k"].sum())
            rf_selection_rate = float(rf_selected / n_c * 100)
            if has_target:
                rf_tp = int((sub["rf_top_k"] & (sub["actual"] == 1)).sum())
                rf_prec = float(rf_tp / rf_selected * 100) if rf_selected > 0 else 0.0
                rf_recall = float(rf_tp / converters * 100) if converters > 0 else 0.0
            else:
                rf_tp, rf_prec, rf_recall = np.nan, np.nan, np.nan

            prof_row.update({
                "mean_rf_score": mean_rf,
                "rf_selected_top_k": rf_selected,
                "rf_selection_rate": rf_selection_rate,
                "rf_conversions_captured": rf_tp,
                "rf_precision_at_k": rf_prec,
                "rf_recall_at_k": rf_recall,
            })

        cluster_profiles.append(prof_row)

    return {
        "n_total": n_total,
        "base_rate": base_rate,
        "cluster_profiles": cluster_profiles,
    }


def generate_segmentation_figures(
    X_prep: np.ndarray,
    pca: PCA,
    km_eval_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    profile_summary: Dict[str, Any],
    output_dir: Path | str,
) -> Dict[str, Path]:
    """Generate and save publication-quality figures for the segmentation analysis.

    Figures:
    1. 13_pca_scree_and_variance.png: Scree and cumulative explained variance.
    2. 14_pca_pc1_vs_pc2_clusters.png: PC1 vs PC2 projection colored by final cluster.
    3. 15_kmeans_inertia_and_silhouette.png: Inertia (elbow) and silhouette score vs k.
    4. 16_cluster_stability_ari.png: Stability metrics (ARI) across seeds and subsamples.
    5. 17_cluster_pre_call_profiles.png: Feature comparison bar chart across clusters.
    6. 18_cluster_post_hoc_conversions_and_rf.png: Conversion rate vs RF selection rate by cluster.

    Args:
        X_prep: Preprocessed feature matrix.
        pca: Fitted PCA model.
        km_eval_df: Evaluation table across k=2..8.
        stability_df: Stability evaluation table.
        cluster_labels: Final cluster labels array.
        profile_summary: Output from profile_clusters.
        output_dir: Target output directory.

    Returns:
        Dict[str, Path]: Mapping of figure names to absolute file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_figures: Dict[str, Path] = {}

    cluster_colors = ["#E67E22", "#27AE60", "#2980B9", "#9B59B6", "#E74C3C"]

    # -------------------------------------------------------------
    # Figure 13: PCA Scree & Cumulative Variance
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    evr = pca.explained_variance_ratio_ * 100
    cum_evr = np.cumsum(evr)
    n_plot = min(len(evr), 20)
    components_range = np.arange(1, n_plot + 1)

    ax1.bar(components_range, evr[:n_plot], color="#34495E", alpha=0.85, edgecolor="white")
    ax1.set_title("PCA Scree Plot (Top 20 Components)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Principal Component", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Explained Variance Ratio (%)", fontsize=10, fontweight="bold")
    ax1.set_xticks(components_range)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(components_range, cum_evr[:n_plot], "o-", color="#2980B9", linewidth=2, markersize=5)
    ax2.axhline(50, color="#E67E22", linestyle="--", alpha=0.8, label="50% Threshold (3 PCs)")
    ax2.axhline(70, color="#27AE60", linestyle="--", alpha=0.8, label="70% Threshold (5 PCs)")
    ax2.axhline(80, color="#8E44AD", linestyle="--", alpha=0.8, label="80% Threshold (7 PCs)")
    ax2.axhline(90, color="#C0392B", linestyle="--", alpha=0.8, label="90% Threshold (11 PCs)")
    ax2.set_title("Cumulative Explained Variance", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Number of Principal Components", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=10, fontweight="bold")
    ax2.set_xticks(components_range)
    ax2.set_ylim(0, 102)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)

    f13_path = out_path / "13_pca_scree_and_variance.png"
    plt.savefig(f13_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures["pca_scree"] = f13_path

    # -------------------------------------------------------------
    # Figure 14: PC1 vs PC2 Projection by Cluster
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    X_pca = pca.transform(X_prep)

    # Subsample 5,000 points for readable scatter
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_pca), size=min(5000, len(X_pca)), replace=False)

    unique_c = np.unique(cluster_labels)
    cluster_names = {
        0: "Cluster 0: Zero-Balance & Indebted First-Timers (15.7%)",
        1: "Cluster 1: Previously Contacted Clients (18.2%)",
        2: "Cluster 2: Positive-Balance First-Time Prospects (66.1%)",
    }

    for c in unique_c:
        mask = cluster_labels[sub_idx] == c
        label_text = cluster_names.get(c, f"Cluster {c}")
        col = cluster_colors[c % len(cluster_colors)]
        ax.scatter(
            X_pca[sub_idx[mask], 0],
            X_pca[sub_idx[mask], 1],
            c=col,
            alpha=0.5,
            s=16,
            edgecolors="none",
            label=label_text,
        )

    ax.set_title("PC1 vs. PC2 Projection Colored by Final K-Means Clusters (k=3)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance) — Prior Contact & Recency Axis", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance) — Age & Wealth Accumulation Axis", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    f14_path = out_path / "14_pca_pc1_vs_pc2_clusters.png"
    plt.savefig(f14_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures["pca_projection"] = f14_path

    # -------------------------------------------------------------
    # Figure 15: K-Means Inertia & Silhouette Curves
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))
    k_vals = km_eval_df["k"].values
    inertia_vals = km_eval_df["inertia"].values
    sil_vals = km_eval_df["silhouette_score"].values

    color_in = "#2980B9"
    color_sil = "#E67E22"

    ax1.set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", color=color_in, fontsize=10, fontweight="bold")
    line1 = ax1.plot(k_vals, inertia_vals, "o-", color=color_in, linewidth=2, label="Inertia (Elbow)")
    ax1.tick_params(axis="y", labelcolor=color_in)
    ax1.set_xticks(k_vals)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color=color_sil, fontsize=10, fontweight="bold")
    line2 = ax2.plot(k_vals, sil_vals, "s-", color=color_sil, linewidth=2, label="Silhouette Score")
    ax2.tick_params(axis="y", labelcolor=color_sil)
    ax2.set_ylim(0.10, 0.45)

    lines = line1 + line2
    labels_leg = [l.get_label() for l in lines]
    ax1.legend(lines, labels_leg, loc="upper right", framealpha=0.95, fontsize=9)
    plt.title("K-Means Candidate Evaluation: Inertia and Silhouette across k = 2..8", fontsize=11, fontweight="bold", pad=12)

    f15_path = out_path / "15_kmeans_inertia_and_silhouette.png"
    plt.savefig(f15_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures["kmeans_evaluation"] = f15_path

    # -------------------------------------------------------------
    # Figure 16: Cluster Stability Analysis (ARI)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    k_stab = stability_df["k"].values
    bar_width = 0.35
    x_idx = np.arange(len(k_stab))

    ax.bar(
        x_idx - bar_width/2,
        stability_df["seed_ari_mean"],
        yerr=stability_df["seed_ari_std"],
        capsize=4,
        width=bar_width,
        color="#3498DB",
        alpha=0.85,
        label="Seed Stability (ARI across Random Seeds)",
    )
    ax.bar(
        x_idx + bar_width/2,
        stability_df["subsample_ari_mean"],
        yerr=stability_df["subsample_ari_std"],
        capsize=4,
        width=bar_width,
        color="#2ECC71",
        alpha=0.85,
        label="Subsample Stability (ARI on 80% Bootstrap Resamples)",
    )

    ax.set_title("Cluster Stability Evaluation via Adjusted Rand Index (ARI)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Candidate Number of Clusters (k)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Adjusted Rand Index (1.0 = Perfect Stability)", fontsize=10, fontweight="bold")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(k_stab)
    ax.set_ylim(0.4, 1.08)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9)

    f16_path = out_path / "16_cluster_stability_ari.png"
    plt.savefig(f16_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures["cluster_stability"] = f16_path

    # -------------------------------------------------------------
    # Figure 17: Pre-Call Cluster Characterization
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    prof_list = profile_summary["cluster_profiles"]
    c_labels = [f"Cluster {p['cluster_id']}\n({p['pct_of_population']:.1f}%)" for p in prof_list]

    attributes = [
        ("% Prior Contact", [p["pct_previously_contacted"] for p in prof_list]),
        ("% Housing Loan", [p["pct_housing_loan"] for p in prof_list]),
        ("% Personal Loan", [p["pct_personal_loan"] for p in prof_list]),
        ("% Debt-Free", [p["pct_debt_free"] for p in prof_list]),
        ("% Positive Balance", [p["pct_positive_balance"] for p in prof_list]),
    ]

    x = np.arange(len(c_labels))
    width = 0.16
    offsets = np.linspace(-width * 2, width * 2, len(attributes))

    attr_colors = ["#9B59B6", "#E67E22", "#E74C3C", "#27AE60", "#2980B9"]

    for i, ((name, vals), color, offset) in enumerate(zip(attributes, attr_colors, offsets)):
        ax.bar(x + offset, vals, width=width, label=name, color=color, alpha=0.85)

    ax.set_title("Pre-Call Structural Profile Across Final Clusters (k=3)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Prevalence / Proportion (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(c_labels, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    f17_path = out_path / "17_cluster_pre_call_profiles.png"
    plt.savefig(f17_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures["pre_call_profiles"] = f17_path

    # -------------------------------------------------------------
    # Figure 18: Post-Hoc Conversion & Supervised RF Overlap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x_c = np.arange(len(prof_list))
    width = 0.35

    conv_rates = [p.get("conversion_rate", 0) for p in prof_list]
    rf_rates = [p.get("rf_selection_rate", 0) for p in prof_list]

    ax.bar(x_c - width/2, conv_rates, width=width, color="#2ECC71", alpha=0.85, label="Observed Conversion Rate (%)")
    ax.bar(x_c + width/2, rf_rates, width=width, color="#3498DB", alpha=0.85, label="Random Forest Top-k Selection Rate (%)")
    ax.axhline(profile_summary.get("base_rate", 11.70), color="#E74C3C", linestyle="--", linewidth=1.5, label="Population Base Rate (11.70%)")

    ax.set_title("Post-Hoc Observed Conversion Rate vs. RF Outreach Selection Rate by Cluster", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Rate (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(x_c)
    ax.set_xticklabels(c_labels, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    f18_path = out_path / "18_cluster_post_hoc_conversions_and_rf.png"
    plt.savefig(f18_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures["conversions_and_rf"] = f18_path

    logger.info(f"Successfully generated and saved 6 segmentation figures to {out_path}")
    return generated_figures
