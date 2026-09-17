"""Supervised model interpretation, diagnostic evaluation, and error analysis.

Provides reusable analytical functions for:
1. Out-of-fold (OOF) prediction generation strictly on the 80% development partition.
2. Capacity-constrained ranking error categorization (Top-k True/False Positives, Missed Positives).
3. Score-distribution and ranked-score cutoff analysis.
4. Comprehensive subgroup diagnostic profiling across pre-call dimensions.
5. Forensic error profiling: Top-k False Positives vs Top-k True Positives, and Missed Positives vs Captured Positives.
6. Permutation feature importance across cross-validation folds.
7. Random Forest vs. Business-Rule Baseline overlap and unique yield analysis.
8. Generation of publication-quality diagnostic figures.

Methodological Guardrails:
- Strictly restricted to the frozen candidate: RandomForestClassifier(n_estimators=100, max_depth=12, class_weight=None, random_state=42).
- Zero access to or reinterpretation of the 20% holdout test partition.
- Programmatic post-call leakage prevention: 'duration' is stripped before all diagnostics.
- Proportional capacity constraint: k_oof = round(36,168 * 5,000 / 45,211) = 4,000.
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.load_data import load_bank_marketing_data
from src.features.build_features import (
    drop_duration,
    split_data,
    create_pre_call_pipeline,
)
from src.features.feature_contract import (
    select_canonical_pre_campaign_features,
    validate_pre_campaign_feature_contract,
    CANONICAL_PRE_CAMPAIGN_RAW_FEATURES,
    CANONICAL_FORBIDDEN_FEATURES,
)
from src.models.baseline import business_rule_baseline_score
from src.models.train import compute_capacity_k
from src.utils.logger import setup_logger, load_config, get_project_root

logger = setup_logger(__name__)


def generate_oof_predictions(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    pos_label: str = "yes",
) -> pd.DataFrame:
    """Generate clean out-of-fold predictions on the development partition.

    Each observation receives exactly one prediction from an estimator trained
    on the other (n_splits - 1) folds. Feature engineering and preprocessing
    are fitted strictly on fold training splits.

    Args:
        X_dev: Feature dataframe for development partition.
        y_dev: Target series for development partition.
        n_splits: Number of cross-validation folds (default: 5).
        random_state: Random state for StratifiedKFold.
        pos_label: Positive class label (default: 'yes').

    Returns:
        pd.DataFrame: OOF predictions containing original index, actuals, predicted scores, and fold IDs.
    """
    X_clean = select_canonical_pre_campaign_features(X_dev)
    validate_pre_campaign_feature_contract(X_clean.columns, raise_on_violation=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_dev = len(X_clean)
    oof_scores = np.zeros(n_dev, dtype=float)
    oof_folds = np.zeros(n_dev, dtype=int)

    rf_params = {
        "n_estimators": 100,
        "max_depth": 12,
        "class_weight": None,
        "random_state": random_state,
        "n_jobs": -1,
    }

    logger.info(
        f"Generating OOF predictions across {n_splits} folds on {n_dev:,} development records "
        "using frozen RandomForest(class_weight=None)..."
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_clean, y_dev)):
        X_fold_train = X_clean.iloc[train_idx]
        y_fold_train = y_dev.iloc[train_idx]
        X_fold_val = X_clean.iloc[val_idx]
        y_fold_val = y_dev.iloc[val_idx]

        clf = RandomForestClassifier(**rf_params)
        pipeline = create_pre_call_pipeline(classifier=clf, raw_feature_df=X_fold_train)
        pipeline.fit(X_fold_train, y_fold_train)

        classes = list(pipeline.classes_)
        if pos_label not in classes:
            raise ValueError(f"Positive label '{pos_label}' not found in fitted classes: {classes}")
        pos_col = classes.index(pos_label)

        scores = pipeline.predict_proba(X_fold_val)[:, pos_col]
        oof_scores[val_idx] = scores
        oof_folds[val_idx] = fold_idx

        logger.info(f"OOF Fold {fold_idx + 1}/{n_splits} completed ({len(val_idx):,} validation rows).")

    y_binary = (pd.Series(y_dev).reset_index(drop=True) == pos_label).astype(int)

    df_oof = pd.DataFrame({
        "orig_idx": X_dev.index,
        "actual_str": y_dev.values,
        "actual": y_binary.values,
        "score": oof_scores,
        "fold": oof_folds,
    }, index=X_dev.index)

    # Verification checks
    assert len(df_oof) == n_dev, f"OOF row count mismatch: {len(df_oof)} vs {n_dev}"
    assert not df_oof["score"].isna().any(), "Missing OOF scores detected"
    assert len(df_oof["orig_idx"].unique()) == n_dev, "Duplicate rows detected in OOF dataset"

    return df_oof


def compute_capacity_diagnostics(
    df_oof: pd.DataFrame,
    full_capacity: int = 5000,
    total_population: int = 45211,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assign ranking categories at fixed capacity constraint k_oof and compute lead-ranking diagnostics.

    Ranking Error Definitions:
    - Top-k True Positive (TP): Actual subscriber inside selected top-k.
    - Top-k False Positive (FP): Non-subscriber inside selected top-k.
    - Missed Positive (FN): Actual subscriber outside selected top-k.
    - Correctly Rejected (TN): Non-subscriber outside selected top-k.

    Args:
        df_oof: Out-of-fold predictions DataFrame.
        full_capacity: Full campaign call capacity (default: 5,000).
        total_population: Total eligible population size (default: 45,211).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Annotated df_oof, diagnostic summary metrics).
    """
    n_samples = len(df_oof)
    k_oof = compute_capacity_k(n_samples, total_population, full_capacity)

    # Deterministic descending rank sort by score, tie-breaking by original index
    df_ranked = df_oof.copy()
    df_ranked["orig_row_num"] = np.arange(len(df_ranked))
    df_ranked = df_ranked.sort_values(by=["score", "orig_row_num"], ascending=[False, True])
    df_ranked["rank"] = np.arange(len(df_ranked))
    df_ranked["in_top_k"] = df_ranked["rank"] < k_oof

    # Categorize errors
    conditions = [
        (df_ranked["in_top_k"]) & (df_ranked["actual"] == 1),
        (df_ranked["in_top_k"]) & (df_ranked["actual"] == 0),
        (~df_ranked["in_top_k"]) & (df_ranked["actual"] == 1),
        (~df_ranked["in_top_k"]) & (df_ranked["actual"] == 0),
    ]
    choices = [
        "Top-k True Positive",
        "Top-k False Positive",
        "Missed Positive",
        "Correctly Rejected",
    ]
    df_ranked["error_category"] = np.select(conditions, choices, default="Unknown")

    # Restore original index order
    df_annotated = df_ranked.sort_index()

    # Calculate metrics
    tp = int(((df_annotated["in_top_k"]) & (df_annotated["actual"] == 1)).sum())
    fp = int(((df_annotated["in_top_k"]) & (df_annotated["actual"] == 0)).sum())
    fn = int(((~df_annotated["in_top_k"]) & (df_annotated["actual"] == 1)).sum())
    tn = int(((~df_annotated["in_top_k"]) & (df_annotated["actual"] == 0)).sum())

    total_positives = int(df_annotated["actual"].sum())
    base_rate = total_positives / n_samples if n_samples > 0 else 0.0

    precision_at_k = tp / k_oof if k_oof > 0 else 0.0
    recall_at_k = tp / total_positives if total_positives > 0 else 0.0
    lift_at_k = precision_at_k / base_rate if base_rate > 0 else 0.0

    # Cutoff threshold score
    cutoff_score = float(df_ranked.iloc[k_oof - 1]["score"]) if k_oof > 0 else 0.0

    # Strict reconciliation checks
    assert tp + fp == k_oof, f"Selected count mismatch: {tp + fp} != {k_oof}"
    assert tp + fn == total_positives, f"Positive count mismatch: {tp + fn} != {total_positives}"
    assert tp + fp + fn + tn == n_samples, f"Population reconciliation mismatch: {tp + fp + fn + tn} != {n_samples}"

    pr_auc = float(average_precision_score(df_annotated["actual"], df_annotated["score"]))
    roc_auc = float(roc_auc_score(df_annotated["actual"], df_annotated["score"]))

    summary = {
        "k_evaluated": k_oof,
        "total_samples": n_samples,
        "total_positives": total_positives,
        "base_rate": base_rate,
        "conversions_at_k": tp,
        "top_k_false_positives": fp,
        "missed_positives": fn,
        "correctly_rejected": tn,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "lift_at_k": lift_at_k,
        "cutoff_score": cutoff_score,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }

    return df_annotated, summary


def build_subgroup_definitions(X: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
    """Construct boolean mask series for all pre-call candidate subgroups.

    Only legitimate pre-campaign CRM and customer dimensions are permitted.
    Current-campaign execution variables (contact, month, campaign, day, duration)
    are strictly forbidden.

    Args:
        X: Feature dataframe (must contain only canonical pre-campaign features).

    Returns:
        Dict mapping dimension name to dictionary of (subgroup_name -> boolean mask).
    """
    validate_pre_campaign_feature_contract(X.columns, raise_on_violation=True)
    subgroups: Dict[str, Dict[str, pd.Series]] = {}

    # 1. Prior Contact History (Previously contacted vs Never contacted)
    if "pdays" in X.columns:
        pdays_series = X["pdays"].fillna(-1)
        subgroups["Prior Contact History"] = {
            "Previously Contacted (pdays != -1)": pdays_series != -1,
            "Never Contacted (pdays == -1)": pdays_series == -1,
        }

    # 2. Prior Campaign Outcome
    if "poutcome" in X.columns:
        subgroups["Prior Outcome"] = {
            "Prior Success": X["poutcome"] == "success",
            "Prior Failure": X["poutcome"] == "failure",
            "Prior Other": X["poutcome"] == "other",
            "Prior Unknown / Missing": X["poutcome"].isna() | (X["poutcome"] == "unknown"),
        }

    # 3. Debt Burden
    if "housing" in X.columns and "loan" in X.columns:
        subgroups["Debt Burden"] = {
            "Debt-Free (No Housing & No Loan)": (X["housing"] == "no") & (X["loan"] == "no"),
            "Housing Loan Only": (X["housing"] == "yes") & (X["loan"] == "no"),
            "Personal Loan Only": (X["housing"] == "no") & (X["loan"] == "yes"),
            "Dual Loan Burden (Housing & Loan)": (X["housing"] == "yes") & (X["loan"] == "yes"),
        }

    # 4. Balance Sign
    if "balance" in X.columns:
        bal = X["balance"].fillna(0)
        subgroups["Balance Sign"] = {
            "Negative Balance (< €0)": bal < 0,
            "Zero Balance (€0)": bal == 0,
            "Positive Balance (> €0)": bal > 0,
        }

    # 5. Financial Balance Bands
    if "balance" in X.columns:
        bal = X["balance"].fillna(0)
        subgroups["Balance Tiers"] = {
            "< €0": bal < 0,
            "€0 - €499": (bal >= 0) & (bal < 500),
            "€500 - €1,999": (bal >= 500) & (bal < 2000),
            "€2,000 - €4,999": (bal >= 2000) & (bal < 5000),
            "€5,000+": bal >= 5000,
        }

    # 6. Age Bands
    if "age" in X.columns:
        age = X["age"]
        subgroups["Age Tiers"] = {
            "< 30 years": age < 30,
            "30 - 39 years": (age >= 30) & (age < 40),
            "40 - 49 years": (age >= 40) & (age < 50),
            "50 - 59 years": (age >= 50) & (age < 60),
            "60+ years": age >= 60,
        }

    # 7. Job Category (legitimate pre-campaign CRM profile)
    if "job" in X.columns:
        jobs = [
            "management", "blue-collar", "technician", "admin.",
            "services", "retired", "self-employed", "entrepreneur",
            "unemployed", "housemaid", "student"
        ]
        job_dict = {j.capitalize(): X["job"] == j for j in jobs if (X["job"] == j).any()}
        if (X["job"] == "unknown").any():
            job_dict["Unknown Job"] = X["job"] == "unknown"
        subgroups["Job Category"] = job_dict

    # 8. Education Tier (legitimate pre-campaign CRM profile)
    if "education" in X.columns:
        subgroups["Education Tier"] = {
            "Tertiary Education": X["education"] == "tertiary",
            "Secondary Education": X["education"] == "secondary",
            "Primary Education": X["education"] == "primary",
            "Unknown Education": X["education"].isna() | (X["education"] == "unknown"),
        }

    # 9. Marital Status (legitimate pre-campaign CRM profile)
    if "marital" in X.columns:
        subgroups["Marital Status"] = {
            "Married": X["marital"] == "married",
            "Single": X["marital"] == "single",
            "Divorced": X["marital"] == "divorced",
        }

    # Engineering safeguard: Verify that no forbidden feature appears in any subgroup dimension
    forbidden_dim_names = {
        "contact",
        "communication channel",
        "contact communication channel",
        "month",
        "outreach month",
        "campaign",
        "campaign outreach intensity",
        "day",
        "day_of_week",
        "contact_day_of_month",
        "duration",
        "y",
    }
    for dim_name in subgroups.keys():
        assert dim_name.lower() not in forbidden_dim_names, f"Forbidden dimension '{dim_name}' in subgroups"

    return subgroups


def analyze_subgroups(
    X_dev: pd.DataFrame,
    df_annotated: pd.DataFrame,
    min_count: int = 30,
) -> pd.DataFrame:
    """Evaluate performance across candidate pre-call subgroups.

    Args:
        X_dev: Feature dataframe for development partition.
        df_annotated: Annotated OOF predictions DataFrame containing 'in_top_k', 'actual', and 'score'.
        min_count: Minimum subgroup size to calculate percentage metrics reliably.

    Returns:
        pd.DataFrame: Formatted subgroup metrics table.
    """
    X_clean = select_canonical_pre_campaign_features(X_dev)
    subgroup_definitions = build_subgroup_definitions(X_clean)
    rows: List[Dict[str, Any]] = []

    for dim_name, group_dict in subgroup_definitions.items():
        for group_name, mask in group_dict.items():
            # Align mask with annotated df index
            aligned_mask = mask.loc[df_annotated.index]
            sub_df = df_annotated[aligned_mask]

            pop_count = len(sub_df)
            if pop_count == 0:
                continue

            total_positives = int(sub_df["actual"].sum())
            prevalence = total_positives / pop_count if pop_count > 0 else 0.0

            selected_count = int(sub_df["in_top_k"].sum())
            selection_rate = selected_count / pop_count if pop_count > 0 else 0.0

            tp = int(((sub_df["in_top_k"]) & (sub_df["actual"] == 1)).sum())
            fp = int(((sub_df["in_top_k"]) & (sub_df["actual"] == 0)).sum())
            fn = int(((~sub_df["in_top_k"]) & (sub_df["actual"] == 1)).sum())

            precision = tp / selected_count if selected_count > 0 else 0.0
            recall = tp / total_positives if total_positives > 0 else 0.0
            lift = precision / prevalence if prevalence > 0 else 0.0
            mean_score = float(sub_df["score"].mean())

            rows.append({
                "dimension": dim_name,
                "subgroup": group_name,
                "population_count": pop_count,
                "prevalence": prevalence,
                "selected_count": selected_count,
                "selection_rate": selection_rate,
                "conversions_captured": tp,
                "false_positives": fp,
                "missed_positives": fn,
                "precision_at_k": precision if selected_count >= min_count else np.nan,
                "recall_at_k": recall if total_positives >= min_count else np.nan,
                "lift_at_k": lift if (selected_count >= min_count and prevalence > 0) else np.nan,
                "mean_score": mean_score,
                "small_sample_flag": bool(selected_count < min_count),
            })

    return pd.DataFrame(rows)


def analyze_false_positives_and_missed_positives(
    X_dev: pd.DataFrame,
    df_annotated: pd.DataFrame,
    boundary_window: int = 50,
) -> Dict[str, Any]:
    """Perform descriptive forensic comparison of Top-k False Positives vs True Positives,
    and Missed Positives vs Captured Positives.

    Args:
        X_dev: Raw feature dataframe.
        df_annotated: Annotated OOF predictions DataFrame.
        boundary_window: Number of prospects to inspect immediately above and below the cutoff.

    Returns:
        Dict[str, Any]: Comparative demographic and financial profiles for error cohorts.
    """
    merged = X_dev.copy()
    merged["actual"] = df_annotated["actual"]
    merged["score"] = df_annotated["score"]
    merged["rank"] = df_annotated["rank"]
    merged["in_top_k"] = df_annotated["in_top_k"]
    merged["error_category"] = df_annotated["error_category"]

    tp_df = merged[merged["error_category"] == "Top-k True Positive"]
    fp_df = merged[merged["error_category"] == "Top-k False Positive"]
    fn_df = merged[merged["error_category"] == "Missed Positive"]
    tn_df = merged[merged["error_category"] == "Correctly Rejected"]

    def profile_cohort(df: pd.DataFrame) -> Dict[str, Any]:
        bal = df["balance"].fillna(0) if "balance" in df.columns else pd.Series(0)
        age = df["age"] if "age" in df.columns else pd.Series(0)
        pdays = df["pdays"].fillna(-1) if "pdays" in df.columns else pd.Series(-1)

        poutcome_dist = df["poutcome"].value_counts(normalize=True).to_dict() if "poutcome" in df.columns else {}
        job_top3 = df["job"].value_counts(normalize=True).head(3).to_dict() if "job" in df.columns else {}
        education_top3 = df["education"].value_counts(normalize=True).head(3).to_dict() if "education" in df.columns else {}
        marital_dist = df["marital"].value_counts(normalize=True).to_dict() if "marital" in df.columns else {}

        return {
            "count": len(df),
            "mean_score": float(df["score"].mean()) if len(df) > 0 else 0.0,
            "median_score": float(df["score"].median()) if len(df) > 0 else 0.0,
            "mean_age": float(age.mean()) if len(df) > 0 else 0.0,
            "median_age": float(age.median()) if len(df) > 0 else 0.0,
            "mean_balance": float(bal.mean()) if len(df) > 0 else 0.0,
            "median_balance": float(bal.median()) if len(df) > 0 else 0.0,
            "pct_previously_contacted": float((pdays != -1).mean() * 100) if len(df) > 0 else 0.0,
            "pct_never_contacted": float((pdays == -1).mean() * 100) if len(df) > 0 else 0.0,
            "pct_prior_success": float((df["poutcome"] == "success").mean() * 100) if ("poutcome" in df.columns and len(df) > 0) else 0.0,
            "pct_debt_free": float(((df["housing"] == "no") & (df["loan"] == "no")).mean() * 100) if ("housing" in df.columns and "loan" in df.columns and len(df) > 0) else 0.0,
            "pct_housing_loan": float((df["housing"] == "yes").mean() * 100) if ("housing" in df.columns and len(df) > 0) else 0.0,
            "pct_personal_loan": float((df["loan"] == "yes").mean() * 100) if ("loan" in df.columns and len(df) > 0) else 0.0,
            "pct_negative_balance": float((bal < 0).mean() * 100) if len(df) > 0 else 0.0,
            "poutcome_dist": poutcome_dist,
            "job_top3": job_top3,
            "education_top3": education_top3,
            "marital_dist": marital_dist,
        }

    profile_tp = profile_cohort(tp_df)
    profile_fp = profile_cohort(fp_df)
    profile_fn = profile_cohort(fn_df)
    profile_tn = profile_cohort(tn_df)

    # Boundary analysis: Positives right at the decision boundary
    # 50 highest-scoring missed positives (just below capacity threshold)
    highest_fn = fn_df.sort_values(by="score", ascending=False).head(boundary_window)
    # 50 lowest-scoring captured positives (just above capacity threshold)
    lowest_tp = tp_df.sort_values(by="score", ascending=True).head(boundary_window)

    profile_highest_fn = profile_cohort(highest_fn)
    profile_lowest_tp = profile_cohort(lowest_tp)

    fn_vs_tp_comparison = {
        "fn_pct_never_contacted": profile_fn["pct_never_contacted"],
        "tp_pct_never_contacted": profile_tp["pct_never_contacted"],
        "fn_pct_prior_success": profile_fn["pct_prior_success"],
        "tp_pct_prior_success": profile_tp["pct_prior_success"],
        "fn_median_balance": profile_fn["median_balance"],
        "tp_median_balance": profile_tp["median_balance"],
        "fn_mean_balance": profile_fn["mean_balance"],
        "tp_mean_balance": profile_tp["mean_balance"],
        "fn_pct_debt_free": profile_fn["pct_debt_free"],
        "tp_pct_debt_free": profile_tp["pct_debt_free"],
        "fn_pct_housing_loan": profile_fn["pct_housing_loan"],
        "tp_pct_housing_loan": profile_tp["pct_housing_loan"],
        "fn_pct_personal_loan": profile_fn["pct_personal_loan"],
        "tp_pct_personal_loan": profile_tp["pct_personal_loan"],
        "fn_mean_score": profile_fn["mean_score"],
        "tp_mean_score": profile_tp["mean_score"],
        "fn_median_score": profile_fn["median_score"],
        "tp_median_score": profile_tp["median_score"],
    }

    return {
        "profile_tp": profile_tp,
        "profile_fp": profile_fp,
        "profile_fn": profile_fn,
        "profile_tn": profile_tn,
        "profile_highest_fn_boundary": profile_highest_fn,
        "profile_lowest_tp_boundary": profile_lowest_tp,
        "fn_vs_tp_comparison": fn_vs_tp_comparison,
    }


def compute_permutation_feature_importance(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute permutation feature importance across cross-validation folds using PR-AUC (Average Precision).

    Evaluated strictly on held-out validation folds to measure genuine generalization loss
    when raw pre-call input features are shuffled.

    Args:
        X_dev: Feature dataframe for development partition.
        y_dev: Target series for development partition.
        n_splits: Number of cross-validation folds.
        n_repeats: Number of shuffle repetitions per feature.
        random_state: Random state for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Permutation importance summary table, Impurity importance summary table).
    """
    X_clean = select_canonical_pre_campaign_features(X_dev)
    validate_pre_campaign_feature_contract(X_clean.columns, raise_on_violation=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rf_params = {
        "n_estimators": 100,
        "max_depth": 12,
        "class_weight": None,
        "random_state": random_state,
        "n_jobs": -1,
    }

    raw_features = list(X_clean.columns)
    fold_importances: List[np.ndarray] = []
    impurity_importances_list: List[np.ndarray] = []
    feature_names_out: Optional[List[str]] = None

    def pr_auc_scorer(estimator, X_val, y_true):
        pos_idx = list(estimator.classes_).index("yes")
        y_proba = estimator.predict_proba(X_val)[:, pos_col]
        y_bin = (pd.Series(y_true).reset_index(drop=True) == "yes").astype(int)
        return average_precision_score(y_bin, y_proba)

    logger.info(f"Computing permutation feature importance on held-out validation folds ({n_splits} folds)...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_clean, y_dev)):
        X_tr = X_clean.iloc[train_idx]
        y_tr = y_dev.iloc[train_idx]
        X_val = X_clean.iloc[val_idx]
        y_val = y_dev.iloc[val_idx]

        clf = RandomForestClassifier(**rf_params)
        pipe = create_pre_call_pipeline(clf, raw_feature_df=X_tr)
        pipe.fit(X_tr, y_tr)

        pos_col = list(pipe.classes_).index("yes")

        perm_res = permutation_importance(
            pipe,
            X_val,
            y_val,
            scoring=pr_auc_scorer,
            n_repeats=n_repeats,
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )
        fold_importances.append(perm_res.importances_mean)

        # Built-in impurity importance of fitted RF
        fitted_rf = pipe.named_steps["classifier"]
        impurity_importances_list.append(fitted_rf.feature_importances_)

        if feature_names_out is None:
            try:
                preproc = pipe.named_steps["preprocessor"]
                feature_names_out = list(preproc.get_feature_names_out())
            except Exception:
                feature_names_out = [f"feat_{i}" for i in range(len(fitted_rf.feature_importances_))]

    # Aggregate permutation importance (raw features)
    perm_matrix = np.array(fold_importances)  # shape: (n_splits, n_raw_features)
    df_perm = pd.DataFrame({
        "feature": raw_features,
        "mean_importance": perm_matrix.mean(axis=0),
        "std_importance": perm_matrix.std(axis=0, ddof=1),
        "min_importance": perm_matrix.min(axis=0),
        "max_importance": perm_matrix.max(axis=0),
    }).sort_values(by="mean_importance", ascending=False).reset_index(drop=True)

    # Aggregate impurity importance (engineered preprocessed features)
    imp_matrix = np.array(impurity_importances_list)
    df_impurity = pd.DataFrame({
        "engineered_feature": feature_names_out,
        "mean_impurity_importance": imp_matrix.mean(axis=0),
        "std_impurity_importance": imp_matrix.std(axis=0, ddof=1),
    }).sort_values(by="mean_impurity_importance", ascending=False).reset_index(drop=True)

    return df_perm, df_impurity


def compare_rf_vs_business_rule(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    df_annotated: pd.DataFrame,
    full_capacity: int = 5000,
    total_population: int = 45211,
    pos_label: str = "yes",
) -> Dict[str, Any]:
    """Compare prospects prioritized by Random Forest vs. the frozen Business-Rule heuristic
    under the identical development capacity constraint k_oof.

    Args:
        X_dev: Feature dataframe for development partition.
        y_dev: Target series for development partition.
        df_annotated: Annotated OOF predictions DataFrame containing 'rank' and 'actual'.
        full_capacity: Full campaign call capacity (default: 5,000).
        total_population: Total eligible population size (default: 45,211).
        pos_label: Positive class label.

    Returns:
        Dict[str, Any]: Overlap metrics, conversion yields, and Jaccard similarity.
    """
    n_samples = len(X_dev)
    k_oof = compute_capacity_k(n_samples, total_population, full_capacity)

    # 1. Random Forest Top-k Indices
    rf_top_k_indices = set(df_annotated[df_annotated["rank"] < k_oof].index)

    # 2. Business-Rule Top-k Indices
    br_scores = business_rule_baseline_score(X_dev)
    bal_series = X_dev["balance"].fillna(0) if "balance" in X_dev.columns else pd.Series(0, index=X_dev.index)
    sort_df = pd.DataFrame({
        "score": br_scores,
        "balance": bal_series,
        "row_idx": np.arange(n_samples),
    }, index=X_dev.index)
    br_ranked_idx = sort_df.sort_values(
        by=["score", "balance", "row_idx"],
        ascending=[False, False, True]
    ).index[:k_oof]
    br_top_k_indices = set(br_ranked_idx)

    # 3. Set Overlaps
    shared_indices = rf_top_k_indices.intersection(br_top_k_indices)
    rf_only_indices = rf_top_k_indices - br_top_k_indices
    br_only_indices = br_top_k_indices - rf_top_k_indices
    union_indices = rf_top_k_indices.union(br_top_k_indices)

    jaccard_similarity = len(shared_indices) / len(union_indices) if len(union_indices) > 0 else 0.0

    # 4. Conversion counts
    y_dev_series = pd.Series(y_dev).copy()
    y_dev_binary = (y_dev_series == pos_label).astype(int)

    shared_conv = int(y_dev_binary.loc[list(shared_indices)].sum()) if shared_indices else 0
    rf_only_conv = int(y_dev_binary.loc[list(rf_only_indices)].sum()) if rf_only_indices else 0
    br_only_conv = int(y_dev_binary.loc[list(br_only_indices)].sum()) if br_only_indices else 0

    rf_total_conv = shared_conv + rf_only_conv
    br_total_conv = shared_conv + br_only_conv

    # 5. Profile RF-only vs BR-only prospects to understand qualitative differences
    rf_only_df = X_dev.loc[list(rf_only_indices)]
    br_only_df = X_dev.loc[list(br_only_indices)]

    def quick_profile(df: pd.DataFrame) -> Dict[str, Any]:
        pdays = df["pdays"].fillna(-1) if "pdays" in df.columns else pd.Series(-1)
        bal = df["balance"].fillna(0) if "balance" in df.columns else pd.Series(0)
        age = df["age"] if "age" in df.columns else pd.Series(0)
        return {
            "mean_age": float(age.mean()),
            "median_balance": float(bal.median()),
            "pct_previously_contacted": float((pdays != -1).mean() * 100),
            "pct_debt_free": float(((df["housing"] == "no") & (df["loan"] == "no")).mean() * 100) if ("housing" in df.columns and "loan" in df.columns) else 0.0,
            "top_jobs": df["job"].value_counts(normalize=True).head(3).to_dict() if "job" in df.columns else {},
            "top_education": df["education"].value_counts(normalize=True).head(3).to_dict() if "education" in df.columns else {},
        }

    return {
        "k_evaluated": k_oof,
        "shared_count": len(shared_indices),
        "rf_only_count": len(rf_only_indices),
        "br_only_count": len(br_only_indices),
        "jaccard_similarity": jaccard_similarity,
        "shared_conversions": shared_conv,
        "shared_precision": shared_conv / len(shared_indices) if shared_indices else 0.0,
        "rf_only_conversions": rf_only_conv,
        "rf_only_precision": rf_only_conv / len(rf_only_indices) if rf_only_indices else 0.0,
        "br_only_conversions": br_only_conv,
        "br_only_precision": br_only_conv / len(br_only_indices) if br_only_indices else 0.0,
        "rf_total_conversions": rf_total_conv,
        "br_total_conversions": br_total_conv,
        "rf_net_conversion_advantage": rf_total_conv - br_total_conv,
        "rf_only_profile": quick_profile(rf_only_df),
        "br_only_profile": quick_profile(br_only_df),
    }


def generate_diagnostic_figures(
    df_annotated: pd.DataFrame,
    summary_metrics: Dict[str, Any],
    perm_importance_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    rf_br_comparison: Dict[str, Any],
    output_dir: Path,
) -> List[Path]:
    """Generate portfolio-quality analytical figures and save to output directory.

    Figures:
    1. 05_oof_score_distribution.png: Score distribution by actual target with cutoff line.
    2. 06_ranked_score_curve.png: Ranked score curve with capacity cutoff.
    3. 07_permutation_importance.png: Top raw feature permutation importances with error bars.
    4. 08_subgroup_precision_and_selection.png: Subgroup precision@capacity across key dimensions.
    5. 09_rf_vs_business_rule_overlap.png: Overlap & yield comparison (RF vs Business Rule).

    Args:
        df_annotated: Annotated OOF predictions DataFrame.
        summary_metrics: Dict with diagnostic summary metrics.
        perm_importance_df: Permutation importance DataFrame.
        subgroup_df: Subgroup metrics DataFrame.
        rf_br_comparison: Overlap comparison Dict.
        output_dir: Output directory path.

    Returns:
        List[Path]: Paths to generated image files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: List[Path] = []
    cutoff_score = summary_metrics["cutoff_score"]
    k_oof = summary_metrics["k_evaluated"]

    # Style settings
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })

    # -------------------------------------------------------------
    # Figure 5: OOF Score Distribution by Actual Target
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    scores_neg = df_annotated[df_annotated["actual"] == 0]["score"]
    scores_pos = df_annotated[df_annotated["actual"] == 1]["score"]

    bins = np.linspace(0, 0.9, 45)
    ax.hist(scores_neg, bins=bins, density=True, alpha=0.55, color="#1f77b4", label=f"Non-Subscribers (N={len(scores_neg):,})", edgecolor="white")
    ax.hist(scores_pos, bins=bins, density=True, alpha=0.65, color="#ff7f0e", label=f"Subscribers (N={len(scores_pos):,})", edgecolor="white")

    ax.axvline(cutoff_score, color="#d62728", linestyle="--", linewidth=2.0, label=f"Pooled OOF Diagnostic Cutoff (Rank 4,000, Score={cutoff_score:.4f})")

    ax.set_title("OOF Ranking Score Distribution by Actual Target Outcome", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Conversion Score (Uncalibrated Probability)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True)
    plt.tight_layout()

    fig5_path = output_dir / "05_oof_score_distribution.png"
    fig.savefig(fig5_path)
    plt.close(fig)
    generated_paths.append(fig5_path)
    logger.info(f"Saved Figure 5: {fig5_path}")

    # -------------------------------------------------------------
    # Figure 6: Ranked Score Curve with Capacity Cutoff
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    df_sorted = df_annotated.sort_values(by="rank")
    ranks = df_sorted["rank"].values
    scores = df_sorted["score"].values

    ax.plot(ranks, scores, color="#2ca02c", linewidth=2.2, label="Prospect Ranking Curve")
    ax.axvline(k_oof, color="#d62728", linestyle="--", linewidth=1.8, label=f"Call Quota k={k_oof:,}")
    ax.axhline(cutoff_score, color="#7f7f7f", linestyle=":", linewidth=1.5, label=f"Pooled OOF Diagnostic Cutoff = {cutoff_score:.4f}")

    # Shading
    ax.fill_between(ranks[:k_oof], 0, scores[:k_oof], color="#2ca02c", alpha=0.18, label="Selected Top-Capacity Cohort (Top 4,000)")
    ax.fill_between(ranks[k_oof:], 0, scores[k_oof:], color="#7f7f7f", alpha=0.08, label="Rejected Prospects (Ranks 4,001 - 36,168)")

    ax.set_title("Cumulative Ranked Score Curve with Fixed Capacity Cutoff", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Prospect Rank (Descending Predicted Score)", fontsize=11)
    ax.set_ylabel("Predicted Conversion Score", fontsize=11)
    ax.set_xlim(0, len(df_annotated))
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True)
    plt.tight_layout()

    fig6_path = output_dir / "06_ranked_score_curve.png"
    fig.savefig(fig6_path)
    plt.close(fig)
    generated_paths.append(fig6_path)
    logger.info(f"Saved Figure 6: {fig6_path}")

    # -------------------------------------------------------------
    # Figure 7: Top Permutation Importances with Uncertainty
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    df_top_perm = perm_importance_df.sort_values(by="mean_importance", ascending=True)

    y_pos = np.arange(len(df_top_perm))
    ax.barh(y_pos, df_top_perm["mean_importance"], xerr=df_top_perm["std_importance"], color="#3b528b", alpha=0.85, edgecolor="#1f2d4d", capsize=4, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top_perm["feature"], fontsize=11)
    ax.set_title("Pre-Call Permutation Feature Importances (PR-AUC Drop on Held-Out Folds)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in Validation PR-AUC (Error Bars = ±1 Fold Std Dev)", fontsize=11)
    ax.grid(axis="x")
    plt.tight_layout()

    fig7_path = output_dir / "07_permutation_importance.png"
    fig.savefig(fig7_path)
    plt.close(fig)
    generated_paths.append(fig7_path)
    logger.info(f"Saved Figure 7: {fig7_path}")

    # -------------------------------------------------------------
    # Figure 8: Subgroup Precision@capacity vs. Population Baseline
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    # Select key representative pre-campaign subgroups (only those meeting N >= 30 safeguard)
    key_subgroups = [
        "Prior Success",
        "Prior Failure",
        "Previously Contacted (pdays != -1)",
        "Never Contacted (pdays == -1)",
        "Debt-Free (No Housing & No Loan)",
        "Housing Loan Only",
        "Dual Loan Burden (Housing & Loan)",
        "€5,000+",
        "€2,000 - €4,999",
        "€500 - €1,999",
        "€0 - €499",
        "60+ years",
        "< 30 years",
        "30 - 39 years",
        "Tertiary Education",
        "Secondary Education",
        "Retired",
        "Management",
        "Married",
        "Single",
    ]
    df_plot_sub = subgroup_df[subgroup_df["subgroup"].isin(key_subgroups)].dropna(subset=["precision_at_k"]).copy()
    df_plot_sub = df_plot_sub.sort_values(by="precision_at_k", ascending=True)

    y_pos = np.arange(len(df_plot_sub))
    bars = ax.barh(y_pos, df_plot_sub["precision_at_k"] * 100, color="#21918c", alpha=0.85, edgecolor="#0f4d4a", height=0.6)

    # Base rate reference
    ax.axvline(summary_metrics["base_rate"] * 100, color="#d62728", linestyle="--", linewidth=1.5, label=f"Development Base Rate ({summary_metrics['base_rate']*100:.2f}%)")
    ax.axvline(summary_metrics["precision_at_k"] * 100, color="#1f77b4", linestyle=":", linewidth=1.8, label=f"Overall Top-k Precision ({summary_metrics['precision_at_k']*100:.2f}%)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot_sub["subgroup"], fontsize=10.5)
    ax.set_title("Selected Subgroup Precision@capacity (% Converting in Top 4,000)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Precision within Subgroup When Selected into Top-k (%)", fontsize=11)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(axis="x")
    plt.tight_layout()

    fig8_path = output_dir / "08_subgroup_precision_and_selection.png"
    fig.savefig(fig8_path)
    plt.close(fig)
    generated_paths.append(fig8_path)
    logger.info(f"Saved Figure 8: {fig8_path}")

    # -------------------------------------------------------------
    # Figure 9: RF vs. Business Rule Overlap & Yield Breakdown
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300)

    # Left: Lead Selection Composition
    categories = ["Shared Selections\n(Both RF & Rule)", "Unique to RF\n(Model Discoveries)", "Unique to Rule\n(Heuristic Only)"]
    counts = [rf_br_comparison["shared_count"], rf_br_comparison["rf_only_count"], rf_br_comparison["br_only_count"]]
    colors = ["#440154", "#21918c", "#fde725"]

    bars1 = ax1.bar(categories, counts, color=colors, alpha=0.85, edgecolor="#333333", width=0.55)
    ax1.set_title(f"Composition of Top {k_oof:,} Leads (Jaccard = {rf_br_comparison['jaccard_similarity']:.3f})", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_ylabel("Number of Leads", fontsize=11)
    ax1.set_ylim(0, max(counts) * 1.15)
    ax1.grid(axis="y")
    for b, count in zip(bars1, counts):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 40, f"{count:,}", ha="center", va="bottom", fontweight="bold")

    # Right: Conversion Yield & Precision
    conversions = [rf_br_comparison["shared_conversions"], rf_br_comparison["rf_only_conversions"], rf_br_comparison["br_only_conversions"]]
    precisions = [rf_br_comparison["shared_precision"] * 100, rf_br_comparison["rf_only_precision"] * 100, rf_br_comparison["br_only_precision"] * 100]

    bars2 = ax2.bar(categories, conversions, color=colors, alpha=0.85, edgecolor="#333333", width=0.55)
    ax2.set_title("Conversions Captured by Lead Cohort", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_ylabel("Actual Subscriptions Captured", fontsize=11)
    ax2.set_ylim(0, max(conversions) * 1.18)
    ax2.grid(axis="y")
    for b, conv, prec in zip(bars2, conversions, precisions):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, f"{conv:,}\n({prec:.1f}% prec)", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()

    fig9_path = output_dir / "09_rf_vs_business_rule_overlap.png"
    fig.savefig(fig9_path)
    plt.close(fig)
    generated_paths.append(fig9_path)
    logger.info(f"Saved Figure 9: {fig9_path}")

    return generated_paths


def run_full_diagnostics(
    output_dir: Optional[Path] = None,
    random_state: int = 42,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Execute end-to-end diagnostics on the 80% development partition.

    Never accesses or revisits the 20% holdout test partition.

    Args:
        output_dir: Path to directory for saving diagnostic figures.
        random_state: Random state for reproducibility.
        n_splits: Number of CV folds for OOF predictions.

    Returns:
        Dict[str, Any]: Comprehensive diagnostic results.
    """
    if output_dir is None:
        root = get_project_root()
        output_dir = root / "reports" / "figures"

    logger.info("Loading Bank Marketing dataset...")
    X, y = load_bank_marketing_data()
    n_total = len(X)

    # Enforce canonical pre-campaign feature contract
    X_clean = select_canonical_pre_campaign_features(X)
    validate_pre_campaign_feature_contract(X_clean.columns, raise_on_violation=True)

    # 80/20 train/test split - test set is strictly quarantined and not accessed
    X_dev, X_test_frozen, y_dev, y_test_frozen = split_data(
        X_clean,
        y,
        test_size=0.2,
        random_state=random_state,
    )
    del X_test_frozen, y_test_frozen

    logger.info(f"Development partition isolated: {len(X_dev):,} records (80.0% of total population).")

    # 1. Generate OOF predictions
    df_oof = generate_oof_predictions(
        X_dev=X_dev,
        y_dev=y_dev,
        n_splits=n_splits,
        random_state=random_state,
    )

    # 2. Assign capacity categories and calculate ranking errors
    df_annotated, summary_metrics = compute_capacity_diagnostics(
        df_oof=df_oof,
        full_capacity=5000,
        total_population=n_total,
    )

    # 3. Subgroup analysis
    subgroup_df = analyze_subgroups(X_dev, df_annotated)

    # 4. Error cohort profiling (False Positives vs True Positives, Missed vs Captured)
    error_profiles = analyze_false_positives_and_missed_positives(X_dev, df_annotated)

    # 5. Permutation feature importance
    perm_importance_df, impurity_importance_df = compute_permutation_feature_importance(
        X_dev=X_dev,
        y_dev=y_dev,
        n_splits=n_splits,
        random_state=random_state,
    )

    # 6. RF vs Business-Rule overlap
    rf_br_comparison = compare_rf_vs_business_rule(
        X_dev=X_dev,
        y_dev=y_dev,
        df_annotated=df_annotated,
        full_capacity=5000,
        total_population=n_total,
    )

    # 7. Generate portfolio figures
    fig_paths = generate_diagnostic_figures(
        df_annotated=df_annotated,
        summary_metrics=summary_metrics,
        perm_importance_df=perm_importance_df,
        subgroup_df=subgroup_df,
        rf_br_comparison=rf_br_comparison,
        output_dir=output_dir,
    )

    return {
        "df_annotated": df_annotated,
        "summary_metrics": summary_metrics,
        "subgroup_df": subgroup_df,
        "error_profiles": error_profiles,
        "perm_importance_df": perm_importance_df,
        "impurity_importance_df": impurity_importance_df,
        "rf_br_comparison": rf_br_comparison,
        "fig_paths": fig_paths,
    }


if __name__ == "__main__":
    logger.info("Executing supervised model interpretation and diagnostics...")
    res = run_full_diagnostics()
    print("\n--- Capacity Diagnostics Summary ---")
    for k, v in res["summary_metrics"].items():
        print(f"{k}: {v}")
    print("\n--- RF vs Business Rule ---")
    for k, v in res["rf_br_comparison"].items():
        if not k.endswith("_profile"):
            print(f"{k}: {v}")
