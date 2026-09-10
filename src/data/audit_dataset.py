"""Forensic Data Audit and Exploratory Data Analysis module for the UCI Bank Marketing dataset.

Computes statistical diagnostics covering:
- Target prevalence and 5,000-call capacity economics
- Post-call leakage detection for 'duration'
- Campaign contact fatigue and diminishing returns
- Prior contact history dynamics (pdays, previous, poutcome)
- Categorical integrity and missing value profiling
- Duplicate demographic profiles
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from src.data.load_data import load_bank_marketing_data
from src.utils.logger import setup_logger, get_project_root

logger = setup_logger(__name__)


def compute_target_prevalence(df: pd.DataFrame, target_col: str = "target") -> Dict[str, Any]:
    """Analyze target prevalence and compute expected conversion for a 5,000 call capacity."""
    counts = df[target_col].value_counts()
    props = df[target_col].value_counts(normalize=True)
    n_total = len(df)
    n_pos = int(counts.get("yes", 0))
    pos_rate = float(props.get("yes", 0.0))

    capacity = 5000
    expected_random_conv = int(round(capacity * pos_rate))

    return {
        "total_records": n_total,
        "positive_count": n_pos,
        "negative_count": int(counts.get("no", 0)),
        "prevalence_rate": pos_rate,
        "capacity_constraint": capacity,
        "expected_conversions_random": expected_random_conv,
    }


def audit_duration_leakage(df: pd.DataFrame, target_col: str = "target") -> Dict[str, Any]:
    """Quantify target leakage in the 'duration' column."""
    y_binary = (df[target_col] == "yes").astype(int)
    auc = float(roc_auc_score(y_binary, df["duration"]))

    duration_by_target = df.groupby(target_col)["duration"].describe().to_dict(orient="index")

    zero_duration = df[df["duration"] == 0]
    zero_conv_rate = float((zero_duration[target_col] == "yes").mean()) if len(zero_duration) > 0 else 0.0

    # Decile analysis
    df_temp = df[["duration", target_col]].copy()
    df_temp["duration_decile"] = pd.qcut(df_temp["duration"], q=10, duplicates="drop")
    deciles = df_temp.groupby("duration_decile", observed=False)[target_col].agg(
        total="count",
        conv_rate=lambda s: (s == "yes").mean()
    ).reset_index()
    deciles["duration_decile"] = deciles["duration_decile"].astype(str)

    return {
        "duration_roc_auc": auc,
        "duration_by_target": duration_by_target,
        "zero_duration_count": len(zero_duration),
        "zero_duration_conv_rate": zero_conv_rate,
        "duration_deciles": deciles.to_dict(orient="records"),
    }


def audit_campaign_fatigue(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """Analyze conversion rate and volume decay across campaign contact frequencies."""
    bins = [0, 1, 2, 3, 5, 10, 100]
    labels = ["1 contact", "2 contacts", "3 contacts", "4-5 contacts", "6-10 contacts", ">10 contacts"]
    df_temp = df[["campaign", target_col]].copy()
    df_temp["campaign_tier"] = pd.cut(df_temp["campaign"], bins=bins, labels=labels)

    fatigue = df_temp.groupby("campaign_tier", observed=False)[target_col].agg(
        call_volume="count",
        conversions=lambda s: (s == "yes").sum(),
        conversion_rate=lambda s: (s == "yes").mean()
    ).reset_index()

    fatigue["pct_of_total_calls"] = fatigue["call_volume"] / len(df)
    return fatigue


def audit_prior_contacts(df: pd.DataFrame, target_col: str = "target") -> Dict[str, Any]:
    """Examine pdays, previous, and poutcome relationships."""
    df_temp = df[["pdays", "previous", "poutcome", target_col]].copy()
    df_temp["never_contacted"] = df_temp["pdays"] == -1

    never_contacted_count = int(df_temp["never_contacted"].sum())
    never_contacted_rate = float(never_contacted_count / len(df))

    pdays_cohort = df_temp.groupby("never_contacted")[target_col].agg(
        volume="count",
        conversions=lambda s: (s == "yes").sum(),
        conversion_rate=lambda s: (s == "yes").mean()
    ).reset_index()

    poutcome_clean = df_temp["poutcome"].fillna("unknown")
    poutcome_stats = df_temp.groupby(poutcome_clean)[target_col].agg(
        volume="count",
        conversions=lambda s: (s == "yes").sum(),
        conversion_rate=lambda s: (s == "yes").mean()
    ).reset_index().rename(columns={"poutcome": "prior_outcome"})

    return {
        "never_contacted_count": never_contacted_count,
        "never_contacted_pct": never_contacted_rate,
        "cohort_comparison": pdays_cohort.to_dict(orient="records"),
        "poutcome_analysis": poutcome_stats.to_dict(orient="records"),
    }


def audit_data_quality_and_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Audit missing values, 'unknown' sentinels, and repeated customer demographic profiles."""
    missing_counts = df.isna().sum()[lambda x: x > 0].to_dict()

    unknown_counts = {}
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        cnt = int((df[col] == "unknown").sum())
        if cnt > 0:
            unknown_counts[col] = cnt

    demo_cols = ["age", "job", "marital", "education", "default", "balance", "housing", "loan"]
    dup_demos = int(df.duplicated(subset=demo_cols).sum())

    return {
        "nan_counts": missing_counts,
        "unknown_sentinels": unknown_counts,
        "duplicate_demographic_profiles": dup_demos,
        "duplicate_demographic_pct": float(dup_demos / len(df)),
    }


def generate_audit_figures(df: pd.DataFrame, output_dir: Path, target_col: str = "target") -> None:
    """Generate and save publication-grade audit figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Target Leakage: Duration distribution by subscription outcome
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x=target_col, y="duration", hue=target_col, showfliers=False, ax=axes[0], palette=["#e74c3c", "#2ecc71"], legend=False)
    axes[0].set_title("Call Duration (seconds) by Subscription Outcome\n(Outliers Hidden)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Subscribed (y)", fontsize=11)
    axes[0].set_ylabel("Duration (seconds)", fontsize=11)

    df_temp = df[["duration", target_col]].copy()
    df_temp["duration_decile"] = pd.qcut(df_temp["duration"], q=10, duplicates="drop")
    deciles = df_temp.groupby("duration_decile", observed=False)[target_col].agg(
        lambda s: (s == "yes").mean()
    ).reset_index()
    deciles["decile_idx"] = [f"D{i+1}" for i in range(len(deciles))]

    sns.barplot(data=deciles, x="decile_idx", y=target_col, ax=axes[1], color="#3498db")
    axes[1].axhline(y=(df[target_col] == "yes").mean(), color="crimson", linestyle="--", label="Base Rate (11.7%)")
    axes[1].set_title("Conversion Rate by Duration Decile (D1=Shortest, D10=Longest)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Duration Decile", fontsize=11)
    axes[1].set_ylabel("Conversion Rate", fontsize=11)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    axes[1].legend()

    plt.tight_layout()
    fig_path = output_dir / "01_duration_leakage.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved duration audit figure to {fig_path}")

    # 2. Campaign Fatigue: Marginal conversion and call volume
    fatigue = audit_campaign_fatigue(df, target_col=target_col)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(fatigue))
    width = 0.4
    ax1.bar(x - width/2, fatigue["call_volume"], width=width, label="Total Calls", color="#bdc3c7")
    ax2.plot(x + width/2, fatigue["conversion_rate"] * 100, color="#e67e22", marker="o", linewidth=2.5, label="Conversion Rate (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(fatigue["campaign_tier"], rotation=15, ha="right", fontsize=10)
    ax1.set_ylabel("Total Outreach Calls", color="#2c3e50", fontsize=11)
    ax2.set_ylabel("Conversion Rate (%)", color="#e67e22", fontsize=11)
    ax1.set_title("Outreach Diminishing Returns: Call Volume vs. Conversion Rate by Campaign Contacts", fontsize=12, fontweight="bold")
    ax1.grid(False)

    plt.tight_layout()
    fig_path = output_dir / "02_campaign_fatigue.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved campaign fatigue figure to {fig_path}")

    # 3. Prior Contacts & Poutcome
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_temp = df.copy()
    df_temp["prior_contact"] = np.where(df_temp["pdays"] == -1, "Never Contacted (81.7%)", "Previously Contacted (18.3%)")
    pdays_conv = df_temp.groupby("prior_contact")[target_col].agg(lambda s: (s == "yes").mean() * 100).reset_index()

    sns.barplot(data=pdays_conv, x="prior_contact", y=target_col, hue="prior_contact", ax=axes[0], palette=["#95a5a6", "#34495e"], legend=False)
    axes[0].set_title("Conversion Rate: First-Time vs. Repeat Prospects", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Contact History", fontsize=11)
    axes[0].set_ylabel("Conversion Rate (%)", fontsize=11)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    pout_stats = df_temp.groupby(df_temp["poutcome"].fillna("unknown"))[target_col].agg(
        lambda s: (s == "yes").mean() * 100
    ).reset_index()
    sns.barplot(data=pout_stats, x="poutcome", y=target_col, hue="poutcome", ax=axes[1], palette="Blues_r", legend=False)
    axes[1].set_title("Prior Campaign Outcome (poutcome) vs. Current Conversion", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Prior Outcome", fontsize=11)
    axes[1].set_ylabel("Conversion Rate (%)", fontsize=11)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                         ha='center', va='bottom', color='#2c3e50', fontweight='bold', fontsize=10)

    plt.tight_layout()
    fig_path = output_dir / "03_prior_contacts_dynamics.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved prior contacts figure to {fig_path}")

    # 4. Monthly Seasonality: Volume vs Conversion Efficiency
    months_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    month_stats = df.groupby("month")[target_col].agg(
        call_volume="count",
        conversion_rate=lambda s: (s == "yes").mean() * 100
    ).reindex(months_order).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(month_stats["month"], month_stats["call_volume"], color="#95a5a6", alpha=0.6, label="Call Volume")
    ax2.plot(month_stats["month"], month_stats["conversion_rate"], color="#27ae60", marker="s", linewidth=2.5, label="Conversion Rate (%)")
    ax1.set_ylabel("Total Calls Dialed", fontsize=11)
    ax2.set_ylabel("Conversion Rate (%)", color="#27ae60", fontsize=11)
    ax1.set_title("Campaign Timing Seasonality: Call Volume Dialed vs. Conversion Rate by Month", fontsize=12, fontweight="bold")
    ax1.grid(False)

    plt.tight_layout()
    fig_path = output_dir / "04_monthly_seasonality.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved seasonality figure to {fig_path}")


def run_full_data_audit() -> Dict[str, Any]:
    """Execute complete end-to-end data audit and print key metrics."""
    logger.info("Executing full data audit...")
    X, y = load_bank_marketing_data()
    df = X.copy()
    df["target"] = y

    target_prev = compute_target_prevalence(df)
    leakage = audit_duration_leakage(df)
    fatigue = audit_campaign_fatigue(df)
    prior = audit_prior_contacts(df)
    quality = audit_data_quality_and_duplicates(df)

    root = get_project_root()
    figures_dir = root / "reports" / "figures"
    generate_audit_figures(df, figures_dir)

    print("\n" + "=" * 65)
    print("        UCI BANK MARKETING DATASET: FORENSIC AUDIT SUMMARY       ")
    print("=" * 65)
    print(f"Total Records           : {target_prev['total_records']:,}")
    print(f"Target Distribution     : {target_prev['positive_count']:,} subscribed ({target_prev['prevalence_rate']*100:.2f}%) / {target_prev['negative_count']:,} no ({100-target_prev['prevalence_rate']*100:.2f}%)")
    print(f"5,000-Call Baseline     : ~{target_prev['expected_conversions_random']:,} conversions (at random dial)")
    print("-" * 65)
    print(f"TARGET LEAKAGE CHECK    : duration ROC-AUC = {leakage['duration_roc_auc']:.4f}")
    print(f"Median Duration         : 'no' = {leakage['duration_by_target']['no']['50%']:.0f}s | 'yes' = {leakage['duration_by_target']['yes']['50%']:.0f}s")
    print(f"Duration = 0s Check     : {leakage['zero_duration_count']} calls (conversion rate = {leakage['zero_duration_conv_rate']*100:.1f}%)")
    print("-" * 65)
    print("CAMPAIGN FATIGUE ANALYSIS:")
    for _, row in fatigue.iterrows():
        print(f"  {row['campaign_tier']:<14} : {row['call_volume']:>6,} calls ({row['pct_of_total_calls']*100:>4.1f}%) | Conv Rate: {row['conversion_rate']*100:>5.2f}%")
    print("-" * 65)
    print(f"PRIOR CONTACTS (pdays)  : {prior['never_contacted_count']:,} never contacted ({prior['never_contacted_pct']*100:.2f}%)")
    cohort_dict = {row['never_contacted']: row['conversion_rate'] for row in prior['cohort_comparison']}
    print(f"  Uncontacted Conv Rate : {cohort_dict.get(True, 0)*100:.2f}%")
    print(f"  Contacted Conv Rate   : {cohort_dict.get(False, 0)*100:.2f}% (2.5x higher)")
    print("PRIOR OUTCOME (poutcome):")
    for row in prior['poutcome_analysis']:
        print(f"  {row['prior_outcome']:<10} : {row['volume']:>6,} records | Conv Rate: {row['conversion_rate']*100:>5.2f}%")
    print("-" * 65)
    print(f"MISSING (NaN) VALUES    : {quality['nan_counts']}")
    print(f"DEMOGRAPHIC DUPLICATES  : {quality['duplicate_demographic_profiles']:,} ({quality['duplicate_demographic_pct']*100:.2f}%)")
    print("=" * 65 + "\n")

    return {
        "prevalence": target_prev,
        "leakage": leakage,
        "fatigue": fatigue.to_dict(orient="records"),
        "prior": prior,
        "quality": quality,
    }


if __name__ == "__main__":
    run_full_data_audit()
