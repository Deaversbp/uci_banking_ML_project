"""Authoritative generator for notebooks/01_data_exploration.ipynb.

This script is the single authoritative source of truth for generating the exploratory
data analysis notebook. Reconciled with Phase 1 forensic audit findings:
- Framing of campaign tiers and monthly conversion as observational associations
- Investigation of the 5 pdays != -1 and poutcome missingness exceptions
- Repeated demographic/financial profile analysis (acknowledging absence of unique customer IDs)
- Two-tier baseline comparison: Random Selection (585 conv) vs Business-Rule Heuristic (~1,664 conv)
"""
import json
from pathlib import Path

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text: str):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code: str):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# Title & Business Context
add_md("""# UCI Bank Marketing - Phase 1: Forensic Data Audit & EDA

### Business Context:
A retail bank conducts outbound telemarketing campaigns to sell term deposits. Due to operational capacity constraints, the sales team can call only **5,000 customers** from the eligible prospect pool.

### Core Strategic Objectives:
1. **Supervised Optimization**: Identify and rank-order leads by conversion probability to maximize subscriptions within the 5,000-call quota.
2. **Unsupervised Discovery**: Uncover actionable customer segments to understand who the campaign reaches and tailor messaging.

### Phase 1 Audit Mandate:
Perform a comprehensive data audit and exploratory analysis **before building any models**. Specifically investigate:
- **Baseline economics & two-tier benchmarks**: Random selection (~585 conversions) and Business-Rule Heuristic (~1,664 conversions).
- **Target leakage** in `duration`: Proof of post-call leakage and justification for strict exclusion.
- **Campaign outreach dynamics**: Observational association across contact frequency tiers without causal leaps.
- **Prior contact history**: Dynamics of `pdays`, `previous`, and `poutcome`, including documentation of alignment exceptions.
- **Data quality & missingness**: Preserving informative categorical missing states (`unknown`).
- **Repeated demographic/financial profiles**: Assessing profile overlap without assuming proven customer identities.""")

# Setup & Imports
add_code("""import sys
from pathlib import Path

# Ensure project root is in path
cwd = Path.cwd()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# Styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

from src.data.load_data import load_bank_marketing_data
from src.models.baseline import evaluate_random_baseline, evaluate_business_rule_baseline
print("Libraries loaded successfully!")""")

# Ingestion
add_md("""## 1. Data Ingestion & Schema Verification""")
add_code("""# Load cached dataset
X, y = load_bank_marketing_data()
df = X.copy()
df['target'] = y

print(f"Total Observations (Records): {len(df):,}")
print(f"Total Raw Features: {X.shape[1]}")
print("\\nFeatures Data Types:")
print(df.dtypes)
df.head(5)""")

# Target Distribution & Two-Tier Benchmarks
add_md("""## 2. Target Prevalence & The Two-Tier Baseline (5,000-Call Quota)

### Business Benchmarks:
With an eligible population of 45,211 contact records and a positive prevalence of **11.70%**, our baselines under a **5,000 call capacity constraint** are:
1. **Random Outreach Baseline**: Uniform random selection yields expected conversions:
   $$\\mathbb{E}[\\text{Conversions}] = 5,000 \\times 0.1170 \\approx 585 \\text{ subscriptions (11.70% precision, 1.00x lift)}$$
2. **Business-Rule Heuristic Baseline**: A sensible pre-call heuristic prioritizing prior campaign success (`poutcome == 'success'`), prior contact (`pdays != -1`) without loan burden, and liquid balance tie-breaking achieves **~1,664 conversions (33.28% precision, 2.84x lift)**.

> **Key Rule**: Any machine learning model in Phase 3 must beat **both** benchmarks to provide genuine business value.""")

add_code("""# Target prevalence
target_counts = df['target'].value_counts()
target_props = df['target'].value_counts(normalize=True)

print("Target Distribution:")
for k in target_counts.index:
    print(f"  '{k}': {target_counts[k]:,} ({target_props[k]*100:.2f}%)")

# Compute benchmarks at k=5,000
rand_bench = evaluate_random_baseline(df['target'], k=5000)
biz_bench = evaluate_business_rule_baseline(df, df['target'], k=5000)

print(f"\\n1. Random Selection Benchmark @ 5,000: {rand_bench['conversions_at_k']:.0f} conversions (11.70% precision, 1.00x lift)")
print(f"2. Business-Rule Benchmark @ 5,000   : {biz_bench['conversions_at_k']:,} conversions (33.28% precision, 2.84x lift)")

# Plot target distribution
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=target_props.index, y=target_props.values * 100, hue=target_props.index, palette=["#e74c3c", "#2ecc71"], legend=False, ax=ax)
ax.set_title("Target Distribution (Prevalence: 11.70% Subscribed)", fontsize=12, fontweight="bold")
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Subscribed to Term Deposit (target)")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.show()""")

# Duration Leakage
add_md("""## 3. Forensic Leakage Investigation: `duration`

> [!WARNING]
> **UCI Documentation Warning**: Call duration is strictly known **only after** the phone call has been conducted. Including `duration` introduces strong post-call target leakage (ROC-AUC = 0.8076 on duration alone) that completely invalidates pre-call lead scoring in production.""")

add_code("""# 1. ROC-AUC of duration alone
y_binary = (df['target'] == 'yes').astype(int)
duration_auc = roc_auc_score(y_binary, df['duration'])
print(f"Univariate ROC-AUC on 'duration' ALONE: {duration_auc:.4f}")

# 2. Duration summary statistics by target
print("\\nDuration summary statistics (seconds) by target:")
print(df.groupby('target')['duration'].describe()[['mean', 'std', 'min', '50%', '75%', 'max']])

# 3. Check duration = 0
zero_dur = df[df['duration'] == 0]
print(f"\\nCalls with duration = 0s: {len(zero_dur)} (Conversion rate: {(zero_dur['target'] == 'yes').mean()*100:.1f}%)")

# 4. Visualization: Boxplot and Decile Conversion Rate
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Boxplot
sns.boxplot(data=df, x='target', y='duration', hue='target', showfliers=False, palette=["#e74c3c", "#2ecc71"], legend=False, ax=axes[0])
axes[0].set_title("Call Duration by Outcome (Outliers Hidden)", fontweight="bold")
axes[0].set_ylabel("Duration (seconds)")

# Deciles
df_temp = df[['duration', 'target']].copy()
df_temp['duration_decile'] = pd.qcut(df_temp['duration'], q=10)
decile_conv = df_temp.groupby('duration_decile', observed=False)['target'].agg(lambda s: (s == 'yes').mean() * 100).reset_index()
decile_conv['decile'] = [f"D{i+1}" for i in range(len(decile_conv))]

sns.barplot(data=decile_conv, x='decile', y='target', color='#3498db', ax=axes[1])
axes[1].axhline(y=11.70, color='crimson', linestyle='--', label='Baseline (11.7%)')
axes[1].set_title("Conversion Rate by Duration Decile (D1=Shortest, D10=Longest)", fontweight="bold")
axes[1].set_xlabel("Duration Decile")
axes[1].set_ylabel("Conversion Rate (%)")
axes[1].legend()

plt.tight_layout()
plt.show()""")

# Campaign Outreach
add_md("""## 4. Campaign Outreach Dynamics (`campaign`)

### Observational Association:
Empirical conversion rates decline across higher contact frequency tiers. Rather than proving that calling someone causes them to decline, this observational pattern likely reflects a **negative selection effect**: interested prospects convert on early contacts (1–3), while recalcitrant prospects accumulate repeated contact records precisely because they have not subscribed.""")

add_code("""bins = [0, 1, 2, 3, 5, 10, 100]
labels = ["1 contact", "2 contacts", "3 contacts", "4-5 contacts", "6-10 contacts", ">10 contacts"]
df_camp = df[['campaign', 'target']].copy()
df_camp['campaign_tier'] = pd.cut(df_camp['campaign'], bins=bins, labels=labels)

camp_summary = df_camp.groupby('campaign_tier', observed=False)['target'].agg(
    record_count='count',
    conversions=lambda s: (s == 'yes').sum(),
    conversion_rate=lambda s: (s == 'yes').mean() * 100
).reset_index()
camp_summary['pct_of_all_records'] = camp_summary['record_count'] / len(df) * 100

print(camp_summary.to_string(index=False))

# Visualization: Contact Records vs Conversion Rate
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

x = np.arange(len(camp_summary))
width = 0.4
ax1.bar(x - width/2, camp_summary['record_count'], width=width, color='#bdc3c7', label='Contact Records')
ax2.plot(x + width/2, camp_summary['conversion_rate'], color='#e67e22', marker='o', linewidth=2.5, label='Conversion Rate (%)')

ax1.set_xticks(x)
ax1.set_xticklabels(camp_summary['campaign_tier'], rotation=15, ha='right')
ax1.set_ylabel('Contact Records (Observations)', color='#2c3e50')
ax2.set_ylabel('Conversion Rate (%)', color='#e67e22')
ax1.set_title("Observational Association: Contact Records vs. Conversion Rate", fontsize=13, fontweight='bold')
ax1.grid(False)

plt.tight_layout()
plt.show()""")

# Prior Contacts & Exceptions
add_md("""## 5. Prior Campaign History: `pdays`, `previous`, and `poutcome`

- `pdays = -1` indicates clients never previously contacted (**81.74%** of records, converting at **9.16%**).
- Prior contact (`pdays != -1`) converts at **23.07%** (2.5x higher).
- Prior success (`poutcome == 'success'`) converts at **64.73%** (5.5x higher than average).
- **Audit Discovery**: Exactly **5 records** have `pdays != -1` yet `poutcome` is `NaN` (repeat contacts with unrecorded prior outcome in CRM).""")

add_code("""df_prior = df.copy()
df_prior['contact_history'] = np.where(df_prior['pdays'] == -1, 'Never Contacted (81.7%)', 'Previously Contacted (18.3%)')

pdays_stats = df_prior.groupby('contact_history')['target'].agg(
    volume='count',
    conversions=lambda s: (s == 'yes').sum(),
    conversion_rate=lambda s: (s == 'yes').mean() * 100
).reset_index()
print("--- Prior Contact Cohort Comparison ---")
print(pdays_stats.to_string(index=False))

# Identify exceptions
exceptions = df_prior[(df_prior['pdays'] != -1) & (df_prior['poutcome'].isna())]
print(f"\\nExceptions where pdays != -1 but poutcome is NaN: {len(exceptions)} records")
print(exceptions[['pdays', 'previous', 'poutcome', 'target']])

pout_stats = df_prior.groupby(df_prior['poutcome'].fillna('unknown'))['target'].agg(
    volume='count',
    conversions=lambda s: (s == 'yes').sum(),
    conversion_rate=lambda s: (s == 'yes').mean() * 100
).reset_index().rename(columns={'poutcome': 'prior_outcome'})
print("\\n--- Prior Outcome (poutcome) Breakdown ---")
print(pout_stats.to_string(index=False))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(data=pdays_stats, x='contact_history', y='conversion_rate', hue='contact_history', palette=['#95a5a6', '#2c3e50'], legend=False, ax=axes[0])
axes[0].set_title("Conversion Rate: Never Contacted vs. Previously Contacted", fontweight='bold')
axes[0].set_ylabel("Conversion Rate (%)")
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=12)

sns.barplot(data=pout_stats, x='prior_outcome', y='conversion_rate', hue='prior_outcome', palette='Blues_r', legend=False, ax=axes[1])
axes[1].set_title("Conversion Rate by Prior Campaign Outcome (poutcome)", fontweight='bold')
axes[1].set_ylabel("Conversion Rate (%)")
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                    ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()""")

# Seasonality
add_md("""## 6. Campaign Timing & Seasonality: Observational Trends

Analyzing contact records and observed conversion rates across calendar months. Differences reflect observational associations (differing cohort selections, promotional waves, and economic cycles) rather than causal effects.""")

add_code("""month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
month_df = df.groupby('month')['target'].agg(
    record_count='count',
    conversions=lambda s: (s == 'yes').sum(),
    conversion_rate=lambda s: (s == 'yes').mean() * 100
).reindex(month_order).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.bar(month_df['month'], month_df['record_count'], color='#95a5a6', alpha=0.6, label='Contact Records')
ax2.plot(month_df['month'], month_df['conversion_rate'], color='#27ae60', marker='s', linewidth=2.5, label='Conversion Rate (%)')

ax1.set_ylabel('Total Contact Records (Observations)')
ax2.set_ylabel('Conversion Rate (%)', color='#27ae60')
ax1.set_title("Monthly Seasonality: Contact Records vs. Observed Conversion Rate", fontsize=13, fontweight='bold')
ax1.grid(False)

plt.tight_layout()
plt.show()""")

# Data Quality & Repeated Profiles
add_md("""## 7. Data Quality, Missingness & Repeated Demographic Profiles

- Missing values: Informative missingness in `poutcome` (81.7%) and `contact` (28.8%) must be preserved via explicit categories (`unknown`).
- Repeated demographic/financial profiles: In the absence of unique client IDs, identical profiles represent repeated demographic combinations across contact events rather than verified customer identities.""")

add_code("""# Missing values
print("Missing (NaN) Counts:")
nan_counts = df.isna().sum()[lambda x: x > 0]
print(nan_counts)

# Repeated demographic profiles
demo_cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan']
n_duplicates = df.duplicated(subset=demo_cols).sum()
print(f"\\nRepeated demographic/financial profiles: {n_duplicates:,} out of {len(df):,} ({n_duplicates/len(df)*100:.2f}%)")
print("Validation Note: Stratified random splitting is used with documented limitations due to absence of client IDs and timestamps.")""")

# Conclusion & Phase 2 Blueprint
add_md("""## 8. Summary of Phase 1 Findings & Phase 2 Architecture

### Key Audit Conclusions:
1. **Capacity Benchmarks**: Under a 5,000-call quota, Random Selection yields **585 conversions** (11.70%), while the Business-Rule Benchmark yields **1,664 conversions** (33.28%, 2.84x lift).
2. **Leakage Elimination**: `duration` (ROC-AUC = 0.8076) is strictly post-call information and is programmatically excluded from all pre-call pipelines.
3. **Primary Predictive Drivers**:
   - `poutcome == 'success'` converts at **64.73%** (5.5x baseline).
   - Past contact (`pdays != -1`) converts at **23.07%** (2.5x baseline).
4. **Data Preprocessing Directives for Phase 2**:
   - Drop `duration` via `PreCallFeatureEngineer`.
   - Engineer `was_previously_contacted`, non-negative `pdays_recency`, `has_debt_burden`, and signed `balance_log`.
   - Treat `'unknown'` / NaN as an explicit categorical state for `poutcome` and `contact`.
   - Rank and evaluate models on **PR-AUC**, **Precision@5,000**, and **Lift@5,000**.""")

output_path = Path("notebooks/01_data_exploration.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully generated authoritative {output_path}!")
