# Bank Marketing Lead Prioritization: Capacity-Constrained Outbound Optimization

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Tests](https://img.shields.io/badge/pytest-57%20passed-brightgreen.svg)]()
[![Methodology](https://img.shields.io/badge/contract-canonical%20pre--campaign-orange.svg)]()

An end-to-end applied machine learning system designed to solve a realistic operational problem in retail banking: **ranking an outbound prospect pool to maximize term deposit conversions under a strict telemarketing capacity quota**.

Based on the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) ($45,211$ records), this project enforces strict prediction-time data availability guardrails, compares non-linear machine learning against two domain baselines across repeated cross-validation folds, audits out-of-fold probability calibration, and conducts parsimonious unsupervised segmentation.

---

## Executive Summary

- **Operational Problem**: A commercial retail bank conducts an outbound telephone campaign to promote term deposits. The call center has operational staffing to reach only **5,000 prospects out of 45,211 eligible accounts** (a fixed capacity fraction of $\sim 11.06\%$).
- **The Core Metric**: **Conversions@capacity** (the absolute number of subscribed term deposits captured within the budgeted 5,000 outbound calls). Arbitrary classification cutoffs (such as F1 at 0.5) are operationally irrelevant under capacity constraints.
- **The Frozen Champion**: An unweighted Random Forest (`max_depth=12`, `n_estimators=100`, `class_weight=None`) evaluated strictly on the 80% development partition ($N_{\text{dev}}=36,168$).
- **Performance vs Baselines** (Evaluated on development partition at $k=4,000$ calls):
  - **Random Dialing**: Captures **$468$ conversions** ($11.70\%$ precision, $1.00\times$ lift).
  - **Domain Business Rule**: Captures **$1,321$ conversions** ($33.03\%$ precision, $2.82\times$ lift).
  - **Random Forest (Out-of-Fold)**: Captures **$1,656$ conversions** ($41.40\%$ precision, $3.54\times$ lift), delivering a net gain of **$+335$ additional customer subscriptions ($+25.4\%$ relative improvement)** over the heuristic rule, and capturing approximately **$1,188$ more conversions** than random selection (about $254\%$ more, corresponding to formal $\text{Lift@capacity} \approx 3.54\times$).
- **Central Operational Limitation**: The model's predictive precision is heavily concentrated among previously contacted prospects ($77.5\%$ recall within cluster). For first-time prospects, available pre-campaign demographic and financial records provide limited separation, yielding an **$18.0\%$ recall** that accounts for **$86.8\%$ of all missed converters**.

---

## Business Problem

Retail bank outbound call campaigns incur substantial staffing expenses and create prospect friction. Calling the entire customer pool is economically inefficient and exhausts client goodwill.

The bank's marketing management established a hard operational constraint:
$$\text{Budget Quota: Call exactly } 5,000 \text{ accounts out of } 45,211 \text{ eligible prospects } (\approx 11.059\%).$$

The business objective is to sort the customer base in descending order of conversion propensity so that the top 5,000 dialed prospects yield the maximum possible number of term deposit contracts.

---

## Decision & Prediction Timestamp

To avoid data leakage, every feature was formally audited against the business decision point:

$$\mathbf{Prediction\ Timestamp:\ Immediately\ BEFORE\ any\ contact\ in\ the\ new/current\ campaign\ occurs.}$$

All candidate predictors must be known CRM records or core banking relationship data existing prior to campaign outreach.

---

## Dataset & Leakage Audit

The project uses the complete UCI Bank Marketing dataset ($45,211$ rows, 16 candidate attributes, binary target `y`).

### Variable Classification & Quarantine

| Raw UCI Feature | Status | Prediction-Time Classification & Quarantine Rationale |
| :--- | :---: | :--- |
| **`duration`** | **FORBIDDEN** | Post-call call duration (seconds). Catastrophic target leakage; unknown prior to call completion. |
| **`contact`** | **FORBIDDEN** | Communication channel (cellular/telephone). Operational routing decision made during campaign rollout. |
| **`month`** | **FORBIDDEN** | Calendar month of call. Campaign scheduling parameter set by operations, not a customer attribute. |
| **`day` / `day_of_week`** | **FORBIDDEN** | Day of month when call occurs. Unknown until call is placed from queue. |
| **`campaign`** | **FORBIDDEN** | Number of contacts during current campaign. Post-launch execution count; zero for all leads pre-campaign. |
| **`y`** | **FORBIDDEN** | Target conversion outcome. Quarantined until post-hoc evaluation. |
| **`age`, `job`, `marital`, `education`** | **PERMITTED** | Demographic CRM records available in account profile. |
| **`default`, `balance`, `housing`, `loan`** | **PERMITTED** | Financial relationship data (account balance, credit default, mortgages, personal loans). |
| **`pdays`, `previous`, `poutcome`** | **PERMITTED** | Historical interaction logs from prior marketing campaigns completed in earlier periods. |

*Detailed Audit Report*: [`reports/prediction_time_feature_contract_audit.md`](reports/prediction_time_feature_contract_audit.md)

---

## Methodology & Workflow

```text
Data Forensic Audit (reports/data_audit_report.md)
    ↓
Prediction-Time Feature Contract (reports/prediction_time_feature_contract_audit.md)
    ↓
Pre-Call Feature Engineering (src/features/build_features.py)
    ↓
Two-Tier Baseline Construction (src/models/baseline.py)
    ↓
Repeated-CV Supervised Model Comparison (reports/supervised_model_comparison.md)
    ↓
Forensic Error Analysis & Cold-Start Diagnosis (reports/supervised_error_analysis_and_interpretation.md)
    ↓
Probability Calibration Evaluation (reports/probability_calibration.md)
    ↓
Parsimonious Unsupervised Segmentation (reports/unsupervised_customer_segmentation.md)
    ↓
Final Decision & Portfolio Synthesis (reports/final_project_summary.md)
```

---

## Evaluation Strategy

- **Development Partition Scope**: All model comparison, error diagnosis, calibration, and clustering were conducted strictly on the 80% development partition ($N_{\text{dev}}=36,168$). The 20% holdout partition ($N=9,043$) was not accessed by the compliant model.
- **Primary Metric**: **Conversions@capacity** ($k_{\text{fold}}=800$ contacts per validation fold; $k_{\text{oof}}=4,000$ contacts pooled).
- **Secondary Metrics**: Precision@capacity, Recall@capacity, Lift@capacity, PR-AUC (Average Precision), and ROC-AUC.
- **Validation Design**: 5-fold cross-validation repeated across 3 random seeds (15 paired evaluation folds).

---

## Supervised Model Results

### Repeated Cross-Validation Comparison (15 Folds, $k=800$ contacts/fold)

| Candidate Model / Configuration | Mean Conversions@800 | Fold Std ($\sigma$) | Mean Precision@800 | Mean Lift@800 | Mean PR-AUC | Mean ROC-AUC | Paired Fold Win Rate vs Ref LR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (unweighted)** | 300.93 | 7.59 | 37.62% | 3.22x | 0.3514 | 0.7191 | 33.3% (5/15) |
| **Logistic Regression (balanced)** | 301.80 | 7.72 | 37.73% | 3.22x | 0.3493 | 0.7199 | Reference (-) |
| **Random Forest (balanced)** | 327.67 | 9.86 | 40.96% | 3.50x | 0.3745 | 0.7295 | 100.0% (15/15) |
| **Random Forest (unweighted)** *(Champion)* | **331.27** | **11.74** | **41.41%** | **3.54x** | **0.3791** | **0.7333** | **100.0% (15/15)** |

### Benchmark Comparison on Development Partition ($k=4,000$ calls)

| Strategy | Selection Basis | Conversions Captured | Precision@capacity | Lift@capacity | Gain vs Random | Gain vs Heuristic |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Dialing** | Random lead ordering | 468.0 | 11.70% | 1.00x | — | — |
| **Business-Rule Heuristic** | Prior successes + debt-free liquidity | 1,321.0 | 33.03% | 2.82x | +853 | — |
| **Random Forest (OOF)** | ML predicted conversion probability | **1,656.0** | **41.40%** | **3.54x** | **+1,188 (+254%)** | **+335 (+25.4%)** |

*Note on Lift Terminology*: Random Forest captures approximately 1,188 more conversions than random selection, or about 254% more conversions, corresponding to formal Lift@capacity ≈ 3.54x (ratio of precision 41.40% to base prevalence 11.70%). Versus the business heuristic, it captures 335 more conversions (+25.4% relative improvement).

### Why Random Forest Earned Its Complexity
- **100% Win Rate Against Linear Models**: Unweighted Random Forest defeated Logistic Regression across **15 out of 15 paired evaluation folds** (mean conversion advantage: **$+30.33$ conversions per 800 calls**).
- **Unweighted Superiority**: Unweighted Random Forest outperformed balanced Random Forest by an average of **$+3.60$ conversions per 800 calls** (winning 10 folds, losing 4, tying 1). Class re-weighting altered probability calibration without improving top-capacity ranking.

*Detailed Comparison Report*: [`reports/supervised_model_comparison.md`](reports/supervised_model_comparison.md)

---

## Model Interpretation

Permutation feature importance was evaluated strictly on held-out validation folds by tracking PR-AUC decrease:

1. **`poutcome`** (0.1086): Dominant predictor reflecting past marketing campaign outcome.
2. **`pdays`** (0.0483): Recency of prior campaign outreach.
3. **`housing`** (0.0399): Primary household credit liability (mortgage).
4. **`age`** (0.0375): Demographic life-stage and retirement transitions.
5. **`balance`** (0.0136): Account liquidity reserves.
6. **`job`** (0.0089): Occupational category.
7. **`marital`** (0.0080): Household marital status.
8. **`loan`** (0.0056): Personal unsecured loan liability.
9. **`education`** (0.0032): Educational attainment tier.
10. **`previous`** (0.0026): Historical touch volume.
11. **`default`** (0.0001): Credit default indicator.

*Methodological Note*: Importance reflects predictive contribution to PR-AUC loss reduction, not causal customer drivers.

---

## Forensic Error Analysis & Cold-Start Diagnosis

Out-of-fold confusion matrix at capacity ($k=4,000$):
- **True Positives ($TP$)**: $1,656$
- **False Positives ($FP$)**: $2,344$
- **Missed Positives ($FN$)**: $2,575$
- **Correctly Rejected ($TN$)**: $29,593$

### The First-Time Prospect Cold-Start Blind Spot
- **$86.83\%$ of missed converters ($FN=2,575$) had no prior campaign interaction (`pdays == -1`)**.
- **Recall Disparity**:
  - Previously contacted prospects (`pdays != -1`): **$77.46\%$ recall** within top capacity.
  - First-time prospects (`pdays == -1`): **$18.01\%$ recall**.
- **Underlying Mechanism**: Missed converters frequently hold mortgage liabilities ($50.10\%$) or personal loans ($13.55\%$). Without prior contact logs to indicate interest, available pre-campaign demographic and balance features provide limited separation from non-subscribers (mean score $0.097$ vs $0.080$ for non-subscribers).

*Detailed Error Analysis Report*: [`reports/supervised_error_analysis_and_interpretation.md`](reports/supervised_error_analysis_and_interpretation.md)

---

## Probability Calibration

The raw RF scores were well aligned with observed conversion frequencies in development OOF validation, though calibration may change under prevalence, population, or campaign drift:
- **Raw Random Forest**: Brier score = **$0.08808$**, Log loss = **$0.30999$**, Intercept $\alpha = -0.0220$, Slope $\beta = 0.9874$.
- **Sigmoid (Platt) Calibration**: Slightly degraded probability loss (Brier $0.08882$, Log loss $0.31289$).
- **Isotonic Calibration**: Degraded PR-AUC from $0.3758$ to $0.3693$ due to prediction score ties and step-function plateaus.
- **Architectural Decision**: **No separate post-processing calibrator was adopted**. Raw ensemble voting proportions are retained.

*Detailed Calibration Report*: [`reports/probability_calibration.md`](reports/probability_calibration.md)

---

## Unsupervised Customer Segmentation

To determine whether prospects group naturally into distinct demographic or behavioral archetypes, PCA and K-Means were fitted on a parsimonious $36,168 \times 33$ feature matrix (4 numeric + 29 symmetric categorical one-hot columns):

### Final $k=3$ Segment Profiles

| Cluster Identifier | Population Share ($N$) | Median Balance | Housing Loan | Personal Loan | Previously Contacted | Observed Conversion Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0: Positive-Balance First-Time** | 66.26% ($23,965$) | €652.00 | 53.39% | 14.73% | 0.20% | 10.04% |
| **Cluster 1: Zero-Balance & Indebted First-Time** | 15.83% ($5,725$) | €0.00 | 58.31% | 25.05% | 1.03% | 5.78% |
| **Cluster 2: Previously Contacted Relationship** | 17.91% ($6,478$) | €623.00 | 62.50% | 13.11% | 100.00% | 23.08% |

### Analytical Finding
The modest silhouette score (~0.2241), together with the observed continuous feature gradients, provides limited evidence for strongly separated compact clusters under this K-Means representation. The $k=3$ solution is therefore treated as a stable descriptive partition rather than evidence of discrete natural customer classes. It functions as a pragmatic descriptive taxonomy for cross-functional communication, but provides negligible new analytical information beyond EDA and supervised error diagnostics.

*Detailed Segmentation Report*: [`reports/unsupervised_customer_segmentation.md`](reports/unsupervised_customer_segmentation.md)

---

## Methodological Correction & Leakage Remediation

A key technical milestone in this project was the formal discovery and remediation of execution-time leakage:

1. **Initial Historical Modeling**: Included `contact`, `month`, `day`, and `campaign`.
2. **Audit Discovery**: These fields reflect mid-campaign execution and scheduling parameters that cannot exist when selecting leads prior to campaign launch.
3. **Remediation**: Rebuilt all feature pipelines under the canonical pre-campaign contract.
4. **Performance Shift**:
   - Historical non-compliant Random Forest: Mean Conversions@800 = **$382.47$** ($47.81\%$ precision).
   - Corrected compliant Random Forest: Mean Conversions@800 = **$331.27$** ($41.41\%$ precision).
   - **Impact**: Performance decreased by **$-51.20$ conversions per 800 calls ($-13.4\%$)**.
5. **Methodological Significance**: This reduction is not an algorithmic regression; it represents the elimination of un-deployable future information, establishing a decision-time-valid pre-campaign development baseline.

---

## Project Limitations & Holdout Status

1. **Cold-Start Conversion Blind Spot**: First-time leads experience low recall ($18.01\%$), accounting for $86.83\%$ of all missed converters.
2. **Holdout Partition Status**: The current compliant unweighted Random Forest has not been evaluated on the historical 20% holdout. However, that partition is no longer a pristine independent test set because it was accessed during earlier historical analyses. Current performance claims therefore rely on development-only repeated cross-validation and out-of-fold evaluation. A genuinely independent final estimate would require future-period or external data.
3. **Predictive Prioritization vs. Prescriptive Interventions**: The model provides predictive lead prioritization under fixed capacity, but the project does not establish causal or prescriptive effects of downstream interventions (such as optimal sales scripting, channel routing, or lead suppression policies). Segmentation remains descriptive.

---

## Repository Structure

```text
uci_banking_ML_project/
├── README.md                                    # Project portfolio entry point
├── requirements.txt                             # Version-bounded Python dependencies
├── config/
│   └── config.yaml                              # Dataset ID, random seed, and CV parameters
├── data/
│   ├── raw/                                     # Cached raw bank features & targets (CSV)
│   └── processed/                               # Processed splits (.gitkeep)
├── models/                                      # Serialized models (.gitkeep, joblib ignored)
├── reports/
│   ├── data_audit_report.md                     # Phase 1: Forensic Data & Pre-Call Leakage Audit
│   ├── prediction_time_feature_contract_audit.md# Prediction-Time Feature Availability Audit
│   ├── supervised_model_comparison.md           # Phase 3: Repeated-CV Supervised Model Comparison
│   ├── supervised_error_analysis_and_interpretation.md # Phase 4: Forensic Error Analysis & Interpretation
│   ├── probability_calibration.md               # Phase 5: Probability Calibration Assessment
│   ├── unsupervised_customer_segmentation.md    # Phase 6: Parsimonious Unsupervised Segmentation
│   ├── final_project_summary.md                 # Definitive Comprehensive Technical Summary
│   └── figures/                                 # 18 publication-quality diagnostic charts (PNG)
├── src/
│   ├── data/
│   │   ├── audit_dataset.py                     # Automated forensic data auditing
│   │   └── load_data.py                         # Dataset loading & caching via ucimlrepo
│   ├── features/
│   │   ├── build_features.py                    # PreCallFeatureEngineer & pre-call transformers
│   │   └── feature_contract.py                  # Canonical pre-campaign contract & validators
│   ├── models/
│   │   ├── baseline.py                          # Random & domain heuristic baseline evaluators
│   │   ├── calibration.py                       # Out-of-fold calibration diagnostics & reliability
│   │   ├── diagnostics.py                       # Error profiling, subgroups, & permutation importance
│   │   ├── evaluate.py                          # Ranking metrics (Conversions@k, PR-AUC, Lift@k)
│   │   ├── segmentation.py                      # PCA, K-Means evaluation, stability, & profiling
│   │   └── train.py                             # Cross-validation model comparison pipeline
│   └── utils/
│       └── logger.py                            # Logging and YAML config management
└── tests/
    ├── test_audit.py                            # Forensic audit calculation tests
    ├── test_calibration.py                      # Probability calibration & loss contract tests
    ├── test_compare.py                          # CV ranking comparison & fold identity tests
    ├── test_data.py                             # Pipeline leakage safety & baseline tests
    ├── test_diagnostics.py                      # Subgroup reconciliation & contract tests
    ├── test_feature_contract.py                 # Canonical feature contract validator tests
    └── test_segmentation.py                     # Parsimonious clustering & stability tests
```

---

## Reproducibility

### 1. Environment Setup

This project requires **Python 3.11** (tested on Python 3.11.9).

```bash
# Clone the repository
git clone https://github.com/Deaversbp/uci_banking_ML_project.git
cd uci_banking_ML_project

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install version-bounded dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite

Verify pipeline integrity, contract enforcement, and metric identities:

```bash
pytest -v
```
*(All 57 unit and integration tests should pass in under 10 seconds).*

### 3. Reproduce Core Analyses & Figures

All core pipeline steps and analytical figures can be reproduced directly using project-relative runner scripts under `scripts/`:

```bash
# 1. Forensic data audit and figures 01-04
python scripts/run_audit.py

# 2. Repeated CV supervised model comparison across 15 folds
python scripts/run_comparison.py

# 3. Model error analysis, interpretation, and diagnostic figures 05-09
python scripts/run_diagnostics.py

# 4. Out-of-fold probability calibration assessment and figures 10-12
python scripts/run_calibration.py

# 5. Parsimonious PCA, segmentation, stability, and figures 13-18
python scripts/run_segmentation.py
```
*(All scripts run self-contained from the project root without hardcoded machine paths, IDE artifacts, or manual copy steps).*

---

## Key Takeaways

1. **Capacity Constraints Dictate Model Selection**: Evaluating models by Conversions@capacity rather than unweighted classification metrics directly aligns data science with business operations.
2. **Feature Timing Integrity Trumps Algorithmic Complexity**: A model with execution leakage appears artificially accurate, but fails in production. Eliminating post-decision variables established an honest, actionable baseline.
3. **Machine Learning Delivers Real Economic Lift**: The compliant Random Forest captures **$+335$ additional subscriptions (+25.4%)** over standard banking business rules across 4,000 calls.
4. **Honest Diagnosis of Cold-Start Limitations**: Highlighting that 86.8% of missed conversions stem from uncontacted prospects provides operational leadership with a realistic roadmap for future data acquisition.
