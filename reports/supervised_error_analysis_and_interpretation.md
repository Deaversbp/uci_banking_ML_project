# Phase 4 Deliverable: Supervised Model Interpretation & Error Analysis Report (Remediated)

**Project**: UCI Bank Marketing Term Deposit Outreach Optimization  
**Phase**: Phase 4 Deliverable — Supervised Model Interpretation & Error Analysis (Remediated Under Canonical Feature Contract)  
**Frozen Model Candidate**: `RandomForestClassifier(n_estimators=100, max_depth=12, class_weight=None, random_state=42, n_jobs=-1)`  
**Date**: September 2026 (Updated Post-Feature Availability Audit)  
**Status**: Completed & Evaluated Strictly on Development Partition  

---

## 1. Executive Summary & Correction Protocol

### Feature Contract & Prediction Timestamp
Following the prediction-time feature availability audit (documented in [`reports/prediction_time_feature_contract_audit.md`](prediction_time_feature_contract_audit.md)), all diagnostics in this phase have been recomputed under the canonical pre-campaign feature contract:
- **Canonical Decision**: *"Before outreach begins for a new campaign, rank an eligible prospect pool and determine which prospects should receive the limited outbound call capacity."*
- **Canonical Prediction Timestamp**: **IMMEDIATELY BEFORE any contact in the new/current campaign occurs.**
- **Canonical Raw Features (11)**: `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `pdays`, `previous`, `poutcome`.
- **Forbidden Execution Variables (Purged)**: `duration` (post-call target leakage), `contact` (telecommunication channel), `month` (campaign execution month), `contact_day_of_month` / `day_of_week` / `day` (dialing day), `campaign` (cumulative attempts during current campaign), and target `y`.

### Holdout Test Set Protocol Rectification
> [!IMPORTANT]
> The current compliant unweighted Random Forest has not been evaluated on the historical 20% holdout. However, that partition is no longer a pristine independent test set because it was accessed during earlier historical analyses. Current performance claims therefore rely on development-only repeated cross-validation and out-of-fold evaluation. A genuinely independent final estimate would require future-period or external data.
> 
> All benchmarks in this report use strictly the **80% development-partition business-rule result**:
> - **Conversions@4000**: **1,321**
> - **Precision@capacity**: **33.025%**
> - **Lift@capacity**: **2.82x**

### Methodological Guardrails
1. **Frozen Candidate Architecture**: No hyperparameter tuning, model family modifications, or probability recalibration were introduced.
2. **Strict Development Scope**: All out-of-fold (OOF) predictions, subgroup analyses, error profiling, permutation importances, and baseline comparisons were executed strictly on the 80% development partition ($N_{\text{dev}} = 36,168$).
3. **Proportional Capacity Constraint**:
   $$\text{capacity\_fraction} = \frac{5{,}000}{45{,}211} \approx 0.11059255 \implies k_{\text{oof}} = \text{round}(36{,}168 \times 0.11059255) = 4{,}000 \quad \text{calls}$$

---

## 2. Diagnostic Methodology & Clean OOF Design

To evaluate ranking error without in-sample training optimism, a 5-fold cross-validation scheme was executed across the development partition:

```
Development Partition (N = 36,168 records, 80% of total)
  ├── Fold 1 Val (7,234 rows) <── Scored by RF fitted on Folds 2-5 Train (28,934 rows)
  ├── Fold 2 Val (7,234 rows) <── Scored by RF fitted on Folds 1,3-5 Train (28,934 rows)
  ├── Fold 3 Val (7,234 rows) <── Scored by RF fitted on Folds 1-2,4-5 Train (28,934 rows)
  ├── Fold 4 Val (7,233 rows) <── Scored by RF fitted on Folds 1-3,5 Train (28,935 rows)
  └── Fold 5 Val (7,233 rows) <── Scored by RF fitted on Folds 1-4 Train (28,935 rows)
```

### Protocol & Verification
- **Resampling Scheme**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Pipeline Isolation**: For every fold, preprocessing (`PreCallFeatureEngineer(enforce_contract=True)` + `ColumnTransformer`) was fit exclusively on that fold's training split.
- **Contract Enforcement**: Every fold model verified that zero forbidden fields entered preprocessing or modeling (`validate_pre_campaign_feature_contract`).
- **Integrity Guarantees**:
  - Every development row received **exactly one** out-of-fold prediction ($\sum N_i = 36{,}168$).
  - Zero observations were scored by a model trained on their own data.
  - Zero missing scores were generated.

---

## 3. Capacity-Based Ranking Performance ($k_{\text{oof}} = 4{,}000$)

In outbound telemarketing with a fixed dialer capacity, default 0.50 classification thresholds are operationally irrelevant. The top $k_{\text{oof}} = 4{,}000$ prospects ranked descending by predicted conversion score receive calls.

### Compliant Ranking Error Cohorts

```
                           Actual Positive (Subscriber)      Actual Negative (Non-Subscriber)
Selected in Top 4,000       Top-k True Positive (TP = 1,656)   Top-k False Positive (FP = 2,344)   ---> Selected = 4,000 (k)
Rejected (Ranks 4,001+)     Missed Positive (FN = 2,575)       Correctly Rejected (TN = 29,593)   ---> Rejected = 32,168
                                    |                                         |
                                    v                                         v
                         Total Positives = 4,231                  Total Negatives = 31,937         ---> Total Dev = 36,168
```

### Compliant Empirical Capacity Ranking Metrics

| Metric | Development OOF Value | Operational Interpretation |
| :--- | :---: | :--- |
| **Development Population ($N_{\text{dev}}$)** | **36,168** | Eligible prospects in 80% development partition. |
| **Outreach Capacity ($k_{\text{oof}}$)** | **4,000** | Budgeted call volume ($11.059\%$ capacity fraction). |
| **Conversions Captured ($TP$)** | **1,656** | Subscribed term deposits within top 4,000 calls. |
| **Top-k False Positives ($FP$)** | **2,344** | Non-converting prospects contacted in top 4,000 calls. |
| **Missed Positives ($FN$)** | **2,575** | Actual subscribers not reached within top 4,000 calls. |
| **Correctly Rejected ($TN$)** | **29,593** | Non-subscribers correctly excluded from call campaign. |
| **Precision@capacity** | **41.400%** | Over 4 out of 10 calls placed result in a term deposit. |
| **Recall@capacity** | **39.140%** | Captures $39.14\%$ of all available subscribers in top $11.06\%$ calls. |
| **Lift@capacity** | **3.539x** | Captures 3.54 times more conversions than random dialing. |
| **Pooled OOF Diagnostic Cutoff** | **0.2007** | Prospect ranking score required to secure rank $\le 4{,}000$. |
| **PR-AUC (Average Precision)** | **0.3758** | Global area under the precision-recall curve across all thresholds. |
| **ROC-AUC** | **0.7339** | Global pairwise discrimination concordance. |

### Mathematical Reconciliation Identities
- $TP + FP = 1{,}656 + 2{,}344 = 4{,}000 = k_{\text{oof}}$ (Exact)
- $TP + FN = 1{,}656 + 2{,}575 = 4{,}231 = \text{Total Actual Positives}$ (Exact)
- $TN + FP = 29{,}593 + 2{,}344 = 31{,}937 = \text{Total Actual Negatives}$ (Exact)
- $TN + FN = 29{,}593 + 2{,}575 = 32{,}168 = N_{\text{dev}} - k_{\text{oof}}$ (Exact)
- $TP + FP + FN + TN = 1{,}656 + 2{,}344 + 2{,}575 + 29{,}593 = 36{,}168 = N_{\text{dev}}$ (Exact)

---

## 4. Score-Distribution Findings & Cutoff Terminology

![OOF Score Distribution by Target](figures/05_oof_score_distribution.png)
*Figure 1: Distribution of compliant OOF predicted conversion scores for actual subscribers ($y=\text{'yes'}$, orange) vs. non-subscribers ($y=\text{'no'}$, blue). The dashed red line marks the pooled OOF diagnostic capacity cutoff ($0.2007$).*

![Ranked Score Curve](figures/06_ranked_score_curve.png)
*Figure 2: Descending ranked score curve across all 36,168 development prospects, highlighting the selected top-capacity cohort (green) vs. rejected prospects (grey).*

### Analytical Insights from Score Distributions
1. **Separation and Distribution Shape**:
   - Non-subscribers cluster heavily near zero: median score is **0.071**, mean is **0.080**, and over 80% score below **0.10**.
   - Actual subscribers exhibit a long right tail with a median of **0.125** and mean of **0.248**; 39.14% score above the cutoff of **0.2007**.
2. **Terminology: "Pooled OOF Diagnostic Capacity Cutoff"**:
   - The capacity boundary corresponds to a score of **0.2007**.
   - **Critical Methodological Clarification**: This score is strictly a **pooled OOF diagnostic capacity cutoff**, not a production operational threshold. It represents the score of the 4,000th prospect when pooling uncalibrated out-of-fold ranking scores across 5 independently fitted models.
   - In live deployment, prospect selection will be performed by scoring the entire candidate pool in batch and taking the top $k$ prospects, rather than applying a fixed numeric cutoff.
3. **Threshold vs. Capacity Outreach**:
   - Prospects with scores $\ge 0.50$ number only **659** in total. A naive 0.50 threshold would utilize only **16.5%** of available capacity, contacting 659 prospects and capturing only 522 conversions (abandoning 1,134 viable conversions).

---

## 5. Subgroup Performance Breakdown (Legitimate Dimensions Only)

![Subgroup Precision at Capacity](figures/08_subgroup_precision_and_selection.png)
*Figure 3: Selected subgroup Precision@capacity (% converting in top 4,000) across legitimate pre-campaign dimensions compared against the development base rate (11.70%, red dashed line) and overall top-k precision (41.40%, blue dotted line). All displayed subgroups satisfy the $N_{\text{selected}} \ge 30$ sample size safeguard.*

### Purge of Non-Compliant Dimensions
In accordance with the pre-campaign feature contract, all subgroup breakdowns based on **`contact` (channel)**, **`month` (calendar timing)**, **`campaign` (call count)**, and **`day` (contact day)** have been **completely removed**. Those variables describe execution choices and are unknown when ranking leads ahead of campaign launch.

### Small-Subgroup Safeguard ($N_{\text{selected}} \ge 30$)
Subgroups with fewer than 30 selected prospects are flagged as `[Unstable: N < 30]`. In the compliant recomputation, all major demographic, financial, and historical interaction categories exceed this threshold.

### Compliant Subgroup Performance Table

| Dimension | Subgroup | Population ($N$) | Prevalence | Selected in Top-k | Selection Rate | Conversions Captured | Precision@k | Recall@k | Lift@k | Mean Score | Safeguard Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prior Contact History** | **Previously Contacted (`pdays != -1`)** | 6,584 | 22.84% | 2,456 | 37.30% | 1,165 | **47.43%** | **77.46%** | 2.08x | 0.2293 | Robust ($N \ge 30$) |
| | **Never Contacted (`pdays == -1`)** | 29,584 | 9.22% | 1,544 | 5.22% | 491 | **31.80%** | **18.01%** | 3.45x | 0.0919 | Robust ($N \ge 30$) |
| **Prior Outcome** | **Prior Success** | 1,205 | 64.65% | 1,202 | 99.75% | 779 | **64.81%** | **100.00%** | 1.00x | 0.6447 | Robust ($N \ge 30$) |
| | **Prior Failure** | 3,889 | 12.29% | 796 | 20.47% | 226 | **28.39%** | **47.28%** | 2.31x | 0.1262 | Robust ($N \ge 30$) |
| | **Prior Other** | 1,485 | 16.50% | 453 | 30.51% | 158 | **34.88%** | **64.49%** | 2.11x | 0.1622 | Robust ($N \ge 30$) |
| | **Prior Unknown / Missing** | 29,589 | 9.22% | 1,549 | 5.24% | 493 | **31.83%** | **18.07%** | 3.45x | 0.0919 | Robust ($N \ge 30$) |
| **Debt Burden** | **Debt-Free (No Housing & No Loan)** | 13,702 | 18.25% | 3,307 | 24.14% | 1,362 | **41.19%** | **54.48%** | 2.26x | 0.1793 | Robust ($N \ge 30$) |
| | **Housing Loan Only** | 16,653 | 8.02% | 541 | 3.25% | 247 | **45.66%** | **18.50%** | 5.70x | 0.0806 | Robust ($N \ge 30$) |
| | **Personal Loan Only** | 2,284 | 7.66% | 92 | 4.03% | 28 | **30.43%** | **16.00%** | 3.97x | 0.0881 | Robust ($N \ge 30$) |
| | **Dual Loan Burden (Both Loans)** | 3,529 | 6.26% | 60 | 1.70% | 19 | **31.67%** | **8.60%** | 5.06x | 0.0647 | Robust ($N \ge 30$) |
| **Balance Sign** | **Negative Balance (< €0)** | 3,001 | 5.36% | 40 | 1.33% | 9 | **22.50%** | **5.59%** | 4.19x | 0.0592 | Robust ($N=40 \ge 30$) |
| | **Zero Balance (€0)** | 2,809 | 8.33% | 141 | 5.02% | 61 | **43.26%** | **26.07%** | 5.19x | 0.0830 | Robust ($N \ge 30$) |
| | **Positive Balance (> €0)** | 30,358 | 12.64% | 3,819 | 12.58% | 1,586 | **41.53%** | **41.35%** | 3.29x | 0.1257 | Robust ($N \ge 30$) |
| **Balance Tiers** | **€0 - €499** | 15,855 | 9.92% | 1,262 | 7.96% | 544 | **43.11%** | **34.58%** | 4.34x | 0.0992 | Robust ($N \ge 30$) |
| | **€500 - €1,999** | 10,511 | 12.98% | 1,353 | 12.87% | 539 | **39.84%** | **39.52%** | 3.07x | 0.1299 | Robust ($N \ge 30$) |
| | **€2,000 - €4,999** | 4,519 | 17.13% | 908 | 20.09% | 396 | **43.61%** | **51.16%** | 2.55x | 0.1624 | Robust ($N \ge 30$) |
| | **€5,000+** | 2,282 | 15.73% | 437 | 19.15% | 168 | **38.44%** | **46.80%** | 2.44x | 0.1654 | Robust ($N \ge 30$) |
| **Age Tiers** | **< 30 years** | 4,270 | 17.54% | 1,167 | 27.33% | 438 | **37.53%** | **58.48%** | 2.14x | 0.1617 | Robust ($N \ge 30$) |
| | **30 - 39 years** | 14,481 | 10.59% | 1,011 | 6.98% | 403 | **39.86%** | **26.27%** | 3.76x | 0.1053 | Robust ($N \ge 30$) |
| | **40 - 49 years** | 9,323 | 9.16% | 471 | 5.05% | 233 | **49.47%** | **27.28%** | 5.40x | 0.0948 | Robust ($N \ge 30$) |
| | **50 - 59 years** | 6,647 | 9.12% | 396 | 5.96% | 169 | **42.68%** | **27.89%** | 4.68x | 0.1022 | Robust ($N \ge 30$) |
| | **60+ years** | 1,447 | 33.72% | 955 | 66.00% | 413 | **43.25%** | **84.63%** | 1.28x | 0.3102 | Robust ($N \ge 30$) |
| **Job Category** | **Management** | 7,511 | 13.86% | 941 | 12.53% | 412 | **43.78%** | **39.58%** | 3.16x | 0.1346 | Robust ($N \ge 30$) |
| | **Blue-collar** | 7,830 | 7.29% | 242 | 3.09% | 96 | **39.67%** | **16.81%** | 5.44x | 0.0762 | Robust ($N \ge 30$) |
| | **Technician** | 6,068 | 10.93% | 497 | 8.19% | 200 | **40.24%** | **30.17%** | 3.68x | 0.1099 | Robust ($N \ge 30$) |
| | **Admin.** | 4,141 | 12.29% | 431 | 10.41% | 172 | **39.91%** | **33.79%** | 3.25x | 0.1162 | Robust ($N \ge 30$) |
| | **Services** | 3,348 | 8.90% | 162 | 4.84% | 73 | **45.06%** | **24.50%** | 5.06x | 0.0912 | Robust ($N \ge 30$) |
| | **Retired** | 1,812 | 22.96% | 760 | 41.94% | 325 | **42.76%** | **78.13%** | 1.86x | 0.2253 | Robust ($N \ge 30$) |
| | **Student** | 758 | 28.89% | 472 | 62.27% | 185 | **39.19%** | **84.47%** | 1.36x | 0.2781 | Robust ($N \ge 30$) |
| | **Self-employed** | 1,243 | 11.42% | 153 | 12.31% | 57 | **37.25%** | **40.14%** | 3.26x | 0.1213 | Robust ($N \ge 30$) |
| | **Unemployed** | 1,024 | 15.43% | 148 | 14.45% | 67 | **45.27%** | **42.41%** | 2.93x | 0.1459 | Robust ($N \ge 30$) |
| | **Entrepreneur** | 1,186 | 8.52% | 58 | 4.89% | 23 | **39.66%** | **22.77%** | 4.66x | 0.0932 | Robust ($N \ge 30$) |
| | **Housemaid** | 1,013 | 8.39% | 94 | 9.28% | 29 | **30.85%** | **34.12%** | 3.68x | 0.1022 | Robust ($N \ge 30$) |
| **Education Tier** | **Tertiary Education** | 10,594 | 14.89% | 1,592 | 15.03% | 658 | **41.33%** | **41.72%** | 2.78x | 0.1437 | Robust ($N \ge 30$) |
| | **Secondary Education** | 18,561 | 10.67% | 1,696 | 9.14% | 713 | **42.04%** | **35.99%** | 3.94x | 0.1070 | Robust ($N \ge 30$) |
| | **Primary Education** | 5,531 | 8.62% | 448 | 8.10% | 183 | **40.85%** | **38.36%** | 4.74x | 0.0927 | Robust ($N \ge 30$) |
| | **Unknown Education** | 1,482 | 13.23% | 264 | 17.81% | 102 | **38.64%** | **52.04%** | 2.92x | 0.1390 | Robust ($N \ge 30$) |
| **Marital Status** | **Married** | 21,771 | 10.04% | 1,766 | 8.11% | 798 | **45.19%** | **36.52%** | 4.50x | 0.1019 | Robust ($N \ge 30$) |
| | **Single** | 10,227 | 15.01% | 1,826 | 17.85% | 680 | **37.24%** | **44.30%** | 2.48x | 0.1470 | Robust ($N \ge 30$) |
| | **Divorced** | 4,170 | 12.25% | 408 | 9.78% | 178 | **43.63%** | **34.83%** | 3.56x | 0.1165 | Robust ($N \ge 30$) |

---

## 6. Forensic Error Analysis: False Positives vs. Missed Positives

### Compliant Error Cohort Profile Matrix

| Customer Pre-Campaign Attribute | Top-k True Positives ($TP = 1,656$) | Top-k False Positives ($FP = 2,344$) | Missed Positives ($FN = 2,575$) | Correctly Rejected ($TN = 29,593$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Predicted Score** | **0.483** | **0.350** | **0.097** | **0.080** |
| **Median Predicted Score** | **0.440** | **0.293** | **0.088** | **0.071** |
| **Mean Age** | 44.20 years | 42.59 years | 39.95 years | 40.66 years |
| **Median Age** | 39.0 years | 36.0 years | 38.0 years | 39.0 years |
| **Mean Balance** | **€2,301.87** | **€2,340.08** | **€1,525.52** | **€1,221.98** |
| **Median Balance** | **€1,036.0** | **€1,101.5** | **€603.0** | **€387.0** |
| **Previously Contacted (`pdays != -1`)** | **70.35%** | **55.08%** | **13.17%** | **12.80%** |
| **Never Contacted (`pdays == -1`)** | **29.65%** | **44.92%** | **86.83%** | **87.20%** |
| **Prior Campaign Success (`poutcome`)** | **47.04%** | **18.05%** | **0.00%** | **0.01%** |
| **Debt-Free (No Housing, No Loan)** | **82.25%** | **82.98%** | **44.19%** | **31.28%** |
| **Holding Housing Loan** | **16.06%** | **14.29%** | **50.10%** | **61.81%** |
| **Holding Personal Loan** | **2.84%** | **4.48%** | **13.55%** | **17.95%** |
| **Negative Balance Flag (< €0)** | **0.54%** | **1.32%** | **5.90%** | **9.49%** |
| **Top 3 Occupations** | Management (25.1%), Retired (19.8%), Tech (12.2%) | Management (22.8%), Retired (18.8%), Tech (12.8%) | Management (24.5%), Blue-collar (18.5%), Tech (18.1%) | Blue-collar (24.2%), Management (20.2%), Tech (17.4%) |
| **Education: Secondary / Tertiary** | 45.9% / 42.3% | 45.1% / 42.8% | 51.1% / 37.0% | 54.8% / 28.4% |
| **Marital: Married / Single** | 48.2% / 41.1% | 41.3% / 48.9% | 53.9% / 33.2% | 62.9% / 25.5% |

### False-Positive Findings ($FP = 2,344$)
1. **High Demographic and Financial Overlap**:
   - False Positives and True Positives are remarkably similar across observable pre-campaign financial and demographic attributes.
   - Median balance is actually higher for False Positives (**€1,101.50** vs **€1,036.00** for True Positives).
   - Debt-free proportion is essentially identical (**82.98%** for FP vs **82.25%** for TP).
   - Housing loan presence is equally low (**14.29%** for FP vs **16.06%** for TP), and negative balances are rare in both (**1.32%** vs **0.54%**).
   - Both cohorts are dominated by management professionals, retirees, and technicians with secondary or tertiary education.
2. **Prior Interaction Discrepancy**:
   - The primary observable divergence is in prior campaign interaction history: True Positives have a higher prevalence of prior success (**47.04%** vs **18.05%** in FP) and prior contacts (**70.35%** vs **55.08%** in FP).
   - However, 423 False Positives were prior campaign successes who did not re-subscribe.
3. **Objective Analytical Takeaway**:
   - The observed pre-campaign features do not distinguish why these qualified, liquid, debt-free prospects declined to subscribe.
   - We make no causal claims or speculative assertions regarding unobserved conversational or situational factors. We document only that under the available pre-campaign feature contract, False Positives represent commercially qualified leads exhibiting strong financial capacity.

### Missed-Positive Findings ($FN = 2,575$) & Cold-Start Impact
1. **Severe First-Time Outreach Cold-Start Problem**:
   - **86.83% of all Missed Positives (2,236 out of 2,575) have never been contacted previously (`pdays == -1`)**.
   - In contrast, only **29.65%** of captured True Positives were first-time contacts.
   - Removing current-campaign variables (`month`, `contact`, `campaign`) directly exacerbates this cold-start dynamic: without in-campaign execution signals, first-time prospects have fewer distinguishing variables.
2. **Zero Prior Success History**:
   - Exactly **0.00%** of Missed Positives had prior campaign success (`poutcome == 'success'`).
3. **Subscribing Under Financial Liability**:
   - While True Positives are 82.25% debt-free, **over half of Missed Positives hold a housing loan (50.10%)**, and **13.55% hold a personal loan** (vs 2.84% in TP). Only 44.19% are debt-free.
   - Median balance is **€603.00** (vs €1,036.00 for TP), and 5.90% have negative balances.
4. **Severe Score Suppression**:
   - Because they lack prior contact history and carry loans, Missed Positives receive low model scores (mean **0.097**, median **0.088**), placing them well below the pooled OOF diagnostic capacity cutoff of 0.2007.
5. **Analytical Takeaway**:
   > [!NOTE]
   > **The current compliant feature set and frozen Random Forest provide limited separation for first-time converters who carry debt burdens and lack previous campaign interaction records.** In the observed pre-campaign feature space, their profiles appear demographically and financially similar to the broader non-subscribing population (mean score 0.097 vs 0.080 for TN).

---

## 7. Model Interpretation: Permutation Feature Importance

![Permutation Feature Importance](figures/07_permutation_importance.png)
*Figure 4: Permutation feature importance of canonical raw pre-campaign features, evaluated on held-out validation folds across 5 cross-validation splits using PR-AUC (Average Precision) decrease. Error bars represent $\pm 1$ standard deviation.*

### Compliant Raw Feature Permutation Importance on Held-Out Folds

Permutation importance was evaluated strictly on held-out validation folds to quantify genuine generalization loss when each raw pre-campaign feature is shuffled:

| Rank | Raw Pre-Campaign Feature | Mean PR-AUC Decrease | Fold Std Dev ($\sigma$) | Min PR-AUC Decrease | Max PR-AUC Decrease | Pre-Campaign Domain Role |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **`poutcome`** | **0.1086** | 0.0123 | 0.0956 | 0.1238 | Prior campaign outcome (success, failure, other). |
| **2** | **`pdays`** | **0.0483** | 0.0051 | 0.0422 | 0.0557 | Days elapsed since last contact from prior campaign. |
| **3** | **`housing`** | **0.0399** | 0.0044 | 0.0330 | 0.0431 | Presence of housing mortgage liability. |
| **4** | **`age`** | **0.0375** | 0.0053 | 0.0300 | 0.0415 | Client age (demographic lifecycle stage). |
| **5** | **`balance`** | **0.0136** | 0.0066 | 0.0057 | 0.0219 | Average yearly balance in euros. |
| **6** | **`job`** | **0.0089** | 0.0034 | 0.0045 | 0.0123 | Client occupation category. |
| **7** | **`marital`** | **0.0080** | 0.0037 | 0.0047 | 0.0137 | Marital status (single, married, divorced). |
| **8** | **`loan`** | **0.0056** | 0.0007 | 0.0046 | 0.0066 | Presence of personal loan liability. |
| **9** | **`education`** | **0.0032** | 0.0023 | 0.0009 | 0.0069 | Highest educational tier attained. |
| **10** | **`previous`** | **0.0026** | 0.0031 | -0.0014 | 0.0065 | Number of historical contacts prior to campaign. |
| **11** | **`default`** | **0.0001** | 0.0006 | -0.0007 | 0.0008 | Credit in default flag. |

### Comparison with Old Historical Non-Compliant Importance Ranking

In the historical non-compliant model, four execution variables distorted the feature hierarchy:
- `month`: Formerly Rank 2 (mean PR-AUC decrease: **0.0929**)
- `contact`: Formerly Rank 3 (mean PR-AUC decrease: **0.0498**)
- `day_of_week` (`contact_day_of_month`): Formerly Rank 7 (mean PR-AUC decrease: **0.0147**)
- `campaign`: Formerly Rank 9 (mean PR-AUC decrease: **0.0048**)

**What variables replace those signals once only legitimate pre-campaign information remains?**
1. **`poutcome` (0.1086)** and **`pdays` (0.0483)** remain the primary predictors of term deposit subscription.
2. In the absence of seasonal and dialing channel shortcuts, **`housing` mortgage debt** rises from rank 5 to **Rank 3** (importance increased from 0.0277 to **0.0399**, a **+44.0% increase in relative reliance**).
3. **`age`** rises from rank 6 to **Rank 4** (importance increased from 0.0172 to **0.0375**, a **+118% increase in relative reliance**).
4. **`balance`** rises from rank 8 to **Rank 5** (importance increased from 0.0062 to **0.0136**, a **+119% increase in relative reliance**).

> [!NOTE]
> **Non-Causal Interpretation**: Feature importance reflects predictive reliance in reducing validation PR-AUC loss, not causal impact. Shuffling `housing` damages model ranking because housing debt strongly correlates with household liquidity constraints, not because taking out a mortgage causes a customer to decline a savings deposit. Furthermore, importance magnitudes are conditional on the fitted model and feature set. Removing correlated/current-campaign variables can redistribute permutation importance across remaining features, so percentage changes are descriptive rather than intrinsic increases in feature importance.

---

## 8. Random Forest vs. Business-Rule Baseline Selection Overlap

![RF vs Business Rule Overlap](figures/09_rf_vs_business_rule_overlap.png)
*Figure 5: Set overlap in lead selections (left) and conversion yield (right) between Random Forest and the frozen Business-Rule heuristic within the identical top $k=4,000$ capacity.*

Evaluated on the identical 80% development partition ($N=36,168$) under the identical capacity constraint ($k_{\text{oof}} = 4,000$):

| Selection Cohort | Number of Leads | % of Top-k | Conversions Captured | Precision within Cohort |
| :--- | :---: | :---: | :---: | :---: |
| **Shared Selections (Both RF & Rule)** | **2,359** | 58.98% | **1,146** | **48.58%** |
| **Unique to Random Forest (ML Discoveries)** | **1,641** | 41.02% | **510** | **31.08%** |
| **Unique to Business Rule (Heuristic Only)** | **1,641** | 41.02% | **175** | **10.66%** |
| **Overall Random Forest ($k=4,000$)** | **4,000** | 100.00% | **1,656** | **41.40%** |
| **Overall Business Rule Baseline ($k=4,000$)** | **4,000** | 100.00% | **1,321** | **33.025%** |
| **Jaccard Similarity Index** | **0.418** | — | — | — |
| **RF Net Conversion Advantage** | **+335 conversions (+25.36% gain)** | — | — | — |

### Qualitative Analysis of RF Discoveries vs. Rule Over-Commitment
1. **The Heuristic's Structural Limitation**:
   - The business rule strictly prioritizes prior successes (+1,000 pts) and prior contacts without personal loans (+500 pts).
   - Once high-propensity repeat contacts are exhausted, the heuristic fills its remaining quota with repeat contacts who hold mortgages or personal loans.
   - As a result, the **1,641 leads selected only by the business rule convert at only 10.66%** (below the 11.70% random dialing baseline).
2. **What Random Forest Discovers**:
   - Random Forest breaks free of the repeat-contact bias: **94.09% of RF-unique selections had never been contacted previously (`pdays == -1`)**.
   - RF discovers liquid, debt-free demographic segments among higher-conversion first-time prospects:
     - **Retirees and students**: 28.26% retired and 17.80% students (combined 46.06%), with a mean age of 43.34 years (vs 40.27 years for rule-unique leads).
     - **High debt-free proportion**: **88.79%** of RF-unique selections are completely debt-free (vs only **25.29%** for rule-unique selections).
   - Consequently, RF's 1,641 unique selections convert at **31.08%** (capturing **510 conversions** vs 175 for the rule), delivering a net gain of **+335 term deposit subscriptions**.

---

## 9. Quantifying Remediation Impact (Old Non-Compliant vs. Corrected Diagnostics)

> [!CAUTION]
> **Historical Comparison Disclaimer**:
> All results labeled **"NON-COMPLIANT HISTORICAL DIAGNOSTICS"** reflect models and diagnostics that included current-campaign execution variables (`month`, `contact`, `campaign`, `day`). They are obsolete and invalid for production planning.

### Diagnostic Comparison Table

| Metric | Non-Compliant Historical Diagnostics | Corrected Compliant Diagnostics | Absolute Difference | Relative Change | Primary Methodological Cause |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **OOF Conversions@4000** | **1,908** | **1,656** | **-252** | **-13.21%** | Removal of execution variables (`month`, `contact`, `campaign`, `day`). |
| **Precision@capacity** | **47.70%** | **41.40%** | **-6.30%** | **-13.21%** | True pre-campaign prospect ranking without operational leakage. |
| **Recall@capacity** | **45.10%** | **39.14%** | **-5.96%** | **-13.21%** | Fewer conversions captured within fixed 4,000 call capacity. |
| **Lift@capacity** | **4.08x** | **3.54x** | **-0.54x** | **-13.24%** | Baseline development prevalence is 11.70%. |
| **PR-AUC (Average Precision)** | **0.4395** | **0.3758** | **-0.0637** | **-14.49%** | Elimination of strong seasonal and channel separation. |
| **ROC-AUC** | **0.7924** | **0.7339** | **-0.0585** | **-7.38%** | True pre-campaign discrimination power. |
| **First-Time Contact Recall** | **27.29%** | **18.01%** | **-9.28%** | **-34.01%** | First-time leads (`pdays == -1`) lose campaign timing signals. |
| **Prior-Contact Recall** | **76.62%** | **77.46%** | **+0.84%** | **+1.10%** | Repeat contact conversion capture remains highly stable. |
| **RF vs. Rule Net Advantage** | **+587 conversions** | **+335 conversions** | **-252** | **-42.93%** | RF advantage remains decisive (+25.36% over compliant rule). |

### Analytical Explanation of Differences
- **Legitimate Degradation**: The reduction in Conversions@4000 (from 1,908 to 1,656) is **not a modeling defect**. It represents the elimination of artificial predictability from variables that cannot exist when selecting prospects ahead of campaign launch.
- **First-Time Prospect Penalty**: The loss in recall is heavily concentrated in first-time contacts (`pdays == -1`), where recall dropped from 27.29% to 18.01%. When `month` and `contact` are unavailable, the model must rely solely on age, balance, debt, and job.
- **Repeat-Contact Invariance**: Recall for previously contacted prospects remained virtually unchanged (76.62% vs 77.46%), proving that the model's core mechanism for prioritizing repeat relationships is robust and genuine.

---

## 10. Summary of Analytical Findings & Key Business Insights

1. **Prior Campaign Success is the Anchor Signal**:
   - `poutcome == 'success'` remains the single most powerful pre-campaign signal, driving a 0.1086 drop in PR-AUC. The model captures **100.0% of available prior successes** within the top 4,000 capacity.
2. **Financial Liabilities Suppress Conversion Propensity**:
   - Holding a housing loan (`housing == 'yes'`) or personal loan (`loan == 'yes'`) severely reduces conversion probability. Debt-free clients convert at 41.19% within top-k, whereas clients with dual loan burdens convert at only 31.67%.
3. **The First-Time Contact Blindspot**:
   - **86.83% of Missed Positives are first-time contacts (`pdays == -1`)**.
   - The current compliant feature set and frozen Random Forest provide limited separation for first-time converters who carry debt burdens.
4. **False Positives are Financially Qualified Prospects**:
   - False Positives share high balances (€1,101.50 median), high debt-free rates (82.98%), and similar occupational distributions with True Positives. They represent commercial prospects who declined rather than disqualified leads.
5. **Decisive Advantage Over Business Heuristics**:
   - Random Forest captures **+335 additional subscriptions (+25.36% gain)** over the compliant domain baseline ($k=4,000$) by identifying liquid retirees and students rather than recycling indebted past contacts.

---

## 11. Remaining Methodological Limitations

1. **Information Ceiling on First-Time Contacts**: Without interaction history, first-time prospects exhibit limited feature variance in core banking records.
2. **Uncalibrated Ranking Scores**: Scores reflect relative rank ordering, not calibrated probabilities. The scores should not be treated as reliable calibrated probabilities for expected-value decisions until calibration quality has been evaluated.
3. **Absence of Real-Time Interaction Data**: Actual conversion depends partly on conversational interaction and real-time customer context that are unobservable at pre-campaign selection time.
4. **Holdout Status**: The current compliant unweighted Random Forest has not been evaluated on the historical 20% holdout. However, that partition is no longer a pristine independent test set because it was accessed during earlier historical analyses. Current performance claims therefore rely on development-only repeated cross-validation and out-of-fold evaluation. A genuinely independent final estimate would require future-period or external data.
