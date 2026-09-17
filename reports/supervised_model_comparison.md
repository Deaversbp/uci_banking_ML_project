# Supervised Model Comparison & Validation Stability Report (Post-Remediation)

**Project**: UCI Bank Marketing Term Deposit Outreach Optimization  
**Phase**: Phase 3 Deliverable — Supervised Model Comparison (Remediated Under Canonical Feature Contract)  
**Date**: September 2026 (Updated Post-Feature Availability Audit)  
**Status**: Completed, Formally Remediated & Frozen  

---

## 1. Executive Summary & Correction History

### Feature Contract Correction Background
In the original Phase 3 execution, candidate supervised models inadvertently included several variables recorded during campaign execution (`contact`, `month`, `contact_day_of_month` / `day_of_week`, and `campaign`). A rigorous prediction-time feature availability audit (documented in `reports/prediction_time_feature_contract_audit.md`) established that:
1. The operational decision is: *"Before outreach begins for a new campaign, rank an eligible prospect pool and choose which prospects should receive the limited outbound call capacity."*
2. The exact prediction timestamp is: **IMMEDIATELY BEFORE any contact in the new/current campaign occurs.**
3. At this timestamp, all current-campaign variables (`contact`, `month`, `contact_day_of_month`, `day_of_week`, `day`, `campaign`, and post-call `duration`) are completely non-existent and represent operational execution leakage.
4. **Correction History**: The feature contract was formally corrected. All candidate models were recomputed strictly on development data using only legitimate pre-campaign variables. **All prior numerical results are formally labeled as NON-COMPLIANT HISTORICAL RESULTS and should not be treated as valid deployment estimates.**

### Core Objective of This Phase
Determine whether **Random Forest's** validation advantage over **Logistic Regression** remains stable, robust, and economically meaningful when evaluated strictly under the canonical pre-campaign feature contract under a fixed-capacity sales constraint (5,000 calls maximum).

### Methodological Guardrails
- **Restricted Candidate Set**: Officially limited strictly to **Logistic Regression** and **Random Forest** (no new model families).
- **Holdout Test Set Status & Protocol Correction**: The holdout was not used for supervised candidate fitting or model selection during the remediation pass. However, its labels were inadvertently included in a full-population business-rule reference calculation. No supervised model was rescored on the holdout. The holdout partition ($N=9,043$) remains quarantined and must not be accessed again. Furthermore, this must be distinguished from the earlier historical balanced-RF holdout evaluation, which remains an old historical artifact and is not a valid independent test estimate for the current compliant unweighted RF.
- **Development-Only Partition**: All repeated cross-validation, candidate comparisons, and business-rule baseline references are evaluated strictly on the 80% development partition ($N=36,168$).
- **Decision Hierarchy**: Primary decision metric is **Conversions@capacity**. Secondary diagnostics are ranking stability, Precision/Recall/Lift, PR-AUC, and ROC-AUC, followed by operational and complexity considerations.

---

## 2. Candidate Models & Canonical Pre-Campaign Pipeline

The candidate models are wrapped in the leak-free `create_pre_call_pipeline`, which strictly enforces `select_canonical_pre_campaign_features`:

| Model Architecture | Base Estimator | Preprocessing & Feature Engineering | Key Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | `PreCallFeatureEngineer(enforce_contract=True)` -> `ColumnTransformer` (Median Imputer + StandardScaler for numeric; Constant 'unknown' + OneHotEncoder for categorical) | `max_iter=1000`, `random_state=42`, `class_weight='balanced'` / `None` |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | `PreCallFeatureEngineer(enforce_contract=True)` -> `ColumnTransformer` (Same canonical pipeline) | `n_estimators=100`, `max_depth=12`, `n_jobs=-1`, `random_state=42`, `class_weight='balanced'` / `None` |

### Features Permitted Under Canonical Contract
- **Valid Raw Features (11)**: `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `pdays`, `previous`, `poutcome`.
- **Approved Engineered Features (6)**: `was_previously_contacted`, `pdays_recency`, `prior_success`, `has_debt_burden`, `negative_balance_flag`, `balance_log`.
- **Forbidden Variables (Purged)**: `duration`, `contact`, `month`, `contact_day_of_month`, `day_of_week`, `day`, `campaign`, `y`.

---

## 3. Validation Design & Capacity Derivation

### Fixed Capacity Derivation
Under the bank's operational budget of 5,000 calls out of 45,211 prospects:
$$\text{capacity\_fraction} = \frac{5{,}000}{45{,}211} \approx 0.11059255 \quad (11.059\%)$$

For any validation partition of size $N_{\text{val}}$, the proportional outreach capacity $k$ is:
$$k_{\text{val}} = \text{round}(N_{\text{val}} \times \text{capacity\_fraction})$$

### Development-Only Partitioning
1. **Outer Split**: Stratified 80/20 split (`random_state=42`). 36,168 development records, 9,043 untouched holdout test records.
2. **Repeated Cross-Validation**: On the development partition, **Repeated Stratified K-Fold** with:
   - **5 Folds** per repeat
   - **3 Repeats**
   - **15 Total Evaluation Splits**
   - Fixed `random_state=42`
3. **Fold Capacity**: Each validation fold contains 7,233 or 7,234 records:
   $$k_{\text{val}} = \text{round}(7{,}233 \times 0.11059255) = 800 \quad \text{calls}$$
   $$k_{\text{val}} = \text{round}(7{,}234 \times 0.11059255) = 800 \quad \text{calls}$$

---

## 4. Class-Weight Sensitivity Analysis (Compliant Recomputation)

Four controlled variants were evaluated across the identical 15 repeated-validation folds under the canonical feature contract:

### Paired Comparison: RandomForest (unweighted) vs. RandomForest (balanced)
$$\Delta_{\text{weight}} = \text{Conversions@}k(\text{RF unweighted}) - \text{Conversions@}k(\text{RF balanced})$$

| Metric | Corrected Paired Weighting Result: RF (unweighted) minus RF (balanced) |
| :--- | :---: |
| **Number of Folds ($N$)** | 15 |
| **Mean $\Delta_{\text{weight}}$** | **+3.60 conversions** |
| **Median $\Delta_{\text{weight}}$** | **+4.00 conversions** |
| **Standard Deviation of $\Delta_{\text{weight}}$** | 7.61 |
| **Minimum / Maximum $\Delta_{\text{weight}}$** | **-12.00 / +20.00 conversions** |
| **Unweighted Wins** | **10 / 15 folds (66.7%)** |
| **Balanced Wins** | **4 / 15 folds (26.7%)** |
| **Ties** | **1 / 15 folds (6.7%)** |

### Secondary Diagnostics: Precision@capacity, PR-AUC, and ROC-AUC
- **Precision@capacity**: Mean paired delta is **+0.45 percentage points** (median: +0.50%, unweighted wins on 10/15 folds).
- **PR-AUC**: Mean paired delta is **+0.0046** (median: +0.0038, unweighted wins on 13/15 folds).
- **ROC-AUC**: Mean paired delta is **+0.0039** (median: +0.0035, unweighted wins on 13/15 folds).

### Weighting Selection Conclusion
In accordance with the project's selection hierarchy:
1. **`class_weight=None` (unweighted) is empirically preferred**: It captures more conversions (+3.60 per fold), higher precision (+0.45%), higher PR-AUC, and wins on 66.7% of folds.
2. Therefore, **`RandomForest(class_weight=None)` is retained as the preferred Random Forest candidate**.

---

## 5. Aggregate Metric Comparison (Compliant Results)

The table below summarizes performance across all 15 validation folds ($k = 800$ contacts per fold). Fold win rate is evaluated against the baseline reference `LogisticRegression (balanced)`:

| Model / Configuration | Mean Conversions@capacity | Std Conversions@capacity | Mean Precision@capacity | Mean Recall@capacity | Mean Lift@capacity | Mean PR-AUC | Mean ROC-AUC | Fold Win Rate vs Ref LR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression (unweighted)** | 300.93 | 7.59 | 37.62% | 35.56% | 3.22x | 0.3514 | 0.7191 | 33.3% (5/15) |
| **LogisticRegression (balanced)** | 301.80 | 7.72 | 37.73% | 35.67% | 3.22x | 0.3493 | 0.7199 | Reference (-) |
| **RandomForest (balanced)** | 327.67 | 9.86 | 40.96% | 38.72% | 3.50x | 0.3745 | 0.7295 | 100.0% (15/15) |
| **RandomForest (unweighted)** | **331.27** | **11.74** | **41.41%** | **39.15%** | **3.54x** | **0.3791** | **0.7333** | **100.0% (15/15)** |

### Non-ML Hurdle Rate Comparison (@ $k=800$)
- **Random Selection Baseline**: Expected **~93.6 conversions** (11.70% precision, 1.00x lift).
- **Compliant Business-Rule Baseline**: Expected **~264.2 conversions** (33.03% precision, 2.82x lift). On the complete 80% development partition ($N_{\text{dev}}=36,168, k_{\text{oof}}=4,000$), the compliant business rule achieves **Conversions@4000 = 1,321** and **Precision@capacity = 33.025%**.
- **Selected Random Forest**: Achieves **331.27 conversions** (41.41% precision, 3.54x lift).
  - Net conversion lift over random outreach: **+237.7 conversions (+254%)**.
  - Net conversion gain over domain heuristic: **+67.1 conversions (+25.4%)**.

---

## 6. Paired Model Comparison & Fold Stability Analysis

Fold-by-fold paired differences were calculated across the 15 identical cross-validation folds:
$$\Delta_{\text{conversions}} = \text{Conversions@}k(\text{Model A}) - \text{Conversions@}k(\text{Model B})$$

### Paired Delta Statistics Table

| Metric | RF (unweighted) vs LR (unweighted) | RF (balanced) vs LR (balanced) | RF (unweighted) vs RF (balanced) | LR (unweighted) vs LR (balanced) |
| :--- | :---: | :---: | :---: | :---: |
| **Number of Folds ($N$)** | 15 | 15 | 15 | 15 |
| **Mean $\Delta_{\text{conversions}}$** | **+30.33 conversions** | **+25.87 conversions** | **+3.60 conversions** | **-0.87 conversions** |
| **Median $\Delta_{\text{conversions}}$** | **+31.00 conversions** | **+28.00 conversions** | **+4.00 conversions** | **-1.00 conversion** |
| **Standard Deviation of $\Delta$** | 12.35 | 10.05 | 7.61 | 2.59 |
| **Minimum $\Delta$** | **+7.00 conversions** | **+8.00 conversions** | **-12.00 conversions** | **-4.00 conversions** |
| **Maximum $\Delta$** | **+50.00 conversions** | **+42.00 conversions** | **+20.00 conversions** | **+3.00 conversions** |
| **Model A Win Rate** | **15 / 15 (100.0%)** | **15 / 15 (100.0%)** | **10 / 15 (66.7%)** | **5 / 15 (33.3%)** |
| **Model B Win Rate** | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** | **4 / 15 (26.7%)** | **9 / 15 (60.0%)** |
| **Tie Rate** | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** | **1 / 15 (6.7%)** | **1 / 15 (6.7%)** |

### Stability Takeaways
- **100% Win Rate Against Logistic Regression**: Random Forest strictly won every single evaluation fold (15/15) against Logistic Regression in both unweighted and balanced settings.
- **Minimum Advantage**: Even in its worst fold, unweighted Random Forest outperformed Logistic Regression by at least **+7.00 conversions**.
- **Illustrative Linear Extrapolation to 5,000 Quota**:
  $$\text{Extrapolated Advantage} = +30.33 \times \frac{5{,}000}{800} \approx 189.6 \text{ conversions}$$

---

## 7. Historical vs. Corrected Performance Comparison

To evaluate the exact impact of removing invalid current-campaign variables, the table below compares historical non-compliant results against the newly recomputed compliant results:

> [!CAUTION]
> **Old Results Warning**: The historical results below are formally labeled:
> **"NON-COMPLIANT HISTORICAL RESULTS — contained current-campaign execution variables"**
> These old numbers reflected execution leakage (dialing channel, call month, contact day, cumulative attempts) that cannot exist when selecting prospects ahead of campaign launch.

### Model Comparison Table (Old Non-Compliant vs. Corrected Compliant)

| Candidate Configuration | Metric | Historical Non-Compliant Value | Corrected Compliant Value | Absolute Difference | Relative Change |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RandomForest (unweighted)** *(Frozen Champion)* | **Mean Conversions@800** | 382.47 | **331.27** | **-51.20** | -13.39% |
| | **Precision@800** | 47.81% | **41.41%** | **-6.40%** | -13.39% |
| | **PR-AUC** | 0.4395 | **0.3791** | **-0.0604** | -13.74% |
| | **ROC-AUC** | 0.7924 | **0.7333** | **-0.0591** | -7.46% |
| **RandomForest (balanced)** | **Mean Conversions@800** | 375.47 | **327.67** | **-47.80** | -12.73% |
| | **Precision@800** | 46.93% | **40.96%** | **-5.97%** | -12.73% |
| | **PR-AUC** | 0.4273 | **0.3745** | **-0.0528** | -12.36% |
| | **ROC-AUC** | 0.7886 | **0.7295** | **-0.0591** | -7.49% |
| **LogisticRegression (unweighted)** | **Mean Conversions@800** | 364.87 | **300.93** | **-63.94** | -17.52% |
| | **Precision@800** | 45.61% | **37.62%** | **-7.99%** | -17.52% |
| | **PR-AUC** | 0.4050 | **0.3514** | **-0.0536** | -13.23% |
| | **ROC-AUC** | 0.7671 | **0.7191** | **-0.0480** | -6.26% |
| **LogisticRegression (balanced)** *(Benchmark Reference)* | **Mean Conversions@800** | 360.73 | **301.80** | **-58.93** | -16.34% |
| | **Precision@800** | 45.09% | **37.73%** | **-7.36%** | -16.34% |
| | **PR-AUC** | 0.4014 | **0.3493** | **-0.0521** | -12.98% |
| | **ROC-AUC** | 0.7679 | **0.7199** | **-0.0480** | -6.25% |

### Analytical Interpretation of Performance Degradation
- **Not a Model Failure**: The drop in Conversions@capacity (-51.20 conversions for RF, -58.93 for LR) is **not** a modeling flaw or algorithmic regression. Rather, it represents the necessary elimination of execution information that is structurally unavailable at prediction time.
- **True Pre-Campaign Baseline**: The corrected figures represent genuine, realistic estimates of lead ranking capability before outbound dialing begins.
- **Relative Invariance**: The relative ranking of models remained completely unchanged: Random Forest still dominates Logistic Regression across 100% of folds, and unweighted Random Forest remains preferred over balanced Random Forest.

---

## 8. Complexity Decision & Final Verdict

### Core Question
> *"Does Random Forest provide a sufficiently consistent increase in conversions at fixed capacity to justify greater complexity than Logistic Regression under the compliant feature contract?"*

### Evaluation
1. **Substantial Conversion Gain**: Unweighted Random Forest delivers an average of **+30.33 conversions per 800-call cohort** over unweighted Logistic Regression (and +29.47 over balanced Logistic Regression). On a 5,000-call quota, this corresponds to an illustrative gain of **~190 additional term deposit subscriptions**.
2. **Total Stability**: Random Forest won on **15 out of 15 folds (100.0%)** against Logistic Regression.
3. **Decisive Hurdle Clearance**: Random Forest captures 331.27 conversions vs 264.2 for the compliant business heuristic (+25.4% gain) and 93.6 for random dialing (+254% gain).
4. **Computational Feasibility**: Fitting Random Forest takes <2 seconds. Batch inference takes <100 ms. Complexity introduces zero operational barrier.

### Verdict
**CONFIRMED.** Random Forest's advantage over Logistic Regression is stable, invariant to weighting, and provides substantial business value under the canonical pre-campaign contract.

---

## 9. New Frozen Supervised Candidate Configuration

| Attribute | New Frozen Supervised Candidate Specification |
| :--- | :--- |
| **Model Family** | **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`) |
| **Pipeline Architecture** | `PreCallFeatureEngineer(enforce_contract=True)` -> `ColumnTransformer` -> `RandomForestClassifier` |
| **Selected Hyperparameters** | `n_estimators=100`, `max_depth=12`, **`class_weight=None` (unweighted)**, `random_state=42`, `n_jobs=-1` |
| **Primary Metric (15-Fold Val)** | **331.27 ± 11.74 Conversions@800** (41.41% Precision, 3.54x Lift) |
| **Secondary Metrics (15-Fold Val)** | **PR-AUC: 0.3791 ± 0.0124**, **ROC-AUC: 0.7333 ± 0.0071**, Recall@800: 39.15% |
| **Historical Test Set Status** | The holdout was not used for supervised candidate fitting or model selection during the remediation pass. However, its labels were inadvertently included in a full-population business-rule reference calculation. No supervised model was rescored on the holdout. The holdout partition ($N=9,043$) remains quarantined and will not be accessed again. Note that this is distinguished from the earlier historical balanced-RF holdout evaluation, which remains an old historical artifact and is not a valid independent test estimate for the current compliant unweighted RF. |

---

## 10. Remaining Methodological Limitations

1. **Information Ceiling for First-Time Prospects**: In the absence of campaign timing and contact channel features, prospects with no prior marketing history (`pdays == -1`) exhibit lower ranking resolution. Models must rely primarily on age, occupation, and financial debt/balance indicators.
2. **Downstream Recomputation Required**: Diagnostic error analysis, permutation importance, probability calibration, PCA, and unsupervised segmentation still reflect historical models and must be systematically recomputed in subsequent phases.
3. **Offline Batch Scoring Assumption**: Models are validated under the assumption that prospect ranking is performed in batch before campaign launch. Real-time dynamic re-ranking during campaign execution is outside the current scope.
