# Project-Wide Prediction-Time Feature Availability Audit

**Project**: UCI Bank Marketing Machine Learning System  
**Date**: September 2026  
**Status**: Formal Audit Completed & Contract Established  
**Audit Scope**: Forensic Verification of Prediction-Time Feature Availability & Anti-Leakage Timing Contract

---

## 1. Business Decision

The operational business decision governing this machine learning project is formally defined as:

> **"Before outreach begins for a new campaign, rank an eligible prospect pool and choose which prospects should receive the limited outbound call capacity."**

### Operational Decision Context
- **Fixed Dialing Capacity**: The sales and telemarketing operation has a fixed labor budget capable of executing outbound outreach to a maximum capacity of **5,000 prospects** out of the full eligible population of **45,211 accounts** (a capacity fraction of $\approx 11.059\%$).
- **Allocation Mechanism**: Prospects must be prioritized and scored in batch *prior* to handing outbound call lists to call center representatives.
- **Decision Purpose**: The purpose of the ranking model is prospect selection (lead prioritization), **not** dynamic call-in-progress guidance or retrospective post-campaign response analysis.

---

## 2. Exact Prediction Timestamp

The exact prediction timestamp enforced across all pre-campaign modeling and segmentation is:

> **IMMEDIATELY BEFORE any contact in the new/current campaign occurs.**

### State of the World at the Prediction Timestamp ($t_0$)
At this precise decision moment:
1. **No outbound call has been placed** to the prospect for the new campaign.
2. **No communication channel** (e.g., cellular vs. telephone) has been selected or utilized for the prospect in this campaign.
3. **No contact date** (day of month, calendar month) has transpired for the current campaign.
4. **Current campaign contact count is identically zero** (`campaign = 0` conceptually; the prospect has received 0 attempts in the current campaign).
5. **Only historical and static CRM data exist**:
   - Customer demographic and occupational records stored in the bank CRM profile.
   - Financial account balances and liability records stored in core banking ledger systems.
   - Historical interaction metrics logged during *previous, concluded* marketing campaigns.

---

## 3. Raw Feature Availability Audit

Every raw attribute in the UCI Bank Marketing dataset (Moro et al., 2014) is audited below against the exact prediction timestamp. Features are classified into:
- **A. AVAILABLE PRE-CAMPAIGN**: Legitimate pre-existing attributes available before outreach starts.
- **B. NOT AVAILABLE PRE-CAMPAIGN**: Post-call leakage or current-campaign operational execution variables that do not exist at $t_0$.
- **C. AMBIGUOUS / REQUIRES ASSUMPTION**: Variables whose pre-campaign availability would require unverified operational assumptions.

### Raw Feature Audit Table

| Feature Name | UCI Category | UCI Definition & Meaning | When Known / Generated | Valid at $t_0$? | Classification | Inclusion / Exclusion Reason |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `age` | Bank client data | Age of the bank client (numeric, 18–95). | At client onboarding / KYC profile. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Demographic attribute stored in CRM; immutable prior to campaign launch. |
| `job` | Bank client data | Occupation category (admin., blue-collar, etc.). | In bank client profile. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Static customer employment status available in bank CRM before outreach. |
| `marital` | Bank client data | Marital status (`married`, `single`, `divorced`). | In bank client profile. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Demographic attribute recorded in bank records prior to campaign. |
| `education` | Bank client data | Education level (`primary`, `secondary`, `tertiary`, `unknown`). | In bank client profile. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Customer educational background available in CRM before outreach. |
| `default` | Bank client data | Credit in default? (`yes`, `no`). | Core banking credit records. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Historical credit risk record known prior to outreach initiation. |
| `balance` | Bank client data | Average yearly balance in euros (-€8,019 to +€102,127). | Core banking transaction system. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Financial liquidity calculated from historical account statements prior to campaign. |
| `housing` | Bank client data | Housing mortgage loan? (`yes`, `no`). | Bank loan book / credit system. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Existing active mortgage liability known prior to outreach. |
| `loan` | Bank client data | Personal loan liability? (`yes`, `no`). | Bank loan book / credit system. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Existing active personal loan liability known prior to outreach. |
| `contact` | Current campaign contact | Contact communication type (`cellular`, `telephone`, `unknown`). | During/after outreach attempt. | **NO** | **B. NOT AVAILABLE PRE-CAMPAIGN** | Grouped by UCI under "last contact of current campaign". 28.8% unknown. Reflects operational channel utilized during dialing. Not a fixed pre-call prospect property. *(See Section 5 for detailed analysis)*. |
| `day` / `day_of_week` / `contact_day_of_month` | Current campaign contact | Last contact day of the month (numeric 1–31). | When call is placed. | **NO** | **B. NOT AVAILABLE PRE-CAMPAIGN** | Dialing calendar day is an operational scheduling outcome of the campaign, completely unknown at batch lead selection time. |
| `month` | Current campaign contact | Last contact month of year (`jan`–`dec`). | When call is placed. | **NO** | **B. NOT AVAILABLE PRE-CAMPAIGN** | Campaign records span May 2008 to Nov 2010. Contact month records when the sales team dialed the lead, not a pre-existing prospect trait. |
| `duration` | Current campaign contact | Last contact duration in seconds (0–4,918s). | After call terminates. | **NO** | **B. NOT AVAILABLE PRE-CAMPAIGN** | Catastrophic post-call target leakage. Explicitly flagged by UCI authors as invalid for realistic predictive modeling. |
| `campaign` | Other attributes | Number of contacts performed *during this campaign* for this client (1–63). | Cumulatively during campaign. | **NO** | **B. NOT AVAILABLE PRE-CAMPAIGN** | Records cumulative calls made in the current campaign. At pre-campaign scoring time, exactly 0 contacts have occurred. Conditioning on future call counts is execution leakage. |
| `pdays` | Other attributes | Days passed after client was last contacted from a *previous* campaign (-1 = never). | Concluded in prior campaigns. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Historical relationship variable reflecting recency of past campaign interaction; static before current campaign begins. |
| `previous` | Other attributes | Number of contacts performed *before this campaign* for this client (0–275). | Concluded in prior campaigns. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Historical relationship variable reflecting cumulative past campaign touchpoints; static before current campaign begins. |
| `poutcome` | Other attributes | Outcome of the *previous* marketing campaign (`success`, `failure`, `other`, `unknown`). | Concluded in prior campaigns. | **YES** | **A. AVAILABLE PRE-CAMPAIGN** | Historical outcome from earlier marketing efforts; fixed before current campaign begins. |
| `y` (target) | Output | Subscribed term deposit? (`yes`, `no`). | After campaign concludes. | **NO** | **B. NOT AVAILABLE PRE-CAMPAIGN** | Supervised target label. Forbidden as input. |

---

## 4. Engineered Feature Availability Audit

An engineered feature is legitimate if and only if **100% of its upstream source variables** are legitimate pre-campaign features (Category A). Any engineered feature derived from or combined with a current-campaign variable (Category B) is strictly invalid.

### Engineered Feature Audit Table

| Engineered Feature | Mathematical Definition / Logic | Upstream Source Variables | Source Availability | Status at $t_0$ | Rationale / Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `was_previously_contacted` | `(pdays != -1).astype(int)` | `pdays` | Pre-Campaign (Cat A) | **VALID** | Strictly derived from historical `pdays`; captures whether customer has prior campaign history. |
| `pdays_recency` | $\log(1 + \max(\text{pdays}, 0))$ | `pdays` | Pre-Campaign (Cat A) | **VALID** | Continuous recency transformation mapping -1 to 0; strictly derived from historical `pdays`. |
| `prior_success` | `(poutcome == "success").astype(int)` | `poutcome` | Pre-Campaign (Cat A) | **VALID** | Historical success indicator; strictly derived from previous campaign outcome. |
| `has_debt_burden` | `(housing == "yes") & (loan == "yes")` | `housing`, `loan` | Pre-Campaign (Cat A) | **VALID** | Dual-debt financial indicator; derived strictly from pre-existing bank liability books. |
| `negative_balance_flag` | `(balance < 0).astype(int)` | `balance` | Pre-Campaign (Cat A) | **VALID** | Overdraft / liquidity distress indicator; derived strictly from pre-existing bank balance. |
| `balance_log` | $\text{sign}(b) \cdot \log(1 + \|b\|)$ | `balance` | Pre-Campaign (Cat A) | **VALID** | Signed log1p transform normalizing financial skewness; derived strictly from pre-existing bank balance. |
| `contact_day_of_month` | Source column `day_of_week` (1–31) | `day` / `day_of_week` | During Campaign (Cat B) | **INVALID** | Direct normalization of an invalid current-campaign operational contact date. |
| Any channel features | Derived from `contact` | `contact` | During Campaign (Cat B) | **INVALID** | Derived from invalid current-campaign channel variable. |
| Any seasonality features | Derived from `month` | `month` | During Campaign (Cat B) | **INVALID** | Derived from invalid current-campaign contact month variable. |
| Any fatigue features | Derived from `campaign` | `campaign` | During Campaign (Cat B) | **INVALID** | Derived from invalid cumulative current-campaign attempt count. |

---

## 5. Current-Campaign vs. Previous-Campaign Distinction

A critical architectural distinction must be maintained between **previous-campaign history** and **current-campaign execution**:

```
+----------------------------------------------------------------------------------------------------+
|                                    TEMPORAL PREDICTION BOUNDARY                                    |
|                                                                                                    |
|    PREVIOUS CAMPAIGNS (Historical CRM Data)         |      CURRENT CAMPAIGN (Outreach Execution)   |
|                                                    |                                               |
|  - pdays (days since prior campaign contact)        |  - contact (channel used for outreach)        |
|  - previous (number of prior campaign contacts)    |  - day / contact_day_of_month (dialing day)   |
|  - poutcome (outcome of prior campaign)            |  - month (dialing calendar month)             |
|                                                    |  - campaign (dialing attempt count)           |
|                                                    |  - duration (call length in seconds)          |
|                                                    |  - y (subscription outcome)                   |
|                                                    |                                               |
|  [STATUS: FULLY FIXED, OBSERVED, VALID AT t_0]     |  [STATUS: FUTURE EVENTS, FORBIDDEN AT t_0]    |
+----------------------------------------------------+-----------------------------------------------+
                                                     ^
                                            Prediction Timestamp
                                                    (t_0)
```

### Deep Dive on Specific Variables
1. **`contact`**:
   - UCI categorizes `contact` under *"Related with the last contact of the current campaign"*.
   - 13,020 records (28.8%) have `contact = 'unknown'`.
   - In real-world telemarketing operations, whether an outreach attempt is made via cellular, landline, or fails to connect (leaving it 'unknown') is an operational execution event occurring during the dialing phase.
   - If a bank CRM possessed known phone types prior to outreach, that field would be populated for all customers as client metadata (e.g. "has_cell_phone"). Grouping it under "last contact" and having 28.8% unknown confirms it is a campaign execution artifact. Under the strict pre-campaign contract, it is excluded.
2. **`month` and `day_of_week` (`contact_day_of_month`)**:
   - The UCI dataset covers contacts from May 2008 to November 2010.
   - `month` records the calendar month in which the call center agent happened to reach the prospect.
   - Prioritizing prospects based on `month = 'oct'` or `month = 'mar'` assumes that the telemarketing manager already knows which future calendar month each prospect will be dialed in. In a pre-campaign batch ranking setting, call lists are ranked *before* calls are distributed across the calendar.
3. **`campaign`**:
   - `campaign` records the total number of calls placed to this client *during the current campaign*.
   - At $t_0$, zero calls have been placed in the current campaign.
   - A client who ultimately receives 10 calls did not start with 10 calls on day 1; they accumulated calls over time because they did not answer or did not subscribe initially. Using `campaign` allows models to leverage future call-frequency dynamics that are fundamentally unknown when selecting who to dial.

---

## 6. Downstream Impact Assessment

An audit of every existing codebase module, analysis phase, and deliverable report was conducted. Each artifact is classified into one of three categories:
- **UNCHANGED**: Output relies strictly on valid pre-campaign features and remains 100% methodologically valid.
- **NEEDS RECOMPUTATION**: Output utilized invalid current-campaign features and must be recalculated.
- **NEEDS WORDING UPDATE ONLY**: Methodological logic or descriptive tables remain valid, but explanatory narrative requires updating to document the pre-campaign feature boundary.

### Downstream Artifact Audit Table

| Project Artifact / Phase | Current State / Role | Utilized Invalid Features? | Classification | Detailed Impact & Remediation Required |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 1 EDA & Report** (`reports/data_audit_report.md`) | Forensic exploratory data audit on full dataset. | Evaluated `campaign`, `month`, `contact` descriptively. | **NEEDS WORDING UPDATE ONLY** | Observational statistics remain factually accurate for the historic dataset, but narrative must explicitly clarify that `contact`, `month`, `day`, and `campaign` are during-campaign execution attributes rather than pre-campaign lead scoring predictors. |
| **Feature Engineering Pipeline** (`src/features/build_features.py`) | Pre-call transformer & ColumnTransformer preprocessor. | Retained `campaign`, `contact`, `month`, and normalized `day_of_week`. | **NEEDS RECOMPUTATION** | Must be refactored to strictly enforce `CANONICAL_PRE_CAMPAIGN_RAW_FEATURES` and exclude all `CANONICAL_FORBIDDEN_FEATURES`. |
| **Business-Rule Baseline** (`src/models/baseline.py`) | Non-ML heuristic ranking baseline. | **NO**. Uses only `poutcome`, `pdays`, `loan`, `housing`, `balance`. | **UNCHANGED** | Zero invalid features used. Baseline ranking logic, 1,664 conversions @ 5,000 quota, and 33.28% precision remain **100% valid and unchanged**. |
| **Logistic Regression Candidate** (`src/models/train.py`) | Supervised benchmark model. | Included one-hot `month`, `contact`, `campaign`, and `contact_day_of_month`. | **NEEDS RECOMPUTATION** | Must be retrained on 15 repeated-validation development folds using only canonical pre-campaign features. |
| **Random Forest Candidate** (`src/models/train.py`) | Supervised champion model (`n_estimators=100`, `max_depth=12`). | Included `month`, `contact`, `campaign`, and `contact_day_of_month`. | **NEEDS RECOMPUTATION** | Must be retrained on 15 repeated-validation development folds using only canonical pre-campaign features. |
| **Supervised Model Comparison** (`reports/supervised_model_comparison.md`) | 15-fold repeated validation, paired deltas, model selection. | Compared models trained with invalid features. | **NEEDS RECOMPUTATION** | All validation metrics, paired conversion deltas, and model selection tables must be recomputed using compliant feature pipelines. |
| **Permutation Feature Importance** (`reports/supervised_error_analysis_and_interpretation.md`) | Feature importance ranking on held-out validation folds. | Ranked `month` (#2), `contact` (#3), `day_of_week` (#7), and `campaign` (#9). | **NEEDS RECOMPUTATION** | 4 of the top 9 features were invalid current-campaign execution variables. Permutation importance must be recomputed on compliant features. |
| **Error Analysis & Subgroup Diagnostics** (`reports/supervised_error_analysis_and_interpretation.md`) | OOF error cohorts (TP, FP, FN, TN), subgroup precision. | Analyzed predictions generated by non-compliant models; profiled `month`/`contact`. | **NEEDS RECOMPUTATION** | OOF predictions, error cohorts, score distributions, and subgroup tables must be regenerated from compliant models. |
| **Probability Calibration** (`reports/probability_calibration.md`) | Nested OOF calibration (Platt scaling & Isotonic regression). | Calibrated scores from non-compliant Random Forest. | **NEEDS RECOMPUTATION** | Calibration curves, Brier scores, reliability tables, and calibrators must be refitted on compliant Random Forest OOF scores. |
| **Clustering Feature Matrix** (`src/models/segmentation.py`) | Standardized matrix for unsupervised segmentation. | Included `campaign`, `contact` (2 dummies), and `month` (11 dummies). | **NEEDS RECOMPUTATION** | 14 out of 41 preprocessed dimensions (34.1%) were invalid current-campaign variables. Must be rebuilt with 27 compliant dimensions. |
| **PCA Dimensionality Analysis** (`reports/unsupervised_customer_segmentation.md`) | PCA variance decomposition and loadings. | Fitted on 41-dimensional matrix containing invalid variables. | **NEEDS RECOMPUTATION** | Scree plots, cumulative variance, and component loadings must be recomputed on compliant customer/financial dimensions. |
| **Cluster Profiling & Personas** (`reports/unsupervised_customer_segmentation.md`) | K-Means clustering ($k=3$), stability, personas. | Clusters formed partly along calendar month and contact channel axes. | **NEEDS RECOMPUTATION** | Customer segments, stability metrics, and profiles must be recomputed strictly on pre-campaign customer traits. |

---

## 7. Quantifying Likely Impact

### Features Removed from Models
The removal of invalid current-campaign variables removes 4 of the top 9 predictors from the previous Random Forest model:
1. **`month`** (Previously Rank 2, PR-AUC drop **0.0929**): Seasonality and campaign batch timing.
2. **`contact`** (Previously Rank 3, PR-AUC drop **0.0498**): Cellular vs. telephone channel logging.
3. **`day_of_week` / `contact_day_of_month`** (Previously Rank 7, PR-AUC drop **0.0147**): Calendar day of dialing.
4. **`campaign`** (Previously Rank 9, PR-AUC drop **0.0048**): Cumulative calls placed during campaign.

Combined, these 4 operational execution features accounted for a cumulative validation PR-AUC drop of **~0.1622**.

### What Models Must Rely On
Following remediation, supervised models will be forced to rely exclusively on legitimate customer signals:
- **Historical Campaign Responsiveness**: `poutcome` (previously Rank 1, PR-AUC drop 0.1104), `pdays` / `pdays_recency` (previously Rank 4, PR-AUC drop 0.0466), `previous`, `was_previously_contacted`, `prior_success`.
- **Customer Financial & Balance Profile**: `housing` (previously Rank 5, PR-AUC drop 0.0277), `balance` / `balance_log` (previously Rank 8, PR-AUC drop 0.0062), `loan`, `default`, `has_debt_burden`, `negative_balance_flag`.
- **Customer Demographics & Life Stage**: `age` (previously Rank 6, PR-AUC drop 0.0172), `job`, `marital`, `education`.

### Impact on Unsupervised Segmentation & PCA
- **Feature Space Reduction**: The preprocessed clustering space decreases from **41 dimensions to 27 dimensions** (dropping 11 `month` dummies, 2 `contact` dummies, and 1 `campaign` numeric).
- **Semantic Shift**: Cluster discovery will no longer group customers by the month they were contacted or whether they were reached by mobile phone; segments will represent pure customer financial/demographic archetypes and historical relationship depth.

### Impact on the Business-Rule Baseline
- **ZERO impact**. The business-rule baseline already relied strictly on `poutcome`, `pdays`, `loan`, `housing`, and `balance`. Its benchmark performance (~1,664 conversions @ 5,000 quota, 33.28% precision) remains completely intact.

*(Note: In accordance with methodological standards, no claims are made regarding future ML performance numbers until models are retrained).*

---

## 8. Required Remediation Plan (Execution Sequence)

To remediate the codebase without introducing data snooping or test-set leakage:

```
[Phase A: Contract Establishment] (COMPLETED)
  - Formalize prediction timestamp and feature contract audit report.
  - Implement src/features/feature_contract.py and automated contract tests.
         |
         v
[Phase B: User Review & Approval] (CURRENT GATE)
  - Wait for explicit user review and acceptance of the audit before modifying models.
         |
         v
[Phase C: Pipeline Refactoring]
  - Refactor src/features/build_features.py to strictly enforce CANONICAL_PRE_CAMPAIGN_RAW_FEATURES.
  - Remove duration, contact, month, day_of_week, contact_day_of_month, campaign from preprocessors.
         |
         v
[Phase D: Supervised Model Retraining & Comparison (Phase 3 Recomputation)]
  - Retrain Logistic Regression and Random Forest (unweighted vs. balanced) across 15 repeated folds.
  - Recompute Conversions@k, paired deltas, PR-AUC, ROC-AUC, Precision@k, Recall@k.
  - Update reports/supervised_model_comparison.md.
         |
         v
[Phase E: Supervised Interpretation & Error Analysis (Phase 4 Recomputation)]
  - Recompute 5-fold development OOF predictions, permutation importance, and error matrices.
  - Update reports/supervised_error_analysis_and_interpretation.md and figures.
         |
         v
[Phase F: Probability Calibration (Phase 5 Recomputation)]
  - Refit nested OOF Platt/Isotonic calibrators on compliant RF scores.
  - Update reports/probability_calibration.md and figures.
         |
         v
[Phase G: Unsupervised Customer Segmentation (Phase 6 Recomputation)]
  - Rebuild 27-dimensional clustering matrix; re-run PCA, scree curves, and K-Means.
  - Update reports/unsupervised_customer_segmentation.md and figures.
         |
         v
[Phase H: Phase 1 EDA Narrative Clarification]
  - Update reports/data_audit_report.md with temporal boundary documentation.
```

---

## 9. Limitations & Assumptions

1. **Absence of Customer Identifiers**: The UCI dataset lacks unique customer IDs, account numbers, or timestamps. The unit of analysis is the telemarketing contact record. We assume that customer records represent independent prospecting opportunities at campaign launch.
2. **CRM Data Accuracy Assumption**: We assume that client demographic and financial variables (`age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`) reflect the static state of customer records immediately prior to campaign outreach.
3. **Immutability of Historical Variables**: We assume that `pdays`, `previous`, and `poutcome` reflect settled outcomes of earlier marketing campaigns and were not retroactively modified during current campaign execution.
4. **Holdout Test Set Status**: The current compliant unweighted Random Forest has not been evaluated on the historical 20% holdout (9,043 records), which remains unaccessed during all compliant development recomputations.
