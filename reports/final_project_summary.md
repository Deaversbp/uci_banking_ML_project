# Comprehensive Project Summary: Outbound Telemarketing Lead Prioritization

**Project**: UCI Bank Marketing Term Deposit Outreach Optimization  
**Phase**: Final Deliverable — Portfolio Synthesis, Methodology Audit, and Validation Summary  
**Evaluated Data**: 80% Development Partition ($N_{\text{dev}} = 36,168$ records; 20% holdout un-evaluated by compliant model)  
**Prediction Timestamp**: Immediately before any contact in the new/current campaign occurs  
**Frozen Supervised Candidate**: Unweighted Random Forest (`max_depth=12`, `n_estimators=100`, `random_state=42`)  
**Status**: Fully Compliant Canonical Pre-Campaign Feature Contract  

---

## 1. Executive Summary

This project delivers a disciplined machine learning solution to a classic operational constraint in retail banking: **allocating limited outbound telemarketing capacity to maximize term deposit subscriptions**. 

Rather than treating the dataset as a generic classification benchmark, the project enforces a realistic operational framing:
- **Calling Capacity Quota**: Outbound sales capacity is constrained to **5,000 calls across 45,211 eligible prospects** (a fixed selection fraction of $\sim 11.06\%$).
- **Objective Metric**: Maximize **Conversions@capacity** (the absolute volume of term deposit conversions captured within the top $11.06\%$ scored leads).
- **Leakage Remediation**: Historical predictive models on this dataset routinely incorporate call duration (post-call leakage) or campaign execution variables (`month`, `contact`, `day`, `campaign`). This project establishes a strict pre-campaign prediction timestamp that purges all execution variables, ensuring models rely strictly on data available *before outreach begins*.
- **Empirical Performance**: Under repeated 5x3-fold cross-validation on the development partition ($k=800$ calls/fold), the frozen compliant unweighted Random Forest captures **$331.27 \pm 11.74$ conversions** ($41.41\%$ precision, $3.54\times$ lift over random dialing). On the pooled development out-of-fold (OOF) evaluation ($k=4,000$ calls), it captures **$1,656$ conversions** ($41.40\%$ precision), outperforming the domain business-rule benchmark ($1,321$ conversions, $33.03\%$ precision) by **$+335$ additional conversions ($+25.4\%$ relative lift)** and random dialing ($468$ conversions) by capturing approximately **$1,188$ more conversions** (about $254\%$ more, corresponding to formal $\text{Lift@capacity} \approx 3.54\times$).
- **Central Operational Limitation**: The model's predictive power is heavily concentrated among prospects with prior campaign interactions ($77.5\%$ recall). For first-time prospects, available pre-campaign demographic and balance features provide limited separation, resulting in an **$18.0\%$ recall** that accounts for **$86.8\%$ of all missed converters**.

---

## 2. Business Decision & Operational Framing

In outbound financial campaigns, call center staffing represents a binding budget constraint. Dialing the entire prospect pool is economically non-viable and results in severe prospect fatigue.

- **The Business Mandate**: *"Immediately before outreach begins for a new campaign, rank an eligible prospect pool of 45,211 accounts and select the top 5,000 prospects to receive outbound calls."*
- **Capacity Scale**: $5,000 / 45,211 \approx 11.059\%$ of the population.
- **Development Partition Scale**: On the 80% development partition ($N_{\text{dev}} = 36,168$), the scaled operational capacity is **$k_{\text{capacity}} = 4,000$ calls** ($800$ calls per fold across 5 cross-validation folds).
- **Optimization Target**: Maximize true positive conversions within the budget quota. Threshold-dependent metrics (such as F1-score at arbitrary cutoffs like $0.5$) are operationally irrelevant; decision-making is governed strictly by capacity-constrained ranking metrics.

---

## 3. Canonical Prediction-Time Feature Contract

A central methodological contribution of this project is the formal auditing and enforcement of feature availability relative to the decision timestamp.

$$\text{Prediction Timestamp: IMMEDIATELY BEFORE any contact in the new/current campaign occurs.}$$

### Variable Classification & Quarantine Rationale

| Raw UCI Variable | Canonical Status | Business Availability Assessment | Operational Quarantine Rationale |
| :--- | :---: | :--- | :--- |
| **`duration`** | **FORBIDDEN** | Measured only after call terminates. | **Catastrophic Target Leakage**: Yields an artificial ROC-AUC of 0.808 on its own. It is impossible to know call duration before dialing. |
| **`contact`** | **FORBIDDEN** | Telecommunication channel (cellular/telephone). | **Operational Routing Choice**: In outbound operations, channel is determined during campaign execution, not an intrinsic prospect property. |
| **`month`** | **FORBIDDEN** | Calendar month of call execution. | **Campaign Scheduling Parameter**: Reflects when management schedules an outbound wave, not a customer attribute available at pre-campaign ranking. |
| **`day` / `day_of_week`** | **FORBIDDEN** | Day of month when call occurred. | **Operational Dialing Timestamp**: Unknown until the call center reaches the prospect in the dialing queue. |
| **`campaign`** | **FORBIDDEN** | Cumulative call attempts in current campaign. | **Execution-Time Counter**: Equal to 0 for all prospects before outreach begins. Non-zero values reflect mid-campaign dialing fatigue. |
| **`y`** | **FORBIDDEN** | Campaign target outcome. | **Target Variable**: Quarantined until evaluation and post-hoc profiling. |
| **`age`, `job`, `marital`, `education`** | **PERMITTED** | Demographic CRM records. | Available in customer profile before campaign launch. |
| **`default`, `balance`, `housing`, `loan`** | **PERMITTED** | Account financial balance & credit liabilities. | Core banking relationship data available at campaign planning. |
| **`pdays`, `previous`, `poutcome`** | **PERMITTED** | Prior campaign historical interaction logs. | Prior interaction records logged during historical marketing waves. |

---

## 4. Dataset Audit & Pre-Call Data Characteristics

The dataset consists of $45,211$ observations from the UCI Bank Marketing repository (bank-full.csv).
- **Partitioning**: Stratified 80/20 train-test split ($N_{\text{dev}} = 36,168$, $N_{\text{holdout}} = 9,043$).
- **Base Prevalence**: $11.698\%$ ($4,231$ converters in development).
- **Historical Interaction Asymmetry**:
  - $81.8\%$ of prospects have never been contacted in a prior campaign (`pdays == -1`, `previous == 0`, `poutcome == 'unknown'`).
  - $18.2\%$ have historical outreach records, exhibiting a conversion prevalence of $22.8\%$ (rising to $64.8\%$ for prior successes).
- **Missing State Preservation**: Missingness in categorical variables (`poutcome` $81.7\%$, `job` $0.6\%$, `education` $4.1\%$) represents an informative operational state ("uncontacted" or "unreported"). It is explicitly preserved as the category `"unknown"`, preventing naive imputation to failure.

---

## 5. Feature Engineering

Engineered features were constructed strictly within a leak-free pipeline obeying the pre-campaign contract:
1. **`balance_log`**: Signed logarithmic transform ($\text{sign}(b) \cdot \log(1 + |b|)$) stabilizing liquidity variance across 4 orders of magnitude (-€8,019 to +€102,128).
2. **`pdays_recency`**: Non-negative logarithmic recency ($\log(1 + \max(\text{pdays}, 0))$), mapping recency continuously without sentinel distortion.
3. **`was_previously_contacted`**: Binary indicator ($I(\text{pdays} \ne -1)$). *(Note: Exactly equal to `pdays_recency > 0` on this dataset).*
4. **`prior_success`**: Binary indicator ($I(\text{poutcome} == \text{'success'})$).
5. **`has_debt_burden`**: Combined liability indicator ($I(\text{housing} == \text{'yes'} \lor \text{loan} == \text{'yes'})$).
6. **`negative_balance_flag`**: Indicator for liquid overdraft ($I(\text{balance} < 0)$).

---

## 6. Two-Tier Baseline Benchmarks

All machine learning models were required to demonstrate statistical and operational superiority over two non-ML baselines:

1. **Random Dialing Baseline**:
   - Dialing randomly yields conversions equal to population prevalence ($11.70\%$).
   - At $k=800$ contacts: **$93.6$ conversions**.
   - At $k=4,000$ contacts: **$468.0$ conversions**.
2. **Compliant Business-Rule Domain Heuristic**:
   - A deterministic rule constructed from domain expertise without model fitting:
     - Tier 1: Prior campaign successes (`poutcome == 'success'`).
     - Tier 2: Previously contacted prospects without debt (`pdays != -1` and debt-free), sorted by balance.
     - Tier 3: Uncontacted prospects without debt (`housing == 'no'`, `loan == 'no'`, `default == 'no'`), sorted by balance.
   - At $k=800$ contacts: **$264.2$ conversions** ($33.03\%$ precision, $2.82\times$ lift).
   - At $k=4,000$ contacts: **$1,321$ conversions** ($33.025\%$ precision, $2.823\times$ lift).

---

## 7. Validation Design

To prevent overfitting, avoid optimistic evaluation bias, and ensure stability:
- **Repeated Cross-Validation**: 5-fold cross-validation repeated across 3 independent random seeds (15 paired evaluation folds).
- **Fold Allocation**: In each validation split, exactly $k_{\text{fold}} = 800$ contacts are selected ($11.06\%$ of the fold validation set).
- **Paired Comparisons**: All candidate models were evaluated on the identical 15 folds, enabling fold-by-fold paired difference calculations:
  $$\Delta_{\text{conversions}} = \text{Conversions@800}(\text{Candidate}) - \text{Conversions@800}(\text{Baseline})$$
- **Out-of-Fold (OOF) Aggregation**: Out-of-fold predictions pooled across folds were used for error analysis and calibration diagnostics.

---

## 8. Supervised Model Comparison & Selection

Four candidate configurations were compared under the compliant pre-campaign contract:

### 15-Fold Repeated Cross-Validation Results ($k=800$ calls/fold)

| Model / Configuration | Mean Conversions@800 | Std Dev ($\sigma$) | Mean Precision@800 | Mean Recall@800 | Mean Lift@800 | Mean PR-AUC | Mean ROC-AUC | Fold Win Rate vs Ref LR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression (unweighted)** | 300.93 | 7.59 | 37.62% | 35.56% | 3.22x | 0.3514 | 0.7191 | 33.3% (5/15) |
| **LogisticRegression (balanced)** | 301.80 | 7.72 | 37.73% | 35.67% | 3.22x | 0.3493 | 0.7199 | Reference (-) |
| **RandomForest (balanced)** | 327.67 | 9.86 | 40.96% | 38.72% | 3.50x | 0.3745 | 0.7295 | 100.0% (15/15) |
| **RandomForest (unweighted)** *(Frozen Champion)* | **331.27** | **11.74** | **41.41%** | **39.15%** | **3.54x** | **0.3791** | **0.7333** | **100.0% (15/15)** |

### Why Random Forest Earned Its Complexity
The unweighted Random Forest was selected based on strict empirical criteria:
1. **Dominant Paired Win Rate**: Random Forest strictly defeated Logistic Regression across **15 out of 15 paired evaluation folds** ($100\%$ win rate).
   - Mean conversion advantage over unweighted LR: **$+30.33$ conversions per 800 calls** (range: $+7.0$ to $+50.0$).
   - Mean conversion advantage over balanced LR: **$+25.87$ conversions per 800 calls** (range: $+8.0$ to $+42.0$).
2. **Unweighted Superiority**: Unweighted Random Forest outperformed balanced Random Forest by an average of **$+3.60$ conversions per 800 calls** (winning 10 folds, losing 4, tying 1). Class-weight balancing shifts probability thresholds without altering rank ordering, but introduces minor tree split distortion under capacity constraints.
3. **Complexity Earning**: Random Forest was adopted not for algorithmic prestige, but because non-linear feature interactions (e.g., between age, balance, and prior contact) delivered an extra **$+67.1$ conversions per 800 calls** over the domain business rule.

---

## 9. Model Interpretation & Forensic Error Analysis

### Compliant Permutation Feature Importance (Validation PR-AUC Decrease)
1. **`poutcome`** (0.1086): Dominant predictor reflecting past campaign outcome.
2. **`pdays`** (0.0483): Recency of prior campaign outreach.
3. **`housing`** (0.0399): Primary household credit liability.
4. **`age`** (0.0375): Demographic life-stage and retirement transitions.
5. **`balance`** (0.0136): Account liquidity reserves.
6. **`job`** (0.0089): Occupational category.
7. **`marital`** (0.0080): Household marital status.
8. **`loan`** (0.0056): Personal loan liability.
9. **`education`** (0.0032): Educational attainment tier.
10. **`previous`** (0.0026): Historical touch frequency.
11. **`default`** (0.0001): Credit default flag.

*Interpretation Caveat*: Importance is strictly predictive within this model, not causal. Correlated variables can redistribute permutation importance.

### Forensic Error Analysis: Missed Positives ($FN = 2,575$)
- **Severe Cold-Start Dilemma**: **$86.83\%$ of all missed converters (2,236 out of 2,575) have never been contacted in a prior campaign (`pdays == -1`)**.
- **Recall Asymmetry**:
  - Previously contacted prospects achieve **$77.46\%$ recall** in the top 4,000 calls.
  - First-time prospects achieve only **$18.01\%$ recall**.
- **Subscribing Under Debt Burden**: While captured true positives are $82.25\%$ debt-free, $50.10\%$ of missed converters hold a housing mortgage and $13.55\%$ hold a personal loan. Lacking prior contact history and carrying debt, their predicted conversion scores are suppressed (mean score $0.097$ vs $0.080$ for non-subscribers), causing the model to bypass them.

---

## 10. Probability Calibration Assessment

Probability calibration was audited using out-of-fold development predictions ($N=36,168$):

| Configuration | Brier Score | Log Loss | Calibration Intercept ($\alpha$) | Calibration Slope ($\beta$) | Conversions Captured ($k=4,000$) | Precision@capacity | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uncalibrated Raw RF** | **0.08808** | **0.30999** | **-0.0220** | **0.9874** | **1,656** | **41.400%** | **0.3758** |
| **Sigmoid Calibrated** | 0.08882 | 0.31289 | +0.0367 | 1.0178 | 1,655 | 41.375% | 0.3759 |
| **Isotonic Calibrated** | 0.08820 | 0.31038 | +0.0090 | 1.0038 | 1,666 | 41.650% | 0.3693 |

### Calibration Decision: No Post-Processing Calibrator Adopted
1. **Development Alignment**: The raw RF scores were well aligned with observed conversion frequencies in development OOF validation, exhibiting near-ideal slope ($0.9874 \approx 1.0$) and intercept ($-0.0220 \approx 0.0$). However, calibration may change under prevalence, population, or campaign drift.
2. **Lowest Probability Error**: Raw scores achieve the lowest Brier score ($0.08808$) and lowest log loss ($0.30999$).
3. **Isotonic Ranking Degradation**: Non-parametric isotonic regression introduces flat step-functions and tied prediction scores, degrading PR-AUC from $0.3758$ to $0.3693$.
4. **Nuance**: "Uncalibrated" means no post-processing calibration layer was added; it does not mean observed development calibration was poor.

---

## 11. Unsupervised Customer Segmentation

To determine whether the pre-campaign customer space contains natural demographic archetypes, PCA and K-Means were fitted on a parsimonious $36,168 \times 33$ feature matrix (4 numeric + 29 symmetric categorical one-hot columns with `drop=None`):

### Final $k=3$ Segment Archetypes

| Cluster Identifier & Persona | Size ($N$) | Population Share | Median Balance | Housing Loan | Personal Loan | Previously Contacted | Observed Conversion Rate | RF Selected in Top 4,000 | Conversions Captured |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0: Positive-Balance First-Time** | 23,965 | 66.26% | €652.00 | 53.39% | 14.73% | 0.20% | 10.04% | 1,486 (6.20%) | 477 (32.10% precision) |
| **Cluster 1: Zero-Balance & Indebted First-Time** | 5,725 | 15.83% | €0.00 | 58.31% | 25.05% | 1.03% | 5.78% | 78 (1.36%) | 20 (25.64% precision) |
| **Cluster 2: Previously Contacted Relationship** | 6,478 | 17.91% | €623.00 | 62.50% | 13.11% | 100.00% | 23.08% | 2,436 (37.60%) | 1,159 (47.58% precision) |

### Segmentation Takeaways
- **Descriptive Partition Rather than Natural Classes**: The modest silhouette score (~0.2241), together with the observed continuous feature gradients, provides limited evidence for strongly separated compact clusters under this K-Means representation. The $k=3$ solution is therefore treated as a stable descriptive partition rather than evidence of discrete natural customer classes.
- **Analytical Value**: Category B/C — provides a useful communication vocabulary for marketing stakeholders, but adds negligible new analytical insight beyond EDA and supervised error diagnostics.

---

## 12. Methodological Remediation Summary

This project explicitly documents the historical remediation of execution-time leakage:

| Analysis Phase | Historical Non-Compliant Implementation | Corrected Compliant Implementation | Impact of Remediation |
| :--- | :--- | :--- | :--- |
| **Feature Space** | Included `contact`, `month`, `day`, and `campaign`. | Quarantined all current-campaign variables; 11 canonical pre-campaign features only. | Eliminates operational data leakage occurring after campaign launch. |
| **Supervised RF Performance** | Mean Conversions@800 = **382.47** (47.81% precision, PR-AUC = 0.4395). | Mean Conversions@800 = **331.27** (41.41% precision, PR-AUC = 0.3791). | Performance decreased by **-51.20 conversions (-13.4%)** as execution shortcuts were purged. |
| **Feature Importance** | Dominated by `month` (Rank 2) and `contact` (Rank 3). | Dominated by `poutcome` (Rank 1), `pdays` (Rank 2), and `housing` (Rank 3). | Replaces dialing schedule shortcuts with genuine customer credit and historical relationship signals. |
| **Clustering Space** | $36,168 \times 41$ matrix using `drop='first'`. | $36,168 \times 33$ parsimonious matrix using `drop=None`. | Purged mid-campaign rotation axes; restored customer balance-sheet structure. |

---

## 13. Final Model Decision

**Adopted Candidate**: Frozen Unweighted Random Forest (`n_estimators=100`, `max_depth=12`, `class_weight=None`, `random_state=42`) without a post-processing calibration layer.
- **Operational Implementation**: In simulated campaign operations, the entire eligible prospect pool is scored in batch at prediction time; the top 5,000 accounts are selected for outbound telemarketing.
- **Expected Lift**: Captures an estimated **$1,656$ conversions per 4,000 calls** ($41.40\%$ precision, $3.54\times$ lift), delivering a net gain of **$+335$ conversions** over domain heuristic rules.

---

## 14. Project Limitations

1. **Cold-Start Vulnerability**: Lacking prior interaction logs, first-time prospects experience low recall ($18.01\%$), accounting for $86.83\%$ of missed subscribers.
2. **Mixed-Data Metric Limitations**: Euclidean distance in K-Means clustering averages discrete categorical indicators into synthetic fractional coordinates.
3. **Cross-Sectional Static Limitation**: Data lacks transactional flow history, credit utilization velocity, or temporal interaction sequences.
4. **Predictive Prioritization vs. Prescriptive Interventions**: The model provides predictive lead prioritization under fixed capacity, but the project does not establish causal or prescriptive effects of downstream interventions (e.g., channel switching, lead suppression, or sales rep routing). Segmentation remains descriptive.

---

## 15. Future Validation Requirements & Holdout Status

- **Status of Historical 20% Holdout ($N=9,043$)**: The current compliant unweighted Random Forest has not been evaluated on the historical 20% holdout. However, that partition is no longer a pristine independent test set because it was accessed during earlier historical analyses. Current performance claims therefore rely on development-only repeated cross-validation and out-of-fold evaluation. A genuinely independent final estimate would require future-period or external data.
- **Requirement for Production Sign-Off**: Genuine production sign-off requires evaluating the frozen pipeline on **new, future-period campaign data** collected after model specification was locked. Current performance evidence rests strictly on development-partition repeated cross-validation and out-of-fold analysis.
