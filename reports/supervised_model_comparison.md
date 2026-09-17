# Supervised Model Comparison & Validation Stability Report

**Project**: UCI Bank Marketing Term Deposit Outreach Optimization  
**Phase**: Phase 3 Deliverable — Supervised Model Comparison  
**Date**: September 2026  
**Status**: Completed & Formally Frozen  

---

## 1. Executive Summary & Objective

The primary objective of this phase is to determine whether **Random Forest's** validation advantage over **Logistic Regression** is sufficiently stable, robust, and economically meaningful to justify its additional model complexity under a fixed-capacity sales constraint (5,000 calls maximum).

### Core Methodological Guardrails
- **Restricted Candidate Set**: Officially limited strictly to **Logistic Regression** and **Random Forest** (no new model families such as XGBoost, CatBoost, or Neural Networks).
- **Frozen Test Set Contract**: The held-out test set was not revisited during this comparison phase; its previously recorded single evaluation remains frozen. Zero test records or test labels were accessed, evaluated, or snooped during this comparison.
- **Development-Only Partition**: All repeated validation, class-weight sensitivity, and model-selection decisions were performed exclusively on the 80% development partition (36,168 records).
- **Leakage Contract**: Post-call `duration` was strictly quarantined and stripped from all preprocessing and model inputs.
- **Decision Hierarchy**: Primary decision metric is **Conversions@capacity** (and Precision@capacity). Secondary diagnostics are stability, Precision/Recall/Lift, PR-AUC, and ROC-AUC, followed by operational and complexity considerations.

---

## 2. Candidate Models & Hyperparameter Specifications

The official candidate set consists of two baseline architectures wrapped in the leak-free `create_pre_call_pipeline`:

| Model Architecture | Base Estimator | Preprocessing & Feature Engineering | Key Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | PreCallFeatureEngineer -> ColumnTransformer (Median Imputer + StandardScaler for numeric; Constant 'unknown' + OneHotEncoder for categorical) | `max_iter=1000`, `random_state=42`, `class_weight='balanced'` / `None` |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | PreCallFeatureEngineer -> ColumnTransformer (Same leak-free pipeline) | `n_estimators=100`, `max_depth=12`, `n_jobs=-1`, `random_state=42`, `class_weight='balanced'` / `None` |

---

## 3. Validation Design & Capacity Derivation

### Capacity Fraction Derivation
The business operates under an operational quota of 5,000 outbound phone calls out of a total eligible population of 45,211 prospects:
$$\text{capacity\_fraction} = \frac{5{,}000}{45{,}211} \approx 0.11059255 \quad (11.059\%)$$

For any validation partition of size $N_{\text{val}}$, the proportional outreach capacity $k$ is calculated as:
$$k_{\text{val}} = \text{round}(N_{\text{val}} \times \text{capacity\_fraction})$$

### Development-Only Partitioning
1. **Outer Split**: The full dataset (45,211 records) was split into an **80% Development Partition** (36,168 records) and a **20% Holdout Test Partition** (9,043 records) using stratified sampling (`random_state=42`). The held-out test set was not revisited during this comparison phase; its previously recorded single evaluation remains frozen.
2. **Repeated Cross-Validation**: On the 36,168-record development set, a **Repeated Stratified K-Fold** design was executed with:
   - **5 Folds** per repeat
   - **3 Repeats**
   - **15 Total Evaluation Splits**
   - Fixed `random_state=42`
3. **Fold Capacity**: Each validation fold contains either 7,233 or 7,234 records:
   $$k_{\text{val}} = \text{round}(7{,}233 \times 0.11059255) = 800 \quad \text{calls}$$
   $$k_{\text{val}} = \text{round}(7{,}234 \times 0.11059255) = 800 \quad \text{calls}$$
   Each candidate model was evaluated by scoring and selecting the top $k = 800$ prospects on each fold's validation split.

---

## 4. Class-Weight Sensitivity Analysis

To rigorously evaluate the effect of class-imbalance weighting without broad hyperparameter snooping, four controlled variants were evaluated across the identical 15 folds:
1. `LogisticRegression (unweighted)`: `class_weight=None`
2. `LogisticRegression (balanced)`: `class_weight='balanced'`
3. `RandomForest (unweighted)`: `class_weight=None`
4. `RandomForest (balanced)`: `class_weight='balanced'`

### Paired Comparison: RandomForest (unweighted) vs. RandomForest (balanced)
Across all 15 validation folds, the paired difference was calculated directly:
$$\Delta_{\text{weight}} = \text{Conversions@}k(\text{RF unweighted}) - \text{Conversions@}k(\text{RF balanced})$$

| Metric | Paired Weighting Result: RF (unweighted) minus RF (balanced) |
| :--- | :---: |
| **Number of Folds ($N$)** | 15 |
| **Mean $\Delta_{\text{weight}}$** | **+7.00 conversions** |
| **Median $\Delta_{\text{weight}}$** | **+8.00 conversions** |
| **Standard Deviation of $\Delta_{\text{weight}}$** | 9.43 |
| **Minimum / Maximum $\Delta_{\text{weight}}$** | **-8.00 / +23.00 conversions** |
| **Unweighted Wins** | **11 / 15 folds (73.3%)** |
| **Balanced Wins** | **4 / 15 folds (26.7%)** |
| **Ties** | **0 / 15 folds (0.0%)** |
| **Fold Deltas ($\Delta$)** | `[-6, +10, +22, +9, +8, +23, +12, +1, -8, +14, -2, +1, +15, -1, +7]` |

### Descriptive Comparison: Precision@capacity & PR-AUC
- **Precision@capacity**:
  - Unweighted RF: Mean **47.81%** (median: 47.75%, std: 1.54%, range: 45.62% - 50.50%).
  - Balanced RF: Mean **46.93%** (median: 47.00%, std: 1.06%, range: 44.50% - 49.00%).
  - Paired Delta: Mean **+0.88 percentage points** (median: +1.00%, std: 1.18%, min: -1.00%, max: +2.88%).
  - Win Rate: Unweighted wins on precision in **11 of 15 folds (73.3%)**.
- **PR-AUC**:
  - Unweighted RF: Mean **0.4395** (median: 0.4386, std: 0.0167, range: 0.4158 - 0.4712).
  - Balanced RF: Mean **0.4273** (median: 0.4218, std: 0.0164, range: 0.4034 - 0.4605).
  - Paired Delta: Mean **+0.0122** (median: +0.0125, std: 0.0045, min: +0.0042, max: +0.0193).
  - Win Rate: Unweighted wins on PR-AUC in **15 of 15 folds (100.0%)**.

### Empirical Sensitivity Conclusions
1. **Weighting Choice**: In accordance with the project's selection hierarchy (primary: Conversions@capacity; secondary: stability, Precision/Recall/Lift, PR-AUC, ROC-AUC), **`class_weight=None` (unweighted) is clearly supported by the development evidence**. It achieves higher average conversions (+7.00 conversions), higher precision (+0.88%), higher PR-AUC (+0.0122 across 100% of folds), and a 73.3% fold win rate on Conversions@capacity.
2. **No Retention Merely for History**: `class_weight='balanced'` is **not** retained merely because it was previously configured. The empirical evidence across 15 development folds demonstrates that unweighted Random Forest yields superior lead-ranking performance.
3. **Model Family Invariance**: In **both** balanced and unweighted settings, Random Forest strictly defeated Logistic Regression on 100% of folds (15/15). The superiority of Random Forest over Logistic Regression is completely invariant to the weighting choice.

---

## 5. Aggregate Metric Comparison

The table below summarizes performance across all 15 validation folds ($k = 800$ contacts per fold). Fold win rate is evaluated against the baseline reference `LogisticRegression (balanced)`.

| Model / Configuration | Mean Conversions@capacity | Std Conversions@capacity | Mean Precision@capacity | Mean Recall@capacity | Mean Lift@capacity | Mean PR-AUC | Mean ROC-AUC | Fold Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression (balanced)** | 360.73 | 9.66 | 45.09% | 42.63% | 3.85x | 0.4014 | 0.7679 | Reference (-) |
| **LogisticRegression (unweighted)** | 364.87 | 10.68 | 45.61% | 43.12% | 3.90x | 0.4050 | 0.7671 | 86.7% (13/15) |
| **RandomForest (balanced)** | 375.47 | 8.48 | 46.93% | 44.37% | 4.01x | 0.4273 | 0.7886 | 100.0% (15/15) |
| **RandomForest (unweighted)** | **382.47** | **12.33** | **47.81%** | **45.20%** | **4.09x** | **0.4395** | **0.7924** | **100.0% (15/15)** |

> [!NOTE]
> All four machine learning variants decisively exceed both non-ML benchmark hurdle rates: Random Selection (expected ~93.6 conversions @ $k=800$, 11.70% precision) and Business-Rule Heuristic (~279 conversions @ $k=800$, 34.90% precision).

---

## 6. Paired Model Comparison & Fold Stability Analysis

Because candidate models were evaluated on identical validation folds, fold-by-fold differences were computed directly:
$$\Delta_{\text{conversions}} = \text{Conversions@}k(\text{RandomForest}) - \text{Conversions@}k(\text{LogisticRegression})$$

### Paired Delta Statistics

| Metric | Configured Balanced: RF (balanced) - LR (balanced) | Unweighted Variants: RF (unweighted) - LR (unweighted) | Preferred RF vs Ref LR: RF (unweighted) - LR (balanced) |
| :--- | :---: | :---: | :---: |
| **Number of Folds ($N$)** | 15 | 15 | 15 |
| **Mean $\Delta_{\text{conversions}}$** | **+14.73 conversions** | **+17.60 conversions** | **+21.73 conversions** |
| **Median $\Delta_{\text{conversions}}$** | **+16.00 conversions** | **+18.00 conversions** | **+21.00 conversions** |
| **Standard Deviation of $\Delta$** | 7.35 | 11.30 | 10.42 |
| **Minimum $\Delta$** | **+1.00 conversion** | **+1.00 conversion** | **+4.00 conversions** |
| **Maximum $\Delta$** | **+28.00 conversions** | **+44.00 conversions** | **+44.00 conversions** |
| **Random Forest Win Rate** | **15 / 15 (100.0%)** | **15 / 15 (100.0%)** | **15 / 15 (100.0%)** |
| **Logistic Regression Win Rate** | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** |
| **Tie Rate** | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** |

### Stability Interpretation
- Random Forest won every single evaluation fold without exception against Logistic Regression. Not a single fold resulted in a loss or tie.
- **Illustrative Linear Extrapolation**: Scaling validation fold differences to the full 5,000-call quota:
  - For balanced RF vs balanced LR: $+14.73 \times (5{,}000 / 800) \approx 92.1$ conversions.
  - For preferred unweighted RF vs balanced LR: $+21.73 \times (5{,}000 / 800) \approx 135.8$ conversions.
  > [!IMPORTANT]
  > These figures (~92 and ~136 conversions) represent **illustrative linear extrapolations from cross-validation fold cohorts**, provided solely to interpret metric scale. They are **not** guaranteed or directly estimated future campaign gains.

---

## 7. Complexity Decision

### Core Question
> *"Does Random Forest provide a sufficiently consistent increase in conversions at fixed capacity to justify greater complexity than Logistic Regression?"*

### Synthesis & Evaluation Across Decision Dimensions

1. **Average Conversions@capacity (Primary Business Objective)**:
   - Preferred unweighted Random Forest captures an average of **382.47 conversions** per fold versus **360.73** for Logistic Regression (balanced) (+21.73 conversions per 800-lead cohort) and **364.87** for Logistic Regression (unweighted) (+17.60 conversions).
   - Even under balanced weighting, Random Forest captures **375.47 conversions** (+14.73 over Logistic Regression).
2. **Fold-to-Fold Stability**:
   - Random Forest achieved a **100.0% win rate (15 out of 15 folds)** against Logistic Regression across all weighting configurations.
   - Fold-to-fold variance remains controlled ($\sigma = 12.33$ for unweighted RF; $\sigma = 8.48$ for balanced RF).
3. **Secondary Ranking Diagnostics**:
   - Random Forest strictly dominates Logistic Regression across all diagnostic measures:
     - PR-AUC: $0.4395$ (unweighted RF) vs $0.4014$ (balanced LR) and $0.4050$ (unweighted LR).
     - ROC-AUC: $0.7924$ vs $0.7679$ and $0.7671$.
     - Precision@k: $47.81\%$ vs $45.09\%$ and $45.61\%$.
     - Recall@k: $45.20\%$ vs $42.63\%$ and $43.12\%$.
     - Lift@k: $4.09x$ vs $3.85x$ and $3.90x$.
4. **Operational & Interpretability Tradeoff**:
   - **Computational Overhead**: Fitting Random Forest (100 trees, depth 12) takes ~1.5 seconds on multicore hardware. Inference on 10,000 records takes <100 milliseconds. Because call lists are generated as an offline batch scoring process (e.g. daily or weekly), the computational delta over Logistic Regression is operationally negligible.
   - **Interpretability**: While Logistic Regression offers simple linear weights, it cannot capture non-linear relationships and interactions without manual feature engineering (e.g., interaction between prior campaign outcome, recency, and debt burden). Random Forest captures these interactions naturally. Feature importance and explainability can be readily provided in subsequent phases via tree-based diagnostics.

### Decision Verdict
**YES.** Random Forest's validation advantage is stable, consistent, and delivers meaningful incremental business value without meaningful operational penalty. The additional model complexity is fully justified.

---

## 8. Final Frozen Supervised Candidate Configuration

The supervised candidate comparison phase is formally **CLOSED**. Based strictly on development-only repeated validation evidence, the selected configuration is:

| Attribute | Frozen Supervised Candidate Configuration |
| :--- | :--- |
| **Model Family** | **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`) |
| **Pipeline Architecture** | `PreCallFeatureEngineer(drop_leakage=True)` -> `ColumnTransformer` -> `RandomForestClassifier` |
| **Selected Hyperparameters** | `n_estimators=100`, `max_depth=12`, **`class_weight=None` (unweighted)**, `random_state=42`, `n_jobs=-1` |
| **Primary Metric (15-Fold Val)** | **382.47 ± 12.33 Conversions@800** (47.81% Precision, 4.09x Lift) |
| **Secondary Metrics (15-Fold Val)** | **PR-AUC: 0.4395**, **ROC-AUC: 0.7924**, Recall@800: 45.20% |
| **Historical Test Set Evaluation** | The held-out test set was not revisited during this comparison phase; its previously recorded single evaluation remains frozen. |

> [!IMPORTANT]
> The held-out test set was not revisited during this comparison phase; its previously recorded single evaluation remains frozen.

---

## 9. Methodological Limitations

1. **Subpopulation Imbalance**: Prospects with no prior campaign contact history (`pdays == -1`) exhibit limited pre-call feature variation, capping ranking resolution within first-time prospect segments.
2. **Fixed Depth & Estimator Hyperparameters**: Hyperparameters (`n_estimators=100`, `max_depth=12`) were frozen to prevent data snooping. Finer hyperparameter optimization (e.g. min_samples_leaf, criterion) was intentionally deferred to later phases.
3. **Absence of Calibrated Probabilities**: While lead ranking depends purely on probability ordering (monotonic invariance), probability calibration has not yet been performed. Probability thresholding for expected value optimization will require future calibration analysis.
4. **Offline Validation Assumption**: Validation assumes that prospect conversion propensities remain stationary over campaign execution waves. Temporal macro shifts (such as interest rate changes) could shift baseline conversion rates across future campaign waves.
