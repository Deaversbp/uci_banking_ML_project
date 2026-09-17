# Phase 5 Deliverable: Probability Calibration Report

**Project**: UCI Bank Marketing Term Deposit Outreach Optimization  
**Phase**: Phase 5 Deliverable — Probability Calibration for Frozen Supervised Random Forest  
**Frozen Model Candidate**: `RandomForestClassifier(n_estimators=100, max_depth=12, class_weight=None, random_state=42, n_jobs=-1)`  
**Date**: September 2026  
**Status**: Completed & Evaluated Strictly on 80% Development Partition  

---

## 1. Executive Summary & Objective

In outbound telemarketing lead prioritization, a predictive model serves two distinct operational functions:
1. **Lead Ranking**: Deciding which prospects to contact first under a constrained outreach capacity ($k$).
2. **Probability Estimation**: Estimating an individual prospect's empirical conversion chance ($P(Y=1 \mid X)$) for economic expected-value calculations, variable cost optimization, or policy thresholding.

The primary objective of this phase is to determine whether the continuous scores produced by the frozen unweighted Random Forest can be calibrated into reliable probabilities that correspond to observed conversion rates, while ensuring that calibration **does not materially harm the existing top-capacity ranking performance** ($k_{\text{oof}} = 4{,}000$).

### Core Methodological Guardrails
- **Frozen Model Architecture**: No hyperparameter tuning, tree restructuring, or new classifier families were introduced.
- **Strict Partition Isolation**: The held-out 20% test set (9,043 records) was **not** revisited, loaded, or scored. All diagnostics and calibration evaluations were conducted strictly on the 80% development partition (36,168 records).
- **Leakage Quarantine**: Post-call `duration` was strictly dropped before all preprocessing, modeling, and calibration workflows.
- **Nested Out-of-Fold Evaluation**: Evaluated using a 5-fold outer cross-validation scheme with internal 3-fold cross-validation inside each outer training fold (`CalibratedClassifierCV(..., ensemble=False)`). Neither the base estimator nor the calibrator ever observed outer-validation labels during fitting.
- **Business Ranking Alignment**: Outreach capacity remains anchored to the business quota: $k_{\text{oof}} = \text{round}(36{,}168 \times 5{,}000 / 45{,}211) = 4{,}000$ contacts ($11.059\%$ capacity fraction).

---

## 2. Why Ranking and Calibration Are Different

It is critical to distinguish between a **ranking score** and a **calibrated probability**:

```
+--------------------------------------------------------------------------------------------------+
|                                    RANKING vs. CALIBRATION                                       |
+------------------------------------+-------------------------------------------------------------+
| Concept                            | Core Question Answered                                      |
+------------------------------------+-------------------------------------------------------------+
| Ranking Score (Relative Order)     | "Is Prospect A more likely to convert than Prospect B?"     |
| Calibrated Probability (Absolute)  | "What is Prospect A's true expected chance of subscribing?" |
+------------------------------------+-------------------------------------------------------------+
```

1. **Ranking Invariance Under Monotonic Transformations**:
   - For a fixed capacity of $k=4,000$ calls, any strictly monotonic transformation of scores $g(s)$ yields the exact same top-$k$ prospect cohort. The absolute magnitude of the score does not matter for fixed-capacity dialing order.
2. **Probability Sensitivity in Downstream Decision Systems**:
   - In contrast, downstream economic optimization (e.g., calling if $p \times \text{DepositValue} > \text{CallCost}$) requires accurate absolute probabilities.
   - If a model outputs a score of $0.75$, but only $55\%$ of prospects with that score actually subscribe, an expected-value calculation will substantially overestimate revenue.
3. **Random Forest Score Characteristics**:
   - Standard Random Forest probability estimates are computed as the empirical proportion of positive votes across ensemble decision trees ($p = \frac{1}{B} \sum_{b=1}^B I(T_b(X) = 1)$).
   - Because tree predictions rarely reach complete leaf consensus, RF scores are historically compressed toward the base rate (underconfident near 0 and 1) and may have slight non-linear distortions.

---

## 3. Development-Only Nested Calibration Design

To prevent validation leakage, calibration must **never** be evaluated on the same data used to fit either the base classifier or the calibrator. 

We deployed a **Nested Out-of-Fold Cross-Validation Scheme**:

```
Development Partition (N = 36,168 rows, 80% of total)
  │
  ├── Outer Fold 1: Train (28,934 rows) ──> [Inner 3-Fold CV] ──> Fit Base RF & Fit Calibrator
  │                 Val   (7,234 rows)  <── Evaluated strictly Out-of-Fold
  │
  ├── Outer Fold 2: Train (28,934 rows) ──> [Inner 3-Fold CV] ──> Fit Base RF & Fit Calibrator
  │                 Val   (7,234 rows)  <── Evaluated strictly Out-of-Fold
  │
  ├── Outer Fold 3: Train (28,934 rows) ──> [Inner 3-Fold CV] ──> Fit Base RF & Fit Calibrator
  │                 Val   (7,234 rows)  <── Evaluated strictly Out-of-Fold
  │
  ├── Outer Fold 4: Train (28,935 rows) ──> [Inner 3-Fold CV] ──> Fit Base RF & Fit Calibrator
  │                 Val   (7,233 rows)  <── Evaluated strictly Out-of-Fold
  │
  └── Outer Fold 5: Train (28,935 rows) ──> [Inner 3-Fold CV] ──> Fit Base RF & Fit Calibrator
                    Val   (7,233 rows)  <── Evaluated strictly Out-of-Fold
```

### Protocol & Verification
1. **Outer Resampling**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
2. **Inner Resampling**: `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` within each outer training fold.
3. **Calibrator Architecture**:
   - Used `CalibratedClassifierCV(estimator=pipeline, method=method, cv=inner_cv, ensemble=False)`.
   - Setting `ensemble=False` guarantees that the base model scored on outer validation is the **exact single pipeline** trained on the entire outer training split, while the calibrator is fit on the unbiased inner out-of-fold predictions.
4. **Guarantees**:
   - Every development record received **exactly one** out-of-fold prediction for each method ($\sum N_i = 36{,}168$).
   - Outer validation folds remained strictly untouched during pipeline preprocessing, RF training, and calibrator parameter estimation.
   - Zero NaN values or infinite probabilities were produced.

---

## 4. Probability Quality Metrics Across Calibration Methods

We evaluated three model configurations across the pooled 36,168 out-of-fold development predictions:
1. **Uncalibrated Random Forest (Raw Tree-Vote Proportions)**
2. **Sigmoid / Platt-Style Calibrated Random Forest**
3. **Isotonic-Calibrated Random Forest**

### Summary Comparison Table

| Dimension | Metric | Uncalibrated RF | Sigmoid Calibrated | Isotonic Calibrated | Ideal / Direction |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Probability Quality** | **Brier Score** | **0.08318** | 0.08454 | **0.08273** | Lower is better |
| | **Log Loss** | **0.29041** | 0.29581 | **0.28889** | Lower is better |
| | **Calibration Intercept ($\alpha$)** | +0.2538 | **+0.0212** | **-0.0072** | Target = 0.0 |
| | **Calibration Slope ($\beta$)** | 1.1418 | **1.0103** | **0.9950** | Target = 1.0 |
| **Discrimination** | **ROC-AUC** | **0.7915** | **0.7915** | 0.7902 | Higher is better |
| | **PR-AUC (Average Precision)** | **0.4370** | **0.4373** | 0.4296 | Higher is better |
| **Ranking Guardrail ($k=4,000$)** | **Conversions Captured** | **1,908** | 1,906 (-2) | **1,909** (+1) | Higher is better |
| | **Precision@capacity** | **47.70%** | 47.65% | **47.73%** | Higher is better |
| | **Recall@capacity** | **45.10%** | 45.05% | **45.12%** | Higher is better |
| | **Lift@capacity** | **4.08x** | 4.07x | **4.08x** | Higher is better |
| **Ranking Stability** | **Top-$k$ Overlap with Raw RF** | — | **99.83% (3,993/4,000)** | **96.35% (3,854/4,000)** | Higher is better |
| | **Prospects Changed in Top-$k$**| — | **7 prospects** | **146 prospects** | Lower churn is better |

---

## 5. Uncalibrated Baseline Probability Quality

The raw Random Forest voting probabilities provide a baseline:
- **Brier Score**: **0.08318** (strong baseline due to solid ranking discrimination).
- **Log Loss**: **0.29041**.
- **Calibration Slope ($\beta = 1.1418$)**: The slope is greater than 1.0, indicating that raw RF probabilities are **underconfident** (compressed toward the central base rate). Low probabilities are not low enough, and high probabilities are slightly dampened.
- **Calibration Intercept ($\alpha = +0.2538$)**: The positive intercept reflects slight systematic underestimation of overall conversion probability across the distribution.
- **Discrimination & Ranking**: PR-AUC is **0.4370**, capturing **1,908 conversions** at top-4,000 capacity ($47.70\%$ precision, $4.08\times$ lift).

> [!NOTE]
> While raw tree-vote fractions are not formal probabilities, the uncalibrated Random Forest exhibits surprisingly good empirical alignment with observed conversion rates in lower-to-middle probability bins.

---

## 6. Sigmoid (Platt) Calibration Results

Sigmoid calibration fits a two-parameter logistic transformation to the raw scores: $P(Y=1 \mid s) = \frac{1}{1 + \exp(A \cdot s + B)}$.

### Empirical Performance
- **Slope & Intercept**: Sigmoid calibration successfully straightens the global logistic log-odds relationship:
  - Slope improves from $1.1418 \rightarrow \mathbf{1.0103}$ (very close to 1.0).
  - Intercept improves from $+0.2538 \rightarrow \mathbf{+0.0212}$ (very close to 0.0).
- **The Degradation in Brier Score & Log Loss**:
  - Despite improving the global slope and intercept, **Sigmoid calibration actually worsens both Brier score ($0.08454$ vs $0.08318$) and Log Loss ($0.29581$ vs $0.29041$)**.
- **Root Cause of Degradation**:
  - The true relationship between RF score and empirical conversion is not strictly sigmoidal.
  - The rigid parametric logistic curve overcorrects in the upper tail ($0.60 \le p \le 0.80$). In this tier, Sigmoid predicts an average probability of $75.4\%$, while the empirical conversion rate is only $57.8\%$ (an absolute calibration error gap of **$17.63\%$**).
  - Heavy penalties incurred by Log Loss on these overconfident upper-tail errors drive the overall loss degradation.

---

## 7. Isotonic Calibration Results

Isotonic regression fits a free-form, non-parametric piecewise constant non-decreasing step function: $\min \sum (y_i - m(s_i))^2 \quad \text{subject to } m(s_i) \le m(s_j) \text{ for } s_i \le s_j$.

### Empirical Performance
- **Probability Losses**:
  - Produces the lowest Brier score (**0.08273**, a marginal $-0.00045$ improvement over uncalibrated).
  - Produces the lowest Log Loss (**0.28889**, a marginal $-0.00152$ improvement over uncalibrated).
  - Slope ($\beta = 0.9950$) and Intercept ($\alpha = -0.0072$) are nearly ideal.
- **The Downside: Step Plateaus and Loss of Rank Resolution**:
  - Because isotonic regression produces piecewise constant step functions, many distinct continuous raw scores are mapped to identical discrete probability values (flat plateaus).
  - This step-coarsening **degrades PR-AUC from $0.4370 \rightarrow 0.4296$** (a drop of $-0.0074$).
  - ROC-AUC drops slightly from $0.7915 \rightarrow 0.7902$.
  - It alters the ranking of **146 prospects** at the top-4,000 cutoff due to arbitrary tie-breaking across step plateaus.

---

## 8. Reliability Analysis & Calibration Curve Breakdown

![Calibration Reliability Curves](figures/10_calibration_reliability_curves.png)
*Figure 1: Top: Out-of-fold reliability diagram for Uncalibrated RF, Sigmoid, and Isotonic calibration evaluated across 10 decile bins against the perfect calibration line ($y=x$). Bottom: Predicted probability density distribution.*

![Calibrated Probability Distributions](figures/11_calibrated_probability_distributions.png)
*Figure 2: Distribution of out-of-fold raw scores vs. Sigmoid and Isotonic calibrated probabilities across the 36,168 development prospects.*

![Absolute Calibration Gaps](figures/12_calibration_binned_gaps.png)
*Figure 3: Absolute calibration gap ($|\text{Mean Pred} - \text{Observed Rate}|$) across probability deciles for all three methods (lower bar is better).*

### Detailed Reliability Decile Tables ($N_{\text{dev}} = 36,168$)

#### Table 1: Uncalibrated Random Forest (Raw Scores)
| Bin | Score Range | Count ($N$) | % of Pop | Positives | Mean Pred Score | Observed Rate | Absolute Gap | Small-Sample Flag ($N < 30$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | $[0.00, 0.10]$ | 24,896 | 68.83% | 1,256 | 0.0606 | 0.0505 | 0.0102 | False |
| 2 | $[0.10, 0.20]$ | 6,618 | 18.30% | 876 | 0.1306 | 0.1324 | **0.0017** | False |
| 3 | $[0.20, 0.30]$ | 1,780 | 4.92% | 588 | 0.2507 | 0.3303 | 0.0796 | False |
| 4 | $[0.30, 0.40]$ | 1,239 | 3.43% | 500 | 0.3433 | 0.4036 | 0.0602 | False |
| 5 | $[0.40, 0.50]$ | 494 | 1.37% | 245 | 0.4412 | 0.4960 | 0.0548 | False |
| 6 | $[0.50, 0.60]$ | 318 | 0.88% | 173 | 0.5489 | 0.5440 | **0.0049** | False |
| 7 | $[0.60, 0.70]$ | 385 | 1.06% | 266 | 0.6531 | 0.6909 | 0.0378 | False |
| 8 | $[0.70, 0.80]$ | 373 | 1.03% | 275 | 0.7417 | 0.7373 | **0.0044** | False |
| 9 | $[0.80, 0.90]$ | 65 | 0.18% | 52 | 0.8230 | 0.8000 | 0.0230 | False |
| 10 | $[0.90, 1.00]$ | 0 | 0.00% | 0 | — | — | — | True ($N=0$) |

#### Table 2: Sigmoid Calibrated Random Forest
| Bin | Probability Range | Count ($N$) | % of Pop | Positives | Mean Pred Prob | Observed Rate | Absolute Gap | Small-Sample Flag ($N < 30$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | $[0.00, 0.10]$ | 28,749 | 79.49% | 1,632 | 0.0687 | 0.0568 | 0.0119 | False |
| 2 | $[0.10, 0.20]$ | 3,566 | 9.86% | 738 | 0.1315 | 0.2070 | 0.0754 | False |
| 3 | $[0.20, 0.30]$ | 1,386 | 3.83% | 510 | 0.2461 | 0.3680 | 0.1219 | False |
| 4 | $[0.30, 0.40]$ | 749 | 2.07% | 303 | 0.3437 | 0.4045 | 0.0608 | False |
| 5 | $[0.40, 0.50]$ | 399 | 1.10% | 194 | 0.4465 | 0.4862 | 0.0397 | False |
| 6 | $[0.50, 0.60]$ | 208 | 0.58% | 104 | 0.5501 | 0.5000 | 0.0501 | False |
| 7 | $[0.60, 0.70]$ | 196 | 0.54% | 106 | 0.6491 | 0.5408 | 0.1083 | False |
| 8 | $[0.70, 0.80]$ | 263 | 0.73% | 152 | 0.7543 | 0.5779 | **0.1763** | False |
| 9 | $[0.80, 0.90]$ | 512 | 1.42% | 382 | 0.8544 | 0.7461 | 0.1083 | False |
| 10 | $[0.90, 1.00]$ | 140 | 0.39% | 110 | 0.9176 | 0.7857 | 0.1319 | False |

#### Table 3: Isotonic Calibrated Random Forest
| Bin | Probability Range | Count ($N$) | % of Pop | Positives | Mean Pred Prob | Observed Rate | Absolute Gap | Small-Sample Flag ($N < 30$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | $[0.00, 0.10]$ | 27,202 | 75.21% | 1,480 | 0.0542 | 0.0544 | **0.0002** | False |
| 2 | $[0.10, 0.20]$ | 3,584 | 9.91% | 491 | 0.1350 | 0.1370 | **0.0020** | False |
| 3 | $[0.20, 0.30]$ | 1,045 | 2.89% | 257 | 0.2482 | 0.2459 | **0.0022** | False |
| 4 | $[0.30, 0.40]$ | 1,898 | 5.25% | 657 | 0.3504 | 0.3462 | **0.0043** | False |
| 5 | $[0.40, 0.50]$ | 1,263 | 3.49% | 567 | 0.4350 | 0.4489 | **0.0140** | False |
| 6 | $[0.50, 0.60]$ | 257 | 0.71% | 130 | 0.5435 | 0.5058 | 0.0376 | False |
| 7 | $[0.60, 0.70]$ | 362 | 1.00% | 229 | 0.6569 | 0.6326 | 0.0243 | False |
| 8 | $[0.70, 0.80]$ | 512 | 1.42% | 389 | 0.7419 | 0.7598 | 0.0179 | False |
| 9 | $[0.80, 0.90]$ | 39 | 0.11% | 26 | 0.8390 | 0.6667 | **0.1724** | False |
| 10 | $[0.90, 1.00]$ | 6 | 0.02% | 5 | 0.9032 | 0.8333 | 0.0699 | True ($N=6 < 30$) |

---

## 9. Ranking-Preservation & Guardrail Analysis

A primary requirement of this phase is that calibration **must not harm the existing business-ranking objective**:
- Outreach quota: $k_{\text{oof}} = 4{,}000$ leads.
- Baseline yield: **1,908 conversions** ($47.70\%$ precision, $4.08\times$ lift).

### Empirical Ranking Guardrail Verification

```
+---------------------------------------------------------------------------------------------------------+
|                                    TOP-k RANKING PRESERVATION SUMMARY                                    |
+----------------------+--------------------+--------------------+--------------------+-------------------+
| Model Configuration  | Top-k Overlap with | Prospects Changed  | Conversions at     | Conversion Delta  |
|                      | Uncalibrated RF    | in Top 4,000 (k)   | Capacity (k=4,000) | vs. Raw Baseline  |
+----------------------+--------------------+--------------------+--------------------+-------------------+
| Uncalibrated RF      | 100.0% (4,000)     | 0 (baseline)       | 1,908 (47.70%)     | Baseline          |
| Sigmoid Calibrated   |  99.83% (3,993)    | 7 prospects        | 1,906 (47.65%)     | -2 conversions    |
| Isotonic Calibrated  |  96.35% (3,854)    | 146 prospects      | 1,909 (47.73%)     | +1 conversion     |
+----------------------+--------------------+--------------------+--------------------+-------------------+
```

### Analysis of Ranking Shifts
1. **Sigmoid Calibration Ranking Invariance**:
   - Because a sigmoid function is strictly monotonic within each fold, intra-fold ranking is 100% preserved.
   - When pooling predictions across 5 outer folds, slight variations in fold calibrator parameters cause a minor boundary shift of **only 7 prospects (0.175%)** across the 4,000 cutoff.
   - The conversion impact is negligible: $-2$ conversions ($1,906$ vs $1,908$).
2. **Isotonic Step Plateaus and Churn**:
   - Because isotonic regression creates constant step functions, ties are introduced among prospects who previously had distinct continuous scores.
   - Across the pooled folds, **146 prospects (3.65% of the outreach quota)** are swapped across the capacity boundary.
   - While net conversions happen to be $+1$ on this development sample ($1,909$ vs $1,908$), PR-AUC drops from $0.4370 \rightarrow 0.4296$, demonstrating that isotonic regression degrades overall ranking discrimination across the full list.

---

## 10. Final Calibration Decision & Trade-Off Synthesis

### Decision Hierarchy Evaluation
1. **Primary: Probability Calibration Quality**:
   - **Sigmoid Calibration**: Fails to improve probability quality. Worsens Brier score ($0.08454$ vs $0.08318$) and Log Loss ($0.29581$ vs $0.29041$). Its rigid parametric shape creates large overconfidence errors ($17.6\%$ gap) in the upper deciles.
   - **Isotonic Calibration**: Yields tiny numerical gains in Brier score ($-0.00045$) and Log Loss ($-0.00152$), but degrades PR-AUC by $-0.0074$ and introduces step plateaus.
2. **Guardrail: Ranking Preservation**:
   - Sigmoid maintains $99.83\%$ ranking fidelity (7 prospects changed).
   - Isotonic causes $3.65\%$ ranking churn (146 prospects changed) due to plateau ties.
3. **Secondary: Robustness, Simplicity, and Overfitting Risk**:
   - Isotonic regression is non-parametric and highly flexible, making it vulnerable to overfitting on small or changing distributions.
   - The uncalibrated Random Forest has smooth continuous scores, zero tie-breaking artifacts, the highest PR-AUC ($0.4370$), and an inherently well-behaved empirical calibration curve (Brier score $0.08318$).

### Formal Strategic Recommendation

> [!IMPORTANT]
> **Decision: Maintain Raw RF Scores for Outreach Ranking; Avoid Deploying Forced Calibration**.
> 
> 1. **For Capacity-Constrained Lead Ranking (Current Operational Mode)**:
>    - Retain the **Uncalibrated Random Forest scores**.
>    - It maximizes ranking discrimination (PR-AUC: **0.4370**), avoids tie-breaking artifacts, and preserves exact prospect prioritization ($1,908$ conversions at $k=4,000$).
>    - Calibration is not needed to rank leads under fixed capacity and introduces unnecessary complexity or ranking distortion.
> 2. **If Downstream Probability Estimation Is Mandated**:
>    - **Do NOT deploy Sigmoid calibration**: It actively degrades Brier score and Log Loss and suffers from severe overconfidence in the $0.60–0.80$ range.
>    - If absolute probabilities are strictly required for external financial systems, **Isotonic calibration** is the only method that improves Brier and Log Loss, but it must be deployed with explicit awareness of its step-plateau behavior and potential PR-AUC loss.
>    - However, the cleanest and most robust engineering conclusion is that **the uncalibrated Random Forest should remain a ranking score rather than being artificially presented as calibrated probabilities**.

---

## 11. What Calibrated Probabilities Would Enable

While the immediate project phase operates under fixed call capacity ($k=5,000$), understanding calibrated probabilities is foundational for future operations:

1. **Expected-Value Sales Dialing ($E[\text{Profit}]$)**:
   - If call costs ($C_{\text{call}}$) and expected deposit revenues ($V_{\text{deposit}}$) are known, a calibrated probability $p_i$ allows dialing whenever:
     $$E[\text{Profit}_i] = p_i \cdot V_{\text{deposit}} - C_{\text{call}} > 0 \iff p_i > \frac{C_{\text{call}}}{V_{\text{deposit}}}$$
   - This frees the organization from an arbitrary fixed quota ($k=5,000$) and enables profit-maximizing dynamic campaign sizing.
2. **Variable Calling Costs & Channel Routing**:
   - Higher-cost outreach channels (e.g. senior branch advisors or in-person meetings) can be reserved for prospects with high calibrated probabilities ($p_i \ge 0.50$), while lower-probability prospects ($p_i \in [0.15, 0.30]$) can be routed to automated SMS, email, or lower-cost call center tiers.
3. **Capacity Planning Across Changing Headcounts**:
   - If call center staffing changes (e.g., from 5,000 to 2,500 or 8,000 calls), calibrated probabilities allow management to forecast exact subscription volumes and conversion yields before dialing begins.

---

## 12. Limitations & Methodological Guardrails

1. **Development-Only Scope**: All results, reliability curves, and tables were generated strictly on the 80% development partition. The untouched 20% holdout test set was **not** revisited.
2. **Frozen Model Candidate Lacks Untouched Test Estimate**: As established in Phase 3, the final unweighted Random Forest configuration (`class_weight=None`) was selected based on 15 repeated-validation development folds and OOF diagnostics; it does not currently have an independent holdout test estimate (the previously recorded test benchmark was on the balanced configuration).
3. **Prevalence and Seasonality Drift**: Calibration curves depend directly on the baseline prevalence ($11.7\%$) and campaign calendar mix. If future campaigns operate in different months (e.g., exclusively May vs. exclusively October) or target cold lists without past CRM interactions, the probability mapping will drift and require recalibration.
4. **No Economic Assumptions Applied**: This analysis deliberately avoids inventing artificial economic values for deposits or call costs; all guardrail evaluations were conducted on pure conversion counts and statistical calibration losses.
