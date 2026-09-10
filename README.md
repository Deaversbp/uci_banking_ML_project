# UCI Banking Machine Learning Project

An end-to-end Machine Learning project designed to solve an outbound telemarketing optimization problem using the **UCI Bank Marketing dataset** (fetched via `ucimlrepo`).

The project frames a realistic pre-call decision problem: **the sales team has operational capacity to call only 5,000 customers from the eligible pool**. The objective combines:
1. **Supervised Lead Prioritization**: Predict and rank-order leads by conversion probability to maximize subscriptions within the 5,000-call quota.
2. **Unsupervised Prospect Segmentation**: Identify natural customer segments to tailor outreach messaging and understand audience composition.

---

## 🎯 Project Goals & Foundational Standards

- **Strict Pre-Call Leakage Prevention**:
  - The dataset contains `duration` (call length in seconds), which alone yields an artificial ROC-AUC of 0.808.
  - Because duration is unknown before dialing, it is programmatically quarantined and excluded from all feature engineering, pre-call models, and prospect clustering.
- **Two-Tier Baseline Benchmarking (5,000-Call Constraint)**:
  - **Random Selection Benchmark**: With an 11.70% base prevalence, random dialing yields **~585 conversions**.
  - **Business-Rule Heuristic Benchmark**: A non-ML domain heuristic prioritizing prior campaign success (`poutcome == 'success'`), prior contacts without debt, and liquid balances achieves **~1,664 conversions (33.28% precision, 2.84x lift)**.
  - All supervised ML models must outperform **both** benchmarks to prove commercial value.
- **Capacity-Constrained Evaluation**:
  - Models are ranked and selected using **PR-AUC (Average Precision)**, **ROC-AUC**, **Precision@k**, and **Lift@k** rather than arbitrary thresholded F1 scores.
- **Phase 2 Pre-Call Feature Engineering**:
  - `was_previously_contacted`: Binary indicator for previous campaign contact (`pdays != -1`).
  - `pdays_recency`: Non-negative transformed recency handling `-1` explicitly via $\log1p(\max(pdays, 0))$.
  - `prior_success`: Binary indicator for prior campaign success (`poutcome == 'success'`).
  - `has_debt_burden`: Combined indicator for housing and personal loan obligations.
  - `negative_balance_flag` and signed $\log1p$ balance representation.
  - Preserved raw `campaign` contact counts.
- **Informative Missing-State Preservation**:
  - Missingness in `poutcome` (81.7%) and `contact` (28.8%) is preserved as an explicit category (`unknown`), preventing naive imputation to `'failure'`.

---

## 📂 Project Structure

```text
uci_banking_ML_project/
├── .gitignore                    # Ignores .venv, datasets, models, checkpoints
├── README.md                     # Project documentation and guide
├── requirements.txt              # Python dependencies with minimum version constraints
├── config/
│   └── config.yaml               # Dataset IDs, split ratios, model seeds
├── data/
│   ├── raw/                      # Cached raw data (CSV)
│   └── processed/                # Preprocessed datasets
├── models/                       # Saved trained model pipelines (.joblib)
├── notebooks/
│   └── 01_data_exploration.ipynb # Pre-computed EDA & audit diagnostics notebook
├── reports/
│   ├── data_audit_report.md      # Comprehensive Phase 1 Forensic Data Audit Report
│   └── figures/                  # Generated diagnostic figures (PNG)
├── scripts/
│   └── build_notebook.py         # Authoritative generator for 01_data_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py           # Explicit package exports
│   │   ├── audit_dataset.py      # Automated forensic audit & figure generation
│   │   └── load_data.py          # Fetches & caches UCI dataset (ID 222)
│   ├── features/
│   │   ├── __init__.py           # Explicit package exports
│   │   └── build_features.py     # PreCallFeatureEngineer & leak-free preprocessing
│   ├── models/
│   │   ├── __init__.py           # Explicit package exports
│   │   ├── baseline.py           # Random & Business-Rule Heuristic benchmarks
│   │   ├── evaluate.py           # Ranking metrics (PR-AUC, Lift@k, Precision@k)
│   │   └── train.py              # End-to-end model training & benchmark comparison
│   └── utils/
│       ├── __init__.py
│       └── logger.py             # Logging and YAML config loader
└── tests/
    ├── __init__.py
    ├── test_audit.py             # Unit tests for audit calculations & exception checks
    └── test_data.py              # Pipeline, leakage safety, features, and ranking tests
```

---

## ⚙️ Environment Setup

This project uses **Python 3.11.9** on Windows.

### 1. Activate the Virtual Environment

In Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

If you encounter execution policy restrictions:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

### 2. Dependencies

To install or update dependencies:
```powershell
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

### Step 1: Fetch and Cache the Dataset
Download the Bank Marketing dataset directly from the UCI ML Repository (ID 222):
```powershell
python -m src.data.load_data
```

### Step 2: Run the Forensic Data Audit
Execute the automated audit suite to recompute statistical diagnostics and update visual figures:
```powershell
python -m src.data.audit_dataset
```
Audit figures are saved to `reports/figures/`, and findings are documented in [`reports/data_audit_report.md`](reports/data_audit_report.md).

### Step 3: Run the Test Suite
Execute unit and regression tests covering leakage exclusion, feature engineering, missingness preservation, and ranking metrics:
```powershell
pytest -v
```

### Step 4: Train Pre-Call Models & Compare Against Baselines
Run the training pipeline to evaluate Logistic Regression and Random Forest against both the Random Baseline and Business-Rule Baseline:
```powershell
python -m src.models.train
```
The best pipeline selected by **PR-AUC** is serialized to `models/best_*_pipeline.joblib`.

### Step 5: Explore the Diagnostic Notebook
Open `notebooks/01_data_exploration.ipynb`:
```powershell
jupyter notebook
```
*(Note: To modify notebook content, edit the authoritative generator `scripts/build_notebook.py` and run `python scripts/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb`)*.
