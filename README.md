# UCI Banking Machine Learning Project

An end-to-end Machine Learning project designed to learn and practice foundational to advanced ML concepts using the **UCI Bank Marketing dataset** (fetched via `ucimlrepo`).

---

## 🎯 Project Goals

- **Learn ML Pipeline Design**: Clean modular code structure separating data ingestion, feature engineering, model training, and evaluation.
- **Handling Real-World Data Challenges**:
  - Class imbalance (~11.7% subscription rate).
  - Mixed data types (numerical and categorical features).
  - Evaluation beyond simple accuracy (Precision, Recall, F1-score, ROC-AUC).
- **Experimentation & Reproducibility**: Using configuration files (`config/config.yaml`), local caching of raw data, and automated testing.

---

## 📂 Project Structure

```text
uci_banking_ML_project/
├── .gitignore                    # Ignores .venv, datasets, models, checkpoints
├── README.md                     # Project documentation and guide
├── requirements.txt              # Pinned Python dependencies
├── config/
│   └── config.yaml               # Dataset IDs, split ratios, model seeds
├── data/
│   ├── raw/                      # Cached raw data (CSV)
│   └── processed/                # Preprocessed datasets
├── models/                       # Saved trained models (.joblib)
├── notebooks/
│   └── 01_data_exploration.ipynb # Initial EDA and data visualization
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py          # Fetches & caches UCI dataset (ID 222)
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py     # Preprocessing pipeline (One-Hot, Scaler)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluate.py           # Evaluation metrics & confusion matrix
│   │   └── train.py              # Baseline model training & serialization
│   └── utils/
│       ├── __init__.py
│       └── logger.py             # Logging and YAML config loader
└── tests/
    ├── __init__.py
    └── test_data.py              # Sanity check tests for pipeline
```

---

## ⚙️ Environment Setup

This project uses **Python 3.11.9** for optimal scikit-learn wheel compatibility on Windows.

### 1. Activate the Virtual Environment

In Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

If you encounter execution policy restrictions in PowerShell:
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
The raw data is cached to `data/raw/bank_features.csv` and `data/raw/bank_targets.csv` so you can work offline.

### Step 2: Run Unit Tests
Ensure everything is configured and operational:
```powershell
pytest
```

### Step 3: Train Baseline Models
Run the end-to-end training pipeline (trains Logistic Regression and Random Forest with balanced class weights):
```powershell
python -m src.models.train
```
The best-performing model pipeline is automatically saved to `models/best_*_pipeline.joblib`.

### Step 4: Explore the Notebook
Start Jupyter and open `notebooks/01_data_exploration.ipynb`:
```powershell
jupyter notebook
```

---

## 🔗 GitHub Integration & Remote Setup

Git is initialized and configured with your Git Credential Manager for user **`Deaversbp`**.

### To push this project to a new GitHub repository:

1. Create a new empty repository on [GitHub](https://github.com/new) named `uci_banking_ML_project`.
2. Link your local repository to the remote:
   ```powershell
   git remote add origin https://github.com/Deaversbp/uci_banking_ML_project.git
   git branch -M main
   ```
3. Commit and push your code:
   ```powershell
   git add .
   git commit -m "Initial commit: UCI Banking ML project setup and pipeline"
   git push -u origin main
   ```
*Git Credential Manager will automatically use your existing authenticated GitHub session.*
