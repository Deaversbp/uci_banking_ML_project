"""Train baseline models on the Bank Marketing dataset."""
from pathlib import Path
from typing import Dict, Any
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.data.load_data import load_bank_marketing_data
from src.features.build_features import identify_feature_types, build_preprocessor, split_data
from src.models.evaluate import evaluate_model
from src.utils.logger import setup_logger, load_config, get_project_root

logger = setup_logger(__name__)

def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Define candidate baseline classifiers."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
    }

def train_baseline_models(save_best: bool = True) -> Dict[str, Dict[str, Any]]:
    """End-to-end baseline training pipeline.

    Loads data, builds preprocessing pipelines, trains candidates, and evaluates.
    """
    config = load_config()
    random_state = config.get("model", {}).get("random_state", 42)
    pos_label = config.get("dataset", {}).get("positive_class", "yes")

    logger.info("Loading Bank Marketing dataset...")
    X, y = load_bank_marketing_data()

    logger.info(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} columns")
    numeric_cols, categorical_cols = identify_feature_types(X)

    X_train, X_test, y_train, y_test = split_data(X, y, random_state=random_state)

    candidates = get_candidate_models(random_state=random_state)
    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, clf in candidates.items():
        logger.info(f"Training pipeline for {name}...")
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test, pos_label=pos_label)
        results[name] = {"metrics": metrics, "pipeline": pipeline}

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_pipeline = pipeline

    logger.info(f"Training complete! Best candidate: {best_model_name} (F1: {best_f1:.4f})")

    if save_best and best_pipeline is not None:
        root = get_project_root()
        models_dir = root / config.get("paths", {}).get("models_dir", "models")
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = models_dir / f"best_{best_model_name.lower()}_pipeline.joblib"
        joblib.dump(best_pipeline, save_path)
        logger.info(f"Saved best model pipeline to {save_path}")

    return results

if __name__ == "__main__":
    logger.info("Starting baseline training run...")
    train_baseline_models()
