"""Orchestrate probability calibration assessment and figure generation."""
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load_data import load_bank_marketing_data
from src.features.build_features import split_data
from src.features.feature_contract import select_canonical_pre_campaign_features
from src.models.calibration import (
    generate_oof_calibrated_predictions,
    compare_calibration_methods,
    generate_calibration_figures,
)

def run_calibration():
    print("Loading data for calibration evaluation...")
    X, y = load_bank_marketing_data()
    X_dev, _, y_dev, _ = split_data(X, y, test_size=0.2, random_state=42)
    X_clean = select_canonical_pre_campaign_features(X_dev)

    print("Generating out-of-fold calibrated predictions (5-fold CV)...")
    df_oof_cal = generate_oof_calibrated_predictions(X_clean, y_dev, n_outer_splits=5, n_inner_splits=3, random_state=42)

    print("Comparing calibration methods (Raw RF, Sigmoid, Isotonic)...")
    comparison_res = compare_calibration_methods(df_oof_cal, k=4000)

    print("Generating calibration figures (10-12)...")
    fig_paths = generate_calibration_figures(
        df_oof_cal=df_oof_cal,
        output_dir=project_root / "reports" / "figures",
    )
    for name, path in fig_paths.items():
        print(f"Generated {name}: {path}")

    print("Calibration evaluation complete.")
    return comparison_res

if __name__ == "__main__":
    run_calibration()
