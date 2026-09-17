"""Orchestrate forensic data audit and figure generation."""
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.audit_dataset import run_full_data_audit

if __name__ == "__main__":
    print("Running forensic data audit...")
    results = run_full_data_audit()
    print("Data audit complete. Figures saved to reports/figures/.")
