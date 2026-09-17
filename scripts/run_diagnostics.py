"""Orchestrate supervised model error analysis, interpretation, and diagnostics."""
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.diagnostics import run_full_diagnostics

if __name__ == "__main__":
    print("Running supervised model error analysis and interpretation...")
    results = run_full_diagnostics()
    print("Diagnostics complete. Figures 05-09 saved to reports/figures/.")
