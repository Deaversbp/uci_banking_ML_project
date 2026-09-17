"""Orchestrate repeated cross-validation supervised model comparison."""
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.compare import compare_supervised_candidates

if __name__ == "__main__":
    print("Running repeated cross-validation supervised model comparison on development partition...")
    compare_supervised_candidates()
    print("Model comparison complete.")
