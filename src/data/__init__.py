"""Data loading and audit modules for UCI Banking Dataset."""
from .load_data import load_bank_marketing_data, fetch_and_save_data

__all__ = [
    "load_bank_marketing_data",
    "fetch_and_save_data",
    "run_full_data_audit",
]

def __getattr__(name):
    if name == "run_full_data_audit":
        from . import audit_dataset
        return getattr(audit_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
