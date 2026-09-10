"""Data loading module for UCI Banking Dataset."""
__all__ = ["load_bank_marketing_data", "fetch_and_save_data"]

def __getattr__(name):
    if name in ("load_bank_marketing_data", "fetch_and_save_data"):
        from .load_data import load_bank_marketing_data, fetch_and_save_data
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
