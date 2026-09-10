"""Feature engineering and preprocessing modules."""
__all__ = ["build_preprocessor", "split_data", "identify_feature_types"]

def __getattr__(name):
    if name in ("build_preprocessor", "split_data", "identify_feature_types"):
        from .build_features import build_preprocessor, split_data, identify_feature_types
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
