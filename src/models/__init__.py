"""Model training and evaluation package."""
__all__ = ["train_baseline_models", "evaluate_model"]

def __getattr__(name):
    if name == "train_baseline_models":
        from .train import train_baseline_models
        return train_baseline_models
    elif name == "evaluate_model":
        from .evaluate import evaluate_model
        return evaluate_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
