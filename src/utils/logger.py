import logging
import sys
from pathlib import Path
import yaml

def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file relative to project root."""
    root = get_project_root()
    full_path = root / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logger(name: str = "uci_banking_ml", level: int = logging.INFO) -> logging.Logger:
    """Set up and return a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
