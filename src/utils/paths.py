"""
Path management module for Driver Safety AI project.
Provides absolute paths to project directories and configuration settings.
"""

import os
from pathlib import Path
import yaml

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Core Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

OUTPUTS_DIR = ROOT_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
SESSIONS_DIR = OUTPUTS_DIR / "sessions"
SCREENSHOTS_DIR = OUTPUTS_DIR / "screenshots"

MODELS_DIR = OUTPUTS_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_driver_safety_net.pt"
SCALER_PATH = MODELS_DIR / "scaler_proposed_12feature.pkl"

CONFIG_DIR = ROOT_DIR / "configs"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

DOCS_DIR = ROOT_DIR / "docs"

def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        METADATA_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        SESSIONS_DIR,
        SCREENSHOTS_DIR,
        DOCS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load configuration YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Ensure directories exist on module import
ensure_directories()
