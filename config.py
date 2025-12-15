"""
Central configuration for the Book of Mormon Textual Parallel Detection project.
Loads settings from YAML files in config/ directory.
"""

import os
from pathlib import Path
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
TEXTS_DIR = PROJECT_ROOT / "texts"
TEXTS_RAW_DIR = TEXTS_DIR / "raw"
TEXTS_STRUCTURED_DIR = TEXTS_DIR / "structured"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

# Database
DATABASE_PATH = DATA_DIR / "matches.db"

# Ensure directories exist
for dir_path in [CONFIG_DIR, DATA_DIR, TEXTS_RAW_DIR, TEXTS_STRUCTURED_DIR,
                 EMBEDDINGS_DIR, RESULTS_DIR, CALIBRATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def load_yaml(filename: str) -> dict:
    """Load a YAML config file from the config directory."""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, 'r') as f:
        return yaml.safe_load(f) or {}


def get_sources_config() -> dict:
    """Load source texts configuration."""
    return load_yaml("sources.yaml")


def get_models_config() -> dict:
    """Load model configuration."""
    return load_yaml("models.yaml")


def get_thresholds_config() -> dict:
    """Load threshold configuration."""
    return load_yaml("thresholds.yaml")


# API Keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# Default settings (can be overridden by YAML)
DEFAULTS = {
    "embedding_model": "text-embedding-3-large",
    "embedding_dimensions": 3072,
    "llm_model_tier1": "gpt-4o-mini",  # Fast/cheap for initial filtering
    "llm_model_tier2": "gpt-4o",       # Accurate for deep analysis
    "batch_size": 100,
    "top_k_candidates": 10,
    "min_score_threshold": 0.5,
}


def get_setting(key: str, default=None):
    """Get a setting, checking YAML configs first, then defaults."""
    # Check models.yaml
    models = get_models_config()
    if key in models:
        return models[key]

    # Check thresholds.yaml
    thresholds = get_thresholds_config()
    if key in thresholds:
        return thresholds[key]

    # Fall back to defaults
    return DEFAULTS.get(key, default)
