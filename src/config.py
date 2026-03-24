"""
Paths and LLM defaults. Override via environment variables (or a repo-root `.env` file).

Load this module before reading paths so `load_dotenv` runs first.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


def _get(key: str, default: str) -> str:
    v = os.getenv(key)
    if v is None or not str(v).strip():
        return default
    return str(v).strip()


# Data files (paths relative to cwd when you run scripts, unless you pass absolute paths)
INPUT_DATA_PATH = _get("SHUNT_INPUT_DATA_PATH")
EVAL_DATA_PATH = _get("SHUNT_EVAL_DATA_PATH")
MERGED_DATA_PATH = _get("SHUNT_MERGED_DATA_PATH")
RESULTS_DATA_PATH = _get("SHUNT_RESULTS_DATA_PATH")

# LLM (used by llm_settings_from_config)
provider = _get("SHUNT_LLM_PROVIDER")
model_id = _get("SHUNT_MODEL_ID")