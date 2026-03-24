"""
Paths and LLM defaults. Override via environment variables (or a repo-root `.env` file).

Path keys: INPUT_DATA_PATH, EVAL_DATA_PATH, MERGED_DATA_PATH, RESULTS_DATA_PATH.
LLM keys: LLM_PROVIDER, MODEL_ID.

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
INPUT_DATA_PATH = _get("INPUT_DATA_PATH", "data/shunt_registry_extract_110326.csv")
EVAL_DATA_PATH = _get("EVAL_DATA_PATH", "data/ground_truth.csv")
MERGED_DATA_PATH = _get("MERGED_DATA_PATH", "data/merged_data_test.csv")
RESULTS_DATA_PATH = _get("RESULTS_DATA_PATH", "all_results.csv")

# LLM (used by llm_settings_from_config)
provider = _get("LLM_PROVIDER", "openai")
model_id = _get("MODEL_ID", "gpt-5-mini")