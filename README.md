# Shunt Registry Data Extraction using LLMs

This repository provides functionality for extracting data from text to populate the national shunt registry database, using Large Language Models (LLMs)

## Usage

1. Collect dataset on patients via CogStack Catalogue (TODO: fill in exactly what to select and expected format)
2. Collect some gold standard data to evaluate against if available (TODO: fill in where your gold standard came from)
3. Add file paths to config.py, including:
- INPUT_DATA_PATH = this is the file path for the dataset from CogStack catalogue
- EVAL_DATA_PATH = this is the file path for your golden dataset
4. Run process_data.py - this will pre-process the data, and create several new files (TODO: explain which)
5. After this you can now run any of the question scripts e.g. q1.py

## Architecture

config / env  →  LLMSettings  →  create_llm_client()  →  llm: LLMClient
                                                              ↓
data + extract_q1(..., llm)  →  llm.generate_chat(...)

**Step 1:** `src/llm_client.py` — `LLMSettings`, `OpenAIClient`, `HuggingFaceClient`, `create_llm_client(settings)`, `llm_settings_from_config()`, `create_llm_client_from_config()`.

**Step 2:** `utils.extract_with_llm(..., llm)` and `append_results_to_csv(..., llm)` take an `LLMClient`. `load_llm()` was removed. `q1.py` builds the client once (`create_llm_client_from_config()`) and passes it to `extract_q1(data_merged, llm)`.