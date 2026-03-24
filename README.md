# Shunt Registry Data Extraction using LLMs

This repository provides functionality for extracting data from text to populate the UK Shunt Registry (UKSR) database using Large Language Models (LLMs). 

## Usage

Clone the repo, create a virtualenv, install dependencies, then work from the repository root (so paths like `data/...` resolve correctly):

Example:
```bash
python -m venv venv
source/venv/bin/activate
pip install -r requirements.txt
```

Steps:
1. **Source data:** Export patient notes from CogStack Catalogue: one row per note with and a 'MRN' patient identifier. Place the CSV under `data/` (or set `INPUT_DATA_PATH` in `.env`).

2. **Gold / evaluation:** Provide a spreadsheet of registry fields to compare against, with `MRN` aligned to the notes export.

3. **Config:** Copy `.env.example` to `.env`. Set `OPENAI_API_KEY` if using OpenAI. Adjust paths and `LLM_PROVIDER` / `MODEL_ID` as needed, or rely on defaults in `src/config.py`.

4. **Pre-process:** Run `python src/process_data.py`. This pivots long-format notes to a **wide** table (one row per MRN, columns such as `Discharge Summary`, `Op Note`, `Clerking`), merges with the evaluation file on `MRN`, and writes the merged CSV (default `data/merged_data.csv` per `MERGED_DATA_PATH`).

5. **Extract:** Run one or more questions via `question_runner.py`:

```bash
# Run a single question
python src/question_runner.py q1

# Run multiple questions
python src/question_runner.py q1 q4 q8

# Run all registered questions
python src/question_runner.py all

# Limit the number of MRNs processed (0 = all)
python src/question_runner.py q1 --max-mrns 50
```

Predictions append to the results CSV (`RESULTS_DATA_PATH`, default `all_results.csv`), including provider, model, merged data path, and timestamp.

To **add a new question**: add its options to `registry_options.py`, add a prompt file to `prompts/`, then add one `QuestionSpec` entry to `QUESTION_REGISTRY` in `questions.py`. No other files need to change.

## Architecture

config / env  →  LLMSettings  →  create_llm_client()  →  llm: LLMClient
                                                              ↓
data + `run_question(..., llm, spec)`  →  `extract_with_llm` → `llm.generate_chat(...)`

Main modules:

```
.
├── prompts/                 # templates named in each QuestionSpec (e.g. q1_prompt.txt)
└── src/
    ├── config.py            # loads .env → paths + LLM settings
    ├── process_data.py      # stage 1: pivot + merge → merged CSV
    ├── questions.py         # QUESTION_REGISTRY: which questions exist
    ├── question_runner.py   # stage 2: loop, CLI, run_question
    ├── llm_client.py        # LLMClient implementations
    ├── registry_options.py  # option text fed into prompts
    └── utils.py             # helpers used by question_runner
```