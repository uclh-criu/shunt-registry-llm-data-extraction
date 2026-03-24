# Shunt Registry Data Extraction using LLMs

This repository provides functionality for extracting data from text to populate the national shunt registry database, using Large Language Models (LLMs)

## Usage

1. Collect dataset on patients via CogStack Catalogue (TODO: fill in exactly what to select and expected format)
2. Collect some gold standard data to evaluate against if available (TODO: fill in where your gold standard came from)
3. Add file paths to config.py, including:
- INPUT_DATA_PATH = this is the file path for the dataset from CogStack catalogue
- EVAL_DATA_PATH = this is the file path for your golden dataset
4. Run process_data.py - this will pre-process the data, and create several new files (TODO: explain which)
5. Run one or more questions via `question_runner.py`:

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

To **add a new question**: add its options to `registry_options.py`, add a prompt file to `prompts/`, then add one `QuestionSpec` entry to `QUESTION_REGISTRY` in `questions.py`. No other files need to change.

## Architecture

config / env  →  LLMSettings  →  create_llm_client()  →  llm: LLMClient
                                                              ↓
data + `run_question(..., llm, spec)`  →  `extract_with_llm` → `llm.generate_chat(...)`


#	Issue	Status	Notes
1	utils.py is a grab-bag — split by responsibility	Not done	Still one file with prompt loading, text combination, evaluation, and CSV logging. LLM init/calling moved to llm_client.py (which covers the llm.py split), but evaluation.py, data_utils.py, results.py haven't been broken out.
5	q1.py mixes orchestration with question config — need generic run_question()	Not done	q1.py still has the full loop inline. Adding Q4 still means copying the file. This is the biggest remaining win.
7	split_data() writes intermediate CSVs nothing reads	Not done	Still writes clerking.csv, op_note.csv, etc. Low priority but worth removing or documenting.
8	config.py hardcodes paths and model choice	Not done	Still hardcoded. Low priority for now; env vars or YAML would be better long-term.



**Step 1:** `src/llm_client.py` — `LLMSettings`, `OpenAIClient`, `HuggingFaceClient`, `create_llm_client(settings)`, `llm_settings_from_config()`, `create_llm_client_from_config()`.

**Step 2:** `utils.extract_with_llm(..., llm)` and `append_results_to_csv(..., llm)` take an `LLMClient`. `load_llm()` was removed.

**Step 3:** `src/question_runner.py` defines `QuestionSpec`, `run_question(data_merged, llm, spec)`, and the CLI entry point. `src/questions.py` is the central registry mapping keys (`"q1"`, `"q4"`, …) to `QuestionSpec` instances. Individual `q*.py` files have been removed — all question config lives in `questions.py`.