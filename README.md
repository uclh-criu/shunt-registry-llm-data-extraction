# Shunt Registry Data Extraction using LLMs

This repository provides functionality for extracting data from text to populate the national shunt registry database, using Large Language Models (LLMs)

## Usage

1. Collect dataset on patients via CogStack Catalogue (TODO: fill in exactly what to select and expected format)
2. Collect some gold standard data to evaluate against if available (TODO: fill in where your gold standard came from)
3. Configure paths and LLM (optional). Copy `.env.example` to `.env` and set `SHUNT_*` variables, or rely on defaults in `src/config.py`. See `OPENAI_API_KEY` in `.env` for OpenAI.
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