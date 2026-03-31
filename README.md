# Shunt Registry Data Extraction using LLMs

Functionality for extracting data from medical text to populate the UK Shunt Registry (UKSR) database using Large Language Models (LLMs).

## Usage

**Set up:** clone the repository, create a Python virtual environment, install dependencies, and run all commands from the **repository root** so paths like `data/...` work as intended.

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Workflow:**

1. **Source notes:** Export patient notes from the CogStack Catalogue as a CSV with **one row per note** and a column for the **medical record number (`MRN`)**. Save the file under `data/`.

2. **Reference answers (evaluation):** Provide a spreadsheet of known registry field values (the “gold standard”) keyed by `MRN`, aligned with the same patients as in the notes export.

3. **Configuration:** Copy `.env.example` to `.env`. Set file paths if yours differ, and choose which LLM to use (`LLM_PROVIDER`, `MODEL_ID`). Defaults are defined in `src/config.py`.
    - **OpenAI:** Set `LLM_PROVIDER=openai`, `MODEL_ID` to a chat model you can access, and add API key to `OPENAI_API_KEY`.
    - **Ollama:** Install [Ollama](https://ollama.com/download), keep the app running, then `ollama pull <model>` (for example `tinyllama`). Set `LLM_PROVIDER=ollama` and `MODEL_ID` equal to the name of the model you downloaded.
    - **Hugging Face:** Set `LLM_PROVIDER=hf` and `MODEL_ID` to a Hugging Face model id (for example `Qwen/Qwen3-0.6B`). The model downloads when first used; in an air-gapped environment you should download weights manually (for example on UCLH TRE).

4. **Merge notes and reference data:** Run the script below once. It turns long-format notes into **one row per patient (`MRN`)** with separate columns for note types (e.g. discharge summary, operation note), merges in your reference spreadsheet on `MRN`, and writes a combined CSV.

```bash
python src/process_data.py
```

5. **Run extractions:** Use `question_runner.py` to ask the model one or more registry questions per patient:

```bash
# Single question
python src/question_runner.py q1

# Several questions
python src/question_runner.py q1 q4 q8

# Every registered question
python src/question_runner.py all

# Cap how many MRNs are processed (0 = no limit)
python src/question_runner.py q1 --max-mrns 50
```

Each run **appends** rows to the results file defined in the .env config.

6. You can then run the below to get a simple performance summary per question:
```bash
python src/evaluate_results.py
```

**Adding a new question:** 

Add the allowed answer list to `registry_options.py`, add a new prompt file under `prompts/`, and register one entry in `QUESTION_REGISTRY` in `questions.py`. Other Python modules should not need changes.

## Architecture

**Settings:** values from `.env` (and defaults) are loaded into `LLMSettings`; `create_llm_client()` builds an `LLMClient` for the selected provider.

**Per patient and question:** `run_question(...)` calls `extract_with_llm`, which runs `generate_chat` on the LLM and, when the reply is JSON with an `answer` field, extracts that string so the stored prediction can be compared to the gold label.

**Repository layout:**

```
.
├── prompts/                 # Text templates; each question points at one file by name
└── src/
    ├── config.py            # Environment-based paths and LLM defaults
    ├── process_data.py      # Script to build merged dataset (notes + gold standard columns)
    ├── questions.py         # Registry of questions (options, prompts, model options)
    ├── question_runner.py   # Main entrypoint
    ├── llm_client.py        # Standard interface with OpenAI, Ollama, and HF clients
    ├── registry_options.py  # Shunt Registry option lists used for prompts and schemas
    └── utils.py             # Utils inc. prompt loading, JSON normalization, metrics etc
```

## Note on running Ollama models in TRE

First download Ollama to your local device.

Then pull the model you are interested in e.g.
```bash
ollama pull gpt-oss:120b
```

The model will be saved to one of the following locations:
```bash
macOS: ~/.ollama/models
Linux: /usr/share/ollama/.ollama/models
Windows: C:/Users/%username%/.ollama/models
```

Upload the .ollama folder into the TRE Airlock and then download into your VM. Then run the commands below (note: these assume a username of 3Thd and that you downloaded into your Desktop)

```bash
sudo cp -a /home/3Thd/Desktop/.ollama/models/. /usr/share/ollama/.ollama/models/
sudo chown -R ollama:ollama /usr/share/ollama/.ollama
sudo systemctl restart ollama
```