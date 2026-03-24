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