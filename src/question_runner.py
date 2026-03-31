"""
Generic LLM extraction loop + CLI entry point.

Run a single question:
    python question_runner.py q1

Run multiple questions:
    python question_runner.py q1 q4 q8

Run all registered questions:
    python question_runner.py all

Override the MRN limit (default = all MRNs; 0 = all MRNs):
    python question_runner.py q1 --max-mrns 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from llm_client import LLMClient, create_llm_client_from_config
from config import RESULTS_DATA_PATH
from utils import (
    append_results_to_csv,
    combine_medical_texts,
    evaluate_predictions,
    extract_with_llm,
    get_gold_standard,
    print_evaluation_summary,
)


@dataclass(frozen=True)
class QuestionSpec:
    """Configuration for one registry question (LLM + notes + gold column)."""

    question_name: str
    """Display name, e.g. 'Q1 - Primary reason for shunting'."""

    gold_standard_col: str
    """Column in data_merged for gold labels."""

    prompt_file: str
    """Filename under prompts/, e.g. 'q1_prompt.txt'."""

    options: str
    """Option list / registry text passed into the prompt template."""

    prediction_key: str
    """Key for the prediction column in per-MRN result rows, e.g. 'Q1_Primary_Reason_Shunting'."""

    note_sources: Tuple[str, ...] = ("Discharge Summary", "Op Note", "Clerking")
    """Wide-table columns to concatenate as input notes."""

    max_mrns: Optional[int] = None
    """If set, only process the first N unique MRNs (after dropna). None = all."""

    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    """Provider-specific kwargs forwarded to generate_chat on every call.

    Examples:
        OpenAI:  {"response_format": {"type": "json_object"}}
        Ollama:  {"format": MyPydanticModel.model_json_schema(), "options": {"temperature": 0}}
    """


def run_question(
    data_merged: pd.DataFrame,
    llm: LLMClient,
    spec: QuestionSpec,
    merged_data_path: str | None = None,
) -> pd.DataFrame:
    """
    For each MRN: combine notes, call LLM, compare to gold, log results, print metrics.

    ``merged_data_path`` is written to the results CSV; defaults to ``MERGED_DATA_PATH``
    from config when omitted.

    Returns a DataFrame with MRN, prediction column, and Gold_Standard.
    """
    if merged_data_path is None:
        from config import MERGED_DATA_PATH

        merged_data_path = MERGED_DATA_PATH
    mrns = data_merged["MRN"].dropna().unique()
    if spec.max_mrns is not None:
        mrns = mrns[: spec.max_mrns]

    results: list[dict] = []
    predictions: list = []
    gold_standards: list = []

    for mrn in tqdm(mrns, total=len(mrns)):
        gold_standard = get_gold_standard(data_merged, mrn, spec.gold_standard_col)

        try:
            note_text = combine_medical_texts(
                data_merged, mrn, list(spec.note_sources)
            )
            prediction = extract_with_llm(
                spec.prompt_file, spec.options, note_text, llm,
                **spec.llm_kwargs,
            )

            results.append(
                {
                    "MRN": mrn,
                    spec.prediction_key: prediction,
                    "Gold_Standard": gold_standard
                    if gold_standard is not None
                    else "Unavailable",
                }
            )
            predictions.append(prediction)
            gold_standards.append(gold_standard)
            print(f"MRN: {mrn} -> {prediction}")

        except Exception as e:
            print(f"Error on MRN {mrn}: {e}")
            err = f"ERROR: {str(e)}"
            results.append(
                {
                    "MRN": mrn,
                    spec.prediction_key: err,
                    "Gold_Standard": gold_standard
                    if gold_standard is not None
                    else "Unavailable",
                }
            )
            predictions.append(err)
            gold_standards.append(gold_standard)

    df_results = pd.DataFrame(results)
    print(f"\nProcessed {len(results)} records")

    rows_logged = append_results_to_csv(
        question_name=spec.question_name,
        predictions=predictions,
        gold_standards=gold_standards,
        mrns=mrns,
        llm=llm,
        merged_data_path=merged_data_path,
    )

    metrics = evaluate_predictions(
        predictions, gold_standards, spec.question_name
    )
    print_evaluation_summary(metrics, spec.question_name)

    results_path = Path(RESULTS_DATA_PATH).resolve()
    if rows_logged:
        print(
            f"Results file: appended {rows_logged} row(s) to {results_path}",
            flush=True,
        )
    else:
        print(
            f"Results file: nothing appended (0 rows). Path: {results_path}",
            flush=True,
        )

    return df_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM extraction for one or more registry questions.",
        epilog="Questions available: see QUESTION_REGISTRY in questions.py",
    )
    parser.add_argument(
        "questions",
        nargs="+",
        metavar="QUESTION",
        help="Question key(s) to run (e.g. q1 q4) or 'all' to run everything.",
    )
    parser.add_argument(
        "--max-mrns",
        type=int,
        default=None,
        metavar="N",
        help="Limit MRNs per question (0 = all MRNs; default = all MRNs).",
    )
    return parser


if __name__ == "__main__":
    from config import MERGED_DATA_PATH
    from questions import QUESTION_REGISTRY

    args = _build_parser().parse_args()

    # Resolve question keys
    if args.questions == ["all"]:
        keys = list(QUESTION_REGISTRY.keys())
    else:
        keys = args.questions
        unknown = [k for k in keys if k not in QUESTION_REGISTRY]
        if unknown:
            raise SystemExit(
                f"Unknown question(s): {unknown}\n"
                f"Available: {list(QUESTION_REGISTRY.keys())}"
            )

    data_merged = pd.read_csv(MERGED_DATA_PATH)
    llm = create_llm_client_from_config()

    for key in keys:
        spec = QUESTION_REGISTRY[key]

        # Apply CLI --max-mrns override if provided
        if args.max_mrns is not None:
            max_mrns = None if args.max_mrns == 0 else args.max_mrns
            from dataclasses import replace
            spec = replace(spec, max_mrns=max_mrns)

        print(f"\n{'='*60}")
        print(f"Running: {spec.question_name}")
        print(f"{'='*60}")
        run_question(data_merged, llm, spec, merged_data_path=MERGED_DATA_PATH)
