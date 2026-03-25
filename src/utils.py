"""
Shared helpers for the extraction pipeline (single module, grouped by concern).

Sections:
  - Prompts + LLM: load_prompt, unwrap_structured_answer, extract_with_llm, options_to_enum_schema
  - Notes: combine_medical_texts
  - Gold + metrics: normalize_text, get_gold_standard, evaluate_predictions, print_evaluation_summary
  - Results: append_results_to_csv
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import pandas as pd

from config import RESULTS_DATA_PATH
from llm_client import LLMClient

# --- Prompts + LLM -----------------------------------------------------------

def load_prompt(prompt_file, options_text, note_text, max_length=4000):
    """Load a prompt template from file and fill in options and note text."""
    with open(f"prompts/{prompt_file}", "r") as f:
        prompt_template = f.read()

    truncated_note = (
        note_text[:max_length] if len(note_text) > max_length else note_text
    )

    return prompt_template.format(options=options_text, note_text=truncated_note)


def extract_with_llm(prompt_file, options_text, note_text, llm: LLMClient, **kwargs):
    """Load prompt template and run chat completion via the given LLM client.

    Extra **kwargs are forwarded to llm.generate_chat (provider-specific).
    """
    prompt_content = load_prompt(prompt_file, options_text, note_text)
    messages = [{"role": "user", "content": prompt_content}]
    raw = llm.generate_chat(messages, **kwargs)
    return unwrap_structured_answer(raw)


def options_to_enum_schema(options_text: str) -> dict:
    """Parse '- Label' lines from an options block into a JSON Schema enum.

    Returns a schema like ``{"type": "object", "properties": {"answer": {...}},
    "required": ["answer"]}``.

    Use with Ollama ``format=``; pair OpenAI ``response_format`` separately
    (e.g. ``{"type": "json_object"}``).
    """
    labels = [
        m.group(1).strip()
        for m in re.finditer(r"^- (.+)$", options_text, re.MULTILINE)
    ]
    if "UNKNOWN" in options_text:
        labels.append("UNKNOWN")
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": labels},
        },
        "required": ["answer"],
    }

def unwrap_structured_answer(raw: str) -> str:
    """Pull ``answer`` out of structured JSON; otherwise return stripped text.

    Handles optional `` ```json ... ``` `` fences and pretty-printed JSON.
    If parsing fails or there is no ``answer`` key, returns ``raw`` stripped.
    """
    s = (raw or "").strip()
    if not s:
        return s
    m = re.match(
        r"^```(?:json)?\s*\r?\n?(.*?)\r?\n?```\s*$",
        s,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        s = m.group(1).strip()
    if not s.lstrip().startswith("{"):
        return (raw or "").strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return (raw or "").strip()
    if isinstance(data, dict) and "answer" in data and data["answer"] is not None:
        return str(data["answer"]).strip()
    return (raw or "").strip()

# --- Notes -------------------------------------------------------------------


def combine_medical_texts(
    data, mrn, sources=("Discharge Summary", "Op Note", "Clerking")
):
    """Combine text from specified wide-table columns for a given MRN."""
    mrn_data = data[data["MRN"] == mrn]

    if mrn_data.empty:
        return ""

    row = mrn_data.iloc[0]
    combined_parts = []

    for source in sources:
        if source in row.index:
            text = row[source]
            if pd.notna(text):
                text_str = str(text).strip()
                if text_str:
                    combined_parts.append(f"{source}: {text_str}")

    return "\n\n".join(combined_parts)


# --- Gold + evaluation -------------------------------------------------------


def normalize_text(text):
    """Normalize text for comparison: lowercase, strip whitespace."""
    if pd.isna(text) or text is None:
        return ""
    return str(text).lower().strip()


def get_gold_standard(data_merged, mrn, column_name):
    """Get gold standard value for a given MRN and column."""
    mrn_data = data_merged[data_merged["MRN"] == mrn]
    if mrn_data.empty:
        return None
    value = mrn_data[column_name].iloc[0]
    return None if pd.isna(value) else value


def _prf_for_class(preds_norm, golds_norm, cls):
    """Precision, recall, F1 for one class (one-vs-rest)."""
    tp = sum(
        1 for p, g in zip(preds_norm, golds_norm) if p == cls and g == cls
    )
    fp = sum(
        1 for p, g in zip(preds_norm, golds_norm) if p == cls and g != cls
    )
    fn = sum(
        1 for p, g in zip(preds_norm, golds_norm) if p != cls and g == cls
    )
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def evaluate_predictions(predictions, gold_standards, question_name):
    """
    Evaluate predictions against gold standards.

    ``question_name`` is accepted for API consistency with call sites; metrics are not
    split by question yet.

    Returns a dict with accuracy, precision, recall, f1, counts, and example lists.
    """
    eval_data = [(p, g) for p, g in zip(predictions, gold_standards) if g is not None]

    if not eval_data:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "total": len(predictions),
            "with_gold_standard": 0,
            "correct": [],
            "incorrect": [],
        }

    preds, golds = zip(*eval_data)

    preds_normalized = [normalize_text(p) for p in preds]
    golds_normalized = [normalize_text(g) for g in golds]

    correct = sum(
        1 for p, g in zip(preds_normalized, golds_normalized) if p == g
    )
    total = len(preds_normalized)
    accuracy = correct / total if total > 0 else 0.0

    all_classes = set(preds_normalized + golds_normalized)

    if len(all_classes) == 2:
        pos_class = list(all_classes)[0]
        precision, recall, f1 = _prf_for_class(
            preds_normalized, golds_normalized, pos_class
        )
    else:
        class_metrics: dict[str, dict] = {}
        for cls in all_classes:
            prec, rec, f1_score = _prf_for_class(
                preds_normalized, golds_normalized, cls
            )
            class_metrics[cls] = {
                "precision": prec,
                "recall": rec,
                "f1": f1_score,
            }

        precision = (
            sum(m["precision"] for m in class_metrics.values()) / len(class_metrics)
            if class_metrics
            else 0.0
        )
        recall = (
            sum(m["recall"] for m in class_metrics.values()) / len(class_metrics)
            if class_metrics
            else 0.0
        )
        f1 = (
            sum(m["f1"] for m in class_metrics.values()) / len(class_metrics)
            if class_metrics
            else 0.0
        )

    correct_examples = []
    incorrect_examples = []

    for pred, gold, pred_norm, gold_norm in zip(
        preds, golds, preds_normalized, golds_normalized
    ):
        if pred_norm == gold_norm:
            correct_examples.append((pred, gold))
        else:
            incorrect_examples.append((pred, gold))

    correct_examples = correct_examples[:5]
    incorrect_examples = incorrect_examples[:5]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": len(predictions),
        "with_gold_standard": total,
        "correct": correct_examples,
        "incorrect": incorrect_examples,
    }


def print_evaluation_summary(metrics, question_name):
    """Print a formatted evaluation summary."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {question_name}")
    print(f"{'='*60}")

    if metrics["with_gold_standard"] == 0:
        print("⚠️  No gold standard available for evaluation")
        print(f"   Total records processed: {metrics['total']}")
        return

    print(f"Total records: {metrics['total']}")
    print(f"Records with gold standard: {metrics['with_gold_standard']}")
    print(
        f"Records without gold standard: {metrics['total'] - metrics['with_gold_standard']}"
    )
    print("\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1']:.3f}")

    if metrics["correct"]:
        print("\n✓ Correct Examples (showing up to 5):")
        for i, (pred, gold) in enumerate(metrics["correct"], 1):
            print(f"  {i}. Predicted: '{pred}' | Gold: '{gold}'")

    if metrics["incorrect"]:
        print("\n✗ Incorrect Examples (showing up to 5):")
        for i, (pred, gold) in enumerate(metrics["incorrect"], 1):
            print(f"  {i}. Predicted: '{pred}' | Gold: '{gold}'")

    print(f"{'='*60}\n")


# --- Results -----------------------------------------------------------------


def append_results_to_csv(
    question_name,
    predictions,
    gold_standards,
    mrns,
    llm: LLMClient,
    merged_data_path: str,
):
    """
    Append per-MRN results for a question to a single CSV file.

    Args:
        question_name: str, e.g. "Q1 - Primary reason for shunting"
        predictions: list of model outputs (len N)
        gold_standards: list of gold values (len N, None if unavailable)
        mrns: list/array of MRNs (len N)
        llm: client used for this run (provider/model_id logged from it)
        merged_data_path: path to the merged input CSV used for this run
    """
    run_ts = datetime.now(timezone.utc).isoformat()

    rows = []
    for mrn, pred, gold in zip(mrns, predictions, gold_standards):
        has_gold = gold is not None
        rows.append(
            {
                "MRN": mrn,
                "Question": question_name,
                "Prediction": pred,
                "Gold_Standard": "" if gold is None else gold,
                "Has_Gold": has_gold,
                "Provider": llm.provider,
                "Model": llm.model_id,
                "Merged_Data_Path": merged_data_path,
                "Run_Timestamp": run_ts,
            }
        )

    df_append = pd.DataFrame(rows)

    write_header = not os.path.exists(RESULTS_DATA_PATH)
    df_append.to_csv(RESULTS_DATA_PATH, mode="a", header=write_header, index=False)
