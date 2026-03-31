"""
Evaluate logged extraction results from RESULTS_DATA_PATH.

Usage:
    python src/evaluate_results.py
    python src/evaluate_results.py --results-path all_results.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from config import RESULTS_DATA_PATH
from utils import evaluate_predictions


REQUIRED_COLUMNS = {"Question", "Prediction", "Gold_Standard"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate per-question performance from the appended results CSV.",
    )
    parser.add_argument(
        "--results-path",
        default=RESULTS_DATA_PATH,
        help=(
            "Path to results CSV (default: RESULTS_DATA_PATH from config/.env)."
        ),
    )
    parser.add_argument(
        "--question-width",
        type=int,
        default=36,
        metavar="N",
        help="Max display width for the Question column (default: 36).",
    )
    return parser


def _read_results(results_path: str) -> pd.DataFrame:
    path = Path(results_path)
    if not path.exists():
        raise SystemExit(f"Results file not found: {path.resolve()}")

    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Results file is empty: {path.resolve()}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(
            "Results file is missing required column(s): "
            f"{sorted(missing)}. Found: {list(df.columns)}"
        )

    return df


def _truncate(text: str, max_width: int) -> str:
    if max_width <= 0:
        return text
    if len(text) <= max_width:
        return text
    if max_width <= 3:
        return "." * max_width
    return text[: max_width - 3] + "..."


def _question_key(question_name: str) -> str:
    m = re.match(r"^\s*(Q\d+)\b", question_name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return question_name


def evaluate_results(df: pd.DataFrame, question_width: int) -> None:
    questions = sorted(str(q) for q in df["Question"].dropna().unique())
    if not questions:
        raise SystemExit("No question values found in 'Question' column.")

    headers = [
        "Q",
        "Total",
        "WithGold",
        "NoGold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]
    rows: list[list[str]] = []

    for question_name in questions:
        subset = df[df["Question"] == question_name]
        predictions = subset["Prediction"].tolist()

        # Keep API parity with question_runner: treat missing/blank as no gold.
        gold_standards = [
            None
            if pd.isna(g) or str(g).strip() == ""
            else g
            for g in subset["Gold_Standard"].tolist()
        ]

        metrics = evaluate_predictions(predictions, gold_standards, question_name)
        without_gold = metrics["total"] - metrics["with_gold_standard"]

        def _fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.3f}"

        rows.append(
            [
                _truncate(_question_key(question_name), question_width),
                str(metrics["total"]),
                str(metrics["with_gold_standard"]),
                str(without_gold),
                _fmt(metrics["accuracy"]),
                _fmt(metrics["precision"]),
                _fmt(metrics["recall"]),
                _fmt(metrics["f1"]),
            ]
        )

    col_widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def _render_row(values: list[str]) -> str:
        return " | ".join(
            value.ljust(col_widths[i]) for i, value in enumerate(values)
        )

    print(_render_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(_render_row(row))


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results_df = _read_results(args.results_path)
    evaluate_results(results_df, question_width=args.question_width)
