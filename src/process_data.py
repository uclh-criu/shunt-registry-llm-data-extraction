"""
Pre-process the combined CSV (notes + gold standard columns) into the merged dataset
used by the LLM extraction pipeline.
"""

import pandas as pd
from config import INPUT_DATA_PATH, MERGED_DATA_PATH


def load_data() -> pd.DataFrame:
    """Load the single combined CSV (notes + GOLD columns)."""
    return pd.read_csv(INPUT_DATA_PATH, encoding="latin-1")


def build_merged_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Rename GOLD columns to the names expected in questions.py and keep only
    MRN, note columns, and the gold-standard columns used for evaluation.
    """
    # Map GOLD columns in the new CSV to the short names used in questions.py
    rename_map = {
        # Q1
        "Primary reason for shunting | Shunt Operation(GOLD)": "Primary reason for shunting",
        # Q2
        "EVD insertion in the last 30 days | Shunt Operation(GOLD)": "EVD insertion in the last 30 days",
        # Q4
        "Primary reason for revision | Shunt Operation": "Primary reason for revision",
        # Q8
        "Choroid plexectomy | Shunt Operation": "Choroid plexectomy",
        # Q9
        "Subtemporal decompression | Shunt Operation": "Subtemporal decompression",
        # Q10
        "Ventricular size prior to surgery | Shunt Operation": "Ventricular size prior to surgery",
        # Q11
        "Concurrent chemoradiotherapy for primary CNS tumour | Shunt Operation(GOLD)": (
            "Concurrent chemoradiotherapy for primary CNS tumour"
        ),
        # Q12
        "Co-existing CNS infection | Shunt Operation(GOLD)": "Co-existing CNS infection",
        # Q13
        "CNS infection in the last 6 months | Shunt Operation(GOLD)": (
            "CNS infection in the last 6 months"
        ),
        # Q18
        "Consultant presence | Shunt Operation": "Consultant presence",
        # Q23
        "Operation title | Shunt Operation": "Operation title",
        # Q25
        "Procedure | Shunt Operation": "Procedure",
        # Q26
        "Post-op plan | Shunt Operation": "Post-op plan",
    }

    data = data.rename(columns=rename_map)

    # Note columns to preserve for prompts (including new ones you listed)
    note_cols = [
        "Clerking",
        "Op Note",
        "Discharge Summary",
        "Imaging Report",
        "MDT Outcome Pre Proc Date",
        "MDT Outcome Pre Proc",
        "MDT Outcome Post Proc Date",
        "MDT Outcome Post Proc",
        "ImplantName",
        "ManufacturerFull",
    ]

    # Gold-standard columns as referenced in questions.py
    gold_cols = [
        "Primary reason for shunting",
        "EVD insertion in the last 30 days",
        "Primary reason for revision",
        "Choroid plexectomy",
        "Subtemporal decompression",
        "Ventricular size prior to surgery",
        "Concurrent chemoradiotherapy for primary CNS tumour",
        "Co-existing CNS infection",
        "CNS infection in the last 6 months",
        "Consultant presence",
        "Operation title",
        "Procedure",
        "Post-op plan",
    ]

    # Identifier columns to always carry through
    id_cols = ["MRN", "CSN"]

    cols_to_keep = id_cols + note_cols + gold_cols

    missing = [c for c in cols_to_keep if c not in data.columns]
    if missing:
        print("Warning: the following expected columns were not found in the input data:")
        for c in missing:
            print(f"  - {c}")

    present_cols = [c for c in cols_to_keep if c in data.columns]
    data_merged = data[present_cols].copy()

    print("Merged dataset created from combined CSV:")
    print(f"  - Total records: {len(data_merged)}")
    if "Primary reason for shunting" in data_merged.columns:
        has_q1 = data_merged["Primary reason for shunting"].notna().sum()
        print(f"  - Records with Q1 gold standard: {has_q1}")
        print(f"  - Records without Q1 gold standard: {len(data_merged) - has_q1}")

    return data_merged


if __name__ == "__main__":
    data = load_data()
    data_merged = build_merged_dataset(data)
    data_merged.to_csv(MERGED_DATA_PATH, index=False)