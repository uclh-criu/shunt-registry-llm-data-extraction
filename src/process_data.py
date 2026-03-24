'''
Script to pre-process data ready for LLM extraction
'''

import pandas as pd
from config import RESULTS_CSV_PATH, INPUT_DATA_PATH, EVAL_DATA_PATH, MERGED_DATA_PATH, CLERKING_DATA_PATH, DISCHARGE_DATA_PATH, OP_NOTE_DATA_PATH, MDT_DATA_PATH

def load_data():
    data = pd.read_csv(INPUT_DATA_PATH, encoding='latin-1')
    evaluation = pd.read_csv(EVAL_DATA_PATH, encoding='latin-1')
    return data, evaluation

def format_input_data(data):
    # 1. Create MRN column from the new ID
    data['MRN'] = data['patient']

    # 2. Map notetype values to the old column names used in the notebook
    note_type_map = {
        # new notetype value -> existing column name
        'Discharge Summary': 'Discharge Summary',
        'Op Note': 'Op Note',
        'Clerking': 'Clerking',
        'MDT Outcome': 'MDT Outcome'
    }

    data['notetype_clean'] = data['notetype'].map(note_type_map).fillna(data['notetype'])

    # 3. Pivot from long (multiple rows per MRN) to wide (one row per MRN)
    # If there are multiple notes of the same type for a patient, concatenate them.
    wide = (
        data
        .sort_values('notecreationinstant')
        .groupby(['MRN', 'notetype_clean'])['note']
        .apply(lambda s: "\n\n".join(str(x) for x in s if pd.notna(x)))
        .unstack('notetype_clean')
        .reset_index()
    )

    # 4. Replace original data with wide-format version expected by rest of notebook
    data = wide

    return data

def split_data(data):
    clerking = data[['MRN', 'Clerking']]
    op_note = data[['MRN', 'Op Note']]
    discharge_summary = data[['MRN', 'Discharge Summary']]
    #imaging = data[['MRN', 'Imaging Report']]
    mdt = data[['MRN', 'MDT Outcome']]

    #Save to csv
    clerking.to_csv(CLERKING_DATA_PATH, index=False)
    op_note.to_csv(OP_NOTE_DATA_PATH, index=False)
    discharge_summary.to_csv(DISCHARGE_DATA_PATH, index=False)
    mdt.to_csv(MDT_DATA_PATH, index=False)

def clean_eval_data(evaluation):
# Clean evaluation column names: remove " | Shunt Operation" suffix
    evaluation = evaluation.rename(
        columns=lambda c: c.replace(" | Shunt Operation", "")
    )
    return evaluation

def check_mrn_overlap(data, evaluation):
    # Check MRN overlaps between datasets
    print("=== Overlap Analysis ===\n")

    # MRN overlaps
    data_mrns = set(data['MRN'].dropna().unique())
    eval_mrns = set(evaluation['MRN'].dropna().unique())
    mrn_overlap = data_mrns.intersection(eval_mrns)

    print(f"MRN Statistics:")
    print(f"  - Unique MRNs in data: {len(data_mrns)}")
    print(f"  - Unique MRNs in evaluation: {len(eval_mrns)}")
    print(f"  - Overlapping MRNs: {len(mrn_overlap)}")
    print(f"  - Overlap percentage (data): {len(mrn_overlap)/len(data_mrns)*100:.1f}%")
    print(f"  - Overlap percentage (evaluation): {len(mrn_overlap)/len(eval_mrns)*100:.1f}%")
    print(f"  - MRNs only in data: {len(data_mrns - eval_mrns)}")
    print(f"  - MRNs only in evaluation: {len(eval_mrns - data_mrns)}")

def merge_data(data, evaluation):
    # Merge data and evaluation datasets on MRN
    # This creates a merged dataset with both source data and gold standard labels
    data_merged = data.merge(
        evaluation,
        on='MRN',
        how='left',
        suffixes=('', '_eval')
    )
    print("Merged dataset created:")
    print(f"  - Total records: {len(data_merged)}")
    print(f"  - Records with gold standard: {data_merged['Primary reason for shunting'].notna().sum()}")
    print(f"  - Records without gold standard: {data_merged['Primary reason for shunting'].isna().sum()}")
    data_merged.to_csv(MERGED_DATA_PATH, index=False)

if __name__ == "__main__":
    data, evaluation = load_data()
    data = format_input_data(data)
    split_data(data)
    evaluation = clean_eval_data(evaluation)
    check_mrn_overlap(data, evaluation)
    merge_data(data, evaluation)