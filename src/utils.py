'''
Utility & helper functions
'''

import os
import pandas as pd
from datetime import datetime

from config import RESULTS_DATA_PATH
from llm_client import LLMClient

def load_prompt(prompt_file, options, note_text, max_length=4000):
    """Load a prompt template from file and fill in options and note text."""
    with open(f'prompts/{prompt_file}', 'r') as f:
        prompt_template = f.read()
    
    # Truncate note if needed
    truncated_note = note_text[:max_length] if len(note_text) > max_length else note_text
    
    return prompt_template.format(options=options, note_text=truncated_note)

def extract_with_llm(prompt_file, options, note_text, llm: LLMClient):
    """Load prompt template and run chat completion via the given LLM client."""
    prompt_content = load_prompt(prompt_file, options, note_text)
    messages = [{"role": "user", "content": prompt_content}]
    return llm.generate_chat(messages)

def combine_medical_texts(data, mrn, sources=('Discharge Summary', 'Op Note', 'Clerking')):
    """Combine text from specified sources for a given MRN."""
    mrn_data = data[data['MRN'] == mrn]

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

def normalize_text(text):
    """Normalize text for comparison: lowercase, strip whitespace."""
    if pd.isna(text) or text is None:
        return ""
    return str(text).lower().strip()

def get_gold_standard(data_merged, mrn, column_name):
    """Get gold standard value for a given MRN and column."""
    mrn_data = data_merged[data_merged['MRN'] == mrn]
    if mrn_data.empty:
        return None
    value = mrn_data[column_name].iloc[0]
    return None if pd.isna(value) else value

def evaluate_predictions(predictions, gold_standards, question_name):
    """
    Evaluate predictions against gold standards.
    
    Args:
        predictions: List of predicted values
        gold_standards: List of gold standard values (None if unavailable)
        question_name: Name of the question for display
    
    Returns:
        Dictionary with metrics and examples
    """
    # Filter to only records with gold standard
    eval_data = [(p, g) for p, g in zip(predictions, gold_standards) if g is not None]
    
    if not eval_data:
        return {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'total': len(predictions),
            'with_gold_standard': 0,
            'correct': [],
            'incorrect': []
        }
    
    preds, golds = zip(*eval_data)
    
    # Normalize for comparison
    preds_normalized = [normalize_text(p) for p in preds]
    golds_normalized = [normalize_text(g) for g in golds]
    
    # Calculate accuracy
    correct = sum(1 for p, g in zip(preds_normalized, golds_normalized) if p == g)
    total = len(preds_normalized)
    accuracy = correct / total if total > 0 else 0.0
    
    # For categorical questions, calculate precision, recall, F1
    # Get unique classes
    all_classes = set(preds_normalized + golds_normalized)
    
    if len(all_classes) == 2:  # Binary classification
        # Assume first class is positive
        pos_class = list(all_classes)[0]
        tp = sum(1 for p, g in zip(preds_normalized, golds_normalized) if p == pos_class and g == pos_class)
        fp = sum(1 for p, g in zip(preds_normalized, golds_normalized) if p == pos_class and g != pos_class)
        fn = sum(1 for p, g in zip(preds_normalized, golds_normalized) if p != pos_class and g == pos_class)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    else:  # Multi-class
        # Calculate macro-averaged precision, recall, F1
        class_metrics = {}
        for cls in all_classes:
            tp = sum(1 for p, g in zip(preds_normalized, golds_normalized) if p == cls and g == cls)
            fp = sum(1 for p, g in zip(preds_normalized, golds_normalized) if p == cls and g != cls)
            fn = sum(1 for p, g in zip(preds_normalized, golds_normalized) if p != cls and g == cls)
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            class_metrics[cls] = {'precision': prec, 'recall': rec, 'f1': f1_score}
        
        precision = sum(m['precision'] for m in class_metrics.values()) / len(class_metrics) if class_metrics else 0.0
        recall = sum(m['recall'] for m in class_metrics.values()) / len(class_metrics) if class_metrics else 0.0
        f1 = sum(m['f1'] for m in class_metrics.values()) / len(class_metrics) if class_metrics else 0.0
    
    # Collect examples
    correct_examples = []
    incorrect_examples = []
    
    for i, (pred, gold, pred_norm, gold_norm) in enumerate(zip(preds, golds, preds_normalized, golds_normalized)):
        if pred_norm == gold_norm:
            correct_examples.append((pred, gold))
        else:
            incorrect_examples.append((pred, gold))
    
    # Limit examples to 5 each
    correct_examples = correct_examples[:5]
    incorrect_examples = incorrect_examples[:5]
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total': len(predictions),
        'with_gold_standard': total,
        'correct': correct_examples,
        'incorrect': incorrect_examples
    }

def print_evaluation_summary(metrics, question_name):
    """Print a formatted evaluation summary."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {question_name}")
    print(f"{'='*60}")
    
    if metrics['with_gold_standard'] == 0:
        print(f"⚠️  No gold standard available for evaluation")
        print(f"   Total records processed: {metrics['total']}")
        return
    
    print(f"Total records: {metrics['total']}")
    print(f"Records with gold standard: {metrics['with_gold_standard']}")
    print(f"Records without gold standard: {metrics['total'] - metrics['with_gold_standard']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1']:.3f}")
    
    if metrics['correct']:
        print(f"\n✓ Correct Examples (showing up to 5):")
        for i, (pred, gold) in enumerate(metrics['correct'], 1):
            print(f"  {i}. Predicted: '{pred}' | Gold: '{gold}'")
    
    if metrics['incorrect']:
        print(f"\n✗ Incorrect Examples (showing up to 5):")
        for i, (pred, gold) in enumerate(metrics['incorrect'], 1):
            print(f"  {i}. Predicted: '{pred}' | Gold: '{gold}'")
    
    print(f"{'='*60}\n")

def append_results_to_csv(question_name, predictions, gold_standards, mrns, llm: LLMClient):
    """
    Append per-MRN results for a question to a single CSV file.

    Args:
        question_name: str, e.g. "Q1 - Primary reason for shunting"
        predictions: list of model outputs (len N)
        gold_standards: list of gold values (len N, None if unavailable)
        mrns: list/array of MRNs (len N)
        llm: client used for this run (provider/model_id logged from it)
    """
    run_ts = datetime.utcnow().isoformat()
    
    rows = []
    for mrn, pred, gold in zip(mrns, predictions, gold_standards):
        has_gold = gold is not None
        rows.append({
            "MRN": mrn,
            "Question": question_name,
            "Prediction": pred,
            "Gold_Standard": "" if gold is None else gold,
            "Has_Gold": has_gold,
            "Provider": llm.provider,
            "Model": llm.model_id,
            "Run_Timestamp": run_ts,
        })
    
    df_append = pd.DataFrame(rows)
    
    write_header = not os.path.exists(RESULTS_DATA_PATH)
    df_append.to_csv(RESULTS_DATA_PATH, mode="a", header=write_header, index=False)