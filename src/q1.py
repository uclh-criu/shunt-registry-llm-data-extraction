import pandas as pd
from tqdm import tqdm

from utils import load_llm, get_gold_standard, combine_medical_texts, extract_with_llm, append_results_to_csv, evaluate_predictions, print_evaluation_summary
from config import MERGED_DATA_PATH, provider, model_id
from registry_options import q1_options

def extract_q1(data_merged):
    
    # Load llm
    client, hf_tokenizer, hf_model, device = load_llm()
    
    # Main loop for Q1 extraction
    results = []

    # Get unique MRNs and process first N (adjust slice as needed)
    unique_mrns = data_merged['MRN'].dropna().unique()[:10]

    gold_standard_col = 'Primary reason for shunting'
    question_name = 'Q1 - Primary reason for shunting'

    predictions = []
    gold_standards = []

    for mrn in tqdm(unique_mrns, total=len(unique_mrns)):
        gold_standard = get_gold_standard(data_merged, mrn, gold_standard_col)

        try:
            note_text = combine_medical_texts(data_merged, mrn, ['Discharge Summary', 'Op Note', 'Clerking'])
            prediction = extract_with_llm('q1_prompt.txt', q1_options, note_text, client, hf_tokenizer, hf_model, device)

            results.append({
                'MRN': mrn,
                'Q1_Primary_Reason_Shunting': prediction,
                'Gold_Standard': gold_standard if gold_standard is not None else 'Unavailable',
            })

            predictions.append(prediction)
            gold_standards.append(gold_standard)

            print(f"MRN: {mrn} -> {prediction}")

        except Exception as e:
            print(f"Error on MRN {mrn}: {e}")
            err = f'ERROR: {str(e)}'
            results.append({
                'MRN': mrn,
                'Q1_Primary_Reason_Shunting': err,
                'Gold_Standard': gold_standard if gold_standard is not None else 'Unavailable',
            })
            predictions.append(err)
            gold_standards.append(gold_standard)

    df_results = pd.DataFrame(results)
    print(f"\nProcessed {len(results)} records")

    append_results_to_csv(
        question_name=question_name,
        predictions=predictions,
        gold_standards=gold_standards,
        mrns=unique_mrns,
        provider=provider,
        model_id=model_id,
    )

    metrics = evaluate_predictions(predictions, gold_standards, question_name)
    print_evaluation_summary(metrics, question_name)

    # df_results.to_csv('q1_extraction_results.csv', index=False)

if __name__ == "__main__":
    data_merged = pd.read_csv(MERGED_DATA_PATH)
    extract_q1(data_merged)