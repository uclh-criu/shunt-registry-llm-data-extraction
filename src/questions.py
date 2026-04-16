"""
Central registry of all QuestionSpec definitions.

To add a new question:
  1. Add its options string to registry_options.py.
  2. Add a prompt file to prompts/.
  3. Add a QuestionSpec entry to QUESTION_REGISTRY below.

Not registered here (non-LLM): Q22 "Include note" — in analysis.ipynb this is a simple
check that Op Note text exists (Yes/No), not a prompt-based extraction.
"""

from question_runner import QuestionSpec
from registry_options import (
    q1_options,
    q4_options,
    q8_options,
    q9_options,
    q10_options,
    q11_options,
    q12_options,
    q13_options,
    q18_options,
    q23_options,
    q25_options,
    q26_options,
)
from utils import free_text_answer_schema, options_to_enum_schema


_q1_schema = options_to_enum_schema(q1_options)
_q4_schema = options_to_enum_schema(q4_options)
_q8_schema = options_to_enum_schema(q8_options)
_q9_schema = options_to_enum_schema(q9_options)
_q10_schema = options_to_enum_schema(q10_options)
_q11_schema = options_to_enum_schema(q11_options)
_q12_schema = options_to_enum_schema(q12_options)
_q13_schema = options_to_enum_schema(q13_options)
_q18_schema = options_to_enum_schema(q18_options)
_q23_schema = free_text_answer_schema()
_q25_schema = free_text_answer_schema()
_q26_schema = free_text_answer_schema()

ALL_NOTES_SOURCES = (
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
)

OP_NOTE_ONLY = ("Op Note",)

# ---------------------------------------------------------------------------
# Question specs — one entry per registry question
# ---------------------------------------------------------------------------

QUESTION_REGISTRY: dict[str, QuestionSpec] = {
    "q1": QuestionSpec(
        question_name="Q1 - Primary reason for shunting",
        gold_standard_col="Primary reason for shunting",
        prompt_file="q1_prompt.txt",
        options=q1_options,
        prediction_key="Q1_Primary_Reason_Shunting",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q1_schema,        # Ollama structured output
            "response_format": {          # OpenAI JSON mode
                "type": "json_object",
            },
            "options": {"temperature": 0},  # Ollama deterministic
        },
    ),
    "q4": QuestionSpec(
        question_name="Q4 - Primary reason for revision",
        gold_standard_col="Primary reason for revision",
        prompt_file="q4_prompt.txt",
        options=q4_options,
        prediction_key="Q4_Primary_Reason_Revision",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q4_schema,  # Ollama structured output
            "response_format": {  # OpenAI JSON mode
                "type": "json_object",
            },
            "options": {"temperature": 0},  # Ollama deterministic
        },
    ),
    "q8": QuestionSpec(
        question_name="Q8 - Choroid plexectomy",
        gold_standard_col="Choroid plexectomy",
        prompt_file="q8_prompt.txt",
        options=q8_options,
        prediction_key="Q8_Choroid_Plexectomy",
        note_sources=OP_NOTE_ONLY,
        llm_kwargs={
            "format": _q8_schema,  # Ollama structured output
            "response_format": {  # OpenAI JSON mode
                "type": "json_object",
            },
            "options": {"temperature": 0},  # Ollama deterministic
        },
    ),
    "q9": QuestionSpec(
        question_name="Q9 - Subtemporal decompression",
        gold_standard_col="Subtemporal decompression",
        prompt_file="q9_prompt.txt",
        options=q9_options,
        prediction_key="Q9_Subtemporal_Decompression",
        note_sources=OP_NOTE_ONLY,
        llm_kwargs={
            "format": _q9_schema,  # Ollama structured output
            "response_format": {  # OpenAI JSON mode
                "type": "json_object",
            },
            "options": {"temperature": 0},  # Ollama deterministic
        },
    ),
    "q10": QuestionSpec(
        question_name="Q10 - Ventricular size prior to surgery",
        gold_standard_col="Ventricular size prior to surgery",
        prompt_file="q10_prompt.txt",
        options=q10_options,
        prediction_key="Q10_Ventricular_Size_Prior_Surgery",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q10_schema,  # Ollama structured output
            "response_format": {  # OpenAI JSON mode
                "type": "json_object",
            },
            "options": {"temperature": 0},  # Ollama deterministic
        },
    ),
    "q11": QuestionSpec(
        question_name="Q11 - Concurrent chemoradiotherapy for primary CNS tumour",
        gold_standard_col="Concurrent chemoradiotherapy for primary CNS tumour",
        prompt_file="q11_prompt.txt",
        options=q11_options,
        prediction_key="Q11_Concurrent_Chemoradiotherapy",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q11_schema,  # Ollama structured output
            "response_format": {  # OpenAI JSON mode
                "type": "json_object",
            },
            "options": {"temperature": 0},  # Ollama deterministic
        },
    ),
    "q12": QuestionSpec(
        question_name="Q12 - Co-existing CNS infection",
        gold_standard_col="Co-existing CNS infection",
        prompt_file="q12_prompt.txt",
        options=q12_options,
        prediction_key="Q12_Coexisting_CNS_Infection",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q12_schema,  # Ollama structured output
            "response_format": {  # OpenAI JSON mode
                "type": "json_object",
            },
            "options": {"temperature": 0},  # Ollama deterministic
        },
    ),
    "q13": QuestionSpec(
        question_name="Q13 - CNS infection in the last 6 months",
        gold_standard_col="CNS infection in the last 6 months",
        prompt_file="q13_prompt.txt",
        options=q13_options,
        prediction_key="Q13_CNS_Infection_Last_6_Months",
        note_sources=ALL_NOTES_SOURCES,
        llm_kwargs={
            "format": _q13_schema,  # Ollama structured output
            "response_format": {  # OpenAI JSON mode
                "type": "json_object",
            },
            "options": {"temperature": 0},  # Ollama deterministic
        },
    ),
    "q18": QuestionSpec(
        question_name="Q18 - Consultant presence",
        gold_standard_col="Consultant presence",
        prompt_file="q18_prompt.txt",
        options=q18_options,
        prediction_key="Q18_Consultant_Presence",
        note_sources=OP_NOTE_ONLY,
        llm_kwargs={
            "format": _q18_schema,  # Ollama structured output
            "response_format": {  # OpenAI JSON mode
                "type": "json_object",
            },
            "options": {"temperature": 0},  # Ollama deterministic
        },
    ),
    # "q23": QuestionSpec(
    #     question_name="Q23 - Operation title",
    #     gold_standard_col="Operation title",
    #     prompt_file="q23_prompt.txt",
    #     options=q23_options,
    #     prediction_key="Q23_Operation_Title",
    #     note_sources=ALL_NOTES_SOURCES,
    #     llm_kwargs={
    #         "format": _q23_schema,  # free-text JSON answer
    #         "response_format": {
    #             "type": "json_object",
    #         },
    #         "options": {"temperature": 0},
    #     },
    # ),
    # "q25": QuestionSpec(
    #     question_name="Q25 - Procedure",
    #     gold_standard_col="Procedure",
    #     prompt_file="q25_prompt.txt",
    #     options=q25_options,
    #     prediction_key="Q25_Procedure",
    #     note_sources=ALL_NOTES_SOURCES,
    #     llm_kwargs={
    #         "format": _q25_schema,
    #         "response_format": {
    #             "type": "json_object",
    #         },
    #         "options": {"temperature": 0},
    #     },
    # ),
    # "q26": QuestionSpec(
    #     question_name="Q26 - Post-operative plan",
    #     gold_standard_col="Post-op plan",
    #     prompt_file="q26_prompt.txt",
    #     options=q26_options,
    #     prediction_key="Q26_Post_Operative_Plan",
    #     note_sources=ALL_NOTES_SOURCES,
    #     llm_kwargs={
    #         "format": _q26_schema,
    #         "response_format": {
    #             "type": "json_object",
    #         },
    #         "options": {"temperature": 0},
    #     },
    # ),
    # -----------------------------------------------------------------------
    # Add further questions here, e.g.:
    # "q4": QuestionSpec(
    #     question_name="Q4 - Primary reason for revision",
    #     gold_standard_col="Primary reason for revision",
    #     prompt_file="q4_prompt.txt",
    #     options=q4_options,
    #     prediction_key="Q4_Primary_Reason_Revision",
    #     llm_kwargs={"format": options_to_enum_schema(q4_options), ...},
    # ),
    # -----------------------------------------------------------------------
}
