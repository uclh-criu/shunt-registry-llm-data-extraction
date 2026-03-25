"""
Central registry of all QuestionSpec definitions.

To add a new question:
  1. Add its options string to registry_options.py.
  2. Add a prompt file to prompts/.
  3. Add a QuestionSpec entry to QUESTION_REGISTRY below.
"""

from question_runner import QuestionSpec
from registry_options import q1_options
from utils import options_to_enum_schema


_q1_schema = options_to_enum_schema(q1_options)

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
        note_sources=("Discharge Summary", "Op Note", "Clerking"),
        llm_kwargs={
            "format": _q1_schema,        # Ollama structured output
            "response_format": {          # OpenAI JSON mode
                "type": "json_object",
            },
            "options": {"temperature": 0},  # Ollama deterministic
        },
    ),
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
