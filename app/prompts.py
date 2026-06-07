from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
import json

# ---------------------------------------------------------------------------
# System prompt – RAG aware
# ---------------------------------------------------------------------------
_LABSENSE_SYSTEM = (
    "You are LabSense, a clinical AI assistant that explains laboratory test reports. "
    "For each parameter, classify the value as Normal, Borderline, Abnormal, or Critical. "
    "Use the supplied `rag_context` passages as the primary knowledge source for reference ranges and interpretations. "
    "If a test is NOT covered in rag_context, use well-established clinical reference ranges from standard medical references "
    "(e.g., Harrison's Principles of Internal Medicine, WHO guidelines, or established lab norms). "
    "NEVER leave reference_range blank or write 'Reference not provided' — always supply a specific numeric range. "
    "Only use values you are confident are accurate; do not guess or hallucinate ranges. "
    "Always recommend discussing results with a doctor. "
    "Keep explanations short (one or two sentences). "
    "Return ONLY a valid JSON object with the required schema."
)

# ---------------------------------------------------------------------------
# Parameter field template (used to build the schema block)
# ---------------------------------------------------------------------------
_PARAMETER_FIELDS = (
    '      "parameter": "string",\n'
    '      "reported_value": "string",\n'
    '      "unit": "string or null",\n'
    '      "severity": "Normal | Borderline | Abnormal | Critical",\n'
    '      "reference_range": "string",\n'
    '      "explanation": "string",\n'
    '      "recommendation": "string"\n'
)

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

def build_labsense_chain(llm):
    """Chain that expects parameters_json, patient_context, and rag_context.
    Returns the full JSON output (including overall_summary)."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", _LABSENSE_SYSTEM),
            (
                "human",
                (
                    "Patient context (may be null):\n"
                    "{patient_context}\n\n"
                    "Extracted parameters:\n"
                    "{parameters_json}\n\n"
                    "RAG context (reference passages):\n"
                    "{rag_context}\n\n"
                    "Schema:\n{{\n"
                    "  \"parameters\": [\n"
                    "    {{\n"
                    f"{_PARAMETER_FIELDS}"
                    "    }}\n"
                    "  ],\n"
                    "  \"overall_summary\": \"string\"\n"
                    "}}\n\n"
                    "Now produce the JSON object."
                ),
            ),
        ]
    )
    return template | llm


def build_labsense_parameters_batch_chain(llm):
    """Batch version – returns only a parameters array (no overall_summary)."""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                _LABSENSE_SYSTEM + " For this request return ONLY {{\"parameters\": [...]}} — do not include overall_summary.",
            ),
            (
                "human",
                (
                    "Patient context (may be null):\n"
                    "{patient_context}\n\n"
                    "Extracted parameters (analyze only this batch):\n"
                    "{parameters_json}\n\n"
                    "Schema:\n{{\n"
                    "  \"parameters\": [\n"
                    "    {{\n"
                    f"{_PARAMETER_FIELDS}"
                    "    }}\n"
                    "  ]\n"
                    "}}\n\n"
                    "Now produce the JSON object."
                ),
            ),
        ]
    )
    return template | llm


def build_overall_summary_chain(llm):
    """Summarize the combined findings – returns only overall_summary JSON."""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You briefly summarize combined laboratory findings for a lay reader. "
                "Do not diagnose. Encourage discussion with a clinician. "
                "Return ONLY valid JSON: {{\"overall_summary\": \"string\"}}",
            ),
            (
                "human",
                (
                    "Patient context:\n{patient_context}\n\n"
                    "Analyzed parameters (compact JSON array):\n{analyzed_json}\n\n"
                    "Produce the JSON object."
                ),
            ),
        ]
    )
    return template | llm


def make_prompt_inputs(parameters: List[Dict], patient_context: Optional[Dict] = None, rag_context: str = "") -> Dict:
    """Prepare the dict passed to the chain.
    rag_context is a plain string containing the concatenated reference passages.
    """
    compact = lambda obj: json.dumps(obj, separators=(",", ":"))
    return {
        "parameters_json": compact(parameters),
        "patient_context": compact(patient_context or {}),
        "rag_context": rag_context,
    }
