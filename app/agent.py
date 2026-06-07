import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .nlp_utils import clean_report_text, extract_parameters
from .llm_client import get_model
from .prompts import (
    build_labsense_chain,
    build_labsense_parameters_batch_chain,
    build_overall_summary_chain,
    make_prompt_inputs,
)

# Add project root so rag_store is importable
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from rag_store import retrieve_context as _retrieve_context
    _RAG_AVAILABLE = True
except Exception:
    _RAG_AVAILABLE = False


def _build_rag_context(parameters: List[Dict]) -> tuple:
    """Query the KB vector store using parameter names.
    Returns (rag_context_str, list_of_source_doc_dicts).
    """
    if not _RAG_AVAILABLE:
        return "", []
    query = " ".join(p["parameter"] for p in parameters)
    try:
        docs = _retrieve_context(query, k=4)
        context_str = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'KB')}]\n{d.page_content}"
            for d in docs
        )
        sources = [
            {"source": d.metadata.get("source", "KB"), "excerpt": d.page_content[:300]}
            for d in docs
        ]
        return context_str, sources
    except Exception:
        return "", []


def _message_text(response) -> str:
    if isinstance(response.content, list):
        text_parts = []
        for item in response.content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            else:
                text_parts.append(str(item))
        return "".join(text_parts).strip()
    return str(response.content).strip()


def _parse_llm_json(raw_output: str) -> dict:
    raw_output = raw_output.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        if raw_output.lower().startswith("json"):
            raw_output = raw_output[4:].strip()

    if not raw_output:
        raise ValueError(
            "LLM returned an empty response. Check API key, model availability, and network access."
        )

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    start_idx = raw_output.find("{")
    if start_idx == -1:
        raise ValueError(
            f"LLM response was not valid JSON. Response starts with: {raw_output[:160]!r}"
        )

    brace_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(raw_output)):
        if raw_output[i] == "{":
            brace_count += 1
        elif raw_output[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if brace_count != 0:
        raise ValueError(
            "LLM JSON looks truncated (often hit output token limit). "
            "Try fewer lab lines, raise GROQ_MAX_OUTPUT_TOKENS / HARD_CAP on a higher tier, "
            "or reduce LABSENSE_PARAMS_PER_BATCH. "
            f"Response preview: {raw_output[:400]!r}"
        )

    try:
        return json.loads(raw_output[start_idx:end_idx])
    except json.JSONDecodeError:
        raise ValueError(
            f"Could not parse JSON from model output. Extracted: {raw_output[start_idx:end_idx][:300]!r}"
        )


def _compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _should_batch(parameters: List[Dict], patient_context: Optional[Dict]) -> bool:
    batch_threshold = int(os.getenv("LABSENSE_PARAMS_PER_BATCH", "6"))
    payload = make_prompt_inputs(parameters, patient_context)
    est_chars = len(payload["parameters_json"]) + len(payload["patient_context"])
    max_chars = int(os.getenv("LABSENSE_SINGLE_CALL_MAX_CHARS", "4500"))
    return len(parameters) > batch_threshold or est_chars > max_chars


def run_labsense_agent(raw_text: str, patient_context: Optional[Dict] = None) -> Dict:
    cleaned = clean_report_text(raw_text)
    parameters = extract_parameters(cleaned)

    if not parameters:
        raise ValueError("No parameters could be extracted from the report text.")

    # ── RAG retrieval ────────────────────────────────────────────────────────
    rag_context, retrieved_sources = _build_rag_context(parameters)
    # ─────────────────────────────────────────────────────────────────────────

    llm = get_model()

    if not _should_batch(parameters, patient_context):
        chain = build_labsense_chain(llm)
        inputs = make_prompt_inputs(parameters, patient_context, rag_context)
        result = _parse_llm_json(_message_text(chain.invoke(inputs)))
        result["_retrieved_sources"] = retrieved_sources
        return result

    batch_size = max(1, int(os.getenv("LABSENSE_PARAMS_PER_BATCH", "6")))
    batch_chain = build_labsense_parameters_batch_chain(llm)
    merged: List[Dict] = []

    for i in range(0, len(parameters), batch_size):
        batch = parameters[i : i + batch_size]
        inputs = make_prompt_inputs(batch, patient_context, rag_context)
        data = _parse_llm_json(_message_text(batch_chain.invoke(inputs)))
        part = data.get("parameters")
        if not isinstance(part, list):
            raise ValueError(
                "Batch response missing a parameters array. "
                f"Got keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )
        merged.extend(part)

    summary_chain = build_overall_summary_chain(llm)
    analyzed_compact = _compact_json(
        [
            {
                "parameter": p.get("parameter"),
                "reported_value": p.get("reported_value"),
                "unit": p.get("unit"),
                "severity": p.get("severity"),
            }
            for p in merged
        ]
    )
    summary_inputs = {
        "patient_context": _compact_json(patient_context or {}),
        "analyzed_json": analyzed_compact,
    }
    summary_data = _parse_llm_json(_message_text(summary_chain.invoke(summary_inputs)))

    overall = summary_data.get("overall_summary", "")
    if not overall:
        overall = (
            "Review all parameters above with your clinician; "
            "this automated summary could not be produced."
        )

    return {"parameters": merged, "overall_summary": overall, "_retrieved_sources": retrieved_sources}
