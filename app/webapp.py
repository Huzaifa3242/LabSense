
import sys
import os

# Ensure the project root (parent of app/) is always on sys.path
# regardless of how Streamlit launches this script.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from collections import Counter
from typing import Optional

import pandas as pd
import streamlit as st

from app.agent import run_labsense_agent


def run_agent(report_text: str, patient_context: Optional[dict], placeholder=None) -> dict:
    """Wrapper around the working CLI agent function."""
    return run_labsense_agent(report_text, patient_context)


def _severity_badge(severity: str) -> str:
    s = (severity or "").strip()
    icons = {
        "Normal": "🟢",
        "Borderline": "🟡",
        "Abnormal": "🟠",
        "Critical": "🔴",
    }
    icon = icons.get(s, "⚪")
    return f"{icon} {s}" if s else "—"


def _format_value_cell(p: dict) -> str:
    unit = (p.get("unit") or "").strip()
    val = p.get("reported_value", "")
    return f"{val} {unit}".strip() if unit else str(val)


def _summary_table_dataframe(parameters: list) -> pd.DataFrame:
    """Compact columns only — Streamlit truncates long text in dataframes."""
    rows = []
    for p in parameters:
        rows.append(
            {
                "Test": p.get("parameter") or "—",
                "Your value": _format_value_cell(p),
                "Status": _severity_badge(p.get("severity", "")),
                "Reference range": p.get("reference_range") or "—",
            }
        )
    return pd.DataFrame(rows)


def _render_interpretation_section(parameters: list) -> None:
    """Merged narrative fields: full width, wraps properly (dataframe does not)."""
    st.markdown("### Interpretation & recommendations")
    st.caption("Open each test to read the full text—tables here cut long sentences.")

    if not parameters:
        return

    for p in parameters:
        param_name = p.get("parameter") or "Test"
        value_str = _format_value_cell(p)
        status = _severity_badge(p.get("severity", ""))
        explanation = (p.get("explanation") or "").strip() or "—"
        recommendation = (p.get("recommendation") or "").strip() or "—"

        title = f"{status} · {param_name}: {value_str}"
        with st.expander(title, expanded=False):
            st.markdown(f"**What it means**\n\n{explanation}")
            st.markdown(f"**Recommendation**\n\n{recommendation}")


def _render_result_dashboard(result: dict) -> None:
    parameters = result.get("parameters") or []
    if not parameters:
        st.warning("No parameters returned.")
        return

    st.markdown("### Overview")
    counts = Counter((p.get("severity") or "Unknown").strip() for p in parameters)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Normal", counts.get("Normal", 0))
    with m2:
        st.metric("Borderline", counts.get("Borderline", 0))
    with m3:
        st.metric("Abnormal", counts.get("Abnormal", 0))
    with m4:
        st.metric("Critical", counts.get("Critical", 0))

    other = sum(v for k, v in counts.items() if k not in {"Normal", "Borderline", "Abnormal", "Critical"})
    if other:
        st.caption(f"Also counted: {other} row(s) with non-standard severity labels.")

    st.markdown("### Lab panel")
    summary_df = _summary_table_dataframe(parameters)
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Test": st.column_config.TextColumn("Test", width="medium"),
            "Your value": st.column_config.TextColumn("Your value", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Reference range": st.column_config.TextColumn("Reference range", width="large"),
        },
    )

    _render_interpretation_section(parameters)

    st.markdown("### Overall summary")
    summary = (result.get("overall_summary") or "").strip()
    if summary:
        st.success(summary)
    else:
        st.info("No overall summary was returned.")


def main():
    st.set_page_config(page_title="LabSense", layout="wide")

    # ── Constrain content width ───────────────────────────────────────────────
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 960px;
            padding-left: 2rem;
            padding-right: 2rem;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # ─────────────────────────────────────────────────────────────────────────

    st.title("LabSense — Lab Report Interpreter")
    st.caption(
        "Educational interpretation only — not medical advice. Discuss all results with a qualified clinician."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("History")
        if not st.session_state.history:
            st.caption("Runs will appear here.")
        for i, item in enumerate(reversed(st.session_state.history)):
            q = item["question"].replace("\n", " ").strip()
            preview = (q[:72] + "…") if len(q) > 72 else q
            with st.expander(f"Run {i + 1}: {preview or '(empty)'}", expanded=False):
                st.text_area("Report snippet", item["question"][:2000], height=120, disabled=True, key=f"h_{i}")
                st.markdown("**Summary**")
                st.write(item.get("answer_summary", "(no summary)"))

    with st.form(key="input_form"):
        report = st.text_area("Paste lab report text here", height=300)
        col1, col2 = st.columns(2)
        with col1:
            age = st.text_input("Age (optional)")
        with col2:
            sex = st.selectbox("Sex (optional)", ["", "male", "female", "other"])

        notes = st.text_input("Notes (optional)")
        run_btn = st.form_submit_button("Run LabSense", type="primary")

    if run_btn:
        if not report.strip():
            st.error("Please paste a lab report before running.")
            return

        patient_context = {"age": age or None, "sex": sex or None, "notes": notes or None}

        with st.spinner("Running LabSense..."):
            try:
                result = run_agent(report, patient_context)
            except Exception as e:
                st.error(f"Error: {e}")
                return

        st.divider()
        _render_result_dashboard(result)

        # ── RAG Sources ───────────────────────────────────────────────────────
        sources = result.get("_retrieved_sources", [])
        if sources:
            unique_names = sorted({s["source"] for s in sources})
            with st.expander("📚 KB Resources", expanded=False):
                for name in unique_names:
                    st.markdown(f"- `{name}`")
        # ─────────────────────────────────────────────────────────────────────

        st.session_state.history.append({"question": report, "answer_summary": result.get("overall_summary", "")})


if __name__ == "__main__":
    main()
