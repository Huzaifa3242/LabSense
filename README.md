# LabSense — AI-Powered Lab Report Interpreter

LabSense interprets medical laboratory reports using a **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain, Google Gemini, ChromaDB, and Streamlit.

---

## System Overview

LabSense combines **NLP-based parameter extraction**, **semantic retrieval from a curated medical knowledge base**, and a **large language model** to produce grounded, structured interpretations of lab reports.

The key idea is that instead of relying purely on LLM memory (which can hallucinate), LabSense first searches its own local Knowledge Base (KB) for trusted reference material and injects those passages into the prompt before calling the LLM. This is the **RAG (Retrieval-Augmented Generation)** pattern.

```
User pastes lab report
         │
         ▼
   Parameter Extraction  ←─ regex NLP parser (nlp_utils.py)
         │
         ▼
   Vector Search         ←─ ChromaDB + sentence-transformers embeddings
         │
         ▼
   Prompt Construction   ←─ LangChain (KB passages injected as context)
         │
         ▼
   LLM Inference         ←─ Google Gemini
         │
         ▼
   Structured JSON Output (severity, range, explanation, recommendation)
         │
         ▼
   Streamlit Dashboard   ←─ table + expandable details + KB Sources
```

---

## What It Does

1. **Paste a lab report** (plain text) into the web UI.
2. **Optionally enter** patient age, sex, and notes for contextualised interpretation.
3. The app **extracts lab parameters** (e.g., `Hemoglobin: 9.2 g/dL`) using a regex-based NLP parser.
4. It **retrieves relevant reference passages** from a local medical knowledge base (KB) using semantic search.
5. Those passages are fed to a **Gemini LLM** via LangChain, which returns structured JSON with:
   - Severity classification: `Normal | Borderline | Abnormal | Critical`
   - Reference range for each parameter
   - Plain-language explanation
   - Clinical recommendation
   - An overall patient summary
6. Results are displayed in a clean **Streamlit dashboard** with a summary table, per-test expandable interpretations, and a **📚 KB Resources** expander showing which knowledge base files were used.

---

## Prerequisites

Make sure the following are installed before running LabSense:

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| uv (package manager) | latest |
| Google Gemini API Key | required |
| Internet connection | for model inference |

Install `uv` if not already installed:
```powershell
pip install uv
```

---

## Project Structure

```
LabSense/
│
├── app/
│   ├── __init__.py      ← makes app/ a Python package
│   ├── webapp.py        ← Streamlit UI (entry point)
│   ├── agent.py         ← Orchestration: extract → RAG → LLM → parse
│   ├── prompts.py       ← LangChain prompt templates (RAG-aware)
│   ├── nlp_utils.py     ← Regex-based parameter extractor & text cleaner
│   └── llm_client.py    ← LLM initialisation (Google Gemini via LangChain)
│
├── kb/                  ← Knowledge Base (plain-text or PDF files)
│   ├── hemoglobin_anemia.txt
│   ├── glucose_diabetes.txt
│   ├── lipid_profile.txt
│   └── thyroid_tsh.txt
│
├── rag_store.py         ← Chroma vector store builder & retriever
├── chroma_db/           ← Auto-generated vector database (built at runtime)
├── main.py              ← Root launcher for Streamlit
├── requirements.txt     ← Python dependencies
├── pyproject.toml       ← uv project config
├── .env                 ← API keys (never commit this file)
└── README.md            ← This file
```

---

## Setup & Installation

### Step 1 — Clone or open the project
```powershell
cd C:\Users\<you>\Desktop\LabSense
```

### Step 2 — Create and activate virtual environment
```powershell
uv venv
.venv\Scripts\activate
```

### Step 3 — Install dependencies
```powershell
uv add -r requirements.txt
```

### Step 4 — Configure environment variables
Create a `.env` file in the project root with your Gemini API key:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

> Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### Step 5 — Build the vector store (first time only)
This downloads the embedding model and indexes all KB files into ChromaDB:
```powershell
python -c "from rag_store import build_vector_store; build_vector_store()"
```
You will see: `✅ Vector store built with N chunks.`

### Step 6 — Launch the app
```powershell
uv run streamlit run app/webapp.py
```
Open your browser at **http://localhost:8501**

---

## Usage Instructions

### Basic Usage

1. **Open** http://localhost:8501 in your browser.
2. **Paste** the full lab report text into the text area (plain text format).
3. **Fill in** optional patient details:
   - Age (e.g., `45`)
   - Sex (`male` / `female` / `other`)
   - Notes (e.g., `patient is diabetic, on metformin`)
4. Click **Run LabSense**.
5. View the results:
   - **Overview metrics** — count of Normal / Borderline / Abnormal / Critical results
   - **Lab panel table** — all parameters with values and severity badges
   - **Interpretation & recommendations** — click each test to expand
   - **Overall summary** — plain-language summary for the full report
   - **📚 KB Resources** — click to see which KB files were used for grounding

### Supported Lab Report Formats

The parser recognises lines in these formats:
```
Hemoglobin: 9.2 g/dL
Glucose (Fasting): 152 mg/dL
WBC: 13.4 x10^9/L
TSH = 0.3 µIU/mL
Platelets 120 x10^9/L
```

### Sample Lab Report (for testing)
```
Hemoglobin: 9.2 g/dL
Hematocrit: 29.5 %
WBC: 13.4 x10^9/L
Platelets: 120 x10^9/L
Glucose (Fasting): 152 mg/dL
Sodium: 138 mmol/L
Creatinine: 1.5 mg/dL
TSH: 0.3 µIU/mL
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | Google Gemini API key for LLM inference |
| `LABSENSE_PARAMS_PER_BATCH` | ❌ Optional | Max parameters per LLM batch call (default: `6`) |
| `LABSENSE_SINGLE_CALL_MAX_CHARS` | ❌ Optional | Character limit before batching kicks in (default: `4500`) |

---

## RAG Pipeline — How Knowledge Grounding Works

When a lab report is submitted:

1. **Parameter names** are extracted (e.g., "Hemoglobin", "Glucose", "TSH").
2. A **semantic query** is built from those names and sent to ChromaDB.
3. The **top-4 most relevant passages** from the KB are retrieved.
4. Those passages are injected into the LangChain prompt as `{rag_context}`.
5. The LLM reads the KB passages **first**, then generates its response grounded in that context.
6. If a test is **not covered** by the KB, the LLM falls back to standard clinical references (Harrison's Principles, WHO guidelines) — it never leaves a reference range blank.

---

## Expanding the Knowledge Base

To add coverage for more lab tests:

1. Create a `.txt` or `.pdf` file in `kb/` (e.g., `kb/wbc_leukocytes.txt`)
2. Include: normal ranges, interpretation notes, and a disclaimer
3. Rebuild the vector store:
   ```powershell
   python -c "from rag_store import build_vector_store; build_vector_store()"
   ```
4. Restart the app — the new content is immediately available to RAG

Suggested files to add:
- `wbc_leukocytes.txt` — WBC, neutrophils, lymphocytes
- `platelets_coagulation.txt` — Platelets, bleeding time
- `kidney_function.txt` — BUN, Creatinine, GFR
- `electrolytes.txt` — Sodium, Potassium, Chloride, Bicarbonate
- `liver_function.txt` — ALT, AST, bilirubin

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini (via `langchain-google-genai`) |
| RAG Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| NLP Parsing | Python `re` (regex) |
| Package Manager | uv |
| Language | Python 3.11 |

---

## Disclaimer

LabSense is an **educational project** built for a university AI assignment. It does **not** provide medical advice and must **not** be used for clinical decision-making. Always consult a qualified healthcare professional for the interpretation of laboratory results.
