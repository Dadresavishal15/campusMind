# CampusMind 🎓

> **RAG-Based Student Experience Intelligence System**  
> Collects, analyzes, and surfaces actionable insights from campus feedback using Retrieval-Augmented Generation.

---

## Overview

CampusMind is an end-to-end NLP pipeline that:
1. **Ingests** anonymized student feedback across 8 departments and 8 categories
2. **Embeds** feedback using a local sentence-transformer model (no API key needed)
3. **Indexes** embeddings in a FAISS vector store for millisecond-level semantic search
4. **Generates** answers, summaries, and recommendations via Google Gemini (RAG)
5. **Presents** everything in a 3-page Streamlit dashboard

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CampusMind                              │
│                                                                 │
│  data/synthetic_feedback.csv                                    │
│          │                                                      │
│          ▼                                                      │
│  src/cleaning.py ──► anonymize, normalize, validate             │
│          │                                                      │
│          ▼                                                      │
│  src/embeddings.py ──► sentence-transformers (all-MiniLM-L6-v2)│
│          │                  ──► FAISS IndexFlatL2               │
│          │                                                      │
│          ▼                                                      │
│  src/retrieval.py ──► semantic search (top-k results)           │
│          │                                                      │
│          ▼                                                      │
│  src/generation.py ──► Google Gemini API (RAG prompt)           │
│          │                                                      │
│          ▼                                                      │
│  app.py (Streamlit) ──► Overview | Ask & Search | Reports       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
campusmind/
├── data/
│   ├── generate_synthetic.py    # Script to create synthetic_feedback.csv
│   ├── synthetic_feedback.csv   # 400-row pre-generated dataset
│   └── raw/                     # Raw uploads (gitignored)
├── notebooks/
│   └── 01_explore.ipynb         # EDA notebook stub
├── src/
│   ├── __init__.py
│   ├── cleaning.py              # Load, anonymize, normalize
│   ├── embeddings.py            # Embedding + FAISS index
│   ├── retrieval.py             # Semantic search
│   ├── generation.py            # Gemini RAG pipeline
│   └── evaluation.py            # Retrieval + generation eval
├── tests/
│   ├── test_cleaning.py
│   └── test_retrieval.py
├── app.py                       # Streamlit UI
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd campusmind
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and paste your Gemini API key
```

Get a free Gemini API key at https://aistudio.google.com/

### 3. Generate the synthetic dataset

```bash
python data/generate_synthetic.py
```

### 4. Build the FAISS index

```python
from src.embeddings import build_and_save
import pandas as pd

df = pd.read_csv("data/synthetic_feedback.csv")
build_and_save(df, index_path="data/feedback.index", meta_path="data/metadata.pkl")
```

Or simply launch the app — it builds the index automatically on first run.

---

## Running the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Pages

| Page | Description |
|------|-------------|
| **Overview** | Feedback count, category bar chart, severity pie chart, top themes |
| **Ask & Search** | Natural-language Q&A powered by RAG + Gemini |
| **Reports** | Per-category AI summaries and recommended actions |

---

## Running Evaluation

```bash
python -c "
from src.evaluation import run_full_eval
results = run_full_eval(
    index_path='data/feedback.index',
    meta_path='data/metadata.pkl',
    eval_path='data/eval_questions.json'
)
print(results)
"
```

---

## Limitations & Ethical Notes

- **Synthetic data only**: The bundled CSV is entirely synthetic. No real student PII is present.
- **Anonymization**: The cleaning module strips emails, phone numbers, and student IDs via regex. Always verify before ingesting real data.
- **Gemini dependency**: The generation module requires a valid `GEMINI_API_KEY`. Without it, the app falls back gracefully.
- **FAISS**: Uses an exact L2 index (`IndexFlatL2`). For production with >100k documents, switch to an approximate index (`IndexIVFFlat`).
- **Bias**: LLM-generated summaries may reflect biases in training data. Outputs should be reviewed by administrators before action.
- **Privacy**: Never ingest feedback containing student names, email addresses, or roll numbers into the RAG system.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data processing | `pandas`, `faker` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector search | `faiss-cpu` |
| LLM | `google-generativeai` (Gemini 1.5 Flash) |
| UI | `streamlit` |
| Env management | `python-dotenv` |
| Testing | `pytest` |
