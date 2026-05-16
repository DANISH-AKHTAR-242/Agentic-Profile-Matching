# Agentic AI Recruitment Assistant (LangGraph + ChromaDB)

Production-structured local recruitment assistant that extends RAG resume matching into an agentic workflow with:

- conversational memory
- explainable ranking
- 3-round screening
- feedback-driven reranking
- candidate comparison
- interview question generation
- CLI + Streamlit interfaces

## Tech Stack

- Python 3.11+
- LangGraph / LangChain
- ChromaDB (persistent)
- sentence-transformers (`all-MiniLM-L6-v2`)
- Streamlit
- PyMuPDF + python-docx
- Pydantic
- FastAPI-ready dependencies (optional API layer)

## Project Structure

```text
project/
├── data/
│   ├── resumes/
│   └── job_descriptions/
├── db/
│   └── chroma_db/
├── graphs/
│   └── workflow.py
├── tools/
│   ├── rag_search.py
│   ├── requirement_extractor.py
│   ├── compare_candidates.py
│   ├── interview_questions.py
│   ├── scoring.py
│   ├── resume_parser.py
│   └── embeddings.py
├── ui/
│   ├── cli.py
│   └── streamlit_app.py
├── models/
│   └── state.py
├── services/
│   ├── ranking_service.py
│   ├── retrieval_service.py
│   └── explainability_service.py
├── utils/
│   ├── logger.py
│   ├── config.py
│   └── synonyms.py
├── scripts/
│   ├── init_chroma.py
│   ├── generate_sample_resumes.py
│   └── test_scenarios.py
├── tests/
├── matching_agent.py
├── requirements.txt
├── README.md
└── notebook.ipynb
```

## Workflow (LangGraph)

Default path:

```text
START
  → parse_input
  → extract_requirements
  → search_resumes
  → rank_candidates
  → generate_report
  → feedback_loop
  → END
```

Additional paths (from feedback loop intent routing):

- `compare_candidates`
- `generate_questions`

## Setup

1. Create and activate venv.
2. Install dependencies.
3. Generate sample resumes (optional).
4. Initialize ChromaDB.
5. Run CLI or Streamlit.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\generate_sample_resumes.py
python scripts\init_chroma.py
python ui\cli.py --index
```

Streamlit:

```bash
streamlit run ui\streamlit_app.py
```

## CLI Usage

```bash
python ui\cli.py
```

Examples:

- `Find Python developers with 5+ years experience`
- `Compare top 3 candidates`
- `Generate interview questions for candidate 1`
- `Prioritize React over Python`
- `Only show AWS candidates`
- `Increase backend weighting`

## Example Recruiter Conversations

1. **Search**
   - Recruiter: `Find Python developers with 5+ years experience`
   - Assistant: returns ranked shortlist with strengths, gaps, and recommendations.

2. **Comparison**
   - Recruiter: `Compare top 3 candidates`
   - Assistant: returns side-by-side comparison of skills, experience, strengths, weaknesses.

3. **Explain ranking difference**
   - Recruiter: `Explain ranking differences`
   - Assistant: compares top candidates and explains score/rank deltas.

4. **Feedback reranking**
   - Recruiter: `Only show AWS candidates`
   - Assistant: updates requirements, reranks shortlist, and reports rank changes.

5. **Interview planning**
   - Recruiter: `Generate interview questions for candidate 1`
   - Assistant: generates targeted technical and risk-focused interview questions.

## Configuration

Use environment variables with prefix `RECRUIT_`:

- `RECRUIT_EMBEDDING_MODEL_NAME`
- `RECRUIT_CHROMA_DIR`
- `RECRUIT_CHROMA_COLLECTION`
- `RECRUIT_RETRIEVAL_TOP_K`
- `RECRUIT_SHORTLIST_SIZE`
- `RECRUIT_FINAL_RECOMMENDATION_SIZE`
- `RECRUIT_LOG_LEVEL`
- `RECRUIT_SEMANTIC_WEIGHT`
- `RECRUIT_SKILL_WEIGHT`
- `RECRUIT_EXPERIENCE_WEIGHT`
- `RECRUIT_EDUCATION_WEIGHT`

## Test Scripts

End-to-end scenarios:

```bash
python scripts\test_scenarios.py
```

Unit tests:

```bash
pytest -q
```

## Optional API Layer

```bash
uvicorn api:app --reload
```

## Notes

- ChromaDB is persisted under `db/chroma_db`.
- Embeddings are cached in-process for query speed.
- Resume parser supports PDF, DOCX, and TXT.
- State is persisted under `db/session_state.json` for conversation continuity.
