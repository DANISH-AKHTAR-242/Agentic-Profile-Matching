"""Optional FastAPI layer for programmatic access."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from matching_agent import RecruitmentAssistant

app = FastAPI(title="Agentic Recruitment Assistant API", version="1.0.0")
assistant = RecruitmentAssistant()


class QueryRequest(BaseModel):
    query: str


@app.post("/index")
def index_resumes(force_reindex: bool = False) -> dict:
    return assistant.bootstrap_index(force_reindex=force_reindex)


@app.post("/query")
def query_candidates(payload: QueryRequest) -> dict:
    return assistant.run(payload.query)


@app.post("/reset")
def reset_state() -> dict[str, str]:
    assistant.reset_state()
    return {"status": "ok"}

