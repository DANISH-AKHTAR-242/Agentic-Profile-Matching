"""RAG retrieval tool wrappers."""

from __future__ import annotations

from typing import Any

from services.retrieval_service import RetrievalService


def rag_resume_search(requirements: dict[str, Any], retrieval_service: RetrievalService) -> list[dict[str, Any]]:
    """Search top candidate resumes for given requirements."""
    return retrieval_service.search(requirements=requirements)

