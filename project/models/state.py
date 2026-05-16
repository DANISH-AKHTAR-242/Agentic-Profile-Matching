"""Typed state models for LangGraph and service boundaries."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class CandidateRecord(BaseModel):
    """Candidate data persisted in retrieval and ranking layers."""

    candidate_id: str
    candidate_name: str
    skills: list[str] = Field(default_factory=list)
    experience: float = 0.0
    education: str = ""
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    semantic_similarity: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    resume_path: str = ""
    resume_text: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendation: str = ""
    risk_indicators: list[str] = Field(default_factory=list)
    interview_focus_areas: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    """Workflow state carried between LangGraph nodes."""

    conversation_history: list[dict[str, str]]
    current_query: str
    job_requirements: dict[str, Any]
    retrieved_candidates: list[dict[str, Any]]
    shortlisted_candidates: list[dict[str, Any]]
    ranking_reasoning: dict[str, Any]
    user_feedback: list[str]
    final_report: dict[str, Any]
    intent: str
    selected_candidate_ids: list[str]
    rerank_changes: list[dict[str, Any]]


def default_agent_state() -> AgentState:
    """Return a clean default state."""
    return AgentState(
        conversation_history=[],
        current_query="",
        job_requirements={},
        retrieved_candidates=[],
        shortlisted_candidates=[],
        ranking_reasoning={},
        user_feedback=[],
        final_report={},
        intent="search",
        selected_candidate_ids=[],
        rerank_changes=[],
    )

