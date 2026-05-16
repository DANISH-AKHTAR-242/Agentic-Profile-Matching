"""LangGraph workflow orchestration for the recruitment assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from models.state import AgentState
from services.explainability_service import ExplainabilityService
from services.ranking_service import RankingService
from services.retrieval_service import RetrievalService
from tools.compare_candidates import compare_candidates as compare_candidates_tool
from tools.interview_questions import generate_interview_questions
from tools.rag_search import rag_resume_search


@dataclass(slots=True)
class WorkflowDependencies:
    """Dependency injection container for graph nodes."""

    requirement_extractor: Any
    retrieval_service: RetrievalService
    ranking_service: RankingService
    explainability_service: ExplainabilityService
    logger: Any


def _detect_intent(query: str) -> str:
    lower = query.lower()
    if "ranking difference" in lower or "rank above" in lower or "why did" in lower:
        return "compare"
    if "compare" in lower:
        return "compare"
    if "interview question" in lower or "generate question" in lower:
        return "questions"
    if any(token in lower for token in ["prioritize", "only show", "increase", "decrease", "weight"]):
        return "feedback"
    return "search"


def _extract_candidate_selection(query: str) -> list[str]:
    # Supports "compare top 3" and "compare 1 2 3".
    lower = query.lower()
    if "ranking difference" in lower or "rank above" in lower or "why did" in lower:
        return ["rank:1", "rank:2"]
    top_match = re.search(r"top\s+(\d+)", lower)
    if top_match:
        return [f"rank:{idx}" for idx in range(1, int(top_match.group(1)) + 1)]
    ranks = re.findall(r"\b(\d+)\b", query)
    return [f"rank:{rank}" for rank in ranks]


def _resolve_selected_candidate_ids(state: AgentState) -> list[str]:
    selected = state.get("selected_candidate_ids", [])
    if not selected:
        return []
    shortlisted = state.get("shortlisted_candidates", [])
    by_rank = {str(candidate.get("rank")): candidate.get("candidate_id") for candidate in shortlisted}
    resolved = []
    for item in selected:
        if item.startswith("rank:"):
            candidate_id = by_rank.get(item.replace("rank:", "").strip())
            if candidate_id:
                resolved.append(candidate_id)
        else:
            resolved.append(item)
    return resolved


def build_workflow(deps: WorkflowDependencies):
    """Construct and compile the LangGraph recruitment workflow."""
    graph = StateGraph(AgentState)

    def parse_input_node(state: AgentState) -> dict[str, Any]:
        query = state.get("current_query", "").strip()
        history = state.get("conversation_history", [])
        history = history + [{"role": "user", "content": query}]
        intent = _detect_intent(query)
        payload: dict[str, Any] = {"conversation_history": history, "intent": intent}
        if intent in {"compare", "questions"}:
            payload["selected_candidate_ids"] = _extract_candidate_selection(query)
        deps.logger.info("graph_parse_input", extra={"event": "graph_transition", "node": "parse_input"})
        return payload

    def extract_requirements_node(state: AgentState) -> dict[str, Any]:
        if state.get("intent") in {"compare", "questions"}:
            return {}
        previous = state.get("job_requirements", {})
        requirements = deps.requirement_extractor(state.get("current_query", ""), previous_requirements=previous)
        deps.logger.info(
            "graph_extract_requirements",
            extra={"event": "graph_transition", "node": "extract_requirements"},
        )
        return {"job_requirements": requirements}

    def search_resumes_node(state: AgentState) -> dict[str, Any]:
        if state.get("intent") in {"compare", "questions"} and state.get("shortlisted_candidates"):
            return {}
        if state.get("intent") == "feedback" and state.get("retrieved_candidates"):
            return {}
        candidates = rag_resume_search(
            requirements=state.get("job_requirements", {}),
            retrieval_service=deps.retrieval_service,
        )
        deps.logger.info("graph_search_resumes", extra={"event": "graph_transition", "node": "search_resumes"})
        return {"retrieved_candidates": candidates}

    def rank_candidates_node(state: AgentState) -> dict[str, Any]:
        if state.get("intent") in {"compare", "questions"} and state.get("shortlisted_candidates"):
            return {}
        ranked, reasoning = deps.ranking_service.rank_candidates(
            state.get("retrieved_candidates", []),
            state.get("job_requirements", {}),
        )
        deps.logger.info("graph_rank_candidates", extra={"event": "graph_transition", "node": "rank_candidates"})
        return {"shortlisted_candidates": ranked, "ranking_reasoning": reasoning}

    def generate_report_node(state: AgentState) -> dict[str, Any]:
        if state.get("intent") in {"compare", "questions"}:
            return {}
        report = deps.explainability_service.generate_report(
            query=state.get("current_query", ""),
            requirements=state.get("job_requirements", {}),
            candidates=state.get("shortlisted_candidates", []),
            ranking_reasoning=state.get("ranking_reasoning", {}),
            rerank_changes=state.get("rerank_changes", []),
        )
        deps.logger.info("graph_generate_report", extra={"event": "graph_transition", "node": "generate_report"})
        return {"final_report": report}

    def feedback_loop_node(state: AgentState) -> dict[str, Any]:
        if state.get("intent") != "feedback":
            return {}

        previous_shortlist = state.get("shortlisted_candidates", [])
        updated_requirements, feedback_summary = deps.ranking_service.apply_feedback(
            state.get("job_requirements", {}),
            state.get("current_query", ""),
        )
        reranked, reasoning = deps.ranking_service.rank_candidates(
            state.get("retrieved_candidates", []),
            updated_requirements,
        )
        changes = deps.explainability_service.explain_rerank_changes(previous_shortlist, reranked)
        report = deps.explainability_service.generate_report(
            query=state.get("current_query", ""),
            requirements=updated_requirements,
            candidates=reranked,
            ranking_reasoning={**reasoning, "feedback_summary": feedback_summary},
            rerank_changes=changes,
        )
        feedback_history = state.get("user_feedback", []) + [state.get("current_query", "")]
        deps.logger.info("graph_feedback_loop", extra={"event": "graph_transition", "node": "feedback_loop"})
        return {
            "job_requirements": updated_requirements,
            "shortlisted_candidates": reranked,
            "ranking_reasoning": {**reasoning, "feedback_summary": feedback_summary},
            "rerank_changes": changes,
            "final_report": report,
            "user_feedback": feedback_history,
        }

    def compare_candidates_node(state: AgentState) -> dict[str, Any]:
        selected_ids = _resolve_selected_candidate_ids(state)
        if not selected_ids:
            selected_ids = [str(c.get("candidate_id")) for c in state.get("shortlisted_candidates", [])[:3]]
        output = compare_candidates_tool(selected_ids, state.get("shortlisted_candidates", []))
        report = {
            "comparison": output.get("comparison", []),
            "summary": output.get("summary", ""),
        }
        deps.logger.info(
            "graph_compare_candidates",
            extra={"event": "graph_transition", "node": "compare_candidates"},
        )
        return {"final_report": report}

    def generate_questions_node(state: AgentState) -> dict[str, Any]:
        selected_ids = _resolve_selected_candidate_ids(state)
        if not selected_ids and state.get("shortlisted_candidates"):
            selected_ids = [str(state["shortlisted_candidates"][0].get("candidate_id"))]
        candidate_id = selected_ids[0] if selected_ids else ""
        questions = generate_interview_questions(
            candidate_id=candidate_id,
            candidates=state.get("shortlisted_candidates", []),
            requirements=state.get("job_requirements", {}),
        )
        report = {
            "candidate_id": candidate_id,
            "candidate_name": next(
                (
                    item.get("candidate_name")
                    for item in state.get("shortlisted_candidates", [])
                    if str(item.get("candidate_id")) == str(candidate_id)
                ),
                "",
            ),
            "interview_questions": questions,
        }
        deps.logger.info(
            "graph_generate_questions",
            extra={"event": "graph_transition", "node": "generate_questions"},
        )
        return {"final_report": report}

    def route_after_feedback(state: AgentState) -> str:
        intent = state.get("intent")
        if intent == "compare":
            return "compare_candidates"
        if intent == "questions":
            return "generate_questions"
        return "end"

    graph.add_node("parse_input", parse_input_node)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("feedback_loop", feedback_loop_node)
    graph.add_node("compare_candidates", compare_candidates_node)
    graph.add_node("generate_questions", generate_questions_node)

    graph.add_edge(START, "parse_input")
    graph.add_edge("parse_input", "extract_requirements")
    graph.add_edge("extract_requirements", "search_resumes")
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "generate_report")
    graph.add_edge("generate_report", "feedback_loop")
    graph.add_conditional_edges(
        "feedback_loop",
        route_after_feedback,
        {
            "compare_candidates": "compare_candidates",
            "generate_questions": "generate_questions",
            "end": END,
        },
    )
    graph.add_edge("compare_candidates", END)
    graph.add_edge("generate_questions", END)
    return graph.compile()

