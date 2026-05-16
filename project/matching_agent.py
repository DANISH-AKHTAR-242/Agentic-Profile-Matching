"""Top-level recruitment assistant orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphs.workflow import WorkflowDependencies, build_workflow
from models.state import AgentState, default_agent_state
from services.explainability_service import ExplainabilityService
from services.ranking_service import RankingService
from services.retrieval_service import RetrievalService
from tools.embeddings import EmbeddingService
from tools.requirement_extractor import extract_requirements
from utils.config import Settings, get_settings
from utils.logger import get_logger


class RecruitmentAssistant:
    """Production-structured agentic recruitment assistant."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger("matching_agent", self.settings.log_level)
        self.root = Path(__file__).resolve().parent
        self.settings.ensure_directories(self.root)
        chroma_path = Path(self.settings.chroma_dir)
        if not chroma_path.is_absolute():
            self.settings.chroma_dir = str((self.root / chroma_path).resolve())

        self.embedding_service = EmbeddingService(self.settings.embedding_model_name)
        self.retrieval_service = RetrievalService(self.settings, self.embedding_service)
        self.ranking_service = RankingService(self.settings)
        self.explainability_service = ExplainabilityService()

        deps = WorkflowDependencies(
            requirement_extractor=extract_requirements,
            retrieval_service=self.retrieval_service,
            ranking_service=self.ranking_service,
            explainability_service=self.explainability_service,
            logger=self.logger,
        )
        self.workflow = build_workflow(deps)
        self.state: AgentState = self._load_state()

    @property
    def _state_file(self) -> Path:
        return self.root / self.settings.session_state_path

    def _load_state(self) -> AgentState:
        state = default_agent_state()
        if not self._state_file.exists():
            return state
        try:
            persisted = json.loads(self._state_file.read_text(encoding="utf-8"))
            state.update(persisted)
        except Exception:  # noqa: BLE001 - keep state loading resilient
            self.logger.warning("state_load_failed", extra={"event": "state", "details": "fallback_to_default"})
        return state

    def _save_state(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def reset_state(self) -> None:
        self.state = default_agent_state()
        self._save_state()

    def bootstrap_index(self, force_reindex: bool = False) -> dict[str, int]:
        """Index all resumes into ChromaDB."""
        result = self.retrieval_service.index_directory(
            directory=self.root / self.settings.resumes_dir,
            force_reindex=force_reindex,
        )
        self.logger.info("bootstrap_index_complete", extra={"event": "retrieval", "details": result})
        return result

    def run(self, query: str) -> dict[str, Any]:
        """Execute one agent turn through the LangGraph workflow."""
        input_state = default_agent_state()
        input_state.update(self.state)
        input_state["current_query"] = query

        output: AgentState = self.workflow.invoke(input_state)
        output_history = output.get("conversation_history", [])
        assistant_reply = self._summarize_for_history(output.get("final_report", {}))
        if assistant_reply:
            output_history = output_history + [{"role": "assistant", "content": assistant_reply}]
            output["conversation_history"] = output_history

        self.state = output
        self._save_state()
        return output.get("final_report", {})

    @staticmethod
    def _summarize_for_history(report: dict[str, Any]) -> str:
        if "top_matches" in report:
            matches = report.get("top_matches", [])
            if not matches:
                return "No candidates found."
            top = matches[0]
            return f"Top candidate: {top.get('candidate_name')} ({top.get('match_score')} match score)."
        if "summary" in report:
            return str(report.get("summary"))
        if "interview_questions" in report:
            return "Generated interview questions."
        return "Response generated."


def build_assistant() -> RecruitmentAssistant:
    """Factory helper for CLI/UI entrypoints."""
    return RecruitmentAssistant()

