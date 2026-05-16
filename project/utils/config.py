"""Application configuration using environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment with sane local defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RECRUIT_",
        extra="ignore",
    )

    project_name: str = "agentic-recruitment-assistant"
    data_dir: str = "data"
    resumes_dir: str = "data/resumes"
    job_descriptions_dir: str = "data/job_descriptions"
    chroma_dir: str = "db/chroma_db"
    chroma_collection: str = "candidate_resumes"
    session_state_path: str = "db/session_state.json"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 20
    shortlist_size: int = 10
    final_recommendation_size: int = 5
    log_level: str = "INFO"
    parser_retries: int = 2
    parser_retry_delay_seconds: float = 0.2
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    skill_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    experience_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    education_weight: float = Field(default=0.1, ge=0.0, le=1.0)

    @property
    def ranking_weights(self) -> dict[str, float]:
        total = (
            self.semantic_weight
            + self.skill_weight
            + self.experience_weight
            + self.education_weight
        )
        if total == 0:
            return {
                "semantic_similarity": 0.5,
                "skill_match_score": 0.3,
                "experience_match_score": 0.1,
                "education_match_score": 0.1,
            }
        return {
            "semantic_similarity": self.semantic_weight / total,
            "skill_match_score": self.skill_weight / total,
            "experience_match_score": self.experience_weight / total,
            "education_match_score": self.education_weight / total,
        }

    def ensure_directories(self, project_root: Path) -> None:
        """Create required directories if they do not exist."""
        for relative in [
            self.data_dir,
            self.resumes_dir,
            self.job_descriptions_dir,
            self.chroma_dir,
            "db",
        ]:
            (project_root / relative).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

