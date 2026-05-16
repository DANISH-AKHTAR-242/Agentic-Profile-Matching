"""Deterministic candidate scoring and reranking."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field

from utils.synonyms import expand_skill_variants, normalize_skill, normalize_skills


class RankingWeights(BaseModel):
    """Ranking component weights."""

    semantic_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    skill_match_score: float = Field(default=0.3, ge=0.0, le=1.0)
    experience_match_score: float = Field(default=0.1, ge=0.0, le=1.0)
    education_match_score: float = Field(default=0.1, ge=0.0, le=1.0)

    def normalized(self) -> "RankingWeights":
        total = (
            self.semantic_similarity
            + self.skill_match_score
            + self.experience_match_score
            + self.education_match_score
        )
        if total == 0:
            return RankingWeights()
        return RankingWeights(
            semantic_similarity=self.semantic_similarity / total,
            skill_match_score=self.skill_match_score / total,
            experience_match_score=self.experience_match_score / total,
            education_match_score=self.education_match_score / total,
        )


def _skill_present(candidate_skill_set: set[str], required_skill: str) -> bool:
    variants = expand_skill_variants(required_skill)
    return any(variant in candidate_skill_set for variant in variants)


def score_skill_match(
    candidate_skills: list[str],
    required_skills: list[str],
    priority_skills: list[str] | None = None,
) -> float:
    """Compute skill match using synonym-aware matching."""
    normalized_candidate = {normalize_skill(skill) for skill in candidate_skills}
    required = normalize_skills(required_skills)
    if not required:
        return 1.0

    matched = sum(1 for skill in required if _skill_present(normalized_candidate, skill))
    base = matched / max(len(required), 1)

    priority = normalize_skills(priority_skills or [])
    if not priority:
        return min(base, 1.0)

    priority_hit = sum(1 for skill in priority if _skill_present(normalized_candidate, skill))
    priority_bonus = 0.15 * (priority_hit / max(len(priority), 1))
    return min(base + priority_bonus, 1.0)


def score_experience_match(candidate_experience: float, required_experience: float) -> float:
    if required_experience <= 0:
        return 1.0
    return min(max(candidate_experience, 0.0) / required_experience, 1.0)


def score_education_match(candidate_education: str, required_education: str) -> float:
    if not required_education:
        return 1.0
    candidate = candidate_education.lower()
    required = required_education.lower()
    if required in candidate:
        return 1.0
    tiers = {"bachelors": 1, "masters": 2, "phd": 3}
    candidate_tier = tiers.get(candidate, 0)
    required_tier = tiers.get(required, 0)
    return 1.0 if candidate_tier >= required_tier and required_tier > 0 else 0.0


def compute_candidate_score(
    candidate: dict[str, Any],
    requirements: dict[str, Any],
    weights: RankingWeights,
) -> tuple[float, dict[str, float]]:
    """Return final score and component breakdown."""
    normalized_weights = weights.normalized()
    semantic_similarity = float(candidate.get("semantic_similarity", 0.0))
    skill_match = score_skill_match(
        candidate_skills=candidate.get("skills", []),
        required_skills=requirements.get("must_have", []),
        priority_skills=requirements.get("priority_skills", []),
    )
    experience_match = score_experience_match(
        candidate_experience=float(candidate.get("experience", 0.0)),
        required_experience=float(requirements.get("min_experience", 0.0)),
    )
    education_match = score_education_match(
        candidate_education=str(candidate.get("education", "")),
        required_education=str(requirements.get("education", "")),
    )
    final_score = (
        normalized_weights.semantic_similarity * semantic_similarity
        + normalized_weights.skill_match_score * skill_match
        + normalized_weights.experience_match_score * experience_match
        + normalized_weights.education_match_score * education_match
    )
    breakdown = {
        "semantic_similarity": round(semantic_similarity, 4),
        "skill_match_score": round(skill_match, 4),
        "experience_match_score": round(experience_match, 4),
        "education_match_score": round(education_match, 4),
    }
    return round(final_score, 6), breakdown


def rerank_candidates(
    candidates: list[dict[str, Any]],
    requirements: dict[str, Any],
    weights: RankingWeights,
) -> list[dict[str, Any]]:
    """Apply deterministic reranking with stable tie-breaking."""
    ranked = []
    for candidate in candidates:
        candidate_copy = deepcopy(candidate)
        score, breakdown = compute_candidate_score(candidate_copy, requirements, weights)
        candidate_copy["final_score"] = score
        candidate_copy["score_breakdown"] = breakdown
        ranked.append(candidate_copy)

    ranked.sort(
        key=lambda item: (
            -float(item.get("final_score", 0.0)),
            -float(item.get("semantic_similarity", 0.0)),
            str(item.get("candidate_name", "")).lower(),
        )
    )
    for idx, candidate in enumerate(ranked, start=1):
        candidate["rank"] = idx
    return ranked

