"""Deterministic ranking and multi-round screening service."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from tools.scoring import RankingWeights, rerank_candidates
from utils.config import Settings
from utils.logger import get_logger
from utils.synonyms import normalize_skill, normalize_skills


class RankingService:
    """Ranking service with feedback-driven reranking logic."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger("ranking_service", settings.log_level)
        self.weights = RankingWeights(**settings.ranking_weights).normalized()

    def get_weights(self) -> dict[str, float]:
        return self.weights.model_dump()

    def _normalize_weights(self, updates: dict[str, float]) -> dict[str, float]:
        merged = self.weights.model_dump()
        merged.update(updates)
        self.weights = RankingWeights(**merged).normalized()
        return self.weights.model_dump()

    def _deep_analysis(self, candidate: dict[str, Any], requirements: dict[str, Any]) -> dict[str, Any]:
        must_have = requirements.get("must_have", [])
        candidate_text = candidate.get("resume_text", "").lower()
        projects = " ".join(candidate.get("projects", [])).lower()
        matched_skills = [skill for skill in must_have if normalize_skill(skill) in candidate.get("skills", [])]
        project_relevance = (
            sum(1 for skill in must_have if skill.lower() in projects or skill.lower() in candidate_text)
            / max(len(must_have), 1)
        )
        experience_depth = min(
            float(candidate.get("experience", 0.0)) / max(float(requirements.get("min_experience", 0.0)), 1.0),
            1.0,
        )
        technical_alignment = len(matched_skills) / max(len(must_have), 1) if must_have else 1.0
        deep_score = round((project_relevance + experience_depth + technical_alignment) / 3, 4)
        return {
            "project_relevance": round(project_relevance, 4),
            "experience_depth": round(experience_depth, 4),
            "technical_alignment": round(technical_alignment, 4),
            "deep_score": deep_score,
        }

    def _round_three_recommendation(self, candidate: dict[str, Any], requirements: dict[str, Any]) -> dict[str, Any]:
        must_have = requirements.get("must_have", [])
        candidate_skills = set(candidate.get("skills", []))
        missing = [skill for skill in must_have if normalize_skill(skill) not in candidate_skills]
        score = float(candidate.get("final_score", 0.0))
        if score >= 0.75 and len(missing) <= 1:
            recommendation = "Hire"
        elif score >= 0.55:
            recommendation = "Proceed to technical interview"
        else:
            recommendation = "No-Hire"

        risk_indicators = []
        if missing:
            risk_indicators.append(f"Missing key skills: {', '.join(missing[:3])}")
        if float(candidate.get("experience", 0.0)) < float(requirements.get("min_experience", 0.0)):
            risk_indicators.append("Experience below requested threshold")
        if not risk_indicators:
            risk_indicators.append("Low risk profile for required stack")

        focus_areas = missing[:3] or candidate.get("gaps", [])[:3]
        if not focus_areas:
            focus_areas = ["System design depth", "Ownership and delivery impact"]
        return {
            "recommendation": recommendation,
            "risk_indicators": risk_indicators,
            "interview_focus_areas": focus_areas,
        }

    @staticmethod
    def _summarize_strengths_and_gaps(
        candidate: dict[str, Any],
        requirements: dict[str, Any],
    ) -> dict[str, list[str]]:
        must_have = requirements.get("must_have", [])
        candidate_skills = set(candidate.get("skills", []))
        matched = [skill for skill in must_have if normalize_skill(skill) in candidate_skills]
        missing = [skill for skill in must_have if normalize_skill(skill) not in candidate_skills]
        strengths = [f"Matched {skill}" for skill in matched[:4]]
        if float(candidate.get("experience", 0.0)) >= float(requirements.get("min_experience", 0.0)):
            strengths.append("Experience aligns with target")
        gaps = [f"Gap in {skill}" for skill in missing[:4]]
        return {"strengths": strengths, "gaps": gaps}

    def rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        requirements: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run 3-stage ranking and screening."""
        round_one = rerank_candidates(
            candidates=candidates[: self.settings.retrieval_top_k],
            requirements=requirements,
            weights=self.weights,
        )

        round_two = []
        for candidate in round_one:
            analysis = self._deep_analysis(candidate, requirements)
            candidate_copy = deepcopy(candidate)
            candidate_copy["deep_analysis"] = analysis
            round_two.append(candidate_copy)
        round_two.sort(
            key=lambda c: (-float(c["deep_analysis"]["deep_score"]), -float(c.get("final_score", 0.0))),
        )
        round_two = round_two[: self.settings.shortlist_size]

        final_round = []
        for candidate in round_two:
            candidate_copy = deepcopy(candidate)
            candidate_copy.update(self._summarize_strengths_and_gaps(candidate_copy, requirements))
            candidate_copy.update(self._round_three_recommendation(candidate_copy, requirements))
            final_round.append(candidate_copy)
        final_round = final_round[: self.settings.final_recommendation_size]
        for idx, candidate in enumerate(final_round, start=1):
            candidate["rank"] = idx

        reasoning = {
            "weights": self.weights.model_dump(),
            "round_1_count": len(round_one),
            "round_2_count": len(round_two),
            "round_3_count": len(final_round),
        }
        self.logger.info(
            "ranking_complete",
            extra={"event": "ranking", "details": reasoning},
        )
        return final_round, reasoning

    def apply_feedback(
        self,
        requirements: dict[str, Any],
        feedback: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Update requirements and component weights from recruiter feedback."""
        updated = deepcopy(requirements)
        lower = feedback.lower().strip()
        weight_updates: dict[str, float] = {}
        feedback_actions: list[str] = []

        prioritize_match = re.search(r"prioritize\s+(.+?)\s+over\s+(.+)", lower)
        if prioritize_match:
            first_skill = normalize_skill(prioritize_match.group(1).strip())
            updated["priority_skills"] = normalize_skills(updated.get("priority_skills", []) + [first_skill])
            feedback_actions.append(f"priority_skill:{first_skill}")

        only_show_match = re.search(r"only show\s+([a-z0-9\+\#\.\- ]+?)\s+(?:candidates|profiles|engineers|developers)?$", lower)
        if only_show_match:
            skill = normalize_skill(only_show_match.group(1).strip())
            updated["must_have"] = normalize_skills(updated.get("must_have", []) + [skill])
            feedback_actions.append(f"must_have_added:{skill}")

        if "aws" in lower and "only show" in lower and "amazon web services" not in updated.get("must_have", []):
            updated["must_have"] = normalize_skills(updated.get("must_have", []) + ["amazon web services"])
            feedback_actions.append("must_have_added:amazon web services")

        if "increase backend weighting" in lower or "increase skill weighting" in lower:
            weight_updates["skill_match_score"] = self.weights.skill_match_score + 0.1
            weight_updates["semantic_similarity"] = max(self.weights.semantic_similarity - 0.1, 0.1)
            feedback_actions.append("weight_shift:skills_up")

        if "increase semantic weighting" in lower:
            weight_updates["semantic_similarity"] = self.weights.semantic_similarity + 0.1
            weight_updates["skill_match_score"] = max(self.weights.skill_match_score - 0.1, 0.1)
            feedback_actions.append("weight_shift:semantic_up")

        if "decrease experience weighting" in lower:
            weight_updates["experience_match_score"] = max(self.weights.experience_match_score - 0.05, 0.05)
            weight_updates["semantic_similarity"] = self.weights.semantic_similarity + 0.05
            feedback_actions.append("weight_shift:experience_down")

        new_weights = self._normalize_weights(weight_updates) if weight_updates else self.weights.model_dump()
        summary = {"actions": feedback_actions, "weights": new_weights}
        self.logger.info(
            "feedback_applied",
            extra={"event": "reranking", "details": summary},
        )
        return updated, summary

