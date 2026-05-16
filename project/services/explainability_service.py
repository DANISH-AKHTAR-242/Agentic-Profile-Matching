"""Explainable ranking output generation."""

from __future__ import annotations

from typing import Any


class ExplainabilityService:
    """Builds transparent explanations and report payloads."""

    @staticmethod
    def build_candidate_explanation(candidate: dict[str, Any], requirements: dict[str, Any]) -> dict[str, Any]:
        must_have = requirements.get("must_have", [])
        candidate_skills = set(candidate.get("skills", []))
        matched_skills = [skill for skill in must_have if skill in candidate_skills]
        missing_skills = [skill for skill in must_have if skill not in candidate_skills]
        strengths = [f"Strong evidence of {skill}" for skill in matched_skills[:4]]
        if float(candidate.get("experience", 0.0)) >= float(requirements.get("min_experience", 0.0)):
            strengths.append("Meets or exceeds experience threshold")

        gaps = [f"Limited evidence for {skill}" for skill in missing_skills[:4]]
        if requirements.get("education") and requirements.get("education", "").lower() not in str(
            candidate.get("education", "")
        ).lower():
            gaps.append(f"Education preference mismatch: expected {requirements.get('education')}")

        breakdown = candidate.get("score_breakdown", {})
        reasoning = (
            f"Semantic={breakdown.get('semantic_similarity', 0):.2f}, "
            f"Skill={breakdown.get('skill_match_score', 0):.2f}, "
            f"Experience={breakdown.get('experience_match_score', 0):.2f}, "
            f"Education={breakdown.get('education_match_score', 0):.2f}."
        )
        return {
            "candidate_name": candidate.get("candidate_name", ""),
            "match_score": round(float(candidate.get("final_score", 0.0)) * 100, 2),
            "strengths": strengths or ["General semantic alignment with role requirements"],
            "gaps": gaps or ["No major gaps detected in extracted profile"],
            "matched_skills": matched_skills,
            "reasoning": reasoning,
            "risk_indicators": candidate.get("risk_indicators", []),
            "suggested_interview_topics": candidate.get("interview_focus_areas", []),
            "recommendation": candidate.get("recommendation", "Review"),
        }

    def explain_rerank_changes(
        self,
        previous: list[dict[str, Any]],
        current: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Describe rank movements between two ordered lists."""
        old_rank = {c.get("candidate_id"): c.get("rank") for c in previous}
        changes = []
        for candidate in current:
            cid = candidate.get("candidate_id")
            old = old_rank.get(cid)
            new = candidate.get("rank")
            if old and new and old != new:
                changes.append(
                    {
                        "candidate_id": cid,
                        "candidate_name": candidate.get("candidate_name"),
                        "from_rank": old,
                        "to_rank": new,
                        "delta": old - new,
                    }
                )
        return changes

    def generate_report(
        self,
        query: str,
        requirements: dict[str, Any],
        candidates: list[dict[str, Any]],
        ranking_reasoning: dict[str, Any],
        rerank_changes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build final report payload consumed by CLI and Streamlit UI."""
        explanations = [self.build_candidate_explanation(candidate, requirements) for candidate in candidates]
        return {
            "job_description": query,
            "requirements": requirements,
            "top_matches": explanations,
            "ranking_reasoning": ranking_reasoning,
            "rerank_changes": rerank_changes or [],
        }

