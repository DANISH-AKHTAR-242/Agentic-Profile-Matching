"""Candidate side-by-side comparison tool."""

from __future__ import annotations

from typing import Any


def compare_candidates(candidate_ids: list[str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare selected candidates and return structured analysis."""
    if not candidate_ids:
        return {"comparison": [], "summary": "No candidate IDs supplied."}

    by_id = {str(candidate.get("candidate_id")): candidate for candidate in candidates}
    selected = [by_id[candidate_id] for candidate_id in candidate_ids if candidate_id in by_id]
    if not selected:
        return {"comparison": [], "summary": "No matching candidates found in shortlist."}

    comparison = []
    for candidate in selected:
        comparison.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "candidate_name": candidate.get("candidate_name"),
                "rank": candidate.get("rank"),
                "match_score": round(float(candidate.get("final_score", 0.0)) * 100, 2),
                "skills": candidate.get("skills", []),
                "experience": candidate.get("experience", 0.0),
                "projects": candidate.get("projects", []),
                "strengths": candidate.get("strengths", []),
                "weaknesses": candidate.get("gaps", []),
                "recommendation": candidate.get("recommendation", ""),
            }
        )

    winner = sorted(comparison, key=lambda x: (-x["match_score"], str(x["candidate_name"]).lower()))[0]
    summary = (
        f"{winner['candidate_name']} leads with score {winner['match_score']} "
        f"based on stronger skill and experience alignment."
    )

    return {"comparison": comparison, "summary": summary}

