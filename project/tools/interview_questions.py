"""Interview question generation based on candidate profile gaps."""

from __future__ import annotations

from typing import Any


def generate_interview_questions(
    candidate_id: str,
    candidates: list[dict[str, Any]],
    requirements: dict[str, Any],
) -> list[str]:
    """Generate targeted interview questions for one candidate."""
    candidate = next((item for item in candidates if str(item.get("candidate_id")) == str(candidate_id)), None)
    if not candidate:
        return ["Candidate not found in current shortlist."]

    must_have = requirements.get("must_have", [])
    missing = [skill for skill in must_have if skill not in candidate.get("skills", [])]
    strengths = candidate.get("strengths", [])
    projects = candidate.get("projects", [])
    questions: list[str] = []

    for skill in candidate.get("skills", [])[:3]:
        questions.append(f"Describe a production challenge you solved using {skill}.")

    for gap in missing[:3]:
        questions.append(f"Your resume shows limited {gap}. How would you ramp up quickly for this role?")

    if float(candidate.get("experience", 0.0)) < float(requirements.get("min_experience", 0.0)):
        questions.append(
            "This role asks for deeper experience. Walk through your most complex ownership and impact."
        )

    for project in projects[:2]:
        questions.append(f"In project '{project}', what architectural trade-offs did you make and why?")

    for strength in strengths[:2]:
        questions.append(f"You list '{strength}'. Can you provide measurable outcomes from that strength?")

    questions.append("What risks do you foresee in this role, and how would you mitigate them in your first 90 days?")
    return questions[:10]

