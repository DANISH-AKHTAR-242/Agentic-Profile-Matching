"""Requirement extraction from recruiter natural-language queries."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from utils.synonyms import COMMON_SKILLS, normalize_skill, normalize_skills

EDUCATION_KEYWORDS = {
    "phd": "PhD",
    "doctorate": "PhD",
    "master": "Masters",
    "m.tech": "Masters",
    "b.tech": "Bachelors",
    "bachelor": "Bachelors",
    "bs": "Bachelors",
}

DOMAIN_KEYWORDS = [
    "fintech",
    "healthcare",
    "ecommerce",
    "edtech",
    "saas",
    "cybersecurity",
]


class Requirements(BaseModel):
    """Structured requirement output from free-form recruiter text."""

    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    min_experience: float = 0.0
    education: str = ""
    domain_preferences: list[str] = Field(default_factory=list)
    priority_skills: list[str] = Field(default_factory=list)
    raw_query: str = ""


def _extract_skills(text: str) -> list[str]:
    lower = text.lower()
    hits = []
    for skill in COMMON_SKILLS:
        variants = {skill}
        if skill == "amazon web services":
            variants.add("aws")
        if any(f" {variant} " in f" {lower} " for variant in variants):
            hits.append(normalize_skill(skill))
    return normalize_skills(hits)


def _extract_nice_to_have(lower_query: str) -> list[str]:
    segments = []
    for marker in ["nice to have", "preferred", "bonus", "plus", "exposure to"]:
        if marker in lower_query:
            start = lower_query.find(marker)
            segments.append(lower_query[start : start + 120])
    skills = []
    for segment in segments:
        skills.extend(_extract_skills(segment))
    return normalize_skills(skills)


def extract_requirements(jd: str, previous_requirements: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract requirements from recruiter queries into structured JSON."""
    query = (jd or "").strip()
    lower = query.lower()
    base = Requirements(**(previous_requirements or {}))
    base.raw_query = query

    years_match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|year)", lower)
    if years_match:
        base.min_experience = float(years_match.group(1))

    extracted_skills = _extract_skills(lower)
    nice_to_have = _extract_nice_to_have(lower)
    must_have = [skill for skill in extracted_skills if skill not in nice_to_have]

    if "only show" in lower:
        forced = _extract_skills(lower[lower.find("only show") :])
        must_have = normalize_skills(must_have + forced)

    prioritize_match = re.search(r"prioritize\s+([a-zA-Z0-9\+\#\.\- ]+?)\s+over", lower)
    if prioritize_match:
        base.priority_skills = normalize_skills([prioritize_match.group(1)])

    if must_have:
        base.must_have = normalize_skills(list(dict.fromkeys(base.must_have + must_have)))
    if nice_to_have:
        base.nice_to_have = normalize_skills(
            [skill for skill in dict.fromkeys(base.nice_to_have + nice_to_have) if skill not in base.must_have]
        )

    for key, normalized in EDUCATION_KEYWORDS.items():
        if key in lower:
            base.education = normalized
            break

    base.domain_preferences = [domain for domain in DOMAIN_KEYWORDS if domain in lower] or base.domain_preferences
    return base.model_dump()

