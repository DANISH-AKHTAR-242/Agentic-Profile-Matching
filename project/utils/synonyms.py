"""Skill synonym mappings and normalization helpers."""

from __future__ import annotations

from collections import defaultdict

SKILL_SYNONYMS: dict[str, list[str]] = {
    "machine learning": ["ml", "machine-learning"],
    "natural language processing": ["nlp", "natural-language-processing"],
    "javascript": ["js", "nodejs", "node.js"],
    "artificial intelligence": ["ai"],
    "amazon web services": ["aws"],
    "react": ["reactjs", "react.js"],
    "python": ["py"],
    "kubernetes": ["k8s"],
}

COMMON_SKILLS: list[str] = sorted(
    {
        "python",
        "javascript",
        "react",
        "java",
        "c++",
        "sql",
        "postgresql",
        "mongodb",
        "docker",
        "kubernetes",
        "machine learning",
        "natural language processing",
        "artificial intelligence",
        "langchain",
        "langgraph",
        "fastapi",
        "flask",
        "django",
        "azure",
        "amazon web services",
        "gcp",
        "terraform",
        "redis",
        "ci/cd",
    }
)


def build_reverse_synonym_map() -> dict[str, str]:
    """Return alias -> canonical skill mapping."""
    reverse = defaultdict(str)
    for canonical, aliases in SKILL_SYNONYMS.items():
        reverse[canonical.lower()] = canonical.lower()
        for alias in aliases:
            reverse[alias.lower()] = canonical.lower()
    return dict(reverse)


REVERSE_SKILL_MAP = build_reverse_synonym_map()


def normalize_skill(skill: str) -> str:
    """Normalize one skill to canonical form."""
    return REVERSE_SKILL_MAP.get(skill.strip().lower(), skill.strip().lower())


def normalize_skills(skills: list[str]) -> list[str]:
    """Normalize and de-duplicate skill lists."""
    seen: set[str] = set()
    normalized: list[str] = []
    for skill in skills:
        canonical = normalize_skill(skill)
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def expand_skill_variants(skill: str) -> set[str]:
    """Return canonical + aliases for matching."""
    canonical = normalize_skill(skill)
    aliases = set(SKILL_SYNONYMS.get(canonical, []))
    aliases.add(canonical)
    return {item.lower() for item in aliases}

