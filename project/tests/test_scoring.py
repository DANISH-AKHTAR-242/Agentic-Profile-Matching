"""Unit tests for deterministic scoring."""

from tools.scoring import RankingWeights, rerank_candidates


def test_rerank_candidates_is_deterministic() -> None:
    requirements = {
        "must_have": ["python", "react"],
        "min_experience": 5,
        "education": "Bachelors",
        "priority_skills": [],
    }
    candidates = [
        {
            "candidate_id": "a",
            "candidate_name": "Alice",
            "skills": ["python", "react", "amazon web services"],
            "experience": 6,
            "education": "Bachelors",
            "semantic_similarity": 0.8,
        },
        {
            "candidate_id": "b",
            "candidate_name": "Bob",
            "skills": ["python"],
            "experience": 7,
            "education": "Bachelors",
            "semantic_similarity": 0.85,
        },
    ]
    ranked = rerank_candidates(candidates, requirements, RankingWeights())
    assert ranked[0]["candidate_id"] == "a"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2

