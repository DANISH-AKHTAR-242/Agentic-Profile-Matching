"""Unit tests for requirement extraction."""

from tools.requirement_extractor import extract_requirements


def test_extract_requirements_from_query() -> None:
    output = extract_requirements("Find React developers with 5 years of experience and AWS exposure")
    assert output["min_experience"] == 5.0
    assert "react" in output["must_have"]
    assert "amazon web services" in output["must_have"] or "amazon web services" in output["nice_to_have"]

