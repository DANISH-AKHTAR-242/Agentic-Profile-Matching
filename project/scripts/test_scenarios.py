"""Run required conversation test scenarios end-to-end."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching_agent import RecruitmentAssistant  # noqa: E402

SCENARIOS = [
    "Find Python developers with 5+ years experience",
    "Compare top 3 candidates",
    "Explain ranking differences",
    "Prioritize AWS skills",
    "Generate interview questions for candidate 1",
]


def main() -> None:
    agent = RecruitmentAssistant()
    agent.bootstrap_index(force_reindex=False)

    for scenario in SCENARIOS:
        print(f"\n=== Scenario: {scenario} ===")
        report = agent.run(scenario)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

