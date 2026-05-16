"""Command-line interface for the recruitment assistant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching_agent import RecruitmentAssistant  # noqa: E402


def _print_report(report: dict) -> None:
    if "top_matches" in report:
        print("\nTop Matches:")
        for idx, candidate in enumerate(report.get("top_matches", []), start=1):
            print(
                f"{idx}. {candidate.get('candidate_name')} | "
                f"Score: {candidate.get('match_score')} | "
                f"Recommendation: {candidate.get('recommendation')}"
            )
            if candidate.get("strengths"):
                print(f"   Strengths: {', '.join(candidate.get('strengths', [])[:2])}")
            if candidate.get("gaps"):
                print(f"   Gaps: {', '.join(candidate.get('gaps', [])[:2])}")
        rerank_changes = report.get("rerank_changes", [])
        if rerank_changes:
            print("\nRerank changes:")
            for change in rerank_changes:
                print(
                    f"- {change.get('candidate_name')}: "
                    f"{change.get('from_rank')} -> {change.get('to_rank')}"
                )
        return

    if "comparison" in report:
        print("\nCandidate Comparison:")
        for item in report.get("comparison", []):
            print(
                f"- {item.get('candidate_name')} | Rank {item.get('rank')} | "
                f"Score {item.get('match_score')}"
            )
        print(f"Summary: {report.get('summary')}")
        return

    if "interview_questions" in report:
        print(f"\nInterview Questions for {report.get('candidate_name') or report.get('candidate_id')}:")
        for idx, question in enumerate(report.get("interview_questions", []), start=1):
            print(f"{idx}. {question}")
        return

    print(json.dumps(report, indent=2))


def run_interactive(agent: RecruitmentAssistant) -> None:
    print("AI Recruitment Assistant CLI")
    print("Type your recruiter query, or use commands: /help, /index, /reset, /exit")
    while True:
        query = input("\nrecruiter> ").strip()
        if not query:
            continue
        if query in {"/exit", "/quit"}:
            break
        if query == "/help":
            print(
                "Examples:\n"
                "- Find Python developers with 5+ years and AWS exposure\n"
                "- Compare top 3 candidates\n"
                "- Generate interview questions for candidate 1\n"
                "- Prioritize React over Python\n"
            )
            continue
        if query == "/index":
            result = agent.bootstrap_index(force_reindex=False)
            print(f"Indexing complete: {result}")
            continue
        if query == "/reset":
            agent.reset_state()
            print("Conversation state reset.")
            continue

        report = agent.run(query)
        _print_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic recruitment assistant CLI")
    parser.add_argument("--query", type=str, default="", help="Single-shot recruiter query")
    parser.add_argument("--index", action="store_true", help="Run indexing before querying")
    parser.add_argument("--force-reindex", action="store_true", help="Reindex all resumes")
    parser.add_argument("--reset-state", action="store_true", help="Clear persisted conversation state")
    args = parser.parse_args()

    agent = RecruitmentAssistant()
    if args.reset_state:
        agent.reset_state()
    if args.index:
        result = agent.bootstrap_index(force_reindex=args.force_reindex)
        print(f"Indexing complete: {result}")

    if args.query:
        _print_report(agent.run(args.query))
        return
    run_interactive(agent)


if __name__ == "__main__":
    main()

