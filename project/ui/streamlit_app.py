"""Streamlit chat UI for the recruitment assistant."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching_agent import RecruitmentAssistant  # noqa: E402

st.set_page_config(page_title="AI Recruitment Assistant", layout="wide")
st.title("Agentic AI Recruitment Assistant")

if "assistant" not in st.session_state:
    st.session_state.assistant = RecruitmentAssistant()
if "messages" not in st.session_state:
    st.session_state.messages = []

assistant: RecruitmentAssistant = st.session_state.assistant

with st.sidebar:
    st.subheader("Actions")
    if st.button("Index resumes"):
        stats = assistant.bootstrap_index(force_reindex=False)
        st.success(f"Indexed resumes: {stats}")
    if st.button("Reset conversation"):
        assistant.reset_state()
        st.session_state.messages = []
        st.info("Session reset.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask for candidates, comparisons, reranking, or interview questions")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    report = assistant.run(prompt)

    with st.chat_message("assistant"):
        if "top_matches" in report:
            matches = report.get("top_matches", [])
            summary = f"Found {len(matches)} shortlisted candidates."
            st.markdown(summary)
            if matches:
                df = pd.DataFrame(
                    [
                        {
                            "Candidate": candidate.get("candidate_name"),
                            "Match Score": candidate.get("match_score"),
                            "Recommendation": candidate.get("recommendation"),
                        }
                        for candidate in matches
                    ]
                )
                st.dataframe(df, use_container_width=True)
                for candidate in matches:
                    with st.expander(f"{candidate.get('candidate_name')} details"):
                        st.write("**Strengths:**", candidate.get("strengths", []))
                        st.write("**Gaps:**", candidate.get("gaps", []))
                        st.write("**Reasoning:**", candidate.get("reasoning", ""))
                        st.write("**Interview Topics:**", candidate.get("suggested_interview_topics", []))
            if report.get("rerank_changes"):
                st.write("### Ranking Changes")
                st.table(pd.DataFrame(report["rerank_changes"]))
            assistant_text = summary
        elif "comparison" in report:
            st.write("### Candidate Comparison")
            st.table(pd.DataFrame(report.get("comparison", [])))
            st.write(report.get("summary", ""))
            assistant_text = report.get("summary", "Comparison generated.")
        elif "interview_questions" in report:
            st.write(f"### Interview Questions for {report.get('candidate_name') or report.get('candidate_id')}")
            for idx, question in enumerate(report.get("interview_questions", []), start=1):
                st.write(f"{idx}. {question}")
            assistant_text = "Interview questions generated."
        else:
            st.json(report)
            assistant_text = "Response generated."

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

