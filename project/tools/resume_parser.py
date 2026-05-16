"""Resume parsing utilities for PDF, DOCX, and TXT content."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import fitz
from docx import Document

from utils.logger import get_logger
from utils.synonyms import COMMON_SKILLS, normalize_skills

LOGGER = get_logger("resume_parser")


class ResumeParseError(RuntimeError):
    """Raised when a resume cannot be parsed."""


def _extract_pdf_text(path: Path) -> str:
    with fitz.open(path) as document:
        return "\n".join(page.get_text("text") for page in document)


def _extract_docx_text(path: Path) -> str:
    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _extract_txt_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_experience(text: str) -> float:
    years = [float(match) for match in re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|year)", text.lower())]
    if years:
        return max(years)
    ranges = re.findall(r"(\d+)\s*-\s*(\d+)\s*(?:years|yrs|year)", text.lower())
    if ranges:
        return max(float(end) for _, end in ranges)
    return 0.0


def _extract_skills(text: str) -> list[str]:
    lower = f" {text.lower()} "
    found = []
    for skill in COMMON_SKILLS:
        if f" {skill.lower()} " in lower:
            found.append(skill)
    if "aws" in lower:
        found.append("amazon web services")
    return normalize_skills(found)


def _extract_education(text: str) -> str:
    lower = text.lower()
    if "phd" in lower or "doctorate" in lower:
        return "PhD"
    if "master" in lower or "m.tech" in lower or "m.s" in lower:
        return "Masters"
    if "bachelor" in lower or "b.tech" in lower or "b.s" in lower:
        return "Bachelors"
    return ""


def _extract_list_lines(text: str, keyword: str, limit: int = 6) -> list[str]:
    lines = [line.strip("-* \t") for line in text.splitlines() if line.strip()]
    matches = [line for line in lines if keyword in line.lower()]
    return matches[:limit]


def parse_resume(path: str | Path, retries: int = 2, retry_delay: float = 0.2) -> dict[str, Any]:
    """Parse one resume file and return normalized candidate metadata."""
    resume_path = Path(path)
    if not resume_path.exists():
        raise ResumeParseError(f"Resume does not exist: {resume_path}")

    parse_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            suffix = resume_path.suffix.lower()
            if suffix == ".pdf":
                text = _extract_pdf_text(resume_path)
            elif suffix == ".docx":
                text = _extract_docx_text(resume_path)
            elif suffix == ".txt":
                text = _extract_txt_text(resume_path)
            else:
                raise ResumeParseError(f"Unsupported resume format: {suffix}")

            text = text.strip()
            if not text:
                raise ResumeParseError("Empty resume text")

            name_match = re.search(r"(?:name|candidate)\s*:\s*([A-Za-z \.\-]{3,60})", text, flags=re.IGNORECASE)
            candidate_name = (
                name_match.group(1).strip()
                if name_match
                else resume_path.stem.replace("_", " ").replace("-", " ").title()
            )

            return {
                "candidate_name": candidate_name,
                "skills": _extract_skills(text),
                "experience": _extract_experience(text),
                "education": _extract_education(text),
                "projects": _extract_list_lines(text, "project"),
                "certifications": _extract_list_lines(text, "cert"),
                "resume_path": str(resume_path),
                "resume_text": text,
            }
        except Exception as exc:  # noqa: BLE001 - deliberate retries for parser robustness
            parse_error = exc
            LOGGER.warning(
                "resume_parse_retry",
                extra={
                    "event": "parsing_failure",
                    "details": {"resume_path": str(resume_path), "attempt": attempt + 1},
                },
            )
            if attempt < retries:
                time.sleep(retry_delay)
            continue

    raise ResumeParseError(f"Failed to parse resume {resume_path}: {parse_error}") from parse_error


def parse_resumes_in_directory(
    directory: str | Path,
    retries: int = 2,
    retry_delay: float = 0.2,
) -> list[dict[str, Any]]:
    """Parse all supported resumes in a directory recursively."""
    root = Path(directory)
    candidates: list[dict[str, Any]] = []
    for extension in ("*.pdf", "*.docx", "*.txt"):
        for file_path in root.rglob(extension):
            try:
                candidates.append(parse_resume(file_path, retries=retries, retry_delay=retry_delay))
            except ResumeParseError:
                LOGGER.error(
                    "resume_parse_error",
                    extra={"event": "parsing_failure", "details": {"resume_path": str(file_path)}},
                )
    return candidates

