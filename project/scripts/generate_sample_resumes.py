"""Generate sample resumes in PDF/DOCX/TXT for local testing."""

from __future__ import annotations

from pathlib import Path

try:
    import fitz
except ImportError:  # pragma: no cover - runtime optional
    fitz = None

try:
    from docx import Document
except ImportError:  # pragma: no cover - runtime optional
    Document = None


SAMPLE_RESUMES = [
    {
        "filename": "john_doe_backend",
        "content": """Name: John Doe
Experience: 7 years
Skills: Python, FastAPI, AWS, PostgreSQL, Docker, Kubernetes
Education: Bachelors in Computer Science
Projects:
- Project Atlas: Built high-scale backend APIs for fintech workloads.
- Project Mercury: Designed CI/CD automation and cloud deployment pipelines.
Certifications:
- AWS Certified Developer Associate
""",
    },
    {
        "filename": "jane_smith_fullstack",
        "content": """Name: Jane Smith
Experience: 6 years
Skills: React, JavaScript, Python, Node.js, AWS, MongoDB
Education: Masters in Software Engineering
Projects:
- Project Nova: Led React migration for enterprise dashboard.
- Project Orion: Developed event-driven microservices with AWS.
Certifications:
- Certified Kubernetes Application Developer
""",
    },
    {
        "filename": "alex_lee_ml",
        "content": """Name: Alex Lee
Experience: 5 years
Skills: Python, Machine Learning, NLP, LangChain, Azure
Education: Masters in Artificial Intelligence
Projects:
- Project Echo: Built NLP classifier for support automation.
- Project Prism: Deployed ML inference platform on Kubernetes.
Certifications:
- Azure AI Engineer Associate
""",
    },
]


def _write_pdf(path: Path, content: str) -> None:
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), content, fontsize=11)
    doc.save(path)
    doc.close()


def _write_docx(path: Path, content: str) -> None:
    if Document is None:
        raise RuntimeError("python-docx not installed")
    document = Document()
    for line in content.splitlines():
        document.add_paragraph(line)
    document.save(path)


def _write_txt(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    resumes_dir = root / "data" / "resumes"
    resumes_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(SAMPLE_RESUMES):
        base = resumes_dir / sample["filename"]
        content = sample["content"].strip() + "\n"
        if idx % 2 == 0:
            try:
                _write_pdf(base.with_suffix(".pdf"), content)
                continue
            except Exception:
                pass
        try:
            _write_docx(base.with_suffix(".docx"), content)
        except Exception:
            _write_txt(base.with_suffix(".txt"), content)
    print(f"Sample resumes generated under: {resumes_dir}")


if __name__ == "__main__":
    main()

