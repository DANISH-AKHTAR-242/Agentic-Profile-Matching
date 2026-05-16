"""Semantic retrieval and ChromaDB indexing service."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools.embeddings import EmbeddingService
from tools.resume_parser import ResumeParseError, parse_resume
from utils.config import Settings
from utils.logger import get_logger
from utils.synonyms import expand_skill_variants, normalize_skills


class RetrievalService:
    """Indexes resumes and retrieves candidates with hybrid filtering."""

    def __init__(self, settings: Settings, embedding_service: EmbeddingService) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.logger = get_logger("retrieval_service", settings.log_level)
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.collection: Any = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)

    @staticmethod
    def _candidate_id(path: Path) -> str:
        return hashlib.sha1(str(path.resolve()).lower().encode("utf-8")).hexdigest()  # noqa: S324

    @staticmethod
    def _requirements_to_query(requirements: dict[str, Any]) -> str:
        skills = requirements.get("must_have", []) + requirements.get("nice_to_have", [])
        domains = requirements.get("domain_preferences", [])
        parts = [
            "skills: " + ", ".join(skills),
            f"minimum experience: {requirements.get('min_experience', 0)} years",
            f"education: {requirements.get('education', '')}",
            "domains: " + ", ".join(domains),
        ]
        return " | ".join(parts)

    @staticmethod
    def _metadata_to_candidate(
        candidate_id: str,
        metadata: dict[str, Any],
        document: str,
        distance: float,
    ) -> dict[str, Any]:
        skills = [skill.strip() for skill in str(metadata.get("skills", "")).split(",") if skill.strip()]
        if not skills and metadata.get("skills_json"):
            try:
                skills = json.loads(str(metadata.get("skills_json")))
            except json.JSONDecodeError:
                skills = []
        similarity = max(0.0, 1.0 - float(distance))
        return {
            "candidate_id": candidate_id,
            "candidate_name": metadata.get("candidate_name", ""),
            "skills": normalize_skills(skills),
            "experience": float(metadata.get("experience", 0.0)),
            "education": metadata.get("education", ""),
            "projects": [x for x in str(metadata.get("projects", "")).split("||") if x],
            "certifications": [x for x in str(metadata.get("certifications", "")).split("||") if x],
            "resume_path": metadata.get("resume_path", ""),
            "resume_text": document,
            "semantic_similarity": round(similarity, 6),
        }

    def _hybrid_filter(
        self,
        candidates: list[dict[str, Any]],
        requirements: dict[str, Any],
    ) -> list[dict[str, Any]]:
        must_have = requirements.get("must_have", [])
        min_experience = float(requirements.get("min_experience", 0.0))
        if not must_have and min_experience <= 0:
            return candidates

        filtered: list[dict[str, Any]] = []
        for candidate in candidates:
            skills = {skill.lower() for skill in candidate.get("skills", [])}
            text = candidate.get("resume_text", "").lower()
            has_must_have = True
            for skill in must_have:
                variants = expand_skill_variants(skill)
                if not any((variant in skills) or (variant in text) for variant in variants):
                    has_must_have = False
                    break
            has_experience = float(candidate.get("experience", 0.0)) >= max(min_experience * 0.5, 0.0)
            if has_must_have and has_experience:
                filtered.append(candidate)
        return filtered

    def index_resume(self, resume_path: str | Path, force_reindex: bool = False) -> str | None:
        """Index one resume into ChromaDB and return candidate ID."""
        path = Path(resume_path)
        candidate_id = self._candidate_id(path)
        existing = self.collection.get(ids=[candidate_id])
        if existing.get("ids") and not force_reindex:
            return None

        if force_reindex and existing.get("ids"):
            self.collection.delete(ids=[candidate_id])

        parsed = parse_resume(
            path=path,
            retries=self.settings.parser_retries,
            retry_delay=self.settings.parser_retry_delay_seconds,
        )
        base_document = Document(
            page_content=parsed["resume_text"],
            metadata={"candidate_name": parsed["candidate_name"]},
        )
        chunks = self.splitter.split_documents([base_document])
        condensed_text = " ".join(chunk.page_content for chunk in chunks[:5]) if chunks else parsed["resume_text"]

        embedding = self.embedding_service.embed_text(condensed_text)
        metadata = {
            "candidate_name": parsed["candidate_name"],
            "skills": ",".join(parsed["skills"]),
            "skills_json": json.dumps(parsed["skills"]),
            "experience": float(parsed["experience"]),
            "education": parsed["education"],
            "resume_path": parsed["resume_path"],
            "projects": "||".join(parsed["projects"]),
            "certifications": "||".join(parsed["certifications"]),
        }
        self.collection.add(
            ids=[candidate_id],
            documents=[condensed_text],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        return candidate_id

    def index_directory(self, directory: str | Path, force_reindex: bool = False) -> dict[str, int]:
        """Incrementally index all resumes in a directory."""
        root = Path(directory)
        indexed = 0
        skipped = 0
        failed = 0
        for extension in ("*.pdf", "*.docx", "*.txt"):
            for resume_path in root.rglob(extension):
                try:
                    result = self.index_resume(resume_path=resume_path, force_reindex=force_reindex)
                    if result:
                        indexed += 1
                    else:
                        skipped += 1
                except ResumeParseError:
                    failed += 1
                    self.logger.error(
                        "indexing_parse_failure",
                        extra={
                            "event": "parsing_failure",
                            "details": {"resume_path": str(resume_path)},
                        },
                    )
        return {"indexed": indexed, "skipped": skipped, "failed": failed}

    def search(self, requirements: dict[str, Any], top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve and hybrid-filter top candidate resumes."""
        query_text = self._requirements_to_query(requirements)
        query_embedding = self.embedding_service.embed_text(query_text)
        n_results = top_k or self.settings.retrieval_top_k
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        candidates = []
        for candidate_id, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
            candidates.append(self._metadata_to_candidate(candidate_id, metadata or {}, document, float(distance)))

        filtered = self._hybrid_filter(candidates, requirements=requirements)
        self.logger.info(
            "retrieval_complete",
            extra={
                "event": "retrieval",
                "details": {
                    "query": query_text,
                    "retrieved": len(candidates),
                    "filtered": len(filtered),
                },
            },
        )
        return filtered[:n_results]

    async def asearch(self, requirements: dict[str, Any], top_k: int | None = None) -> list[dict[str, Any]]:
        """Async wrapper for semantic retrieval."""
        return await asyncio.to_thread(self.search, requirements, top_k)

    def get_candidates_by_ids(self, candidate_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch candidates from persistent store using IDs."""
        if not candidate_ids:
            return []
        raw = self.collection.get(ids=candidate_ids, include=["documents", "metadatas"])
        ids = raw.get("ids", [])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])
        candidates = []
        for candidate_id, document, metadata in zip(ids, documents, metadatas, strict=False):
            candidates.append(self._metadata_to_candidate(candidate_id, metadata or {}, document or "", 0.0))
        return candidates

