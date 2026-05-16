"""Embedding utilities backed by sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=20000)
def _embed_cached(model_name: str, text: str) -> tuple[float, ...]:
    model = _load_model(model_name)
    vector = model.encode(text, normalize_embeddings=True)
    return tuple(float(v) for v in vector.tolist())


class EmbeddingService:
    """Reusable embedding service with in-process caching."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed_text(self, text: str) -> list[float]:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            vector_size = _load_model(self.model_name).get_sentence_embedding_dimension()
            return [0.0 for _ in range(vector_size)]
        return list(_embed_cached(self.model_name, cleaned))

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        a = np.array(vec_a, dtype=float)
        b = np.array(vec_b, dtype=float)
        if np.linalg.norm(a) == 0.0 or np.linalg.norm(b) == 0.0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

