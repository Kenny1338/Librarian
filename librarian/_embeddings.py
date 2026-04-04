"""Optional embedding-based semantic search for Librarian.

Uses sentence-transformers (all-MiniLM-L6-v2) when available.
Falls back gracefully to None when the package is not installed.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

# Default model — small and fast (~80 MB)
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingIndex:
    """Lazy-loaded embedding index for semantic search.

    The underlying model is only downloaded / loaded when :meth:`embed` is
    first called, so importing this module has zero cost.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model: Any = None  # SentenceTransformer instance, loaded lazily

    @property
    def available(self) -> bool:
        """Return True if sentence-transformers is installed."""
        return _HAS_SENTENCE_TRANSFORMERS

    def _load_model(self) -> None:
        if self._model is None:
            if not _HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is required for embedding search. "
                    "Install it with: pip install 'librarian-ai[embeddings]'"
                )
            self._model = SentenceTransformer(self._model_name)
            logger.info("[librarian] Loaded embedding model: %s", self._model_name)

    def embed(self, text: str) -> List[float]:
        """Compute the embedding vector for *text*."""
        self._load_model()
        vec = self._model.encode(text, show_progress_bar=False)
        return vec.tolist()

    def search(
        self,
        query: str,
        facts: List[Dict[str, Any]],
        *,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return *facts* ranked by cosine similarity to *query*.

        Each fact dict may contain a pre-computed ``"embedding"`` field.
        Facts without an embedding are embedded on-the-fly (but not cached
        here — the caller is responsible for persisting).
        """
        if not facts:
            return []

        query_vec = self.embed(query)

        scored: List[tuple[float, Dict[str, Any]]] = []
        for fact in facts:
            fact_vec = fact.get("embedding")
            if fact_vec is None:
                fact_vec = self.embed(fact.get("text", ""))
            score = _cosine_similarity(query_vec, fact_vec)
            scored.append((score, fact))

        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:top_k]]
