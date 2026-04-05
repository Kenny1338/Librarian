"""Deduplication engine: MD5 hash + trigram Jaccard similarity.

Two-tier approach:
  1. Normalized text hash for exact/near-exact matches — O(1)
  2. Trigram similarity for coarse-hash bucket collisions — rare

Average case is O(1) even with 1000+ facts per bank.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Set

# Similarity threshold for deduplication (0.0 = no match, 1.0 = identical)
DEDUP_THRESHOLD = 0.75

_STRIP_RE = re.compile(r"[^a-z0-9 ]")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _STRIP_RE.sub("", text.lower()).strip()


def _norm_hash(text: str) -> str:
    """Hash of normalized text — catches exact and near-exact duplicates."""
    normed = _normalize(text)
    return hashlib.md5(normed.encode()).hexdigest()[:12]


def _coarse_hash(text: str) -> str:
    """Coarser hash: sorted words — catches rewordings with same vocabulary."""
    normed = _normalize(text)
    words = sorted(set(normed.split()))
    return hashlib.md5(" ".join(words).encode()).hexdigest()[:12]


def _trigrams(text: str) -> Set[str]:
    """Return set of character trigrams from normalized text."""
    normed = _normalize(text)
    if len(normed) < 3:
        return {normed}
    return {normed[i:i + 3] for i in range(len(normed) - 2)}


def _trigram_similarity(a: str, b: str) -> float:
    """Trigram Jaccard similarity between two strings."""
    ta = _trigrams(a)
    tb = _trigrams(b)
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union if union else 0.0


class DedupIndex:
    """In-memory dedup index for a set of texts.

    Two-tier approach:
      - Tier 1: exact normalized hash -> O(1) lookup
      - Tier 2: coarse (sorted-word) hash -> bucket of texts for trigram check
    """

    def __init__(self) -> None:
        self._exact_hashes: Set[str] = set()
        self._coarse_buckets: Dict[str, List[str]] = {}

    def load(self, texts: List[str]) -> None:
        """Bulk-load existing texts into the index."""
        self._exact_hashes.clear()
        self._coarse_buckets.clear()
        for t in texts:
            self._add_to_index(t)

    def _add_to_index(self, text: str) -> None:
        eh = _norm_hash(text)
        self._exact_hashes.add(eh)
        ch = _coarse_hash(text)
        bucket = self._coarse_buckets.setdefault(ch, [])
        bucket.append(text)

    def is_duplicate(self, new_text: str, threshold: float = DEDUP_THRESHOLD) -> bool:
        """Check if new_text is a duplicate. O(1) average case."""
        # Tier 1: exact hash
        eh = _norm_hash(new_text)
        if eh in self._exact_hashes:
            return True
        # Tier 2: coarse hash bucket -> trigram check
        ch = _coarse_hash(new_text)
        bucket = self._coarse_buckets.get(ch)
        if bucket:
            for existing in bucket:
                if _trigram_similarity(new_text, existing) >= threshold:
                    return True
        return False

    def add(self, text: str) -> None:
        """Add a text to the index after it's been stored."""
        self._add_to_index(text)

    @property
    def size(self) -> int:
        return len(self._exact_hashes)


def _is_duplicate(new_text: str, existing_texts: List[str], threshold: float = DEDUP_THRESHOLD) -> bool:
    """Convenience function — builds a temporary index."""
    idx = DedupIndex()
    idx.load(existing_texts)
    return idx.is_duplicate(new_text, threshold)
