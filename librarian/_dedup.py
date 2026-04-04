"""Optimized deduplication with two-tier approach: hash + trigram similarity.

Replaces the old O(n^2) SequenceMatcher approach with:
  1. Normalized text hash for exact/near-exact matches — O(1)
  2. Trigram similarity only for hash bucket collisions — rare
  
Average case is O(1) for 1000+ facts per bank.
"""

from __future__ import annotations

import hashlib
import re
import logging
from typing import Dict, List, Optional, Set, Tuple

from librarian._tools import DEDUP_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization + trigram helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dedup Index — loaded once per bank, updated incrementally
# ---------------------------------------------------------------------------

class DedupIndex:
    """In-memory dedup index for a set of texts.
    
    Two-tier approach:
      - Tier 1: exact normalized hash → O(1) lookup
      - Tier 2: coarse (sorted-word) hash → bucket of texts for trigram check
    """

    def __init__(self):
        # exact normalized hashes
        self._exact_hashes: Set[str] = set()
        # coarse hash → list of original texts in that bucket
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
            logger.debug("[librarian] Dedup exact hash hit: %.50s", new_text)
            return True

        # Tier 2: coarse hash bucket → trigram check
        ch = _coarse_hash(new_text)
        bucket = self._coarse_buckets.get(ch)
        if bucket:
            for existing in bucket:
                sim = _trigram_similarity(new_text, existing)
                if sim >= threshold:
                    logger.debug(
                        "[librarian] Dedup trigram hit: '%.50s' ~ '%.50s' (%.2f >= %.2f)",
                        new_text, existing, sim, threshold,
                    )
                    return True

        return False

    def add(self, text: str) -> None:
        """Add a text to the index after it's been stored."""
        self._add_to_index(text)

    @property
    def size(self) -> int:
        return len(self._exact_hashes)


# ---------------------------------------------------------------------------
# Module-level convenience (backward compat)
# ---------------------------------------------------------------------------

def _is_duplicate(new_text: str, existing_texts: List[str], threshold: float = DEDUP_THRESHOLD) -> bool:
    """Legacy API — builds a temporary index. Use DedupIndex for hot paths."""
    idx = DedupIndex()
    idx.load(existing_texts)
    return idx.is_duplicate(new_text, threshold)
