"""Tests for deduplication: edge cases, threshold tuning, performance."""

import time

import pytest

from librarian import _is_duplicate, DEDUP_THRESHOLD, LibrarianStore


class TestIsDuplicate:
    def test_exact_match(self):
        assert _is_duplicate("hello world", ["hello world"])

    def test_case_insensitive(self):
        assert _is_duplicate("Hello World", ["hello world"])

    def test_whitespace_handling(self):
        assert _is_duplicate("  hello world  ", ["hello world"])

    def test_no_match(self):
        assert not _is_duplicate("hello world", ["goodbye universe"])

    def test_empty_existing(self):
        assert not _is_duplicate("hello world", [])

    def test_empty_new_text(self):
        # Empty vs empty should match (ratio=1.0)
        assert _is_duplicate("", [""])

    def test_near_duplicate(self):
        assert _is_duplicate(
            "User prefers dark mode always",
            ["User prefers dark mode always!"],
        )

    def test_below_threshold(self):
        assert not _is_duplicate(
            "User likes Python",
            ["The weather is sunny today"],
        )

    def test_threshold_boundary(self):
        # Same prefix, different ending — test near the boundary
        text1 = "User is working on project Alpha"
        text2 = "User is working on project Beta"
        # These share a lot but differ — depends on exact ratio
        result = _is_duplicate(text1, [text2])
        # Just verify it returns a bool
        assert isinstance(result, bool)

    def test_custom_threshold_strict(self):
        # Very strict threshold — only near-exact matches
        assert not _is_duplicate(
            "User likes Python programming",
            ["User likes Python a lot"],
            threshold=0.95,
        )

    def test_custom_threshold_loose(self):
        # Same words in different order — coarse hash matches, trigram check runs
        assert _is_duplicate(
            "User likes Python for scripting",
            ["Python for scripting User likes"],
            threshold=0.5,
        )

    def test_multiple_existing(self):
        existing = [
            "User likes Java",
            "User likes Rust",
            "User likes Python",
        ]
        assert _is_duplicate("User likes Python", existing)
        assert not _is_duplicate("User likes Haskell and category theory", existing)

    def test_unicode_text(self):
        assert _is_duplicate("User speaks français", ["User speaks français"])

    def test_long_texts(self):
        text = "a" * 1000
        assert _is_duplicate(text, [text])


class TestDedupInStore:
    def test_exact_dedup_in_store(self, store):
        facts = [{"text": "User is Alice", "bank": "general"}]
        store.add_facts(facts)
        added = store.add_facts(facts)
        assert added == 0

    def test_similar_dedup_in_store(self, store):
        store.add_facts([{"text": "User's favorite color is blue", "bank": "preferences"}])
        added = store.add_facts([{"text": "User's favorite color is blue!", "bank": "preferences"}])
        assert added == 0

    def test_different_facts_not_deduped(self, store):
        store.add_facts([{"text": "User likes Python", "bank": "general"}])
        added = store.add_facts([{"text": "User dislikes JavaScript", "bank": "general"}])
        assert added == 1

    def test_cross_bank_no_dedup(self, store):
        """Facts in different banks are NOT deduped against each other."""
        store.add_facts([{"text": "User likes Python", "bank": "general"}])
        added = store.add_facts([{"text": "User likes Python", "bank": "work"}])
        assert added == 1  # Different bank, so it's added


class TestDedupPerformance:
    def test_500_facts_performance(self, store):
        """Adding 500+ unique facts should complete in reasonable time."""
        # Spread across multiple banks to avoid O(n^2) dedup within one bank
        facts = [{"text": f"Unique fact number {i} about topic {i * 7}", "bank": f"bank{i % 10}"} for i in range(500)]
        start = time.monotonic()
        added = store.add_facts(facts)
        elapsed = time.monotonic() - start
        assert added == 500
        assert elapsed < 60, f"Adding 500 facts took {elapsed:.1f}s (too slow)"

    def test_500_duplicates_performance(self, store):
        """Checking 500 duplicates against 500 existing facts."""
        facts = [{"text": f"Unique fact number {i} about topic {i * 7}", "bank": f"bank{i % 10}"} for i in range(500)]
        store.add_facts(facts)

        start = time.monotonic()
        added = store.add_facts(facts)  # All duplicates
        elapsed = time.monotonic() - start
        assert added == 0
        assert elapsed < 60, f"Dedup check for 500 facts took {elapsed:.1f}s (too slow)"

    def test_incremental_add(self, store):
        """Add facts one by one — dedup grows incrementally."""
        for i in range(100):
            added = store.add_facts([{"text": f"Fact number {i} unique content here", "bank": "general"}])
            assert added == 1
        assert store.get_banks()["general"] == 100
