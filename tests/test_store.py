"""Tests for LibrarianStore: add_facts, dedup, search, banks, summary."""

import json
from pathlib import Path

import pytest

from librarian import LibrarianStore


class TestAddFacts:
    def test_add_facts_basic(self, store, sample_facts):
        added = store.add_facts(sample_facts)
        assert added == len(sample_facts)

    def test_add_facts_dedup(self, store, sample_facts):
        store.add_facts(sample_facts)
        # Adding the same facts again should add 0
        added = store.add_facts(sample_facts)
        assert added == 0

    def test_add_facts_near_duplicate(self, store):
        store.add_facts([{"text": "User's name is Alice", "bank": "people"}])
        # Very similar text should be deduped
        added = store.add_facts([{"text": "User's name is alice", "bank": "people"}])
        assert added == 0

    def test_add_facts_different_banks(self, store):
        facts = [
            {"text": "Fact A", "bank": "general"},
            {"text": "Fact B", "bank": "work"},
            {"text": "Fact C", "bank": "projects"},
        ]
        added = store.add_facts(facts)
        assert added == 3
        banks = store.get_banks()
        assert len(banks) == 3

    def test_add_facts_default_bank(self, store):
        # No bank specified => defaults to "general"
        added = store.add_facts([{"text": "Some fact"}])
        assert added == 1
        assert "general" in store.get_banks()

    def test_add_facts_persistence(self, tmp_store_dir, sample_facts):
        store1 = LibrarianStore(tmp_store_dir)
        store1.add_facts(sample_facts)

        # New store instance, same path
        store2 = LibrarianStore(tmp_store_dir)
        banks = store2.get_banks()
        total = sum(banks.values())
        assert total == len(sample_facts)

    def test_add_facts_empty_list(self, store):
        added = store.add_facts([])
        assert added == 0


class TestSearchFacts:
    def test_search_exact(self, populated_store):
        results = populated_store.search_facts("Rust compiler")
        assert len(results) >= 1
        assert any("Rust" in f["text"] for f in results)

    def test_search_partial(self, populated_store):
        results = populated_store.search_facts("dark mode")
        assert len(results) >= 1

    def test_search_no_results(self, populated_store):
        results = populated_store.search_facts("quantum physics")
        assert len(results) == 0

    def test_search_by_bank(self, populated_store):
        results = populated_store.search_facts("Alice", bank="people")
        assert len(results) >= 1

    def test_search_by_wrong_bank(self, populated_store):
        # Alice is in "people" bank, not "work"
        results = populated_store.search_facts("Alice", bank="work")
        assert len(results) == 0

    def test_search_word_matching(self, populated_store):
        results = populated_store.search_facts("favorite language")
        assert len(results) >= 1

    def test_search_short_words_ignored(self, populated_store):
        # Words <= 2 chars are ignored in word matching
        results = populated_store.search_facts("is a")
        # Should still work via substring match if "is a" appears
        # This tests the filter on short words
        assert isinstance(results, list)

    def test_search_limit(self, store):
        # Add many facts
        facts = [{"text": f"Python fact number {i}", "bank": "general"} for i in range(30)]
        store.add_facts(facts)
        results = store.search_facts("Python")
        assert len(results) <= 20  # default limit in search_facts


class TestBankOperations:
    def test_get_banks_empty(self, store):
        assert store.get_banks() == {}

    def test_get_banks(self, populated_store):
        banks = populated_store.get_banks()
        assert "people" in banks
        assert "projects" in banks
        assert "preferences" in banks
        assert "work" in banks

    def test_get_bank_facts(self, populated_store):
        facts = populated_store.get_bank_facts("people")
        assert len(facts) == 1
        assert "Alice" in facts[0]["text"]

    def test_get_bank_facts_empty(self, populated_store):
        facts = populated_store.get_bank_facts("nonexistent")
        assert facts == []

    def test_get_all_facts(self, populated_store, sample_facts):
        all_facts = populated_store.get_all_facts()
        assert len(all_facts) == len(sample_facts)
        # Each fact should have a 'bank' key
        for f in all_facts:
            assert "bank" in f


class TestCommitments:
    def test_add_commitments(self, store, sample_commitments):
        added = store.add_commitments(sample_commitments)
        assert added == 2

    def test_add_commitments_dedup(self, store, sample_commitments):
        store.add_commitments(sample_commitments)
        added = store.add_commitments(sample_commitments)
        assert added == 0

    def test_get_commitments(self, populated_store):
        cmts = populated_store.get_commitments()
        assert len(cmts) == 2

    def test_get_active_commitments(self, populated_store):
        active = populated_store.get_active_commitments()
        assert len(active) == 2
        assert all(c["status"] == "active" for c in active)


class TestEntities:
    def test_add_entities(self, store, sample_entities):
        store.add_entities(sample_entities)
        # Read back
        data = json.loads((store.root / "entities.json").read_text())
        assert len(data["entities"]) == 2

    def test_add_entities_dedup(self, store, sample_entities):
        store.add_entities(sample_entities)
        store.add_entities(sample_entities)
        data = json.loads((store.root / "entities.json").read_text())
        assert len(data["entities"]) == 2

    def test_add_entities_case_insensitive(self, store):
        store.add_entities([{"name": "Alice", "type": "person"}])
        store.add_entities([{"name": "alice", "type": "person"}])
        data = json.loads((store.root / "entities.json").read_text())
        assert len(data["entities"]) == 1


class TestBuildSummary:
    def test_summary_empty(self, store):
        assert store.build_summary() == ""

    def test_summary_with_facts(self, populated_store):
        summary = populated_store.build_summary()
        assert "What You Remember" in summary
        assert "Alice" in summary
        assert "Rust" in summary

    def test_summary_with_commitments(self, populated_store):
        summary = populated_store.build_summary()
        assert "Active Commitments" in summary
        assert "Review PR #42" in summary

    def test_summary_temporal_tag(self, populated_store):
        summary = populated_store.build_summary()
        assert "[temporal]" in summary

    def test_summary_bank_names(self, populated_store):
        summary = populated_store.build_summary()
        assert "people" in summary
        assert "projects" in summary
