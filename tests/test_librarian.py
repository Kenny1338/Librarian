"""Tests for the Librarian class (framework-agnostic API)."""

import json
from unittest.mock import patch, MagicMock

import pytest

from librarian import Librarian, RECALL_SCHEMA, BANKS_SCHEMA, COMMITMENTS_SCHEMA


class TestLibrarianInit:
    def test_init_with_api_key(self, tmp_store_dir, mock_groq_client):
        lib = Librarian(api_key="fake-key", store_path=str(tmp_store_dir))
        assert lib._api_key == "fake-key"

    def test_init_no_key_raises(self, tmp_store_dir):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Groq API key required"):
                Librarian(store_path=str(tmp_store_dir))

    def test_init_env_key(self, tmp_store_dir, mock_groq_client):
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}):
            lib = Librarian(store_path=str(tmp_store_dir))
            assert lib._api_key == "env-key"

    def test_store_property(self, librarian):
        from librarian import LibrarianStore
        assert isinstance(librarian.store, LibrarianStore)


class TestObserve:
    def test_observe_blocking(self, librarian, mock_groq_client):
        librarian.observe("I'm Alice", "Nice to meet you!", blocking=True)
        # Facts should be stored
        banks = librarian.banks()
        total = sum(banks.values())
        assert total > 0

    def test_observe_nonblocking(self, librarian, mock_groq_client):
        librarian.observe("I'm building a compiler", "Cool!")
        # Wait for thread
        librarian.flush()
        banks = librarian.banks()
        total = sum(banks.values())
        assert total > 0

    def test_observe_calls_groq(self, librarian, mock_groq_client):
        librarian.observe("test message", "test response", blocking=True)
        mock_groq_client.chat.completions.create.assert_called_once()


class TestRecall:
    def test_recall_empty(self, librarian):
        results = librarian.recall("anything")
        assert results == []

    def test_recall_with_data(self, librarian, sample_facts):
        librarian.store.add_facts(sample_facts)
        results = librarian.recall("Rust")
        assert len(results) >= 1
        assert any("Rust" in f["text"] for f in results)

    def test_recall_with_bank(self, librarian, sample_facts):
        librarian.store.add_facts(sample_facts)
        results = librarian.recall("Alice", bank="people")
        assert len(results) >= 1

    def test_recall_limit(self, librarian):
        facts = [{"text": f"Python fact {i}", "bank": "general"} for i in range(30)]
        librarian.store.add_facts(facts)
        results = librarian.recall("Python", limit=5)
        assert len(results) <= 5


class TestSummary:
    def test_summary_empty(self, librarian):
        assert librarian.summary() == ""

    def test_summary_with_data(self, librarian, sample_facts):
        librarian.store.add_facts(sample_facts)
        summary = librarian.summary()
        assert "Alice" in summary


class TestBanksAndCommitments:
    def test_banks_empty(self, librarian):
        assert librarian.banks() == {}

    def test_banks_with_data(self, librarian, sample_facts):
        librarian.store.add_facts(sample_facts)
        banks = librarian.banks()
        assert len(banks) > 0

    def test_commitments_empty(self, librarian):
        assert librarian.commitments() == []

    def test_commitments_with_data(self, librarian, sample_commitments):
        librarian.store.add_commitments(sample_commitments)
        cmts = librarian.commitments()
        assert len(cmts) == 2


class TestFlush:
    def test_flush_no_thread(self, librarian):
        # Should not raise
        librarian.flush()

    def test_flush_waits(self, librarian, mock_groq_client):
        librarian.observe("test", "test")
        librarian.flush()
        # After flush, queue should be empty
        assert librarian._work_queue.empty()


class TestToolSchemas:
    def test_tool_schemas_returns_list(self, librarian):
        schemas = librarian.tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 3

    def test_tool_schemas_names(self, librarian):
        schemas = librarian.tool_schemas()
        names = {s["name"] for s in schemas}
        assert names == {"librarian_recall", "librarian_banks", "librarian_commitments"}


class TestHandleToolCall:
    def test_recall_tool(self, librarian, sample_facts):
        librarian.store.add_facts(sample_facts)
        result = json.loads(librarian.handle_tool_call("librarian_recall", {"query": "Rust"}))
        assert "result" in result
        assert "Rust" in result["result"]

    def test_recall_tool_no_results(self, librarian):
        result = json.loads(librarian.handle_tool_call("librarian_recall", {"query": "nonexistent"}))
        assert "No memories found" in result["result"]

    def test_banks_tool_list(self, librarian, sample_facts):
        librarian.store.add_facts(sample_facts)
        result = json.loads(librarian.handle_tool_call("librarian_banks", {}))
        assert "result" in result

    def test_banks_tool_specific_bank(self, librarian, sample_facts):
        librarian.store.add_facts(sample_facts)
        result = json.loads(librarian.handle_tool_call("librarian_banks", {"bank": "people"}))
        assert "result" in result
        assert "Alice" in result["result"]

    def test_banks_tool_empty_bank(self, librarian):
        result = json.loads(librarian.handle_tool_call("librarian_banks", {"bank": "nonexistent"}))
        assert "not found" in result["result"] or "No banks" in result.get("result", "")

    def test_commitments_tool(self, librarian, sample_commitments):
        librarian.store.add_commitments(sample_commitments)
        result = json.loads(librarian.handle_tool_call("librarian_commitments", {}))
        assert "Review PR" in result["result"]

    def test_commitments_tool_empty(self, librarian):
        result = json.loads(librarian.handle_tool_call("librarian_commitments", {}))
        assert "No active commitments" in result["result"]

    def test_unknown_tool(self, librarian):
        result = json.loads(librarian.handle_tool_call("unknown_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]


class TestForget:
    """Test forget method if it exists."""

    def test_forget_not_available(self, librarian):
        if not hasattr(librarian, "forget"):
            pytest.skip("forget method not yet implemented")

    def test_forget_by_bank(self, librarian):
        if not hasattr(librarian, "forget"):
            pytest.skip("forget method not yet implemented")
        librarian.store.add_facts([{"text": "Secret fact", "bank": "general"}])
        removed = librarian.forget(bank="general")
        assert removed == 1
        assert librarian.store.get_bank_facts("general") == []

    def test_forget_by_query(self, librarian):
        if not hasattr(librarian, "forget"):
            pytest.skip("forget method not yet implemented")
        librarian.store.add_facts([
            {"text": "User likes Python", "bank": "general"},
            {"text": "User likes Rust", "bank": "general"},
        ])
        removed = librarian.forget("Python")
        assert removed == 1
        remaining = librarian.store.get_bank_facts("general")
        assert len(remaining) == 1
        assert "Rust" in remaining[0]["text"]

    def test_forget_all(self, librarian):
        if not hasattr(librarian, "forget_all"):
            pytest.skip("forget_all not yet implemented")
        librarian.store.add_facts([
            {"text": "Fact 1", "bank": "general"},
            {"text": "Fact 2", "bank": "work"},
        ])
        removed = librarian.forget_all()
        assert removed == 2
        assert librarian.banks() == {} or sum(librarian.banks().values()) == 0
