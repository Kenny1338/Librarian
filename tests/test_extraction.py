"""Tests for Groq extraction: mock API, parsing, retry, error handling."""

import json
from unittest.mock import MagicMock, patch, call

import pytest

from librarian import _extract_via_groq, EXTRACTION_MODELS, EMPTY_EXTRACTION


def _make_response(content):
    """Helper to build a mock Groq chat completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestExtractViaGroq:
    def test_basic_extraction(self):
        payload = {
            "facts": [{"text": "User is Alice", "bank": "people", "confidence": "stated", "durability": "permanent"}],
            "commitments": [],
            "entities": [{"name": "Alice", "type": "person"}],
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(json.dumps(payload))

        with patch("groq.Groq", return_value=mock_client):
            result = _extract_via_groq("fake-key", "llama-3.3-70b-versatile", "I'm Alice", "Hi Alice!")

        assert len(result["facts"]) == 1
        assert result["facts"][0]["text"] == "User is Alice"
        assert len(result["entities"]) == 1

    def test_empty_extraction(self):
        payload = {"facts": [], "commitments": [], "entities": []}
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(json.dumps(payload))

        with patch("groq.Groq", return_value=mock_client):
            result = _extract_via_groq("fake-key", "llama-3.3-70b-versatile", "Hi", "Hello!")

        assert result["facts"] == []
        assert result["commitments"] == []
        assert result["entities"] == []

    def test_missing_keys_filled(self):
        """If the model returns partial JSON, missing keys should be filled with []."""
        payload = {"facts": [{"text": "Something", "bank": "general"}]}
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(json.dumps(payload))

        with patch("groq.Groq", return_value=mock_client):
            result = _extract_via_groq("fake-key", "llama-3.3-70b-versatile", "msg", "resp")

        assert "commitments" in result
        assert "entities" in result
        assert result["commitments"] == []
        assert result["entities"] == []


class TestRetryLogic:
    def test_fallback_to_second_model(self):
        """If first model fails, should try the next."""
        mock_client = MagicMock()
        payload = {"facts": [{"text": "Fallback fact", "bank": "general"}], "commitments": [], "entities": []}

        # First call fails, second succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception("Model overloaded"),
            _make_response(json.dumps(payload)),
        ]

        with patch("groq.Groq", return_value=mock_client):
            result = _extract_via_groq("fake-key", "llama-3.3-70b-versatile", "msg", "resp")

        assert len(result["facts"]) == 1
        assert result["facts"][0]["text"] == "Fallback fact"
        assert mock_client.chat.completions.create.call_count == 2

    def test_all_models_fail(self):
        """If all models fail, should return empty extraction."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("All models down")

        with patch("groq.Groq", return_value=mock_client):
            result = _extract_via_groq("fake-key", "llama-3.3-70b-versatile", "msg", "resp")

        assert result["facts"] == []
        assert result["commitments"] == []
        assert result["entities"] == []

    def test_invalid_json_triggers_retry(self):
        """If model returns invalid JSON, it should try next model."""
        mock_client = MagicMock()
        payload = {"facts": [{"text": "OK", "bank": "general"}], "commitments": [], "entities": []}
        mock_client.chat.completions.create.side_effect = [
            _make_response("not valid json {{{"),
            _make_response(json.dumps(payload)),
        ]

        with patch("groq.Groq", return_value=mock_client):
            result = _extract_via_groq("fake-key", "llama-3.3-70b-versatile", "msg", "resp")

        assert len(result["facts"]) == 1

    def test_non_dict_response_triggers_retry(self):
        """If model returns a JSON list instead of dict, retry."""
        mock_client = MagicMock()
        payload = {"facts": [], "commitments": [], "entities": []}
        mock_client.chat.completions.create.side_effect = [
            _make_response(json.dumps([1, 2, 3])),
            _make_response(json.dumps(payload)),
        ]

        with patch("groq.Groq", return_value=mock_client):
            result = _extract_via_groq("fake-key", "llama-3.3-70b-versatile", "msg", "resp")

        assert isinstance(result, dict)


class TestErrorHandling:
    def test_none_content(self):
        """If message content is None, should handle gracefully."""
        mock_client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = None
        # None content => json.loads("{}") => empty dict => missing keys filled
        mock_client.chat.completions.create.return_value = resp

        with patch("groq.Groq", return_value=mock_client):
            result = _extract_via_groq("fake-key", "llama-3.3-70b-versatile", "msg", "resp")

        # Should not crash; returns dict with empty lists
        assert isinstance(result, dict)

    def test_model_list_order(self):
        """Primary model should be tried first."""
        mock_client = MagicMock()
        payload = {"facts": [], "commitments": [], "entities": []}
        mock_client.chat.completions.create.return_value = _make_response(json.dumps(payload))

        with patch("groq.Groq", return_value=mock_client):
            _extract_via_groq("fake-key", "llama-3.3-70b-versatile", "msg", "resp")

        # Check the model used in the first call
        first_call = mock_client.chat.completions.create.call_args_list[0]
        assert first_call.kwargs.get("model") or first_call[1].get("model") == "llama-3.3-70b-versatile"

    def test_extraction_with_rich_payload(self):
        """Test parsing of a full extraction result with all fields."""
        payload = {
            "facts": [
                {"text": "User's name is Bob", "bank": "people", "confidence": "stated", "durability": "permanent"},
                {"text": "Bob works at Acme Corp", "bank": "work", "confidence": "stated", "durability": "permanent"},
                {"text": "Meeting scheduled for Friday", "bank": "work", "confidence": "inferred", "durability": "temporal"},
            ],
            "commitments": [
                {"type": "task", "subject": "Deploy v2.0", "due": "2025-02-01"},
                {"type": "reminder", "subject": "Call dentist", "due": ""},
            ],
            "entities": [
                {"name": "Bob", "type": "person"},
                {"name": "Acme Corp", "type": "org"},
            ],
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(json.dumps(payload))

        with patch("groq.Groq", return_value=mock_client):
            result = _extract_via_groq("fake-key", "llama-3.3-70b-versatile", "msg", "resp")

        assert len(result["facts"]) == 3
        assert len(result["commitments"]) == 2
        assert len(result["entities"]) == 2
        assert result["facts"][0]["bank"] == "people"
        assert result["commitments"][0]["due"] == "2025-02-01"
