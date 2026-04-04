"""Shared fixtures for librarian tests."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from librarian import LibrarianStore, Librarian, _is_duplicate, DEDUP_THRESHOLD


@pytest.fixture
def tmp_store_dir(tmp_path):
    """Provide a temporary directory for LibrarianStore."""
    return tmp_path / "store"


@pytest.fixture
def store(tmp_store_dir):
    """Provide a fresh LibrarianStore."""
    return LibrarianStore(tmp_store_dir)


@pytest.fixture
def sample_facts():
    """Sample facts for testing."""
    return [
        {"text": "User's name is Alice", "bank": "people", "confidence": "stated", "durability": "permanent"},
        {"text": "User is building a Rust compiler", "bank": "projects", "confidence": "stated", "durability": "permanent"},
        {"text": "User prefers dark mode", "bank": "preferences", "confidence": "stated", "durability": "permanent"},
        {"text": "User has a meeting on 2025-01-15", "bank": "work", "confidence": "stated", "durability": "temporal"},
        {"text": "User's favorite language is Python", "bank": "preferences", "confidence": "stated", "durability": "permanent"},
    ]


@pytest.fixture
def sample_commitments():
    return [
        {"type": "task", "subject": "Review PR #42", "due": "2025-01-20"},
        {"type": "reminder", "subject": "Send weekly report", "due": ""},
    ]


@pytest.fixture
def sample_entities():
    return [
        {"name": "Alice", "type": "person"},
        {"name": "Acme Corp", "type": "org"},
    ]


@pytest.fixture
def populated_store(store, sample_facts, sample_commitments, sample_entities):
    """Store pre-populated with sample data."""
    store.add_facts(sample_facts)
    store.add_commitments(sample_commitments)
    store.add_entities(sample_entities)
    return store


@pytest.fixture
def mock_groq_response():
    """Return a mock Groq API response with extraction results."""
    def _make(facts=None, commitments=None, entities=None):
        payload = {
            "facts": facts or [],
            "commitments": commitments or [],
            "entities": entities or [],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(payload)
        return mock_response
    return _make


@pytest.fixture
def mock_groq_client(mock_groq_response, sample_facts):
    """Patch groq.Groq so no real API calls are made."""
    response = mock_groq_response(
        facts=sample_facts,
        commitments=[{"type": "task", "subject": "Fix the bug", "due": ""}],
        entities=[{"name": "Alice", "type": "person"}],
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response
    with patch("groq.Groq", return_value=mock_client) as patcher:
        yield mock_client


@pytest.fixture
def librarian(tmp_store_dir, mock_groq_client):
    """Provide a Librarian instance with mocked Groq."""
    return Librarian(api_key="fake-key", store_path=str(tmp_store_dir))
