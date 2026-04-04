"""Tool schemas and constants for Librarian."""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXTRACTION_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

EMPTY_EXTRACTION: Dict[str, Any] = {"facts": [], "commitments": [], "entities": []}

# Similarity threshold for deduplication (0.0 = no match, 1.0 = identical)
DEDUP_THRESHOLD = 0.75

# Default TTL for temporal facts (seconds) — 30 days
DEFAULT_TEMPORAL_TTL = 30 * 24 * 60 * 60

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-compatible function calling)
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "librarian_recall",
    "description": (
        "Search the Librarian's memory banks for relevant facts about the user. "
        "Banks: general, people, work, health, projects, preferences. "
        "Returns matching facts ranked by relevance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "bank": {"type": "string", "description": "Optional: restrict to a specific bank."},
        },
        "required": ["query"],
    },
}

BANKS_SCHEMA = {
    "name": "librarian_banks",
    "description": "List all memory banks and their fact counts, or show all facts in a specific bank.",
    "parameters": {
        "type": "object",
        "properties": {
            "bank": {"type": "string", "description": "Optional: show facts from this specific bank."},
        },
    },
}

COMMITMENTS_SCHEMA = {
    "name": "librarian_commitments",
    "description": "List all active commitments (promises, tasks, reminders) tracked by the Librarian.",
    "parameters": {"type": "object", "properties": {}},
}
