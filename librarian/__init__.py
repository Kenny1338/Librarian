"""Librarian — persistent memory for AI agents, powered by Groq.

Observes conversation turns, extracts facts / commitments / entities via a
fast LLM sidecar, and stores them in categorised memory banks.  Before each
turn it assembles a context packet so the agent has instant recall.

Works with **any agent framework** (OpenAI, LangChain, Claude, etc.) or standalone.
Also works as a native **Hermes Agent plugin**.

Quick start::

    from librarian import Librarian

    lib = Librarian(api_key="your-groq-key")
    lib.observe("I'm building a Rust compiler", "Cool! What stage?")
    print(lib.summary())          # markdown for your system prompt
    print(lib.recall("compiler")) # search stored facts

Environment variables::

    GROQ_API_KEY          — Groq API key (required)
    LIBRARIAN_MODEL       — Extraction model (default: llama-3.3-70b-versatile)
"""

from __future__ import annotations

__version__ = "0.2.0"

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Internal modules
from librarian._tools import (
    RECALL_SCHEMA,
    BANKS_SCHEMA,
    COMMITMENTS_SCHEMA,
    EXTRACTION_MODELS,
    EMPTY_EXTRACTION,
    DEDUP_THRESHOLD,
    DEFAULT_TEMPORAL_TTL,
)
from librarian._extraction import (
    EXTRACTION_SYSTEM,
    EXTRACTION_USER,
    _call_groq,
)
from librarian._dedup import _is_duplicate, DedupIndex
from librarian._store import LibrarianStore
from librarian._provider import LibrarianMemoryProvider

__all__ = [
    "Librarian",
    "LibrarianMemoryProvider",
    "LibrarianStore",
    "EXTRACTION_SYSTEM",
    "register",
]

logger = logging.getLogger(__name__)

# Queue size for Librarian standalone worker
_QUEUE_MAX = 50


# ---------------------------------------------------------------------------
# Framework-agnostic API
# ---------------------------------------------------------------------------

class Librarian:
    """Simple, universal interface — works with any agent framework or none.

    Usage::

        from librarian import Librarian

        lib = Librarian(api_key="your-groq-key", store_path="./memory")

        # after each conversation turn
        lib.observe("I'm building a Rust compiler", "Cool! What stage?")

        # before the next turn — get context for the system prompt
        print(lib.summary())

        # search for specific memories
        results = lib.recall("compiler")

        # get OpenAI-compatible tool schemas to expose to your agent
        tools = lib.tool_schemas()
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        store_path: str | Path = "",
        search_mode: str = "text",
    ):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Groq API key required. Pass api_key= or set GROQ_API_KEY env var. "
                "Get one at https://console.groq.com/keys"
            )
        self._model = model
        store = Path(store_path) if store_path else Path.home() / ".librarian"
        self._store = LibrarianStore(store, search_mode=search_mode)

        # Queue-based worker for non-blocking observe()
        self._work_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="librarian-observe-worker"
        )
        self._worker_thread.start()

        logger.info("[librarian] Ready — store=%s, model=%s", store, model)

    def _worker_loop(self) -> None:
        """Process extraction tasks sequentially from the queue."""
        while not self._shutdown_event.is_set():
            try:
                task = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                task()
            except Exception as e:
                logger.warning("[librarian] Worker task failed: %s", e)
            finally:
                self._work_queue.task_done()

    def _enqueue(self, task) -> None:
        """Add task to queue. Drop oldest on overflow."""
        try:
            self._work_queue.put_nowait(task)
        except queue.Full:
            try:
                self._work_queue.get_nowait()
                logger.warning("[librarian] Queue full, dropped oldest task")
            except queue.Empty:
                pass
            try:
                self._work_queue.put_nowait(task)
            except queue.Full:
                logger.warning("[librarian] Failed to enqueue task even after drop")

    @property
    def store(self) -> LibrarianStore:
        """Direct access to the underlying memory store."""
        return self._store

    def observe(self, user_message: str, agent_response: str, *, blocking: bool = False) -> None:
        """Extract and store facts from a conversation turn.

        By default runs async in a background thread (non-blocking).
        Set ``blocking=True`` to wait for extraction to complete.
        """
        def _run():
            result = _call_groq(self._api_key, self._model, EXTRACTION_SYSTEM, EXTRACTION_USER.format(
                today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                user_message=user_message,
                agent_response=agent_response,
            ))
            if result["facts"]:
                self._store.add_facts(result["facts"])
            if result["commitments"]:
                self._store.add_commitments(result["commitments"])
            if result["entities"]:
                self._store.add_entities(result["entities"])

        if blocking:
            _run()
        else:
            self._enqueue(_run)

    def recall(self, query: str, *, bank: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search memory banks for facts matching a query.

        Returns a list of fact dicts sorted by relevance.
        """
        results = self._store.search_facts(query, bank=bank)
        return results[:limit]

    def summary(self) -> str:
        """Build a markdown summary of all stored memories.

        Suitable for injecting into a system prompt.
        """
        return self._store.build_summary()

    def banks(self) -> Dict[str, int]:
        """Return ``{bank_name: fact_count}`` for all memory banks."""
        return self._store.get_banks()

    def commitments(self, *, active_only: bool = True) -> List[Dict[str, Any]]:
        """Return tracked commitments."""
        if active_only:
            return self._store.get_active_commitments()
        return self._store.get_commitments()

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible function-calling tool schemas."""
        return [RECALL_SCHEMA, BANKS_SCHEMA, COMMITMENTS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Dispatch a tool call and return a JSON-string result."""
        if tool_name == "librarian_recall":
            results = self.recall(args.get("query", ""), bank=args.get("bank"))
            if not results:
                return json.dumps({"result": f"No memories found for '{args.get('query', '')}'."})
            lines = [f"[{f.get('bank', '?')}] {f.get('text', '')}" for f in results]
            return json.dumps({"result": "\n".join(lines)})

        elif tool_name == "librarian_banks":
            bank = args.get("bank")
            if bank:
                facts = self._store.get_bank_facts(bank)
                if not facts:
                    return json.dumps({"result": f"Bank '{bank}' not found or empty."})
                lines = [f"- {f.get('text', '')}" for f in facts]
                return json.dumps({"result": f"Bank '{bank}' ({len(facts)} facts):\n" + "\n".join(lines)})
            return json.dumps({"result": "\n".join(f"- {n}: {c} facts" for n, c in self.banks().items()) or "No banks yet."})

        elif tool_name == "librarian_commitments":
            cmts = self.commitments()
            if not cmts:
                return json.dumps({"result": "No active commitments."})
            lines = [f"- [{c.get('type', 'task')}] {c.get('subject', '')}'" for c in cmts]
            return json.dumps({"result": "Active commitments:\n" + "\n".join(lines)})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ------------------------------------------------------------------
    # Forget operations
    # ------------------------------------------------------------------

    def forget(self, query: str = "", *, bank: str = "") -> int:
        """Remove matching facts. If bank is given, clears that bank.
        If query is given, removes facts matching the query.
        Returns count deleted.
        """
        if bank:
            return self._store.forget_bank(bank)
        if query:
            return self._store.forget(query)
        return 0

    def forget_all(self) -> int:
        """Wipe all memories. Returns count deleted."""
        return self._store.forget_all()

    def cleanup_expired(self) -> int:
        """Remove expired temporal facts. Returns count removed."""
        return self._store.cleanup_expired()

    def flush(self) -> None:
        """Wait for any pending background extraction to finish."""
        try:
            self._work_queue.join()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Hermes plugin registration hook
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by Hermes Agent plugin loader. Not needed for standalone use."""
    ctx.register_memory_provider(LibrarianMemoryProvider())
