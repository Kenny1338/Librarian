"""LibrarianMemoryProvider — Hermes Agent MemoryProvider implementation.

Uses a persistent worker thread + queue.Queue for non-blocking observe/sync_turn.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from librarian._compat import MemoryProvider
from librarian._extraction import _extract_via_groq
from librarian._store import LibrarianStore
from librarian._tools import RECALL_SCHEMA, BANKS_SCHEMA, COMMITMENTS_SCHEMA

logger = logging.getLogger(__name__)

# Max queue size for pending extractions
_QUEUE_MAX = 50


class LibrarianMemoryProvider(MemoryProvider):
    def __init__(self):
        self._api_key: str = ""
        self._model: str = "llama-3.3-70b-versatile"
        self._store: Optional[LibrarianStore] = None
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._turn_count: int = 0
        self._session_id: str = ""
        # Queue-based worker for non-blocking sync
        self._work_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

    def _start_worker(self) -> None:
        """Start the persistent worker thread if not already running."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="librarian-worker"
        )
        self._worker_thread.start()

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
            # Drop oldest
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
    def name(self) -> str:
        return "librarian"

    def is_available(self) -> bool:
        return bool(os.environ.get("GROQ_API_KEY", ""))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "Groq API key for the Librarian extraction model",
                "secret": True,
                "required": True,
                "env_var": "GROQ_API_KEY",
                "url": "https://console.groq.com/keys",
            },
            {
                "key": "model",
                "description": "Groq model for extraction",
                "default": "llama-3.3-70b-versatile",
                "choices": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_dir = Path(hermes_home) / "librarian"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._api_key = os.environ.get("GROQ_API_KEY", "")
        self._model = os.environ.get("LIBRARIAN_MODEL", "llama-3.3-70b-versatile")

        hermes_home = kwargs.get("hermes_home", "")
        if hermes_home:
            store_dir = Path(hermes_home) / "librarian"
        else:
            try:
                from hermes_constants import get_hermes_home  # type: ignore[import-untyped]
                store_dir = get_hermes_home() / "librarian"
            except ImportError:
                store_dir = Path.home() / ".hermes" / "librarian"

        self._store = LibrarianStore(store_dir)
        self._start_worker()

        stats = self._store.get_banks()
        total = sum(stats.values())
        logger.info(
            "[librarian] Initialized: %d banks, %d total facts, model=%s, store=%s",
            len(stats), total, self._model, store_dir,
        )

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        summary = self._store.build_summary()
        if not summary:
            return (
                "# Librarian Memory\n"
                "Active. No memories stored yet — they'll be extracted automatically from conversations.\n"
                "Use librarian_banks to browse, librarian_recall to search, librarian_commitments to check tasks."
            )
        return (
            "# Librarian Memory\n"
            "The following memories were automatically extracted from past conversations.\n\n"
            + summary
            + "\n\nUse librarian_recall to search for more specific memories."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Librarian Context\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._store or not self._api_key:
            return

        def _run():
            try:
                results = self._store.search_facts(query)
                if results:
                    text = "\n".join(f"- [{f.get('bank', '?')}] {f.get('text', '')}" for f in results[:10])
                    with self._prefetch_lock:
                        self._prefetch_result = text
                    logger.debug("[librarian] Prefetched %d facts for query: %.60s", len(results), query)
            except Exception as e:
                logger.debug("[librarian] Prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="librarian-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._api_key or not self._store:
            return

        self._turn_count += 1
        turn = self._turn_count
        store = self._store
        api_key = self._api_key
        model = self._model
        logger.info("[librarian] Turn %d queued for extraction (user: %.60s...)", turn, user_content)

        def _sync():
            start = time.monotonic()
            try:
                result = _extract_via_groq(api_key, model, user_content, assistant_content)

                added_facts = 0
                added_cmts = 0
                if result["facts"]:
                    added_facts = store.add_facts(result["facts"])
                if result["commitments"]:
                    added_cmts = store.add_commitments(result["commitments"])
                if result["entities"]:
                    store.add_entities(result["entities"])

                elapsed = time.monotonic() - start
                extracted = len(result["facts"])
                logger.info(
                    "[librarian] Turn %d done in %.1fs: extracted %d facts (added %d, deduped %d), %d commitments (added %d)",
                    turn, elapsed,
                    extracted, added_facts, extracted - added_facts,
                    len(result["commitments"]), added_cmts,
                )
            except Exception as e:
                logger.warning("[librarian] Turn %d extraction failed: %s", turn, e)

        # Non-blocking: enqueue the task
        self._start_worker()
        self._enqueue(_sync)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        logger.debug("[librarian] Turn %d starting: %.60s...", turn_number, message)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # Wait for pending work to finish
        try:
            self._work_queue.join()
        except Exception:
            pass

        if self._store:
            stats = self._store.get_banks()
            total = sum(stats.values())
            logger.info(
                "[librarian] Session ended. Total: %d banks, %d facts, %d active commitments",
                len(stats), total, len(self._store.get_active_commitments()),
            )

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._api_key or not self._store:
            return ""

        user_texts = []
        agent_texts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            if role == "user":
                user_texts.append(content[:500])
            elif role == "assistant":
                agent_texts.append(content[:500])

        if user_texts and agent_texts:
            combined_user = " | ".join(user_texts[-3:])
            combined_agent = " | ".join(agent_texts[-3:])
            logger.info("[librarian] Pre-compress extraction from %d messages", len(messages))
            try:
                result = _extract_via_groq(self._api_key, self._model, combined_user, combined_agent)
                if result["facts"]:
                    added = self._store.add_facts(result["facts"])
                    return f"Librarian extracted {added} new facts before compression."
            except Exception as e:
                logger.warning("[librarian] Pre-compress extraction failed: %s", e)

        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, BANKS_SCHEMA, COMMITMENTS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._store:
            return json.dumps({"error": "Librarian not initialized"})

        if tool_name == "librarian_recall":
            query = args.get("query", "")
            bank = args.get("bank")
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            results = self._store.search_facts(query, bank=bank)
            if not results:
                return json.dumps({"result": f"No memories found for '{query}'."})
            lines = [f"[{f.get('bank', '?')}] {f.get('text', '')}" for f in results]
            return json.dumps({"result": "\n".join(lines)})

        elif tool_name == "librarian_banks":
            bank = args.get("bank")
            if bank:
                facts = self._store.get_bank_facts(bank)
                if not facts:
                    return json.dumps({"result": f"Bank '{bank}' not found or empty."})
                lines = []
                for f in facts:
                    dur = f" [{f['durability']}]" if f.get("durability") == "temporal" else ""
                    lines.append(f"- {f.get('text', '')}{dur}")
                return json.dumps({"result": f"Bank '{bank}' ({len(facts)} facts):\n" + "\n".join(lines)})
            banks = self._store.get_banks()
            if not banks:
                return json.dumps({"result": "No memory banks yet."})
            lines = [f"- {name}: {count} facts" for name, count in banks.items()]
            return json.dumps({"result": "Memory banks:\n" + "\n".join(lines)})

        elif tool_name == "librarian_commitments":
            cmts = self._store.get_active_commitments()
            if not cmts:
                return json.dumps({"result": "No active commitments."})
            lines = []
            for c in cmts:
                due = f" (due: {c['due']})" if c.get("due") else ""
                lines.append(f"- [{c.get('type', 'task')}] {c.get('subject', '')}{due}")
            return json.dumps({"result": "Active commitments:\n" + "\n".join(lines)})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        self._shutdown_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
        if self._store:
            stats = self._store.get_banks()
            logger.info("[librarian] Shutdown. Banks: %s", stats)
