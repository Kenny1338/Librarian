"""Librarian — persistent memory for AI agents, powered by Groq.

Observes conversation turns, extracts facts / commitments / entities via a
fast LLM sidecar, and stores them in categorised memory banks.  Before each
turn it assembles a context packet so the agent has instant recall.

Works with **any agent framework** (OpenAI, LangChain, Claude, etc.) or standalone.
Also ships as a native **Hermes Agent plugin**.

Quick start::

    from hermes_librarian import Librarian

    lib = Librarian(api_key="gsk_...")
    lib.observe("I'm building a Rust compiler", "Cool! What stage?")
    print(lib.summary())          # markdown for your system prompt
    print(lib.recall("compiler")) # search stored facts

Environment variables::

    GROQ_API_KEY          — Groq API key (required)
    LIBRARIAN_MODEL       — Extraction model (default: llama-3.3-70b-versatile)
"""

from __future__ import annotations

__version__ = "0.1.0"

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_librarian._compat import MemoryProvider

__all__ = [
    "Librarian",
    "LibrarianMemoryProvider",
    "LibrarianStore",
    "EXTRACTION_SYSTEM",
    "register",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt — instructs the LLM what to extract and what to ignore
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You are a memory librarian. Extract ALL meaningful facts from a conversation between a user and an AI agent.

WHAT TO EXTRACT:
- User facts: name, preferences, projects, relationships, skills, habits, goals
- Agent statements: what the agent said, decided, proposed, recommended, or promised
- Decisions and outcomes: "User decided X", "Agent recommended Y and user agreed"
- Agreements: "User confirmed the API design is OK", "Agent will use approach X"
- Any context that would help continue this conversation seamlessly later

WHAT TO SKIP:
- Generic pleasantries ("Hi", "Thanks", "How are you")
- Meta-commentary about conversation mechanics ("The user asked a question")
- How the AI system/tools/memory works internally ("The agent uses tool X")
- Redundant facts — merge similar statements into one concise fact

RULES:
- Write facts as concise third-person statements
- Distinguish who said/decided what: "User wants X" vs "Agent suggested X"
- For time-sensitive facts, include the absolute date if known (not "tomorrow")
- Keep facts atomic — one idea per fact

Return a JSON object with these keys:
- "facts": array of objects with:
  - "text" (string): the fact
  - "bank" (string): "general", "people", "work", "health", "projects", "preferences", "decisions"
  - "confidence" (string): "stated" or "inferred"
  - "durability" (string): "permanent" (name, preference, relationship) or "temporal" (meeting, deadline, task, decision)
- "commitments": array of objects with "type" (string), "subject" (string), "due" (string, ISO date if known, else "")
- "entities": array of objects with "name" (string), "type" ("person", "org", or "place")

If nothing meaningful to extract, return: {"facts": [], "commitments": [], "entities": []}

Always return valid JSON. No explanations."""

EXTRACTION_USER = """Today's date: {today}

USER: {user_message}
AGENT: {agent_response}

Extract ALL meaningful facts as JSON:"""

EXTRACTION_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

EMPTY_EXTRACTION: Dict[str, Any] = {"facts": [], "commitments": [], "entities": []}

# Similarity threshold for deduplication (0.0 = no match, 1.0 = identical)
DEDUP_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Tool schemas
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


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

def _is_duplicate(new_text: str, existing_texts: List[str], threshold: float = DEDUP_THRESHOLD) -> bool:
    """Check if new_text is too similar to any existing text."""
    new_lower = new_text.lower().strip()
    for existing in existing_texts:
        ratio = SequenceMatcher(None, new_lower, existing.lower().strip()).ratio()
        if ratio >= threshold:
            logger.debug("[librarian] Dedup: '%.50s' ~ '%.50s' (%.2f >= %.2f)", new_text, existing, ratio, threshold)
            return True
    return False


# ---------------------------------------------------------------------------
# Groq extraction (synchronous, runs in thread)
# ---------------------------------------------------------------------------

def _extract_via_groq(
    api_key: str,
    model: str,
    user_message: str,
    agent_response: str,
) -> Dict[str, Any]:
    from groq import Groq

    client = Groq(api_key=api_key)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user_prompt = EXTRACTION_USER.format(
        today=today,
        user_message=user_message,
        agent_response=agent_response,
    )

    models_to_try = [model] + [m for m in EXTRACTION_MODELS if m != model]

    for m in models_to_try:
        try:
            logger.debug("[librarian] Trying extraction with %s", m)
            response = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict, got {type(parsed).__name__}")
            for key in ("facts", "commitments", "entities"):
                if key not in parsed:
                    parsed[key] = []

            n_facts = len(parsed["facts"])
            n_cmts = len(parsed["commitments"])
            n_ents = len(parsed["entities"])
            logger.info(
                "[librarian] Extraction OK with %s: %d facts, %d commitments, %d entities",
                m, n_facts, n_cmts, n_ents,
            )
            for f in parsed["facts"][:5]:
                logger.debug(
                    "[librarian]   fact [%s/%s] %s",
                    f.get("bank", "?"), f.get("durability", "?"), f.get("text", "")[:100],
                )
            return parsed

        except Exception as exc:
            logger.warning("[librarian] Extraction with %s failed: %s", m, exc)

    logger.error("[librarian] All extraction models failed")
    return dict(EMPTY_EXTRACTION)


# ---------------------------------------------------------------------------
# Memory store (JSON files, bank-based, with deduplication)
# ---------------------------------------------------------------------------

class LibrarianStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "banks").mkdir(exist_ok=True)

    def add_facts(self, facts: List[Dict[str, Any]]) -> int:
        """Add facts with deduplication. Returns number actually added."""
        added = 0
        for fact in facts:
            bank = fact.get("bank", "general")
            bank_file = self.root / "banks" / f"{bank}.json"
            data = self._read(bank_file) or {"bank": bank, "facts": []}

            existing_texts = [f["text"] for f in data["facts"]]
            if _is_duplicate(fact["text"], existing_texts):
                logger.debug("[librarian] Skipped duplicate: %.60s", fact["text"])
                continue

            data["facts"].append({
                "text": fact["text"],
                "confidence": fact.get("confidence", "stated"),
                "durability": fact.get("durability", "permanent"),
                "added": datetime.now(timezone.utc).isoformat(),
            })
            self._write(bank_file, data)
            added += 1
        return added

    def add_commitments(self, commitments: List[Dict[str, Any]]) -> int:
        """Add commitments with deduplication. Returns number actually added."""
        cmt_file = self.root / "commitments.json"
        data = self._read(cmt_file) or {"commitments": []}

        existing_subjects = [c["subject"] for c in data["commitments"]]
        added = 0
        for c in commitments:
            subject = c.get("subject", "")
            if _is_duplicate(subject, existing_subjects):
                continue
            data["commitments"].append({
                "type": c.get("type", "task"),
                "subject": subject,
                "due": c.get("due", ""),
                "status": "active",
                "created": datetime.now(timezone.utc).isoformat(),
            })
            existing_subjects.append(subject)
            added += 1

        if added:
            self._write(cmt_file, data)
        return added

    def add_entities(self, entities: List[Dict[str, Any]]) -> None:
        ent_file = self.root / "entities.json"
        data = self._read(ent_file) or {"entities": []}
        existing = {e["name"].lower() for e in data["entities"]}
        for e in entities:
            if e["name"].lower() not in existing:
                data["entities"].append({
                    "name": e["name"],
                    "type": e.get("type", "person"),
                    "added": datetime.now(timezone.utc).isoformat(),
                })
                existing.add(e["name"].lower())
        self._write(ent_file, data)

    def get_banks(self) -> Dict[str, int]:
        banks_dir = self.root / "banks"
        result = {}
        if banks_dir.exists():
            for f in sorted(banks_dir.glob("*.json")):
                data = self._read(f)
                if data:
                    result[f.stem] = len(data.get("facts", []))
        return result

    def get_bank_facts(self, bank: str) -> List[Dict[str, Any]]:
        data = self._read(self.root / "banks" / f"{bank}.json")
        return data.get("facts", []) if data else []

    def get_all_facts(self) -> List[Dict[str, Any]]:
        all_facts = []
        for bank, _count in self.get_banks().items():
            for f in self.get_bank_facts(bank):
                f["bank"] = bank
                all_facts.append(f)
        return all_facts

    def get_commitments(self) -> List[Dict[str, Any]]:
        data = self._read(self.root / "commitments.json")
        return data.get("commitments", []) if data else []

    def get_active_commitments(self) -> List[Dict[str, Any]]:
        return [c for c in self.get_commitments() if c.get("status") == "active"]

    def search_facts(self, query: str, bank: Optional[str] = None) -> List[Dict[str, Any]]:
        q = query.lower()
        words = [w for w in q.split() if len(w) > 2]
        facts = self.get_bank_facts(bank) if bank else self.get_all_facts()
        scored = []
        for f in facts:
            text = f.get("text", "").lower()
            if q in text:
                scored.append((3, f))
            else:
                word_hits = sum(1 for w in words if w in text)
                if word_hits > 0:
                    scored.append((word_hits, f))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:20]]

    def build_summary(self) -> str:
        parts = []
        banks = self.get_banks()
        if banks:
            lines = []
            for bank, count in banks.items():
                facts = self.get_bank_facts(bank)
                lines.append(f"**{bank}** ({count} facts):")
                for f in facts[:8]:
                    durability = f.get("durability", "")
                    tag = f" [{durability}]" if durability == "temporal" else ""
                    lines.append(f"  - {f.get('text', '')}{tag}")
            parts.append("## What You Remember About The User\n" + "\n".join(lines))

        active_cmts = self.get_active_commitments()
        if active_cmts:
            cmt_lines = []
            for c in active_cmts[:10]:
                due = f" (due: {c['due']})" if c.get("due") else ""
                cmt_lines.append(f"- [{c.get('type', 'task')}] {c.get('subject', '')}{due}")
            parts.append("## Active Commitments\n" + "\n".join(cmt_lines))

        if not parts:
            return ""
        return "\n\n".join(parts)

    def _read(self, path: Path) -> Optional[Dict]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write(self, path: Path, data: Dict) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class LibrarianMemoryProvider(MemoryProvider):
    def __init__(self):
        self._api_key: str = ""
        self._model: str = "llama-3.3-70b-versatile"
        self._store: Optional[LibrarianStore] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._turn_count: int = 0
        self._session_id: str = ""

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
        logger.info("[librarian] Turn %d queued for extraction (user: %.60s...)", turn, user_content)

        def _sync():
            start = time.monotonic()
            try:
                result = _extract_via_groq(
                    self._api_key, self._model,
                    user_content, assistant_content,
                )

                added_facts = 0
                added_cmts = 0
                if result["facts"]:
                    added_facts = self._store.add_facts(result["facts"])
                if result["commitments"]:
                    added_cmts = self._store.add_commitments(result["commitments"])
                if result["entities"]:
                    self._store.add_entities(result["entities"])

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

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="librarian-sync")
        self._sync_thread.start()

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        logger.debug("[librarian] Turn %d starting: %.60s...", turn_number, message)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            logger.info("[librarian] Waiting for final extraction to complete...")
            self._sync_thread.join(timeout=15.0)

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
        for t in (self._sync_thread, self._prefetch_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        if self._store:
            stats = self._store.get_banks()
            logger.info("[librarian] Shutdown. Banks: %s", stats)


# ---------------------------------------------------------------------------
# Framework-agnostic API
# ---------------------------------------------------------------------------

class Librarian:
    """Simple, universal interface — works with any agent framework or none.

    Usage::

        from hermes_librarian import Librarian

        lib = Librarian(api_key="gsk_...", store_path="./memory")

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
    ):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Groq API key required. Pass api_key= or set GROQ_API_KEY env var. "
                "Get one at https://console.groq.com/keys"
            )
        self._model = model
        store = Path(store_path) if store_path else Path.home() / ".librarian"
        self._store = LibrarianStore(store)
        self._sync_thread: Optional[threading.Thread] = None
        logger.info("[librarian] Ready — store=%s, model=%s", store, model)

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
            result = _extract_via_groq(self._api_key, self._model, user_message, agent_response)
            if result["facts"]:
                self._store.add_facts(result["facts"])
            if result["commitments"]:
                self._store.add_commitments(result["commitments"])
            if result["entities"]:
                self._store.add_entities(result["entities"])

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

        if blocking:
            _run()
        else:
            self._sync_thread = threading.Thread(target=_run, daemon=True, name="librarian-observe")
            self._sync_thread.start()

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
        """Return OpenAI-compatible function-calling tool schemas.

        Plug these into any framework that supports OpenAI-style tools:
        OpenAI SDK, LangChain, LlamaIndex, Anthropic, etc.
        """
        return [RECALL_SCHEMA, BANKS_SCHEMA, COMMITMENTS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Dispatch a tool call and return a JSON-string result.

        Use with the schemas from :meth:`tool_schemas`.
        """
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
            lines = [f"- [{c.get('type', 'task')}] {c.get('subject', '')}" for c in cmts]
            return json.dumps({"result": "Active commitments:\n" + "\n".join(lines)})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def flush(self) -> None:
        """Wait for any pending background extraction to finish."""
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=15.0)


# ---------------------------------------------------------------------------
# Hermes plugin registration hook
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by Hermes Agent plugin loader. Not needed for standalone use."""
    ctx.register_memory_provider(LibrarianMemoryProvider())
