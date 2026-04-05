"""Librarian MCP Server — persistent super-memory for AI agents.

Tools:
  - librarian_recall: Smart search across memory banks
  - librarian_banks: Browse banks and their contents
  - librarian_commitments: List active tasks/promises
  - librarian_observe: Extract facts from conversation text
  - librarian_inject: Full memory context for system prompt
  - librarian_forget: Remove facts
  - librarian_consolidate: Merge/clean a bank via LLM
  - librarian_stats: Memory usage statistics
  - librarian_remember: Directly store a fact (no LLM needed)

Usage:
    python -m librarian.server
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ._store import LibrarianStore
from ._extraction import extract_facts

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("LIBRARIAN_DATA_DIR", "/root/metis/data/librarian"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("librarian")

# ---------------------------------------------------------------------------
# Initialize store and MCP server
# ---------------------------------------------------------------------------

store = LibrarianStore(DATA_DIR)
mcp = FastMCP("librarian")


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def librarian_recall(query: str, bank: str = None) -> str:
    """Search memory for facts matching a query. Uses smart ranking (exact match > all words > stems > partial).

    Args:
        query: What to search for.
        bank: Optional — restrict to a specific bank (general, people, work, health, projects, preferences, decisions, finance).
    """
    results = store.search_facts(query, bank=bank)
    if not results:
        return f"No memories found for '{query}'."

    lines = []
    for f in results:
        bank_name = f.get("bank", "?")
        text = f.get("text", "")
        imp = f.get("importance", 3)
        tags = []
        if imp >= 5:
            tags.append("★")
        if f.get("durability") == "temporal":
            tags.append("temporal")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"[{bank_name}] {text}{tag_str}")

    return f"Found {len(results)} memories:\n" + "\n".join(lines)


@mcp.tool()
def librarian_banks(bank: str = None) -> str:
    """List all memory banks with fact counts, or show all facts in a specific bank.

    Args:
        bank: Optional — show facts from this specific bank.
    """
    if bank:
        facts = store.get_bank_facts(bank)
        if not facts:
            return f"Bank '{bank}' is empty or doesn't exist."
        lines = []
        for f in facts:
            imp = f.get("importance", 3)
            star = " ★" if imp >= 5 else ""
            lines.append(f"- {f.get('text', '')}{star}")
        return f"Bank '{bank}' ({len(facts)} facts):\n" + "\n".join(lines)

    banks = store.get_banks()
    if not banks:
        return "No memories stored yet."
    lines = [f"- {name}: {count} facts" for name, count in banks.items()]
    return "Memory banks:\n" + "\n".join(lines)


@mcp.tool()
def librarian_commitments() -> str:
    """List all active commitments (promises, tasks, reminders)."""
    cmts = store.get_active_commitments()
    if not cmts:
        return "No active commitments."

    lines = []
    for c in cmts:
        due = f" (due: {c['due']})" if c.get("due") else ""
        lines.append(f"- [{c.get('type', 'task')}] {c.get('subject', '')}{due}")

    return "Active commitments:\n" + "\n".join(lines)


@mcp.tool()
def librarian_observe(text: str) -> str:
    """Extract and store facts from conversation text via LLM.

    Pass conversation text (user message + agent response). The Librarian extracts 
    only HIGH-VALUE facts (importance >= 3) and stores them.

    Args:
        text: Conversation text. Format: "USER: ... AGENT: ..." or free text.
    """
    result = extract_facts(text)

    added_facts = 0
    added_cmts = 0

    if result["facts"]:
        added_facts = store.add_facts(result["facts"])
    if result["commitments"]:
        added_cmts = store.add_commitments(result["commitments"])
    if result["entities"]:
        store.add_entities(result["entities"])

    parts = []
    if added_facts:
        # Show what was stored
        fact_texts = [f['text'] for f in result['facts'][:added_facts]]
        parts.append(f"{added_facts} facts stored: " + "; ".join(fact_texts))
    if added_cmts:
        parts.append(f"{added_cmts} commitments tracked")

    if not parts:
        return "Nothing worth remembering in this text."
    return "Observed: " + " | ".join(parts)


@mcp.tool()
def librarian_remember(fact: str, bank: str = "general", importance: int = 4, durability: str = "permanent") -> str:
    """Directly store a fact without LLM extraction. Use when you know exactly what to remember.

    Args:
        fact: The fact to store (e.g. "Manu prefers dark mode").
        bank: Memory bank (general, people, work, health, projects, preferences, decisions, finance).
        importance: 3-5 (3=context, 4=active project/decision, 5=identity/correction).
        durability: "permanent" or "temporal" (temporal expires after 30 days).
    """
    added = store.add_facts([{
        "text": fact,
        "bank": bank,
        "confidence": "stated",
        "durability": durability,
        "importance": max(3, min(5, importance)),
    }])
    if added:
        return f"Remembered: {fact} → [{bank}]"
    return f"Already known (duplicate detected)."


@mcp.tool()
def librarian_inject() -> str:
    """Return full memory context block for system prompt injection.

    Builds summary of all memories sorted by importance. High-importance 
    facts (★) are always included. Call at session start to load context.
    """
    summary = store.build_summary()
    if not summary:
        return "[Librarian: No memories stored yet. I'll learn as we talk.]"
    return summary


@mcp.tool()
def librarian_forget(text: str) -> str:
    """Forget facts matching the given text across all memory banks.

    Args:
        text: Search text — all facts containing this (case-insensitive) will be removed.
    """
    removed = store.forget(text)
    if removed == 0:
        return f"No facts found matching '{text}'."
    return f"Forgot {removed} fact(s) matching '{text}'."


@mcp.tool()
def librarian_consolidate(bank: str) -> str:
    """Consolidate a memory bank — merge duplicates, remove outdated facts via LLM.

    Use periodically to keep memory clean and efficient. Banks with < 5 facts 
    are skipped (too few to consolidate).

    Args:
        bank: Which bank to consolidate (e.g. "general", "projects").
    """
    result = store.consolidate(bank)
    if "message" in result:
        return result["message"]
    return f"Consolidated '{bank}': {result['before']} facts → {result['after']} facts"


@mcp.tool()
def librarian_stats() -> str:
    """Show memory usage statistics — total facts, bank sizes, most accessed facts."""
    stats = store.get_stats()

    lines = [
        f"Total facts: {stats['total_facts']}",
        f"Active commitments: {stats['active_commitments']}",
        f"Known entities: {stats['entities']}",
        "",
        "Banks:",
    ]
    for bank, count in stats.get("banks", {}).items():
        lines.append(f"  - {bank}: {count}")

    if stats.get("importance_distribution"):
        lines.append("")
        lines.append("Importance distribution:")
        for imp in sorted(stats["importance_distribution"].keys(), reverse=True):
            count = stats["importance_distribution"][imp]
            stars = "★" * imp
            lines.append(f"  {stars} ({imp}): {count} facts")

    if stats.get("most_accessed"):
        lines.append("")
        lines.append("Most accessed facts:")
        for f in stats["most_accessed"][:5]:
            if f.get("hit_count", 0) > 0:
                lines.append(f"  [{f.get('hit_count', 0)}x] {f.get('text', '')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the Librarian MCP server on stdio transport."""
    logger.info("Librarian MCP server starting — data_dir=%s", DATA_DIR)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
