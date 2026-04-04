"""LibrarianStore — JSON-file memory store with dedup, TTL, and forget()."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from librarian._dedup import DedupIndex, _is_duplicate
from librarian._tools import DEFAULT_TEMPORAL_TTL

logger = logging.getLogger(__name__)


class LibrarianStore:
    def __init__(self, root: Path, *, search_mode: str = "text"):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "banks").mkdir(exist_ok=True)
        # Per-bank dedup indexes, loaded lazily
        self._dedup_indexes: Dict[str, DedupIndex] = {}

        # Embedding search support
        self._search_mode = search_mode  # "text" or "embedding"
        self._embedding_index: Optional[Any] = None
        if search_mode == "embedding":
            from librarian._embeddings import EmbeddingIndex
            self._embedding_index = EmbeddingIndex()

    # ------------------------------------------------------------------
    # Dedup index management
    # ------------------------------------------------------------------

    def _get_dedup_index(self, bank: str) -> DedupIndex:
        """Get or build the dedup index for a bank."""
        if bank not in self._dedup_indexes:
            idx = DedupIndex()
            bank_file = self.root / "banks" / f"{bank}.json"
            data = self._read(bank_file)
            if data:
                texts = [f["text"] for f in data.get("facts", [])]
                idx.load(texts)
            self._dedup_indexes[bank] = idx
        return self._dedup_indexes[bank]

    def _invalidate_dedup_index(self, bank: str) -> None:
        """Remove cached dedup index (e.g. after forget)."""
        self._dedup_indexes.pop(bank, None)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_facts(self, facts: List[Dict[str, Any]]) -> int:
        """Add facts with deduplication. Returns number actually added."""
        added = 0
        for fact in facts:
            bank = fact.get("bank", "general")
            bank_file = self.root / "banks" / f"{bank}.json"
            data = self._read(bank_file) or {"bank": bank, "facts": []}

            idx = self._get_dedup_index(bank)
            if idx.is_duplicate(fact["text"]):
                logger.debug("[librarian] Skipped duplicate: %.60s", fact["text"])
                continue

            entry: Dict[str, Any] = {
                "text": fact["text"],
                "confidence": fact.get("confidence", "stated"),
                "durability": fact.get("durability", "permanent"),
                "added": datetime.now(timezone.utc).isoformat(),
            }

            # TTL for temporal facts
            durability = fact.get("durability", "permanent")
            if durability == "temporal":
                ttl = fact.get("ttl", DEFAULT_TEMPORAL_TTL)
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
                entry["expires_at"] = expires_at.isoformat()

            # Compute and cache embedding if in embedding mode
            if self._search_mode == "embedding" and self._embedding_index is not None:
                try:
                    entry["embedding"] = self._embedding_index.embed(fact["text"])
                except Exception:
                    pass  # graceful fallback — fact stored without embedding

            data["facts"].append(entry)
            self._write(bank_file, data)
            idx.add(fact["text"])
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

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_banks(self) -> Dict[str, int]:
        banks_dir = self.root / "banks"
        result = {}
        if banks_dir.exists():
            for f in sorted(banks_dir.glob("*.json")):
                data = self._read(f)
                if data:
                    result[f.stem] = len(data.get("facts", []))
        return result

    def get_bank_facts(self, bank: str, *, include_expired: bool = True) -> List[Dict[str, Any]]:
        data = self._read(self.root / "banks" / f"{bank}.json")
        facts = data.get("facts", []) if data else []
        if not include_expired:
            now = datetime.now(timezone.utc)
            facts = [f for f in facts if not self._is_expired(f, now)]
        return facts

    def get_all_facts(self, *, include_expired: bool = True) -> List[Dict[str, Any]]:
        all_facts = []
        for bank, _count in self.get_banks().items():
            for f in self.get_bank_facts(bank, include_expired=include_expired):
                f["bank"] = bank
                all_facts.append(f)
        return all_facts

    def get_commitments(self) -> List[Dict[str, Any]]:
        data = self._read(self.root / "commitments.json")
        return data.get("commitments", []) if data else []

    def get_active_commitments(self) -> List[Dict[str, Any]]:
        return [c for c in self.get_commitments() if c.get("status") == "active"]

    def search_facts(self, query: str, bank: Optional[str] = None, *, mark_expired: bool = True) -> List[Dict[str, Any]]:
        facts = self.get_bank_facts(bank) if bank else self.get_all_facts()
        now = datetime.now(timezone.utc)

        # Embedding-based search when available
        if self._search_mode == "embedding" and self._embedding_index is not None:
            try:
                results = self._embedding_index.search(query, facts, top_k=20)
            except Exception:
                # Fall back to text search on any embedding error
                results = self._text_search(query, facts)
        else:
            results = self._text_search(query, facts)

        # Mark expired facts
        if mark_expired:
            for f in results:
                if self._is_expired(f, now):
                    if not f.get("text", "").startswith("[expired]"):
                        f["text"] = "[expired] " + f.get("text", "")

        return results

    @staticmethod
    def _text_search(query: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Original text-based substring/word search."""
        q = query.lower()
        words = [w for w in q.split() if len(w) > 2]
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
        now = datetime.now(timezone.utc)
        parts = []
        banks = self.get_banks()
        if banks:
            lines = []
            for bank, count in banks.items():
                facts = self.get_bank_facts(bank)
                # Skip expired facts in summary
                active_facts = [f for f in facts if not self._is_expired(f, now)]
                if not active_facts:
                    continue
                lines.append(f"**{bank}** ({len(active_facts)} facts):")
                for f in active_facts[:8]:
                    durability = f.get("durability", "")
                    tag = f" [{durability}]" if durability == "temporal" else ""
                    lines.append(f"  - {f.get('text', '')}{tag}")
            if lines:
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

    # ------------------------------------------------------------------
    # Forget operations
    # ------------------------------------------------------------------

    def forget(self, query: str) -> int:
        """Remove facts matching a query across all banks. Returns count deleted."""
        q = query.lower()
        total_removed = 0
        banks_dir = self.root / "banks"
        if not banks_dir.exists():
            return 0
        for f in banks_dir.glob("*.json"):
            data = self._read(f)
            if not data:
                continue
            original = data.get("facts", [])
            remaining = [fact for fact in original if q not in fact.get("text", "").lower()]
            removed = len(original) - len(remaining)
            if removed:
                data["facts"] = remaining
                self._write(f, data)
                self._invalidate_dedup_index(f.stem)
                total_removed += removed
        return total_removed

    def forget_bank(self, bank: str) -> int:
        """Clear all facts from a specific bank. Returns count deleted."""
        bank_file = self.root / "banks" / f"{bank}.json"
        data = self._read(bank_file)
        if not data:
            return 0
        count = len(data.get("facts", []))
        data["facts"] = []
        self._write(bank_file, data)
        self._invalidate_dedup_index(bank)
        return count

    def forget_all(self) -> int:
        """Wipe everything. Returns total count deleted."""
        total = 0
        banks_dir = self.root / "banks"
        if banks_dir.exists():
            for f in banks_dir.glob("*.json"):
                data = self._read(f)
                if data:
                    total += len(data.get("facts", []))
                f.unlink(missing_ok=True)
        self._dedup_indexes.clear()

        # Also clear commitments and entities
        for name in ("commitments.json", "entities.json"):
            p = self.root / name
            if p.exists():
                p.unlink(missing_ok=True)

        return total

    def cleanup_expired(self) -> int:
        """Remove all expired facts from all banks. Returns count removed."""
        now = datetime.now(timezone.utc)
        total_removed = 0
        banks_dir = self.root / "banks"
        if not banks_dir.exists():
            return 0
        for f in banks_dir.glob("*.json"):
            data = self._read(f)
            if not data:
                continue
            original = data.get("facts", [])
            remaining = [fact for fact in original if not self._is_expired(fact, now)]
            removed = len(original) - len(remaining)
            if removed:
                data["facts"] = remaining
                self._write(f, data)
                self._invalidate_dedup_index(f.stem)
                total_removed += removed
        return total_removed

    # ------------------------------------------------------------------
    # TTL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_expired(fact: Dict[str, Any], now: Optional[datetime] = None) -> bool:
        """Check if a fact has expired based on its expires_at field."""
        expires_at = fact.get("expires_at")
        if not expires_at:
            return False
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            exp = datetime.fromisoformat(expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return now > exp
        except (ValueError, TypeError):
            return False

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _read(self, path: Path) -> Optional[Dict]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write(self, path: Path, data: Dict) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
