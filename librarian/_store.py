"""LibrarianStore — JSON-file memory store with dedup, TTL, search, and consolidation.

Data layout under root/:
    banks/{bank_name}.json   — fact banks
    commitments.json         — tracked commitments
    entities.json            — known entities
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._dedup import DedupIndex, _is_duplicate

logger = logging.getLogger(__name__)

# Default TTL for temporal facts — 30 days
DEFAULT_TEMPORAL_TTL = 30 * 24 * 60 * 60


class LibrarianStore:
    """JSON-file backed memory store with per-bank dedup indexes."""

    def __init__(self, root) -> None:
        self.root = Path(root) if not isinstance(root, Path) else root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "banks").mkdir(exist_ok=True)
        self._dedup_indexes: Dict[str, DedupIndex] = {}

    # ------------------------------------------------------------------
    # Dedup index management
    # ------------------------------------------------------------------

    def _get_dedup_index(self, bank: str) -> DedupIndex:
        if bank not in self._dedup_indexes:
            idx = DedupIndex()
            data = self._read(self.root / "banks" / f"{bank}.json")
            if data:
                idx.load([f["text"] for f in data.get("facts", [])])
            self._dedup_indexes[bank] = idx
        return self._dedup_indexes[bank]

    def _invalidate_dedup_index(self, bank: str) -> None:
        self._dedup_indexes.pop(bank, None)

    # ------------------------------------------------------------------
    # Core write operations
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
                logger.debug("Skipped duplicate: %.60s", fact["text"])
                continue

            entry: Dict[str, Any] = {
                "text": fact["text"],
                "confidence": fact.get("confidence", "stated"),
                "durability": fact.get("durability", "permanent"),
                "importance": fact.get("importance", 3),
                "added": datetime.now(timezone.utc).isoformat(),
                "hit_count": 0,  # Track how often this fact is retrieved
            }

            if fact.get("durability") == "temporal":
                ttl = fact.get("ttl", DEFAULT_TEMPORAL_TTL)
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
                entry["expires_at"] = expires_at.isoformat()

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
        """Add named entities with deduplication."""
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
    # Smart search — multi-strategy ranking
    # ------------------------------------------------------------------

    # Common German synonyms/related terms for semantic-ish matching
    _RELATED_TERMS: Dict[str, List[str]] = {
        "wohnt": ["lebt", "wohnung", "adresse", "umgezogen", "zuhause", "wohnort"],
        "lebt": ["wohnt", "wohnung", "umgezogen", "zuhause", "wohnort"],
        "umzug": ["umgezogen", "wohnung", "neue", "lebt", "wohnt"],
        "kosten": ["preis", "geld", "zahlen", "euro", "budget", "usage", "billing", "bezahlt"],
        "geld": ["kosten", "finanzen", "euro", "budget", "zahlen", "konto", "bezahlt"],
        "api": ["usage", "token", "kosten", "billing", "extra", "anthropic"],
        "arbeit": ["job", "arbeitet", "firma", "beruf", "arbeitgeber"],
        "job": ["arbeit", "arbeitet", "firma", "beruf", "arbeitgeber"],
        "familie": ["mutter", "vater", "bruder", "schwester", "eltern"],
        "mutter": ["familie", "mama", "eltern"],
        "projekt": ["projekte", "arbeitet", "baut", "entwickelt", "migration"],
        "migration": ["migrieren", "umgebaut", "wechsel", "umzug", "hermes", "metis"],
    }

    def search_facts(self, query: str, bank: str | None = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search facts using multi-strategy ranking.
        
        Strategies (in order of weight):
        1. Exact phrase match (weight: 5)
        2. All words present (weight: 4) 
        3. Word stem matching (weight: 3)
        4. Related terms / synonym matching (weight: 2.5)
        5. Partial word matching (weight: 2)
        6. Importance boost (+0.5 per importance level above 3)
        7. Hit count boost (frequently accessed = more relevant)
        """
        facts = self.get_bank_facts(bank) if bank else self.get_all_facts(include_expired=False)
        if not facts:
            return []

        q = query.lower().strip()
        words = [w for w in q.split() if len(w) > 1]
        stems = [self._stem(w) for w in words]
        
        # Expand query with related terms
        expanded_words = set(words)
        for w in words:
            w_lower = w.lower()
            stem = self._stem(w_lower)
            # Check direct match and stem match in related terms
            for key, related in self._RELATED_TERMS.items():
                if w_lower == key or stem == self._stem(key) or w_lower in related:
                    expanded_words.update(related)
                    expanded_words.add(key)

        scored = []
        for f in facts:
            text = f.get("text", "").lower()
            text_words = text.split()
            score = 0.0

            # Strategy 1: Exact phrase match
            if q in text:
                score += 5.0
            
            # Strategy 2: All query words present
            elif words and all(w in text for w in words):
                score += 4.0
            
            else:
                # Strategy 3: Stem matching
                text_stems = [self._stem(w) for w in text_words]
                stem_hits = sum(1 for s in stems if any(s == ts or (len(s) > 3 and (s in ts or ts in s)) for ts in text_stems))
                if stem_hits > 0:
                    score += 2.0 + (stem_hits / max(len(stems), 1))
                
                # Strategy 4: Related terms matching
                if score == 0 and expanded_words:
                    related_hits = sum(1 for ew in expanded_words if ew in text)
                    if related_hits > 0:
                        score += 1.5 + (related_hits / max(len(expanded_words), 1))

                # Strategy 5: Partial word matching (substring in words)
                if score == 0:
                    partial_hits = sum(1 for w in words if any(w in tw or tw in w for tw in text_words if len(tw) > 2))
                    if partial_hits > 0:
                        score += 1.0 + (partial_hits / max(len(words), 1))

            if score == 0:
                continue

            # Importance boost
            importance = f.get("importance", 3)
            score += max(0, (importance - 3) * 0.5)

            # Hit count boost (log scale)
            hit_count = f.get("hit_count", 0)
            if hit_count > 0:
                import math
                score += math.log2(hit_count + 1) * 0.2

            scored.append((score, f))

        scored.sort(key=lambda x: -x[0])
        results = [f for _, f in scored[:limit]]

        # Bump hit counts for returned results
        self._bump_hit_counts(results)

        return results

    @staticmethod
    def _stem(word: str) -> str:
        """Basic stemming — strip common German/English suffixes."""
        word = word.lower().strip()
        # German suffixes first (longer to shorter)
        for suffix in ("ierung", "ungen", "heit", "keit", "lich", "isch", "ung", "igt", "ert", "est", "end", "ens", "ing", "tion", "ment", "ness", "able", "ible", "ous", "ive", "en", "er", "es", "ed", "em", "ly", "st", "te", "et"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[:-len(suffix)]
        return word

    def _bump_hit_counts(self, facts: List[Dict[str, Any]]) -> None:
        """Increment hit_count for facts that were returned in search results."""
        # Group by bank
        by_bank: Dict[str, List[str]] = {}
        for f in facts:
            bank = f.get("bank", "general")
            by_bank.setdefault(bank, []).append(f.get("text", ""))

        for bank, texts in by_bank.items():
            bank_file = self.root / "banks" / f"{bank}.json"
            data = self._read(bank_file)
            if not data:
                continue
            changed = False
            for stored_fact in data.get("facts", []):
                if stored_fact.get("text", "") in texts:
                    stored_fact["hit_count"] = stored_fact.get("hit_count", 0) + 1
                    changed = True
            if changed:
                self._write(bank_file, data)

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_banks(self) -> Dict[str, int]:
        """Return {bank_name: fact_count} for all banks."""
        banks_dir = self.root / "banks"
        result = {}
        if banks_dir.exists():
            for f in sorted(banks_dir.glob("*.json")):
                data = self._read(f)
                if data:
                    count = len([
                        fact for fact in data.get("facts", [])
                        if not self._is_expired(fact)
                    ])
                    if count > 0:
                        result[f.stem] = count
        return result

    def get_bank_facts(self, bank: str, *, include_expired: bool = False) -> List[Dict[str, Any]]:
        """Return all facts from a specific bank."""
        data = self._read(self.root / "banks" / f"{bank}.json")
        facts = data.get("facts", []) if data else []
        if not include_expired:
            now = datetime.now(timezone.utc)
            facts = [f for f in facts if not self._is_expired(f, now)]
        # Add bank field
        for f in facts:
            f["bank"] = bank
        return facts

    def get_all_facts(self, *, include_expired: bool = False) -> List[Dict[str, Any]]:
        """Return all facts across all banks, with 'bank' field added."""
        all_facts = []
        for bank in self.get_banks():
            all_facts.extend(self.get_bank_facts(bank, include_expired=include_expired))
        return all_facts

    def get_commitments(self) -> List[Dict[str, Any]]:
        data = self._read(self.root / "commitments.json")
        return data.get("commitments", []) if data else []

    def get_active_commitments(self) -> List[Dict[str, Any]]:
        return [c for c in self.get_commitments() if c.get("status") == "active"]

    def get_entities(self) -> List[Dict[str, Any]]:
        data = self._read(self.root / "entities.json")
        return data.get("entities", []) if data else []

    def get_stats(self) -> Dict[str, Any]:
        """Return overall stats about the memory store."""
        banks = self.get_banks()
        total_facts = sum(banks.values())
        all_facts = self.get_all_facts()
        
        importance_dist = {}
        for f in all_facts:
            imp = f.get("importance", 3)
            importance_dist[imp] = importance_dist.get(imp, 0) + 1

        return {
            "total_facts": total_facts,
            "banks": banks,
            "active_commitments": len(self.get_active_commitments()),
            "entities": len(self.get_entities()),
            "importance_distribution": importance_dist,
            "most_accessed": sorted(all_facts, key=lambda f: f.get("hit_count", 0), reverse=True)[:5],
        }

    # ------------------------------------------------------------------
    # Summary for system prompt
    # ------------------------------------------------------------------

    def build_summary(self, max_facts_per_bank: int = 15) -> str:
        """Build a markdown summary for system prompt injection.
        
        Facts are sorted by importance (desc) then recency (desc).
        High-importance facts (5) always included. Lower importance 
        facts fill remaining slots.
        """
        now = datetime.now(timezone.utc)
        parts = []
        banks = self.get_banks()

        if banks:
            lines = []
            for bank in banks:
                facts = self.get_bank_facts(bank)
                if not facts:
                    continue

                # Sort: importance desc, then recency desc
                facts.sort(key=lambda f: (
                    -f.get("importance", 3),
                    -(f.get("hit_count", 0)),
                    f.get("added", ""),
                ), reverse=False)
                # Reverse because we sorted importance descending
                facts.sort(key=lambda f: (-f.get("importance", 3), -f.get("hit_count", 0)))

                display_facts = facts[:max_facts_per_bank]
                lines.append(f"**{bank}** ({len(facts)} facts):")
                for f in display_facts:
                    imp = f.get("importance", 3)
                    durability = f.get("durability", "")
                    tags = []
                    if durability == "temporal":
                        tags.append("temporal")
                    if imp >= 5:
                        tags.append("★")
                    tag_str = f" [{', '.join(tags)}]" if tags else ""
                    lines.append(f"  - {f.get('text', '')}{tag_str}")

            if lines:
                parts.append("## What You Remember\n" + "\n".join(lines))

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
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, bank: str) -> Dict[str, int]:
        """Consolidate a bank — merge duplicates, remove outdated facts via LLM.
        
        Returns {"before": N, "after": M}.
        """
        from ._extraction import consolidate_bank

        facts = self.get_bank_facts(bank, include_expired=True)
        if len(facts) < 5:
            return {"before": len(facts), "after": len(facts), "message": "Too few facts to consolidate"}

        before = len(facts)
        new_facts = consolidate_bank(bank, facts)
        
        if not new_facts or len(new_facts) >= before:
            return {"before": before, "after": before, "message": "No consolidation needed"}

        # Replace bank contents
        bank_file = self.root / "banks" / f"{bank}.json"
        data = {"bank": bank, "facts": []}
        for f in new_facts:
            data["facts"].append({
                "text": f.get("text", ""),
                "confidence": f.get("confidence", "stated"),
                "durability": f.get("durability", "permanent"),
                "importance": f.get("importance", 3),
                "added": f.get("added", datetime.now(timezone.utc).isoformat()),
                "hit_count": 0,
            })
        self._write(bank_file, data)
        self._invalidate_dedup_index(bank)

        return {"before": before, "after": len(new_facts)}

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
    def _is_expired(fact: Dict[str, Any], now: datetime | None = None) -> bool:
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

    def _read(self, path: Path) -> Dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write(self, path: Path, data: Dict) -> None:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
