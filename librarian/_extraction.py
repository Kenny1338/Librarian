"""Groq-based fact extraction from conversation text.

Uses Groq LLM to extract facts, commitments, and entities from
conversation turns. Includes retry logic with exponential backoff
and model fallback.

QUALITY OVER QUANTITY — only extract what's worth remembering long-term.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Models in order of preference
EXTRACTION_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

EMPTY_EXTRACTION: Dict[str, Any] = {"facts": [], "commitments": [], "entities": []}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You are a memory librarian for a personal AI assistant called Metis. Your job is to decide what's WORTH REMEMBERING from a conversation.

Think like a real human memory: you don't remember every word someone said. You remember the IMPORTANT stuff — who people are, what they care about, what decisions were made, what needs to happen next.

## EXTRACT (high-value, long-term useful):
- Identity facts: name, job, timezone, relationships, where they live
- Preferences: how they like things done, communication style, pet peeves
- Decisions made: "User decided to migrate from X to Y"
- Corrections: if the user corrected the agent, ALWAYS capture this — it prevents repeating mistakes
- Projects & goals: what they're working on, what they want to achieve
- Important context: upcoming events, deadlines, financial situations
- Relationship dynamics: inside jokes, shared references, communication patterns
- Technical discoveries: "X doesn't work because Y" — saves debugging time later

## SKIP (noise, not worth storing):
- Greetings, thanks, acknowledgments
- Anything the agent said that's just doing its job (searching, reading files, etc.)
- Interim discussion that led to a final decision (only store the decision)
- Facts already known (don't re-extract what's already in memory)
- Vague or obvious statements ("AI is useful")
- Tool outputs, error messages, status updates (unless they revealed something important)
- Meta-commentary about the conversation itself ("let me check", "good question")

## QUALITY RULES:
- MAX 5 facts per conversation turn. If you have more, keep only the 5 most important.
- Write facts as crisp third-person statements: "Manu works at Cognizant Mobility"
- Each fact must be SELF-CONTAINED — readable without context
- Merge related micro-facts into one: not "User has a cat" + "Cat's name is Felix" → "User has a cat named Felix"
- Include WHO said/decided what: "Manu decided X" vs "Metis suggested X"
- For temporal facts, include the date: "Manu's lease starts March 2026" not "User's lease starts soon"
- Preserve the user's actual words for preferences: if they said "Scheiße", don't sanitize

## IMPORTANCE SCORING:
Rate each fact 1-5:
- 5: Identity, corrections, strong preferences (would be embarrassing to forget)
- 4: Active projects, decisions, upcoming deadlines
- 3: Context that helps personalization (hobbies, opinions, habits)
- 2: Mentioned once, might be relevant later
- 1: Trivia, might never come up again
Only extract facts scored 3 or higher.

Return JSON:
{
  "facts": [
    {"text": "...", "bank": "...", "confidence": "stated|inferred", "durability": "permanent|temporal", "importance": 3-5}
  ],
  "commitments": [
    {"type": "task|promise|reminder", "subject": "...", "due": "ISO date or empty"}
  ],
  "entities": [
    {"name": "...", "type": "person|org|place"}
  ]
}

Banks: general, people, work, health, projects, preferences, decisions, finance
If nothing worth remembering: {"facts": [], "commitments": [], "entities": []}
Always return valid JSON. No explanations."""

EXTRACTION_USER = """Today: {today}

USER: {user_message}
AGENT: {agent_response}

Extract only what's WORTH REMEMBERING (max 5 facts, importance >= 3):"""

# ---------------------------------------------------------------------------
# Consolidation prompt — merges related facts
# ---------------------------------------------------------------------------

CONSOLIDATION_SYSTEM = """You are a memory librarian. You have a list of facts in a memory bank that may have redundancies, outdated entries, or facts that can be merged.

Your job: CONSOLIDATE. Merge duplicates, update outdated facts with newer info, remove noise.

Rules:
- If two facts say the same thing differently, keep the better-worded one
- If a newer fact contradicts an older one, keep the newer one
- Merge related micro-facts into one: "Manu lives in Hoffenheim" + "Manu moved to Hoffenheim in March 2026" → "Manu moved to Hoffenheim in March 2026"
- Remove facts that are obviously outdated (e.g. "meeting tomorrow" from weeks ago)
- Keep the importance scores from the original facts (use the highest if merging)
- Return the consolidated list — should be SHORTER than the input

Return JSON: {"facts": [{"text": "...", "bank": "...", "confidence": "stated|inferred", "durability": "permanent|temporal", "importance": 3-5}]}"""

# ---------------------------------------------------------------------------
# Retry config
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BACKOFF_SCHEDULE = [1.0, 2.0, 4.0]


def extract_facts(
    text: str,
    *,
    api_key: str = "",
    model: str = "",
) -> Dict[str, Any]:
    """Extract facts from conversation text via Groq.

    Args:
        text: Conversation text to extract facts from.
        api_key: Groq API key (falls back to GROQ_API_KEY env var).
        model: Primary model to use (falls back to default list).

    Returns:
        Dict with keys: facts, commitments, entities.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.error("No GROQ_API_KEY available for extraction")
        return dict(EMPTY_EXTRACTION)

    model = model or os.environ.get("LIBRARIAN_MODEL", EXTRACTION_MODELS[0])

    # Try to split text into user/agent parts
    user_message = text
    agent_response = ""
    for sep in ["AGENT:", "Assistant:", "Response:", "METIS:"]:
        if sep in text:
            parts = text.split(sep, 1)
            user_message = parts[0].replace("USER:", "").replace("User:", "").strip()
            agent_response = parts[1].strip()
            break

    result = _call_groq(api_key, model, EXTRACTION_SYSTEM, EXTRACTION_USER.format(
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        user_message=user_message,
        agent_response=agent_response,
    ))

    # Filter by importance threshold
    if result.get("facts"):
        result["facts"] = [f for f in result["facts"] if f.get("importance", 3) >= 3]

    return result


def consolidate_bank(
    bank_name: str,
    facts: list,
    *,
    api_key: str = "",
    model: str = "",
) -> list:
    """Consolidate a bank's facts — merge duplicates, remove outdated entries.
    
    Returns a new (shorter) list of facts.
    """
    if len(facts) < 5:
        return facts  # Not worth consolidating

    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return facts

    model = model or os.environ.get("LIBRARIAN_MODEL", EXTRACTION_MODELS[0])

    facts_text = "\n".join(
        f"- [{f.get('durability', 'permanent')}] {f.get('text', '')} (added: {f.get('added', '?')})"
        for f in facts
    )

    user_prompt = f"Bank: {bank_name}\nFacts to consolidate:\n{facts_text}\n\nReturn consolidated JSON:"

    result = _call_groq(api_key, model, CONSOLIDATION_SYSTEM, user_prompt)

    if result.get("facts"):
        # Preserve bank name
        for f in result["facts"]:
            f["bank"] = bank_name
        return result["facts"]
    return facts


def _call_groq(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """Call Groq API with retry and model fallback."""
    from groq import Groq

    client = Groq(api_key=api_key)
    models_to_try = [model] + [m for m in EXTRACTION_MODELS if m != model]

    for m in models_to_try:
        for attempt in range(_MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected dict, got {type(parsed).__name__}")
                for key in ("facts", "commitments", "entities"):
                    if key not in parsed:
                        parsed[key] = []
                return parsed

            except Exception as exc:
                retry_after = None
                try:
                    from groq import RateLimitError
                    if isinstance(exc, RateLimitError):
                        if hasattr(exc, "response") and exc.response is not None:
                            retry_after = exc.response.headers.get("retry-after")
                            if retry_after:
                                try:
                                    retry_after = float(retry_after)
                                except (ValueError, TypeError):
                                    retry_after = None
                except ImportError:
                    pass

                if attempt < _MAX_RETRIES - 1:
                    wait = retry_after if retry_after else _BACKOFF_SCHEDULE[attempt]
                    logger.warning("Groq %s attempt %d failed: %s — retrying in %.1fs", m, attempt + 1, exc, wait)
                    time.sleep(wait)
                else:
                    logger.warning("Groq %s failed after %d attempts: %s", m, _MAX_RETRIES, exc)
                    break

    logger.error("All Groq models failed")
    return dict(EMPTY_EXTRACTION)
