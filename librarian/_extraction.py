"""Extraction prompts and Groq LLM call with retry logic."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from librarian._tools import EMPTY_EXTRACTION, EXTRACTION_MODELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You are a memory librarian. Extract ALL meaningful facts from a conversation between a user and an AI agent.

WHAT TO EXTRACT — be thorough, capture EVERYTHING that happened:
- User facts: name, preferences, projects, relationships, skills, habits, goals, opinions, emotions, jokes
- Agent statements: what the agent said, decided, proposed, recommended, promised, joked about, or refused
- Decisions and outcomes: "User decided X", "Agent recommended Y and user agreed"
- Agreements and disagreements: "User confirmed X", "User pushed back on Y"
- Banter, insults, humor, personality: "User called agent X", "Agent roasted user about Y" — this IS meaningful context for relationship continuity
- Any context that would help continue this conversation seamlessly later

WHAT TO SKIP:
- Pure routing/mechanics with zero content ("Hi", "Thanks", "OK")
- Redundant facts — merge similar statements into one concise fact

DO NOT SKIP:
- Opinions, reactions, humor, trash-talk — these define the relationship
- Tool/technical discussion — if the user or agent talked about HOW something works, that's a fact
- Corrections — if the user corrected the agent, ALWAYS capture that

RULES:
- Write facts as concise third-person statements
- Distinguish who said/decided what: "User wants X" vs "Agent suggested X"
- For time-sensitive facts, include the absolute date if known (not "tomorrow")
- Keep facts atomic — one idea per fact
- Preserve tone: if user said something was "Scheiße", don't sanitize to "not ideal"

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


# ---------------------------------------------------------------------------
# Groq extraction with exponential backoff retry
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BACKOFF_SCHEDULE = [1.0, 2.0, 4.0]  # seconds


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
        for attempt in range(_MAX_RETRIES):
            try:
                logger.debug("[librarian] Trying extraction with %s (attempt %d)", m, attempt + 1)
                response = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM},
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
                # Check for rate limit error with Retry-After header
                retry_after = None
                try:
                    from groq import RateLimitError
                    if isinstance(exc, RateLimitError):
                        # Try to get Retry-After from response headers
                        if hasattr(exc, 'response') and exc.response is not None:
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
                    logger.warning(
                        "[librarian] Extraction with %s attempt %d failed: %s — retrying in %.1fs",
                        m, attempt + 1, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.warning("[librarian] Extraction with %s failed after %d attempts: %s", m, _MAX_RETRIES, exc)
                    break  # try next model

    logger.error("[librarian] All extraction models failed")
    return dict(EMPTY_EXTRACTION)
