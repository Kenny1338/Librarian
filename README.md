# hermes-librarian

Real-time memory extraction plugin for [Hermes Agent](https://github.com/hermes-agent/hermes-agent). A fast LLM sidecar (Groq) observes every conversation turn, extracts facts, commitments, and entities, and persists them into categorised memory banks — giving your agent persistent recall across sessions.

## The Problem

LLM agents suffer from **cold-start amnesia**: every new session starts with zero context about the user. Stuffing entire conversation logs into the context window is expensive and noisy. RAG helps, but requires explicit indexing and doesn't capture implicit facts ("the user prefers dark mode").

## How It Works

```
User message ──► Agent responds ──► Librarian observes (async)
                                         │
                                    Groq LLM extracts:
                                    ├─ facts (categorised into banks)
                                    ├─ commitments (tasks, promises)
                                    └─ entities (people, orgs, places)
                                         │
                                    Dedup & persist to JSON
                                         │
                                    Next turn: inject as context
```

**Two-phase context injection:**

1. **System prompt block** — overview of all stored facts (max 8 per bank)
2. **Prefetch** — keyword-matched facts relevant to the current query

The agent never has to explicitly "remember" — the Librarian handles it silently.

## Features

- **Async extraction** — runs in background threads, zero latency impact on the agent
- **Deduplication** — `SequenceMatcher`-based similarity check (0.75 threshold) prevents redundant facts
- **Durability tags** — facts are tagged `permanent` (name, preference) or `temporal` (meeting, deadline)
- **Bank categorisation** — `general`, `people`, `work`, `health`, `projects`, `preferences`, `decisions`
- **Agent + user facts** — captures what both sides said, decided, and agreed on
- **Pre-compression hook** — extracts facts from messages before Hermes compresses them away
- **Tool exposure** — `librarian_recall`, `librarian_banks`, `librarian_commitments` for agent self-service
- **Model fallback** — tries `llama-3.3-70b-versatile`, falls back to `llama-3.1-8b-instant`

## Installation

### As a Hermes plugin

```bash
pip install hermes-librarian
```

Then set the memory provider in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: librarian
```

And add your Groq API key to `~/.hermes/.env`:

```bash
GROQ_API_KEY=gsk_...
```

### As a standalone library

```bash
pip install hermes-librarian
```

```python
from hermes_librarian import LibrarianMemoryProvider

lib = LibrarianMemoryProvider()
lib.initialize("session-1", hermes_home="/path/to/storage")

# After a conversation turn
lib.sync_turn(
    "I'm working on a Rust compiler project",
    "That sounds interesting! What stage are you at?"
)

# On the next turn, get context
lib.queue_prefetch("How's my compiler going?")
context = lib.prefetch("How's my compiler going?")
prompt_block = lib.system_prompt_block()
```

### From source

```bash
git clone https://github.com/manu/hermes-librarian
cd hermes-librarian
pip install -e ".[dev]"
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key ([get one here](https://console.groq.com/keys)) |
| `LIBRARIAN_MODEL` | `llama-3.3-70b-versatile` | Groq model for extraction |
| `LIBRARIAN_BANK_DIR` | `$HERMES_HOME/librarian` | Override storage directory |

## Memory Banks

Facts are categorised into banks stored as individual JSON files:

```
~/.hermes/librarian/
├── banks/
│   ├── general.json
│   ├── people.json
│   ├── work.json
│   ├── health.json
│   ├── projects.json
│   ├── preferences.json
│   └── decisions.json
├── commitments.json
└── entities.json
```

Each fact includes:

```json
{
  "text": "User's name is Manu",
  "confidence": "stated",
  "durability": "permanent",
  "added": "2026-04-03T14:30:00+00:00"
}
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Hermes Agent                     │
│                                                   │
│  system prompt ◄── system_prompt_block()          │
│  turn context  ◄── prefetch()                     │
│                                                   │
│  after turn ───► sync_turn() ──► background thread│
│                                      │            │
│                                 Groq API call     │
│                                      │            │
│                                 LibrarianStore    │
│                                 ├── add_facts()   │
│                                 ├── add_commits() │
│                                 └── dedup check   │
│                                                   │
│  tools: librarian_recall, librarian_banks,        │
│         librarian_commitments                     │
└──────────────────────────────────────────────────┘
```

## Extraction Prompt

The Librarian extracts:

- **User facts** — name, preferences, projects, relationships, skills, habits, goals
- **Agent statements** — what the agent said, decided, proposed, recommended, or promised
- **Decisions and outcomes** — "User decided X", "Agent recommended Y and user agreed"
- **Agreements** — "User confirmed the API design is OK"
- **Commitments** — tasks, promises, reminders with optional due dates
- **Entities** — people, organisations, places mentioned

It explicitly skips pleasantries, meta-commentary about conversation mechanics, and internal system details.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
