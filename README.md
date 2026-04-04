<p align="center">
  <img src="logo.jpg" alt="Librarian" width="200">
</p>

<h1 align="center">Librarian</h1>

<p align="center">
  <strong>Persistent memory for AI agents</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/librarian-ai/"><img src="https://img.shields.io/pypi/v/librarian-ai?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/librarian-ai/"><img src="https://img.shields.io/pypi/pyversions/librarian-ai" alt="Python versions"></a>
  <a href="https://github.com/Kenny1338/Librarian/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="https://github.com/Kenny1338/Librarian/actions/workflows/ci.yml"><img src="https://github.com/Kenny1338/Librarian/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

A fast LLM sidecar (Groq) silently observes conversation turns, extracts facts, commitments, and entities, and stores them in categorised memory banks — giving any agent instant recall across sessions.

Works with **any framework** — OpenAI, LangChain, LlamaIndex, Anthropic Claude, or plain Python. Also ships as a native [Hermes Agent](https://github.com/hermes-agent/hermes-agent) plugin.

## The Problem

LLM agents suffer from cold-start amnesia: every new session starts with zero context. Stuffing full conversation logs into the prompt is expensive and noisy. RAG helps, but requires explicit indexing and misses implicit facts like "the user prefers dark mode".

The Librarian solves this by silently observing every turn and building a structured memory that persists across sessions — no manual saving, no explicit commands.

## Install

```bash
pip install librarian-ai
```

You need a [Groq API key](https://console.groq.com/keys) (free tier available).

## Quick Start

```python
from librarian import Librarian

lib = Librarian(api_key="gsk_...")

# after each conversation turn — extracts facts in the background
lib.observe("I'm Manu, working on a Rust compiler", "Nice! What stage are you at?")

# before the next turn — inject into your system prompt
print(lib.summary())
# ## What You Remember About The User
# **projects** (1 facts):
#   - User's name is Manu
#   - User is building a Rust compiler

# search for specific memories
results = lib.recall("compiler")
```

That's it. Three core methods: `observe()`, `summary()`, `recall()` — plus `forget()` when you need to erase.

## Integration Examples

### OpenAI SDK

```python
import openai
from librarian import Librarian

client = openai.OpenAI()
lib = Librarian()

messages = [{"role": "system", "content": f"You are a helpful assistant.\n\n{lib.summary()}"}]

while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[{"type": "function", "function": t} for t in lib.tool_schemas()],
    )

    assistant_msg = response.choices[0].message

    # handle tool calls
    if assistant_msg.tool_calls:
        for tc in assistant_msg.tool_calls:
            result = lib.handle_tool_call(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
        continue

    reply = assistant_msg.content
    messages.append({"role": "assistant", "content": reply})
    print(reply)

    # observe the turn — extraction happens in background
    lib.observe(user_input, reply)
```

### Anthropic Claude

```python
import anthropic
from librarian import Librarian

client = anthropic.Anthropic()
lib = Librarian()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=f"You are a helpful assistant.\n\n{lib.summary()}",
    messages=[{"role": "user", "content": user_input}],
    tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
           for t in lib.tool_schemas()],
)

# after the turn
lib.observe(user_input, response.content[0].text)
```

### LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from librarian import Librarian

lib = Librarian()
llm = ChatOpenAI(model="gpt-4o")

messages = [SystemMessage(content=lib.summary()), HumanMessage(content="What's my name?")]
response = llm.invoke(messages)

lib.observe("What's my name?", response.content)
```

### Plain Python (no framework)

```python
from librarian import Librarian

lib = Librarian(store_path="./my-memory")

# manually feed conversation data
lib.observe("My meeting with Lisa is tomorrow at 10", "Got it, I'll remind you!")
lib.observe("I prefer dark mode in all apps", "Noted!")

lib.flush()  # wait for background extraction

print(lib.banks())
# {'general': 1, 'preferences': 1, 'people': 1}

print(lib.recall("Lisa"))
# [{'text': 'User has a meeting with Lisa on 2026-04-04 at 10:00', 'bank': 'people', ...}]

print(lib.commitments())
# [{'type': 'reminder', 'subject': 'Meeting with Lisa', 'due': '2026-04-04T10:00', ...}]
```

## How It Works

```
User message ──► Agent responds ──► Librarian.observe() [async]
                                         │
                                    Groq LLM extracts:
                                    ├─ facts (categorised into banks)
                                    ├─ commitments (tasks, promises)
                                    └─ entities (people, orgs, places)
                                         │
                                    Dedup & persist to JSON
                                         │
                                    Next turn: summary() / recall()
```

## API Reference

### `Librarian(*, api_key=..., model="llama-3.3-70b-versatile", store_path="", search_mode="text")`

Create a new Librarian instance. All parameters are optional.

- `api_key` — Groq API key. Falls back to `GROQ_API_KEY` env var.
- `model` — Groq model for extraction. Default: `llama-3.3-70b-versatile`.
- `store_path` — Where to store memory banks. Default: `~/.librarian`.
- `search_mode` — `"text"` (default) or `"embedding"` for semantic search.

### `.observe(user_message, agent_response, *, blocking=False)`

Extract facts from a conversation turn. Runs async by default.

### `.recall(query, *, bank=None, limit=20) → list[dict]`

Search memory banks. Returns facts sorted by relevance.

### `.summary() → str`

Markdown summary of all memories — ready for system prompt injection.

### `.banks() → dict[str, int]`

Bank names and fact counts.

### `.commitments(*, active_only=True) → list[dict]`

Tracked commitments (tasks, promises, reminders).

### `.tool_schemas() → list[dict]`

OpenAI-compatible function-calling schemas. Works with OpenAI, Anthropic, LangChain, LlamaIndex, etc.

### `.handle_tool_call(tool_name, args) → str`

Dispatch a tool call. Returns JSON string.

### `.flush()`

Wait for any pending background extraction to complete.

### `.forget(query) → int`

Delete facts matching a search query. Returns the number of facts removed.

### `.forget_bank(bank) → int`

Delete an entire memory bank. Returns the number of facts removed.

### `.forget_all() → int`

Wipe all memory banks, commitments, and entities. Returns total items removed.

## Features

- **Zero-config observation** — just call `observe()` after each turn
- **Async extraction** — background threads, no latency impact
- **Deduplication** — similarity-based (0.75 threshold) prevents redundant facts
- **Durability tags** — `permanent` (name, preference) vs `temporal` (meeting, deadline)
- **Bank categorisation** — general, people, work, health, projects, preferences, decisions
- **Full conversation capture** — extracts both user and agent facts, decisions, agreements
- **Model fallback** — tries `llama-3.3-70b-versatile`, falls back to `llama-3.1-8b-instant`
- **Framework-agnostic tools** — OpenAI-compatible schemas for any agent framework
- **Semantic search** — optional embedding-based recall via `sentence-transformers`
- **TTL / expiry** — temporal facts auto-expire after 30 days
- **Forget** — selective memory deletion (by query, bank, or full wipe)

## Semantic Search

By default, `recall()` uses fast text matching. For higher-quality results you can enable embedding-based semantic search powered by `sentence-transformers` (uses the `all-MiniLM-L6-v2` model, ~80 MB one-time download).

```bash
pip install 'librarian-ai[embeddings]'
```

```python
from librarian import Librarian

lib = Librarian(search_mode="embedding")

# recall() now ranks results by cosine similarity
results = lib.recall("what programming language is the user learning?")
```

Embeddings are computed once and cached inside each bank's JSON file, so subsequent searches are instant. Set `search_mode="text"` (the default) to skip the dependency entirely.

## Memory Banks

Stored as simple JSON files:

```
~/.librarian/
├── banks/
│   ├── general.json
│   ├── people.json
│   ├── projects.json
│   └── preferences.json
├── commitments.json
└── entities.json
```

## Hermes Agent Plugin

If you use [Hermes Agent](https://github.com/hermes-agent/hermes-agent), the Librarian also works as a native memory plugin:

```yaml
# ~/.hermes/config.yaml
memory:
  provider: librarian
```

```bash
# ~/.hermes/.env
GROQ_API_KEY=gsk_...
```

The plugin automatically hooks into Hermes's memory lifecycle (`sync_turn`, `prefetch`, `system_prompt_block`, `on_pre_compress`).

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | [Groq API key](https://console.groq.com/keys) |
| `LIBRARIAN_MODEL` | `llama-3.3-70b-versatile` | Extraction model |

## Development

```bash
git clone https://github.com/Kenny1338/Librarian
cd Librarian
pip install -e ".[dev]"
```

### From PyPI (once published)

```bash
pip install librarian-ai
pytest
```

## License

MIT
