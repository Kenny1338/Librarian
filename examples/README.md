# Librarian Examples

Runnable examples showing how to use Librarian with different setups.

## Prerequisites

All examples require a **Groq API key** (free tier available):

```bash
export GROQ_API_KEY="your-groq-key"
```

Get one at https://console.groq.com/keys

Install Librarian from the repo root:

```bash
pip install -e .
```

## Examples

### standalone.py — No framework needed

Demonstrates the core API: `observe()`, `recall()`, `summary()`, `banks()`, and `commitments()`.

```bash
python examples/standalone.py
```

No extra dependencies required.

---

### openai_chat.py — OpenAI SDK + Librarian

Interactive chat using GPT-4o-mini with persistent Librarian memory and tool calling.

```bash
pip install openai
export OPENAI_API_KEY="your-openai-key"
python examples/openai_chat.py
```

---

### claude_chat.py — Anthropic Claude + Librarian

Interactive chat using Claude with persistent Librarian memory and tool calling.

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"
python examples/claude_chat.py
```

---

## Notes

- All examples store memory in `./example-memory/` by default.
- Memory persists between runs — restart an example and it remembers previous conversations.
- Delete `./example-memory/` to start fresh.
