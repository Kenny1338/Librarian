#!/usr/bin/env python3
"""OpenAI SDK + Librarian memory — interactive chat with persistent recall.

Usage:
    export GROQ_API_KEY="your-groq-key"
    export OPENAI_API_KEY="your-openai-key"
    pip install openai
    python examples/openai_chat.py
"""

import json
import os
import sys


def main():
    missing = []
    if not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("  GROQ_API_KEY  — https://console.groq.com/keys")
        print("  OPENAI_API_KEY — https://platform.openai.com/api-keys")
        sys.exit(1)

    try:
        import openai
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        sys.exit(1)

    from librarian import Librarian

    client = openai.OpenAI()
    lib = Librarian(store_path="./example-memory")

    print("=== OpenAI + Librarian Chat ===")
    print("Type 'quit' to exit.\n")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with persistent memory.\n\n"
                + (lib.summary() or "No memories yet — they'll build up as we talk.")
            ),
        }
    ]

    # Register Librarian tools so the model can search memory
    tools = [{"type": "function", "function": t} for t in lib.tool_schemas()]

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools if tools else None,
        )

        assistant_msg = response.choices[0].message

        # Handle tool calls (memory lookups)
        if assistant_msg.tool_calls:
            messages.append(assistant_msg)
            for tc in assistant_msg.tool_calls:
                result = lib.handle_tool_call(
                    tc.function.name, json.loads(tc.function.arguments)
                )
                messages.append(
                    {"role": "tool", "content": result, "tool_call_id": tc.id}
                )
            # Get final response after tool use
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools if tools else None,
            )
            assistant_msg = response.choices[0].message

        reply = assistant_msg.content or ""
        messages.append({"role": "assistant", "content": reply})
        print(f"\n{reply}\n")

        # Observe the turn — extraction happens in the background
        lib.observe(user_input, reply)


if __name__ == "__main__":
    main()
