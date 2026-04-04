#!/usr/bin/env python3
"""Anthropic Claude + Librarian memory — interactive chat with persistent recall.

Usage:
    export GROQ_API_KEY="your-groq-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"
    pip install anthropic
    python examples/claude_chat.py
"""

import json
import os
import sys


def main():
    missing = []
    if not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("  GROQ_API_KEY     — https://console.groq.com/keys")
        print("  ANTHROPIC_API_KEY — https://console.anthropic.com/settings/keys")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    from librarian import Librarian

    client = anthropic.Anthropic()
    lib = Librarian(store_path="./example-memory")

    print("=== Claude + Librarian Chat ===")
    print("Type 'quit' to exit.\n")

    system_prompt = (
        "You are a helpful assistant with persistent memory.\n\n"
        + (lib.summary() or "No memories yet — they'll build up as we talk.")
    )

    # Convert Librarian tool schemas to Anthropic format
    tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in lib.tool_schemas()
    ]

    messages = []

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

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
            tools=tools,
        )

        # Process response blocks — handle tool use if present
        reply_text = ""
        while response.stop_reason == "tool_use":
            # Collect assistant content blocks
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Process each tool use block
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    result = lib.handle_tool_call(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

            # Continue the conversation
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

        # Extract final text response
        for block in response.content:
            if hasattr(block, "text"):
                reply_text += block.text

        messages.append({"role": "assistant", "content": response.content})
        print(f"\n{reply_text}\n")

        # Observe the turn — extraction happens in the background
        lib.observe(user_input, reply_text)


if __name__ == "__main__":
    main()
