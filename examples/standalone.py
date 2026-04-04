#!/usr/bin/env python3
"""Standalone Librarian demo — no framework needed.

Demonstrates observe(), recall(), summary(), banks(), and commitments().

Usage:
    export GROQ_API_KEY="your-key-here"
    python examples/standalone.py
"""

import os
import sys

def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is required.")
        print("Get a free key at https://console.groq.com/keys")
        sys.exit(1)

    from librarian import Librarian

    # Create a Librarian with a local store path
    lib = Librarian(store_path="./example-memory")

    print("=== Librarian Standalone Demo ===\n")

    # Observe some conversation turns
    print("1. Observing conversation turns...")
    lib.observe(
        "My name is Alice and I'm working on a Python web scraper.",
        "Nice to meet you, Alice! What sites are you scraping?",
        blocking=True,
    )
    lib.observe(
        "I prefer using httpx over requests. My meeting with Bob is Friday at 3pm.",
        "Good choice! I'll note that meeting for you.",
        blocking=True,
    )

    # Check what banks were created
    print("\n2. Memory banks:")
    banks = lib.banks()
    for bank, count in banks.items():
        print(f"   {bank}: {count} facts")

    # Search for specific memories
    print("\n3. Recall 'Alice':")
    results = lib.recall("Alice")
    for r in results:
        print(f"   [{r.get('bank', '?')}] {r.get('text', '')}")

    # Get full summary (ready for system prompt injection)
    print("\n4. Summary for system prompt:")
    summary = lib.summary()
    if summary:
        for line in summary.split("\n"):
            print(f"   {line}")
    else:
        print("   (no summary yet)")

    # Check commitments
    print("\n5. Commitments:")
    commitments = lib.commitments()
    if commitments:
        for c in commitments:
            due = f" (due: {c['due']})" if c.get("due") else ""
            print(f"   [{c.get('type', 'task')}] {c.get('subject', '')}{due}")
    else:
        print("   (none tracked)")

    print("\nDone! Memory stored in ./example-memory/")


if __name__ == "__main__":
    main()
