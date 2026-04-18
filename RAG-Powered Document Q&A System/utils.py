"""
utils.py
========
Helper functions for token estimation, text cleaning, and source formatting.
"""

import re
from typing import List, Dict


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 chars per token for English text.
    Good enough for display; use tiktoken for billing-accurate counts.
    """
    return max(1, len(text) // 4)


def clean_text(text: str) -> str:
    """
    Remove common PDF extraction artifacts:
    - Multiple blank lines → single blank line
    - Hyphenated line breaks → joined words
    - Non-breaking spaces → regular spaces
    - Control characters
    """
    # Join hyphenated line breaks
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Non-breaking spaces
    text = text.replace("\xa0", " ")
    # Collapse 3+ newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def format_sources(sources: List[Dict]) -> str:
    """Format sources list into a readable markdown string."""
    if not sources:
        return "_No sources retrieved._"
    lines = []
    for src in sources:
        lines.append(
            f"- **{src['source']}** — chunk #{src['chunk_id']} "
            f"(relevance: {src['score']:.0%})\n  _{src['preview']}_"
        )
    return "\n".join(lines)


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
