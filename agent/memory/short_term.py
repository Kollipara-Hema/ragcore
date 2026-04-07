"""
Short-term memory — in-process conversation buffer for a single session.

Stores the last N turns (query + answer pairs) so the agent can refer
back to earlier context within a conversation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    query: str
    answer: str
    query_type: str = "semantic"
    citations: list[dict] = field(default_factory=list)


class ShortTermMemory:
    """
    Thread-local, in-memory conversation buffer.

    Usage:
        mem = ShortTermMemory(max_turns=5)
        mem.add(query="What is X?", answer="X is ...")
        context = mem.get_context()   # formatted string for prompts
    """

    def __init__(self, max_turns: int = 5):
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(
        self,
        query: str,
        answer: str,
        query_type: str = "semantic",
        citations: Optional[list[dict]] = None,
    ) -> None:
        self._turns.append(
            Turn(query=query, answer=answer, query_type=query_type, citations=citations or [])
        )

    def get_turns(self) -> list[Turn]:
        return list(self._turns)

    def get_context(self) -> str:
        """Return a formatted string of previous turns for prompt injection."""
        if not self._turns:
            return ""
        lines = []
        for i, turn in enumerate(self._turns, 1):
            lines.append(f"[Turn {i}] Q: {turn.query}\nA: {turn.answer}")
        return "\n\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
