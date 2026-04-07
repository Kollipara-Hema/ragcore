"""Tests for agent/memory — short_term and long_term."""
import asyncio
from agent.memory.short_term import ShortTermMemory
from agent.memory.long_term import LongTermMemory


class TestShortTermMemory:
    def test_empty_on_init(self):
        mem = ShortTermMemory()
        assert len(mem) == 0

    def test_add_and_retrieve_turn(self):
        mem = ShortTermMemory()
        mem.add(query="What is X?", answer="X is Y.")
        assert len(mem) == 1
        turns = mem.get_turns()
        assert turns[0].query == "What is X?"
        assert turns[0].answer == "X is Y."

    def test_respects_max_turns(self):
        mem = ShortTermMemory(max_turns=2)
        mem.add("q1", "a1")
        mem.add("q2", "a2")
        mem.add("q3", "a3")
        assert len(mem) == 2
        assert mem.get_turns()[0].query == "q2"

    def test_get_context_empty_string_when_no_turns(self):
        mem = ShortTermMemory()
        assert mem.get_context() == ""

    def test_get_context_includes_turns(self):
        mem = ShortTermMemory()
        mem.add("What is X?", "X is Y.")
        ctx = mem.get_context()
        assert "What is X?" in ctx
        assert "X is Y." in ctx

    def test_clear(self):
        mem = ShortTermMemory()
        mem.add("q", "a")
        mem.clear()
        assert len(mem) == 0


class TestLongTermMemory:
    """Test long-term memory using the in-process fallback (no Redis needed)."""

    def test_set_and_get(self):
        mem = LongTermMemory(prefix="test")
        asyncio.run(mem.set("key1", "value1"))
        result = asyncio.run(mem.get("key1"))
        assert result == "value1"

    def test_get_missing_returns_default(self):
        mem = LongTermMemory(prefix="test")
        result = asyncio.run(mem.get("nonexistent", default="fallback"))
        assert result == "fallback"

    def test_delete(self):
        mem = LongTermMemory(prefix="test")
        asyncio.run(mem.set("to_delete", 42))
        asyncio.run(mem.delete("to_delete"))
        result = asyncio.run(mem.get("to_delete"))
        assert result is None

    def test_exists(self):
        mem = LongTermMemory(prefix="test")
        asyncio.run(mem.set("exists_key", True))
        assert asyncio.run(mem.exists("exists_key")) is True
        assert asyncio.run(mem.exists("missing_key")) is False

    def test_stores_complex_value(self):
        mem = LongTermMemory(prefix="test")
        value = {"nested": [1, 2, 3], "flag": True}
        asyncio.run(mem.set("complex", value))
        result = asyncio.run(mem.get("complex"))
        assert result == value
