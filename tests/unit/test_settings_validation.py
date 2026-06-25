"""Validation tests for config.settings field constraints."""
import pytest
from pydantic import ValidationError

from config.settings import Settings


def test_generation_strategy_rejects_unwired_agentic(monkeypatch):
    """agentic is documented but not wired — it must be rejected at construction,
    not silently fall through to basic generation."""
    monkeypatch.setenv("GENERATION_STRATEGY", "agentic")
    with pytest.raises(ValidationError, match="generation_strategy must be one of"):
        Settings()


def test_generation_strategy_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("GENERATION_STRATEGY", "not_a_strategy")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("strategy", ["basic", "self_rag", "flare"])
def test_generation_strategy_accepts_supported_values(monkeypatch, strategy):
    monkeypatch.setenv("GENERATION_STRATEGY", strategy)
    assert Settings().generation_strategy == strategy
