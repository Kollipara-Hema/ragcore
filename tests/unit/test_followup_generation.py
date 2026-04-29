"""
Unit tests for GenerationService.generate_followups().

All tests mock self._llm.generate to avoid real API calls.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from generation.llm_service import GenerationService


def _make_service(raw_response: str) -> GenerationService:
    """Build a GenerationService whose LLM returns raw_response."""
    svc = GenerationService.__new__(GenerationService)
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=(raw_response, 50))
    svc._llm = mock_llm
    svc._redis = None
    return svc


QUESTION = "What is a Roth IRA?"
ANSWER = "A Roth IRA is a tax-advantaged retirement account funded with after-tax dollars."


@pytest.mark.asyncio
async def test_well_formed_response():
    questions = [
        "How does a Roth IRA differ from a Traditional IRA?",
        "What are the income limits for contributing to a Roth IRA?",
        "Can I withdraw from a Roth IRA before retirement without penalty?",
    ]
    svc = _make_service(json.dumps(questions))
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == questions


@pytest.mark.asyncio
async def test_markdown_wrapped():
    questions = [
        "What happens to a Roth IRA when I reach age 72?",
        "Can I have both a Roth IRA and a 401k at the same time?",
        "What is the annual contribution limit for a Roth IRA in 2025?",
    ]
    raw = "```json\n" + json.dumps(questions) + "\n```"
    svc = _make_service(raw)
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == questions


@pytest.mark.asyncio
async def test_not_a_list():
    raw = json.dumps({"question1": "How does a Roth IRA work?"})
    svc = _make_service(raw)
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == []


@pytest.mark.asyncio
async def test_wrong_count():
    questions = [
        "What is a Roth IRA contribution limit?",
        "Can I open a Roth IRA for my child?",
    ]
    svc = _make_service(json.dumps(questions))
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == []


@pytest.mark.asyncio
async def test_non_string_elements():
    raw = json.dumps([{"q": "What is a Roth IRA?"}, {"q": "How does it work?"}, 42])
    svc = _make_service(raw)
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == []


@pytest.mark.asyncio
async def test_empty_string_in_list():
    raw = json.dumps([
        "How does a Roth IRA compare to a Traditional IRA?",
        "",
        "What is the penalty for early Roth IRA withdrawal?",
    ])
    svc = _make_service(raw)
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == []


@pytest.mark.asyncio
async def test_extra_text_after_array():
    """LLM appends prose after the JSON array — the real failure seen on Render."""
    questions = [
        "How does a Roth IRA differ from a Traditional IRA?",
        "What are the income limits for contributing to a Roth IRA?",
        "Can I withdraw Roth IRA contributions early without penalty?",
    ]
    raw = json.dumps(questions) + "\nNote: these are common follow-up questions."
    svc = _make_service(raw)
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == questions


@pytest.mark.asyncio
async def test_extra_text_before_array():
    """LLM prefixes the JSON array with prose."""
    questions = [
        "What is the Roth IRA contribution limit for 2025?",
        "How does a Roth IRA conversion work for high earners?",
        "When can I start withdrawing from a Roth IRA tax-free?",
    ]
    raw = "Here are three follow-up questions:\n" + json.dumps(questions)
    svc = _make_service(raw)
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == questions


@pytest.mark.asyncio
async def test_three_separate_single_element_arrays():
    """Production Llama shape: three separate single-element arrays on consecutive lines."""
    q1 = "How does a Roth IRA differ from a Traditional IRA?"
    q2 = "What are the income limits for contributing to a Roth IRA?"
    q3 = "Can I convert a 401k to a Roth IRA without penalty?"
    raw = f'{json.dumps([q1])}\n{json.dumps([q2])}\n{json.dumps([q3])}'
    svc = _make_service(raw)
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == [q1, q2, q3]


@pytest.mark.asyncio
async def test_five_element_array_returns_first_three():
    """LLM emits 5 questions — parser takes the first 3."""
    questions = [
        "How does a Roth IRA differ from a Traditional IRA?",
        "What are the income limits for a Roth IRA in 2025?",
        "Can I have both a Roth IRA and a 401k simultaneously?",
        "What happens to a Roth IRA after I turn 72?",
        "Is a backdoor Roth IRA still legal in 2025?",
    ]
    svc = _make_service(json.dumps(questions))
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == questions[:3]


@pytest.mark.asyncio
async def test_mixed_strings_and_arrays_on_lines():
    """Lines contain a mix of bare JSON strings and single-element arrays."""
    q1 = "How does a Roth IRA differ from a Traditional IRA?"
    q2 = "What is the annual Roth IRA contribution limit?"
    q3 = "When can I start withdrawing from a Roth IRA tax-free?"
    raw = f'{json.dumps(q1)}\n{json.dumps([q2])}\n{json.dumps(q3)}'
    svc = _make_service(raw)
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == [q1, q2, q3]


@pytest.mark.asyncio
async def test_mixed_lines_insufficient_returns_empty():
    """Only 2 parseable strings across lines — must return []."""
    q1 = "How does a Roth IRA differ from a Traditional IRA?"
    q2 = "What is the annual Roth IRA contribution limit?"
    raw = f'{json.dumps([q1])}\n{json.dumps([q2])}'
    svc = _make_service(raw)
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == []


@pytest.mark.asyncio
async def test_markdown_wrapped_multi_array():
    """Markdown fences around three separate single-element arrays."""
    q1 = "How does a Roth IRA differ from a Traditional IRA?"
    q2 = "What are the income limits for a Roth IRA?"
    q3 = "Can I convert a 401k to a Roth IRA?"
    inner = f'{json.dumps([q1])}\n{json.dumps([q2])}\n{json.dumps([q3])}'
    raw = f"```json\n{inner}\n```"
    svc = _make_service(raw)
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == [q1, q2, q3]


@pytest.mark.asyncio
async def test_llm_exception():
    svc = GenerationService.__new__(GenerationService)
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("API timeout"))
    svc._llm = mock_llm
    svc._redis = None
    result = await svc.generate_followups(QUESTION, ANSWER)
    assert result == []
