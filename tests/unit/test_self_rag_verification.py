"""
Unit tests for SelfRAGGenerator claim verification and extraction parsers.

Covers the JSON parsing layer in _verify_claim and _extract_claims,
mocking the OpenAI client so no real API calls are made.
"""
from __future__ import annotations
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch


def _make_openai_response(content: str):
    """Build a minimal fake openai ChatCompletion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _run_verify(content: str) -> tuple:
    """Call _verify_claim with a mocked LLM returning `content`."""
    from generation.advanced_generation import SelfRAGGenerator

    gen = SelfRAGGenerator()
    fake_response = _make_openai_response(content)

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=fake_response)
        return asyncio.run(gen._verify_claim("some claim", "some context"))


def _run_extract(content: str) -> list:
    """Call _extract_claims with a mocked LLM returning `content`."""
    from generation.advanced_generation import SelfRAGGenerator

    gen = SelfRAGGenerator()
    fake_response = _make_openai_response(content)

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=fake_response)
        return asyncio.run(gen._extract_claims("some answer text"))


class TestVerifyClaimParser:

    def test_standard_dict_true_with_evidence(self):
        """Well-formed {"supported": true, "evidence": "quote"} → (True, "quote")."""
        result = _run_verify(json.dumps({"supported": True, "evidence": "direct quote"}))
        assert result == (True, "direct quote")

    def test_standard_dict_true_no_evidence_field(self):
        """{"supported": true} with no evidence key → (True, "") without raising."""
        result = _run_verify(json.dumps({"supported": True}))
        assert result == (True, "")

    def test_dict_with_string_bool_false(self):
        """{"supported": "false"} → (False, "") — string "false" must not be truthy."""
        result = _run_verify(json.dumps({"supported": "false", "evidence": ""}))
        assert result == (False, "")

    def test_bare_string_supported(self):
        """LLM returns bare JSON string "supported" instead of object → (True, "")."""
        result = _run_verify('"supported"')
        assert result == (True, "")

    def test_bare_string_gibberish_fails_closed(self):
        """Unknown bare string → (False, "") — verification fails closed."""
        result = _run_verify('"xyz_gibberish_unknown"')
        assert result == (False, "")


class TestExtractClaimsParser:

    def test_non_claims_key_statements(self):
        """{"statements": ["claim a long enough", "claim b long enough"]} uses the statements key."""
        content = json.dumps({"statements": [
            "The transformer uses self-attention mechanism.",
            "BERT was pretrained on masked language modeling.",
        ]})
        claims = _run_extract(content)
        assert len(claims) == 2
        assert any("self-attention" in c for c in claims)
