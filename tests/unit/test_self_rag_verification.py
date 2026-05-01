"""
Unit tests for SelfRAGGenerator claim verification and extraction parsers.

Covers the JSON parsing layer in _verify_claim and _extract_claims,
mocking the GenerationService interface so no real API calls are made.
"""
from __future__ import annotations
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock


def _make_mock_service(content: str):
    """Build a mock GenerationService returning `content` as .answer."""
    mock_result = MagicMock()
    mock_result.answer = content
    mock_service = AsyncMock()
    mock_service.generate = AsyncMock(return_value=mock_result)
    return mock_service


def _run_verify(content: str) -> tuple:
    """Call _verify_claim with a mocked LLM service returning `content`."""
    from generation.advanced_generation import SelfRAGGenerator

    gen = SelfRAGGenerator()
    mock_service = _make_mock_service(content)
    return asyncio.run(gen._verify_claim("some claim", "some context", mock_service))


def _run_extract(content: str) -> list:
    """Call _extract_claims with a mocked LLM service returning `content`."""
    from generation.advanced_generation import SelfRAGGenerator

    gen = SelfRAGGenerator()
    mock_service = _make_mock_service(content)
    return asyncio.run(gen._extract_claims("some answer text", mock_service))


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


class TestSelfRAGVerificationCrossProvider:

    def test_self_rag_verification_with_anthropic_provider(self):
        """Anthropic path: raw string response (no native JSON mode), parsed defensively."""
        from generation.advanced_generation import SelfRAGGenerator

        gen = SelfRAGGenerator()
        mock_service = _make_mock_service(
            json.dumps({"supported": True, "evidence": "direct context quote"})
        )
        result = asyncio.run(gen._verify_claim("some claim", "some context", mock_service))
        assert result == (True, "direct context quote")
        mock_service.generate.assert_awaited_once()

    def test_self_rag_verification_with_groq_provider(self):
        """Groq path: same interface, confirm no regression on string-bool edge case."""
        from generation.advanced_generation import SelfRAGGenerator

        gen = SelfRAGGenerator()
        mock_service = _make_mock_service(
            json.dumps({"supported": "false", "evidence": ""})
        )
        result = asyncio.run(gen._verify_claim("unverifiable claim", "unrelated context", mock_service))
        assert result == (False, "")
        mock_service.generate.assert_awaited_once()
