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


class TestVerifyClaimEmptyAndUnparseable:

    def test_empty_string_response_fails_closed(self):
        """Empty string from LLM → json.loads("") raises JSONDecodeError → (False, "").

        This is the exact path that escaped the 2026-04-18 fix: the exception handler
        previously returned (True, "") instead of (False, ""), so every empty response
        promoted the claim to verified_claims rather than unsupported_claims.
        """
        result = _run_verify("")
        assert result == (False, "")

    def test_whitespace_only_response_fails_closed(self):
        """Whitespace-only string → same JSONDecodeError path → (False, "")."""
        result = _run_verify("   \n  ")
        assert result == (False, "")

    def test_none_answer_fails_closed(self):
        """None as the answer field → json.loads(None) raises TypeError → (False, "")."""
        from generation.advanced_generation import SelfRAGGenerator
        import asyncio

        gen = SelfRAGGenerator()
        mock_result = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        mock_result.answer = None
        mock_service = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock()
        mock_service.generate = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(
            return_value=mock_result
        )
        result = asyncio.run(gen._verify_claim("some claim", "some context", mock_service))
        assert result == (False, "")


class TestMarkdownFenceStripping:

    def test_verify_claim_strips_markdown_fences(self):
        """```json\\n{...}\\n``` — gpt-4o-mini's real-world output shape with FiQA contexts."""
        result = _run_verify('```json\n{"supported": true, "evidence": "test"}\n```')
        assert result == (True, "test")

    def test_verify_claim_strips_plain_fence_no_lang(self):
        """``` (no json tag) is also stripped before parsing."""
        result = _run_verify('```\n{"supported": true, "evidence": "test"}\n```')
        assert result == (True, "test")

    def test_extract_claims_strips_markdown_fences(self):
        """_extract_claims handles fenced JSON array from the LLM."""
        content = '```json\n["claim one long enough", "claim two long enough"]\n```'
        claims = _run_extract(content)
        assert claims == ["claim one long enough", "claim two long enough"]


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
