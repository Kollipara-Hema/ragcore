"""Unit tests for _extract_attributed_spans in orchestrator.py."""
import pytest
from orchestrator import _extract_attributed_spans


# ── 1. Single trailing marker ────────────────────────────────────────────────

def test_single_trailing_marker():
    raw = 'Claim sentence <cite source="1">.'
    clean, spans = _extract_attributed_spans(raw)

    assert clean == "Claim sentence ."
    assert len(spans) == 1
    s = spans[0]
    assert s["source"] == 1
    assert s["text"] == "Claim sentence"
    assert s["start"] == 0
    assert s["end"] == len("Claim sentence")
    assert clean[s["start"] : s["end"]] == s["text"]


# ── 2. Two trailing markers across two sentences ─────────────────────────────

def test_two_trailing_markers_two_sentences():
    raw = 'First claim <cite source="1">. Second claim <cite source="2">.'
    clean, spans = _extract_attributed_spans(raw)

    assert clean == "First claim . Second claim ."
    assert len(spans) == 2

    assert spans[0]["source"] == 1
    assert spans[0]["text"] == "First claim"
    assert clean[spans[0]["start"] : spans[0]["end"]] == spans[0]["text"]

    assert spans[1]["source"] == 2
    assert spans[1]["text"] == "Second claim"
    assert clean[spans[1]["start"] : spans[1]["end"]] == spans[1]["text"]


# ── 3. No markers ─────────────────────────────────────────────────────────────

def test_no_markers():
    raw = "Plain answer with no citations."
    clean, spans = _extract_attributed_spans(raw)

    assert clean == raw
    assert spans == []


# ── 4. Marker at start of text — empty preceding clause, skipped ─────────────

def test_marker_at_start():
    """
    Marker at position 0 leaves no preceding clause; span is skipped.
    Represents malformed LLM output where the marker precedes any text.
    """
    raw = '<cite source="1"> Claim follows.'
    clean, spans = _extract_attributed_spans(raw)

    assert clean == " Claim follows."
    assert spans == []


# ── 5. Invalid (non-integer) source number — graceful skip ───────────────────

def test_invalid_source_number():
    raw = '<cite source="abc">.'
    clean, spans = _extract_attributed_spans(raw)

    assert clean == "."
    assert spans == []


# ── 6. Consecutive markers — second has empty clause, skipped ────────────────

def test_consecutive_markers():
    """
    'First <cite source="1"><cite source="2">.'
    First span: source 1, text "First ".
    Second span: zero text between markers → empty clause → skipped.
    """
    raw = 'First <cite source="1"><cite source="2">.'
    clean, spans = _extract_attributed_spans(raw)

    assert clean == "First ."
    assert len(spans) == 1
    assert spans[0]["source"] == 1
    assert spans[0]["text"] == "First"
    assert clean[spans[0]["start"] : spans[0]["end"]] == spans[0]["text"]


# ── 7. Realistic multi-sentence answer ───────────────────────────────────────

def test_realistic_multi_sentence():
    raw = (
        "Investment risk is the chance that an investment will lose value "
        '<cite source="1">. '
        "It involves balancing the desire for return against potential losses "
        '<cite source="1">. '
        "Diversification across asset classes can reduce overall portfolio risk "
        '<cite source="2">. '
        "Bond prices move inversely to interest rates "
        '<cite source="3">. '
        "Rebalancing annually helps maintain target allocations "
        '<cite source="2">.'
    )
    clean, spans = _extract_attributed_spans(raw)

    # No markers remain in cleaned answer
    assert "<cite" not in clean

    # 5 markers, each with a non-empty preceding clause → 5 spans
    assert len(spans) == 5

    # All offsets resolve correctly in cleaned answer
    for s in spans:
        assert clean[s["start"] : s["end"]] == s["text"], (
            f"Offset mismatch: span={s}, clean[{s['start']}:{s['end']}]="
            f"{clean[s['start']:s['end']]!r}"
        )

    # Sources are attributed correctly
    assert spans[0]["source"] == 1
    assert spans[1]["source"] == 1
    assert spans[2]["source"] == 2
    assert spans[3]["source"] == 3
    assert spans[4]["source"] == 2


# ── 8. [Source N] bracket stripped from span text and cleaned answer ──────────

def test_strip_source_bracket():
    """
    Model emits [Source N] inline alongside the <cite> marker.
    Both should be stripped: [Source N] from cleaned_answer and span.text.
    """
    raw = 'Claim text [Source 1] <cite source="1">.'
    clean, spans = _extract_attributed_spans(raw)

    assert clean == "Claim text ."
    assert len(spans) == 1
    s = spans[0]
    assert s["source"] == 1
    assert s["text"] == "Claim text"
    assert clean[s["start"] : s["end"]] == s["text"]


# ── 9. Multiple [Source N] brackets in one clause — all stripped ──────────────

def test_multiple_brackets():
    """
    Multiple [Source N] tokens in one clause before a single <cite> marker.
    All bracket notation stripped from both cleaned_answer and span.text.
    """
    raw = 'First [Source 1] and second [Source 2] <cite source="1">.'
    clean, spans = _extract_attributed_spans(raw)

    assert clean == "First and second ."
    assert len(spans) == 1
    s = spans[0]
    assert s["source"] == 1
    assert s["text"] == "First and second"
    assert clean[s["start"] : s["end"]] == s["text"]
