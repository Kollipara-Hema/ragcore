"""
/ingest/file hardening: magic-byte sniff drives BOTH validation and loader
dispatch; PDF page-cap fires BEFORE embed.

The two load-bearing properties verified here:
  1. The loader that RUNS matches the sniffed bytes, NOT the client's
     filename extension or Content-Type. Verified by spying on each
     loader's load() and asserting which was reached.
  2. PDF page-cap rejects oversized PDFs BEFORE the embed runs. Verified
     by spying on the embedder and asserting it's never called.

Test harness mirrors test_session_ingest.py: TestClient(app) inside `with`
so lifespan fires; module-level session_store is replaced with a tmp_path
SessionStore so files don't bleed across tests.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from config.settings import settings
from vectorstore.session_store import SessionStore


@pytest.fixture
def client_and_store(tmp_path, monkeypatch):
    """Same shape as test_session_ingest.py's fixture: real lifespan, then
    a tmp-rooted SessionStore patched onto api.main so handlers see it."""
    with TestClient(app) as c:
        store = SessionStore(root=tmp_path)
        monkeypatch.setattr(api_main, "session_store", store)
        yield c, store


def _minimal_pdf_bytes(n_pages: int = 1) -> bytes:
    """Build a minimal valid PDF of n_pages via pymupdf. Used to drive the
    PDF-specific code path with bytes that pymupdf can actually parse."""
    import pymupdf
    doc = pymupdf.open()
    for _ in range(n_pages):
        doc.new_page()
    buf = doc.tobytes()
    doc.close()
    return buf


def _spy_registry_loaders(monkeypatch):
    """Replace each loader instance's `load` method on the live registry
    with a MagicMock wrapping the bound method. Returns a dict
    {loader_class_name: spy_mock} keyed by class name so tests can assert
    which loader ran. Bound-method wrapping is required because
    `MagicMock(wraps=Cls.method)` would drop self; wrapping the bound
    instance method keeps it intact."""
    from ingestion.loaders import document_loaders as dl
    spies: dict[str, MagicMock] = {}
    for inst in dl.loader_registry._loaders:
        cls_name = type(inst).__name__
        spy = MagicMock(wraps=inst.load)
        monkeypatch.setattr(inst, "load", spy)
        spies[cls_name] = spy
    return spies


# ──────────────────────────────────────────────────────────────────────────────
# A. Sniff drives dispatch — the loader that runs matches the bytes
# ──────────────────────────────────────────────────────────────────────────────

def test_A1_spoofed_mime_real_txt_routes_to_text_loader(client_and_store, monkeypatch):
    """Upload claims application/pdf with filename 'report.pdf' but the bytes
    are plain TXT. Sniffed as 'text' → TextLoader runs, PDFLoader does NOT."""
    client, _ = client_and_store
    txt_bytes = b"This is just plain text. No magic bytes anywhere here.\n"

    spies = _spy_registry_loaders(monkeypatch)
    r = client.post(
        "/ingest/file",
        files={"file": ("report.pdf", io.BytesIO(txt_bytes), "application/pdf")},
    )

    assert r.status_code == 200, r.text
    assert spies["TextLoader"].call_count == 1, "TextLoader must run — bytes are text"
    assert spies["PDFLoader"].call_count == 0, "PDFLoader must NOT run — bytes are not PDF"


def test_A2_spoofed_extension_real_pdf_routes_to_pdf_loader(client_and_store, monkeypatch):
    """Filename '.csv', Content-Type 'text/csv', body is real PDF bytes.
    Sniff says PDF → PDFLoader runs."""
    client, _ = client_and_store
    pdf_bytes = _minimal_pdf_bytes(n_pages=1)

    spies = _spy_registry_loaders(monkeypatch)
    r = client.post(
        "/ingest/file",
        files={"file": ("report.csv", io.BytesIO(pdf_bytes), "text/csv")},
    )

    assert r.status_code == 200, r.text
    assert spies["PDFLoader"].call_count == 1
    assert spies["CSVLoader"].call_count == 0, "CSVLoader must NOT run on PDF bytes"


def test_A3_html_bytes_in_pdf_named_file_routes_to_text_loader(client_and_store, monkeypatch):
    """HTML bytes inside a '.pdf'-named upload. filetype.guess returns None
    on HTML (text-shaped, no magic). UTF-8 decode succeeds → sniff = 'text'
    → TextLoader runs. HTMLLoader (BeautifulSoup) is NEVER reached on the
    user-input path; that's the security property under test."""
    client, _ = client_and_store
    html_bytes = b"<!doctype html><html><body><p>hi</p></body></html>"

    spies = _spy_registry_loaders(monkeypatch)
    r = client.post(
        "/ingest/file",
        files={"file": ("trick.pdf", io.BytesIO(html_bytes), "application/pdf")},
    )

    assert r.status_code == 200, r.text
    assert spies["HTMLLoader"].call_count == 0, (
        "HTMLLoader MUST NOT run on the user-input path — that's the parser-"
        "pathology surface we eliminated"
    )
    assert spies["TextLoader"].call_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# B. Allowlist reduction — DOCX and unknown binary rejected; HTML/CSV/text
#    bytes route to safe TextLoader
# ──────────────────────────────────────────────────────────────────────────────

def test_B4_csv_bytes_route_to_text_loader_not_csv_loader(client_and_store, monkeypatch):
    """CSV bytes pass UTF-8 decode → sniff='text' → TextLoader. The pandas
    csv.DictReader-based CSVLoader is bypassed structurally."""
    client, _ = client_and_store
    csv_bytes = b"col1,col2,col3\n1,2,3\n4,5,6\n"

    spies = _spy_registry_loaders(monkeypatch)
    r = client.post(
        "/ingest/file",
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert r.status_code == 200, r.text
    assert spies["CSVLoader"].call_count == 0
    assert spies["TextLoader"].call_count == 1


def test_B6_docx_upload_rejected_415(client_and_store):
    """DOCX (zip-based binary) is identified by filetype as 'docx' → reject.
    The python-docx loader stays registered for seed-corpus CLI use but is
    unreachable via the HTTP path."""
    client, _ = client_and_store
    # Minimal-ish DOCX: a zip container with the OOXML structure. filetype
    # detects this by magic bytes (PK zip header + content_types.xml).
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/></Types>',
        )
        zf.writestr("word/document.xml", "<document/>")
    docx_bytes = buf.getvalue()

    r = client.post(
        "/ingest/file",
        files={"file": (
            "report.docx",
            io.BytesIO(docx_bytes),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )},
    )
    assert r.status_code == 415, r.text
    assert "Allowed: PDF, TXT, MD" in r.json()["detail"]


def test_B7_real_pdf_upload_succeeds(client_and_store):
    """No regression: a real, small, well-formed PDF still ingests."""
    client, _ = client_and_store
    pdf_bytes = _minimal_pdf_bytes(n_pages=2)

    r = client.post(
        "/ingest/file",
        files={"file": ("ok.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert r.status_code == 200, r.text


def test_B8_real_txt_upload_succeeds(client_and_store):
    client, _ = client_and_store
    r = client.post(
        "/ingest/file",
        files={"file": ("notes.txt", io.BytesIO(b"hello world " * 20), "text/plain")},
    )
    assert r.status_code == 200, r.text


def test_B9_real_md_upload_succeeds(client_and_store):
    """Markdown content with an inline HTML block — would have been false-
    rejected by an angle-bracket heuristic. The simplified sniffer (no
    sub-classification) accepts it as text."""
    client, _ = client_and_store
    md_bytes = (
        b"# Title\n\nSome prose.\n\n"
        b"<div class='callout'>This is legal Markdown with inline HTML.</div>\n\n"
        b"More prose afterward. " * 5
    )
    r = client.post(
        "/ingest/file",
        files={"file": ("doc.md", io.BytesIO(md_bytes), "text/markdown")},
    )
    assert r.status_code == 200, r.text


# ──────────────────────────────────────────────────────────────────────────────
# C. PDF page cap — fires BEFORE embed
# ──────────────────────────────────────────────────────────────────────────────

def test_C10_pdf_over_page_cap_rejected_embedder_not_called(client_and_store, monkeypatch):
    """A PDF with page_count > cap returns 413, and the embedder is NEVER
    called. The cap probe sits between the temp write and the
    pipeline.ingest_file() call which would chunk+embed."""
    client, _ = client_and_store
    monkeypatch.setattr(settings, "ragcore_pdf_max_pages", 5)
    pdf_bytes = _minimal_pdf_bytes(n_pages=6)

    embedder_spy = AsyncMock(side_effect=AssertionError(
        "embed_chunks reached despite over-cap PDF"
    ))
    ingest_spy = AsyncMock(side_effect=AssertionError(
        "pipeline.ingest_file reached despite over-cap PDF"
    ))
    with patch("ingestion.pipeline.IngestionPipeline.ingest_file", ingest_spy), \
         patch("embeddings.embedder.BGEEmbedder.embed_chunks", embedder_spy):
        r = client.post(
            "/ingest/file",
            files={"file": ("big.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )

    assert r.status_code == 413, r.text
    assert "page limit" in r.json()["detail"]
    embedder_spy.assert_not_called()
    ingest_spy.assert_not_called()


def test_C11_pdf_at_page_cap_boundary_accepted(client_and_store, monkeypatch):
    """Inclusive limit — a PDF with exactly N pages and cap=N is accepted."""
    client, _ = client_and_store
    monkeypatch.setattr(settings, "ragcore_pdf_max_pages", 3)
    pdf_bytes = _minimal_pdf_bytes(n_pages=3)

    r = client.post(
        "/ingest/file",
        files={"file": ("ok.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert r.status_code == 200, r.text


def test_C12_pdf_one_over_cap_rejected(client_and_store, monkeypatch):
    client, _ = client_and_store
    monkeypatch.setattr(settings, "ragcore_pdf_max_pages", 3)
    pdf_bytes = _minimal_pdf_bytes(n_pages=4)

    r = client.post(
        "/ingest/file",
        files={"file": ("over.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert r.status_code == 413, r.text


def test_C13_malformed_pdf_with_valid_magic_returns_400_not_500(client_and_store):
    """A file with the %PDF-1.x magic header but truncated/garbage body
    sniffs as 'pdf', passes the allowlist, and reaches the page-cap probe
    where pymupdf.open raises. That raise becomes a 400 — NOT a 500, NOT
    a hang. The embedder must never be called."""
    client, _ = client_and_store
    # Valid magic, broken body. filetype.guess() will still classify as PDF.
    malformed = b"%PDF-1.4\n" + b"\x00\x01\x02" * 200 + b"garbage no trailer"

    embedder_spy = AsyncMock(side_effect=AssertionError(
        "embed_chunks reached on malformed PDF"
    ))
    with patch("embeddings.embedder.BGEEmbedder.embed_chunks", embedder_spy):
        r = client.post(
            "/ingest/file",
            files={"file": ("broken.pdf", io.BytesIO(malformed), "application/pdf")},
        )

    assert r.status_code == 400, r.text
    assert "PDF could not be parsed" in r.json()["detail"]
    embedder_spy.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# D. Random binary noise rejected
# ──────────────────────────────────────────────────────────────────────────────

def test_D_random_binary_noise_rejected_415(client_and_store):
    """Non-UTF-8 random bytes with no recognized magic → 'reject' → 415."""
    client, _ = client_and_store
    # High-bit bytes that won't be valid UTF-8 and don't carry any magic.
    noise = bytes(range(128, 256)) * 16  # 2 KiB of >=0x80 bytes

    r = client.post(
        "/ingest/file",
        files={"file": ("blob.bin", io.BytesIO(noise), "application/octet-stream")},
    )
    assert r.status_code == 415, r.text


# ──────────────────────────────────────────────────────────────────────────────
# E. No regression: missing/odd Content-Type doesn't punish honest clients
# ──────────────────────────────────────────────────────────────────────────────

def test_E14_pdf_with_octet_stream_content_type_accepted(client_and_store):
    """Some browsers send application/octet-stream for unfamiliar extensions.
    The old MIME allowlist would 400; the new sniff-driven path accepts on
    bytes alone."""
    client, _ = client_and_store
    pdf_bytes = _minimal_pdf_bytes(n_pages=1)

    r = client.post(
        "/ingest/file",
        files={"file": ("x.pdf", io.BytesIO(pdf_bytes), "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
