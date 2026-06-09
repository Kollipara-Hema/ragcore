"""Unit tests for _seed_faiss_default — the FAISS self-heal seed step.

Scope: the fail-hard contract. A unit test catches future refactors that
silently soften the seed into the Apple-style soft-fallback. The four other
verification cases (fresh / repeat / unwritable-dest / unwritable-parent) are
local smoke tests; only the missing-source path is committed here because it's
the single assertion that proves the divergence-from-Apple-seed premise.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_missing_source_index_raises_runtime_error(tmp_path, monkeypatch):
    """Missing source index file → RuntimeError + 'source index missing' message.

    Defends the divergence from _seed_apple_collections's soft-fallback: any
    refactor that catches the exception and continues will fail this test.
    """
    from api import main as api_main

    # Point source at an empty tmp dir (no faiss_index.idx present).
    empty_source = tmp_path / "fake_seed_source"
    empty_source.mkdir()
    monkeypatch.setattr(api_main, "_REPO_FAISS_SOURCE", empty_source)

    # Point dest at a fresh writable tmp dir so the failure is unambiguously
    # the missing source, not a dest-side issue.
    dest = tmp_path / "fake_dest"
    monkeypatch.setattr(api_main.settings, "faiss_data_dir", str(dest))

    with pytest.raises(RuntimeError, match="source index missing"):
        api_main._seed_faiss_default()

    # Dest must not have been created — fail-hard means no silent partial state.
    assert not dest.exists(), (
        "Dest dir was created despite missing source — fail-hard contract broken; "
        "this is the silent-outage path the seed exists to prevent."
    )
