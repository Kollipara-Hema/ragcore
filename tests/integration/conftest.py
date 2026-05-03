"""
Integration-test fixtures shared across test files in this directory.
"""
import pytest


@pytest.fixture(autouse=True)
def _use_faiss_for_get_vector_store():
    """
    Force FAISS provider when calling get_vector_store().

    The singleton and pipeline tests were written against FAISSVectorStore and
    rely on get_vector_store() returning a FAISS instance.  Without this
    fixture, a local .env with VECTOR_STORE_PROVIDER=chroma would route those
    tests through ChromaVectorStore with no tmp_path isolation, causing data
    bleed between tests.

    Chroma tests (test_chroma_vector_store.py) instantiate ChromaVectorStore
    directly with persist_dir=tmp_path and do not call get_vector_store(), so
    this fixture does not affect them.
    """
    from config.settings import settings, VectorStoreProvider
    original = settings.vector_store_provider
    settings.vector_store_provider = VectorStoreProvider.FAISS
    yield
    settings.vector_store_provider = original
