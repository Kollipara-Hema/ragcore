__all__ = ["search_docs", "summarize_doc", "compare_docs", "get_metadata"]


def __getattr__(name):
    if name == "search_docs":
        from agent.tools.search_docs import search_docs
        return search_docs
    if name == "summarize_doc":
        from agent.tools.summarize_doc import summarize_doc
        return summarize_doc
    if name == "compare_docs":
        from agent.tools.compare_docs import compare_docs
        return compare_docs
    if name == "get_metadata":
        from agent.tools.get_metadata import get_metadata
        return get_metadata
    raise AttributeError(f"module 'agent.tools' has no attribute {name!r}")
