"""LangChain tool definitions for the agent graph.

Wraps existing retrieval infrastructure as LangChain tools that the
orchestrator can call during its self-correction loop.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import tool

from app.services.retrieval import hybrid_search_with_legs, get_effective_datastore_ids
from app.services.retrieval import score_retrieval


@tool
def search_documents(query: str, kb_ids: list, top_k: int = 10) -> str:
    """Search documents across knowledge bases for evidence related to the user question.

    Use this as the first retrieval step when answering a question that requires
    knowledge base lookup.

    Args:
        query: Focused search query with concrete keywords from the question.
        kb_ids: List of knowledge base IDs to search within.
        top_k: Maximum number of results to return (default: 10).

    Returns:
        String of retrieved documents with citation numbers.
    """
    # This tool's actual implementation is provided by a db injection
    # at graph compile time. The docstring above guides the orchestrator LLM.
    return ""


@tool
def retrieve_context(
    question: str,
    kb_ids: list,
    file_markdown: str = "",
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
    org_id: int = None,
    db: object = None,
) -> str:
    """Retrieve the best context chunks from the knowledge base for answering a question.

    This runs hybrid search (dense + sparse + exact) with reranking and
    confidence scoring.

    Args:
        question: The question to find context for.
        kb_ids: Knowledge base IDs to search.
        file_markdown: Optional markdown file content to include.
        use_dense: Enable vector (dense) search.
        use_sparse: Enable sparse (SPLADE) search.
        use_exact: Enable exact (BM25) search.
        use_graph_rag: Enable graph-based retrieval.
        org_id: Organization ID for multi-tenant filtering.
        db: SQLAlchemy database session.

    Returns:
        String of retrieved documents with citation numbers and a confidence score.
    """
    return ""
