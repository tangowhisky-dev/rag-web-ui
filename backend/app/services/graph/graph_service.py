"""Compatibility shim — re-exports from split sub-modules.

All functionality has been split into sibling modules within this package:
  setup.py       — driver/schema/progress/LLM client lifecycle
  extraction.py  — LLM entity-relation extraction pipeline
  build.py       — document graph ingestion
  expand.py      — graph-expanded retrieval and enrichment
  delete.py      — graph deletion and stale data purge

This module exists solely so legacy imports of the form
``from app.services.graph.graph_service import X`` continue to work.
New code should import from the specific sub-module or from
``app.services.graph`` (the package __init__).
"""

from .setup import (
    _get_driver,
    _ensure_schema,
    close_llm_clients,
    get_graph_batch_progress,
    _chunk_id_to_point_id,
)
from .extraction import (
    _get_extractor_and_writer,
    _build_extraction_batches,
    _acquire_global_llm_sem,
    _extract_with_llm,
    _strip_overlap,
)
from .build import (
    build_graph_for_document,
    _extract_seen_point_ids,
)
from .expand import (
    _build_graph_scope_filter,
    _build_traversal_patterns,
    _traverse_graph_for_expansion,
    _fetch_expanded_docs_from_qdrant,
    expand_docs_via_graph,
    enrich_docs_with_graph,
)
from .delete import (
    delete_graph_for_document,
    delete_graph_for_kb,
    _batch_delete_chunks,
    _batch_delete_orphaned_entities,
    purge_stale_graph_data,
)
