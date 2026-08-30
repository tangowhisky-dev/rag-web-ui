"""Neo4j driver, schema, and global lifecycle management.

Owns all module-level mutable singletons shared across the graph
sub-modules:
  _neo4j_driver          — lazy Neo4j driver singleton
  _graph_batch_progress  — in-memory extraction progress tracker
  _graph_progress_lock   — thread-safe guard for the progress dict
  _global_llm_sem        — caps concurrent LLM calls across all builds
  _global_extractor      — LLM entity-relation extractor singleton
  _global_writer         — Neo4j writer singleton

Also provides:
  _get_driver()              — lazy driver init + schema ensure
  _ensure_schema()           — creates Neo4j indexes for fast traversal
  close_llm_clients()        — closes httpx clients before loop teardown
  get_graph_batch_progress() — reads in-memory progress for the API
  _chunk_id_to_point_id()    — SHA-256 hex → Qdrant UUID conversion
"""

import logging
import threading
from typing import Optional

import neo4j

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (lazy) ────────────────────────────────────────────
_neo4j_driver: Optional[neo4j.Driver] = None

# In-memory graph extraction progress: task_id → (completed_batches, total_batches)
# Ephemeral by design — lost on restart, but so is the interrupted extraction.
# Thread-safe via _graph_progress_lock (concurrent graph builds from recovery
# and live ingestion can update this dict simultaneously).
_graph_batch_progress: dict[int, tuple[int, int]] = {}
_graph_progress_lock = threading.Lock()

# Global semaphore limiting concurrent LLM calls across ALL graph builds.
# Each graph build thread runs in its own event loop, so we use
# threading.Semaphore (not asyncio.Semaphore) which works across
# threads/loops.  This caps total in-flight LLM calls at 4 regardless
# of how many documents are being graph-extracted simultaneously,
# preventing GPU endpoint overload.
_global_llm_sem = threading.Semaphore(4)

# LLM extractor/writer singletons — shared between setup.close_llm_clients
# and extraction._get_extractor_and_writer.  Defined here so both modules
# mutate the same objects.
_global_extractor = None
_global_writer = None


def get_graph_batch_progress(task_id: int) -> tuple[int, int] | None:
    """Return (completed_batches, total_batches) for an in-progress graph extraction."""
    with _graph_progress_lock:
        return _graph_batch_progress.get(task_id)


def _get_driver() -> neo4j.Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = neo4j.GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        _ensure_schema(_neo4j_driver)
    return _neo4j_driver


def _ensure_schema(driver: neo4j.Driver) -> None:
    """Create indexes so graph-expansion traversals aren't full label scans.

    Plain indexes (not uniqueness constraints) are used so repeated calls with
    IF NOT EXISTS are true no-ops — a uniqueness constraint creates a
    same-named backing index, which then collides on re-creation attempts.
    """
    try:
        with driver.session() as session:
            # Named distinctly from the earlier "chunk_qdrant_point_id" constraint
            # attempt to avoid a same-name constraint/index collision on rerun.
            session.run(
                "CREATE INDEX idx_chunk_qdrant_point_id IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.qdrant_point_id)"
            )
            session.run(
                "CREATE INDEX idx_chunk_qdrant_collection IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.qdrant_collection)"
            )
    except Exception as exc:
        logger.warning("GraphService: failed to ensure Neo4j schema indexes: %s", exc)


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """
    Convert a SHA-256 hex chunk ID to the deterministic UUID Qdrant uses.

    Delegates to the canonical implementation in document_qdrant.
    """
    from app.services.ingestion.document_qdrant import _chunk_id_to_point_id as _impl
    return _impl(chunk_id)


async def close_llm_clients():
    """Close the global extractor's httpx async clients and reset globals.

    Call this before closing the event loop that ran graph extraction.
    If left open, Python's GC tries to close the httpx AsyncClient after
    the loop is already closed, producing "Event loop is closed" errors.
    The next call to _get_extractor_and_writer() will recreate everything.
    """
    global _global_extractor, _global_writer
    if _global_extractor is not None:
        try:
            llm = getattr(_global_extractor, "llm", None)
            if llm is not None:
                # Close sync client first (no event loop needed)
                sync_client = getattr(llm, "client", None)
                if sync_client and hasattr(sync_client, "close"):
                    sync_client.close()
                # Close async client on the current loop
                async_client = getattr(llm, "async_client", None)
                if async_client and hasattr(async_client, "close"):
                    await async_client.close()
        except Exception as e:
            logger.debug("GraphService[llm]: error closing LLM clients: %s", e)
    _global_extractor = None
    _global_writer = None
