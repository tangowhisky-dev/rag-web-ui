"""
Neo4j knowledge graph service.

Architecture
------------
Strict separation of concerns:
  Qdrant  — source of truth for all chunk TEXT and VECTORS
  Neo4j   — source of truth for GRAPH TOPOLOGY (entities, relationships, chunk linkage)

Cross-reference
---------------
The link between the two systems is the Qdrant point UUID, stored as
`qdrant_point_id` on every Chunk node in Neo4j. This enables:

  1. Graph-expanded retrieval — vector search returns point UUIDs → traverse
     Neo4j to find entity-connected Chunk nodes NOT in the original result set
     → fetch those Chunks from Qdrant by UUID → merge into candidate pool.

  2. Entity context enrichment — for the final top-k docs, traverse FROM_CHUNK
     edges to Entity nodes, collect relationship triples, append a [Graph context]
     block to each chunk's text so the LLM sees the entity graph.

No vectors are stored in Neo4j. All text retrieval goes through Qdrant. Neo4j
is only traversed for graph topology.

Extraction backend
------------------
  LLM pipeline  (GRAPHRAG_LLM=<model_name>):
    neo4j-graphrag Pipeline + LLMEntityRelationExtractor with
    use_structured_output=True — JSON Schema-constrained output, no free-form
    JSON drift.

Ingestion (build_graph_for_document)
-------------------------------------
  1. Write Chunk nodes keyed by qdrant_point_id (the UUID Qdrant uses as point ID).
     Also carry document_id + chunk_index for backward-compatible lookups.
  2. Run extraction → write Entity nodes + typed relationship edges.
  3. Link Entity nodes to Chunk nodes via FROM_CHUNK edges.

Retrieval
---------
  expand_docs_via_graph(docs, kb_ids)
    Takes already-retrieved docs (from vector search + RRF).
    Extracts their Qdrant point IDs → traverses Neo4j graph
    (chunk → entity → entity → chunk) to find RELATED chunks.
    Fetches those related chunks from Qdrant by UUID.
    Returns them as additional LangchainDocument objects to merge into
    the candidate pool BEFORE reranking.

  enrich_docs_with_graph(docs)
    For the final top-k docs: append entity relationship triples as
    [Graph context] text so the LLM sees the graph alongside the chunk.
    Non-fatal per doc.

Deletion
--------
  delete_graph_for_document / delete_graph_for_kb
  Always run regardless of GRAPHRAG_ENABLED — data may exist from a prior
  ingest run. Also cleans up orphaned Entity nodes with no remaining chunk links.
"""

import asyncio
import logging
import threading
import uuid
from typing import Optional
from sqlalchemy.orm import Session


import neo4j
from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings
from app.services.agentic_rag.retry import with_retry_sync
from app.services.settings_service import get_setting

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


# ── LLM pipeline path ─────────────────────────────────────────────────────────

_global_extractor = None
_global_writer = None


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


def _get_extractor_and_writer():
    """
    Build (once) the LLM entity-relation extractor and Neo4j writer as
    separate components so we can intercept the extracted graph between
    extraction and writing — specifically to link only the batch's extracted
    entities to the batch's chunks via FROM_CHUNK edges.

    use_structured_output=True makes the LLM return a JSON Schema-validated
    Neo4jGraph object — no regex post-processing, no json.loads guesswork.
    The LLM must support response_format=json_schema (OpenAI GPT-4 class and
    most modern OpenAI-compatible endpoints do).
    """
    global _global_extractor, _global_writer
    if _global_extractor is not None and _global_writer is not None:
        return _global_extractor, _global_writer

    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.experimental.components.entity_relation_extractor import (
        LLMEntityRelationExtractor,
        OnError,
    )
    from neo4j_graphrag.experimental.components.kg_writer import (
        KGWriterModel,
        LexicalGraphConfig,
        Neo4jWriter,
    )
    from neo4j_graphrag.experimental.components.types import Neo4jNode, Neo4jGraph

    class _SafeNeo4jWriter(Neo4jWriter):
        """Strips empty/non-finite embedding_properties before writing to Neo4j.

        The LLM occasionally returns malformed entity nodes that end up with
        embedding_properties={"embedding": []} — an empty list that Neo4j's
        db.create.setNodeVectorProperty() rejects with
        'Vector must only contain finite values. Provided: List{}'.

        We don't store vectors in Neo4j (Qdrant is the vector store), so the
        safest fix is to clear all embedding_properties before each write.
        """

        @staticmethod
        def _clean_nodes(nodes: list[Neo4jNode]) -> list[Neo4jNode]:
            cleaned = []
            for node in nodes:
                if node.embedding_properties:
                    # Keep only entries that are non-empty lists of finite floats
                    valid = {
                        k: v for k, v in node.embedding_properties.items()
                        if v and all(isinstance(x, (int, float)) and not (x != x or x == float("inf") or x == float("-inf")) for x in v)
                    }
                    if valid != node.embedding_properties:
                        node = Neo4jNode(
                            id=node.id,
                            label=node.label,
                            properties=node.properties,
                            embedding_properties=valid,
                        )
                cleaned.append(node)
            return cleaned

        def _upsert_nodes(self, nodes, lexical_graph_config):  # type: ignore[override]
            super()._upsert_nodes(self._clean_nodes(nodes), lexical_graph_config)

        async def run(
            self,
            graph: Neo4jGraph,
            lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
        ) -> KGWriterModel:
            # Delegate to parent — _upsert_nodes override cleans nodes
            # before writing via the parent's run() method.
            return await super().run(graph, lexical_graph_config)

    # Graph extraction LLM credentials are super_admin-only (app scope).
    from app.services.settings_service import get_setting
    from app.db.session import SessionLocal
    _db = SessionLocal()
    try:
        graph_model = get_setting(_db, "GRAPHRAG_LLM", None) or get_setting(_db, "OPENAI_MODEL", None)
        api_key = get_setting(_db, "GRAPHRAG_API_KEY", None) or get_setting(_db, "OPENAI_API_KEY", None)
        api_base = get_setting(_db, "GRAPHRAG_API_BASE", None) or get_setting(_db, "OPENAI_API_BASE", None)
    finally:
        _db.close()

    if not api_key:
        api_key = "not-required"

    llm = OpenAILLM(
        model_name=graph_model,
        model_params={
            "temperature": 0,
        },
        base_url=api_base,
        api_key=api_key,
    )

    _global_extractor = LLMEntityRelationExtractor(
        llm=llm,
        use_structured_output=True,
        on_error=OnError.IGNORE,
        create_lexical_graph=False,
        max_concurrency=1,   # 1 = fully sequential; local models OOM at >1
    )

    _global_writer = _SafeNeo4jWriter(
        driver=_get_driver(),
        neo4j_database="neo4j",
        batch_size=500,
    )

    logger.info("GraphService[llm]: extractor+writer built with model=%s", graph_model)
    return _global_extractor, _global_writer


def _strip_overlap(prev: str, curr: str, max_search: int) -> str:
    """Strip the overlapping prefix from *curr* that duplicates the tail of *prev*.

    Searches for the longest suffix of *prev* (up to *max_search* chars) that
    appears as a prefix of *curr* and strips it.  Returns *curr* unchanged if
    no overlap is found.
    """
    search_len = min(len(prev), len(curr), max_search)
    for length in range(search_len, 0, -1):
        if prev[-length:] == curr[:length]:
            return curr[length:]
    return curr


def _build_extraction_batches(
    chunks: list[str],
    point_ids: list[str],
    context_budget: int,
) -> list[tuple[str, list[str]]]:
    """Group consecutive chunks into batches whose combined text fits within
    *context_budget* characters (80% of NEO4J_LLM_CONTEXT, reserving headroom
    for the system prompt and JSON schema output).

    Within each batch, chunk overlap is stripped from chunk[i+1] before
    concatenation so the LLM sees clean, non-redundant prose.

    Returns a list of (combined_text, [point_id, ...]) tuples — one per batch.
    Each entry in the point_id list corresponds to one of the original Qdrant
    chunks so FROM_CHUNK edges fan out to all of them after extraction.
    """
    budget = int(context_budget * 0.33)
    # neo4j-graphrag's system prompt + JSON schema for structured output consumes
    # ~800-1200 tokens. Using 33% of the char budget leaves 67% headroom for
    # prompt overhead + output tokens, keeping total well within the model's limit.
    max_overlap_search = max(200, getattr(settings, "chunk_overlap", 200) * 2)

    batches: list[tuple[str, list[str]]] = []
    batch_texts: list[str] = []
    batch_ids: list[str] = []
    running_len = 0

    for i, (text, pid) in enumerate(zip(chunks, point_ids)):
        # Strip overlap with the previous chunk in this batch
        if batch_texts:
            deduped = _strip_overlap(batch_texts[-1], text, max_overlap_search)
        else:
            deduped = text

        if running_len + len(deduped) > budget and batch_texts:
            # Flush current batch and start a new one with the full chunk text
            batches.append((" ".join(batch_texts), batch_ids))
            batch_texts = [text]      # new batch starts with full text
            batch_ids = [pid]
            running_len = len(text)
        else:
            batch_texts.append(deduped)
            batch_ids.append(pid)
            running_len += len(deduped)

    if batch_texts:
        batches.append((" ".join(batch_texts), batch_ids))

    return batches


async def _acquire_global_llm_sem(cancel_event) -> bool:
    """Acquire the global LLM semaphore, yielding to the event loop.

    Returns True if acquired, False if cancelled while waiting.
    Uses non-blocking poll (50 ms interval) so the event loop stays
    responsive and no thread-pool threads are consumed — unlike
    ``run_in_executor`` with a blocking ``acquire()`` which would
    exhaust the default thread pool under high batch concurrency.
    """
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return False
        if _global_llm_sem.acquire(blocking=False):
            return True
        await asyncio.sleep(0.05)


async def _extract_with_llm(
    document_id: int,
    file_name: str,
    chunks: list[str],
    qdrant_point_ids: list[str],
    pt=None,            # optional ProgressTimeout for periodic pings
    max_chunks: int = 0,
    neo4j_llm_context: int = 12000,
    kb_id: Optional[int] = None,
    data_store_id: Optional[int] = None,
    task_id: Optional[int] = None,
    cancel_event=None,  # optional threading.Event — abort if set
) -> tuple[int, int, int]:
    """Run neo4j-graphrag LLM extraction on context-sized batches of chunks.

    Consecutive chunks are merged into batches sized to NEO4J_LLM_CONTEXT
    (with overlap stripped) before being sent to the LLM. This gives the
    extractor broader context than single-chunk calls, reducing duplicate
    entities at chunk boundaries and surfacing intra-document relationships.

    The extractor and writer are called separately (not via Pipeline) so we
    can intercept the extracted Neo4jGraph and link ONLY the entities from
    this batch to the batch's chunks via FROM_CHUNK edges.

    Up to 4 batches run concurrently. All synchronous Neo4j I/O is
    dispatched via run_in_executor.

    Returns (total_nodes, total_rels, skipped_batches). skipped_batches > 0
    means some batches were skipped due to cancellation/pause — the caller
    should mark the graph build as pending, not completed, so it gets retried.
    """
    from neo4j_graphrag.experimental.components.types import TextChunks, TextChunk

    loop = asyncio.get_event_loop()
    extractor, writer = _get_extractor_and_writer()
    driver = _get_driver()

    # Cap chunks to avoid OOM on low-RAM local models. Qdrant still has all chunks.
    cap = max_chunks
    effective_chunks = chunks if cap <= 0 else chunks[:cap]
    effective_ids = qdrant_point_ids if cap <= 0 else qdrant_point_ids[:cap]
    if cap > 0 and len(chunks) > cap:
        logger.info(
            "GraphService[llm]: doc %d — capping graph extraction at %d/%d chunks (GRAPHRAG_MAX_CHUNKS=%d)",
            document_id, cap, len(chunks), cap,
        )

    batches = _build_extraction_batches(
        effective_chunks, effective_ids, neo4j_llm_context
    )
    logger.info(
        "GraphService[llm]: doc %d — %d chunks → %d extraction batches (NEO4J_LLM_CONTEXT=%d)",
        document_id, len(effective_chunks), len(batches), neo4j_llm_context,
    )

    # Track per-batch progress in-memory for the API to read
    total_batches = len(batches)
    completed_batches = 0
    skipped_batches = 0
    if task_id is not None:
        with _graph_progress_lock:
            _graph_batch_progress[task_id] = (0, total_batches)

    def _is_cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _inject_scope_props(graph):
        scope_props = {}
        if kb_id is not None:
            scope_props["kb_id"] = str(kb_id)
        if data_store_id is not None:
            scope_props["data_store_id"] = str(data_store_id)
        if scope_props:
            from neo4j_graphrag.experimental.components.types import Neo4jNode as _Neo4jNode
            scoped_nodes = []
            for n in graph.nodes:
                merged_props = {**n.properties, **scope_props}
                scoped_nodes.append(_Neo4jNode(
                    id=n.id,
                    label=n.label,
                    properties=merged_props,
                    embedding_properties=n.embedding_properties,
                ))
            from neo4j_graphrag.experimental.components.types import Neo4jGraph as _Neo4jGraph
            graph = _Neo4jGraph(nodes=scoped_nodes, relationships=graph.relationships)
        return graph, scope_props

    async def _attempt_batch_extraction(
        batch_idx: int, combined_text: str, batch_point_ids: list[str]
    ) -> tuple[int, int, bool]:
        graph = await extractor.run(
            chunks=TextChunks(chunks=[TextChunk(text=combined_text, index=batch_idx)]),
            examples="",
        )
        if _is_cancelled():
            logger.info(
                "GraphService[llm]: doc %d batch %d — cancelled during LLM call, discarding result",
                document_id, batch_idx,
            )
            return 0, 0, True
        if data_store_id is not None:
            from app.services.ingestion.ingestion_dispatcher import is_datastore_deleted
            if is_datastore_deleted(data_store_id):
                logger.info(
                    "GraphService[llm]: doc %d batch %d — datastore %s deleted during LLM call, discarding",
                    document_id, batch_idx, data_store_id,
                )
                return 0, 0, False
        graph, scope_props = _inject_scope_props(graph)
        writer_result = await writer.run(graph=graph)
        status = getattr(writer_result, "status", None)
        if status == "FAILURE":
            logger.warning(
                "GraphService[llm]: writer FAILURE for doc %d batch %d — skipping FROM_CHUNK links",
                document_id, batch_idx,
            )
            return 0, 0, False
        batch_entities = [
            (n.properties.get("name"), n.label)
            for n in graph.nodes
            if n.properties.get("name")
        ]
        if not batch_entities:
            return 0, 0, False

        def _link_batch_chunks(
            entities=batch_entities, pids=batch_point_ids,
            scope=scope_props
        ):
            linked_total = 0
            with driver.session() as session:
                scope_clauses = []
                if "kb_id" in scope:
                    scope_clauses.append("e.kb_id = $kb_id")
                if "data_store_id" in scope:
                    scope_clauses.append("e.data_store_id = $ds_id")
                scope_filter = " AND ".join(scope_clauses) if scope_clauses else "true"

                rec = session.run(
                    f"""
                    UNWIND $entities AS ent
                    UNWIND $point_ids AS pid
                    MATCH (e:__Entity__)
                    WHERE e.name = ent.name
                      AND ent.label IN labels(e)
                      AND {scope_filter}
                    MATCH (c:Chunk {{qdrant_point_id: pid}})
                    MERGE (e)-[:FROM_CHUNK]->(c)
                    RETURN count(*) AS linked
                    """,
                    entities=[
                        {"name": name, "label": label}
                        for name, label in entities
                    ],
                    point_ids=pids,
                    kb_id=scope.get("kb_id"),
                    ds_id=scope.get("data_store_id"),
                ).single()
                linked_total += rec["linked"] if rec else 0
            return linked_total

        linked = await loop.run_in_executor(None, _link_batch_chunks)
        if pt:
            pt.ping()
        return linked, 0, False

    async def _run_batch_with_retries(
        batch_idx: int, combined_text: str, batch_point_ids: list[str]
    ) -> tuple[int, int, bool]:
        last_exc = None
        for attempt in range(1, 4):
            try:
                return await _attempt_batch_extraction(batch_idx, combined_text, batch_point_ids)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "GraphService[llm]: extraction failed for doc %d batch %d (attempt %d/3): %s",
                    document_id, batch_idx, attempt, exc,
                )
                if attempt < 3:
                    await asyncio.sleep(1)
        logger.error(
            "GraphService[llm]: all 3 attempts failed for doc %d batch %d — giving up: %s",
            document_id, batch_idx, last_exc,
        )
        return 0, 0, False

    async def _process_batch(
        batch_idx: int, combined_text: str, batch_point_ids: list[str]
    ) -> tuple[int, int]:
        nonlocal skipped_batches, completed_batches
        if _is_cancelled():
            logger.info(
                "GraphService[llm]: doc %d batch %d — cancelled, skipping",
                document_id, batch_idx,
            )
            skipped_batches += 1
            return 0, 0
        acquired = await _acquire_global_llm_sem(cancel_event)
        if not acquired:
            logger.info(
                "GraphService[llm]: doc %d batch %d — cancelled while waiting for LLM semaphore, skipping",
                document_id, batch_idx,
            )
            skipped_batches += 1
            return 0, 0
        try:
            if _is_cancelled():
                logger.info(
                    "GraphService[llm]: doc %d batch %d — cancelled while waiting for semaphore, skipping",
                    document_id, batch_idx,
                )
                skipped_batches += 1
                return 0, 0
            linked, rel_count, skip = await _run_batch_with_retries(
                batch_idx, combined_text, batch_point_ids
            )
            if skip:
                skipped_batches += 1
                return 0, 0
            if task_id is not None:
                completed_batches += 1
                with _graph_progress_lock:
                    _graph_batch_progress[task_id] = (completed_batches, total_batches)
            return linked, rel_count
        finally:
            _global_llm_sem.release()

    results = await asyncio.gather(*[
        _process_batch(idx, text, pids)
        for idx, (text, pids) in enumerate(batches)
    ])

    # Clean up in-memory progress tracking
    if task_id is not None:
        with _graph_progress_lock:
            _graph_batch_progress.pop(task_id, None)

    # Final accurate count from Neo4j.
    def _count_nodes_rels():
        with driver.session() as session:
            rec = session.run(
                """
                MATCH (c:Chunk {document_id: $doc_id})<-[:FROM_CHUNK]-(e)
                WITH collect(DISTINCT e) AS entities
                UNWIND entities AS e
                OPTIONAL MATCH (e)-[r]-(e2)
                WHERE e2 IN entities
                RETURN size(entities) AS node_count,
                       count(r)      AS rel_count
                """,
                doc_id=str(document_id),
            ).single()
            return (rec["node_count"] if rec else 0, rec["rel_count"] if rec else 0)

    total_nodes, total_rels = await loop.run_in_executor(None, _count_nodes_rels)
    return total_nodes, total_rels, skipped_batches


# ── Ingest ─────────────────────────────────────────────────────────────────────

async def build_graph_for_document(
    kb_id: Optional[int],
    document_id: int,
    file_name: str,
    chunks: list[str],
    chunk_ids: list[str],
    data_store_id: Optional[int] = None,
    pt=None,            # optional ProgressTimeout for periodic pings
    db: Optional[Session] = None,
    org_id: Optional[int] = None,
    task_id: Optional[int] = None,
    cancel_event=None,  # optional threading.Event — abort if set
) -> int:
    """Extract entity/relationship graph from document chunks and store in Neo4j.

    Returns the number of extraction batches skipped due to cancellation/pause.

    chunk_ids are the SHA-256 hex strings from document_processor — we convert
    them to the actual Qdrant point UUIDs here via the same deterministic
    uuid5 transform, so qdrant_point_id in Neo4j always matches the Qdrant
    point ID exactly.

    Chunk nodes carry:
      qdrant_point_id    — primary cross-reference key (UUID string)
      qdrant_collection  — which Qdrant collection (kb_<kb_id> or ds_<data_store_id>)
      document_id        — for deletion queries
      chunk_index        — ordinal position within the document
      kb_id              — knowledge base
      data_store_id      — datastore (optional)
      file_name          — human-readable source
    """
    if not get_setting(db, "GRAPHRAG_ENABLED", None):
        return

    # Resolve app-level ingestion settings from the settings service
    from app.db.session import SessionLocal
    _db = SessionLocal()
    try:
        graphrag_enabled = get_setting(_db, "GRAPHRAG_ENABLED", None)
        max_chunks = get_setting(_db, "GRAPHRAG_MAX_CHUNKS", None)
        neo4j_llm_context = get_setting(_db, "NEO4J_LLM_CONTEXT", None)
    finally:
        _db.close()

    if not graphrag_enabled:
        return

    driver = _get_driver()
    loop = asyncio.get_event_loop()

    # Convert SHA-256 chunk IDs → the actual UUIDs Qdrant uses as point IDs
    qdrant_point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
    # Determine collection based on document source
    if data_store_id:
        collection_name = f"ds_{data_store_id}"
    else:
        collection_name = f"kb_{kb_id}"

    logger.info(
        "GraphService[llm]: writing %d Chunk nodes for doc %d (kb=%s, ds=%s)",
        len(chunks), document_id, kb_id, data_store_id,
    )

    # Write Chunk nodes — sync Neo4j I/O, must run in executor to avoid
    # blocking the event loop and causing ECONNRESET on concurrent requests.
    def _write_chunk_nodes():
        with driver.session() as session:
            for idx, (text, point_id) in enumerate(zip(chunks, qdrant_point_ids)):
                session.run(
                    """
                    MERGE (c:Chunk {qdrant_point_id: $point_id})
                    SET c.qdrant_collection = $collection,
                        c.document_id       = $document_id,
                        c.chunk_index       = $chunk_index,
                        c.kb_id             = $kb_id,
                        c.data_store_id     = $data_store_id,
                        c.file_name         = $file_name
                    """,
                    point_id=point_id,
                    collection=collection_name,
                    document_id=str(document_id),
                    chunk_index=idx,
                    kb_id=str(kb_id),
                    data_store_id=str(data_store_id) if data_store_id else None,
                    file_name=file_name,
                )

    await loop.run_in_executor(None, _write_chunk_nodes)

    # Check if cancelled before starting expensive LLM extraction.
    if cancel_event is not None and cancel_event.is_set():
        logger.info("GraphService[llm]: doc %d — cancelled before extraction", document_id)
        return 0

    # Extract entities and relationships.  Concurrency is controlled by the
    # global LLM semaphore inside _extract_with_llm, which caps total
    # concurrent LLM calls across all documents at 4.
    total_entities, total_relations, skipped_batches = await _extract_with_llm(
        document_id=document_id,
        file_name=file_name,
        chunks=chunks,
        qdrant_point_ids=qdrant_point_ids,
        pt=pt,
        max_chunks=max_chunks,
        neo4j_llm_context=neo4j_llm_context,
        kb_id=kb_id,
        data_store_id=data_store_id,
        task_id=task_id,
        cancel_event=cancel_event,
    )

    logger.info(
        "GraphService[llm]: doc %d — %d entities, %d relations written to Neo4j (skipped_batches=%d)",
        document_id, total_entities, total_relations, skipped_batches,
    )
    return skipped_batches


# ── Retrieval: graph expansion ─────────────────────────────────────────────────

def _extract_seen_point_ids(docs: list[LangchainDocument]) -> set[str]:
    """Extract Qdrant point UUIDs from the retrieved docs' metadata."""
    seen = set()
    for doc in docs:
        pid = doc.metadata.get("qdrant_point_id")
        if pid:
            seen.add(pid)
    return seen


def _build_graph_scope_filter(
    kb_ids: list[int],
    datastore_ids: Optional[list[int]],
) -> tuple[list[str], list[str], str]:
    """Build entity scope filter for graph traversal.

    Ensures graph traversal stays within the queried KB(s)/datastore(s)
    and doesn't cross-contaminate via shared entity nodes from other KBs.
    Entities without scope props are from older ingestion runs —
    include them for backward compatibility.
    """
    kb_scope = [str(k) for k in kb_ids] if kb_ids else []
    ds_scope = [str(d) for d in datastore_ids] if datastore_ids else []
    scope_clauses = []
    if kb_scope:
        scope_clauses.append("e.kb_id IN $kb_scope")
    if ds_scope:
        scope_clauses.append("e.data_store_id IN $ds_scope")
    if scope_clauses:
        scope_filter = "(" + " OR ".join(scope_clauses) + " OR e.kb_id IS NULL AND e.data_store_id IS NULL)"
    else:
        scope_filter = "true"
    return kb_scope, ds_scope, scope_filter


def _build_traversal_patterns(hops_val: int) -> tuple[str, str]:
    """Build Cypher path pattern and intermediate entity filter for N-hop traversal.

    1 hop: (e)-[:FROM_CHUNK]->(c2)
    2 hops: (e)-[r1]-(e2)-[:FROM_CHUNK]->(c2)
    N hops: chain of N entity nodes with N-1 relationships
    """
    hops = max(1, hops_val)
    if hops == 1:
        rest_pattern = "(e)-[:FROM_CHUNK]->(c2)"
    else:
        parts = ["(e)"]
        for i in range(2, hops + 1):
            parts.append(f"-[r{i - 1}]-(e{i})")
        parts.append("-[:FROM_CHUNK]->(c2)")
        rest_pattern = " ".join(parts)

    if hops > 1:
        interm_clauses = []
        for i in range(2, hops + 1):
            interm_clauses.append(f"(e{i}.kb_id IN $kb_scope OR e{i}.data_store_id IN $ds_scope OR (e{i}.kb_id IS NULL AND e{i}.data_store_id IS NULL))")
        interm_filter = " AND ".join(interm_clauses)
    else:
        interm_filter = "true"
    return rest_pattern, interm_filter


def _traverse_graph_for_expansion(
    driver: neo4j.Driver,
    seen_point_ids: set[str],
    collections: list[str],
    scope_filter: str,
    rest_pattern: str,
    interm_filter: str,
    fanout_val: int,
    limit_val: int,
    kb_scope: list[str],
    ds_scope: list[str],
) -> list[tuple[str, str]]:
    """Traverse from seed chunks via entity relationships to connected chunks.

    The first hop's distinct entities are capped (GRAPHRAG_ENTITY_FANOUT_CAP)
    before expanding further, so a handful of highly-connected "hub"
    entities (e.g. generic terms shared by hundreds of chunks) can't blow
    up the traversal into a combinatorial cross product.
    Return qdrant_point_id + qdrant_collection of chunks NOT already seen.
    """
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (c:Chunk)
            WHERE c.qdrant_point_id IN $seen_ids
              AND c.qdrant_collection IN $collections
            MATCH (c)<-[:FROM_CHUNK]-(e)
            WHERE {scope_filter}
            WITH DISTINCT e LIMIT $entity_cap
            MATCH {rest_pattern}
            WHERE c2.qdrant_point_id IS NOT NULL
              AND NOT c2.qdrant_point_id IN $seen_ids
              AND c2.qdrant_collection IN $collections
              AND {interm_filter}
            RETURN DISTINCT c2.qdrant_point_id AS point_id,
                            c2.qdrant_collection AS collection
            LIMIT $limit
            """,
            seen_ids=list(seen_point_ids),
            collections=collections,
            entity_cap=max(1, fanout_val),
            limit=max(1, limit_val),
            kb_scope=kb_scope,
            ds_scope=ds_scope,
        )
        return [
            (rec["point_id"], rec["collection"]) for rec in result
        ]


def _fetch_expanded_docs_from_qdrant(
    expansion_targets: list[tuple[str, str]],
) -> list[LangchainDocument]:
    """Fetch chunk text from Qdrant by point UUID and build LangchainDocuments.

    Groups targets by collection and fetches text/payload only (no re-embedding).
    """
    from collections import defaultdict
    from qdrant_client import QdrantClient

    by_collection: dict[str, list[str]] = defaultdict(list)
    for point_id, collection in expansion_targets:
        by_collection[collection].append(point_id)

    qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    expanded_docs: list[LangchainDocument] = []

    for collection, point_ids in by_collection.items():
        try:
            points = qdrant.retrieve(
                collection_name=collection,
                ids=point_ids,
                with_payload=True,
                with_vectors=False,   # text only — Qdrant is the source of truth
            )
            for pt in points:
                payload = pt.payload or {}
                chunk_text = payload.get("chunk_text", "")
                if not chunk_text:
                    continue
                meta = {k: v for k, v in payload.items() if k != "chunk_text"}
                meta["_graph_expanded"] = True
                meta["qdrant_point_id"] = str(pt.id)
                expanded_docs.append(
                    LangchainDocument(page_content=chunk_text, metadata=meta)
                )
        except Exception as exc:
            logger.warning(
                "GraphService.expand: Qdrant retrieve failed for collection %s: %s",
                collection, exc,
            )
    return expanded_docs


@with_retry_sync(max_attempts=3)
def expand_docs_via_graph(
    docs: list[LangchainDocument],
    kb_ids: list[int],
    db: Optional[Session] = None,
    org_id: Optional[int] = None,
    datastore_ids: Optional[list[int]] = None,
) -> list[LangchainDocument]:
    """
    Graph-expanded retrieval: find additional chunks via Neo4j graph traversal
    and fetch their text from Qdrant.

    Flow:
      1. Extract Qdrant point UUIDs from the already-retrieved docs.
      2. Query Neo4j: traverse chunk → entity → entity → chunk to find
         entity-connected chunks whose qdrant_point_id is NOT already in
         the current result set.
      3. Fetch those new points from Qdrant by UUID (text/payload only,
         no vector computation needed).
      4. Return them as LangchainDocument objects with metadata flag
         `_graph_expanded=True` so the caller can annotate them.

    This surfaces chunks that are SEMANTICALLY linked via entity relationships
    but would not have been returned by vector similarity alone.

    When db and org_id are provided, org-overridable settings (hops, limit,
    fanout) are resolved via the settings service.

    Non-fatal — returns [] on any failure so the caller's pipeline continues
    with only the original vector search results.
    """
    if not get_setting(db, "GRAPHRAG_ENABLED", None) or not docs:
        return []

    from qdrant_client import QdrantClient

    # Resolve org-overridable settings
    hops_val = get_setting(db, "GRAPHRAG_RETRIEVAL_HOPS", org_id)
    fanout_val = get_setting(db, "GRAPHRAG_ENTITY_FANOUT_CAP", org_id)
    limit_val = get_setting(db, "GRAPHRAG_RETRIEVAL_LIMIT", org_id)

    seen_point_ids = _extract_seen_point_ids(docs)

    if not seen_point_ids:
        # Docs came from before the qdrant_point_id payload field was added —
        # fall back gracefully rather than blowing up.
        logger.debug("GraphService.expand: no qdrant_point_id in doc metadata, skipping expansion")
        return []

    try:
        driver = _get_driver()
        collections = [f"kb_{kb_id}" for kb_id in kb_ids]
        if datastore_ids:
            collections += [f"ds_{ds_id}" for ds_id in datastore_ids]

        kb_scope, ds_scope, scope_filter = _build_graph_scope_filter(kb_ids, datastore_ids)
        rest_pattern, interm_filter = _build_traversal_patterns(hops_val)

        expansion_targets = _traverse_graph_for_expansion(
            driver, seen_point_ids, collections, scope_filter,
            rest_pattern, interm_filter, fanout_val, limit_val,
            kb_scope, ds_scope,
        )
        if not expansion_targets:
            logger.debug("GraphService.expand: no graph-connected chunks found beyond current result set")
            return []

        logger.info(
            "GraphService.expand: found %d graph-connected chunks to fetch from Qdrant",
            len(expansion_targets),
        )

        expanded_docs = _fetch_expanded_docs_from_qdrant(expansion_targets)

        logger.info(
            "GraphService.expand: fetched %d graph-expanded docs from Qdrant",
            len(expanded_docs),
        )
        return expanded_docs

    except Exception as exc:
        logger.warning("GraphService.expand: expansion failed (non-fatal): %s", exc)
        return []


# ── Retrieval: entity context enrichment ──────────────────────────────────────

@with_retry_sync(max_attempts=3)
def enrich_docs_with_graph(
    docs: list[LangchainDocument],
    db: Optional[Session] = None,
    org_id: Optional[int] = None,
) -> list[LangchainDocument]:
    """
    Append [Graph context] entity relationship triples to each doc's text.

    Looks up each Chunk in Neo4j by qdrant_point_id (primary key), traverses
    FROM_CHUNK edges to Entity nodes, collects entity-to-entity relationship
    triples, and appends them as a compact text block.

    This gives the LLM explicit relationship context alongside the chunk text.
    Called AFTER expansion+reranking so only the final top-k docs are enriched.

    Falls back to (document_id, chunk_index) lookup for docs that predate the
    qdrant_point_id field being added to Qdrant payloads.

    Non-fatal per doc — failures return the doc unchanged.
    """
    if not get_setting(db, "GRAPHRAG_ENABLED", None) or not docs:
        return docs

    driver = _get_driver()
    enriched = []
    graph_hits = 0

    for doc in docs:
        point_id  = doc.metadata.get("qdrant_point_id")
        doc_id    = doc.metadata.get("document_id")
        chunk_idx = doc.metadata.get("chunk_index")

        try:
            with driver.session() as session:
                if point_id:
                    # Primary path: lookup by Qdrant point UUID (O(1) index hit).
                    # Neighbor traversal is scoped to entities that also have a
                    # FROM_CHUNK edge into this same Qdrant collection — prevents
                    # stale inter-entity edges from deleted KBs bleeding in.
                    result = session.run(
                        """
                        MATCH (c:Chunk {qdrant_point_id: $point_id})
                        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c)
                        OPTIONAL MATCH (e)-[r]-(neighbor)
                        WHERE EXISTS { MATCH (neighbor)-[:FROM_CHUNK]->(:Chunk {qdrant_collection: c.qdrant_collection}) }
                        WITH e.name AS ename, type(r) AS rel, neighbor.name AS nname
                        WHERE ename IS NOT NULL AND nname IS NOT NULL
                        RETURN collect(DISTINCT [ename, rel, nname])[..40] AS triples
                        """,
                        point_id=point_id,
                    )
                elif doc_id is not None and chunk_idx is not None:
                    # Fallback: legacy lookup by (document_id, chunk_index).
                    # No collection scoping available — best-effort.
                    result = session.run(
                        """
                        MATCH (c:Chunk {document_id: $document_id, chunk_index: $chunk_index})
                        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c)
                        OPTIONAL MATCH (e)-[r]-(neighbor)
                        WITH e.name AS ename, type(r) AS rel, neighbor.name AS nname
                        WHERE ename IS NOT NULL AND nname IS NOT NULL
                        RETURN collect(DISTINCT [ename, rel, nname])[..40] AS triples
                        """,
                        document_id=str(doc_id),
                        chunk_index=int(chunk_idx),
                    )
                else:
                    enriched.append(doc)
                    continue

                record = result.single()
                triples = record["triples"] if record else []

            if triples:
                graph_ctx = "\n[Graph context]\n" + "".join(
                    f"{t[0]} -[{t[1]}]-> {t[2]}\n" for t in triples
                )
                enriched_doc = LangchainDocument(
                    page_content=doc.page_content + graph_ctx,
                    metadata={**doc.metadata, "_graph_triples": len(triples)},
                )
                graph_hits += 1
            else:
                enriched_doc = doc

        except Exception as exc:
            logger.warning(
                "GraphService.enrich: failed for point_id=%s doc_id=%s chunk=%s: %s",
                point_id, doc_id, chunk_idx, exc,
            )
            enriched_doc = doc

        enriched.append(enriched_doc)

    logger.info(
        "GraphService.enrich: %d/%d docs enriched with entity triples",
        graph_hits, len(docs),
    )
    return enriched


# ── Deletion ───────────────────────────────────────────────────────────────────

def delete_graph_for_document(
    kb_id: Optional[int],
    document_id: int,
    data_store_id: Optional[int] = None,
) -> None:
    """
    Remove all Neo4j Chunk nodes for a deleted document, and clean up
    any Entity nodes that no longer have any Chunk connections.

    NOT gated on GRAPHRAG_ENABLED — data may exist from a prior ingest
    run when the flag was on.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})
            DETACH DELETE c
            RETURN count(c) AS deleted
            """,
            doc_id=str(document_id),
        ).single()
        logger.info(
            "GraphService: deleted %d Chunk nodes for doc %d",
            rec["deleted"] if rec else 0, document_id,
        )

        # Clean up orphaned entity nodes. With per-KB/datastore scoping,
        # entities from other KBs are separate nodes and won't be affected.
        # Only delete entities that have no remaining FROM_CHUNK edges.
        # Scope the cleanup to avoid deleting orphaned entities from other KBs.
        rec = session.run(
            """
            MATCH (e)
            WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
              AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
              AND (
                ($kb_id IS NOT NULL AND e.kb_id = $kb_id)
                OR ($ds_id IS NOT NULL AND e.data_store_id = $ds_id)
                OR (e.kb_id IS NULL AND e.data_store_id IS NULL)
              )
            DETACH DELETE e
            RETURN count(e) AS cleaned
            """,
            kb_id=str(kb_id) if kb_id is not None else None,
            ds_id=str(data_store_id) if data_store_id is not None else None,
        ).single()
        logger.info(
            "GraphService: cleaned %d orphaned entity nodes after doc %d deletion",
            rec["cleaned"] if rec else 0, document_id,
        )


def delete_graph_for_kb(kb_id: int) -> None:
    """
    Remove all Neo4j nodes for an entire deleted knowledge base.

    NOT gated on GRAPHRAG_ENABLED — same reasoning as delete_graph_for_document.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    with driver.session() as session:
        # 1. Delete all inter-entity relationships stamped with this KB's id.
        #    The ReLiK pipeline writes these as MERGE (a)-[:REL {kb_id: ...}]->(b).
        #    Must run before the chunk DETACH DELETE while entity nodes still exist.
        rec = session.run(
            """
            MATCH ()-[r {kb_id: $kb_id}]->()
            DELETE r
            RETURN count(r) AS deleted_rels
            """,
            kb_id=str(kb_id),
        ).single()
        logger.info(
            "GraphService: deleted %d inter-entity relationships for kb_%d",
            rec["deleted_rels"] if rec else 0, kb_id,
        )

        # 2. Delete Chunk nodes for direct uploads only (data_store_id IS NULL).
        #    DataStore document chunks are preserved — they belong to the datastore, not the KB.
        rec = session.run(
            """
            MATCH (c:Chunk {kb_id: $kb_id})
            WHERE c.data_store_id IS NULL
            DETACH DELETE c
            RETURN count(c) AS deleted
            """,
            kb_id=str(kb_id),
        ).single()
        logger.info(
            "GraphService: deleted %d Chunk nodes (direct uploads only) for kb_%d",
            rec["deleted"] if rec else 0, kb_id,
        )

        # 3. Sweep entity nodes scoped to this KB that have no remaining
        #    FROM_CHUNK edges. With per-KB scoping, entities from other KBs
        #    are separate nodes and won't be affected.
        #    Also covers legacy entities (no kb_id property) that were orphaned.
        rec = session.run(
            """
            MATCH (e)
            WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
              AND (e.kb_id = $kb_id OR e.kb_id IS NULL)
              AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
            DETACH DELETE e
            RETURN count(e) AS cleaned
            """,
            kb_id=str(kb_id),
        ).single()
        logger.info(
            "GraphService: cleaned %d orphaned entity nodes after kb_%d deletion",
            rec["cleaned"] if rec else 0, kb_id,
        )


def _batch_delete_chunks(driver: neo4j.Driver, stale_id: str) -> int:
    total = 0
    while True:
        with driver.session() as session:
            rec = session.run(
                """
                MATCH (c:Chunk {kb_id: $kb_id})
                WITH c LIMIT 100
                DETACH DELETE c
                RETURN count(c) AS n
                """,
                kb_id=stale_id,
            ).single()
            n = rec["n"] if rec else 0
            total += n
            if n == 0:
                break
    return total


def _batch_delete_orphaned_entities(driver: neo4j.Driver, stale_id: str) -> int:
    total = 0
    while True:
        with driver.session() as session:
            rec = session.run(
                """
                MATCH (e)
                WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
                  AND (e.kb_id = $kb_id OR e.kb_id IS NULL)
                  AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
                WITH e LIMIT 500
                DETACH DELETE e
                RETURN count(e) AS n
                """,
                kb_id=stale_id,
            ).single()
            n = rec["n"] if rec else 0
            total += n
            if n == 0:
                break
    return total


def purge_stale_graph_data(active_kb_ids: list[int]) -> None:
    """
    Delete any Chunk nodes (and their dependent entities) whose kb_id is not
    in active_kb_ids.  Call this after every KB deletion to sweep historical
    debris left by prior code paths that skipped Neo4j cleanup.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    active_str = [str(i) for i in active_kb_ids]

    with driver.session() as session:
        # Find kb_ids present in Neo4j that are no longer in MySQL.
        # Only consider direct uploads (data_store_id IS NULL) — DataStore docs persist.
        # Wrap in a try-catch to handle case when Chunk label doesn't exist.
        try:
            stale_rec = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.data_store_id IS NULL
                  AND NOT c.kb_id IN $active_ids
                RETURN DISTINCT c.kb_id AS stale_kb_id
                """,
                active_ids=active_str,
            )
            stale_ids = [r["stale_kb_id"] for r in stale_rec if r["stale_kb_id"] is not None]
        except Exception:
            # Chunk label doesn't exist or other error - no stale data to purge
            stale_ids = []

    if not stale_ids:
        return

    logger.info("GraphService: found stale kb_ids in Neo4j not in MySQL: %s", stale_ids)

    for stale_id in stale_ids:
        logger.info("GraphService: purging stale kb_%s in batches", stale_id)

        # Inter-entity rels stamped with this kb_id (usually small, single pass ok)
        with driver.session() as session:
            r1 = session.run(
                "MATCH ()-[r {kb_id: $kb_id}]->() DELETE r RETURN count(r) AS n",
                kb_id=stale_id,
            ).single()
            logger.info("GraphService: purged %d inter-entity rels for stale kb_%s", r1["n"] if r1 else 0, stale_id)

        # Chunk nodes in batches — each chunk can have hundreds of FROM_CHUNK
        # relationships, so deleting all at once blows the transaction memory limit.
        total_chunks = _batch_delete_chunks(driver, stale_id)
        logger.info("GraphService: purged %d chunks for stale kb_%s", total_chunks, stale_id)

        # Entity nodes scoped to this KB now orphaned — batch delete
        total_entities = _batch_delete_orphaned_entities(driver, stale_id)
        logger.info("GraphService: purged %d orphaned entities for stale kb_%s", total_entities, stale_id)
