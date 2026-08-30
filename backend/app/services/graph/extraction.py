"""LLM entity-relation extraction pipeline.

Builds and runs the neo4j-graphrag LLMEntityRelationExtractor on
context-sized batches of document chunks.  The extractor and Neo4j
writer are called separately (not via Pipeline) so the extracted graph
can be intercepted to link only the batch's entities to the batch's
chunks via FROM_CHUNK edges.

Key components:
  _get_extractor_and_writer()  — lazy singleton builder (LLM + writer)
  _strip_overlap()             — deduplicates chunk overlap text
  _build_extraction_batches()  — groups chunks into context-sized batches
  _acquire_global_llm_sem()    — non-blocking semaphore acquire
  _extract_with_llm()          — runs extraction across all batches

Shared state (_global_extractor, _global_writer, _global_llm_sem,
_graph_batch_progress, _graph_progress_lock) lives in .setup and is
accessed via module attribute to ensure singleton semantics across
all graph sub-modules.
"""

import asyncio
import logging
from typing import Optional

from app.core.config import settings
from app.services.settings_service import get_setting

from .setup import (
    _get_driver,
    _global_llm_sem,
    _graph_batch_progress,
    _graph_progress_lock,
)
from . import setup as _setup

logger = logging.getLogger(__name__)


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
    if _setup._global_extractor is not None and _setup._global_writer is not None:
        return _setup._global_extractor, _setup._global_writer

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

    _setup._global_extractor = LLMEntityRelationExtractor(
        llm=llm,
        use_structured_output=True,
        on_error=OnError.IGNORE,
        create_lexical_graph=False,
        max_concurrency=1,   # 1 = fully sequential; local models OOM at >1
    )

    _setup._global_writer = _SafeNeo4jWriter(
        driver=_get_driver(),
        neo4j_database="neo4j",
        batch_size=500,
    )

    logger.info("GraphService[llm]: extractor+writer built with model=%s", graph_model)
    return _setup._global_extractor, _setup._global_writer


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
