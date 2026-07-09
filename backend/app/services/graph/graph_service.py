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
import uuid
from typing import Optional


import neo4j
from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (lazy) ────────────────────────────────────────────
_neo4j_driver: Optional[neo4j.Driver] = None
_llm_pipeline = None   # neo4j_graphrag Pipeline — only built when GRAPHRAG_LLM is set

def _get_driver() -> neo4j.Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = neo4j.GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _neo4j_driver


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """
    Convert a SHA-256 hex chunk ID to the deterministic UUID Qdrant uses.

    Mirrors document_processor._chunk_id_to_point_id so the UUID written
    to Neo4j is always the exact UUID stored as the Qdrant point ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


# ── LLM pipeline path ─────────────────────────────────────────────────────────

def _get_llm_pipeline():
    """
    Build (once) a neo4j-graphrag Pipeline with LLMEntityRelationExtractor.

    Pipeline topology:
      extractor (LLMEntityRelationExtractor)
          └─[graph: Neo4jGraph]──► writer (Neo4jWriter)

    use_structured_output=True makes the LLM return a JSON Schema-validated
    Neo4jGraph object — no regex post-processing, no json.loads guesswork.
    The LLM must support response_format=json_schema (OpenAI GPT-4 class and
    most modern OpenAI-compatible endpoints do).
    """
    global _llm_pipeline
    if _llm_pipeline is not None:
        return _llm_pipeline

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
    from neo4j_graphrag.experimental.pipeline.pipeline import Pipeline

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

    llm = OpenAILLM(
        model_name=settings.GRAPHRAG_LLM,
        model_params={
            "temperature": 0,
            # "max_tokens": 1024,  # entity/relation JSON is small; cap prevents full context reservation
        },
        base_url=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY,
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        use_structured_output=True,
        on_error=OnError.IGNORE,
        create_lexical_graph=False,
        max_concurrency=1,   # 1 = fully sequential; local models OOM at >1
    )

    writer = _SafeNeo4jWriter(
        driver=_get_driver(),
        neo4j_database="neo4j",
        batch_size=500,
    )

    pipe = Pipeline()
    pipe.add_component(extractor, "extractor")
    pipe.add_component(writer, "writer")
    pipe.connect("extractor", "writer", input_config={"graph": "extractor"})

    _llm_pipeline = pipe
    logger.info("GraphService[llm]: pipeline built with model=%s", settings.GRAPHRAG_LLM)
    return _llm_pipeline


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


async def _extract_with_llm(
    document_id: int,
    file_name: str,
    chunks: list[str],
    qdrant_point_ids: list[str],
    pt=None,            # optional ProgressTimeout for periodic pings
) -> tuple[int, int]:
    """Run neo4j-graphrag LLM pipeline on context-sized batches of chunks.

    Consecutive chunks are merged into batches sized to NEO4J_LLM_CONTEXT
    (with overlap stripped) before being sent to the LLM. This gives the
    extractor broader context than single-chunk calls, reducing duplicate
    entities at chunk boundaries and surfacing intra-document relationships.

    After each pipe.run(), FROM_CHUNK edges are written to ALL Qdrant chunk
    nodes in the batch — so entity→chunk links remain granular even though
    extraction ran on the merged text.

    Up to 4 batches run concurrently (local _chunk_sem). All synchronous Neo4j
    I/O is dispatched via run_in_executor.
    """
    from neo4j_graphrag.experimental.components.types import TextChunks, TextChunk

    loop = asyncio.get_event_loop()
    pipe = _get_llm_pipeline()
    driver = _get_driver()

    # Cap chunks to avoid OOM on low-RAM local models. Qdrant still has all chunks.
    cap = settings.GRAPHRAG_MAX_CHUNKS
    effective_chunks = chunks if cap <= 0 else chunks[:cap]
    effective_ids = qdrant_point_ids if cap <= 0 else qdrant_point_ids[:cap]
    if cap > 0 and len(chunks) > cap:
        logger.info(
            "GraphService[llm]: doc %d — capping graph extraction at %d/%d chunks (GRAPHRAG_MAX_CHUNKS=%d)",
            document_id, cap, len(chunks), cap,
        )

    batches = _build_extraction_batches(
        effective_chunks, effective_ids, settings.NEO4J_LLM_CONTEXT
    )
    logger.info(
        "GraphService[llm]: doc %d — %d chunks → %d extraction batches (NEO4J_LLM_CONTEXT=%d)",
        document_id, len(effective_chunks), len(batches), settings.NEO4J_LLM_CONTEXT,
    )

    _batch_sem = asyncio.Semaphore(4)

    async def _process_batch(
        batch_idx: int, combined_text: str, batch_point_ids: list[str]
    ) -> tuple[int, int]:
        async with _batch_sem:
            last_exc = None
            for attempt in range(1, 4):  # up to 3 attempts
                try:
                    pipe_result = await pipe.run({
                        "extractor": {
                            "chunks": TextChunks(chunks=[TextChunk(text=combined_text, index=batch_idx)]),
                            "examples": "",
                        },
                        "writer": {},
                    })
                    writer_output = None
                    raw = getattr(pipe_result, "result", None)
                    if isinstance(raw, dict):
                        writer_output = raw.get("writer")
                    elif hasattr(raw, "status"):
                        writer_output = raw

                    status = getattr(writer_output, "status", None)
                    if status == "FAILURE":
                        logger.warning(
                            "GraphService[llm]: writer FAILURE for doc %d batch %d — skipping FROM_CHUNK links",
                            document_id, batch_idx,
                        )
                        return 0, 0

                    # Fan out FROM_CHUNK edges to every chunk in the batch.
                    def _link_batch_chunks(pids=batch_point_ids):
                        linked_total = 0
                        with driver.session() as session:
                            for pid in pids:
                                rec = session.run(
                                    """
                                    MATCH (e:__Entity__)
                                    WHERE NOT EXISTS {
                                        MATCH (e)-[:FROM_CHUNK]->(:Chunk {qdrant_point_id: $point_id})
                                    }
                                    WITH e LIMIT 500
                                    MATCH (c:Chunk {qdrant_point_id: $point_id})
                                    MERGE (e)-[:FROM_CHUNK]->(c)
                                    RETURN count(e) AS linked
                                    """,
                                    point_id=pid,
                                ).single()
                                linked_total += rec["linked"] if rec else 0
                        return linked_total

                    linked = await loop.run_in_executor(None, _link_batch_chunks)
                    if pt:
                        pt.ping()  # signal progress after LLM extraction batch
                    return linked, 0

                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "GraphService[llm]: pipeline failed for doc %d batch %d (attempt %d/3): %s",
                        document_id, batch_idx, attempt, exc,
                    )
                    if attempt < 3:
                        await asyncio.sleep(1)

            logger.error(
                "GraphService[llm]: all 3 attempts failed for doc %d batch %d — giving up: %s",
                document_id, batch_idx, last_exc,
            )
            return 0, 0

    results = await asyncio.gather(*[
        _process_batch(idx, text, pids)
        for idx, (text, pids) in enumerate(batches)
    ])

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
    return total_nodes, total_rels


# ── Ingest ─────────────────────────────────────────────────────────────────────

async def build_graph_for_document(
    kb_id: Optional[int],
    document_id: int,
    file_name: str,
    chunks: list[str],
    chunk_ids: list[str],
    data_store_id: Optional[int] = None,
    pt=None,            # optional ProgressTimeout for periodic pings
) -> None:
    """
    Extract entity/relationship graph from document chunks and store in Neo4j.

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
    if not settings.GRAPHRAG_ENABLED:
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

    # Extract entities and relationships — throttled by semaphore so we don't
    # run all 20 documents' extraction simultaneously on one local LLM.
    # Lazily create semaphore inside the function so it is bound to the
    # *current* event loop (the module-level one is created at import time
    # and may be on a different loop from the running task).
    sem = asyncio.Semaphore(1)
    async with sem:
        total_entities, total_relations = await _extract_with_llm(
            document_id=document_id,
            file_name=file_name,
            chunks=chunks,
            qdrant_point_ids=qdrant_point_ids,
            pt=pt,
        )

    logger.info(
        "GraphService[llm]: doc %d — %d entities, %d relations written to Neo4j",
        document_id, total_entities, total_relations,
    )


# ── Retrieval: graph expansion ─────────────────────────────────────────────────

def expand_docs_via_graph(
    docs: list[LangchainDocument],
    kb_ids: list[int],
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

    Non-fatal — returns [] on any failure so the caller's pipeline continues
    with only the original vector search results.
    """
    if not settings.GRAPHRAG_ENABLED or not docs:
        return []

    from qdrant_client import QdrantClient

    # Extract the Qdrant point UUIDs from the retrieved docs
    seen_point_ids = set()
    for doc in docs:
        pid = doc.metadata.get("qdrant_point_id")
        if pid:
            seen_point_ids.add(pid)

    if not seen_point_ids:
        # Docs came from before the qdrant_point_id payload field was added —
        # fall back gracefully rather than blowing up.
        logger.debug("GraphService.expand: no qdrant_point_id in doc metadata, skipping expansion")
        return []

    try:
        driver = _get_driver()
        collections = [f"kb_{kb_id}" for kb_id in kb_ids]

        # Traverse the graph: found_chunk → entity → entity → connected_chunk
        # Return qdrant_point_id + qdrant_collection of chunks NOT already seen.
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.qdrant_point_id IN $seen_ids
                  AND c.qdrant_collection IN $collections
                MATCH (c)<-[:FROM_CHUNK]-(e)-[r]-(e2)-[:FROM_CHUNK]->(c2:Chunk)
                WHERE c2.qdrant_point_id IS NOT NULL
                  AND NOT c2.qdrant_point_id IN $seen_ids
                  AND c2.qdrant_collection IN $collections
                RETURN DISTINCT c2.qdrant_point_id AS point_id,
                                c2.qdrant_collection AS collection
                LIMIT 40
                """,
                seen_ids=list(seen_point_ids),
                collections=collections,
            )
            expansion_targets = [(r["point_id"], r["collection"]) for r in result]

        if not expansion_targets:
            logger.debug("GraphService.expand: no graph-connected chunks found beyond current result set")
            return []

        logger.info(
            "GraphService.expand: found %d graph-connected chunks to fetch from Qdrant",
            len(expansion_targets),
        )

        # Group by collection and fetch from Qdrant (text only, no re-embedding)
        from collections import defaultdict
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

        logger.info(
            "GraphService.expand: fetched %d graph-expanded docs from Qdrant",
            len(expanded_docs),
        )
        return expanded_docs

    except Exception as exc:
        logger.warning("GraphService.expand: expansion failed (non-fatal): %s", exc)
        return []


# ── Retrieval: entity context enrichment ──────────────────────────────────────

def enrich_docs_with_graph(docs: list[LangchainDocument]) -> list[LangchainDocument]:
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
    if not settings.GRAPHRAG_ENABLED or not docs:
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

def delete_graph_for_document(kb_id: Optional[int], document_id: int) -> None:
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

        rec = session.run(
            """
            MATCH (e)
            WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
              AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
            DETACH DELETE e
            RETURN count(e) AS cleaned
            """
        ).single()
        logger.info(
            "GraphService: cleaned %d orphaned entity nodes after doc %d deletion",
            rec["cleaned"] if rec else 0, document_id,
        )

        # Defensive: sweep any Chunk nodes for this document that somehow
        # survived (e.g. from a prior run that was interrupted mid-transaction).
        rec = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})
            DETACH DELETE c
            RETURN count(c) AS cleaned
            """,
            doc_id=str(document_id),
        ).single()
        if rec and rec["cleaned"]:
            logger.info(
                "GraphService: cleaned %d residual Chunk nodes for doc %d",
                rec["cleaned"], document_id,
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
            MATCH (n {kb_id: $kb_id})
            WHERE n.data_store_id IS NULL
            DETACH DELETE n
            RETURN count(n) AS deleted
            """,
            kb_id=str(kb_id),
        ).single()
        logger.info(
            "GraphService: deleted %d Chunk nodes (direct uploads only) for kb_%d",
            rec["deleted"] if rec else 0, kb_id,
        )

        # 3. Sweep entity nodes that have no remaining FROM_CHUNK edges.
        #    Covers: (a) ReLiK Entity nodes (b) LLM-pipeline __Entity__ nodes
        #            (c) neo4j-graphrag __KGBuilder__ bookkeeping nodes.
        #    The FROM_CHUNK direction is always entity→chunk, so checking the
        #    outgoing side is sufficient.
        rec = session.run(
            """
            MATCH (e)
            WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
              AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
            DETACH DELETE e
            RETURN count(e) AS cleaned
            """
        ).single()
        logger.info(
            "GraphService: cleaned %d orphaned entity nodes after kb_%d deletion",
            rec["cleaned"] if rec else 0, kb_id,
        )

        # 4. Defensive: sweep any Chunk nodes that still carry this kb_id.
        #    These would only survive if a prior deletion was blocked (e.g. by the
        #    old GRAPHRAG_ENABLED gate) or interrupted partway through.
        #    Running this last means entities above were already swept, so
        #    DETACH DELETE here only removes the chunk nodes themselves.
        rec = session.run(
            """
            MATCH (c:Chunk {kb_id: $kb_id})
            DETACH DELETE c
            RETURN count(c) AS cleaned
            """,
            kb_id=str(kb_id),
        ).single()
        if rec and rec["cleaned"]:
            logger.info(
                "GraphService: cleaned %d residual Chunk nodes for kb_%d",
                rec["cleaned"], kb_id,
            )
            # Entity nodes whose only FROM_CHUNK edges pointed at those
            # now-deleted chunks become newly orphaned — sweep again.
            rec2 = session.run(
                """
                MATCH (e)
                WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
                  AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
                DETACH DELETE e
                RETURN count(e) AS cleaned
                """
            ).single()
            if rec2 and rec2["cleaned"]:
                logger.info(
                    "GraphService: cleaned %d newly-orphaned entity nodes after residual chunk sweep for kb_%d",
                    rec2["cleaned"], kb_id,
                )


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
        total_chunks = 0
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
                total_chunks += n
                if n == 0:
                    break
        logger.info("GraphService: purged %d chunks for stale kb_%s", total_chunks, stale_id)

        # Entity nodes now orphaned — also batch in case there are many
        total_entities = 0
        while True:
            with driver.session() as session:
                rec = session.run(
                    """
                    MATCH (e)
                    WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
                      AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
                    WITH e LIMIT 500
                    DETACH DELETE e
                    RETURN count(e) AS n
                    """
                ).single()
                n = rec["n"] if rec else 0
                total_entities += n
                if n == 0:
                    break
        logger.info("GraphService: purged %d orphaned entities for stale kb_%s", total_entities, stale_id)
