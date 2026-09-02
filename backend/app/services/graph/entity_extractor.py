"""
Entity-aware retrieval: LLM entity extraction, Neo4j expansion, and entity boost scoring.

Query flow (ENTITY_CENTRIC queries only):
  1. extract_entities_from_query  — LLM extracts named entities from the query text
                                    using the same GRAPHRAG_LLM model used for ingestion.
                                    Falls back to empty list on error (non-fatal).
  2. expand_query_entities        — each query entity is matched in Neo4j and 1-hop
                                    neighbors are collected. The matching is fuzzy
                                    (case-insensitive CONTAINS) so "Apple" matches
                                    "Apple Inc.", "Apple Computer", etc.
  3. apply_entity_boost           — chunks whose text mentions query entities or their
                                    neighbors receive a score boost, and entity match
                                    metadata is attached for UI rendering.

Integration:
  retrieval.py calls extract_expand_boost(query, docs, kb_ids) after RRF fusion
  for ENTITY_CENTRIC queries.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document as LangchainDocument
from openai import OpenAI as SyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """Named entity extracted from a query."""
    name: str
    type: str  # PERSON, ORG, GPE, PRODUCT, EVENT, etc.
    confidence: float = 1.0


@dataclass
class EntityNeighbor:
    """Entity neighbor found via Neo4j 1-hop expansion."""
    name: str
    type: str
    relation: str
    score: float = 1.0


# ── LLM client (lazy singleton) ───────────────────────────────────────────────

_llm_client: Optional[SyncOpenAI] = None

_ENTITY_EXTRACT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entity_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":  {"type": "string"},
                            "type":  {"type": "string"},
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

_ENTITY_EXTRACT_SYSTEM = (
    "You are a named entity extractor. "
    "Extract named entities from the user query. "
    "Return only entities that are proper nouns (people, organizations, places, products, events). "
    "Do NOT return common nouns, verbs, or adjectives. "
    "For each entity return its name exactly as it appears in the query and its type. "
    "Valid types: PERSON, ORG, GPE, PRODUCT, EVENT, WORK_OF_ART, LAW, FAC, NORP, LOC. "
    "Return an empty entities list if no named entities are found."
)


def _get_llm_client(db: Any = None, org_id: Any = None) -> SyncOpenAI:
    """Get or create the LLM client for entity extraction.

    When db and org_id are provided, resolves per-org graph-role config.
    Falls back to the global singleton (app-level settings) when not.
    """
    global _llm_client
    if db is not None:
        from app.services.agentic_rag.llm_factory import get_org_llm
        cfg = get_org_llm(org_id, db, role="graph")
        return SyncOpenAI(base_url=cfg["api_base"], api_key=cfg["api_key"])
    # Fallback: global singleton using app-level settings
    if _llm_client is None:
        from app.services.settings_service import get_setting
        from app.db.session import SessionLocal
        _db = SessionLocal()
        try:
            api_key = get_setting(_db, "GRAPHRAG_API_KEY", None) or get_setting(_db, "OPENAI_API_KEY", None)
            api_base = get_setting(_db, "GRAPHRAG_API_BASE", None) or get_setting(_db, "OPENAI_API_BASE", None)
        finally:
            _db.close()
        if not api_key:
            api_key = "not-required"
        _llm_client = SyncOpenAI(base_url=api_base, api_key=api_key)
    return _llm_client


def _extraction_model(db: Any = None, org_id: Any = None) -> str:
    """Model to use for entity extraction. Prefers GRAPHRAG_LLM, falls back to OPENAI_MODEL."""
    if db is not None:
        from app.services.agentic_rag.llm_factory import get_org_llm
        cfg = get_org_llm(org_id, db, role="graph")
        return cfg["model_name"]
    from app.services.settings_service import get_setting
    from app.db.session import SessionLocal
    _db = SessionLocal()
    try:
        return get_setting(_db, "GRAPHRAG_LLM", None) or get_setting(_db, "OPENAI_MODEL", None)
    finally:
        _db.close()


# ── T01: LLM Query Entity Extraction ─────────────────────────────────────────

def extract_entities_from_query(query: str, db: Any = None, org_id: Any = None) -> List[Entity]:
    """
    Extract named entities from a query using the GRAPHRAG_LLM model.

    Uses the same OpenAI-compatible endpoint used during graph ingestion.
    Returns an empty list on error (non-fatal — extraction failure degrades
    gracefully to plain RRF results).

    Args:
        query: User query text.
        db: Optional database session for per-org resolution.
        org_id: Optional organisation ID for per-org resolution.

    Returns:
        Deduplicated list of Entity objects ordered by appearance.
    """
    if not query.strip():
        return []

    model = _extraction_model(db, org_id)
    client = _get_llm_client(db, org_id)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ENTITY_EXTRACT_SYSTEM},
                {"role": "user",   "content": query},
            ],
            response_format=_ENTITY_EXTRACT_SCHEMA,
            temperature=0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        raw_entities = data.get("entities", [])
    except Exception as exc:
        logger.warning("[ENTITY] extraction failed (non-fatal): model=%s err=%s", model, exc)
        return []

    seen: Set[str] = set()
    entities: List[Entity] = []
    for item in raw_entities:
        name = (item.get("name") or "").strip()
        etype = (item.get("type") or "UNKNOWN").upper()
        if not name or len(name) < 2:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(Entity(name=name, type=etype))

    if entities:
        logger.debug(
            "[ENTITY] extracted %d entities | query=%.80s | entities=%s",
            len(entities),
            query,
            ", ".join(f"{e.name}({e.type})" for e in entities),
        )
    return entities


# ── T02: Neo4j Entity Expansion ──────────────────────────────────────────────

def _get_neo4j_driver():
    """Reuse the Neo4j driver singleton from the graph setup module."""
    from app.services.graph.setup import _get_driver
    return _get_driver()


def expand_query_entities(
    entities: List[Entity],
    kb_ids: List[int],
) -> List[EntityNeighbor]:
    """
    Expand query entities via Neo4j 1-hop traversal.

    Matches each query entity against __Entity__ nodes in Neo4j using
    case-insensitive name matching (entities written during ingestion may
    differ in capitalisation from the query). For each matched entity,
    collects 1-hop neighbors connected by any relationship type.

    Args:
        entities: Entities extracted from the query.
        kb_ids:   Knowledge base IDs to scope the search.

    Returns:
        Deduplicated list of EntityNeighbor objects.
    """
    from app.core.settings_registry import get_def
    if not get_def("GRAPHRAG_ENABLED").default or not entities:
        return []

    driver = _get_neo4j_driver()
    neighbors: List[EntityNeighbor] = []
    seen: Set[str] = set()

    try:
        with driver.session() as session:
            for entity in entities:
                # Match against __Entity__ nodes (written by neo4j-graphrag LLM pipeline)
                # Use case-insensitive CONTAINS so "Apple" matches "Apple Inc." etc.
                result = session.run(
                    """
                    MATCH (e:__Entity__)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                    MATCH (e)-[r]-(n:__Entity__)
                    RETURN DISTINCT n.name AS name,
                                    coalesce(n.type, 'Entity') AS type,
                                    type(r) AS rel
                    LIMIT 30
                    """,
                    name=entity.name,
                )
                for record in result:
                    neighbor_name = (record.get("name") or "").strip()
                    if not neighbor_name or neighbor_name.lower() in seen:
                        continue
                    # Skip if it's just the original entity itself
                    if neighbor_name.lower() == entity.name.lower():
                        continue
                    seen.add(neighbor_name.lower())
                    neighbors.append(EntityNeighbor(
                        name=neighbor_name,
                        type=record.get("type") or "Entity",
                        relation=record.get("rel") or "RELATED_TO",
                        score=1.0,
                    ))

    except Exception as exc:
        logger.warning("[ENTITY] Neo4j expansion failed (non-fatal): %s", exc)

    if neighbors:
        logger.debug(
            "[ENTITY] expanded %d query entities → %d Neo4j neighbors",
            len(entities), len(neighbors),
        )
    return neighbors


# ── T03: Entity Boost Scoring ─────────────────────────────────────────────────

def apply_entity_boost(
    docs: List[LangchainDocument],
    entities: List[Entity],
    neighbors: Optional[List[EntityNeighbor]] = None,
    db: Any = None,
    org_id: Any = None,
) -> List[LangchainDocument]:
    """
    Apply entity mention boost to chunk scores.

    For each document, counts how many times query entities (and their
    1-hop neighbors) appear in the chunk text. Each mention adds
    ENTITY_BOOST_FACTOR to the score multiplier.

    Entity match metadata is stored in doc.metadata["entity_matches"] for
    UI rendering (highlighting, faceting).

    Args:
        docs:      Retrieved documents from RRF fusion.
        entities:  Entities extracted from the query.
        neighbors: Optional expanded entity neighbors from Neo4j.

    Returns:
        Documents with score boost applied and entity_matches metadata set.
    """
    if not entities:
        return docs

    from app.services.settings_service import get_setting
    from app.core.settings_registry import get_def
    boost_factor = get_setting(db, "ENTITY_BOOST_FACTOR", org_id) if db is not None else get_def("ENTITY_BOOST_FACTOR").default

    # Build (pattern, entity_name) pairs — query entities get full weight,
    # expanded neighbors get half weight (they're one hop away).
    patterns: List[Tuple[re.Pattern, str, float]] = []
    for e in entities:
        pat = re.compile(rf"\b{re.escape(e.name)}\b", re.IGNORECASE)
        patterns.append((pat, e.name, 1.0))
    if neighbors:
        for n in neighbors:
            pat = re.compile(rf"\b{re.escape(n.name)}\b", re.IGNORECASE)
            patterns.append((pat, n.name, 0.5))

    boosted_count = 0
    for doc in docs:
        text = doc.page_content or ""
        matches: Dict[str, int] = {}

        for pattern, entity_name, weight in patterns:
            hits = len(pattern.findall(text))
            if hits:
                matches[entity_name] = matches.get(entity_name, 0) + int(hits * weight)

        if matches:
            total_boost = sum(matches.values()) * boost_factor
            current_score = float(doc.metadata.get("score", 0.0))
            doc.metadata["score"] = current_score * (1.0 + total_boost)
            doc.metadata["entity_matches"] = [
                {"entity": k, "count": v} for k, v in matches.items()
            ]
            boosted_count += 1

    logger.debug(
        "[ENTITY] boost applied | boosted=%d/%d docs | factor=%.2f",
        boosted_count, len(docs), boost_factor,
    )
    return docs


# ── Convenience wrapper ───────────────────────────────────────────────────────

def extract_expand_boost(
    query: str,
    docs: List[LangchainDocument],
    kb_ids: List[int],
    db: Any = None,
    org_id: Any = None,
) -> List[LangchainDocument]:
    """
    Full entity-aware retrieval pipeline: extract → expand → boost.

    Called from retrieval.py after RRF fusion for ENTITY_CENTRIC queries.
    All failures are swallowed — the function always returns the original
    (or boosted) doc list without raising.

    Args:
        query:  User query text.
        docs:   Retrieved documents from RRF fusion.
        kb_ids: Knowledge base IDs.
        db:     Optional database session for per-org LLM resolution.
        org_id: Optional organisation ID for per-org LLM resolution.

    Returns:
        Documents with entity boost applied (or originals if extraction fails).
    """
    entities = extract_entities_from_query(query, db=db, org_id=org_id)
    if not entities:
        return docs

    neighbors = expand_query_entities(entities, kb_ids)
    return apply_entity_boost(docs, entities, neighbors, db=db, org_id=org_id)
