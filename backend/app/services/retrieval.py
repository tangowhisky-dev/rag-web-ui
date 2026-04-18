"""
Hybrid retrieval: MySQL InnoDB FULLTEXT (sparse) + ChromaDB cosine similarity (dense),
fused with Reciprocal Rank Fusion (RRF).

Sparse leg: MySQL MATCH ... AGAINST (NATURAL LANGUAGE MODE) — uses the server's
built-in BM25/TF-IDF scoring on the `chunk_text` LONGTEXT column.  No in-memory
index rebuild; results come directly from the persistent FULLTEXT index.

Four tunable parameters (all in .env / settings):
  HYBRID_DENSE_WEIGHT       — weight applied to the dense ranking leg
  HYBRID_SPARSE_WEIGHT      — weight applied to the sparse (FTS) ranking leg
  RETRIEVAL_TOP_K           — number of documents to return
  HYBRID_MAX_DENSE_DISTANCE — cosine distance cutoff for the dense leg
                              (Chroma: 0 = identical, 2 = opposite)

Absent leg vs. no signal — design rationale
--------------------------------------------
MySQL FTS returns no row when the relevance score would be 0 (no query-term
overlap), so those documents are simply absent from the sparse leg.  They
can still be returned via the dense leg — that case (paraphrase / synonym
match with no exact keyword overlap) is exactly what dense retrieval is for.

HYBRID_MAX_DENSE_DISTANCE is the symmetric control for the dense leg: a
document with both FTS score = 0 *and* high cosine distance carries no
signal from either leg and should be excluded.  Set HYBRID_MAX_DENSE_DISTANCE
< 2.0 (e.g. 1.2) to enforce this.  Default 2.0 keeps all dense results.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import List, Dict

from langchain_core.documents import Document as LangchainDocument
from sqlalchemy import text, bindparam
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.knowledge import DocumentChunk

logger = logging.getLogger(__name__)

# RRF smoothing constant — standard value from the original paper.
_RRF_K = 60


@dataclass
class _Candidate:
    doc: LangchainDocument
    content_hash: str
    dense_rank: int = -1   # -1 = absent from this leg
    sparse_rank: int = -1

    @property
    def rrf_score(self) -> float:
        score = 0.0
        if self.dense_rank >= 0:
            score += settings.HYBRID_DENSE_WEIGHT / (_RRF_K + self.dense_rank)
        if self.sparse_rank >= 0:
            score += settings.HYBRID_SPARSE_WEIGHT / (_RRF_K + self.sparse_rank)
        return score


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dense_search(query: str, vector_stores: list, candidates: int) -> Dict[str, _Candidate]:
    all_pairs: list[tuple] = []
    for vs in vector_stores:
        all_pairs.extend(vs.similarity_search_with_score(query, k=candidates))

    all_pairs.sort(key=lambda pair: pair[1])  # ascending distance = best first

    max_distance = settings.HYBRID_MAX_DENSE_DISTANCE
    result: Dict[str, _Candidate] = {}
    rank = 0
    for doc, distance in all_pairs:
        if distance > max_distance:
            break  # list is sorted — all remaining are farther away
        h = _content_hash(doc.page_content)
        if h not in result:
            result[h] = _Candidate(doc=doc, content_hash=h, dense_rank=rank)
            rank += 1
    return result


def _sparse_search(query: str, kb_ids: List[int], db: Session, candidates: int) -> Dict[str, _Candidate]:
    """MySQL InnoDB FULLTEXT search (BM25/TF-IDF scoring, server-side).

    Returns only rows with a non-zero relevance score, ordered best-first.
    Avoids loading all chunks into memory — the DB does the ranking.
    """
    if not query.strip():
        return {}

    sql = text(
        """
        SELECT chunk_text, chunk_metadata,
               MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) AS fts_score
        FROM   document_chunks
        WHERE  kb_id IN :kb_ids
          AND  MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0
        ORDER  BY fts_score DESC
        LIMIT  :candidates
        """
    ).bindparams(bindparam("kb_ids", expanding=True))

    rows = db.execute(sql, {"query": query, "kb_ids": kb_ids, "candidates": candidates}).fetchall()

    result: Dict[str, _Candidate] = {}
    for rank, row in enumerate(rows):
        chunk_text = row.chunk_text or ""
        h = _content_hash(chunk_text)
        if h not in result:
            doc = LangchainDocument(
                page_content=chunk_text,
                metadata=dict(row.chunk_metadata or {}),
            )
            result[h] = _Candidate(doc=doc, content_hash=h, sparse_rank=rank)
    return result


def _rrf_merge(dense: Dict[str, _Candidate], sparse: Dict[str, _Candidate], top_k: int) -> List[LangchainDocument]:
    merged: Dict[str, _Candidate] = {**dense}
    for h, c in sparse.items():
        if h in merged:
            merged[h].sparse_rank = c.sparse_rank
        else:
            merged[h] = c

    ranked = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)
    return [c.doc for c in ranked[:top_k]]


async def hybrid_search(
    query: str,
    kb_ids: List[int],
    db: Session,
    vector_stores: list,
) -> List[LangchainDocument]:
    top_k = settings.RETRIEVAL_TOP_K
    pool = top_k * 4  # fetch more candidates per leg so RRF has room to rerank

    logger.info("hybrid_search | kb_ids=%s top_k=%d dense_w=%.2f sparse_w=%.2f",
                kb_ids, top_k, settings.HYBRID_DENSE_WEIGHT, settings.HYBRID_SPARSE_WEIGHT)

    dense = _dense_search(query, vector_stores, pool)
    sparse = _sparse_search(query, kb_ids, db, pool)
    docs = _rrf_merge(dense, sparse, top_k)

    logger.info("hybrid_search returned %d documents", len(docs))
    return docs
