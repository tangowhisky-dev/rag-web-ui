"""
Debug script: pulls real chunks from MySQL for a given doc, then calls
_extract_with_llm exactly as the app does — same pipeline, same settings.

Run inside the backend container:
    docker compose -f docker-compose.dev.yml exec backend python3 /app/debug_pipeline.py
"""

import asyncio
import sys
import os

sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")

DOC_ID     = 26
MAX_CHUNKS = 20   # set to 0 to run all chunks (will take a long time)

async def main():
    # Import app modules so we use the exact same settings, pipeline, driver
    from app.core.config import settings
    from app.services.graph import (
        _extract_with_llm,
        _build_extraction_batches,
        _strip_overlap,
    )
    from neo4j_graphrag.generation.prompts import ERExtractionTemplate

    print(f"GRAPHRAG_LLM      : {settings.GRAPHRAG_LLM}")
    print(f"OPENAI_API_BASE   : {settings.OPENAI_API_BASE}")
    print(f"NEO4J_LLM_CONTEXT : {settings.NEO4J_LLM_CONTEXT}")
    print(f"GRAPHRAG_MAX_CHUNKS: {settings.GRAPHRAG_MAX_CHUNKS}")

    # Pull chunks from MySQL via SQLAlchemy (same DB session as app)
    from app.db.session import SessionLocal
    from app.models.knowledge import DocumentChunk

    db = SessionLocal()
    try:
        rows = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == DOC_ID)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
    finally:
        db.close()

    print(f"\nFetched {len(rows)} chunks for doc {DOC_ID}")

    chunks    = [r.chunk_text for r in rows]
    point_ids = [str(r.id) for r in rows]

    # Show batch sizes exactly as the app will see them
    cap = settings.GRAPHRAG_MAX_CHUNKS
    effective_chunks = chunks if cap <= 0 else chunks[:cap]
    effective_ids    = point_ids if cap <= 0 else point_ids[:cap]

    batches = _build_extraction_batches(effective_chunks, effective_ids, settings.NEO4J_LLM_CONTEXT)
    budget  = int(settings.NEO4J_LLM_CONTEXT * 0.33)
    tpl     = ERExtractionTemplate()

    print(f"Batch text budget : {budget} chars  ({budget//3} tokens @3c)")
    print(f"Total batches     : {len(batches)}")
    print()
    print("Batch sizes (first 10 + range 26-33):")
    indices = list(range(min(10, len(batches)))) + list(range(26, min(34, len(batches))))
    for i in sorted(set(indices)):
        text, pids = batches[i]
        prompt = tpl.format(text=text, schema={}, examples="")
        print(f"  [{i:3d}] text={len(text):6d} chars  prompt={len(prompt):6d} chars  ~{len(prompt)//3:4d} tok@3c  chunks={len(pids)}")

    # Run extraction using exact app code, capped at MAX_CHUNKS
    run_chunks    = chunks[:MAX_CHUNKS]   if MAX_CHUNKS > 0 else chunks
    run_point_ids = point_ids[:MAX_CHUNKS] if MAX_CHUNKS > 0 else point_ids
    print(f"\nRunning _extract_with_llm for doc {DOC_ID} with {len(run_chunks)} chunks ...")

    # Temporarily override GRAPHRAG_MAX_CHUNKS so _extract_with_llm respects our cap
    original_cap = settings.GRAPHRAG_MAX_CHUNKS
    settings.GRAPHRAG_MAX_CHUNKS = MAX_CHUNKS

    try:
        entities, relations = await _extract_with_llm(
            document_id=DOC_ID,
            file_name=f"debug_doc_{DOC_ID}",
            chunks=run_chunks,
            qdrant_point_ids=run_point_ids,
        )
        print(f"\nResult: {entities} entities, {relations} relations written to Neo4j")
    except Exception as e:
        print(f"\nFAILED: {e}")
    finally:
        settings.GRAPHRAG_MAX_CHUNKS = original_cap


if __name__ == "__main__":
    asyncio.run(main())
