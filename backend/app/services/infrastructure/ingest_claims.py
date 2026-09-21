"""Redis-backed claim registry for ingestion task submissions.

Tracks which ``ProcessingTask`` IDs currently have a live executor future —
submitted but not yet resolved (completed, failed, or cancelled).  The DB
``status`` column alone cannot distinguish "pending with a queued future"
from "pending and orphaned" (the future was lost to a scan cancel or a
process restart), which previously forced timestamp heuristics and still
allowed the same document to be submitted twice.

Semantics:

- ``claim_ingestion(task_id)`` — ``SET NX EX`` at submit time.  Returns
  ``False`` when a claim already exists → the caller must NOT submit.
  This is the atomic dedup gate.
- ``release_ingestion_claim(task_id)`` — ``DEL`` when the future
  resolves.  Called from the worker's ``finally`` AND from submit-site
  done-callbacks (a future cancelled before it starts never enters the
  worker).
- ``touch_ingestion_claim(task_id)`` — ``EXPIRE`` refresh, called on every
  task progress write so a live task's claim never expires.  A dead
  task's claim self-frees after ``_CLAIM_TTL_S``.
- ``get_claimed_task_ids(task_ids)`` — pipelined ``EXISTS`` for the
  requeue paths (scan, watcher tick, recovery).
- ``clear_ingestion_claims()`` — wipe all claims once at startup; every
  future from the previous process died with it, so all claims are stale.

Fail-open: when Redis is unavailable, claims behave as "not claimed" —
requeue paths stay correct and submission dedup degrades to the pre-claims
behaviour (no worse than before).
"""
from __future__ import annotations

import logging
from typing import Iterable, Set

from app.services.infrastructure.cancel_registry import _get_redis

logger = logging.getLogger(__name__)

_CLAIM_PREFIX = "ingest:claim:"
# Refreshed on every task progress write; comfortably above the silence
# timeout (default 600s) so a merely-quiet live task stays claimed while a
# dead task's claim self-expires.
_CLAIM_TTL_S = 900


def _claim_key(task_id: int) -> str:
    return f"{_CLAIM_PREFIX}{task_id}"


def claim_ingestion(task_id: int) -> bool:
    """Atomically claim a task's ingestion slot.

    Returns ``True`` when the claim was acquired — the caller should submit
    the ingestion future.  ``False`` means another future already owns this
    task → skip the submission.
    """
    r = _get_redis()
    if r is None:
        return True
    try:
        return bool(r.set(_claim_key(task_id), "1", nx=True, ex=_CLAIM_TTL_S))
    except Exception:
        return True


def release_ingestion_claim(task_id: int) -> None:
    """Release a claim when the ingestion future resolves.  Idempotent."""
    r = _get_redis()
    if r is None:
        return
    try:
        r.delete(_claim_key(task_id))
    except Exception:
        pass


def touch_ingestion_claim(task_id: int) -> None:
    """Refresh the claim TTL — called on every task progress write."""
    r = _get_redis()
    if r is None:
        return
    try:
        r.expire(_claim_key(task_id), _CLAIM_TTL_S)
    except Exception:
        pass


def get_claimed_task_ids(task_ids: Iterable[int]) -> Set[int]:
    """Return the subset of *task_ids* that currently hold a claim."""
    ids = [t for t in task_ids if t is not None]
    if not ids:
        return set()
    r = _get_redis()
    if r is None:
        return set()
    try:
        pipe = r.pipeline()
        for tid in ids:
            pipe.exists(_claim_key(tid))
        return {tid for tid, exists in zip(ids, pipe.execute()) if exists}
    except Exception:
        return set()


def clear_ingestion_claims() -> int:
    """Delete all claims.  Call once at startup before recovery begins —
    every executor future from the previous process is dead."""
    r = _get_redis()
    if r is None:
        return 0
    try:
        keys = list(r.scan_iter(f"{_CLAIM_PREFIX}*", count=500))
        if keys:
            r.delete(*keys)
        logger.info("[INGEST-CLAIM] startup_clear removed=%d", len(keys))
        return len(keys)
    except Exception as e:
        logger.warning("[INGEST-CLAIM] startup_clear failed: %s", e)
        return 0
