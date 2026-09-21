"""E2E driver for claims + failed-task gating + parity healing.

Runs INSIDE the backend container. Uses real API (localhost:8000), real
MySQL/Qdrant/Redis. Creates a scratch datastore on /app/data/e2e_test.
"""
import json
import sys
import time

import httpx

BASE = "http://localhost:8000"


def login():
    r = httpx.post(f"{BASE}/api/auth/token",
                   data={"username": "super_admin", "password": "super_admin123"})
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def wait_tasks_settle(ds_id, timeout=180):
    """Wait until no pending/processing tasks remain for the datastore."""
    from app.db.session import SessionLocal
    from app.models.knowledge import ProcessingTask
    deadline = time.time() + timeout
    while time.time() < deadline:
        db = SessionLocal()
        try:
            active = db.query(ProcessingTask).filter(
                ProcessingTask.data_store_id == ds_id,
                ProcessingTask.status.in_(["pending", "processing"]),
            ).count()
            if active == 0:
                return True
        finally:
            db.close()
        time.sleep(2)
    return False


def task_statuses(ds_id):
    from app.db.session import SessionLocal
    from app.models.knowledge import Document, ProcessingTask
    db = SessionLocal()
    try:
        rows = (
            db.query(Document.file_name, ProcessingTask.id, ProcessingTask.status)
            .join(ProcessingTask, ProcessingTask.document_id == Document.id)
            .filter(Document.data_store_id == ds_id)
            .all()
        )
        return {name: (tid, status) for name, tid, status in rows}
    finally:
        db.close()


def chunk_count(ds_id):
    from app.db.session import SessionLocal
    from app.models.knowledge import DocumentChunk
    db = SessionLocal()
    try:
        return db.query(DocumentChunk).filter(
            DocumentChunk.data_store_id == ds_id).count()
    finally:
        db.close()


def main():
    hdr = login()
    step = lambda n, s: print(f"\n=== {n}: {s} ===", flush=True)

    # --- create datastore -------------------------------------------------
    step(1, "create datastore + files")
    r = httpx.post(f"{BASE}/api/admin/datastores", headers=hdr, json={
        "name": "e2e-claims-parity",
        "folder_path": "/app/data/e2e_test",
        "scan_pattern": "*",
        "select_all_files": True,
        "auto_process_enabled": False,
    })
    if r.status_code == 400 and "already exists" in r.text:
        # leftover from a previous run — reuse it
        from app.db.session import SessionLocal
        from app.models.datastore import DataStore
        db = SessionLocal()
        ds = db.query(DataStore).filter(
            DataStore.folder_path == "/app/data/e2e_test").first()
        ds_id = ds.id
        db.close()
        print(f"reusing existing datastore id={ds_id}")
    else:
        r.raise_for_status()
        ds_id = r.json()["id"]
        print(f"created datastore id={ds_id}")

    # --- scan 1: ingest good files, bad file fails -------------------------
    step(2, "scan 1 — ingest")
    r = httpx.post(f"{BASE}/api/admin/datastores/{ds_id}/scan", headers=hdr)
    print("scan:", r.status_code, r.json().get("message"))
    ok = wait_tasks_settle(ds_id)
    print("settled:", ok)
    print("tasks:", task_statuses(ds_id))
    print("chunks:", chunk_count(ds_id))
    st = task_statuses(ds_id)
    assert st["good1.txt"][1] == "completed", st
    assert st["good2.txt"][1] == "completed", st
    assert st["bad.docx"][1] == "completed", st

    # Fabricate the post-failure state for bad.docx: no chunks, no points,
    # task failed — exactly what a failed ingestion leaves behind.
    step("2b", "fabricate failed task for bad.docx")
    from app.db.session import SessionLocal
    from app.models.knowledge import Document, DocumentChunk, ProcessingTask
    from app.services.infrastructure import get_qdrant_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    db = SessionLocal()
    bad = db.query(Document).filter(
        Document.data_store_id == ds_id, Document.file_name == "bad.docx").first()
    bad_doc_id = bad.id
    db.query(DocumentChunk).filter(DocumentChunk.document_id == bad_doc_id).delete()
    db.commit()
    db.close()
    get_qdrant_client().delete(
        collection_name=f"ds_{ds_id}",
        points_selector=Filter(must=[FieldCondition(
            key="document_id", match=MatchValue(value=bad_doc_id))]),
    )
    db = SessionLocal()
    t = db.query(ProcessingTask).filter(
        ProcessingTask.document_id == bad_doc_id).first()
    t.status = "failed"
    t.error_message = "simulated failure for e2e"
    db.commit()
    db.close()
    print("bad.docx → failed (chunks+points removed)")

    # claims all released after completion
    from app.services.infrastructure.ingest_claims import (
        get_claimed_task_ids, claim_ingestion, release_ingestion_claim)
    ids = [tid for tid, _ in st.values()]
    assert get_claimed_task_ids(ids) == set(), "claims leaked after completion"
    print("claims released after completion: OK")

    # --- scan 2: failed task must NOT be retried ---------------------------
    step(3, "scan 2 — failed task stays failed")
    r = httpx.post(f"{BASE}/api/admin/datastores/{ds_id}/scan", headers=hdr)
    ok = wait_tasks_settle(ds_id)
    st = task_statuses(ds_id)
    print("tasks:", st)
    assert st["bad.docx"][1] == "failed", "failed task was retried!"

    # --- claim gate: flip failed→pending, hold claim, scan must skip -------
    step(4, "retry-failed + claim gate")
    r = httpx.post(f"{BASE}/api/admin/datastores/{ds_id}/retry-failed", headers=hdr)
    print("retry-failed:", r.json())
    assert r.json()["retried"] == 1
    st = task_statuses(ds_id)
    bad_task_id = st["bad.docx"][0]
    assert st["bad.docx"][1] == "pending"

    # hold the claim → scan must not consume the pending task
    assert claim_ingestion(bad_task_id)
    r = httpx.post(f"{BASE}/api/admin/datastores/{ds_id}/scan", headers=hdr)
    time.sleep(3)  # let scan's requeue pass run
    st = task_statuses(ds_id)
    print("while claimed:", st["bad.docx"])
    assert st["bad.docx"][1] == "pending", "claimed task was re-ingested!"
    print("claim gate respected: OK")

    release_ingestion_claim(bad_task_id)

    # --- scan 3: unclaimed pending gets consumed (re-ingests fine) --------
    step(5, "scan 3 — unclaimed pending consumed")
    r = httpx.post(f"{BASE}/api/admin/datastores/{ds_id}/scan", headers=hdr)
    ok = wait_tasks_settle(ds_id)
    st = task_statuses(ds_id)
    print("tasks:", st)
    assert st["bad.docx"][1] == "completed", st["bad.docx"]
    print("pending consumed → re-ingested: OK")

    # put it back into the failed state for the remaining checks? no —
    # remaining steps only touch good1/good2.

    # --- parity heal: pending task + full vectors → completed --------------
    step(6, "parity heal — vectors present → healed")
    from app.db.session import SessionLocal
    from app.models.knowledge import ProcessingTask
    good1_tid = task_statuses(ds_id)["good1.txt"][0]
    db = SessionLocal()
    t = db.query(ProcessingTask).get(good1_tid)
    t.status = "pending"  # simulate lost status write
    t.progress = 0
    db.commit()
    db.close()
    r = httpx.post(f"{BASE}/api/admin/datastores/{ds_id}/scan", headers=hdr)
    wait_tasks_settle(ds_id)
    st = task_statuses(ds_id)
    print("after heal-scan:", st["good1.txt"])
    assert st["good1.txt"][1] == "completed", st["good1.txt"]
    print("healed to completed via parity: OK")

    # --- parity heal negative: chunks exist, vectors gone → re-ingest -----
    step(7, "parity fail — points deleted → re-ingest, not heal")
    from app.services.infrastructure import get_qdrant_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qc = get_qdrant_client()
    from app.models.knowledge import Document
    db = SessionLocal()
    g1 = db.query(Document).filter(
        Document.data_store_id == ds_id, Document.file_name == "good1.txt").first()
    g1_id = g1.id
    db.close()
    qc.delete(
        collection_name=f"ds_{ds_id}",
        points_selector=Filter(must=[FieldCondition(
            key="document_id", match=MatchValue(value=g1_id))]),
    )
    # put task back to pending so stuck_docs picks it up
    db = SessionLocal()
    t = db.query(ProcessingTask).get(good1_tid)
    t.status = "pending"
    db.commit()
    db.close()
    r = httpx.post(f"{BASE}/api/admin/datastores/{ds_id}/scan", headers=hdr)
    wait_tasks_settle(ds_id)
    st = task_statuses(ds_id)
    print("after parity-fail scan:", st["good1.txt"])
    assert st["good1.txt"][1] == "completed", st["good1.txt"]
    # verify vectors actually restored
    cnt = qc.count(collection_name=f"ds_{ds_id}",
                   count_filter=Filter(must=[FieldCondition(
                       key="document_id", match=MatchValue(value=g1_id))]),
                   exact=True)
    print("good1 points after re-ingest:", cnt.count)
    assert cnt.count > 0, "vectors not restored"

    # --- reconciliation: wipe doc points → chunks deleted, task pending ----
    step(8, "reconciliation missing-vector repair")
    g2_tid = task_statuses(ds_id)["good2.txt"][0]
    db = SessionLocal()
    g2 = db.query(Document).filter(
        Document.data_store_id == ds_id, Document.file_name == "good2.txt").first()
    g2_id = g2.id
    db.close()
    qc.delete(
        collection_name=f"ds_{ds_id}",
        points_selector=Filter(must=[FieldCondition(
            key="document_id", match=MatchValue(value=g2_id))]),
    )
    chunks_before = chunk_count(ds_id)
    from app.services.cleanup.reconciliation_service import run_reconciliation
    summary = run_reconciliation()
    print("reconcile summary:", json.dumps(summary.get("qdrant"), indent=1))
    chunks_after = chunk_count(ds_id)
    st = task_statuses(ds_id)
    print("chunks", chunks_before, "->", chunks_after, "| task:", st["good2.txt"])
    assert chunks_after < chunks_before, "stale chunks not removed"
    assert st["good2.txt"][1] == "pending", st["good2.txt"]

    # --- scan 4: pending from reconciliation consumed → re-ingested --------
    step(9, "scan 4 — reconciliation-requeued doc re-ingests")
    r = httpx.post(f"{BASE}/api/admin/datastores/{ds_id}/scan", headers=hdr)
    wait_tasks_settle(ds_id)
    st = task_statuses(ds_id)
    cnt2 = qc.count(collection_name=f"ds_{ds_id}",
                    count_filter=Filter(must=[FieldCondition(
                        key="document_id", match=MatchValue(value=g2_id))]),
                    exact=True)
    print("tasks:", st, "| good2 points:", cnt2.count)
    assert st["good2.txt"][1] == "completed"
    assert cnt2.count > 0

    print("\nALL E2E CHECKS PASSED")
    return ds_id


if __name__ == "__main__":
    ds_id = main()
    sys.exit(0)
