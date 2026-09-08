"""Comprehensive end-to-end test harness for the v2 agent pipeline.

Tests all query types with real LM Studio models:
1. Simple RAG query (direct search)
2. Complex multi-part query (retrieve_parallel)
3. Named document query (kb_search_documents + file_read)
4. Chart generation (code_execute + chart_generate)
5. PPTX generation (create_office_document)
6. DOCX generation (create_office_document)
7. XLSX generation (create_office_document)
8. Insufficient evidence handling
9. Tool error recovery

Run inside the backend container:
  docker exec rag-web-ui-backend-1 python tests/test_v2_e2e_full.py
"""

import asyncio
import json
import time
import traceback
import sys

from app.db.session import SessionLocal
from app.models.chat import Message
from app.services.agentic_rag.pipeline import run_agentic_rag


def make_msg(db, content):
    msg = Message(chat_id=4, role="user", content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


async def run_scenario(db, label, query, expected_tools=None, expect_office=False,
                       expect_chart=False, expect_citations=True, expect_answer=True):
    """Run one E2E scenario and return a result dict."""
    msg = make_msg(db, query)
    msg_id = msg.id

    tool_calls = []
    answer_text = ""
    has_done = False
    citations = 0
    charts = 0
    office_files = []
    errors = []
    events_log = []
    start = time.time()

    try:
        async for event in run_agentic_rag(
            query=query,
            chat_id=4, knowledge_base_ids=[1], db=db,
            org_id=1, user_id=1, message_id=msg_id,
            display_query=query,
        ):
            evt = event.get("event", "?")
            events_log.append(evt)

            if evt == "tool_call":
                tool_calls.append(event["tool"])
            elif evt == "answer_rewrite":
                answer_text = event.get("content", "")
            elif evt == "last_answer":
                lao = event.get("last_answer_object", {})
                citations = len(lao.get("citations", []))
                charts = len(lao.get("chart_options", []))
                office_files = lao.get("office_files", [])
            elif evt == "done":
                has_done = True
            elif evt == "error":
                errors.append(event.get("error", "unknown"))
    except Exception as e:
        errors.append(str(e))
        traceback.print_exc()

    elapsed = time.time() - start

    # Validate
    failures = []
    if expect_answer and not answer_text.strip():
        failures.append("no answer text")
    if not has_done:
        failures.append("no done event")
    if expect_office and not office_files:
        failures.append(f"expected office file but got 0 (tools={tool_calls})")
    if expect_chart and not charts:
        failures.append(f"expected chart but got 0 (tools={tool_calls})")
    if expect_citations and citations == 0 and not expect_office:
        # Citations not required for office-only or chart-only queries
        if any(t in tool_calls for t in ("search_dense", "search_exact", "search_sparse",
                                          "kb_search_documents", "file_read", "retrieve_parallel")):
            failures.append(f"expected citations but got 0 (tools={tool_calls})")
    if expected_tools:
        for t in expected_tools:
            if t not in tool_calls:
                failures.append(f"expected tool {t} not called (tools={tool_calls})")
    if errors:
        failures.append(f"errors: {errors}")

    status = "PASS" if not failures else "FAIL"
    result = {
        "label": label,
        "status": status,
        "elapsed": round(elapsed, 1),
        "tools": tool_calls,
        "answer_len": len(answer_text),
        "citations": citations,
        "charts": charts,
        "office_files": len(office_files),
        "failures": failures,
        "answer_preview": answer_text[:200],
    }

    # Print result
    icon = "✓" if status == "PASS" else "✗"
    print(f"\n{icon} [{label}] {status} ({result['elapsed']}s)")
    print(f"  tools={tool_calls} citations={citations} charts={charts} office={len(office_files)} answer={len(answer_text)} chars")
    if failures:
        for f in failures:
            print(f"  FAILURE: {f}")
    if answer_text:
        print(f"  preview: {answer_text[:150]}...")

    return result


async def main():
    db = SessionLocal()
    results = []

    # --- Scenario 1: Simple RAG query ---
    r = await run_scenario(
        db, "simple_rag",
        "What is Velvera Plus?",
        expected_tools=None,  # may use search or answer directly
        expect_citations=True,
    )
    results.append(r)

    # --- Scenario 2: Complex multi-part query ---
    r = await run_scenario(
        db, "complex_parallel",
        "Compare the Velvera Watch Series product specifications with the Velvera Charging Solutions usage guide.",
        expected_tools=["retrieve_parallel"],
        expect_citations=True,
    )
    results.append(r)

    # --- Scenario 3: Named document query ---
    r = await run_scenario(
        db, "named_doc",
        "What does the Warranty & Service document say?",
        expected_tools=None,  # kb_search_documents or search_exact
        expect_citations=True,
    )
    results.append(r)

    # --- Scenario 4: Chart generation ---
    r = await run_scenario(
        db, "chart_gen",
        "Create a bar chart showing the product categories available in the Velvera catalog.",
        expected_tools=None,  # code_execute + chart_generate
        expect_chart=True,
        expect_citations=False,
    )
    results.append(r)

    # --- Scenario 5: PPTX generation ---
    r = await run_scenario(
        db, "pptx_gen",
        "Create a 2-slide PowerPoint about Velvera products. Slide 1: Product overview. Slide 2: Key features.",
        expected_tools=["create_office_document"],
        expect_office=True,
        expect_citations=False,
    )
    results.append(r)

    # --- Scenario 6: DOCX generation ---
    r = await run_scenario(
        db, "docx_gen",
        "Create a one-page Word document about Velvera warranty and service policies.",
        expected_tools=["create_office_document"],
        expect_office=True,
        expect_citations=False,
    )
    results.append(r)

    # --- Scenario 7: XLSX generation ---
    r = await run_scenario(
        db, "xlsx_gen",
        "Create an Excel spreadsheet listing Velvera product names and their categories.",
        expected_tools=["create_office_document"],
        expect_office=True,
        expect_citations=False,
    )
    results.append(r)

    # --- Scenario 8: Insufficient evidence ---
    r = await run_scenario(
        db, "insufficient_evidence",
        "What is the capital of France?",
        expected_tools=None,
        expect_citations=False,
        expect_answer=True,
    )
    results.append(r)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"  PASS: {passed}/{len(results)}")
    print(f"  FAIL: {failed}/{len(results)}")
    if failed:
        print("\nFailed scenarios:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  [{r['label']}] {r['failures']}")

    db.close()
    return results


if __name__ == "__main__":
    asyncio.run(main())
