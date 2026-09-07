"""End-to-end test for agentic-v2 pipeline with real LM Studio models.

Tests multiple scenarios:
1. Simple RAG query (search + answer)
2. Multi-tool query (search + rerank + read)
3. Chart generation query
4. Office document generation (PPTX)

Usage: docker exec rag-web-ui-backend-1 python tests/test_v2_e2e.py
"""

import asyncio
import json
import sys
import time
import traceback
from typing import Optional

from app.db.session import SessionLocal
from app.models.chat import Chat, Message
from app.services.agentic_rag.pipeline import run_agentic_rag


def _create_test_message(db, chat_id: int, role: str = "user", content: str = "") -> int:
    """Create a message row and return its id."""
    msg = Message(chat_id=chat_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg.id


def _print_event(event: dict, indent: str = "  "):
    evt_type = event.get("event", "?")
    if evt_type == "agent_step":
        print(f"{indent}STEP: {event['node']} {event['status']}")
    elif evt_type == "tool_call":
        args_str = json.dumps(event.get("arguments", {}), default=str)[:120]
        print(f"{indent}TOOL_CALL: {event['tool']} args={args_str}")
    elif evt_type == "tool_observation":
        print(f"{indent}TOOL_OBS: {event['tool']} summary={event.get('summary', '')[:100]}")
    elif evt_type == "tool_retry":
        print(f"{indent}TOOL_RETRY: {event['tool']} attempt={event.get('attempt')} success={event.get('success')}")
    elif evt_type == "answer_rewrite":
        content = event.get("content", "")
        print(f"{indent}ANSWER ({len(content)} chars): {content[:200]}...")
    elif evt_type == "last_answer":
        lao = event.get("last_answer_object", {})
        print(f"{indent}LAST_ANSWER: citations={len(lao.get('citations', []))} charts={len(lao.get('chart_options', []))} office={len(lao.get('office_files', []))} followups={len(lao.get('followups', []))}")
    elif evt_type == "context":
        print(f"{indent}CONTEXT: {len(event.get('docs', []))} docs, confidence={event.get('confidence')}")
    elif evt_type == "done":
        usage = event.get("usage", {})
        print(f"{indent}DONE: prompt={usage.get('promptTokens')} completion={usage.get('completionTokens')} estimated={usage.get('estimated')}")
    elif evt_type == "token":
        pass  # skip individual tokens
    elif evt_type == "progress":
        print(f"{indent}PROGRESS: {event.get('message', '')}")
    else:
        print(f"{indent}EVENT: {evt_type} keys={list(event.keys())}")


async def run_test(
    test_name: str,
    query: str,
    chat_id: int,
    kb_ids: list[int],
    org_id: int = 1,
    user_id: int = 1,
    file_markdown: Optional[str] = None,
    timeout: float = 180.0,
) -> dict:
    """Run one test scenario and return results."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")

    db = SessionLocal()
    msg_id = _create_test_message(db, chat_id, "user", query)

    events_log: list[dict] = []
    tool_calls: list[str] = []
    answer_text = ""
    has_answer = False
    has_done = False
    error = None

    start = time.time()
    try:
        async def _run():
            nonlocal answer_text, has_answer, has_done
            async for event in run_agentic_rag(
                query=query,
                chat_id=chat_id,
                knowledge_base_ids=kb_ids,
                db=db,
                file_markdown=file_markdown,
                org_id=org_id,
                user_id=user_id,
                message_id=msg_id,
                display_query=query,
            ):
                events_log.append(event)
                _print_event(event)
                if event.get("event") == "tool_call":
                    tool_calls.append(event["tool"])
                if event.get("event") == "answer_rewrite":
                    answer_text = event.get("content", "")
                    has_answer = True
                if event.get("event") == "done":
                    has_done = True

        await asyncio.wait_for(_run(), timeout=timeout)
    except asyncio.TimeoutError:
        error = f"TIMEOUT after {timeout}s"
        print(f"\n  *** {error} ***")
    except Exception as exc:
        error = str(exc)
        print(f"\n  *** ERROR: {error} ***")
        traceback.print_exc()
    finally:
        elapsed = time.time() - start
        db.close()

    # Evaluate
    print(f"\n--- RESULTS ({elapsed:.1f}s) ---")
    print(f"  Tool calls: {tool_calls}")
    print(f"  Has answer: {has_answer}")
    print(f"  Has done: {has_done}")
    print(f"  Answer length: {len(answer_text)} chars")
    if error:
        print(f"  ERROR: {error}")

    result = {
        "test": test_name,
        "query": query,
        "elapsed": elapsed,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "has_answer": has_answer,
        "has_done": has_done,
        "answer_length": len(answer_text),
        "answer_preview": answer_text[:300],
        "error": error,
        "events": events_log,
    }
    return result


async def main():
    db = SessionLocal()
    chat = db.query(Chat).first()
    if not chat:
        print("No chat found. Creating one...")
        chat = Chat(title="V2 E2E Test", org_id=1)
        db.add(chat)
        db.commit()
        db.refresh(chat)
    chat_id = chat.id
    # Use KB 1 which has 9 documents
    kb_ids = [1]
    db.close()

    results: list[dict] = []

    # Test 1: Simple RAG query
    r1 = await run_test(
        "Simple RAG query",
        "What is the StreamVC voice conversion system about?",
        chat_id=chat_id,
        kb_ids=kb_ids,
    )
    results.append(r1)

    # Test 2: Multi-tool query (search + rerank)
    r2 = await run_test(
        "Multi-tool: search + rerank",
        "What are the key concepts in distributed data processing?",
        chat_id=chat_id,
        kb_ids=kb_ids,
    )
    results.append(r2)

    # Test 3: Document-specific query
    r3 = await run_test(
        "Document-specific query",
        "What is the Thuraya monitoring system proposal about?",
        chat_id=chat_id,
        kb_ids=kb_ids,
    )
    results.append(r3)

    # Test 4: Office document generation
    r4 = await run_test(
        "Office: PPTX generation",
        "Create a 2-slide PowerPoint about the StreamVC voice conversion system. Slide 1: title and overview. Slide 2: key features.",
        chat_id=chat_id,
        kb_ids=kb_ids,
        timeout=240.0,
    )
    results.append(r4)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "PASS" if r["has_answer"] and r["has_done"] and not r["error"] else "FAIL"
        print(f"  [{status}] {r['test']}: {r['tool_call_count']} tool calls, {r['elapsed']:.1f}s, answer={r['answer_length']} chars")
        if r["error"]:
            print(f"         Error: {r['error']}")

    all_pass = all(r["has_answer"] and r["has_done"] and not r["error"] for r in results)
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
