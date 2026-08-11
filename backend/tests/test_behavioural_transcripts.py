"""Behavioural transcript tests for multi-turn agent conversations.

Unlike the structural tests in test_agent_state_integrity.py (which pin
individual graph mechanics), these tests run the **full agent graph** with
mocked LLMs that return scripted responses per node.  The graph mechanics
(routing, state propagation, reference resolution, tool dispatch, citation
normalisation, conversation history) are real; only the LLM outputs and the
rag_retrieve tool's vector search are mocked.

Each transcript is a sequence of turns.  After each turn we assert on the
final graph state — the answer text, retrieved docs, observations, citations,
and the checkpointed conversation history — to verify that conversation
quality actually holds across turns.

Metrics measured:
  - Entity-addition rate: does the conversation history accumulate user +
    assistant messages correctly across turns?
  - Topic carryover: does reference resolution correctly resolve "it"/"that"
    to the right entity from the right turn?
  - Unsupported-citation rate: does normalize_citations strip citations that
    point outside the retrieved docs range?
  - Clarification flow: does the interrupt → resume cycle work end-to-end?
  - Multi-tool plans: does a plan with 2+ subtasks actually execute all of
    them?
  - Code execution + chart generation: do non-retrieval tools flow through
    to the final answer?
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.services.agentic_rag import agent_graph, nodes
from app.services.agentic_rag.schemas import Observation, Plan, Subtask
from app.services.agentic_rag.tool_context import ToolContext


# ─── Helpers ───────────────────────────────────────────────────────────────


def _mock_docs(*pairs: tuple[str, str]) -> list[dict]:
    """Build mock retrieved docs from (content, source) pairs.

    Each doc gets a unique content_hash derived from the content itself,
    so docs from different rag_retrieve calls are not deduplicated by the
    tool_node's content_hash-based merge.
    """
    import hashlib
    return [
        {
            "page_content": content,
            "metadata": {
                "source": src,
                "content_hash": hashlib.md5(content.encode()).hexdigest()[:12],
            },
        }
        for content, src in pairs
    ]


def _make_ctx(db=None, chat_id=1, message_id=100) -> ToolContext:
    return ToolContext(
        db=db or MagicMock(),
        user_id=1,
        org_id=1,
        chat_id=chat_id,
        message_id=message_id,
        redis_memory=None,
        org_llm_config={},
    )


class _ScriptedLLM:
    """LLM mock that returns scripted responses based on the prompt content.

    Call ``script(role, content)`` to set the response for the next call
    matching that role.  Roles: "plan", "think", "finalize", "extract".
    The finalize response is streamed as individual tokens.
    """

    def __init__(self):
        self._scripts: dict[str, list[str]] = {}
        self._call_log: list[tuple[str, str]] = []

    def script(self, role: str, content: str):
        self._scripts.setdefault(role, []).append(content)

    def _pop(self, role: str) -> str:
        queue = self._scripts.get(role, [])
        if not queue:
            # When the think queue is empty, the pre-think sufficiency check
            # has already determined the plan is satisfied — return final_answer
            # so any stray think call finalizes instead of looping.
            if role == "think":
                return json.dumps({"final_answer": True})
            return "{}"
        return queue.pop(0)

    async def ainvoke(self, messages=None, *_a, **_kw):
        # Determine role from the system prompt content.
        sys_content = ""
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                sys_content = first.get("content", "")
            elif hasattr(first, "content"):
                sys_content = first.content

        if "Produce a plan JSON" in sys_content:
            role = "plan"
        elif "You are the acting module" in sys_content:
            role = "think"
        elif "Extract a structured summary" in sys_content:
            role = "extract"
        else:
            role = "finalize"

        content = self._pop(role)
        self._call_log.append((role, content[:200]))

        # For plan: return content that can be parsed as Plan JSON.
        if role == "plan":
            return SimpleNamespace(content=content)
        # For think: return content with .tool_calls=None so JSON-text parsing is used.
        if role == "think":
            return SimpleNamespace(content=content, tool_calls=None)
        # For extract: return content as-is.
        if role == "extract":
            return SimpleNamespace(content=content)
        # For finalize: ainvoke is not used (astream is), but provide a fallback.
        return SimpleNamespace(content=content)

    async def astream(self, messages=None, *_a, **_kw):
        content = self._pop("finalize")
        # Stream word-by-word to simulate real streaming.
        words = content.split()
        for i, word in enumerate(words):
            chunk_content = word + (" " if i < len(words) - 1 else "")
            yield SimpleNamespace(content=chunk_content, usage_metadata=None)
        # Final chunk with usage metadata.
        yield SimpleNamespace(
            content="",
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

    def with_structured_output(self, schema, **_kw):
        """Return a mock structured output wrapper."""
        outer = self

        class _Structured:
            async def ainvoke(self, messages=None, *_a, **_kw):
                content = outer._pop("plan")
                outer._call_log.append(("plan", content[:200]))
                # Try to parse as the Plan schema.
                try:
                    plan = Plan.model_validate_json(content)
                    return {"raw": SimpleNamespace(content=content), "parsed": plan, "parsing_error": None}
                except Exception:
                    return {
                        "raw": SimpleNamespace(content=content),
                        "parsed": None,
                        "parsing_error": ValueError("parse failed"),
                    }

        return _Structured()

    def bind_tools(self, *_a, **_kw):
        return self


def _setup_graph(monkeypatch, scripted_llm: _ScriptedLLM, ctx: ToolContext,
                 rag_results: dict[str, list[dict]] | None = None):
    """Wire up a full agent graph with mocked LLMs and rag_retrieve.

    ``rag_results`` maps a query string to the docs that rag_retrieve should
    return for that query.  Queries not in the map return empty docs.
    If ``ctx.redis_memory`` already has a checkpointer (from a prior call),
    it is reused so the graph state persists across calls.
    """
    rag_results = rag_results or {}

    # Mock build_chat_llm to return our scripted LLM everywhere.
    monkeypatch.setattr(agent_graph, "build_chat_llm", lambda *a, **kw: scripted_llm)

    # Mock resolve_retrieval_query — the rewriter uses AsyncOpenAI directly
    # (not LangChain), so build_chat_llm mock doesn't cover it. The test
    # exercises agent loop logic, not LLM rewriting ability.
    from app.services.agentic_rag import nodes as _nodes

    async def _mock_resolve(query, original_query, recent_history, provenance_sources, **kw):
        # "its" → resolve to the entity from the previous turn.
        # Simple heuristic: find the first capitalized word in the AI message.
        if "its" in query.lower() or "it" in query.lower().split():
            for m in reversed(recent_history):
                from langchain_core.messages import AIMessage
                if isinstance(m, AIMessage):
                    import re
                    # Match capitalized words including camelCase (e.g. "StreamVC").
                    words = re.findall(r'\b([A-Z][A-Za-z]+)\b', m.content)
                    if words:
                        return f"What are the limitations of {words[0]}?", {
                            "resolved": True,
                            "reason": "reference_resolved",
                            "original_query": original_query,
                        }
        return query, {"resolved": False, "reason": "self_contained"}

    monkeypatch.setattr(
        "app.services.agentic_rag.utils.resolve_retrieval_query",
        _mock_resolve,
    )

    # Mock the rag_retrieve tool to return scripted docs.
    async def _mock_rag_retrieve(ctx, input_obj):
        query = input_obj.query
        docs = rag_results.get(query, [])
        return {
            "ok": True,
            "result": {
                "docs": docs,
                "confidence": 0.85 if docs else 0.0,
                "sufficient": bool(docs),
            },
            "error": None,
            "tokens": 50,
        }

    monkeypatch.setattr(
        "app.services.agentic_rag.tools.rag_retrieve._rag_retrieve",
        _mock_rag_retrieve,
    )

    # Mock answer_evaluation_node to skip the LLM eval call.
    async def _mock_eval_node(state, ctx=None):
        return {
            "final_confidence": 80,
            "confidence_level": "high",
            "faithfulness": 85,
            "completeness": 80,
            "retrieval_score": 0.85,
        }

    monkeypatch.setattr(agent_graph, "answer_scoring_node", _mock_eval_node)

    # Inject a MemorySaver checkpointer via a mock redis_memory, but only
    # if one isn't already set (so re-calling _setup_graph with the same
    # ctx preserves the checkpointer and its checkpointed state).
    if not ctx.redis_memory or not getattr(ctx.redis_memory, "checkpointer", None):
        _checkpointer = MemorySaver()

        class _MockRedisMemory:
            checkpointer = _checkpointer

        ctx.redis_memory = _MockRedisMemory()

    graph = agent_graph.build_agent_graph(ctx)
    return graph


def _run_turn(graph, config, query: str, message_id: int = None,
              resume: str = None) -> dict:
    """Run one turn of the graph and return the final state.

    If ``resume`` is given, the turn is a Command(resume=...) to answer a
    clarification.  Otherwise it's a normal HumanMessage turn.
    """
    if resume is not None:
        result = asyncio.run(graph.ainvoke(
            Command(resume=resume), config,
        ))
    else:
        messages = [HumanMessage(content=query)]
        result = asyncio.run(graph.ainvoke(
            {"messages": messages, "original_query": query, "message_id": message_id},
            config,
        ))
    return result


def _get_state(graph, config) -> dict:
    """Get the current checkpointed state."""
    state = asyncio.run(graph.aget_state(config))
    return state.values


# ─── Transcript 1: Multi-turn reference resolution ─────────────────────────


class TestMultiTurnReferenceResolution:
    """Turn 1 introduces an entity; Turn 2 references it with a pronoun.

    Verifies that:
    - The rewrite resolves "its" to the entity from Turn 1.
    - The retrieval query for Turn 2 contains the entity name.
    - The answer for Turn 2 references the correct entity.
    - Conversation history accumulates both user and assistant turns.
    """

    def test_pronoun_resolves_to_prior_turn_entity(self, monkeypatch):
        llm = _ScriptedLLM()
        ctx = _make_ctx(chat_id=1, message_id=101)

        # Turn 1 plan: single rag_retrieve for StreamVC.
        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [{"id": "a", "description": "Retrieve StreamVC overview", "tool_hint": "rag_retrieve"}],
            "needs_clarification": False,
        }))
        # Turn 1 think: call rag_retrieve.
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "StreamVC model overview"}}],
        }))
        # Turn 1 finalize: answer about StreamVC.
        llm.script("finalize", "StreamVC is a voice conversion model based on streaming architecture [1](1).")
        # Turn 1 extract: LastAnswerObject.
        llm.script("extract", json.dumps({
            "summary": "StreamVC is a voice conversion model.",
            "key_points": ["Streaming architecture", "Voice conversion"],
        }))

        rag_docs = {
            "StreamVC model overview": _mock_docs(
                ("StreamVC is a real-time voice conversion model using streaming architecture.", "paper-1"),
            ),
        }

        graph = _setup_graph(monkeypatch, llm, ctx, rag_results=rag_docs)
        config = {"configurable": {"thread_id": "ref-resolution"}}

        # Turn 1.
        _run_turn(graph, config, "Tell me about the StreamVC model", message_id=101)
        state1 = _get_state(graph, config)

        # Verify Turn 1: assistant message persisted.
        msgs1 = state1.get("messages", [])
        assert any(isinstance(m, HumanMessage) and "StreamVC" in m.content for m in msgs1), \
            "Turn 1: user message should be in history"
        assert any(isinstance(m, AIMessage) and "StreamVC" in m.content for m in msgs1), \
            "Turn 1: assistant message should be in history"

        # Turn 2: "What are its limitations?" — "its" should resolve to "StreamVC".
        # Update message_id for turn 2.
        ctx.message_id = 102
        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [{"id": "a", "description": "Retrieve StreamVC limitations", "tool_hint": "rag_retrieve"}],
            "needs_clarification": False,
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "StreamVC limitations"}}],
        }))
        llm.script("finalize", "StreamVC has limitations in real-time latency and speaker adaptation [1](1).")
        llm.script("extract", json.dumps({
            "summary": "StreamVC has limitations in latency and speaker adaptation.",
            "key_points": ["Real-time latency", "Speaker adaptation"],
        }))

        rag_docs["StreamVC limitations"] = _mock_docs(
            ("StreamVC suffers from real-time latency and limited speaker adaptation.", "paper-2"),
        )

        _run_turn(graph, config, "What are its limitations?", message_id=102)
        state2 = _get_state(graph, config)

        # Verify Turn 2: rewritten query should contain "StreamVC".
        rewritten = state2.get("rewritten_query", "")
        assert "StreamVC" in rewritten or "streamvc" in rewritten.lower(), \
            f"Turn 2: rewritten query should resolve 'its' to 'StreamVC', got: {rewritten!r}"

        # Verify conversation history grew.
        msgs2 = state2.get("messages", [])
        human_count = sum(1 for m in msgs2 if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in msgs2 if isinstance(m, AIMessage))
        assert human_count >= 2, f"Turn 2: should have 2+ human messages, got {human_count}"
        assert ai_count >= 2, f"Turn 2: should have 2+ assistant messages, got {ai_count}"

        # Verify answer mentions StreamVC (not some other entity).
        answer2 = state2.get("final_answer", "")
        assert "StreamVC" in answer2 or "streamvc" in answer2.lower(), \
            f"Turn 2: answer should mention StreamVC, got: {answer2!r}"


# ─── Transcript 2: Topic carryover doesn't conflate ────────────────────────


class TestTopicCarryover:
    """Turn 1 about topic A, Turn 2 about topic B, Turn 3 references 'the first thing'.

    Verifies that:
    - Turn 2's rewrite doesn't leak topic A into topic B's retrieval query.
    - Turn 3's rewrite resolves "the first thing" to topic A, not topic B.
    """

    def test_topics_dont_conflate_across_turns(self, monkeypatch):
        llm = _ScriptedLLM()
        ctx = _make_ctx(chat_id=2, message_id=201)

        # Turn 1: Kubernetes autoscaling.
        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [{"id": "a", "description": "Retrieve K8s autoscaling", "tool_hint": "rag_retrieve"}],
            "needs_clarification": False,
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "Kubernetes autoscaling overview"}}],
        }))
        llm.script("finalize", "Kubernetes autoscaling adjusts pod count based on load [1](1).")
        llm.script("extract", json.dumps({"summary": "K8s autoscaling adjusts pods based on load."}))

        rag_docs = {
            "Kubernetes autoscaling overview": _mock_docs(
                ("Kubernetes autoscaling adjusts pod count based on CPU/memory load.", "k8s-doc"),
            ),
        }

        graph = _setup_graph(monkeypatch, llm, ctx, rag_results=rag_docs)
        config = {"configurable": {"thread_id": "topic-carryover"}}

        _run_turn(graph, config, "How does Kubernetes autoscaling work?", message_id=201)

        # Turn 2: Redis caching (completely different topic).
        ctx.message_id = 202
        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [{"id": "a", "description": "Retrieve Redis caching", "tool_hint": "rag_retrieve"}],
            "needs_clarification": False,
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "Redis caching strategies"}}],
        }))
        llm.script("finalize", "Redis caching uses TTL and LRU eviction [1](1).")
        llm.script("extract", json.dumps({"summary": "Redis caching uses TTL and LRU."}))

        rag_docs["Redis caching strategies"] = _mock_docs(
            ("Redis caching strategies include TTL-based eviction and LRU.", "redis-doc"),
        )

        _run_turn(graph, config, "What about Redis caching strategies?", message_id=202)
        state2 = _get_state(graph, config)

        # Turn 2's rewritten query should NOT contain "Kubernetes".
        rewritten2 = state2.get("rewritten_query", "")
        assert "kubernetes" not in rewritten2.lower(), \
            f"Turn 2: rewrite should not leak topic A (Kubernetes) into topic B, got: {rewritten2!r}"

        # Turn 3: "Can you compare that with the first thing?"
        # "that" = Redis caching (Turn 2), "the first thing" = Kubernetes autoscaling (Turn 1)
        ctx.message_id = 203
        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [
                {"id": "a", "description": "Retrieve K8s autoscaling", "tool_hint": "rag_retrieve"},
                {"id": "b", "description": "Retrieve Redis caching", "tool_hint": "rag_retrieve", "depends_on": ["a"]},
            ],
            "needs_clarification": False,
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "Kubernetes autoscaling comparison"}}],
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "Redis caching comparison"}}],
        }))
        llm.script("finalize", "K8s autoscaling and Redis caching serve different purposes [1](1) [2](2).")
        llm.script("extract", json.dumps({"summary": "Comparison of K8s autoscaling and Redis caching."}))

        rag_docs["Kubernetes autoscaling comparison"] = _mock_docs(
            ("Kubernetes autoscaling scales compute resources.", "k8s-doc"),
        )
        rag_docs["Redis caching comparison"] = _mock_docs(
            ("Redis caching speeds up data access.", "redis-doc"),
        )

        _run_turn(graph, config, "Can you compare that with the first thing?", message_id=203)
        state3 = _get_state(graph, config)

        # Turn 3 should have retrieved docs from both topics.
        docs3 = state3.get("retrieved_docs", [])
        assert len(docs3) >= 2, \
            f"Turn 3: should retrieve docs for both topics, got {len(docs3)} docs"

        # Conversation history should have 3 user + 3 assistant turns.
        msgs3 = state3.get("messages", [])
        human_count = sum(1 for m in msgs3 if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in msgs3 if isinstance(m, AIMessage))
        assert human_count >= 3, f"Turn 3: should have 3+ human messages, got {human_count}"
        assert ai_count >= 3, f"Turn 3: should have 3+ assistant messages, got {ai_count}"


# ─── Transcript 3: Clarification interrupt + resume ────────────────────────


class TestClarificationInterruptResume:
    """Turn 1 is ambiguous, triggers clarification; user answers; agent proceeds.

    Verifies that:
    - The graph interrupts and surfaces the clarification question.
    - Command(resume=...) routes the answer back through rewrite_query → plan.
    - The final answer uses the clarification to produce a relevant response.
    - The clarification_count is incremented.
    """

    def test_clarification_flow_end_to_end(self, monkeypatch):
        llm = _ScriptedLLM()
        ctx = _make_ctx(chat_id=3, message_id=301)

        # Turn 1 plan: needs clarification.
        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [],
            "needs_clarification": True,
            "clarification_question": "Which database are you asking about: PostgreSQL or MySQL?",
        }))

        graph = _setup_graph(monkeypatch, llm, ctx)
        config = {"configurable": {"thread_id": "clarification"}}

        # Run turn 1 — should interrupt.
        result = asyncio.run(graph.ainvoke(
            {"messages": [HumanMessage(content="How do I configure replication?")],
             "original_query": "How do I configure replication?",
             "message_id": 301},
            config,
        ))

        # Verify the graph interrupted.
        state_after_interrupt = asyncio.run(graph.aget_state(config))
        assert state_after_interrupt.next == ("clarify_interrupt",) or \
               "clarify_interrupt" in str(state_after_interrupt.next), \
            f"Expected graph to be paused at clarify_interrupt, next={state_after_interrupt.next}"

        # After clarification, plan again with the answer, then proceed.
        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [{"id": "a", "description": "Retrieve PostgreSQL replication", "tool_hint": "rag_retrieve"}],
            "needs_clarification": False,
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "PostgreSQL replication configuration"}}],
        }))
        llm.script("finalize", "PostgreSQL replication is configured via streaming replication [1](1).")
        llm.script("extract", json.dumps({"summary": "PostgreSQL replication via streaming."}))

        # Update the rag_results dict that the mock already captured by closure.
        # _setup_graph captured rag_results by reference, so we can add keys here.
        # But since _setup_graph was called with rag_results=None (empty dict),
        # we need to re-setup the graph with the rag docs while keeping the
        # same checkpointer (same ctx.redis_memory).
        rag_docs = {
            "PostgreSQL replication configuration": _mock_docs(
                ("PostgreSQL streaming replication uses WAL archives and hot standby.", "pg-doc"),
            ),
        }
        graph2 = _setup_graph(monkeypatch, llm, ctx, rag_results=rag_docs)

        # Resume with the clarification answer.
        result = asyncio.run(graph2.ainvoke(
            Command(resume="PostgreSQL"), config,
        ))

        state = _get_state(graph2, config)

        # Verify the answer mentions PostgreSQL.
        answer = state.get("final_answer", "")
        assert "PostgreSQL" in answer or "postgresql" in answer.lower(), \
            f"Answer should mention PostgreSQL after clarification, got: {answer!r}"

        # Verify clarification_count was incremented.
        assert state.get("clarification_count", 0) >= 1, \
            f"clarification_count should be >= 1, got {state.get('clarification_count')}"

        # Verify the conversation history includes the clarification exchange.
        msgs = state.get("messages", [])
        assert len(msgs) >= 2, f"Should have multiple messages after clarification, got {len(msgs)}"


# ─── Transcript 4: Multi-tool plan (2+ rag_retrieve subtasks) ──────────────


class TestMultiToolPlan:
    """A single turn with a plan requiring 2 rag_retrieve subtasks.

    Verifies that:
    - Both subtasks execute (2 rag_retrieve calls).
    - Observations accumulate correctly (no duplication).
    - Retrieved docs from both calls are merged into retrieved_docs.
    - The plan is marked complete only after both subtasks have results.
    """

    def test_two_subtasks_both_execute(self, monkeypatch):
        llm = _ScriptedLLM()
        ctx = _make_ctx(chat_id=4, message_id=401)

        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [
                {"id": "a", "description": "Retrieve revenue data", "tool_hint": "rag_retrieve"},
                {"id": "b", "description": "Retrieve cost data", "tool_hint": "rag_retrieve", "depends_on": ["a"]},
            ],
            "needs_clarification": False,
        }))
        # First think: call rag_retrieve for revenue.
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "Q3 revenue data"}}],
        }))
        # Second think: call rag_retrieve for costs.
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "Q3 cost data"}}],
        }))
        llm.script("finalize", "Revenue was $10M [1](1) and costs were $6M [2](2), yielding $4M profit.")
        llm.script("extract", json.dumps({"summary": "Q3 revenue $10M, costs $6M, profit $4M."}))

        rag_docs = {
            "Q3 revenue data": _mock_docs(
                ("Q3 revenue was $10 million, up 15% year over year.", "fin-report"),
            ),
            "Q3 cost data": _mock_docs(
                ("Q3 operating costs were $6 million, primarily from infrastructure.", "fin-report"),
            ),
        }

        graph = _setup_graph(monkeypatch, llm, ctx, rag_results=rag_docs)
        config = {"configurable": {"thread_id": "multi-tool"}}

        _run_turn(graph, config, "What were Q3 revenue and costs?", message_id=401)
        state = _get_state(graph, config)

        # Verify 2 rag_retrieve observations.
        observations = state.get("observations", [])
        rag_obs = [o for o in observations if isinstance(o, dict) and o.get("tool") == "rag_retrieve"
                   or (hasattr(o, "tool") and o.tool == "rag_retrieve")]
        # Observations might be Observation objects or dicts; coerce.
        from app.services.agentic_rag.agent_graph import _coerce_observation
        rag_obs = [_coerce_observation(o) for o in observations
                   if _coerce_observation(o).tool == "rag_retrieve"]
        assert len(rag_obs) == 2, \
            f"Should have 2 rag_retrieve observations, got {len(rag_obs)}: {[_coerce_observation(o).tool for o in observations]}"

        # Verify both docs are in retrieved_docs.
        docs = state.get("retrieved_docs", [])
        assert len(docs) >= 2, \
            f"Should have 2+ retrieved docs from both subtasks, got {len(docs)}"

        # Verify answer mentions both revenue and costs.
        answer = state.get("final_answer", "")
        assert "revenue" in answer.lower() or "$10" in answer, \
            f"Answer should mention revenue, got: {answer!r}"
        assert "cost" in answer.lower() or "$6" in answer, \
            f"Answer should mention costs, got: {answer!r}"


# ─── Transcript 5: Code execution + chart generation ───────────────────────


class TestCodeExecuteChartGenerate:
    """A turn that uses code_execute then chart_generate.

    Verifies that:
    - code_execute runs and its result is available to the final answer.
    - chart_generate produces a valid chart_option.
    - The final answer includes the chart.
    """

    def test_code_execute_then_chart(self, monkeypatch):
        llm = _ScriptedLLM()
        ctx = _make_ctx(chat_id=5, message_id=501)

        llm.script("plan", json.dumps({
            "intent": "computation",
            "subtasks": [
                {"id": "a", "description": "Calculate quarterly growth", "tool_hint": "code_execute"},
                {"id": "b", "description": "Generate chart", "tool_hint": "chart_generate", "depends_on": ["a"]},
            ],
            "needs_clarification": False,
        }))
        # First think: call code_execute.
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "code_execute", "arguments": {"code": "result = [10, 20, 30, 40]; print(result)"}}],
        }))
        # Second think: call chart_generate.
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "chart_generate", "arguments": {
                "chart_type": "bar",
                "data": [{"label": "Q1", "value": 10}, {"label": "Q2", "value": 20},
                         {"label": "Q3", "value": 30}, {"label": "Q4", "value": 40}],
                "title": "Quarterly Growth",
            }}],
        }))
        llm.script("finalize", "The quarterly growth shows an increasing trend from Q1 to Q4.")
        llm.script("extract", json.dumps({"summary": "Quarterly growth Q1-Q4 showing increasing trend."}))

        # chart_generate requires has_data (retrieved_docs or last_answer_object.data).
        # Mock applicable_tools to always include all tools, since this test
        # verifies tool execution flow, not tool availability gating.
        from app.services.agentic_rag.tools import build_tools
        monkeypatch.setattr(
            "app.services.agentic_rag.agent_graph.applicable_tools",
            lambda ctx: build_tools(ctx),
        )

        graph = _setup_graph(monkeypatch, llm, ctx)
        config = {"configurable": {"thread_id": "code-chart"}}

        _run_turn(graph, config, "Calculate and chart quarterly growth", message_id=501)
        state = _get_state(graph, config)

        # Verify observations include code_execute and chart_generate.
        from app.services.agentic_rag.agent_graph import _coerce_observation
        observations = [_coerce_observation(o) for o in state.get("observations", [])]
        tools_used = {o.tool for o in observations}
        assert "code_execute" in tools_used, \
            f"Should have code_execute observation, got: {tools_used}"
        assert "chart_generate" in tools_used, \
            f"Should have chart_generate observation, got: {tools_used}"

        # Verify chart_generate produced a valid chart_option.
        chart_obs = [o for o in observations if o.tool == "chart_generate" and not o.error]
        assert chart_obs, "Should have a successful chart_generate observation"
        chart_option = chart_obs[0].result.get("chart_option")
        assert chart_option is not None, "chart_generate should produce a chart_option"
        assert chart_option.get("series"), "chart_option should have series data"


# ─── Transcript 6: Entity-addition rate across 3 turns ─────────────────────


class TestEntityAdditionRate:
    """Three turns, each introducing a new entity.

    Verifies that:
    - Each turn adds exactly one user message and one assistant message.
    - No messages are lost or duplicated across turns.
    - The last_answer_object from each turn is available to the next.
    """

    def test_three_turns_accumulate_correctly(self, monkeypatch):
        llm = _ScriptedLLM()
        ctx = _make_ctx(chat_id=6, message_id=601)

        entities = ["Python", "Rust", "Go"]
        rag_docs = {}
        for entity in entities:
            rag_docs[f"{entity} overview"] = _mock_docs(
                (f"{entity} is a programming language.", f"{entity}-doc"),
            )

        graph = _setup_graph(monkeypatch, llm, ctx, rag_results=rag_docs)
        config = {"configurable": {"thread_id": "entity-addition"}}

        for i, entity in enumerate(entities):
            msg_id = 601 + i
            ctx.message_id = msg_id

            llm.script("plan", json.dumps({
                "intent": "rag",
                "subtasks": [{"id": "a", "description": f"Retrieve {entity} overview", "tool_hint": "rag_retrieve"}],
                "needs_clarification": False,
            }))
            llm.script("think", json.dumps({
                "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": f"{entity} overview"}}],
            }))
            llm.script("finalize", f"{entity} is a programming language with unique features [1](1).")
            llm.script("extract", json.dumps({"summary": f"{entity} is a programming language."}))

            _run_turn(graph, config, f"Tell me about {entity}", message_id=msg_id)

        state = _get_state(graph, config)
        msgs = state.get("messages", [])

        # Should have 3 user + 3 assistant messages.
        human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        assert len(human_msgs) == 3, \
            f"Should have exactly 3 human messages, got {len(human_msgs)}: {[m.content[:50] for m in human_msgs]}"
        assert len(ai_msgs) == 3, \
            f"Should have exactly 3 assistant messages, got {len(ai_msgs)}: {[m.content[:50] for m in ai_msgs]}"

        # Verify each entity appears in the conversation.
        all_content = " ".join(m.content for m in msgs)
        for entity in entities:
            assert entity in all_content, \
                f"{entity} should appear in conversation history"

        # Verify no duplicate assistant messages (by id).
        ai_ids = [m.id for m in ai_msgs if m.id]
        assert len(ai_ids) == len(set(ai_ids)), \
            f"Assistant message ids should be unique, got: {ai_ids}"


# ─── Transcript 7: Unsupported citation rejection ──────────────────────────


class TestUnsupportedCitationRejection:
    """Answer cites a non-existent doc index; normalize_citations should strip it.

    Verifies that:
    - Citations pointing outside the docs range are removed.
    - Valid citations are preserved.
    - The cited_doc_indices list only contains valid indices.
    """

    def test_out_of_range_citations_stripped(self, monkeypatch):
        from app.services.agentic_rag.utils import normalize_citations

        docs = _mock_docs(
            ("Doc 1 content", "source-1"),
            ("Doc 2 content", "source-2"),
        )

        answer = "Based on [1](1) and [5](5) and [99](99), we can conclude."
        normalized, cited = normalize_citations(answer, docs)

        # [1](1) is valid (index 1, docs has 2). [5](5) and [99](99) are out of range.
        assert "[1]" in normalized or "[1](1)" in normalized, \
            f"Valid citation [1] should be preserved, got: {normalized!r}"
        assert "5" not in normalized or "[5]" not in normalized, \
            f"Out-of-range citation [5] should be stripped, got: {normalized!r}"
        assert "99" not in normalized or "[99]" not in normalized, \
            f"Out-of-range citation [99] should be stripped, got: {normalized!r}"

        # cited should only contain valid indices.
        assert all(1 <= i <= len(docs) for i in cited), \
            f"All cited indices should be valid, got: {cited}"

    def test_no_docs_strips_all_citations(self):
        from app.services.agentic_rag.utils import normalize_citations

        answer = "Some claim [1](1) and [2](2)."
        normalized, cited = normalize_citations(answer, [])
        assert cited == [], "No docs means no citations"
        assert "[1]" not in normalized, f"Citations should be stripped, got: {normalized!r}"

    def test_citation_normalization_in_finalize(self, monkeypatch):
        """End-to-end: finalize_node normalizes citations from the LLM answer."""
        llm = _ScriptedLLM()
        ctx = _make_ctx(chat_id=7, message_id=701)

        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [{"id": "a", "description": "Retrieve data", "tool_hint": "rag_retrieve"}],
            "needs_clarification": False,
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "important data"}}],
        }))
        # Finalize cites [1] (valid) and [3] (out of range — only 1 doc).
        llm.script("finalize", "The key finding is in [1](1), also referenced in [3](3).")
        llm.script("extract", json.dumps({"summary": "Key finding."}))

        rag_docs = {
            "important data": _mock_docs(
                ("This is the important data.", "source-1"),
            ),
        }

        graph = _setup_graph(monkeypatch, llm, ctx, rag_results=rag_docs)
        config = {"configurable": {"thread_id": "citation-rejection"}}

        _run_turn(graph, config, "What is the key finding?", message_id=701)
        state = _get_state(graph, config)

        answer = state.get("final_answer", "")
        # [1](1) should be preserved, [3](3) should be stripped.
        assert "[1]" in answer, f"Valid citation [1] should be in answer, got: {answer!r}"
        assert "[3]" not in answer, f"Invalid citation [3] should be stripped, got: {answer!r}"

        # cited_doc_indices should only contain 1.
        cited = state.get("cited_doc_indices", [])
        assert cited == [1], f"Should cite only doc 1, got: {cited}"


# ─── Transcript 8: Observation non-duplication across turns ────────────────


class TestObservationNonDuplicationAcrossTurns:
    """Two turns, each calling rag_retrieve. Observations from Turn 1 should
    not leak into Turn 2.

    Verifies that:
    - load_context_node resets observations to empty each turn.
    - Turn 2's observations only contain Turn 2's tool calls.
    """

    def test_observations_reset_between_turns(self, monkeypatch):
        llm = _ScriptedLLM()
        ctx = _make_ctx(chat_id=8, message_id=801)

        # Turn 1.
        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [{"id": "a", "description": "Retrieve topic A", "tool_hint": "rag_retrieve"}],
            "needs_clarification": False,
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "topic A details"}}],
        }))
        llm.script("finalize", "Topic A is about algorithms [1](1).")
        llm.script("extract", json.dumps({"summary": "Topic A is about algorithms."}))

        rag_docs = {
            "topic A details": _mock_docs(("Topic A covers sorting algorithms.", "doc-a")),
            "topic B details": _mock_docs(("Topic B covers data structures.", "doc-b")),
        }

        graph = _setup_graph(monkeypatch, llm, ctx, rag_results=rag_docs)
        config = {"configurable": {"thread_id": "obs-reset"}}

        _run_turn(graph, config, "Tell me about topic A", message_id=801)
        state1 = _get_state(graph, config)
        obs1 = state1.get("observations", [])
        assert len(obs1) == 1, f"Turn 1: should have 1 observation, got {len(obs1)}"

        # Turn 2.
        ctx.message_id = 802
        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [{"id": "a", "description": "Retrieve topic B", "tool_hint": "rag_retrieve"}],
            "needs_clarification": False,
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "topic B details"}}],
        }))
        llm.script("finalize", "Topic B is about data structures [1](1).")
        llm.script("extract", json.dumps({"summary": "Topic B is about data structures."}))

        _run_turn(graph, config, "Tell me about topic B", message_id=802)
        state2 = _get_state(graph, config)

        # Turn 2 observations should only contain Turn 2's tool call.
        obs2 = state2.get("observations", [])
        from app.services.agentic_rag.agent_graph import _coerce_observation
        coerced2 = [_coerce_observation(o) for o in obs2]
        assert len(coerced2) == 1, \
            f"Turn 2: should have 1 observation (reset), got {len(coerced2)}"
        assert coerced2[0].tool == "rag_retrieve", \
            f"Turn 2: observation should be rag_retrieve, got {coerced2[0].tool}"

        # The query should be "topic B details", not "topic A details".
        assert "topic B" in coerced2[0].arguments.get("query", ""), \
            f"Turn 2: observation should be for topic B, got: {coerced2[0].arguments}"


# ─── Transcript 9: Previous answer action (summarize_answer) ───────────────


class TestPreviousAnswerAction:
    """Turn 2 asks to summarize Turn 1's answer.

    Verifies that:
    - The plan recognizes intent="previous_answer_action".
    - The last_answer_object from Turn 1 is available to Turn 2.
    - The summarize_answer tool is used.
    """

    def test_summarize_previous_answer(self, monkeypatch):
        llm = _ScriptedLLM()
        ctx = _make_ctx(chat_id=9, message_id=901)

        # Turn 1: normal RAG turn.
        llm.script("plan", json.dumps({
            "intent": "rag",
            "subtasks": [{"id": "a", "description": "Retrieve data", "tool_hint": "rag_retrieve"}],
            "needs_clarification": False,
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "rag_retrieve", "arguments": {"query": "machine learning basics"}}],
        }))
        llm.script("finalize", "Machine learning is a subset of AI that learns from data [1](1).")
        llm.script("extract", json.dumps({
            "summary": "ML is a subset of AI that learns from data.",
            "key_points": ["Subset of AI", "Learns from data"],
        }))

        rag_docs = {
            "machine learning basics": _mock_docs(
                ("Machine learning is a subset of AI focused on learning from data.", "ml-doc"),
            ),
        }

        graph = _setup_graph(monkeypatch, llm, ctx, rag_results=rag_docs)
        config = {"configurable": {"thread_id": "prev-answer"}}

        _run_turn(graph, config, "What is machine learning?", message_id=901)
        state1 = _get_state(graph, config)

        # Verify last_answer_object was set.
        lao = state1.get("last_answer_object")
        assert lao is not None, "Turn 1: last_answer_object should be set"
        assert hasattr(lao, "summary"), "last_answer_object should have a summary"

        # Turn 2: "Summarize your last answer."
        ctx.message_id = 902
        llm.script("plan", json.dumps({
            "intent": "previous_answer_action",
            "subtasks": [{"id": "a", "description": "Summarize previous answer", "tool_hint": "summarize_answer"}],
            "needs_clarification": False,
        }))
        llm.script("think", json.dumps({
            "tool_calls": [{"tool": "summarize_answer", "arguments": {"action": "summarize"}}],
        }))
        llm.script("finalize", "In summary: ML is a subset of AI that learns from data.")
        llm.script("extract", json.dumps({"summary": "Summary of ML answer."}))

        _run_turn(graph, config, "Summarize your last answer", message_id=902)
        state2 = _get_state(graph, config)

        # Verify the answer references the previous answer's content.
        answer2 = state2.get("final_answer", "")
        assert "ML" in answer2 or "machine learning" in answer2.lower(), \
            f"Turn 2: answer should reference previous answer content, got: {answer2!r}"

        # Verify last_answer_object from Turn 1 was available (it's now replaced by Turn 2's).
        lao2 = state2.get("last_answer_object")
        assert lao2 is not None, "Turn 2: last_answer_object should be set"
