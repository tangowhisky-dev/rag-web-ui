"""Structural regression tests for agent conversation/context state.

Each test here pins one defect found in the enterprise agent review:
assistant turns missing from the checkpoint, observation duplication through
an append-only reducer, undeclared state keys being silently dropped, a
swallowed clarification interrupt, compaction that grew the checkpoint, and
recalled memory leaking into citable evidence.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.services.agentic_rag.graph_state import AgentState, accumulate
from app.services.agentic_rag.schemas import Observation, Plan, Subtask
from app.services.agentic_rag.tool_context import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(
        db=MagicMock(), user_id=1, org_id=1, chat_id=None, message_id=None,
        redis_memory=None, org_llm_config={},
    )


class TestAssistantTurnsArePersisted:
    """The checkpointed thread previously held user questions only.

    Every consumer of history (reference resolution, think, compaction) was
    therefore reading half a conversation.
    """

    def _finalize_update(self, monkeypatch, answer: str, message_id: int) -> dict:
        from app.services.agentic_rag import agent_graph

        class _FakeLLM:
            async def ainvoke(self, *_a, **_kw):
                return SimpleNamespace(content="not json")

        monkeypatch.setattr(agent_graph, "build_chat_llm", lambda *a, **kw: _FakeLLM())
        return asyncio.run(agent_graph.finalize_node(
            {
                "precomputed_answer": answer,
                "original_query": "q",
                "rewritten_query": "q",
                "observations": [],
                "retrieved_docs": [],
                "messages": [HumanMessage(content="q")],
                "message_id": message_id,
            },
            _ctx(),
        ))

    def test_finalize_appends_an_assistant_message(self, monkeypatch):
        update = self._finalize_update(monkeypatch, "the answer", message_id=7)
        assert [type(m) for m in update["messages"]] == [AIMessage]
        assert update["messages"][0].content == "the answer"

    def test_turn_two_sees_turn_one_answer_exactly_once(self, monkeypatch):
        from app.services.agentic_rag.nodes import select_recent_history

        graph = StateGraph(AgentState)
        graph.add_node("noop", lambda _s: {})
        graph.add_edge(START, "noop")
        graph.add_edge("noop", END)
        app = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "two-turn"}}

        # Turn 1: user question, then the assistant answer from finalize.
        app.invoke({"messages": [HumanMessage(content="what is a mutex?")]}, config)
        turn1 = self._finalize_update(monkeypatch, "A mutex is a lock.", message_id=1)
        app.invoke({"messages": turn1["messages"]}, config)

        # Turn 2: a new user question lands on the same thread.
        state = app.invoke({"messages": [HumanMessage(content="what about its limits?")]}, config)

        contents = [m.content for m in state["messages"]]
        assert contents == [
            "what is a mutex?", "A mutex is a lock.", "what about its limits?",
        ]
        assert contents.count("A mutex is a lock.") == 1

        # The resolver now sees the answer it is asked to resolve against.
        history = select_recent_history(state["messages"])
        assert any(isinstance(m, AIMessage) for m in history)

    def test_replayed_finalize_does_not_duplicate_the_turn(self, monkeypatch):
        graph = StateGraph(AgentState)
        graph.add_node("noop", lambda _s: {})
        graph.add_edge(START, "noop")
        graph.add_edge("noop", END)
        app = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "replay"}}

        app.invoke({"messages": [HumanMessage(content="q")]}, config)
        update = self._finalize_update(monkeypatch, "answer", message_id=42)
        app.invoke({"messages": update["messages"]}, config)
        state = app.invoke({"messages": update["messages"]}, config)

        # The stable per-message id makes the append idempotent.
        assert [m.content for m in state["messages"]] == ["q", "answer"]


class TestDeclaredStateKeys:
    """LangGraph silently discards updates for keys absent from the schema.

    started_at / force_finalize / precomputed_tool_calls were undeclared, which
    made AGENT_MAX_WALL_SECONDS a no-op and reflect_node entirely inert.
    """

    def _roundtrip(self, update: dict) -> dict:
        seen = {}

        def writer(_state):
            return update

        def reader(state):
            seen.update({k: state.get(k) for k in update})
            return {}

        graph = StateGraph(AgentState)
        graph.add_node("writer", writer)
        graph.add_node("reader", reader)
        graph.add_edge(START, "writer")
        graph.add_edge("writer", "reader")
        graph.add_edge("reader", END)
        graph.compile().invoke({"messages": [HumanMessage(content="q")]})
        return seen

    def test_started_at_survives_a_node_hop(self):
        assert self._roundtrip({"started_at": 1234.5})["started_at"] == 1234.5

    def test_force_finalize_survives_a_node_hop(self):
        assert self._roundtrip({"force_finalize": True})["force_finalize"] is True

    def test_precomputed_tool_calls_survive_a_node_hop(self):
        calls = [{"tool": "extract_data", "arguments": {"source": "retrieved_docs"}}]
        assert self._roundtrip({"precomputed_tool_calls": calls})["precomputed_tool_calls"] == calls

    def test_wall_clock_budget_actually_terminates(self, monkeypatch):
        from app.services.settings_service import get_setting as _real_get_setting
        from app.services.agentic_rag.agent_graph import _wall_clock_exceeded, route_think

        def _mock_get_setting(db, key, org_id=None):
            if key == "AGENT_MAX_WALL_SECONDS":
                return 0.0
            return _real_get_setting(db, key, org_id)

        monkeypatch.setattr("app.services.agentic_rag.agent_graph.get_setting", _mock_get_setting)
        state = {"started_at": 0.0, "iteration": 1, "tool_calls": [{"tool": "rag_retrieve"}]}
        assert _wall_clock_exceeded(state) is True
        assert route_think(state) == "reflect_final"

    def test_clarification_question_declared_once(self):
        source = open("/app/app/services/agentic_rag/graph_state.py").read()
        assert source.count("clarification_question: Annotated") == 1


class TestObservationAccumulation:
    """tool_node must return only the observations it created.

    The `observations` channel uses the append-style `accumulate` reducer, so
    returning prior + new grew the list 1 -> 3 -> 7 -> 15 across tool rounds.
    """

    def _obs(self, n: int) -> Observation:
        return Observation(
            tool="rag_retrieve",
            arguments={"query": f"q{n}"},
            result={"docs": [{"page_content": f"doc{n}"}]},
            error=None,
            tokens=1,
        )

    def test_three_rounds_persist_three_observations(self):
        from app.services.agentic_rag.agent_graph import tool_node

        ctx = _ctx()
        state = {"observations": [], "tool_call_count": {}, "retrieved_docs": [], "plan": Plan()}
        for n in range(1, 4):
            update = asyncio.run(tool_node(
                {**state, "tool_calls": [{"tool": "nonexistent_tool", "arguments": {"n": n}}]},
                ctx,
            ))
            # Exactly one new observation is returned per round, never the
            # accumulated list.
            assert len(update["observations"]) == 1
            state["observations"] = accumulate(state["observations"], update["observations"])
            state["tool_call_count"] = update["tool_call_count"]

        assert len(state["observations"]) == 3

    def test_compaction_replaces_observations_instead_of_appending(self):
        from app.services.agentic_rag.agent_graph import _compact_observations

        existing = [self._obs(1), self._obs(2)]
        compacted = _compact_observations(existing)
        update = [{"__reset__": True}, *compacted]
        assert len(accumulate(existing, update)) == len(compacted)


class TestRecalledMemoryIsNotEvidence:
    """A prior model answer is conversational memory, not a knowledge-base source."""

    def test_load_context_keeps_memory_out_of_retrieved_docs(self):
        from app.services.agentic_rag.agent_graph import load_context_node

        recalled = [{"page_content": "you said X last week", "metadata": {}}]
        memory = SimpleNamespace(
            search_memory=lambda **kwargs: asyncio.sleep(0, result=recalled)
        )
        ctx = ToolContext(
            db=MagicMock(), user_id=1, org_id=1, chat_id=None, message_id=None,
            redis_memory=memory, org_llm_config={},
        )
        update = asyncio.run(load_context_node({"original_query": "q"}, ctx))

        assert update["recalled_memories"] == recalled
        assert update["retrieved_docs"] == []

    def test_tool_node_does_not_promote_memory_into_evidence(self):
        from app.services.agentic_rag.agent_graph import tool_node

        recalled = [{"page_content": "recalled memory text", "metadata": {}}]
        update = asyncio.run(tool_node(
            {
                "observations": [],
                "tool_call_count": {},
                "retrieved_docs": [],
                "recalled_memories": recalled,
                "plan": Plan(),
                "tool_calls": [{"tool": "nonexistent_tool", "arguments": {}}],
            },
            _ctx(),
        ))
        assert "retrieved_docs" not in update or update["retrieved_docs"] == []


class TestSubtaskVerification:
    """Three subtasks with the same tool_hint need three successful observations."""

    def _plan(self, n: int) -> Plan:
        return Plan(
            intent="rag",
            subtasks=[
                Subtask(id=chr(97 + i), description=f"part {i}", tool_hint="rag_retrieve")
                for i in range(n)
            ],
        )

    def _obs(self, n: int) -> Observation:
        return Observation(
            tool="rag_retrieve", arguments={"query": f"q{n}"},
            result={"docs": [{"page_content": f"doc{n}"}]}, error=None, tokens=1,
        )

    def test_one_retrieval_does_not_complete_three_subtasks(self):
        from app.services.agentic_rag.agent_graph import _build_execution_summary, _verify_execution

        summary = _build_execution_summary({
            "plan": self._plan(3),
            "observations": [self._obs(1)],
            "tool_call_count": {"rag_retrieve": 1},
            "iteration": 1,
        })
        assert [s["completed"] for s in summary["subtasks"]] == [True, False, False]
        assert _verify_execution(summary)[0] is False

    def test_three_retrievals_complete_three_subtasks(self):
        from app.services.agentic_rag.agent_graph import _build_execution_summary, _verify_execution

        summary = _build_execution_summary({
            "plan": self._plan(3),
            "observations": [self._obs(1), self._obs(2), self._obs(3)],
            "tool_call_count": {"rag_retrieve": 3},
            "iteration": 3,
        })
        assert all(s["completed"] for s in summary["subtasks"])
        assert _verify_execution(summary)[0] is True


class TestClarificationFlow:
    """interrupt() raises GraphInterrupt, which subclasses Exception.

    A broad `except Exception` around it swallowed the pause; the graph then
    ran on with an empty clarification answer.
    """

    def test_graph_interrupt_is_an_exception_subclass(self):
        from langgraph.errors import GraphInterrupt

        assert issubclass(GraphInterrupt, Exception)

    def test_clarify_node_propagates_the_interrupt(self):
        from app.services.agentic_rag.agent_graph import clarify_interrupt_node

        graph = StateGraph(AgentState)
        graph.add_node("clarify", clarify_interrupt_node)
        graph.add_edge(START, "clarify")
        graph.add_edge("clarify", END)
        app = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "clarify-test"}}

        plan = Plan(intent="rag", needs_clarification=True, clarification_question="Which report?")

        async def _first_pass():
            return [
                c async for c in app.astream(
                    {"messages": [HumanMessage(content="show me the numbers")], "plan": plan},
                    config,
                    stream_mode="updates",
                )
            ]

        chunks = asyncio.run(_first_pass())
        # The graph paused rather than completing with an empty answer.
        assert any("__interrupt__" in c for c in chunks)
        assert asyncio.run(app.aget_state(config)).next == ("clarify",)

        asyncio.run(app.ainvoke(Command(resume="the Q3 revenue report"), config))
        final = asyncio.run(app.aget_state(config)).values
        assert final["clarification_response"] == "the Q3 revenue report"
        assert final["clarification_count"] == 1
        assert final["needs_clarification"] is False
        # Only the new clarification message was appended.
        assert [m.content for m in final["messages"]] == [
            "show me the numbers", "the Q3 revenue report",
        ]

    def test_clarification_budget_is_capped(self, monkeypatch):
        from app.services.settings_service import get_setting as _real_get_setting
        from app.services.agentic_rag import agent_graph

        def _mock_get_setting(db, key, org_id=None):
            if key == "AGENT_MAX_CLARIFICATIONS":
                return 1
            return _real_get_setting(db, key, org_id)

        monkeypatch.setattr(agent_graph, "get_setting", _mock_get_setting)

        plan = Plan(intent="rag", needs_clarification=True, clarification_question="Which one?")

        class _FakeLLM:
            def with_structured_output(self, *_a, **_kw):
                return self

            async def ainvoke(self, *_a, **_kw):
                return {"parsed": plan, "parsing_error": None}

        monkeypatch.setattr(agent_graph, "build_chat_llm", lambda *a, **kw: _FakeLLM())

        first = asyncio.run(agent_graph.plan_node(
            {"original_query": "q", "rewritten_query": "q", "clarification_count": 0}, _ctx(),
        ))
        assert first["needs_clarification"] is True
        assert agent_graph.route_plan(first) == "clarify_interrupt"

        second = asyncio.run(agent_graph.plan_node(
            {"original_query": "q", "rewritten_query": "q", "clarification_count": 1}, _ctx(),
        ))
        assert second["needs_clarification"] is False
        assert agent_graph.route_plan(second) == "think"


class TestCompactionReplacesHistory:
    """`add_messages` appends: `[summary] + recent` grew the checkpoint."""

    def test_compaction_removes_old_messages_and_keeps_one_summary(self, monkeypatch):
        from app.core.settings_registry import get_def as _real_get_def
        from app.services.agentic_rag import agent_graph

        def _mock_get_def(key):
            if key == "COMPACTION_KEEP_RECENT":
                return SimpleNamespace(default=2)
            return _real_get_def(key)

        monkeypatch.setattr(agent_graph, "get_def", _mock_get_def)

        class _FakeLLM:
            async def ainvoke(self, *_a, **_kw):
                return SimpleNamespace(content="## Goal\nDiscussed mutexes.")

        monkeypatch.setattr(agent_graph, "_build_compaction_llm", lambda ctx: _FakeLLM())

        history = [
            HumanMessage(content="q1", id="1"),
            AIMessage(content="a1", id="2"),
            HumanMessage(content="q2", id="3"),
            AIMessage(content="a2", id="4"),
            HumanMessage(content="q3", id="5"),
        ]

        def compact(state):
            updates, _local, _summary = asyncio.run(
                agent_graph._compact_messages_llm(state["messages"])
            )
            return {"messages": updates}

        updates, resolved, summary = asyncio.run(
            agent_graph._compact_messages_llm(history)
        )
        assert summary is not None
        assert [m.content for m in resolved][1:] == ["a2", "q3"]

        graph = StateGraph(AgentState)
        graph.add_node("compact", compact)
        graph.add_edge(START, "compact")
        graph.add_edge("compact", END)
        out = graph.compile().invoke({"messages": history})

        contents = [m.content for m in out["messages"]]
        assert len(contents) == 3, contents
        assert contents[0].startswith("[Conversation summary]")
        assert contents[1:] == ["a2", "q3"]
        # Old turns are gone from the checkpoint, not merely shadowed.
        assert "q1" not in contents and "a1" not in contents

    def test_second_compaction_replaces_the_first_summary(self, monkeypatch):
        from app.core.settings_registry import get_def as _real_get_def
        from app.services.agentic_rag import agent_graph

        def _mock_get_def(key):
            if key == "COMPACTION_KEEP_RECENT":
                return SimpleNamespace(default=2)
            return _real_get_def(key)

        monkeypatch.setattr(agent_graph, "get_def", _mock_get_def)

        class _FakeLLM:
            async def ainvoke(self, *_a, **_kw):
                return SimpleNamespace(content="summary text")

        monkeypatch.setattr(agent_graph, "_build_compaction_llm", lambda ctx: _FakeLLM())

        graph = StateGraph(AgentState)
        graph.add_node("noop", lambda _s: {})
        graph.add_edge(START, "noop")
        graph.add_edge("noop", END)
        app = graph.compile()

        history = [
            HumanMessage(content=f"m{i}", id=str(i)) for i in range(6)
        ]
        for _ in range(2):
            updates, _resolved, _summary = asyncio.run(
                agent_graph._compact_messages_llm(history)
            )
            history = app.invoke({"messages": history})["messages"] + updates
            history = app.invoke({"messages": history})["messages"]

        summaries = [m for m in history if str(m.content).startswith("[Conversation summary]")]
        assert len(summaries) == 1


class TestEvidenceTrimming:
    """Finalize overflow is an evidence-payload problem, not a history problem."""

    def test_lowest_scoring_chunks_are_dropped_first(self):
        from app.services.agentic_rag.agent_graph import _trim_docs_to_budget

        docs = [
            {"page_content": "low " * 100, "_reranker_score": 0.1},
            {"page_content": "high " * 100, "_reranker_score": 9.0},
            {"page_content": "mid " * 100, "_reranker_score": 5.0},
        ]
        kept = _trim_docs_to_budget(docs, overflow_tokens=50)
        assert docs[0] not in kept
        assert docs[1] in kept

    def test_never_trims_to_an_empty_context(self):
        from app.services.agentic_rag.agent_graph import _trim_docs_to_budget

        docs = [{"page_content": "x " * 500, "_reranker_score": 0.1}]
        assert len(_trim_docs_to_budget(docs, overflow_tokens=10_000)) == 1


class TestQueryResolution:
    """Resolution must be conditional and provenance-bound."""

    def test_self_contained_query_passes_through_byte_for_byte(self):
        from app.services.agentic_rag.utils import resolve_retrieval_query

        query = "What is a mutex?"
        resolved, provenance = asyncio.run(resolve_retrieval_query(
            query=query,
            original_query=query,
            recent_history=[HumanMessage(content="tell me about Linux")],
            provenance_sources=["tell me about Linux"],
        ))
        assert resolved == query
        assert provenance["resolved"] is False
        assert provenance["reason"] == "self_contained"

    def test_no_history_means_no_resolver_call(self):
        from app.services.agentic_rag.utils import resolve_retrieval_query

        resolved, provenance = asyncio.run(resolve_retrieval_query(
            query="what are its limitations?",
            original_query="what are its limitations?",
            recent_history=[],
        ))
        assert resolved == "what are its limitations?"
        assert provenance["resolved"] is False

    def test_untraceable_terms_are_rejected(self):
        from app.services.agentic_rag.utils import validate_resolution_provenance

        ok, unsupported = validate_resolution_provenance(
            original_query="what about its limitations?",
            rewritten="limitations of Kubernetes autoscaling",
            provenance_sources=["User: tell me about the StreamVC paper"],
        )
        assert ok is False
        assert "kubernetes" in unsupported

    def test_terms_traceable_to_history_are_accepted(self):
        from app.services.agentic_rag.utils import validate_resolution_provenance

        ok, unsupported = validate_resolution_provenance(
            original_query="what about its limitations?",
            rewritten="limitations of the StreamVC model",
            provenance_sources=["User: tell me about the StreamVC model"],
        )
        assert ok is True
        assert unsupported == []

    def test_resolver_failure_falls_back_to_the_original_query(self, monkeypatch):
        from app.services.agentic_rag import utils

        async def _boom(**_kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr(utils, "_call_rewriter", _boom)
        resolved, provenance = asyncio.run(utils.resolve_retrieval_query(
            query="what about its limitations?",
            original_query="what about its limitations?",
            recent_history=[HumanMessage(content="tell me about StreamVC")],
            provenance_sources=["tell me about StreamVC"],
        ))
        assert resolved == "what about its limitations?"
        assert provenance["reason"].startswith("resolver_failed")


class TestContiguousOverlapPruning:
    """Neighbour lookup must use enumeration order, not dict equality."""

    def test_identical_chunk_dicts_do_not_confuse_the_neighbour_lookup(self):
        from app.services.agentic_rag.agent_graph import _prune_contiguous_overlaps

        shared = "ABCDEFGHIJ"
        docs = [
            {"page_content": shared, "metadata": {"document_id": 1, "chunk_index": 0}},
            {"page_content": shared, "metadata": {"document_id": 1, "chunk_index": 1}},
            {"page_content": shared + "TAIL", "metadata": {"document_id": 1, "chunk_index": 2}},
        ]
        pruned = _prune_contiguous_overlaps(docs)
        assert len(pruned) == 3
        assert pruned[0]["page_content"] == shared
