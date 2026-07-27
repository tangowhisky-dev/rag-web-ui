# Enterprise Agent — Implementation Plan

> **Status**: Planning document. Supersedes the former `docs/agentic-architecture-v2.md` (deleted; it proposed a narrower tool-loop and was never implemented).
> **Constraint**: Offline. No internet access, no web search, no hosted model APIs. All models, tools, and sandboxes run inside the Docker Compose network against a local OpenAI-compatible gateway (LM Studio / Ollama / vLLM).
> **Scope**: Turn the existing RAG Web UI into a fully autonomous enterprise agent that handles (1) KB-grounded Q&A, (2) fluent multi-turn intent routing, (3) complex multi-subtask queries, (4) file/previous-answer actions, (5) automatic context-window management, and (6) chart/table generation.

---

## 1. Executive summary

The codebase already has a sophisticated LangGraph pipeline: 3-leg hybrid retrieval, cross-encoder reranking, Neo4j expansion, Redis checkpointer + semantic long-term memory, subtask decomposition with parallel/sequential execution, compaction, chart validation, and SSE streaming. The infrastructure is solid.

What it lacks is **agency**: the pipeline is a rigid sequence (`rewrite → classify → retrieve → generate → evaluate`). The agent cannot decide mid-turn to call a tool, observe the result, and call another. It cannot act *on* an attached file (only summarize it). It cannot compute or transform data before charting. It cannot reliably reference "the previous answer" as a structured object. Intent classification is a one-shot gate, not a continuous planner.

The plan replaces the rigid topology with a **tool-calling agent loop** that wraps the existing retrieval/memory/chart infrastructure as tools, adds a local code-execution sandbox and structured-data tools, and rebuilds intent routing, context management, and the chat UX around the loop. The stack (FastAPI + LangGraph + LangChain + Qdrant + Neo4j + Redis + MySQL backend; Next.js 14 + shadcn/ui + echarts frontend) stays. The work is in the loop, the tools, the memory, and the UX — not a framework swap.

**Offline-first constraints applied throughout:** (1) a JSON-text fallback for tool-calling because many local gateways (LM Studio/Ollama/vLLM) do not reliably support the OpenAI `tools` parameter; (2) a unified agent guardrail prompt enforcing offline-only behavior, citation-or-refuse, and agent bounds; (3) per-tool RBAC re-checks and a tool-call audit log so the autonomous agent is provably constrained to the user's accessible data; (4) parallel dispatch of independent subtasks so complex multi-part queries are not serialized.

---

## 2. Framework decisions (and why we keep the stack)

The user asked to evaluate backend and frontend framework choices. Conclusion: **keep everything, with targeted upgrades**. Rationale below.

### 2.1 Backend — keep FastAPI + LangGraph + LangChain

| Option | Verdict | Reason |
|---|---|---|
| **FastAPI** (current) | Keep | Async-native, Pydantic validation, OpenAPI docs, SSE streaming first-class. No benefit to switching to Litestar/Flask. |
| **LangGraph** (current) | Keep, upgrade usage | The right primitive for a stateful agent loop with checkpointer, interrupts (clarification), and parallel `Send()`. The current code uses it as a fixed DAG; the plan uses it as a true loop with tool nodes. |
| **LangChain** (current) | Keep for tool/LLM abstractions | `ChatOpenAI`, `BaseTool`, structured output, `RecursiveCharacterTextSplitter` are all in use. Avoid pulling in a second agent framework (CrewAI, AutoGen) — it would duplicate LangGraph's role and add offline-packaging risk. |
| LlamaIndex | Reject | Overlaps with LangChain/Qdrant usage; would force a second retrieval stack. |
| Custom agent loop (no framework) | Reject | Re-implements checkpointer, streaming, interrupts, parallelism that LangGraph already provides. |

**Target LangGraph version**: pin `langgraph>=0.2.50` (stable tool-node + interrupt APIs). Current `>=0.2.0` is too loose.

### 2.2 Frontend — keep Next.js 14 + shadcn/ui + echarts

| Option | Verdict | Reason |
|---|---|---|
| **Next.js 14 App Router** (current) | Keep | Server components for auth gating, API rewrites for backend proxy, file-based routing. Upgrading to 15/React 19 is optional and not required for this plan; 14.2 is stable. |
| **shadcn/ui + Tailwind 3** (current) | Keep | Component ownership (copy-paste) fits an offline enterprise app where the UI is customized per deployment. No migration to a packaged lib. |
| **echarts 6** (current) | Keep | Already wired for chart rendering. Add a structured-data → echarts-option builder on the backend so charts are generated from data, not free-form LLM JSON. |
| **mermaid** (current) | Keep | Diagram rendering already works. |
| State management (React Context) | Keep | Chat/KB context is fine as Context. The new agent-loop SSE events (`pl:`/`tc:`/`to:`/`la:`) are handled with a small `useReducer` in the chat page — no new state lib for v1. A `zustand` store can be added later if re-renders become a measured problem. |
| SSE via fetch ReadableStream (current) | Keep | `EventSource` cannot POST with auth cookies; the current fetch-stream approach is correct. |

### 2.3 Data layer — keep, no changes

Qdrant (vectors), Neo4j (graph), Redis (checkpointer + semantic memory), MySQL (metadata, chats, messages). All correct for offline enterprise. No swap.

### 2.4 New dependencies (all offline-installable)

| Dependency | Purpose | Risk |
|---|---|---|
| `RestrictedPython` | Local Python sandbox for `code_execute` tool (v1) | Pure-Python, pip-installable offline, AST-level restrictions. Sufficient for the internal-user threat model. `nsjail` is a later hardening for hostile multi-tenant deployments, not a v1 requirement. |
| `pandas`, `openpyxl`, `lxml` | Structured-data tool (CSV/Excel/HTML table extraction) | `pandas` already transitively present via `pyarrow`; add explicitly. |
| `matplotlib` (optional) | Fallback chart renderer if echarts JSON fails | Already in eval harness; add to backend. |
| `tiktoken` or `transformers` tokenizer | Token-accurate context budgeting (current code estimates tokens) | Offline; matches the deployed model's tokenizer where possible. |

---

## 3. Document index

| Doc | Covers |
|---|---|
| `01-gap-analysis.md` | Current state vs the 5 requirements; what works, what's missing, with file:line evidence. |
| `02-target-architecture.md` | Agent loop topology, intent router, tool registry, tool-calling fallback for offline LLMs, unified guardrail prompt, RBAC + audit, parallel subtask execution, memory/context model, SSE protocol v4. |
| `03-tool-specifications.md` | Contract for every tool the agent can call (inputs, outputs, streaming, errors, RBAC, audit). |
| `04-implementation-plan.md` | Backend and frontend changes at file/function granularity; migration from current pipeline. |
| `05-context-memory.md` | Token-based compaction, structured last-answer object, long-term recall, sliding window. |
| `06-roadmap.md` | Phased milestones, verification per phase, risks, rollback strategy. |

Read `01-gap-analysis.md` first if you want the evidence base; read `02-target-architecture.md` first if you want the design.
