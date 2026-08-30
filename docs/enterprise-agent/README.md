# Enterprise Agent — Architecture & Tools

> **Status**: Implemented. The agent loop, tool registry, RBAC, audit, and SSE streaming are all live in `backend/app/services/agentic_rag/`.

---

## 1. Executive summary

The system uses a LangGraph-based agent loop with 11 tools. The LLM autonomously decides which tool to call based on the current state, observes the result, and loops until the plan is satisfied or a budget is exhausted.

The infrastructure wraps the existing retrieval, reranking, memory, and chart systems as tools, adds a local code-execution sandbox, structured-data extraction, and KB exploration tools (grep/outline/read) for last-resort access to document content.

**Key design decisions:**
- JSON-text fallback for tool-calling (many local gateways don't support the OpenAI `tools` parameter)
- Per-tool RBAC re-checks and a tool-call audit log
- Per-turn tool-call budgets (configurable via settings registry)
- Parallel dispatch of independent tool calls within a single think iteration

---

## 2. Document index

| Doc | Covers |
|---|---|
| `02-target-architecture.md` | Agent loop topology, intent router, tool registry, tool-calling fallback, unified guardrail prompt, RBAC + audit, parallel subtask execution, memory/context model, SSE protocol. |
| `03-tool-specifications.md` | Contract for every tool the agent can call (inputs, outputs, RBAC, audit, caps). |
| `05-context-memory.md` | Token-based compaction, structured last-answer object, long-term recall, sliding window. |
| `pipeline-analysis.md` | Complete node-by-node topology, branches, and iterations of the agent loop. |

Read `02-target-architecture.md` for the design, `03-tool-specifications.md` for tool contracts, and `pipeline-analysis.md` for the runtime topology.
