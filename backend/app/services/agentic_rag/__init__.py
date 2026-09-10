"""Autonomous Agentic Agent — LangGraph-powered pipeline.

Public API:
  run_agentic_rag() - async generator that streams SSE events

The v2 agent operates via a lean think ⇄ tool loop:
1. Load conversation context (load_context)
2. Think: one LLM call with all tools bound — emits tool calls or final answer
3. Tool: dispatch tool calls in parallel, record observations
4. Post-process: generate/finalize answer, score quality, save memory

All tokens, progress, thinking traces, tool calls, and final answers stream in real-time
via the unified timeline event protocol (tl: SSE events).

LangGraph components:
  agent_graph/       - Shared modules (helpers, tooling, observations, compaction,
                       finalization, load_context)
  agent_graph_v2/    - v2 graph builder, think node, tool node, post-process node
  agent_runner_v2.py - v2 graph execution runner
  graph_state.py     - AgentState with accumulator reducers
  nodes.py           - Shared node helpers (agent_step, history, LLM factory, evaluation)
  prompts_v2.py      - System/user prompts for v2 agent
  schemas.py         - Pydantic models for state and tool schemas (CitationRef, Subtask, etc.)
  retrieval_subagent.py - Parallel retrieval sub-agent
  office_subagent.py    - Office document generation sub-agent
  utils.py         - Helper functions (context formatting, citation normalization)
  token_budget.py  - Context-window budget management
  redis_memory.py  - Redis-backed checkpoint memory
  evaluator.py     - Answer evaluation helpers
  llm_factory.py   - LLM client construction
  tools/           - Tool implementations (keyword_search, semantic_search,
                     graph_expand, file_read, kb_grep,
                     kb_outline, title_search, kb_metadata, code_execute,
                     chart_generate, extract_data, summarize, current_datetime,
                     file_extract_table)

SSE Event Protocol:
  p:  progress         - transient status messages
  t:  task_list        - subtask list with status
  th: thinking         - reasoning model chain-of-thought
  0:  token            - streaming answer text
  2:  context          - retrieved documents
  3:  error            - exception message
  pl: plan             - agent subtask plan
  tc: tool_call        - tool invocation
  to: tool_observation - tool result
  la: last_answer      - structured summary + chart option
  r:  answer_rewrite   - citation-normalised full answer + cited docs
  c:  interrupt        - human-in-the-loop clarification request
  4:  agent_step       - per-node step status
  d:  done             - finish reason + usage
"""

from .pipeline import run_agentic_rag

__all__ = ["run_agentic_rag"]
