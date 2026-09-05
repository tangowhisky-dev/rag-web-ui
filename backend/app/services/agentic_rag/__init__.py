"""Autonomous Agentic Agent — LangGraph-powered pipeline.

Public API:
  run_agentic_rag() - async generator that streams SSE events

The agent operates via a loop of LangGraph nodes:
1. Load conversation context (load_context)
2. Plan the reasoning steps (plan_node)
3. Think / reason through each step (think_node)
4. Execute atomic search/read tools (tool_node)
5. Check sufficiency (sufficiency_check)
6. Finalize and stream the answer (finalize_node)
7. Score answer quality (answer_scoring)
8. Save memory (save_memory)

Context compaction is not a node: it runs as a budget guard immediately
before any LLM call with variable-length context (think, finalize).

All tokens, progress, thinking traces, tool calls, and final answers stream in real-time.

LangGraph components:
  agent_graph/     - Graph builder, node implementations, sufficiency check,
                     execution check, planning, thinking, tooling, finalization,
                     reflection (answer scoring + clarification), observations
  agent_runner.py  - Graph execution runner
  graph_state.py   - AgentState with accumulator reducers
  nodes.py         - Shared node helpers (agent_step, history, LLM factory, evaluation)
  prompts.py       - System/user prompts for planning, reasoning, and evaluation
  schemas.py       - Pydantic models for state and tool schemas (CitationRef, Subtask, etc.)
  streaming.py     - v3 stream transformer to SSE events
  utils.py         - Helper functions (context formatting, citation normalization)
  token_budget.py  - Context-window budget management
  redis_memory.py  - Redis-backed checkpoint memory
  evaluator.py     - Answer evaluation helpers
  llm_factory.py   - LLM client construction
  tools/           - Atomic tool implementations (search_exact, search_sparse,
                     search_dense, rerank_results, graph_expand, kb_read, kb_grep,
                     kb_outline, kb_search_documents, kb_metadata, code_execute,
                     chart_generate, extract_data, summarize_answer, current_datetime,
                     file_read, file_summarize, file_extract_table)

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
