"""Autonomous Agentic Agent — LangGraph-powered pipeline.

Public API:
  run_agentic_rag() - async generator that streams SSE events

The agent operates via a loop of LangGraph nodes:
1. Load conversation context (load_context)
2. Resolve the retrieval query from chat history (rewrite_query_node)
3. Plan the reasoning steps (plan_node)
4. Think / reason through each step (think_node)
5. Execute tools, including retrieval (tool_node)
6. Reflect on tool results (reflect_node)
7. Finalize and stream the answer (finalize_node)

Context compaction is not a node: it runs as a budget guard immediately
before any LLM call with variable-length context (think, finalize).

All tokens, progress, thinking traces, tool calls, and final answers stream in real-time.

LangGraph components:
  agent_graph.py   - Main agent graph definition and node wiring
  agent_runner.py  - Graph execution runner
  graph_state.py   - AgentState with accumulator reducers
  nodes.py         - Node implementations (query resolution, retrieval, evaluation, etc.)
  prompts.py       - System/user prompts for planning, reasoning, and evaluation
  schemas.py       - Pydantic models for state and tool schemas
  streaming.py     - v3 stream transformer to SSE events
  utils.py         - Helper functions (token estimation, formatting)
  token_budget.py  - Context-window budget management
  redis_memory.py  - Redis-backed checkpoint memory
  evaluator.py     - Answer evaluation helpers
  llm_factory.py   - LLM client construction
  tools/           - Tool implementations (RAG retrieval, file tools, etc.)

SSE Event Protocol:
  p:  progress         - transient status messages
  t:  task_list        - subtask list with status
  th: thinking         - reasoning model chain-of-thought
  0:  token            - streaming answer text
  1:  rewritten_query  - standalone query
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
