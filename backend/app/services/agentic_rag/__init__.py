"""Autonomous Agentic Agent — LangGraph-powered pipeline.

Public API:
  run_agentic_rag() - async generator that streams SSE events

The agent operates via LangGraph StateGraph nodes:
1. Rewrite query using chat history (rewrite_query_node)
2. Classify query (classify_query_node)
3. For simple queries: direct retrieval (direct_retrieval_node) → stream answer
4. For complex queries: subtask decomposition → iterate retrieval/generation per subtask → synthesize
5. All tokens, progress, and thinking traces stream in real-time

LangGraph components:
  graph_state.py  - AgentState with accumulator reducers
  nodes.py        - Node implementations (rewrite, classify, retrieve, generate, etc.)
  callbacks.py    - SSE event bridge
  schemas.py      - Pydantic models (QueryAnalysis)
  utils.py        - Helper functions (token estimation, formatting)
  graph_runner.py - Pipeline execution routing nodes to SSE output

SSE Event Protocol:
  p:  progress       - transient status messages
  t:  task_list      - subtask list with status
  th: thinking       - reasoning model chain-of-thought
  0:  token          - streaming answer text
  1:  rewritten_query - standalone query
  2:  context        - retrieved documents
  3:  error          - exception message
  d:  done           - finish reason + usage
"""

from .pipeline import run_agentic_rag

__all__ = ["run_agentic_rag"]
