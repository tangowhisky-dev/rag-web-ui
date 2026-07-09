"""Autonomous Agentic Agent - single agent, real-time streaming.

Public API:
  run_agentic_rag() - async generator that streams SSE events

The agent operates as a single autonomous pipeline:
1. Rewrite query using chat history
2. Decide simple vs complex (heuristic)
3. For simple: direct search -> stream answer
4. For complex: decompose -> iterate subtasks -> stream each -> synthesize
5. All tokens, progress, and thinking traces stream in real-time

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
