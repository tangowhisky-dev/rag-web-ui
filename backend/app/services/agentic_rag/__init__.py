"""Autonomous Agentic Agent - LangGraph-powered pipeline.

Public API:
  run_agentic_rag() - async generator that streams SSE events

The agent operates via a LangGraph StateGraph with nested subgraph architecture.
Routes between the existing generator-based pipeline and the new LangGraph
pipeline via the USE_LANGGRAPH feature flag.

The agent operates as a single autonomous pipeline:
1. Rewrite query using chat history
2. Decide simple vs complex (LLM-based structured classification)
3. For simple: direct search -> stream answer
4. For complex: decompose -> iterate subtasks -> stream each -> synthesize
5. All tokens, progress, and thinking traces stream in real-time

LangGraph components:
  graph_state.py  - AgentState with accumulator reducers
  graph.py        - Main graph compilation
  nodes.py        - Node implementations (rewrite, classify, retrieve, generate)
  edges.py        - Conditional routing logic
  tools.py        - LangChain tool definitions
  schemas.py      - Pydantic models (QueryAnalysis)
  callbacks.py    - SSE event bridge
  graph_runner.py - Graph execution bridging to SSE protocol

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
