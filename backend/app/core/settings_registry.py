"""
settings_registry.py — Single source of metadata for all runtime-settable keys.

The registry drives:
  - API response shape (types, labels, categories, validation)
  - UI rendering (tabs, fields, types)
  - Settings resolution (2-tier precedence: org override → app DB value → registry default)

Adding a setting = one line here + one read in the consuming service.
No new table, no new migration for the schema itself (only seed data).
"""
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple


Scope = Literal["app", "org"]          # "app" = app-only; "org" = app-default + org-override
Reload = Literal["next_request", "restart", "ingest"]


_DEFAULT_CLASSIFIER_PROMPT = (
    "Classify this query into exactly one category. Respond with only the category name.\n\n"
    "Categories:\n"
    "FACTUAL — Questions asking for specific facts, definitions, or concrete information (e.g., 'What is RRF?', 'Define BM25 scoring')\n"
    "ENTITY_CENTRIC — Questions about specific entities, organizations, people, or products (e.g., 'What did Apple acquire?', 'Who founded Microsoft?')\n"
    "MULTI_PART — Questions comparing/contrasting things, asking for pros/cons, or requesting multiple aspects (e.g., 'Compare RRF and BM25', 'Pros and cons of vector databases')\n"
    "AMBIGUOUS — Vague, context-dependent, or incomplete queries (e.g., 'Tell me about that', 'What do you think?')\n\n"
    "Query: {query}\n\n"
    "Category: "
)

import json as _json

_DEFAULT_RETRIEVAL_PRESETS = _json.dumps({
    "FACTUAL": {
        "dense_weight": 0.5, "sparse_weight": 0.3, "exact_weight": 0.2,
        "top_k": 10
    },
    "ENTITY_CENTRIC": {
        "dense_weight": 0.6, "sparse_weight": 0.2, "exact_weight": 0.2,
        "top_k": 10
    },
    "MULTI_PART": {
        "dense_weight": 0.5, "sparse_weight": 0.5, "exact_weight": 0.0,
        "top_k": 10
    },
    "AMBIGUOUS": {
        "dense_weight": 0.4, "sparse_weight": 0.4, "exact_weight": 0.2,
        "top_k": 15
    }
})


@dataclass(frozen=True)
class SettingDef:
    key: str                           # canonical key, matches config.py attr name
    category: str                      # UI tab grouping
    label: str                         # human label
    value_type: Literal["str", "int", "float", "bool", "json", "text"]
    default: Any                       # config.py hardcoded default (Tier 0 fallback)
    scope: Scope                       # "app" = app-only; "org" = app-default + org-override
    reload: Reload = "next_request"    # when a change takes effect
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[Tuple[str, ...]] = None
    requires_reindex: bool = False
    secret: bool = False
    description: str = ""


# ── App-only settings (Super Admin; no org override) ──────────────────────
# These are app-only because of shared infrastructure, process singletons,
# or shared DataStore ingestion constraints.

_APP_ONLY = [
    # LLM & Models — process-global resources
    SettingDef("DENSE_EMBEDDINGS_MODEL", "LLM & Models", "Dense embeddings model",
               "str", "local-embedding-model", scope="app", reload="restart",
               description="Qdrant collections are dimension-locked; change requires reindex + restart."),
    SettingDef("DENSE_EMBEDDING_DIM", "LLM & Models", "Embedding dimension",
               "int", 1024, scope="app", reload="restart", min_value=1,
               description="Must match the embeddings model; tied to collection schema."),
    SettingDef("EMBEDDING_API_KEY", "LLM & Models", "Embeddings API key",
               "str", None, scope="app", reload="restart", secret=True,
               description="API key for dense embeddings. Falls back to .env OPENAI_API_KEY."),
    SettingDef("EMBEDDING_API_BASE", "LLM & Models", "Embeddings API base URL",
               "str", None, scope="app", reload="restart",
               description="Base URL for dense embeddings. Falls back to OPENAI_API_BASE."),
    SettingDef("VISION_MODEL", "LLM & Models", "Vision/OCR model",
               "str", None, scope="app", reload="ingest",
               description="Multimodal model for OCR during ingestion. Super admin only."),
    SettingDef("VISION_API_KEY", "LLM & Models", "Vision API key",
               "str", None, scope="app", reload="ingest", secret=True,
               description="API key for vision/OCR. Falls back to OPENAI_API_KEY."),
    SettingDef("OPENAI_VISION_API_BASE", "LLM & Models", "Vision API base URL",
               "str", None, scope="app", reload="ingest",
               description="Falls back to OPENAI_API_BASE when unset."),
    SettingDef("GRAPHRAG_LLM", "GraphRAG", "Graph extraction model",
               "str", None, scope="app", reload="ingest",
               description="LLM for graph extraction during ingestion. Super admin only."),
    SettingDef("GRAPHRAG_API_KEY", "GraphRAG", "Graph extraction API key",
               "str", None, scope="app", reload="ingest", secret=True,
               description="API key for graph extraction. Falls back to OPENAI_API_KEY."),
    SettingDef("GRAPHRAG_API_BASE", "GraphRAG", "Graph extraction API base URL",
               "str", None, scope="app", reload="ingest",
               description="Base URL for graph extraction. Falls back to OPENAI_API_BASE."),
    SettingDef("MEMORY_EMBEDDING_MODEL", "System", "Memory embedding model",
               "str", None, scope="app", reload="restart",
               description="Embedding model for Redis store; tied to global embeddings."),
    SettingDef("MEMORY_ENABLED", "System", "Enable Redis long-term memory",
               "bool", True, scope="app", reload="restart",
               description="Redis checkpointer singleton; restart required."),

    # Ingestion — shared DataStores mean these cannot differ per org
    SettingDef("CHUNK_SIZE", "Ingestion", "Chunk size (chars)",
               "int", 1500, scope="app", reload="ingest", min_value=100, max_value=8000,
               requires_reindex=True,
               description="DataStores are shared across orgs; per-org chunking would produce inconsistent indexes."),
    SettingDef("OVERLAP_PERCENTAGE", "Ingestion", "Overlap fraction",
               "float", 0.20, scope="app", reload="ingest", min_value=0.0, max_value=0.9,
               requires_reindex=True,
               description="Fraction of CHUNK_SIZE repeated at the start of the next chunk."),
    SettingDef("GRAPHRAG_ENABLED", "Ingestion", "Enable graph extraction",
               "bool", True, scope="app", reload="ingest",
               description="Graph extraction runs during ingestion of shared DataStores."),
    SettingDef("GRAPHRAG_MAX_CHUNKS", "Ingestion", "Max chunks for graph extraction",
               "int", 0, scope="app", reload="ingest", min_value=0,
               requires_reindex=True,
               description="0 = unlimited. Chunks beyond this are skipped for graph extraction."),
    SettingDef("NEO4J_LLM_CONTEXT", "Ingestion", "Graph extraction LLM context budget",
               "int", 12000, scope="app", reload="ingest", min_value=1000,
               requires_reindex=True,
               description="Char budget for extraction batches. ~3-4 chars per token."),

    # System — process-level services
    SettingDef("WATCHER_ENABLED", "System", "Enable file watcher",
               "bool", True, scope="app", reload="restart",
               description="One watcher process watches all DataStore folders."),
    SettingDef("WATCH_POLL_INTERVAL", "System", "Watcher poll interval (s)",
               "int", 2, scope="app", reload="restart", min_value=1,
               description="PollingObserver timeout in seconds."),
    SettingDef("SANDBOX_TIMEOUT_S", "System", "Sandbox timeout (s)",
               "int", 10, scope="app", reload="next_request", min_value=1,
               description="Code execution sandbox policy."),
    SettingDef("TOOL_CALL_MODE", "System", "Tool call protocol",
               "str", "auto", scope="app", reload="next_request",
               choices=("native", "json_text", "auto"),
               description="Agent protocol choice: native, json_text, or auto."),
    SettingDef("QUERY_CLASSIFIER_PROMPT", "Query Classification", "Classifier prompt template",
               "text", _DEFAULT_CLASSIFIER_PROMPT, scope="app", reload="next_request",
               description="Large template; should be consistent across orgs. Enable toggle is org-overridable."),
]


# ── App-default + org-override settings ───────────────────────────────────
# Super Admin sets the default (Tier 2); Admin may override per-org (Tier 3).

_ORG_OVERRIDABLE = [
    # LLM endpoints & model selection
    SettingDef("OPENAI_API_KEY", "LLM & Models", "API key",
               "str", None, scope="org", reload="next_request", secret=True,
               description="OpenAI-compatible API key for chat generation."),
    SettingDef("OPENAI_API_BASE", "LLM & Models", "Base API URL",
               "str", "http://localhost:1234/v1", scope="org", reload="next_request",
               description="OpenAI-compatible base URL."),
    SettingDef("OPENAI_MODEL", "LLM & Models", "Response model",
               "str", "local-model", scope="org", reload="next_request",
               description="Chat/response-generation model."),
    SettingDef("OPENAI_MODEL_CONTEXT_SIZE", "LLM & Models", "Context window size",
               "int", 131072, scope="org", reload="next_request", min_value=1024,
               description="Total context window of the chat model in tokens."),
    SettingDef("QUERY_MODEL", "LLM & Models", "Query rewrite model",
               "str", None, scope="org", reload="next_request",
               description="Falls back to OPENAI_MODEL when unset."),
    SettingDef("QUERY_API_KEY", "LLM & Models", "Query rewrite API key",
               "str", None, scope="org", reload="next_request", secret=True,
               description="API key for query rewriting. Falls back to OPENAI_API_KEY."),
    SettingDef("QUERY_API_BASE", "LLM & Models", "Query rewrite API base URL",
               "str", None, scope="org", reload="next_request",
               description="Base URL for query rewriting. Falls back to OPENAI_API_BASE."),
    SettingDef("REASONING_MODEL", "LLM & Models", "Reasoning model",
               "str", None, scope="org", reload="next_request",
               description="Falls back to OPENAI_MODEL when unset."),
    SettingDef("REASONING_API_KEY", "LLM & Models", "Reasoning API key",
               "str", None, scope="org", reload="next_request", secret=True,
               description="API key for reasoning. Falls back to OPENAI_API_KEY."),
    SettingDef("REASONING_API_BASE", "LLM & Models", "Reasoning API base URL",
               "str", None, scope="org", reload="next_request",
               description="Base URL for reasoning. Falls back to OPENAI_API_BASE."),

    # Retrieval tuning
    SettingDef("RETRIEVAL_TOP_K", "Retrieval", "Top-K",
               "int", 20, scope="org", reload="next_request", min_value=1, max_value=200),
    SettingDef("DENSE_MIN_SCORE", "Retrieval", "Dense min score",
               "float", 0.5, scope="org", reload="next_request", min_value=0.0),
    SettingDef("SPARSE_MIN_SCORE", "Retrieval", "Sparse min score",
               "float", 5.0, scope="org", reload="next_request", min_value=0.0,
               description="SPLADE term weight scale, not MySQL FTS scale."),
    SettingDef("EXACT_MIN_SCORE", "Retrieval", "Exact min score",
               "float", 0.5, scope="org", reload="next_request", min_value=0.0,
               description="MySQL FTS relevance scale (typically 0-3)."),
    SettingDef("HYBRID_DENSE_WEIGHT", "Retrieval", "Dense weight",
               "float", 0.5, scope="org", reload="next_request", min_value=0.0),
    SettingDef("HYBRID_SPARSE_WEIGHT", "Retrieval", "Sparse weight",
               "float", 0.3, scope="org", reload="next_request", min_value=0.0),
    SettingDef("HYBRID_EXACT_WEIGHT", "Retrieval", "Exact weight",
               "float", 0.2, scope="org", reload="next_request", min_value=0.0),
    SettingDef("RETRIEVAL_DENSE_ENABLED", "Retrieval", "Enable dense leg",
               "bool", True, scope="org", reload="next_request",
               description="Ingestion always indexes all legs."),
    SettingDef("RETRIEVAL_SPARSE_ENABLED", "Retrieval", "Enable sparse leg",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("RETRIEVAL_EXACT_ENABLED", "Retrieval", "Enable exact leg",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("RETRIEVAL_GRAPH_ENABLED", "Retrieval", "Enable graph leg",
               "bool", True, scope="org", reload="next_request",
               description="Graph retrieval leg toggle at query time. Ingestion unaffected."),
    SettingDef("RETRIEVAL_CONFIG_PRESETS", "Retrieval", "Per-query-type presets",
               "json", _DEFAULT_RETRIEVAL_PRESETS, scope="org", reload="next_request",
               description="JSON object: FACTUAL, ENTITY_CENTRIC, MULTI_PART, AMBIGUOUS presets."),
    SettingDef("ENTITY_AWARE_ENABLED", "Retrieval", "Enable entity-aware retrieval",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("ENTITY_BOOST_FACTOR", "Retrieval", "Entity boost factor",
               "float", 0.1, scope="org", reload="next_request", min_value=0.0,
               description="Score boost per entity mention. 0.1 = +10%."),

    # Adaptive retrieval
    SettingDef("ADAPTIVE_RETRIEVAL_ENABLED", "Adaptive Retrieval", "Enable adaptive retrieval",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("ADAPTIVE_RETRIEVAL_THRESHOLD", "Adaptive Retrieval", "Adaptive threshold",
               "float", 55, scope="org", reload="next_request", min_value=0.0, max_value=100.0,
               description="Confidence 0-100 below which adaptive expansion triggers."),
    SettingDef("ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD", "Adaptive Retrieval", "Adaptive reranker threshold",
               "float", -5.0, scope="org", reload="next_request",
               description="Must be lower than RERANKER_SCORE_THRESHOLD."),
    SettingDef("RETRIEVAL_RELAX_LEVEL2_RERANKER_THRESHOLD", "Adaptive Retrieval", "Level-2 relax threshold",
               "float", -8.0, scope="org", reload="next_request",
               description="Deepest relaxation tier for agentic rag_retrieve ladder."),

    # Reranker (model is app-only; enabled/threshold are org-overridable)
    SettingDef("RERANKER_ENABLED", "Reranker", "Enable reranker",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("RERANKER_SCORE_THRESHOLD", "Reranker", "Reranker score threshold",
               "float", -2.0, scope="org", reload="next_request"),

    # GraphRAG query-time
    SettingDef("GRAPHRAG_RETRIEVAL_HOPS", "GraphRAG", "Graph query hops",
               "int", 1, scope="org", reload="next_request", min_value=1, max_value=5),
    SettingDef("GRAPHRAG_RETRIEVAL_LIMIT", "GraphRAG", "Graph query limit",
               "int", 20, scope="org", reload="next_request", min_value=1),
    SettingDef("GRAPHRAG_ENTITY_FANOUT_CAP", "GraphRAG", "Entity fanout cap",
               "int", 50, scope="org", reload="next_request", min_value=1,
               description="Bounds hub-entity fan-out."),

    # Query classification (prompt is app-only; enable is org-overridable)
    SettingDef("QUERY_CLASSIFIER_ENABLED", "Query Classification", "Enable query classifier",
               "bool", True, scope="org", reload="next_request"),

    # Agentic features
    SettingDef("TOOL_CALLING_ENABLED", "Agentic", "Enable tool calling",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("SYNTHESIS_MODE_ENABLED", "Agentic", "Enable synthesis mode",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("AGENT_MAX_ITERATIONS", "Agentic", "Max agent iterations",
               "int", 8, scope="org", reload="next_request", min_value=1),
    SettingDef("AGENT_MAX_RETRIEVALS", "Agentic", "Max agent retrievals",
               "int", 3, scope="org", reload="next_request", min_value=0),
    SettingDef("AGENT_MAX_CODE_EXEC", "Agentic", "Max code executions",
               "int", 3, scope="org", reload="next_request", min_value=0),
    SettingDef("AGENT_MAX_REFLECTIONS", "Agentic", "Max reflections",
               "int", 2, scope="org", reload="next_request", min_value=0),
    SettingDef("AGENT_REFLECT_EVERY", "Agentic", "Reflect every N iterations",
               "int", 2, scope="org", reload="next_request", min_value=1),
    SettingDef("AGENT_MAX_TOOL_RETRIES", "Agentic", "Max tool retries",
               "int", 3, scope="org", reload="next_request", min_value=0),
    SettingDef("AGENT_RETRY_BACKOFF_BASE", "Agentic", "Retry backoff base (s)",
               "float", 0.5, scope="org", reload="next_request", min_value=0.0),
    SettingDef("AGENT_MAX_CLARIFICATIONS", "Agentic", "Max clarifications",
               "int", 1, scope="org", reload="next_request", min_value=0),
    SettingDef("AGENT_HISTORY_PAIRS", "Agentic", "Agent history pairs",
               "int", 3, scope="org", reload="next_request", min_value=0),
    SettingDef("AGENT_MAX_WALL_SECONDS", "Agentic", "Agent wall-clock budget (s)",
               "float", 120, scope="org", reload="next_request", min_value=1.0),

    # Historical memory
    SettingDef("HISTORICAL_MEMORY_ENABLED", "Memory", "Enable historical memory",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("HISTORICAL_MEMORY_TOP_K", "Memory", "Historical memory top-K",
               "int", 5, scope="org", reload="next_request", min_value=1),
    SettingDef("HISTORICAL_MEMORY_SCORE_THRESHOLD", "Memory", "Historical memory score threshold",
               "float", 2.0, scope="org", reload="next_request"),

    # Context, compaction & quality
    SettingDef("CONTEXT_RESERVED_GENERATION", "Context", "Reserved generation tokens",
               "int", 4096, scope="org", reload="next_request", min_value=256),
    SettingDef("CONTEXT_TOOL_BUDGET", "Context", "Tool context budget",
               "int", 8192, scope="org", reload="next_request", min_value=512),
    SettingDef("HIGHLIGHTS_TOKEN_CAP", "Context", "Highlights token cap",
               "int", 2000, scope="org", reload="next_request", min_value=100),
    SettingDef("CONTEXT_COMPACTION_TRIGGER_RATIO", "Context", "Compaction trigger ratio",
               "float", 0.85, scope="org", reload="next_request", min_value=0.1, max_value=1.0),
    SettingDef("COMPACTION_ENABLED", "Memory", "Enable compaction",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("COMPACTION_KEEP_RECENT", "Memory", "Compaction keep recent",
               "int", 10, scope="org", reload="next_request", min_value=1),
    SettingDef("COMPACTION_SUMMARY_MAX_CHARS", "Memory", "Compaction summary max chars",
               "int", 2000, scope="org", reload="next_request", min_value=100),
    SettingDef("ANSWER_QUALITY_GRADING_ENABLED", "Quality", "Enable answer grading",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("PROCESSING_TIMEOUT_SILENCE_S", "Quality", "Processing silence timeout (s)",
               "int", 300, scope="org", reload="next_request", min_value=10),
]


REGISTRY: list[SettingDef] = _APP_ONLY + _ORG_OVERRIDABLE

# Lookup by key for fast access
REGISTRY_BY_KEY: dict[str, SettingDef] = {d.key: d for d in REGISTRY}

# Keys that are org-overridable
ORG_OVERRIDABLE_KEYS: frozenset[str] = frozenset(d.key for d in _ORG_OVERRIDABLE)

# Keys that are app-only
APP_ONLY_KEYS: frozenset[str] = frozenset(d.key for d in _APP_ONLY)


def get_def(key: str) -> Optional[SettingDef]:
    """Return the SettingDef for a key, or None if unknown."""
    return REGISTRY_BY_KEY.get(key)


def is_org_overridable(key: str) -> bool:
    """Check if a key can be overridden at the org level."""
    return key in ORG_OVERRIDABLE_KEYS


def all_keys() -> list[str]:
    """Return all registered keys."""
    return [d.key for d in REGISTRY]
