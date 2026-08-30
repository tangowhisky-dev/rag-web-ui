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
    model_picker: bool = False         # render as combobox with "fetch models" button
    api_base_ref: Optional[str] = None # setting key for the associated API base URL
    api_key_ref: Optional[str] = None  # setting key for the associated API key


# ── App-only settings (Super Admin; no org override) ──────────────────────
# These are app-only because of shared infrastructure, process singletons,
# or shared DataStore ingestion constraints.

_APP_ONLY = [
    # Embeddings — process-global, dimension-locked
    SettingDef("EMBEDDING_API_BASE", "Embeddings", "Embeddings API base URL",
               "str", None, scope="app", reload="restart",
               description="Base URL for dense embeddings. Falls back to OPENAI_API_BASE."),
    SettingDef("EMBEDDING_API_KEY", "Embeddings", "Embeddings API key",
               "str", None, scope="app", reload="restart", secret=True,
               description="API key for dense embeddings. Falls back to .env OPENAI_API_KEY."),
    SettingDef("DENSE_EMBEDDINGS_MODEL", "Embeddings", "Dense embeddings model",
               "str", "local-embedding-model", scope="app", reload="restart",
               description="Qdrant collections are dimension-locked; change requires reindex + restart.",
               model_picker=True, api_base_ref="EMBEDDING_API_BASE", api_key_ref="EMBEDDING_API_KEY"),
    SettingDef("DENSE_EMBEDDING_DIM", "Embeddings", "Embedding dimension",
               "int", 1024, scope="app", reload="restart", min_value=1,
               description="Must match the embeddings model; tied to collection schema."),
    # Vision / OCR — ingestion-time, super admin only
    SettingDef("OPENAI_VISION_API_BASE", "Vision / OCR", "Vision API base URL",
               "str", None, scope="app", reload="ingest",
               description="Falls back to OPENAI_API_BASE when unset."),
    SettingDef("VISION_API_KEY", "Vision / OCR", "Vision API key",
               "str", None, scope="app", reload="ingest", secret=True,
               description="API key for vision/OCR. Falls back to OPENAI_API_KEY."),
    SettingDef("VISION_MODEL", "Vision / OCR", "Vision/OCR model",
               "str", None, scope="app", reload="ingest",
               description="Multimodal model for OCR during ingestion. Super admin only.",
               model_picker=True, api_base_ref="OPENAI_VISION_API_BASE", api_key_ref="VISION_API_KEY"),
    SettingDef("VISION_MAX_TOKENS", "Vision / OCR", "Max output tokens per page/image",
               "int", 4096, scope="app", reload="ingest", min_value=100,
               description="Caps the vision model's output per OCR call (one page or one image). "
                           "Prevents runaway generation on hallucinated or verbose responses. "
                           "A single page typically needs 500-2000 tokens."),
    SettingDef("MARKDOWN_ENGINE", "Ingestion", "Markdown conversion engine",
               "str", "anydoc", scope="app", reload="ingest",
               choices=("anydoc", "markitdown"),
               description="Engine for document→markdown conversion. 'anydoc' = anydoc+pdf-inspector (default), 'markitdown' = legacy."),
    SettingDef("GRAPHRAG_API_BASE", "GraphRAG", "Graph extraction API base URL",
               "str", None, scope="app", reload="ingest",
               description="Base URL for graph extraction. Falls back to OPENAI_API_BASE."),
    SettingDef("GRAPHRAG_API_KEY", "GraphRAG", "Graph extraction API key",
               "str", None, scope="app", reload="ingest", secret=True,
               description="API key for graph extraction. Falls back to OPENAI_API_KEY."),
    SettingDef("GRAPHRAG_LLM", "GraphRAG", "Graph extraction model",
               "str", None, scope="app", reload="ingest",
               description="LLM for graph extraction during ingestion. Super admin only.",
               model_picker=True, api_base_ref="GRAPHRAG_API_BASE", api_key_ref="GRAPHRAG_API_KEY"),
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
    SettingDef("INGESTION_CONCURRENCY", "Ingestion", "Ingestion concurrency",
               "int", 8, scope="app", reload="restart", min_value=1, max_value=32,
               description="Max concurrent document ingestion tasks (convert → embed → store). "
                           "Raise for fast networks / local SSD; lower for slow CIFS/SMB mounts."),

    # System — process-level services
    SettingDef("WATCHER_ENABLED", "System", "Enable file watcher",
               "bool", True, scope="app", reload="restart",
               description="One watcher process watches all DataStore folders."),
    SettingDef("WATCH_POLL_INTERVAL", "System", "Watcher poll interval (s)",
               "int", 2, scope="app", reload="restart", min_value=1,
               description="PollingObserver timeout. For CIFS/SMB mounts, "
                           "use 30-60s to reduce network stat traffic."),
    SettingDef("TOOL_CALL_MODE", "System", "Tool call protocol",
               "str", "auto", scope="app", reload="next_request",
               choices=("native", "json_text", "auto"),
               description="Agent protocol choice: native, json_text, or auto."),
]


# ── App-default + org-override settings ───────────────────────────────────
# Super Admin sets the default (Tier 2); Admin may override per-org (Tier 3).

_ORG_OVERRIDABLE = [
    # Response Model — primary chat/response generation
    SettingDef("OPENAI_API_BASE", "Response Model", "Base API URL",
               "str", "http://localhost:1234/v1", scope="org", reload="next_request",
               description="OpenAI-compatible base URL."),
    SettingDef("OPENAI_API_KEY", "Response Model", "API key",
               "str", None, scope="org", reload="next_request", secret=True,
               description="OpenAI-compatible API key for chat generation."),
    SettingDef("OPENAI_MODEL", "Response Model", "Response model",
               "str", "local-model", scope="org", reload="next_request",
               description="Chat/response-generation model.",
               model_picker=True, api_base_ref="OPENAI_API_BASE", api_key_ref="OPENAI_API_KEY"),
    SettingDef("OPENAI_MODEL_CONTEXT_SIZE", "Response Model", "Context window size",
               "int", 131072, scope="org", reload="next_request", min_value=1024,
               description="Total context window of the chat model in tokens."),
    # Query Rewrite Model — falls back to Response Model
    SettingDef("QUERY_API_BASE", "Query Rewrite Model", "Query rewrite API base URL",
               "str", None, scope="org", reload="next_request",
               description="Base URL for query rewriting. Falls back to OPENAI_API_BASE."),
    SettingDef("QUERY_API_KEY", "Query Rewrite Model", "Query rewrite API key",
               "str", None, scope="org", reload="next_request", secret=True,
               description="API key for query rewriting. Falls back to OPENAI_API_KEY."),
    SettingDef("QUERY_MODEL", "Query Rewrite Model", "Query rewrite model",
               "str", None, scope="org", reload="next_request",
               description="Falls back to OPENAI_MODEL when unset.",
               model_picker=True, api_base_ref="QUERY_API_BASE", api_key_ref="QUERY_API_KEY"),
    # Reasoning Model — falls back to Response Model
    SettingDef("REASONING_API_BASE", "Reasoning Model", "Reasoning API base URL",
               "str", None, scope="org", reload="next_request",
               description="Base URL for reasoning. Falls back to OPENAI_API_BASE."),
    SettingDef("REASONING_API_KEY", "Reasoning Model", "Reasoning API key",
               "str", None, scope="org", reload="next_request", secret=True,
               description="API key for reasoning. Falls back to OPENAI_API_KEY."),
    SettingDef("REASONING_MODEL", "Reasoning Model", "Reasoning model",
               "str", None, scope="org", reload="next_request",
               description="Falls back to OPENAI_MODEL when unset.",
               model_picker=True, api_base_ref="REASONING_API_BASE", api_key_ref="REASONING_API_KEY"),

    # Retrieval tuning
    SettingDef("RETRIEVAL_TOP_K", "Retrieval", "Top-K",
               "int", 20, scope="org", reload="next_request", min_value=1, max_value=200,
               description="Candidate chunks fetched per retrieval leg before RRF fusion and reranking. Higher = more recall, more noise."),
    SettingDef("DENSE_MIN_SCORE", "Retrieval", "Dense min score",
               "float", 0.5, scope="org", reload="next_request", min_value=0.0,
               description="Minimum cosine similarity for dense vector results. Lower accepts more semantic matches; adaptive retrieval relaxes this on retries."),
    SettingDef("SPARSE_MIN_SCORE", "Retrieval", "Sparse min score",
               "float", 5.0, scope="org", reload="next_request", min_value=0.0,
               description="SPLADE term weight scale, not MySQL FTS scale."),
    SettingDef("EXACT_MIN_SCORE", "Retrieval", "Exact min score",
               "float", 0.5, scope="org", reload="next_request", min_value=0.0,
               description="MySQL FTS relevance scale (typically 0-3)."),
    SettingDef("QDRANT_MMR_DIVERSITY", "Retrieval", "Qdrant MMR diversity",
               "float", 0.3, scope="org", reload="next_request", min_value=0.0, max_value=1.0,
               description="0.0 = pure relevance (no MMR), 1.0 = pure diversity. "
                           "Applied to dense and sparse Qdrant legs via native MMR."),
    SettingDef("DEDUP_SEMANTIC_THRESHOLD", "Retrieval", "Semantic dedup threshold",
               "float", 0.95, scope="org", reload="next_request", min_value=0.0, max_value=1.0,
               description="Cosine similarity above which chunks from different documents "
                           "are considered near-duplicates; the one from the latest "
                           "modified_at document is kept. 1.0 = disabled."),
    SettingDef("RETRIEVAL_DENSE_ENABLED", "Retrieval", "Enable dense leg",
               "bool", True, scope="org", reload="next_request",
               description="Ingestion always indexes all legs."),
    SettingDef("RETRIEVAL_SPARSE_ENABLED", "Retrieval", "Enable sparse leg",
               "bool", True, scope="org", reload="next_request",
               description="SPLADE keyword search leg. Disabling skips keyword matching at query time; ingestion still indexes sparse vectors."),
    SettingDef("RETRIEVAL_EXACT_ENABLED", "Retrieval", "Enable exact leg",
               "bool", True, scope="org", reload="next_request",
               description="MySQL FULLTEXT exact-match leg. Disabling skips exact term matching at query time; ingestion still indexes for FTS."),
    SettingDef("RETRIEVAL_GRAPH_ENABLED", "Retrieval", "Enable graph leg",
               "bool", True, scope="org", reload="next_request",
               description="Graph retrieval leg toggle at query time. Ingestion unaffected."),
    SettingDef("ENTITY_AWARE_ENABLED", "Retrieval", "Enable entity-aware retrieval",
               "bool", True, scope="org", reload="next_request",
               description="Extracts entities from the query, expands via Neo4j graph, and boosts chunks mentioning those entities."),
    SettingDef("ENTITY_BOOST_FACTOR", "Retrieval", "Entity boost factor",
               "float", 0.1, scope="org", reload="next_request", min_value=0.0,
               description="Score boost per entity mention. 0.1 = +10%."),

    # Adaptive retrieval
    SettingDef("ADAPTIVE_RETRIEVAL_ENABLED", "Adaptive Retrieval", "Enable adaptive retrieval",
               "bool", True, scope="org", reload="next_request",
               description="Retries retrieval with progressively looser score thresholds when initial results are insufficient."),
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
               "bool", True, scope="org", reload="next_request",
               description="Cross-encoder re-scoring of retrieval candidates after RRF fusion. Disabling skips reranking; results use raw fusion scores."),
    SettingDef("RERANKER_SCORE_THRESHOLD", "Reranker", "Reranker score threshold",
               "float", -2.0, scope="org", reload="next_request",
               description="Minimum cross-encoder logit to pass reranking. Lower = more results pass; adaptive retrieval uses progressively lower thresholds on retries."),

    # GraphRAG query-time
    SettingDef("GRAPHRAG_RETRIEVAL_HOPS", "GraphRAG", "Graph query hops",
               "int", 1, scope="org", reload="next_request", min_value=1, max_value=5,
               description="Relationship hops traversed in Neo4j during graph expansion. Higher = more distant connections found, but slower queries."),
    SettingDef("GRAPHRAG_RETRIEVAL_LIMIT", "GraphRAG", "Graph query limit",
               "int", 20, scope="org", reload="next_request", min_value=1,
               description="Maximum chunks returned from graph expansion. Caps related-chunk additions to retrieval results."),
    SettingDef("GRAPHRAG_ENTITY_FANOUT_CAP", "GraphRAG", "Entity fanout cap",
               "int", 50, scope="org", reload="next_request", min_value=1,
               description="Bounds hub-entity fan-out."),

    # Agentic features
    SettingDef("AGENT_MAX_ITERATIONS", "Agentic", "Max agent iterations",
               "int", 8, scope="org", reload="next_request", min_value=1,
               description="Hard cap on think-act-observe cycles. When reached, the agent finalizes with whatever it has."),
    SettingDef("AGENT_MAX_RETRIEVALS", "Agentic", "Max agent retrievals",
               "int", 3, scope="org", reload="next_request", min_value=0,
               description="Cap on rag_retrieve tool calls per turn. Prevents retrieval loops; exceeded calls return an error."),
    SettingDef("AGENT_MAX_CODE_EXEC", "Agentic", "Max code executions",
               "int", 3, scope="org", reload="next_request", min_value=0,
               description="Cap on code_execute tool calls per turn. Prevents infinite code execution loops."),
    SettingDef("AGENT_MAX_KB_GREP", "Agentic", "Max KB grep calls",
               "int", 5, scope="org", reload="next_request", min_value=0,
               description="Cap on kb_grep tool calls per turn. Prevents excessive grep loops."),
    SettingDef("AGENT_MAX_KB_READ", "Agentic", "Max KB read/outline calls",
               "int", 10, scope="org", reload="next_request", min_value=0,
               description="Combined cap on kb_read + kb_outline tool calls per turn."),
    SettingDef("AGENT_REFLECT_EVERY", "Agentic", "Reflect every N iterations",
               "int", 2, scope="org", reload="next_request", min_value=1,
               description="Runs the reflect node every N iterations for mid-loop recovery and replanning checks."),
    SettingDef("AGENT_MAX_TOOL_RETRIES", "Agentic", "Max tool retries",
               "int", 3, scope="org", reload="next_request", min_value=0,
               description="Maximum retry attempts for failed tool calls. Uses exponential backoff for transient errors."),
    SettingDef("AGENT_RETRY_BACKOFF_BASE", "Agentic", "Retry backoff base (s)",
               "float", 0.5, scope="org", reload="next_request", min_value=0.0,
               description="Base delay in seconds for exponential backoff. Retry delay = base × 2^attempt."),
    SettingDef("AGENT_MAX_CLARIFICATIONS", "Agentic", "Max clarifications",
               "int", 1, scope="org", reload="next_request", min_value=0,
               description="Maximum clarification rounds before the agent proceeds without asking further."),
    SettingDef("AGENT_HISTORY_PAIRS", "Agentic", "Agent history pairs",
               "int", 3, scope="org", reload="next_request", min_value=0,
               description="Conversation pairs (user+assistant) included in agent context. Higher = more multi-turn context, more tokens."),
    SettingDef("AGENT_MAX_WALL_SECONDS", "Agentic", "Agent wall-clock budget (s)",
               "float", 120, scope="org", reload="next_request", min_value=1.0,
               description="Wall-clock time limit for the agent loop. When exceeded, the agent is forced to finalize."),
    SettingDef("GENERATION_TEMPERATURE", "Agentic", "Answer generation temperature",
               "float", 0.7, scope="org", reload="next_request", min_value=0.0, max_value=2.0,
               description="Temperature for final answer generation. Higher = more creative; lower = more deterministic."),

    # Context, compaction & quality
    SettingDef("CONTEXT_RESERVED_GENERATION", "Context", "Reserved generation tokens",
               "int", 4096, scope="org", reload="next_request", min_value=256,
               description="Tokens reserved for the model's response. Available context = context window − this − tool budget. Increase if answers get truncated."),
    SettingDef("CONTEXT_TOOL_BUDGET", "Context", "Tool context budget",
               "int", 8192, scope="org", reload="next_request", min_value=512,
               description="Tokens reserved for tool outputs (retrieval results, file reads). Ensures room for tool results in long conversations."),
    SettingDef("CONTEXT_COMPACTION_TRIGGER_RATIO", "Context", "Compaction trigger ratio",
               "float", 0.85, scope="org", reload="next_request", min_value=0.1, max_value=1.0,
               description="Fraction of available context at which compaction triggers. 0.85 = compact at 85% usage. Lower = more aggressive compaction."),
    SettingDef("COMPACTION_ENABLED", "Memory", "Enable compaction",
               "bool", True, scope="org", reload="next_request",
               description="Summarizes old conversation history when context budget is exceeded. Disabling risks context overflow on long conversations."),
    SettingDef("COMPACTION_KEEP_RECENT", "Memory", "Compaction keep recent",
               "int", 10, scope="org", reload="next_request", min_value=1,
               description="Recent messages kept verbatim during compaction. Older messages are summarized into a compact form."),
    SettingDef("COMPACTION_SUMMARY_MAX_CHARS", "Memory", "Compaction summary max chars",
               "int", 2000, scope="org", reload="next_request", min_value=100,
               description="Maximum character length for the LLM-generated conversation summary. Longer summaries preserve more detail but use more context."),
    SettingDef("ANSWER_QUALITY_GRADING_ENABLED", "Quality", "Enable answer grading",
               "bool", True, scope="org", reload="next_request",
               description="Grades answers on faithfulness, completeness, and quality after generation. Adds one LLM call per answer."),
    SettingDef("PROCESSING_TIMEOUT_SILENCE_S", "Quality", "Processing silence timeout (s)",
               "int", 300, scope="app", reload="next_request", min_value=10,
               description="Warns if ingestion processing shows no progress for this many seconds. Non-fatal — used for monitoring stalled tasks."),
    SettingDef("ABBREVIATION_EXPANSION_ENABLED", "Retrieval", "Abbreviation expansion",
               "bool", True, scope="org", reload="next_request", requires_reindex=True,
               description="Enable suffix expansion of abbreviations during ingestion and query. Requires re-ingestion of existing documents when toggled."),
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
