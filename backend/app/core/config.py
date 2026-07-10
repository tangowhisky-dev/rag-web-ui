import json
import os
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "InsightCore"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # MySQL
    MYSQL_SERVER: str = os.getenv("MYSQL_SERVER", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "ragwebui")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "ragwebui")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "ragwebui")
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @property
    def get_database_url(self) -> str:
        if self.SQLALCHEMY_DATABASE_URI:
            return self.SQLALCHEMY_DATABASE_URI
        return (
            f"mysql+mysqlconnector://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "360"))

    # File storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")

    # Watcher — per-org directory watch settings (defaults in Organisation model)
    WATCH_DIR: str = os.getenv("WATCH_DIR", "/app/uploads")  # legacy: default fallback
    WATCH_POLL_INTERVAL: int = int(os.getenv("WATCH_POLL_INTERVAL", "2"))  # seconds between scans
    WATCHER_ENABLED: bool = os.getenv("WATCHER_ENABLED", "true").lower() == "true"
    # Use inotify (Linux native) instead of polling observer.
    # inotify provides near-instant event delivery on ext4/xfs with Docker bind-mounts.
    # Falls back to PollingObserver on macOS, Windows, or when inotify is unavailable.
    WATCHER_USE_INOTIFY: bool = os.getenv("WATCHER_USE_INOTIFY", "true").lower() == "true"

    # LLM + Embeddings (OpenAI-compatible)
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "lmstudio")

    # Chat / response-generation model
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "local-model")

    # Total context window size of OPENAI_MODEL in tokens.
    # 25% is reserved for injected chat-file content.
    OPENAI_MODEL_CONTEXT_SIZE: int = int(os.getenv("OPENAI_MODEL_CONTEXT_SIZE", "131072"))

    # Query-rewriting model (used for standalone-question condensation and
    # rolling-summary generation). Falls back to OPENAI_MODEL when unset.
    # A smaller/faster model works well here — the task is mechanical rewording,
    # not complex reasoning.
    QUERY_MODEL: Optional[str] = os.getenv("QUERY_MODEL") or None

    # Vision model for OCR of embedded images (scanned PDFs, images in DOCX/
    # PPTX/XLSX). Must be a multimodal (vision-capable) model.
    # When unset, markitdown-ocr is loaded without an llm_client and OCR is
    # silently skipped — behaviour identical to before.
    VISION_MODEL: Optional[str] = os.getenv("VISION_MODEL") or None

    # Optional separate base URL for the vision model. When unset, falls back
    # to OPENAI_API_BASE (same server as chat/embeddings).
    OPENAI_VISION_API_BASE: Optional[str] = os.getenv("OPENAI_VISION_API_BASE") or None

    # Reasoning / thinking model — used in "Thinking" answering mode.
    # Should be a CoT/reasoning-enabled model (e.g. qwq-32b, deepseek-r1).
    # Falls back to OPENAI_MODEL when unset.
    REASONING_MODEL: Optional[str] = os.getenv("REASONING_MODEL") or None

    DENSE_EMBEDDINGS_MODEL: str = os.getenv("DENSE_EMBEDDINGS_MODEL", "local-embedding-model")
    # Dimension of the dense embedding model output. Must match DENSE_EMBEDDINGS_MODEL.
    # qwen3-embedding-0.6b = 1024, text-embedding-3-small = 1536, text-embedding-ada-002 = 1536
    DENSE_EMBEDDING_DIM: int = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))

    @property
    def effective_query_model(self) -> str:
        """Model to use for query rewriting and summarisation. Falls back to OPENAI_MODEL."""
        return self.QUERY_MODEL or self.OPENAI_MODEL

    @property
    def effective_reasoning_model(self) -> str:
        """Model to use for the Thinking answering mode. Falls back to OPENAI_MODEL."""
        return self.REASONING_MODEL or self.OPENAI_MODEL

    @property
    def effective_vision_api_base(self) -> str:
        """Base URL for vision/OCR calls. Falls back to OPENAI_API_BASE."""
        return self.OPENAI_VISION_API_BASE or self.OPENAI_API_BASE

    # ── Query Classification ──────────────────────────────────────────────────
    # Enable/disable LLM-based query classification for adaptive retrieval routing.
    QUERY_CLASSIFIER_ENABLED: bool = os.getenv("QUERY_CLASSIFIER_ENABLED", "true").lower() == "true"

    # Zero-shot classification prompt for the query classifier.
    QUERY_CLASSIFIER_PROMPT: str = (
        "Classify this query into exactly one category. Respond with only the category name.\n\n"
        "Categories:\n"
        "FACTUAL — Questions asking for specific facts, definitions, or concrete information (e.g., 'What is RRF?', 'Define BM25 scoring')\n"
        "ENTITY_CENTRIC — Questions about specific entities, organizations, people, or products (e.g., 'What did Apple acquire?', 'Who founded Microsoft?')\n"
        "MULTI_PART — Questions comparing/contrasting things, asking for pros/cons, or requesting multiple aspects (e.g., 'Compare RRF and BM25', 'Pros and cons of vector databases')\n"
        "AMBIGUOUS — Vague, context-dependent, or incomplete queries (e.g., 'Tell me about that', 'What do you think?')\n\n"
        "Query: {query}\n\n"
        "Category: "
    )

    # JSON string of retrieval config presets per query type.
    # Each preset: use_dense, use_sparse, use_exact, dense_weight, sparse_weight, exact_weight, top_k
    RETRIEVAL_CONFIG_PRESETS: str = json.dumps({
        "FACTUAL": {
            "use_dense": True, "use_sparse": True, "use_exact": True,
            "dense_weight": 0.5, "sparse_weight": 0.3, "exact_weight": 0.2,
            "top_k": 10
        },
        "ENTITY_CENTRIC": {
            "use_dense": True, "use_sparse": True, "use_exact": True,
            "dense_weight": 0.6, "sparse_weight": 0.2, "exact_weight": 0.2,
            "top_k": 10
        },
        "MULTI_PART": {
            "use_dense": True, "use_sparse": True, "use_exact": False,
            "dense_weight": 0.5, "sparse_weight": 0.5, "exact_weight": 0.0,
            "top_k": 10
        },
        "AMBIGUOUS": {
            "use_dense": True, "use_sparse": True, "use_exact": True,
            "dense_weight": 0.4, "sparse_weight": 0.4, "exact_weight": 0.2,
            "top_k": 15
        }
    })

    @property
    def retrieval_config_presets(self) -> Dict[str, Any]:
        """Parse retrieval config presets from JSON string."""
        return json.loads(self.RETRIEVAL_CONFIG_PRESETS)

    # Qdrant vector store
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

    # SPLADE sparse embedding model (FastEmbed / ONNX — CPU-optimised)
    SPLADE_MODEL: str = os.getenv("SPLADE_MODEL", "prithivida/Splade_PP_en_v1")
    # Directory where FastEmbed caches downloaded ONNX models.
    # Mount as a volume so the model survives container restarts.
    FASTEMBED_CACHE_DIR: str = os.getenv("FASTEMBED_CACHE_DIR", "/app/assets/fastembed")

    # ── Retrieval ──────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    # Minimum RRF score to include a chunk in the context passed to the LLM.
    # RRF scores range roughly 0.003–0.02 for a 3-leg setup with K=60.
    # Chunks below this threshold are dropped before the LLM sees them.
    # Set to 0.0 to disable filtering.
    RETRIEVAL_MIN_RRF_SCORE: float = float(os.getenv("RETRIEVAL_MIN_RRF_SCORE", "0.005"))

    # ── Cross-encoder reranker ───────────────────────────────────────────────────
    # When enabled, the top-K RRF candidates are re-scored by a dedicated
    # cross-encoder model and re-ordered by relevance score before being passed
    # to the LLM. More accurate than RRF alone for cross-KB disambiguation.
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-12-v2")
    RERANKER_CACHE_DIR: str = os.getenv("RERANKER_CACHE_DIR", "/app/assets/reranker")
    # How many chunks to keep after reranking. Must be <= RETRIEVAL_TOP_K.
    # Reducing this keeps only the most relevant chunks, further limiting noise.
    RERANKER_SCORE_THRESHOLD: float = float(os.getenv("RERANKER_SCORE_THRESHOLD", "-2.0"))

    # ── Adaptive Retrieval ───────────────────────────────────────────────────
    # Enable/disable adaptive two-pass retrieval (low-confidence expansion).
    # When enabled and retrieval confidence < 55, a second context event is
    # emitted with a wider document set at the adaptive threshold.
    ADAPTIVE_RETRIEVAL_ENABLED: bool = os.getenv("ADAPTIVE_RETRIEVAL_ENABLED", "true").lower() == "true"
    # Confidence score below which adaptive expansion is triggered (0–100).
    ADAPTIVE_RETRIEVAL_THRESHOLD: float = float(os.getenv("ADAPTIVE_RETRIEVAL_THRESHOLD", "55"))
    # Expanded document set threshold — reranker score cutoff for the
    # second-pass adaptive context event. Must be lower (more inclusive)
    # than RERANKER_SCORE_THRESHOLD so that more documents are returned.
    ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD: float = float(
        os.getenv("ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD", "-5.0")
    )

    # ── Historical Memory Retrieval ────────────────────────────────────────────
    # Enable/disable historical memory retrieval (querying past assistant
    # messages from MySQL, reranking, and returning top-K as context blocks).
    HISTORICAL_MEMORY_ENABLED: bool = os.getenv("HISTORICAL_MEMORY_ENABLED", "true").lower() == "true"
    # Number of historical memory docs to return (default 5).
    HISTORICAL_MEMORY_TOP_K: int = int(os.getenv("HISTORICAL_MEMORY_TOP_K", "5"))
    # Minimum cross-encoder reranker score to include a historical memory doc.
    HISTORICAL_MEMORY_SCORE_THRESHOLD: float = float(os.getenv("HISTORICAL_MEMORY_SCORE_THRESHOLD", "2.0"))

    # ── Chunking ────────────────────────────────────────────────────────────────
    # WARNING: changing these values after documents have been ingested creates
    # inconsistent chunk sizes across the knowledge base. If you change them,
    # delete and re-upload all existing documents to re-index with the new settings.
    #
    # CHUNK_SIZE: target chunk size in characters. Keep <= 1800 chars when using
    # SPLADE (prithivida/Splade_PP_en_v1) — BERT's 512-token limit means longer
    # chunks are silently truncated in the sparse leg (~4 chars/token for English).
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1500"))
    # OVERLAP_PERCENTAGE: fraction of CHUNK_SIZE repeated at the start of the next
    # chunk (0.0–1.0). 0.20 = 20% overlap = 300 chars at CHUNK_SIZE=1500.
    OVERLAP_PERCENTAGE: float = float(os.getenv("OVERLAP_PERCENTAGE", "0.20"))

    @property
    def chunk_overlap(self) -> int:
        return int(self.CHUNK_SIZE * self.OVERLAP_PERCENTAGE)

    # RRF weights for each leg. Weights don't need to sum to 1; they are
    # relative multipliers on the RRF term 1/(k + rank).
    HYBRID_DENSE_WEIGHT: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.5"))
    HYBRID_SPARSE_WEIGHT: float = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.3"))
    HYBRID_EXACT_WEIGHT: float = float(os.getenv("HYBRID_EXACT_WEIGHT", "0.2"))

    # Per-leg retrieval enable/disable.
    # Affects retrieval ONLY — ingestion always indexes all three pipelines
    # so re-enabling a leg later requires no re-indexing.
    RETRIEVAL_DENSE_ENABLED: bool = os.getenv("RETRIEVAL_DENSE_ENABLED", "true").lower() == "true"
    RETRIEVAL_SPARSE_ENABLED: bool = os.getenv("RETRIEVAL_SPARSE_ENABLED", "true").lower() == "true"
    RETRIEVAL_EXACT_ENABLED: bool = os.getenv("RETRIEVAL_EXACT_ENABLED", "true").lower() == "true"

    # ── Neo4j / GraphRAG ────────────────────────────────────────────────────────
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "ragwebui_neo4j")

    # Set false to disable graph extraction during ingestion.
    GRAPHRAG_ENABLED: bool = os.getenv("GRAPHRAG_ENABLED", "true").lower() == "true"

    # LLM to use for entity/relationship extraction during graph ingestion.
    # Use an OpenAI-compatible model name, e.g. "gpt-4o" or your local model.
    # Requires use_structured_output support (OpenAI-compatible /chat/completions
    # with response_format=json_schema — most GPT-4 class models and compatible
    # local models with JSON schema support).
    # Also used for query-level entity extraction in entity-aware retrieval.
    GRAPHRAG_LLM: Optional[str] = os.getenv("GRAPHRAG_LLM") or None

    # Enable/disable the graph retrieval leg at query time (ingestion unaffected).
    RETRIEVAL_GRAPH_ENABLED: bool = os.getenv("RETRIEVAL_GRAPH_ENABLED", "true").lower() == "true"

    # Number of graph hops to traverse from seed nodes at query time.
    GRAPHRAG_RETRIEVAL_HOPS: int = int(os.getenv("GRAPHRAG_RETRIEVAL_HOPS", "2"))

    # Maximum number of chunks to run graph extraction on per document.
    # Chunks beyond this limit are skipped for graph extraction but still
    # fully indexed in Qdrant. Set to 0 to disable the cap (default: 300).
    # For large documents on low-RAM local models (e.g. 2B), keep this at
    # 200–400. The first N chunks usually cover the most concept-dense content.
    GRAPHRAG_MAX_CHUNKS: int = int(os.getenv("GRAPHRAG_MAX_CHUNKS", "0"))

    # Context window size (in characters) available to the LLM used for graph
    # extraction. Consecutive chunks are merged into batches up to this budget
    # (with ~20% headroom reserved for the system prompt + JSON schema output).
    # Overlap between adjacent chunks is stripped before concatenation so the
    # LLM sees clean, non-redundant text.
    # Rule of thumb: 1 token ≈ 3–4 chars. For a 4K token context set ~12000.
    # Default: 6000 (safe for 2K-token local models like Qwen-3.5-4B).
    NEO4J_LLM_CONTEXT: int = int(os.getenv("NEO4J_LLM_CONTEXT", "12000"))

    # ── Entity-Aware Retrieval ──────────────────────────────────────────────────
    # Enable/disable entity-aware retrieval (NER extraction + Neo4j expansion).
    ENTITY_AWARE_ENABLED: bool = os.getenv("ENTITY_AWARE_ENABLED", "true").lower() == "true"
    # Score boost factor per entity mention found in a chunk.
    # 0.1 = +10% score per mention. Higher = stronger bias toward entity-rich chunks.
    ENTITY_BOOST_FACTOR: float = float(os.getenv("ENTITY_BOOST_FACTOR", "0.1"))

    # ── Agentic Tool Calling ───────────────────────────────────────────────────
    # Enable/disable LLM tool calling. When enabled, chat requests include the
    # registered tool schemas in the tools= parameter and the backend executes
    # tool_calls returned by the LLM, feeding results back in a loop.
    TOOL_CALLING_ENABLED: bool = os.getenv("TOOL_CALLING_ENABLED", "true").lower() == "true"
    # Maximum tool call iterations per chat turn to prevent infinite loops.
    MAX_TOOL_ITERATIONS: int = int(os.getenv("MAX_TOOL_ITERATIONS", "5"))

    # ── Multi-Document Synthesis ───────────────────────────────────────────────
    # Enable synthesis mode for MULTI_PART queries with synthesis keywords.
    # When enabled, a synthesis-specific system prompt replaces the QA prompt,
    # guiding the LLM through synthesize_documents → extract_entities → summarize_chunks.
    SYNTHESIS_MODE_ENABLED: bool = os.getenv("SYNTHESIS_MODE_ENABLED", "true").lower() == "true"

    PROCESSING_TIMEOUT_SILENCE_S: int = int(os.getenv('PROCESSING_TIMEOUT_SILENCE_S', '300'))

    # ── Answer Quality Grading ────────────────────────────────────────────────
    # Enable/disable automatic quality grading of generated answers. When
    # enabled and retrieval confidence is below 55, the pipeline runs
    # _grade_answer_quality() to check faithfulness/completeness/coherence
    # and may regenerate or add a disclaimer.
    ANSWER_QUALITY_GRADING_ENABLED: bool = os.getenv(
        "ANSWER_QUALITY_GRADING_ENABLED", "true"
    ).lower() == "true"

    # ── Autonomous Agentic Agent ──────────────────────────────────────────────
    # Maximum retry iterations for the autonomous agent loop (supervisor → worker → critic).
    # Each iteration re-runs the supervisor with critic feedback to adjust the plan.
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "3"))
    # Minimum quality score (0-100) for the critic to accept an answer.
    # Below this threshold, the agent retries with adjusted strategy.
    AGENT_QUALITY_THRESHOLD: int = int(os.getenv("AGENT_QUALITY_THRESHOLD", "70"))
    # Enable/disable the autonomous agent mode. When false, falls back to
    # the existing fast/thinking/agentic pipelines.
    AGENT_ENABLED: bool = os.getenv("AGENT_ENABLED", "true").lower() == "true"
    # Enable/disable LangGraph-based pipeline execution. When false, uses
    # the existing generator-based pipeline. When true, routes through
    # the LangGraph StateGraph with nested subgraph architecture.
    USE_LANGGRAPH: bool = os.getenv("USE_LANGGRAPH", "false").lower() == "true"
    # Optional model override for the supervisor and critic LLM calls.
    # Falls back to OPENAI_MODEL when unset.
    AGENT_SUPERVISOR_MODEL: Optional[str] = os.getenv("AGENT_SUPERVISOR_MODEL") or None

    @property
    def graphrag_model(self) -> str:
        """Model to use for entity/relationship extraction. Falls back to OPENAI_MODEL."""
        return self.GRAPHRAG_LLM or self.OPENAI_MODEL

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
