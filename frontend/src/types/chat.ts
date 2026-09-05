/** Shared SSE event type definitions for the chat streaming protocol. */

export interface CitationRef {
  document_id?: number;
  citation_kind?: string; // chunk, file, section, range, grep, table, outline
  chunk_index?: number;
  section?: string;
  start_char?: number;
  end_char?: number;
  start_line?: number;
  end_line?: number;
  page?: number;
  match_line?: number;
  quoted_text?: string;
  source_tool?: string;
  citation_id?: string;
}

export interface CitationMetadata {
  kb_id?: number;
  document_id?: number;
  source?: string;
  citation_ref?: CitationRef;
  [key: string]: unknown;
}

export interface Citation {
  id: number;
  text: string;
  metadata: CitationMetadata;
  kb_id?: number;
  data_store_id?: number;
  document_id?: number;
  file_name?: string;
  chunk_index?: number;
  score?: number;
  dense_rank?: number;
  sparse_rank?: number;
  exact_rank?: number;
  retrieval_leg?: string;
  _legs?: string[];
  _reranker_score?: number;
  qdrant_point_id?: string;
  citation_ref?: CitationRef;
}

// SSE event payloads
export interface AnswerRewriteEvent {
  content: string;
  citations?: Array<{ page_content: string; metadata: Record<string, any> }>;
}

export interface ToolCallEvent {
  tool: string;
  args?: Record<string, unknown>;
  id?: string;
}

export interface ToolObservationEvent {
  tool: string;
  result?: unknown;
  error?: string | null;
  tokens?: number;
}

export interface ToolRetryEvent {
  tool: string;
  attempt: number;
  max_retries: number;
  success: boolean;
  error?: string | null;
}

export interface ProgressEvent {
  phase: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface AgentStep {
  node: string;
  status: string;
  latency_ms: number;
  [key: string]: unknown;
}

export interface LastAnswerEvent {
  summary?: string;
  key_points?: string[];
  data?: Array<{ label: string; value: number; unit?: string; context?: string }>;
  citations?: Array<CitationRef & { page_content?: string }>;
  chart_option?: Record<string, unknown> | null;
  followups?: string[];
  suggestion?: string;
  retry_strategy?: string;
}
