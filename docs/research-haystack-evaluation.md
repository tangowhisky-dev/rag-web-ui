# Haystack Evaluation: LangGraph → Haystack Migration Research

> **⚠️ OUTDATED — `rag_graph.py` has been removed.** This evaluation analyzed our LangGraph `rag_graph.py` pipeline for a potential migration to Haystack. The `rag_graph/` package has been deleted. The current pipeline is `agentic_rag/agentic_rag.py` — a simpler agentic agent with simple/complex branching. This document is retained for historical comparison purposes only.

---

## 1. What is Haystack v2.30?

Haystack (by deepset, €45.6M raised) is an open-source Python framework for **production-ready AI Agents, RAG pipelines, and multimodal search**. Core version is `haystack-ai` (lightweight core + optional extras).

### Core building blocks
| Block | Purpose |
|-------|---------|
| `@component` | Decorator for custom components with typed I/O sockets |
| `Pipeline` | DAG wiring of components (`add_component` + `connect`) |
| `Agent` | ReAct-style agentic loop (LLM + tool calling + iteration) |
| `State` / `state_schema` | Typed state for agent tool communication |
| `Document` / `DocumentStore` | RAG data model |
| `ConditionalRouter` | Declarative Jinja2 routing with fallback |
| `SuperComponent` | Wrap a Pipeline as a single Component (like LangGraph SubGraph) |

### Install
```
pip install haystack-ai
```
Optional extras via `[qdrant]`, `[openai]`, `[anthropic]`, `[fastembed]`, etc.

---

## 2. Concept Mapping: LangGraph → Haystack

| LangGraph | Haystack | Notes |
|-----------|----------|-------|
| `StateGraph` | `Pipeline` + `Agent` | DAG + agentic loop |
| Graph node | `@component` class with `run()` | Same decorator-based approach |
| `add_node()` | `add_component()` | Named instances |
| `add_edge()` | `connect(sender, receiver)` | Port-to-port wiring |
| `add_conditional_edges()` | `ConditionalRouter` | Jinja2 conditions, fallback route |
| `subgraph.compile()` | `SuperComponent` | Pipeline-as-component wrapper |
| `create_react_agent()` | `Agent(chat_generator, tools, ...)` | Drop-in agentic loop |
| `MessagesState` | `State` + `state_schema` | Typed state management |
| `@tool` decorator | `@tool` decorator | Same pattern, also `ComponentTool` |
| `langchain_mcp_adapters` | `MCPToolset` / `MCPTool` | Native MCP support |
| Checkpointing | `DocumentWriter` + DocumentStore | Write docs, read later |

---

## 3. Mapping Our 8-Node `rag_graph` to Haystack

### Current graph (`_rag_graph.py`)
```
__start__ → rewrite_query → context_router → kb_retrieval
                                                → grade_documents
                                                → merge_context
                                                → generate_answer
        ↕ conditional edges (route_after_grade)
        extract_file_sections (optional path)
```

### Haystack equivalent architecture

#### Option A: Full pipeline + Agent for tool loop
```python
# Pipeline (deterministic parts)
rag_pipe = Pipeline()
rag_pipe.add_component("rewrite_query", QueryRewriterComponent())
rag_pipe.add_component("classifier", QueryClassifierComponent())
rag_pipe.add_component("router", ConditionalRouter(routes=[...]))  # FACTUAL/ENTITY/etc
rag_pipe.add_component("retriever", QdrantHybridRetriever(...))
rag_pipe.add_component("grade_docs", GradeDocumentsComponent())
rag_pipe.add_component("joiner", DocumentJoiner())
rag_pipe.add_component("synthesizer", SynthesisComponent())

# Agent (tool-calling loop)
agent = Agent(
    chat_generator=OpenAIChatGenerator(model=OPENAI_MODEL),
    tools=[
        ComponentTool(component=SearchDocuments(...)),
        ComponentTool(component=ExtractEntities(...)),
    ],
    system_prompt=SYSTEM_PROMPT,
    state_schema={"documents": {"type": list[Document]}, "user_name": {"type": str}},
    streaming_callback=print_streaming_chunk,
)
```

#### Option B: Single `Agent` wrapping the full pipeline
```python
rag_pipeline = Pipeline()
# ... all components wired ...

# Wrap as a single tool
rag_tool = SuperComponent(
    pipeline=rag_pipeline,
    input_mapping={"query": ["rewrite_query.query"]},
    output_mapping={"retriever.documents": "documents"}
)

agent = Agent(
    chat_generator=...,
    tools=[ComponentTool(component=rag_tool), ...],
    state_schema={"user_context": {"type": str}},
)
```

### Mapping each node

| LangGraph Node | Haystack Equivalent | Effort |
|---------------|---------------------|--------|
| `rewrite_query` | Custom `@component` with Generator | Low |
| `context_router` | `ConditionalRouter` (declarative) or custom component | Low |
| `kb_retrieval` | `QdrantHybridRetriever` (built-in, hybrid dense+sparse+exact) | **Low** — drops our 1,000+ line `retrieval.py` |
| `grade_documents` | Custom `@component` or cross-encoder + filter | Medium |
| `extract_file_sections` | `@component` wrapping existing logic | Low |
| `merge_context` | `DocumentJoiner` (built-in RRF/joiner) or custom | Low |
| `generate_answer` | `Agent` component (built-in agentic loop) | Low |
| Query classifier (LLM) | `Generator` + custom component | Low |
| Tool caller loop | `Agent` component (already built-in!) | **Eliminates custom loop** |
| Synthesis | `Agent` with synthesis tools | Low |

---

## 4. Qdrant Integration — Direct Compatibility

Our current stack: Qdrant (dense) + MySQL FTS (exact) + SPLADE (sparse).

Haystack provides **first-class Qdrant integration** (`haystack-integrations`):

```python
# Dense embedding retriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantDenseEmbeddingRetriever

# Sparse embedding retriever (SPLADE)
from haystack_integrations.components.retrievers.qdrant import QdrantSparseEmbeddingRetriever
from haystack.components.embedders import FastembedSparseDocumentEmbedder, FastembedSparseTextEmbedder

# Hybrid retriever (combines dense + sparse in one call)
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
```

### Migration path for retrieval
1. `QdrantDocumentStore` connects to our existing Qdrant instance (same `QDRANT_HOST`, `QDRANT_PORT`)
2. `migrate_to_sparse_embeddings_support()` migrates existing data to sparse embedding index
3. `QdrantHybridRetriever` replaces our custom `hybrid_search()` logic (~1,000 lines in `retrieval.py`)
4. `DocumentJoiner` replaces our custom RRF merge logic

### MySQL FTS (exact match) — no native Haystack equivalent
Haystack's document stores are vector-native. For our MySQL FTS exact leg, we'd need a custom `@component` wrapping the existing `search.py` MySQL logic — or drop the exact leg and rely on dense+sparse (which is common in production).

---

## 5. Agent Component — The Biggest Win

Our current custom tool-calling loop in `rag_graph.py` (manual tool iteration, state tracking, max-tool-call limits) would be replaced by:

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool, tool
from haystack.utils import Secret

# Define tools (same pattern as current)
@tool
def search_documents(query: str, user_context: str) -> dict:
    """Search knowledge base for relevant documents."""
    ...

@tool
def extract_entities(query: str) -> list:
    """Extract entities from query."""
    ...

# Built-in agentic loop
agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model=Secret.from_env_var("OPENAI_MODEL"),
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
    ),
    tools=[
        ComponentTool(component=SearchDocuments(...)),
        tool(extract_entities),
    ],
    system_prompt=SYSTEM_PROMPT,
    state_schema={"documents": {"type": list[Document]}, "user_name": {"type": str}},
    streaming_callback=print_streaming_chunk,
)

# State injection — user context auto-passed
result = agent.run(
    messages=[ChatMessage.from_user(query)],
    user_name=user_context,  # injected via state_schema
)
# result["last_message"].text — streaming output
# result["documents"] — state data
```

### Built-in features our current code lacks
| Feature | Current | Haystack `Agent` |
|---------|---------|------------------|
| Tool iteration loop | Manual `while` loop with max iterations | Built-in, configurable |
| Human-in-the-loop | None | `intercept()` on tool calls |
| Streaming | Token-level streaming callback | First-class `streaming_callback` |
| State management | Dict passed through nodes | Typed `State` + `state_schema` |
| Error recovery | Manual try/catch | Graceful degradation |
| Max tool calls | `MAX_TOOL_ITERATIONS` env var | Built-in via `max_tool_calls` |

---

## 6. Conditional Routing — Our Query Classifier

Current: LLM-based classification (`QUERY_MODEL` with zero-shot prompt)

Haystack: `ConditionalRouter` (Jinja2 templates, no LLM needed)

```python
from haystack.components.routers import ConditionalRouter

routes = [
    {
        "condition": '{{ query_type == "FACTUAL" }}',
        "output": "{{ query }}",
        "output_name": "factual_route",
        "output_type": str,
    },
    {
        "condition": '{{ query_type == "ENTITY_CENTRIC" }}',
        "output": "{{ query }}",
        "output_name": "entity_route",
        "output_type": str,
    },
    # ... MULTI_PART, AMBIGUOUS ...
    {
        "condition": "{{ True }}",  # fallback
        "output": "{{ query }}",
        "output_name": "default_route",
        "output_type": str,
    },
]

router = ConditionalRouter(routes, optional_variables=["query_type"])
```

**Note:** Our LLM-based classifier provides better accuracy than template matching. We'd keep the LLM classifier as a custom component and use `ConditionalRouter` for routing based on its output.

---

## 7. Performance & Dependency Impact

| Metric | LangGraph (current) | Haystack |
|--------|---------------------|----------|
| Core package size | `langgraph` (~500KB) | `haystack-ai` (~300KB) |
| Dependencies | `langchain`, `langgraph`, `langchain-core`, `langchain-openai`, etc. (100+ deps) | Minimal core; extras installed on-demand |
| LLM call overhead | Identical (both are async wrappers) | Identical |
| Retrieval | Custom 1,000-line `retrieval.py` | `QdrantHybridRetriever` (~50 lines) |
| Tool calling | Custom loop (~200 lines) | `Agent` component (built-in) |
| Serialization | Custom `StateGraph` serialization | YAML/TOML pipeline serialization built-in |

**Performance:** No measurable difference — both are async Python wrappers around the same LLM calls. The bottleneck is always the LLM.

**Dependency bloat:** Haystack's core is lighter. LangChain ecosystem pulls in 100+ transitive deps. Haystack uses on-demand optional imports (e.g., `ImportError` with helpful message).

---

## 8. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Loss of our custom retrieval logic** | High | `QdrantHybridRetriever` handles dense+sparse; MySQL FTS needs custom component |
| **State migration** | Medium | Haystack `State` is typed dict; mapping from LangGraph `ChatModel` state is straightforward |
| **Streaming changes** | Medium | Haystack streaming callback has different signature; need to update SSE endpoint |
| **Testing impact** | High | All 354 tests need updating; tool signatures change |
| **No LangGraph visualization** | Low | Haystack has no Studio equivalent; but our graph is already in code |
| **Agent tool loop quality** | Medium | Haystack's Agent is simpler; test thoroughly with real queries |
| **LLM model compatibility** | Low | Both support any OpenAI-compatible API (LM Studio, local models) |

---

## 9. Recommendation: Hybrid Approach (Recommended)

**Don't do a full migration. Use Haystack incrementally.**

### Rationale
1. **Our LangGraph graph is well-designed** — 8 nodes, clean separation, 16 tests pass
2. **Haystack's strengths align with specific parts** — retrieval (`QdrantHybridRetriever`), tool calling (`Agent`), serialization
3. **Our custom parts are actually good** — query classifier with LLM, entity extraction, adaptive routing presets
4. **Full migration is risky** — 5+ files rewritten, regression potential, no immediate user value
5. **Incremental migration de-risks everything** — test each replacement independently

### Phase Plan

#### Phase 1: Add Haystack + Replace Retrieval (Low risk)
- Add `haystack-ai[qdrant,openai,fastembed]` to `requirements.txt`
- Write `haystack_retrieval.py` wrapper using `QdrantHybridRetriever`
- Keep current `retrieval.py` as fallback (feature flag)
- Add tests for Haystack retriever
- **No API changes** — same `retrieval.py` interface

#### Phase 2: Replace Tool Loop with `Agent` (Medium risk)
- Replace custom tool-calling loop with `Agent` component
- Keep LangGraph graph for query routing + retrieval
- `Agent` only handles tool calls + synthesis
- Test with real queries, compare quality vs current
- **No API changes** — `chat_service.generate_response()` still works

#### Phase 3: Evaluate & Decide (Informed choice)
- Compare: latency, quality, token usage, code complexity
- If `Agent` + `QdrantHybridRetriever` outperforms: proceed with full migration
- If not: keep hybrid (LangGraph + Haystack components)
- If regression: revert, still learned valuable patterns

### If Full Migration Is Chosen (Phase 3+ decision)

```
Files to modify:
├── backend/rag_graph.py        → Haystack Pipeline + Agent
├── backend/retrieval.py        → QdrantHybridRetriever (eliminate ~1000 lines)
├── backend/chat_service.py     → Simplify (Agent handles streaming)
├── backend/app/services/builtin_tools.py → ComponentTool wrappers
├── backend/requirements.txt    → Add haystack-ai, remove langgraph
└── backend/tests/              → Update 354 tests
```

Estimated effort: **2-3 weeks** for full migration (including testing).

---

## 10. Key Decision Points

| Decision | Recommended | Why |
|----------|-------------|-----|
| Full migration now? | **No** | Risky, no immediate user value |
| Incremental adoption? | **Yes** | De-risked, learn while building |
| Start with retrieval? | **Yes** | Biggest code reduction (~1000 lines), lowest risk |
| Keep MySQL FTS exact leg? | **Yes (custom)** | No Haystack equivalent, but valuable |
| Drop LangGraph entirely? | **Defer** | Evaluate after Phase 2 |
| Keep LLM query classifier? | **Yes** | Better than template matching |

---

## Appendix A: Haystack-to-LangGraph Migration Code Examples (Official)

### LangGraph: Agent with Tool Loop
```python
from langgraph.graph import StateGraph, START

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()
```

### Haystack Equivalent
```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool

@tool
def search(query: str) -> dict:
    """Search for relevant documents."""
    ...

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o"),
    tools=[search],
    system_prompt="You are a helpful assistant.",
)

result = agent.run(messages=[ChatMessage.from_user("Find Python docs")])
```

### LangGraph: Subgraph
```python
subgraph = StateGraph(MessagesState)
subgraph.add_node("retrieve", retrieve)
subgraph.add_node("generate", generate)
subgraph.add_edge(START, "retrieve")
subgraph.add_edge("retrieve", "generate")
subgraph.add_edge("generate", END)
graph.add_node("subgraph", subgraph.compile())
```

### Haystack Equivalent
```python
from haystack import Pipeline, SuperComponent

sub_pipeline = Pipeline()
sub_pipeline.add_component("retriever", retriever)
sub_pipeline.add_component("generator", generator)
sub_pipeline.connect("retriever.documents", "generator.documents")

sub_component = SuperComponent(sub_pipeline, input_mapping={...}, output_mapping={...})
```

---

---

## 11. Dify Comparison — A Different Category Entirely

### What is Dify?
Dify is an **open-source full-stack LLM application platform** (GitHub: 80k+ stars). It is NOT a Python library — it's a complete application with its own:
- Flask backend API
- Next.js frontend UI
- PostgreSQL database
- Celery worker (async tasks)
- Redis message broker
- Sandbox execution environment
- Plugin daemon

### Architecture Comparison: All Three Frameworks

| Aspect | LangGraph (current) | Haystack | Dify |
|--------|-------------------|----------|------|
| **Type** | Python library | Python library | Full-stack platform |
| **Embeddable** | ✅ `pip install langgraph` | ✅ `pip install haystack-ai` | ❌ Runs as separate service |
| **Code integration** | Direct Python calls | Direct Python calls | API only (HTTP/REST) |
| **UI included** | No | No | Yes (workflow builder, datasets, prompts) |
| **Database** | Your own | Your own (via DocumentStore) | PostgreSQL (its own) |
| **Deployment** | Part of your app | Part of your app | Docker Compose stack |
| **Min services** | Your app | Your app | 10 services (api, web, db, redis, worker, sandbox, plugin_daemon, nginx, ssrf_proxy, worker_beat) |
| **Customizability** | Full (source code) | Full (source code) | Limited to API + plugin SDK |
| **RAG capabilities** | Build yourself | Built-in components | Built-in (datasets, chunking, multimodal) |
| **Agent capabilities** | Build yourself | Built-in `Agent` component | Built-in (ReAct + Function Calling) |
| **Model support** | Any OpenAI-compatible | Any OpenAI-compatible | 200+ providers built-in |
| **Workflow builder** | Code-only (StateGraph) | Code + YAML | Visual drag-and-drop canvas |
| **Observability** | Build yourself | Build yourself | Built-in (tracing, tokens, cost) |
| **Multi-tenant** | Build yourself | Build yourself | Built-in (workspaces, tenant isolation) |
| **Team collaboration** | N/A | N/A | Built-in (shared datasets, prompts, workflows) |

### Why Dify Is NOT a Replacement Option

Dify is in a fundamentally different category:

1. **Not embeddable**: You cannot `import dify` in your Python code. Dify runs as a separate HTTP service. Your app would need to communicate via REST API.

2. **Massive infrastructure overhead**: A minimal Dify deployment requires **10 Docker services** (api, web, db_postgres, redis, worker, worker_beat, sandbox, plugin_daemon, ssrf_proxy, nginx). Compare to our current stack which already runs Qdrant, MySQL, Neo4j.

3. **Duplicate functionality**: Dify provides RAG, agent orchestration, model management, dataset management, and observability — all of which we already have or are building. It would be layering a second application framework on top of ours.

4. **API-only integration**: The Dify integration path is through its REST API:
   ```python
   # Dify API call from our backend
   import requests
   response = requests.post(
       "http://dify-api:5001/v1/workflows/run",
       json={"inputs": {"query": user_query}, "response_mode": "streaming", "user": user_id},
   )
   ```
   This adds network latency, a new failure mode, and a dependency on an external service.

5. **No workflow graph transparency**: Dify's workflow graph is defined in its UI/JSON and executed internally. We'd have no visibility into the execution flow — everything becomes a black box HTTP call.

6. **Migration complexity**: If we adopted Dify, we'd need to:
   - Rebuild our entire RAG pipeline as a Dify workflow
   - Rebuild our agent tool-calling as a Dify agent
   - Replace our existing chat API with Dify workflow API calls
   - Maintain both our codebase AND the Dify platform
   - Handle cross-service communication failures

### When Dify Makes Sense

Dify is ideal for:
- **Non-technical teams** who want a visual workflow builder
- **Multi-tenant SaaS platforms** that need workspace isolation
- **Organizations** that want rapid prototyping without engineering overhead
- **Teams** that need shared datasets, prompts, and workflows across multiple apps

### Our Use Case

Our project is an **embedded RAG chat application** with:
- Custom adaptive retrieval (4-way query classification)
- Complex agentic flow (8-node LangGraph with conditional edges)
- Existing infrastructure (Qdrant, MySQL, Neo4j, Docker Compose)
- Active development with 354 tests

Dify would be overkill and under-integrated. We need a **Python library** that fits into our existing application — that's Haystack (or keeping LangGraph).

---

## 12. Final Three-Way Comparison: LangGraph vs Haystack vs Dify

| Criterion | LangGraph (keep) | Haystack (migrate) | Dify (not applicable) |
|-----------|-----------------|-------------------|----------------------|
| **Embeddable in our codebase** | ✅ Native | ✅ Native | ❌ External service |
| **Infrastructure cost** | Zero | Zero | 10 Docker services |
| **Custom retrieval logic** | ✅ Full control | ✅ `@component` + integrations | ❌ Via workflow UI only |
| **Custom agent logic** | ✅ StateGraph control | ✅ `Agent` component | ⚠️ Limited to ReAct/Function Calling |
| **Code quality impact** | — | ✅ Cleaner (built-in components) | ❌ Adds abstraction layer |
| **Performance** | Baseline | Same (LLM bottleneck) | Worse (HTTP overhead) |
| **Testability** | ✅ 354 tests pass | ⚠️ Need rewrites | ❌ Black box |
| **Deployment complexity** | Same | Same | +10 services |
| **Future agentic features** | Possible but manual | Built-in (`Agent`, `MCPToolset`) | Built-in but external |
| **Team collaboration** | Code review | Code review | Built-in (not relevant) |
| **Vendor lock-in** | Anthropic/LangChain | deepset | Dify (closed platform) |

### Recommendation Summary

| Framework | Verdict | Action |
|-----------|---------|--------|
| **Dify** | ❌ Not applicable | Skip — wrong category (platform vs library) |
| **Haystack** | ⚠️ Evaluate incrementally | Phase 1-3 plan (retrieval → agent → decide) |
| **LangGraph** | ✅ Keep as fallback | Current graph is well-designed, 354 tests pass |

**Final recommendation: Keep LangGraph as the agentic framework, adopt Haystack incrementally for specific components (retrieval, tool calling).**

Dify is a powerful platform for teams that need a visual application builder, but it's the wrong tool for an embedded Python application with custom RAG logic. It would add complexity, latency, and a new failure surface without solving any problem we don't already have.

---

*This research was produced from Context7 official docs (haystack-ai v2.30, langgenius/dify) + web research. All code examples are from official documentation.*
