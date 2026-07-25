# RAG Web UI - Comprehensive Features Guide

## Chat Features

### Message Branching
**Edit and continue from any message**

The branching feature allows you to edit any message in a conversation and continue from that point, creating alternate conversation paths. This is useful for:
- Exploring different directions in a conversation
- Correcting mistakes in earlier messages
- Testing different query formulations
- Comparing answers from different approaches

**How it works:**
1. Click the branch icon on any message
2. Edit the message content
3. Send - the system creates a new branch from that point
4. Switch between branches using the branch picker in the UI

**Technical details:**
- Messages are stored with `parent_message_id` and `branch_index`
- The system maintains a tree structure of conversation branches
- Each branch has its own message history
- Citations and context are preserved per branch

### Chat Folders
**Organize your conversations**

Chat folders help you organize conversations by topic, project, or any other classification.

**Features:**
- Create folders with custom names
- Assign chats to folders
- Move chats between folders
- Delete folders (chats become unassigned)

**API endpoints:**
- `POST /api/folders` - Create folder
- `GET /api/folders` - List folders
- `PATCH /api/folders/{id}` - Rename folder
- `DELETE /api/folders/{id}` - Delete folder
- `PATCH /api/folders/{id}/chats/{chat_id}` - Assign chat
- `DELETE /api/folders/{id}/chats/{chat_id}` - Unassign chat

### Chat File Upload (Ephemeral)
**Attach files to conversations without indexing**

Files attached to chat messages are processed ephemerally - they are NOT indexed in any knowledge base.

**How it works:**
1. Upload file via chat input attachment
2. File saved to `uploads/ephemeral/{chat_id}/`
3. MarkItDown converts to Markdown
4. Content stored in MySQL `chat_files` table
5. On next message, file content is injected into the pipeline
6. Files deleted when chat is deleted

**Upload guards:**
- 10 MB file size limit
- Token budget = 25% of `OPENAI_MODEL_CONTEXT_SIZE`
- Files exceeding budget rejected with clear error

**Pipeline handling:**
- Full approved content passed to LLM (no truncation)
- `extract_file_sections` node uses LLM to select 3-6 most relevant sections; files ≤ 12,000 chars passed through unchanged

**API endpoints:**
- `POST /api/chat/{chat_id}/files` - Upload file
- `GET /api/chat/{chat_id}/files/{file_id}` - Get file status
- `DELETE /api/chat/{chat_id}/files/{file_id}` - Delete file
- `GET /api/chat/{chat_id}/files/{file_id}/download` - Download file

### Conversation Compaction
**Automatically summarize long conversations**

For long conversations, the system automatically compacts older messages to save context space and maintain conversation fluency.

**Configuration:**
```env
COMPACTION_ENABLED=true
COMPACTION_HISTORY_THRESHOLD=50
COMPACTION_KEEP_RECENT=10
COMPACTION_SUMMARY_MAX_CHARS=2000
COMPACTION_ASSISTANT_MAX_CHARS=1000
```

**How it works:**
- When message count exceeds threshold, older messages are summarized
- Recent messages (configurable count) are kept in full
- Summaries are stored in `history_summary` field
- LLM uses summary + recent messages for context

### Full-Text Search Across Messages
**Search your conversation history**

Search across all messages in your chats using MySQL FULLTEXT search.

**API endpoint:**
- `GET /api/chat/search?q=query` - Search messages

**Features:**
- Full-text search across message content
- Returns matching messages with context
- Supports natural language queries
- Fast search via FULLTEXT index

## Advanced Retrieval Features

### Historical Memory Retrieval
**Query past assistant messages**

The system can retrieve relevant context from past assistant messages in the conversation history.

**Configuration:**
```env
HISTORICAL_MEMORY_ENABLED=true
HISTORICAL_MEMORY_TOP_K=5
HISTORICAL_MEMORY_SCORE_THRESHOLD=0.7
```

**How it works:**
- Past assistant messages are embedded and stored
- At query time, system searches past messages
- Relevant historical context is added to retrieval
- Uses same embedding model as document chunks

**Use cases:**
- Referring to previous answers in the conversation
- Maintaining context across multiple related queries
- Avoiding repetition of information

### Entity-Aware Retrieval
**NER + score boosting for entity-centric queries**

When queries are entity-centric, the system uses named entity recognition to boost retrieval scores for chunks containing those entities.

**Configuration:**
```env
ENTITY_AWARE_ENABLED=true
ENTITY_BOOST_FACTOR=1.5
```

**How it works:**
1. Extract entities from query using NER
2. Identify chunks containing those entities
3. Boost retrieval scores for matching chunks
4. Re-rank results with entity-boosted scores

**Use cases:**
- Queries about specific people, organizations, or locations
- Technical queries with specific component names
- Domain-specific terminology

### Adaptive Retrieval
**Two-pass retrieval with confidence threshold**

The system can perform a second retrieval pass if the first pass doesn't yield high-confidence results.

**Configuration:**
```env
ADAPTIVE_RETRIEVAL_ENABLED=true
ADAPTIVE_RETRIEVAL_THRESHOLD=0.7
ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD=-2.0
```

**How it works:**
1. First pass: standard hybrid retrieval
2. Calculate confidence score from results
3. If confidence below threshold, trigger second pass
4. Second pass: relaxed reranking threshold
5. Merge and re-rank all results

**Use cases:**
- Difficult queries with low initial confidence
- Queries requiring broader context
- Handling ambiguous or multi-interpretation queries

### Answer Quality Grading
**Automatic faithfulness and completeness scoring**

The system can automatically grade answer quality based on faithfulness to retrieved context and completeness of coverage.

**Configuration:**
```env
ANSWER_QUALITY_GRADING_ENABLED=true
AGENT_QUALITY_THRESHOLD=0.7
```

**How it works:**
- LLM evaluates answer against retrieved context
- Scores faithfulness (answer sticks to facts)
- Scores completeness (answers all aspects)
- Overall quality score from 0-1
- Low-quality answers can trigger regeneration

**Output:**
- `faithfulness` score (0-1)
- `completeness` score (0-1)
- Overall quality level (low/medium/high)

### Synthesis Mode
**Multi-document structured reports**

In agentic mode, the LLM can synthesize information across multiple retrieved contexts to create structured reports.

**Configuration:**
```env
SYNTHESIS_MODE_ENABLED=true
```

**How it works:**
- When multiple documents contain relevant information
- LLM synthesizes across all sources
- Creates structured, coherent answer
- Cites multiple sources appropriately

**Use cases:**
- Compare and contrast queries
- Multi-source research
- Comprehensive overviews

## Knowledge Base Features

### DataStore Watching
**Automatic file system monitoring**

DataStores can be monitored for file changes, triggering automatic ingestion when files are added, modified, or deleted.

**Configuration:**
```env
WATCHER_ENABLED=true
WATCHER_USE_INOTIFY=true
WATCH_POLL_INTERVAL=60
```

**How it works:**
- File system watcher monitors configured directories
- Events: created, modified, deleted, moved
- Debounced to prevent duplicate processing
- Batching for efficiency
- Progress tracking with SSE streaming

**API endpoints:**
- `POST /api/admin/datastores/{id}/scan` - Manual scan
- `POST /api/admin/datastores/{id}/stop-scan` - Stop scan
- `GET /api/admin/datastores/{id}/scan-progress` - Scan progress
- `GET /api/admin/datastores/{id}/scan-progress-stream` - SSE scan progress

### Startup Recovery Service
**Background ingestion on app start**

When the application starts, a recovery service scans for any interrupted ingestion tasks and resumes them.

**How it works:**
- On startup, checks for incomplete `ProcessingTask` records
- Resumes processing from last known state
- Handles crashed or interrupted ingestions
- Ensures data consistency

**API endpoints:**
- `GET /api/admin/datastores/recovery-status` - All recovery status
- `GET /api/admin/datastores/{id}/recovery-status` - Specific recovery status
- `GET /api/admin/datastores/{id}/recovery-stream` - SSE recovery stream
- `POST /api/admin/datastores/{id}/recover` - Trigger recovery

### Chunking Preview
**Test chunking before ingestion**

Preview how documents will be chunked before committing to ingestion.

**API endpoint:**
- `POST /api/knowledge-base/{kb_id}/preview` - Preview chunking

**Features:**
- Upload documents for preview only
- See chunk boundaries and sizes
- Test different chunking parameters
- No permanent storage or indexing

## Multi-Tenancy Features

### Hierarchical Organizations
**Tree structure for organization management**

Organizations are arranged in a hierarchical tree structure using materialized path for efficient queries.

**Features:**
- Parent-child relationships
- Path-based tree traversal
- Efficient subtree queries
- Users inherit access from parent orgs

**API endpoints:**
- `GET /api/admin/orgs` - List orgs (hierarchical)
- `POST /api/admin/orgs` - Create org (parent_id required)
- `PATCH /api/admin/orgs/{id}` - Update org
- `DELETE /api/admin/orgs/{id}` - Delete org

### Per-Organization LLM Configuration
**Custom LLM settings per organization**

Each organization can have its own LLM configuration, overriding the global defaults.

**Configuration stored in database:**
- `api_base` - LLM API endpoint
- `model_name` - Main model
- `query_model` - Query processing model

**API endpoints:**
- `GET /api/admin/orgs/{org_id}/llm-config` - Get org LLM config
- `PUT /api/admin/orgs/{org_id}/llm-config` - Upsert org LLM config

### Organization Abbreviations
**Org-specific terminology expansion**

Organizations can define abbreviations and their expansions for domain-specific terminology.

**How it works:**
- Admins define abbreviations per organization
- Query expander replaces abbreviations with full terms
- Improves retrieval for domain-specific content
- Stored in `org_abbreviations` table

### DataStore Sharing
**Share data sources across organizations**

DataStores can be shared across multiple organizations, allowing efficient use of shared document repositories.

**Features:**
- Many-to-many relationship between orgs and datastores
- Shared ingestion, separate access control
- Efficient storage, flexible access

**API endpoints:**
- `POST /api/admin/datastores/{id}/assign` - Assign to orgs
- `DELETE /api/admin/datastores/{id}/assign` - Unassign from orgs

## User Features

### User Profiles
**Persistent agent preferences**

Users can have profiles with preferences for how the agent interacts with them.

**Stored preferences:**
- `preferences_json` - General preferences
- `query_patterns_json` - Common query patterns
- `domain_focus_json` - Domain focus areas
- `communication_style` - Communication style preferences

**How it works:**
- Profile stored in `user_profiles` table
- Agent adapts responses based on profile
- Preferences persist across sessions

### Clarification Requests
**Agent can ask for clarification**

When a query is ambiguous, the agent can ask the user for clarification before proceeding.

**How it works:**
- Agent detects ambiguous queries
- Generates clarification question
- Presents options to user
- User selects or provides custom answer
- Agent proceeds with clarified query

**Stored in:**
- `clarification_requests` table
- Links to chat and message
- Tracks conversation state

## Admin Features

### Ingestion Status Monitoring
**Track document processing progress**

Monitor the status of document ingestion across all knowledge bases and datastores.

**API endpoints:**
- `GET /api/query/kb/{kb_id}/ingest-status` - KB processing status
- `GET /api/admin/orgs/{org_id}/ingestion-status` - Org ingestion status

**Status information:**
- Processing tasks and their states
- Progress percentages
- Error messages
- Document counts

### Scan Progress Streaming
**Real-time scan progress via SSE**

DataStore scans can be monitored in real-time using Server-Sent Events.

**API endpoint:**
- `GET /api/admin/datastores/{id}/scan-progress-stream` - SSE scan progress

**Streamed events:**
- Scan start/stop
- File processing progress
- Error notifications
- Completion status

## API Features

### Stateless Query Endpoint
**RAG queries without chat session**

For one-off queries without creating a chat session, use the stateless query endpoint.

**API endpoint:**
- `POST /api/query` - Stateless RAG query

**Features:**
- No chat session required
- Specify knowledge base ID
- Uses the agentic pipeline with automatic query adaptation
- Returns answer with citations
- No conversation history

### Configuration Endpoint
**Client configuration**

Frontend can fetch configuration values from the backend.

**API endpoint:**
- `GET /api/config` - Client config

**Returns:**
- `chunk_size` - Current chunk size
- `chunk_overlap` - Current chunk overlap
- Other client-relevant settings

## Streaming Features

### Server-Sent Events (SSE)
**Real-time progress updates**

Multiple endpoints use SSE for real-time progress updates:

1. **Chat responses** - Token-by-token streaming
2. **Agent timeline** - Step-by-step progress
3. **Scan progress** - File processing updates
4. **Recovery progress** - Recovery status updates

**Benefits:**
- Real-time feedback
- Reduced perceived latency
- Better user experience
- Cancellation support

### Cancellation Support
**Cancel in-flight operations**

Long-running operations can be cancelled by the user.

**API endpoints:**
- `POST /api/chat/{chat_id}/cancel` - Cancel streaming response
- `POST /api/admin/datastores/{id}/stop-scan` - Stop scan

**How it works:**
- Client sends cancellation request
- Server stops processing
- Partial results preserved
- Clean shutdown

## Testing Features

### Retrieval Testing UI
**Test retrieval on knowledge base**

Test retrieval settings and results before using in production.

**API endpoint:**
- `POST /api/knowledge-base/{kb_id}/test-retrieval` - Test retrieval

**Features:**
- Test query against KB
- See retrieved chunks
- View relevance scores
- Test different configurations
- Debug retrieval issues

## Additional Features

### Citations with Metadata
**Source attribution for answers**

All answers include citations with rich metadata.

**Citation metadata:**
- Document ID and filename
- Chunk index
- Source location (page number, section)
- Relevance score
- Direct link to source

### Message Editing
**Edit sent messages**

Messages can be edited after sending (with branching).

**API endpoint:**
- `PATCH /api/chat/{chat_id}/messages/{msg_id}` - Edit message

**How it works:**
- Edit creates new branch
- Original message preserved
- Conversation continues from edit
- Full audit trail maintained

### Message Deletion
**Delete individual messages**

Individual messages can be deleted from conversations.

**API endpoint:**
- `DELETE /api/chat/{chat_id}/messages/{msg_id}` - Delete message

**Considerations:**
- Deletes message and its citations
- Does not affect chat file attachments
- Conversation continuity maintained

### Pinned Chats
**Pin important conversations**

Chats can be pinned for easy access.

**API endpoint:**
- `PATCH /api/chat/{id}` - Update chat (including pinned status)

**How it works:**
- Pinned chats appear first in list
- Useful for important or ongoing conversations
- Persistent across sessions

### Knowledge Base Data Source Linking
**Connect KBs to DataStores**

Knowledge bases can be linked to DataStores for automatic ingestion.

**API endpoints:**
- `POST /api/knowledge-base/{kb_id}/data-sources` - Link data sources
- `DELETE /api/knowledge-base/{kb_id}/data-sources/{ds_id}` - Unlink data source

**Benefits:**
- Automatic ingestion from DataStores
- Centralized document management
- Efficient updates
- Shared document repositories
