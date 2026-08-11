# Operator Runbook: Settings Migration

This runbook covers operational concerns for the runtime settings system
introduced in the settings migration. Settings are now resolved via a 2-tier
precedence: **org override → app-level DB value → registry default**.

## Restart-required changes

These settings require a backend restart to take effect (they load
process-global resources at startup):

| Setting | Reason |
|---------|--------|
| `DENSE_EMBEDDINGS_MODEL` | Qdrant collections are dimension-locked |
| `DENSE_EMBEDDING_DIM` | Tied to collection schema |
| `SPLADE_MODEL` | Loaded once by FastEmbed into a process-global ONNX session |
| `RERANKER_MODEL` | Loaded once into a process-global cross-encoder |
| `MEMORY_EMBEDDING_MODEL` | Embedding model for Redis store |
| `MEMORY_ENABLED` | Redis checkpointer singleton |
| `WATCHER_ENABLED` | One watcher process watches all DataStore folders |
| `WATCH_POLL_INTERVAL` | PollingObserver timeout |

## Reindex-required changes

These settings change index geometry and require re-indexing or
re-ingestion to take full effect:

| Setting | Action required |
|---------|----------------|
| `DENSE_EMBEDDINGS_MODEL` | Delete and re-upload all documents (new embedding dimension) |
| `DENSE_EMBEDDING_DIM` | Same as above |
| `CHUNK_SIZE` | Delete and re-upload all documents (new chunk boundaries) |
| `OVERLAP_PERCENTAGE` | Delete and re-upload all documents |
| `GRAPHRAG_MAX_CHUNKS` | Re-run graph extraction on affected documents |
| `NEO4J_LLM_CONTEXT` | Re-run graph extraction (different batch boundaries) |

## Reingest-required changes

These settings affect graph extraction during ingestion:

| Setting | Action required |
|---------|----------------|
| `GRAPHRAG_ENABLED` | Re-ingest documents to extract/skip graph entities |
| `GRAPHRAG_MAX_CHUNKS` | Re-ingest to re-run extraction with new cap |
| `NEO4J_LLM_CONTEXT` | Re-ingest to re-run extraction with new context budget |

## OrgLLMConfig compatibility

The existing `OrgLLMConfig` table and the `/api/admin/orgs/{org_id}/llm-config`
endpoints are preserved for backward compatibility. The LLM factory
(`llm_factory.py`) and chat service (`chat_service.py`) now read from the
unified settings service instead of `OrgLLMConfig` directly.

During the migration window:
- The old LLM Config dialog on the Orgs page continues to work.
- The new Org Settings page provides the same LLM settings plus all other
  org-overridable settings.
- No data migration is required — the old `OrgLLMConfig` rows are simply not
  read by the new code path. Organisations that had LLM config via the old
  UI will need to reconfigure via the new Settings page.

## Cache behavior

The settings service uses a 30-second in-memory TTL cache. After changing a
setting via the API, the cache is invalidated immediately for the affected
key. Other keys remain cached until their TTL expires.

If you need to force-clear the cache (e.g., after a direct DB edit), restart
the backend.

## Migration verification

After deploying the settings migration:

1. Verify the `settings` table exists:
   ```sql
   DESCRIBE settings;
   ```

2. Verify app-level rows:
   ```sql
   SELECT key, scope, LEFT(value, 50) FROM settings WHERE scope = 'app';
   ```

3. Test the API:
   ```bash
   # As super admin
   curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/admin/settings
   curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/admin/settings/schema
   ```

4. Test the UI:
   - Navigate to Admin → Settings (super admin only)
   - Navigate to Admin → Orgs → Settings button (per-org)
