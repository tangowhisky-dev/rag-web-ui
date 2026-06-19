       # File Changes Detection and Ingestion

       Complete flow from file change to ingestion to UI feedback.

       ---

       ## 1. Change Detection

       ### How it works

       ```
       Filesystem event (inotify on Linux, FSEvents on macOS)
       │
       ▼
       DatastoreFileEventHandler.dispatch()    [skips directories]
       │
       ▼
       on_created / on_modified / on_deleted / on_moved
       │
       ├── _resolve_datastore()
       │     Finds which datastore's folder_path contains the event path.
       │     Sorts by path length (longest first) so /app/data/reports/2024
       │     doesn't match /app/data/reports.
       │
       ├── _should_process()
       │     Per-file debounce: 1s window. Prevents duplicate events
       │     from VS Code's temp-file-write-and-rename pattern.
       │     Only after an event is processed does the timer reset.
       │
       ├── _dispatch()
       │     ├── _Debouncer.touch()
       │     │     Coalesces rapid repeated events for the SAME path
       │     │     into one handling. 1s delay.
       │     │
       │     └── 1s write-completion delay
       │           Spawns a daemon thread that sleeps 1s then calls
       │           _queue_change + _process_pending_changes.
       │           This ensures the file write is fully complete before
       │           we try to read it for ingestion.
       │           A _SyntheticEvent object is created with the path and
       │           event type, so the original watchdog event (which may
       │           have changed state by then) is not used.
       │
       └── _queue_change()
             Appends to pending_changes[datastore_id] as a dict:
             {path, event_type, datastore_id, org_id, timestamp}
       ```

       ### When it fires

       | Event     | Action                                           |
       |-----------|--------------------------------------------------|
       | `created` | File appears in watched directory → routed through `_handle_file()` |
       | `modified` | File content changes → routed through `_handle_file()` (re-ingestion if hash changed) |
       | `deleted` | File removed → routed through `_handle_file()` (removes doc + vectors + graph) |
       | `moved`   | Split into two events:                           |
       |           | - Source path: treated as deleted (old document cleaned up) |
       |           | - Dest path: treated as created (new document ingested)   |

       > **Note:** File deletion removes the Document record, all its DocumentChunk rows, its Qdrant vectors, and Neo4j graph nodes — then commits all deletions in a single `db.commit()`. A Qdrant vector deletion failure logs a warning but does not abort the DB deletion (the DB deletion is the primary action; Qdrant cleanup is best-effort).
       
       > **Note:** The event type does **not** directly map 1-to-1 to `_ingest_file()` or `_update_document()`. All events are routed through `_handle_file()`, which then checks whether the Document exists and whether the hash has changed to decide between ingestion and re-ingestion. For example, a `created` event can still trigger `_update_document()` if a Document already exists with a different hash (e.g., a file is re-created with changed content).

       ### Per-datastore queueing

       All events are grouped by `datastore_id` into `pending_changes[datastore_id]`. This means events for different datastores are processed independently — a
       slow ingestion for one datastore doesn't block others.

       ### `_processing` flag

       Once `_process_pending_changes()` starts, it sets `self._processing` to include the `datastore_id` (a set, not a boolean). While processing, new events for the same datastore still get queued (they go into `pending_changes` but won't trigger a second processing run until the current one finishes and the `datastore_id` is removed from `_processing`). Events for **different** datastores are unaffected — they start processing independently.

       > **Note:** The batch timer has been removed entirely — event-driven processing happens immediately after the 1s write-completion delay, with no batching delay. The `_start_batch_timer()` method still exists but is never called in normal event-driven operation. It is only called by the old `_flush_batch` path which is also effectively dead code (removed to avoid 5-minute delays in event-driven processing).

       ---

       ## 2. Ingestion Pipeline

       ### For event-driven changes

       The flow uses **two classes** working together:
       - `DatastoreFileEventHandler` — the watchdog event handler with `_handle_file()`, `_dispatch()`, etc.
       - `DataStoreWatcher` — the service wrapper with `scan_single_datastore()`, `_init_scan()`, `_complete_scan()`, etc.
       - `_on_changes()` is called on `DatastoreFileEventHandler` which delegates to `DataStoreWatcher._on_changes()` which then delegates to `DatastoreFileEventHandler._handle_file()`.

       ```
       _process_pending_changes(datastore_id)
       │
       ├── Pops pending_changes[datastore_id] from queue
       ├── Sets _processing.add(datastore_id) (in finally block: .discard(datastore_id))
       │
       ├── _on_changes(datastore_id, org_id, changes)  [called on DatastoreFileEventHandler, delegates to DataStoreWatcher._on_changes, then back to DatastoreFileEventHandler._handle_file]
       │     For each change in changes:
       │       _handle_file(path, datastore_id, event_type)
       │         │
       │         ├── event_type == "deleted"
       │         │     _handle_deletion():
       │         │       - Delete DocumentChunk rows from DB
       │         │       - Delete ProcessingTask rows from DB
       │         │       - Delete Qdrant vectors from ds_{datastore_id}
       │         │       - Clean Neo4j graph nodes for this document
       │         │       - Delete Document row from DB
       │         │
       │         ├── event_type == "created" or "modified" (Document exists, hash changed)
       │         │     _update_document():
       │         │       - Update Document.file_hash, file_size
       │         │       - Reset ProcessingTask to "pending"
       │         │       - Submit to executor:
       │         │           process_document_background(temp_path=file_path, ...)
       │         │       - Returns Future
       │         │
       │         └── event_type == "created" (new Document, hash unchanged)
       │               _ingest_file():
       │                 - Check if Document already exists
       │                 - If hash unchanged → skip
       │                 - If hash changed → _update_document()
       │                 - If new → create Document + ProcessingTask
       │                 - Submit to executor:
       │                     process_document_background(temp_path=file_path, ...)
       │                 - Returns Future
       │
       └── _refresh_file_count(datastore_id)  [called from _on_changes, after processing all changes]
             Counts files on disk and updates DataStore.last_scan_total_files. Note: _update_scan_progress (called after each event ingestion) does `ds.last_scan_processed = processed` (direct assignment), while _on_changes does `ds.last_scan_processed += changes_processed` (accumulation). This means when a manual scan and event-driven processing overlap, they could double-count `last_scan_processed`.
       ```

       > **Note:** Event type does **not** directly map to `_ingest_file()` or `_update_document()`. All events are routed through `_handle_file()`, which checks
       Document existence and hash to decide between ingestion and re-ingestion. A `created` event can still trigger `_update_document()` if the Document already
       exists with a different hash.

       ### For manual scans

       **Scan a single datastore** (user clicks "Scan" on one datastore):

       ```
       POST /datastores/{id}/scan
       │
       ├── DataStoreWatcher.scan_single_datastore(datastore_id)
       │     │
       │     ├── _init_scan(datastore_id)
       │     │     - Assign scan_id
       │     │     - Count total files matching scan_pattern
       │     │     - Set DataStore.last_scan_status = "running"
       │     │
       │     ├── Walk all files in datastore folder
       │     │     For each file:
       │     │       _handle_file_in_scan(path, datastore_id, scan_id)
       │     │         │
       │     │         ├── Check if Document exists
       │     │         ├── If hash unchanged AND chunks exist → skip
       │     │         ├── If hash unchanged but NO chunks → re-ingest
       │     │         ├── If hash changed → re-ingest
       │     │         └── If new → ingest
       │     │           - Submit to executor:
       │     │               process_document_background(temp_path=file_path, ...)
       │     │           - Track Future in _scan_futures[scan_id]
       │     │
       │     ├── Wait for all ingestion Futures (up to 1 hour each)
       │     │
       │     └── _complete_scan(datastore_id, success=bool(errors==0))
             - Sets last_scan_new, last_scan_modified, last_scan_skipped, last_scan_errors from scan summary
       ```

       **Scan all datastores** (batch operation — not exposed as a REST endpoint):

       ```
       DataStoreWatcher.scan()
       │
       ├── Walk all active datastores' folders
       │     For each datastore:
       │       For each file:
       │         _handle_file(fpath, datastore_id, "created")
       │         (delegates to DatastoreFileEventHandler._handle_file)
       │     (Note: does NOT track scan progress or wait for ingestion Futures —
       │      fires-and-forget with no UI feedback)
       ```

       ---

       ## 3. UI Reflection

       ### Real-time updates via polling

       The frontend polls two endpoints:

       **`GET /api/admin/datastores`** (list with scan progress):

       ```
       GET /api/admin/datastores
       │
       ├── DataStoreWatcher.get_status() returns:
       │   - running: whether the watcher service is running
       │   - last_scan_at: last scan timestamp
       │   - files_scanned: total files scanned across all datastores
       │   - active_scans: list of active scan info dicts
       │   - datastores: list per-datastore with:
       │       - datastore_id, path, pending_changes, min_interval_seconds, processing
       │
       ├── For each datastore, the endpoint populates:
       │   - last_scan_status (never, running, completed, error, idle)
       │   - pending_changes (count from pending_changes[datastore_id])
       │   - processing (true while _process_pending_changes is running, per-datastore)
       │   - scan_progress (from active_scans when manual scan is running; from DB after completion)
       │
       ├── Polling frequency:
       │   - 2s when processing=true (event-driven processing active)
       │   - 500ms when a manual scan is active (scanPollRef)
       │   - 5s when last_scan_status='running' (background scan)
       │   - No polling when idle
       ```

       **`GET /api/admin/datastores/{id}/scan-progress`** (polling endpoint for scan progress):

       ```
       GET /api/admin/datastores/{id}/scan-progress
       │
       ├── If active scan exists in _active_scans:
       │   - Returns real-time scan progress from memory
       │   - new_files, modified_files, skipped_files, error_files from _active_scans
       │
       └── If no active scan:
           - Returns DB values: last_scan_new, last_scan_modified, last_scan_skipped, last_scan_errors
           - status = last_scan_status (never, running, completed, error, idle)
       ```

       **Why polling instead of SSE?** The SSE endpoint (`scan-progress-stream`) does not work through
       Next.js rewrites — the rewriter buffers `text/event-stream` responses instead of forwarding them.
       The SSE connection fails, causing the frontend to receive undefined values. Polling every 500ms
       is the reliable fallback.

       ### Scan result fields

       When a scan completes, the DataStore record and scan-progress endpoint include:

       | Field | Description |
       |-------|-------------|
       | `last_scan_new` | Number of previously unseen files ingested |
       | `last_scan_modified` | Number of existing files with changed content that were re-ingested |
       | `last_scan_skipped` | Number of files skipped (unsupported extension, pattern mismatch, hidden file) |
       | `last_scan_errors` | Number of files that failed ingestion |

       These fields are populated from `scan_single_datastore()`'s summary dict:
       `{scanned, new, modified, skipped, errors}`. They are persisted to the DB on scan completion
       by the POST /datastores/{id}/scan endpoint.

       ### What the UI shows

       | State                | Status badge                       | Files column                 | Actions               |
       |----------------------|------------------------------------|------------------------------|-----------------------|
       | Event processing     | Orange pulsing dot + "Processing"  | "Processing changes..."      | —                     |
       | Pending changes      | Yellow pulsing dot + N             | "N pending"                  | ⟳ Flush button        |
       | Manual scan running  | Blue pulsing dot + "Running"       | Progress bar                 | Stop button           |
       | Completed            | Green "Completed"                  | "N / M" (processed/total) + breakdown | Scan button |
       | Error                | Red "Error"                        | Error message + breakdown    | Scan button           |
       | Idle (no scans)      | Gray "—"                           | "N files"                    | Scan button           |

       ### The flush button (⟳)

       When `pending_changes > 0` and not currently processing, the flush button appears. Clicking it calls `POST /datastores/{id}/flush` which immediately
       processes all queued changes by calling `DataStoreWatcher._handler._process_pending_changes(datastore_id)`. This is useful when a user drops many files
       and wants to process them immediately rather than waiting for the debounce window.
       ---

       ## 4. Event-Driven + Manual Scan Coexistence

       These two modes are independent and don't conflict:

       |                       | Event-driven mode                              | Manual scan mode                             |
       |-----------------------|------------------------------------------------|----------------------------------------------|
       | Trigger               | File system event (inotify/FSEvents)           | User clicks "Scan" button                    |
       | Delay                 | 1s write-completion delay + 1s debounce        | None (starts immediately)                    |
       | Processing            | Immediate, per-file, in executor threads       | Sequential walk of all files                 |
       | Queue                 | `pending_changes[datastore_id]`, processed once per event | No queue — processes everything in one go |
       | UI                    | Orange "Processing" dot when active, yellow "N pending" for unprocessed | Progress bar, "Running" status |

       ### How they interact

       - Event-driven processing does not depend on manual scan being disabled. A datastore can have both: event-driven detection running in the background AND a
       manual scan triggered by the user.
       - Manual scan doesn't clear the event-driven queue. If a manual scan runs while files are being dropped, those files will still be in `pending_changes`
       and processed after the scan completes.
       - The `_processing` flag is now a **per-datastore set** (not a global boolean). It only guards against duplicate processing runs for the **same** datastore
       in event-driven mode — events for other datastores are unaffected. Manual scans don't use this flag at all.
       - Both use the same ingestion pipeline (`process_document_background`), so the results are identical.
       - `scan_single_datastore` waits for all ingestion Futures to complete, so the "Scan" button in the UI will show a progress bar until all files are done.

### When event-driven might be delayed

1. **Debounce window (1s):** If multiple events fire for the same file within 1 second, only the first one is processed.
2. **Write-completion delay (1s):** After the event fires, we wait 1 second before processing to ensure the file is fully written.
3. **`_processing` flag:** If the previous batch for **this** datastore is still being processed, new events get queued and will be processed after the
current batch finishes. Events for **other** datastores are unaffected.
4. **Manual scan in progress:** Manual scans don't block event-driven processing, but if many events fire during a scan, they'll queue up and be processed
afterward.

### Scan cancellation

The `POST /datastores/{id}/stop-scan` endpoint cancels a running manual scan. It works by setting `last_scan_status = "idle"` and `last_scan_error = "Scan cancelled by admin"` on the datastore, then setting `info["status"] = "cancelled"` in the `_active_scans` entry. The scan thread checks `_is_scan_cancelled()` between files and stops early. The cancelled entry is NOT removed from `_active_scans` — the SSE endpoint may still be reading from it and needs to find the cancelled entry to emit the final status event. Stale scans are cleaned up in `_init_scan` before a new scan starts.

       ---

       ## 5. Implementation Notes

       ### Two-class architecture

       The event-driven flow involves **two classes** collaborating:

       1. **`DatastoreFileEventHandler`** (extends `watchdog.events.FileSystemEventHandler`)
       - Receives filesystem events from the watchdog observer
       - Handles `_resolve_datastore()`, `_should_process()`, `_dispatch()`, `_queue_change()`, `_process_pending_changes()`, `_handle_file()`, etc.
       - Maintains `folder_paths`, `pending_changes`, `_processing`, `_last_call`, `_batch_timers`
       - Contains `_handle_deletion()`, `_ingest_file()`, `_update_document()`

       2. **`DataStoreWatcher`** (service wrapper)
       - Wraps `DatastoreFileEventHandler` and the watchdog Observer
       - Manages lifecycle: `start()`, `stop()`, `sync_watchers_with_database()`, `add_datastore()`, `remove_datastore()`
       - Manages scan tracking: `_init_scan()`, `_complete_scan()`, `_update_scan_progress()`, `scan_single_datastore()`, `scan()`
       - Manages status reporting: `get_status()`
       - Holds references to `_handler` (the DatastoreFileEventHandler) and `_active_scans` / `_scan_futures`

       ### `_SyntheticEvent` class

       When `_dispatch()` creates the 1s write-completion delay, it spawns a daemon thread that sleeps 1s and then creates a `_SyntheticEvent` object. This
       synthetic event has `src_path`, `is_directory=False`, and `event_type` — mimicking a watchdog event so that `_queue_change()` and `_process_pending_changes()`
       work without modification. This is needed because the original watchdog event object may have changed state by the time the 1s delay expires. The race condition
       this prevents:

       1. File is created → watchdog fires `on_created` event with `src_path=/app/data/test.pdf`
       2. File write continues (writing actual content to the file)
       3. 1s delay expires → the daemon thread creates a `_SyntheticEvent` with the same `src_path`
       4. If the original watchdog event were used, it might reference the old file state (e.g., the file was renamed/deleted between events)
       5. The synthetic event guarantees the path and type are captured at the time of detection, not at the time of processing

       ### `_flush_batch()` method

       A `_flush_batch()` method exists on `DatastoreFileEventHandler` that pops `pending_changes` and calls the callback. It's used when `remove_folder()` is
       invoked (flushing pending changes on datastore teardown). It is **never** called by a timer during normal event-driven operation — the batch timer has been
       removed entirely to avoid 5-minute delays. The `_start_batch_timer()` method still exists but is never called in normal operation.

       ### `DataStoreWatcher.scan()` method (batch scan)

       The `DataStoreWatcher` has a `scan()` method that walks all active datastores' folders and calls `_handle_file(fpath, datastore_id, "created")` for each
       file. This fires-and-forgets — it does **not** track scan progress, does **not** wait for ingestion Futures, and does **not** have a REST endpoint. It's
       a legacy/internal method not used by the UI.