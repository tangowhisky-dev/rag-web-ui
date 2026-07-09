import sys
sys.path.insert(0, '/Users/tango16/code/rag-web-ui/backend')

# Test datastore_watcher package
try:
    from app.services.datastore_watcher import DataStoreWatcher, DatastoreFileEventHandler, _Debouncer, _SyntheticEvent
    print("OK datastore_watcher package imports OK")
except Exception as e:
    print(f"FAIL datastore_watcher import failed: {e}")

# Test document modules
try:
    from app.services.ingestion import (
    process_document_background,
    upload_document,
    SUPPORTED_EXTENSIONS,
)
    from app.services.ingestion import (
    _convert_to_markdown,
    CONTENT_TYPE_MAP,
    SUPPORTED_EXTENSIONS as SE2,
)
    from app.services.ingestion import (
    _chunk_id_to_point_id,
    PreviewResult,
    UploadResult,
)
    print("OK document_processor/converter/qdrant imports OK")
except Exception as e:
    print(f"FAIL document modules import failed: {e}")

# Test API routers
try:
    from app.api.api_v1 import datastores, datastore_scan, datastore_recovery
    print("OK API routers import OK")
except Exception as e:
    print(f"FAIL API routers import failed: {e}")

print("\nAll imports checked.")
