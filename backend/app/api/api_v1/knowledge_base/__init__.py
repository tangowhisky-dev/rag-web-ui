"""Knowledge-base API endpoints.

Endpoints (prefix ``/api/knowledge-base``):
    POST   /                                   - create knowledge base
    GET    /                                   - list knowledge bases
    GET    /ocr-availability                   - check OCR availability
    GET    /available-datastores               - list linkable datastores
    GET    /documents/{doc_id}                 - get document by ID
    GET    /documents/{doc_id}/download        - download document by ID
    GET    /{kb_id}                            - get knowledge base
    PUT    /{kb_id}                            - update knowledge base
    DELETE /{kb_id}                            - delete knowledge base
    POST   /{kb_id}/documents/upload           - upload documents
    POST   /{kb_id}/documents/preview          - preview document chunks
    POST   /{kb_id}/documents/process          - process uploaded documents
    POST   /cleanup                            - clean up expired temp files
    GET    /{kb_id}/documents/tasks            - get processing task statuses
    DELETE /{kb_id}/documents/{doc_id}         - delete a document
    POST   /{kb_id}/documents/{doc_id}/retry   - retry failed ingestion
    GET    /{kb_id}/documents/{doc_id}         - get document details
    GET    /{kb_id}/documents/{doc_id}/download - download document
    POST   /{kb_id}/link-datastore             - link datastore to KB
    DELETE /{kb_id}/unlink-datastore/{ds_id}   - unlink datastore from KB

This package was split from a single ``knowledge_base.py`` module.  The
``router`` is defined here and each submodule imports it to register
routes.  Shared helpers live in ``helpers`` and are re-exported here so
that existing import paths (e.g. ``from app.api.api_v1.knowledge_base
import _get_user_org_ids``) continue to work.
"""

from fastapi import APIRouter

router = APIRouter()

# Re-export helpers so existing import paths keep working.
from app.api.api_v1.knowledge_base.helpers import (  # noqa: E402, F401
    _get_user_org_ids,
    _kb_owner_filter,
    _file_chunks,
    _get_chunk_scope_filter,
    _delete_qdrant_points,
)

# Import submodules to register their routes on ``router``.
from app.api.api_v1.knowledge_base import kb  # noqa: E402, F401
from app.api.api_v1.knowledge_base import datastores  # noqa: E402, F401
from app.api.api_v1.knowledge_base import documents  # noqa: E402, F401
from app.api.api_v1.knowledge_base import ingestion  # noqa: E402, F401
