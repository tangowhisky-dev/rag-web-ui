from typing import Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, field_serializer, ConfigDict


def _as_utc_iso(dt: datetime) -> str:
    """Serialise a naive-UTC datetime to ISO 8601 with Z suffix.
    MySQL stores datetimes without timezone info but they are always UTC."""
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

class KnowledgeBaseBase(BaseModel):
    name: str
    description: Optional[str] = None

class KnowledgeBaseCreate(KnowledgeBaseBase):
    pass

class KnowledgeBaseUpdate(KnowledgeBaseBase):
    pass

class DocumentBase(BaseModel):
    file_name: str
    file_path: str
    file_hash: str
    file_size: int
    content_type: str

class DocumentCreate(DocumentBase):
    knowledge_base_id: int

class DocumentUploadBase(BaseModel):
    file_name: str
    file_hash: str
    file_size: int
    content_type: str
    temp_path: str
    status: str = "pending"
    error_message: Optional[str] = None

class DocumentUploadCreate(DocumentUploadBase):
    knowledge_base_id: int

class DocumentUploadResponse(DocumentUploadBase):
    id: int
    created_at: datetime
    
    @field_serializer("created_at")
    def serialise_created_at(self, v): return _as_utc_iso(v)

    model_config = ConfigDict(from_attributes=True)

class ProcessingTaskBase(BaseModel):
    status: str
    error_message: Optional[str] = None

class ProcessingTaskCreate(ProcessingTaskBase):
    document_id: int
    knowledge_base_id: int

class ProcessingTask(ProcessingTaskBase):
    id: int
    document_id: Optional[int] = None
    knowledge_base_id: int
    progress: Optional[int] = None
    progress_message: Optional[str] = None
    graph_status: Optional[str] = None
    graph_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def serialise_datetimes(self, v): return _as_utc_iso(v)

    model_config = ConfigDict(from_attributes=True)

class DocumentResponse(DocumentBase):
    id: int
    knowledge_base_id: int
    created_at: datetime
    updated_at: datetime
    processing_tasks: List[ProcessingTask] = []

    @field_serializer("created_at", "updated_at")
    def serialise_datetimes(self, v): return _as_utc_iso(v)

    model_config = ConfigDict(from_attributes=True)

class DataStoreInfo(BaseModel):
    id: int
    name: str
    folder_path: str
    auto_process_enabled: bool = False
    document_count: int = 0

    model_config = ConfigDict(from_attributes=True)

class KnowledgeBaseResponse(KnowledgeBaseBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    documents: List[DocumentResponse] = []
    data_sources: List[DataStoreInfo] = []
    data_source_count: int = 0

    @field_serializer("created_at", "updated_at")
    def serialise_datetimes(self, v): return _as_utc_iso(v)

    model_config = ConfigDict(from_attributes=True)

class PreviewRequest(BaseModel):
    document_ids: List[int]
    # When omitted the server uses CHUNK_SIZE / OVERLAP_PERCENTAGE from .env.
    # Explicitly passing values here overrides the defaults for this preview only
    # and does NOT affect what is used during actual ingestion.
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None 