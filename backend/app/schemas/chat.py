from pydantic import BaseModel, model_validator, field_serializer, ConfigDict
from typing import Any, List, Optional
from datetime import datetime


def _as_utc_iso(dt: datetime) -> str:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class MessageBase(BaseModel):
    content: str
    role: str


class MessageCreate(MessageBase):
    chat_id: int


class MessageEditRequest(BaseModel):
    content: str


class MessageResponse(MessageBase):
    id: int
    chat_id: int
    parent_message_id: Optional[int] = None
    branch_index: int = 0
    created_at: datetime
    updated_at: datetime
    confidence_level: Optional[str] = None
    confidence_score: Optional[int] = None
    confidence_breakdown: Optional[str] = None  # JSON string, parsed by frontend
    # Final answer evaluation (from answer_evaluation_node)
    final_confidence: Optional[float] = None
    final_confidence_level: Optional[str] = None
    faithfulness: Optional[int] = None
    completeness: Optional[int] = None
    retrieval_score: Optional[int] = None
    rewritten_query: Optional[str] = None
    file_name: Optional[str] = None  # filename if a chat file was attached to this message
    file_id: Optional[int] = None      # chat_files.id — used to build download URL
    # citations is added by the API endpoint after model_validate

    @field_serializer("created_at", "updated_at")
    def serialise_datetimes(self, v): return _as_utc_iso(v)

    model_config = ConfigDict(from_attributes=True)


class ChatBase(BaseModel):
    title: str


class ChatCreate(ChatBase):
    knowledge_base_ids: List[int]


class ChatUpdate(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None
    knowledge_base_ids: Optional[List[int]] = None


class FolderCreate(BaseModel):
    name: str


class FolderUpdate(BaseModel):
    name: Optional[str] = None


class FolderResponse(BaseModel):
    id: int
    name: str
    user_id: int
    created_at: datetime

    @field_serializer("created_at")
    def serialise_created_at(self, v): return _as_utc_iso(v)

    model_config = ConfigDict(from_attributes=True)


class SearchResult(BaseModel):
    chat_id: int
    chat_title: str
    snippet: str
    message_id: int


class KbInfo(BaseModel):
    id: int
    name: str


class ChatResponse(ChatBase):
    id: int
    user_id: int
    folder_id: Optional[int] = None
    pinned:        bool = False
    temperature:   Optional[float] = None
    model_name:    Optional[str] = None
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []
    knowledge_base_ids: List[int] = []
    knowledge_bases: List[KbInfo] = []

    @field_serializer("created_at", "updated_at")
    def serialise_datetimes(self, v): return _as_utc_iso(v)

    @model_validator(mode='before')
    @classmethod
    def extract_kb_info(cls, data: Any) -> Any:
        if hasattr(data, 'knowledge_bases'):
            data.__dict__['knowledge_base_ids'] = [kb.id for kb in data.knowledge_bases]
            data.__dict__['knowledge_bases'] = [{"id": kb.id, "name": kb.name} for kb in data.knowledge_bases]
        return data

    model_config = ConfigDict(from_attributes=True)
