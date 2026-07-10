from pydantic import BaseModel, field_serializer, ConfigDict
from typing import Optional
from datetime import datetime


def _as_utc_iso(dt: datetime) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class OrgBase(BaseModel):
    name: str


class OrgCreate(OrgBase):
    parent_id: int


class OrgUpdate(BaseModel):
    name: Optional[str] = None
    parent_id: Optional[int] = None
    remove_parent: bool = False


class OrgResponse(OrgBase):
    id: int
    parent_id: Optional[int] = None
    path: Optional[str] = None
    level: int = 0
    user_count: int = 0
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def serialise_datetimes(self, v: datetime) -> Optional[str]:
        return _as_utc_iso(v)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# OrgLLMConfig schemas
# ---------------------------------------------------------------------------

class OrgLLMConfigBase(BaseModel):
    api_base: Optional[str] = None
    model_name: Optional[str] = None
    query_model: Optional[str] = None


class OrgLLMConfigCreate(OrgLLMConfigBase):
    pass


class OrgLLMConfigUpdate(OrgLLMConfigBase):
    pass


class OrgLLMConfigResponse(OrgLLMConfigBase):
    org_id: int

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Org ingestion status schema
# ---------------------------------------------------------------------------

from typing import Literal


class OrgIngestionStatusResponse(BaseModel):
    org_id: int
    status: Literal["idle", "running", "completed", "failed"]
    total_docs: int
    pending_docs: int
    processing_docs: int
    completed_docs: int
    failed_docs: int
    last_run_at: Optional[datetime] = None

    @field_serializer("last_run_at")
    def serialise_last_run_at(self, v: Optional[datetime]) -> Optional[str]:
        return _as_utc_iso(v) if v else None

