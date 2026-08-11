from pydantic import BaseModel
from typing import Any, Optional, List


class SettingItem(BaseModel):
    """A single setting with metadata for API responses."""
    key: str
    value: Any
    value_type: str
    category: str
    label: str
    scope: str
    source: Optional[str] = None          # "database" | "install_default" (app endpoint)
    overridden: Optional[bool] = None      # org endpoint only
    app_default: Optional[Any] = None      # org endpoint only
    effective: Optional[Any] = None        # org endpoint only
    reload: Optional[str] = None
    requires_reindex: bool = False
    description: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[str]] = None


class SettingsListResponse(BaseModel):
    settings: List[SettingItem]


class SettingUpdate(BaseModel):
    key: str
    value: Any


class SettingsBulkUpdate(BaseModel):
    settings: List[SettingUpdate]


class SettingSchemaItem(BaseModel):
    """Registry metadata for UI form generation."""
    key: str
    value_type: str
    category: str
    label: str
    scope: str
    reload: str
    requires_reindex: bool
    description: str
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[str]] = None


class SettingsSchemaResponse(BaseModel):
    settings: List[SettingSchemaItem]
