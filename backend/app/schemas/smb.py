"""Pydantic schemas for SMB share management API endpoints."""

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class SMBConfigRequest(BaseModel):
    """Request body for configuring or updating an SMB share."""

    host: str = Field(..., description="SMB server hostname or IP address")
    share: str = Field(..., description="SMB share name")
    username: str = Field(..., description="Username for SMB authentication")
    password: str = Field(..., description="Password for SMB authentication")
    domain: Optional[str] = Field(None, description="Windows domain or workgroup (optional)")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class SMBConfigResponse(BaseModel):
    """Response after saving or clearing SMB configuration."""

    org_id: int
    smb_host: Optional[str] = None
    smb_share: Optional[str] = None
    smb_username: Optional[str] = None
    smb_domain: Optional[str] = None
    status: str  # "configured" | "not_configured"
    last_scan_at: Optional[float] = None
    last_scan_status: Optional[str] = None
    last_scan_files: int = 0

    class Config:
        from_attributes = True


class SMBTestConnectionResponse(BaseModel):
    """Response from testing an SMB connection."""

    connected: bool
    share_accessible: bool
    error: Optional[str] = None


class SMBScanResponse(BaseModel):
    """Response from a manual SMB share scan."""

    scanned: int
    new: int
    skipped: int
    errors: int
