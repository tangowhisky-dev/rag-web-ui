from pydantic import BaseModel, EmailStr, field_serializer
from typing import Optional
from datetime import datetime


def _as_utc_iso(dt: datetime) -> str:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class UserBase(BaseModel):
    email: str
    username: str
    is_active: bool = True

class UserCreate(UserBase):
    password: str

class UserUpdate(UserBase):
    password: Optional[str] = None


class UserAdminCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"
    org_id: int
    is_active: bool = True


class UserAdminUpdate(BaseModel):
    role: Optional[str] = None
    org_id: Optional[int] = None
    is_active: Optional[bool] = None


class UserDeleteResponse(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


class UserResponse(UserBase):
    id: int
    role: Optional[str] = None
    org_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def serialise_datetimes(self, v): return _as_utc_iso(v)

    class Config:
        from_attributes = True
