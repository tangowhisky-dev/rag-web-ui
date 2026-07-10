from pydantic import BaseModel, EmailStr, field_serializer, field_validator, ConfigDict
from typing import Optional
from datetime import datetime
from app.core.security import validate_password_strength


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

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        error = validate_password_strength(v)
        if error:
            raise ValueError(error)
        return v


class UserUpdate(UserBase):
    password: Optional[str] = None


class UserAdminCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"
    org_id: int
    is_active: bool = True

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        error = validate_password_strength(v)
        if error:
            raise ValueError(error)
        return v


class UserAdminUpdate(BaseModel):
    role: Optional[str] = None
    org_id: Optional[int] = None
    is_active: Optional[bool] = None


class UserDeleteResponse(BaseModel):
    id: int
    username: str
    email: str

    model_config = ConfigDict(from_attributes=True)


class UserResponse(UserBase):
    id: int
    role: Optional[str] = None
    org_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def serialise_datetimes(self, v): return _as_utc_iso(v)

    model_config = ConfigDict(from_attributes=True)


class PasswordChange(BaseModel):
    new_password: str

    @field_validator('new_password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        error = validate_password_strength(v)
        if error:
            raise ValueError(error)
        return v
