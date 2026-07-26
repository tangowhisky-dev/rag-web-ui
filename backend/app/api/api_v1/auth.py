import logging
import time
from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from requests.exceptions import RequestException

from app.core import security
from app.core.security import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models.user import User
from app.schemas.token import Token
from app.schemas.user import UserCreate, UserResponse, PasswordChange

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Rate limiting for login attempts ──────────────────────────────────────
# Tracks failed login attempts per IP address with exponential backoff.
# Format: { ip: { "attempts": int, "first_attempt_time": float, "backoff_until": float | None, "backoff_level": int } }
# backoff_level: how many times backoff has escalated (0 = first backoff, 1 = second, etc.)
# NOTE: In-progress redesign — current version has issues with correct-login reset and post-expiry escalation.
_failed_login_attempts: dict[str, dict[str, Any]] = {}

# Max attempts before first backoff kicks in
MAX_LOGIN_ATTEMPTS = 3
# Exponential backoff: 15s, 30s, 60s, 120s, 240s, 480s, 900s...
BASE_BACKOFF_SECONDS = 15
MAX_BACKOFF_SECONDS = 900


def _get_client_ip(request: Request) -> str:
    """Extract client IP, trusting proxy headers only for known proxies.

    Priority when peer is trusted: X-Real-IP > X-Forwarded-For > peer IP.
    """
    peer = request.client.host if request.client else None
    trusted = {p.strip() for p in settings.TRUSTED_PROXIES.split(",") if p.strip()}
    if "*" in trusted or peer in trusted:
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    return peer or "unknown"


def _check_rate_limit(ip: str) -> tuple[bool, int]:
    """Check if IP is rate limited.

    Returns:
        (is_limited, retry_after_seconds)
    """
    now = time.time()

    # Clean up entries older than 10 minutes (prevent memory leak)
    expired_ips = [
        ip_key for ip_key, data in _failed_login_attempts.items()
        if now - data["first_attempt_time"] > 600
    ]
    for ip_key in expired_ips:
        del _failed_login_attempts[ip_key]

    if ip not in _failed_login_attempts:
        return False, 0

    data = _failed_login_attempts[ip]
    backoff_until = data.get("backoff_until")

    # If still in backoff window, reject
    if backoff_until and now < backoff_until:
        retry_after = max(0, int(backoff_until - now))
        return True, retry_after

    # Backoff expired — increment level and immediately apply next backoff
    if backoff_until:
        data["backoff_level"] = data.get("backoff_level", 0) + 1
        data["backoff_until"] = None
        # Apply escalated backoff immediately if there are enough failures
        if data["attempts"] >= MAX_LOGIN_ATTEMPTS:
            backoff = min(
                BASE_BACKOFF_SECONDS * (2 ** data["backoff_level"]),
                MAX_BACKOFF_SECONDS
            )
            data["backoff_until"] = now + backoff
            return True, int(backoff)

    return False, 0


def _record_failed_attempt(ip: str) -> int:
    """Record a failed login attempt for the given IP.

    Returns:
        Current attempt count for this IP
    """
    now = time.time()

    if ip not in _failed_login_attempts:
        _failed_login_attempts[ip] = {
            "attempts": 1,
            "first_attempt_time": now,
            "backoff_until": None,
            "backoff_level": 0,
        }
    else:
        _failed_login_attempts[ip]["attempts"] += 1

    data = _failed_login_attempts[ip]

    # First backoff: triggers at MAX_LOGIN_ATTEMPTS failures
    # Subsequent backoffs are applied in _check_rate_limit when previous backoff expires
    if data["attempts"] >= MAX_LOGIN_ATTEMPTS and not data.get("backoff_until"):
        backoff = min(
            BASE_BACKOFF_SECONDS * (2 ** data.get("backoff_level", 0)),
            MAX_BACKOFF_SECONDS
        )
        data["backoff_until"] = now + backoff

    return data["attempts"]

def _reset_failed_attempts(ip: str) -> None:
    """Reset failed attempts after successful login."""
    if ip in _failed_login_attempts:
        del _failed_login_attempts[ip]

@router.post("/register", response_model=UserResponse)
def register(*, db: Session = Depends(get_db), user_in: UserCreate) -> Any:
    """Public registration is disabled. Users must be created by an admin or the seed process."""
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Public registration is disabled.",
    )

@router.post("/token", response_model=Token)
def login_access_token(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    Rate limited: 3 attempts max, then exponential backoff.
    Sets the token in an HttpOnly cookie for browser clients.
    """
    client_ip = _get_client_ip(request)

    # Check rate limit
    is_limited, retry_after = _check_rate_limit(client_ip)
    if is_limited:
        logger.warning(
            "[AUTH] rate_limited ip=%s attempts=%d retry_after=%ds",
            client_ip,
            _failed_login_attempts[client_ip]["attempts"],
            retry_after,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Please try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        # Record failed attempt
        attempts = _record_failed_attempt(client_ip)
        logger.warning(
            "[AUTH] login_failed ip=%s username=%s attempts=%d",
            client_ip,
            form_data.username,
            attempts,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    elif not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Successful login — reset failed attempts
    _reset_failed_attempts(client_ip)

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={
            "sub": user.username,
            "role": user.role.value,
            "org_id": user.org_id,
            "token_version": user.token_version,
        },
        expires_delta=access_token_expires,
    )
    response.set_cookie(
        key="token",
        value=access_token,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        path="/",
    )
    logger.info("[AUTH] token_issued username=%s role=%s org_id=%s", user.username, user.role.value, user.org_id)
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/admin-only", response_model=UserResponse)
def admin_only(current_user: User = Depends(security.require_admin)) -> Any:
    return current_user


@router.post("/change-password", status_code=200)
def change_password(
    payload: PasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Change current user's password. Invalidates existing tokens by bumping token_version.
    """
    if not security.verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    current_user.hashed_password = security.get_password_hash(payload.new_password)
    current_user.token_version += 1
    db.commit()
    db.refresh(current_user)
    logging.info("[AUTH] password_changed username=%s user_id=%s", current_user.username, current_user.id)
    return {"message": "Password changed successfully. Please log in again."}


@router.post("/test-token", response_model=UserResponse)
def test_token(current_user: User = Depends(get_current_user)) -> Any:
    """
    Test access token by getting current user.
    """
    return current_user


@router.post("/logout", status_code=200)
def logout(response: Response) -> Any:
    """Clear the HttpOnly auth cookie."""
    response.delete_cookie(key="token", path="/")
    return {"message": "Logged out"}
