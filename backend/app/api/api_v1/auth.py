import logging
from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from requests.exceptions import RequestException

from app.core import security
from app.core.security import get_current_user
from app.core.config import settings
from app.core.rate_limiter import check_rate_limit, record_failed_attempt, reset_failed_attempts
from app.db.session import get_db
from app.models.user import User, UserRole
from app.schemas.token import Token, HeartbeatResponse
from app.schemas.user import UserCreate, UserResponse, PasswordChange

logger = logging.getLogger(__name__)

router = APIRouter()


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
    is_limited, retry_after = check_rate_limit(client_ip)
    if is_limited:
        logger.warning(
            "[AUTH] rate_limited ip=%s retry_after=%ds",
            client_ip,
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
        attempts = record_failed_attempt(client_ip)
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

    if not user.is_active:
        attempts = record_failed_attempt(client_ip)
        logger.warning(
            "[AUTH] login_failed ip=%s username=%s attempts=%d",
            client_ip,
            form_data.username,
            attempts,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Successful login — reset failed attempts
    reset_failed_attempts(client_ip)

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
        secure=settings.COOKIE_SECURE,
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


@router.get("/test-token", response_model=UserResponse)
def test_token(current_user: User = Depends(get_current_user)) -> Any:
    """
    Test access token by getting current user.
    """
    return current_user


@router.get("/preflight")
def preflight(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Post-login settings validation.

    Returns a list of settings that are unset and would break functionality
    for the authenticated user's role and org. The frontend uses this to
    show a popup after login if any required settings are missing.
    """
    from app.services.settings_preflight import check_required_settings
    result = check_required_settings(db, current_user.role.value, current_user.org_id)
    return result.to_dict()


@router.post("/logout", status_code=200)
def logout(response: Response) -> Any:
    """Clear the HttpOnly auth cookie."""
    response.delete_cookie(key="token", path="/", secure=settings.COOKIE_SECURE)
    return {"message": "Logged out"}


@router.get("/heartbeat", response_model=HeartbeatResponse)
def heartbeat(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Session heartbeat with active-work awareness.

    Called by the frontend every 5 minutes to keep the session alive.
    The endpoint itself is excluded from the sliding renewal in
    ``get_current_user`` — instead, it renews the token only when
    there is active background work for the user. This means:

    - Active users (making real API calls) get their token renewed by
      ``get_current_user`` on every request.
    - Idle users with no active work: token expires after the lifetime.
    - Idle users with active work (ingestion, graph build): token is
      renewed here so they stay logged in until work completes.
    """
    has_work = _check_active_work(db, current_user)

    if has_work:
        # Renew the token — background work is running, keep session alive.
        new_token = security.create_access_token(
            data={
                "sub": current_user.username,
                "role": current_user.role.value,
                "org_id": current_user.org_id,
                "token_version": current_user.token_version,
            }
        )
        response.set_cookie(
            key="token",
            value=new_token,
            httponly=True,
            max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            samesite="lax",
            secure=settings.COOKIE_SECURE,
            path="/",
        )

    return {"has_active_work": has_work}


def _check_active_work(db: Session, user: User) -> bool:
    """Check whether the user has any active background work.

    Active work = ProcessingTask with status in (pending, processing)
    or graph_status in (pending, processing), scoped to the user's
    accessible datastores and knowledge bases.
    """
    from app.models.knowledge import ProcessingTask, KnowledgeBase
    from app.models.datastore import DataStore, OrganizationDataStore
    from app.core.security import get_admin_org_ids

    active_statuses = ("pending", "processing")

    if user.role == UserRole.super_admin:
        # Super admin: any active task counts.
        return (
            db.query(ProcessingTask)
            .filter(
                ProcessingTask.status.in_(active_statuses)
                | ProcessingTask.graph_status.in_(active_statuses)
            )
            .first()
            is not None
        )

    # Admin: tasks for datastores in their org hierarchy + KBs in their org.
    # Regular user: tasks for KBs they own.
    if user.role == UserRole.admin:
        admin_org_ids = get_admin_org_ids(db, user)
        if admin_org_ids is None:
            # No org restriction — same as super admin.
            return (
                db.query(ProcessingTask)
                .filter(
                    ProcessingTask.status.in_(active_statuses)
                    | ProcessingTask.graph_status.in_(active_statuses)
                )
                .first()
                is not None
            )
        if not admin_org_ids:
            return False

        # Datastores assigned to the user's org hierarchy.
        ds_ids = [
            row[0]
            for row in db.query(OrganizationDataStore.data_store_id)
            .filter(OrganizationDataStore.org_id.in_(admin_org_ids))
            .all()
        ]
        # KBs in the user's org hierarchy.
        kb_ids = [
            row[0]
            for row in db.query(KnowledgeBase.id)
            .filter(KnowledgeBase.org_id.in_(admin_org_ids))
            .all()
        ]
    else:
        # Regular user: only their own KBs.
        ds_ids = []
        kb_ids = [
            row[0]
            for row in db.query(KnowledgeBase.id)
            .filter(KnowledgeBase.user_id == user.id)
            .all()
        ]

    if not ds_ids and not kb_ids:
        return False

    query = db.query(ProcessingTask).filter(
        ProcessingTask.status.in_(active_statuses)
        | ProcessingTask.graph_status.in_(active_statuses)
    )
    if ds_ids and kb_ids:
        query = query.filter(
            (ProcessingTask.data_store_id.in_(ds_ids))
            | (ProcessingTask.knowledge_base_id.in_(kb_ids))
        )
    elif ds_ids:
        query = query.filter(ProcessingTask.data_store_id.in_(ds_ids))
    elif kb_ids:
        query = query.filter(ProcessingTask.knowledge_base_id.in_(kb_ids))

    return query.first() is not None
