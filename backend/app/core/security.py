from datetime import datetime, timedelta, timezone
from typing import List, Optional
from jose import JWTError, jwt
import bcrypt
from app.core.config import settings
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.user import User, UserRole
from app.models.organisation import Organisation

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login/access-token", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def get_org_descendants(db: Session, org_id: int) -> List[int]:
    """Return [org_id, all descendant org IDs] via BFS over the hierarchy.

    Safety-capped at 100 iterations to prevent infinite loops on cyclic data.
    """
    org_ids = [org_id]
    to_check = [org_id]
    for _ in range(100):
        if not to_check:
            break
        parent_id = to_check.pop(0)
        children = db.query(Organisation).filter(Organisation.parent_id == parent_id).all()
        for child in children:
            if child.id not in org_ids:
                org_ids.append(child.id)
                to_check.append(child.id)
    return org_ids


def get_admin_org_ids(db: Session, current_user: User) -> Optional[List[int]]:
    """Return the org IDs a non-super admin is allowed to manage.

    - super_admin: None (no org-level scoping restriction)
    - admin with org_id: [own org, all descendant orgs]
    - users without an org: empty list
    """
    if current_user.role == UserRole.super_admin:
        return None
    if current_user.org_id is None:
        return []
    return get_org_descendants(db, current_user.org_id)


def validate_password_strength(password: str) -> str:
    """Validate password strength.

    Returns empty string if valid, or a human-readable error message.
    Rules: minimum 8 characters, must contain at least one letter and one digit.
    """
    if len(password) < 8:
        return "Password must be at least 8 characters"
    if not any(c.isalpha() for c in password):
        return "Password must contain at least one letter"
    if not any(c.isdigit() for c in password):
        return "Password must contain at least one number"
    return ""

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt 

def get_current_user(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        token = request.cookies.get("token")
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        token_version = payload.get("token_version")
        if username is None or token_version is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None or user.token_version != token_version:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Sliding renewal: if the token has less than 50% of its lifetime
    # remaining, mint a new one and set it in the response cookie. This
    # keeps active users logged in without a refresh token. The heartbeat
    # endpoint is excluded — it does its own conditional renewal based on
    # whether background work is running.
    if not request.url.path.endswith("/heartbeat"):
        exp = payload.get("exp")
        if exp is not None:
            remaining = int(exp) - int(datetime.now(timezone.utc).timestamp())
            if remaining < settings.ACCESS_TOKEN_EXPIRE_MINUTES * 30:  # < 50%
                new_token = create_access_token(
                    data={
                        "sub": user.username,
                        "role": user.role.value,
                        "org_id": user.org_id,
                        "token_version": user.token_version,
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

    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role not in (UserRole.admin, UserRole.super_admin):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def require_super_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.super_admin:
        raise HTTPException(status_code=403, detail="Super admin access required")
    return current_user