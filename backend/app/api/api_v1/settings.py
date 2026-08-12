"""
settings.py — Super Admin and Admin settings API endpoints.

Super Admin (app-level):
  GET    /api/admin/settings                         → list all app settings
  GET    /api/admin/settings/schema                  → registry metadata for UI
  PUT    /api/admin/settings                         → bulk upsert app settings
  POST   /api/admin/settings/{key}                   → upsert single app setting
  DELETE /api/admin/settings/{key}                   → reset app setting to .env default
  GET    /api/admin/settings/effective               → full effective app config snapshot

Admin (org-level):
  GET    /api/admin/orgs/{org_id}/settings           → list org settings (with override flags)
  GET    /api/admin/orgs/{org_id}/settings/schema    → registry metadata for org-overridable keys
  PUT    /api/admin/orgs/{org_id}/settings           → bulk upsert org overrides
  POST   /api/admin/orgs/{org_id}/settings/{key}     → upsert single org override
  DELETE /api/admin/orgs/{org_id}/settings/{key}     → delete org override
  DELETE /api/admin/orgs/{org_id}/settings           → clear all org overrides
"""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import require_super_admin, require_admin, get_admin_org_ids
from app.db.session import get_db
from app.models.user import User
from app.models.organisation import Organisation
from app.core.settings_registry import REGISTRY, get_def, is_org_overridable
from app.services.settings_service import (
    get_all_app_settings_with_meta,
    get_all_org_settings_with_meta,
    upsert_app_setting,
    upsert_org_setting,
    reset_app_setting,
    reset_org_setting,
    reset_all_org_settings,
    validate_value,
    clear_cache,
)
from app.schemas.setting import (
    SettingItem, SettingsListResponse, SettingUpdate,
    SettingsBulkUpdate, SettingSchemaItem, SettingsSchemaResponse,
)

logger = logging.getLogger(__name__)

app_router = APIRouter()
org_router = APIRouter()


# ── Super Admin: app-level settings ───────────────────────────────────────

@app_router.get("/settings", response_model=SettingsListResponse)
def list_app_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """List all app-level settings with metadata and effective values."""
    items = get_all_app_settings_with_meta(db)
    return SettingsListResponse(settings=[SettingItem(**item) for item in items])


@app_router.get("/settings/schema", response_model=SettingsSchemaResponse)
def get_app_settings_schema(
    current_user: User = Depends(require_super_admin),
):
    """Return registry metadata for UI form generation."""
    items = []
    for d in REGISTRY:
        items.append(SettingSchemaItem(
            key=d.key,
            value_type=d.value_type,
            category=d.category,
            label=d.label,
            scope=d.scope,
            reload=d.reload,
            requires_reindex=d.requires_reindex,
            description=d.description,
            min=d.min_value,
            max=d.max_value,
            choices=list(d.choices) if d.choices else None,
            secret=d.secret,
            model_picker=d.model_picker,
            api_base_ref=d.api_base_ref,
            api_key_ref=d.api_key_ref,
        ))
    return SettingsSchemaResponse(settings=items)


@app_router.put("/settings")
def update_app_settings(
    payload: SettingsBulkUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Bulk upsert app-level settings."""
    results = []
    for item in payload.settings:
        try:
            upsert_app_setting(db, item.key, item.value, user_id=current_user.id)
            results.append({"key": item.key, "status": "ok"})
        except ValueError as e:
            results.append({"key": item.key, "status": "error", "detail": str(e)})
    return {"results": results}


@app_router.post("/settings/{key}")
def update_app_setting(
    key: str,
    payload: SettingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Upsert a single app-level setting."""
    if payload.key != key:
        raise HTTPException(status_code=400, detail="Key in path must match key in body")
    try:
        upsert_app_setting(db, key, payload.value, user_id=current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"key": key, "status": "ok"}


@app_router.delete("/settings/{key}")
def delete_app_setting(
    key: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Reset an app setting to its .env/config.py default."""
    try:
        reset_app_setting(db, key)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"key": key, "status": "reset"}


@app_router.get("/settings/effective")
def get_effective_app_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Return the fully resolved app config snapshot (for debugging)."""
    from app.services.settings_service import get_org_settings
    return get_org_settings(db, None)


def _fetch_models_from_endpoint(api_base: str, api_key: str | None) -> list[str]:
    """Call an OpenAI-compatible /models endpoint and return model IDs."""
    from openai import SyncOpenAI
    if not api_key:
        api_key = "not-required"
    client = SyncOpenAI(api_key=api_key, base_url=api_base)
    models = client.models.list()
    return sorted([m.id for m in models.data])


@app_router.get("/settings/models")
def fetch_app_models(
    api_base: str,
    api_key: str | None = None,
    current_user: User = Depends(require_super_admin),
):
    """Fetch available models from an OpenAI-compatible endpoint (app scope)."""
    try:
        return {"models": _fetch_models_from_endpoint(api_base, api_key)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch models: {exc}")


# ── Admin: org-level settings ─────────────────────────────────────────────

def _check_org_scope(db: Session, current_user: User, org_id: int) -> Organisation:
    """Verify the org exists and is within the admin's scope."""
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Organisation not found")
    admin_org_ids = get_admin_org_ids(db, current_user)
    if admin_org_ids is not None and org_id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="Organisation is outside your scope")
    return org


@org_router.get("/orgs/{org_id}/settings", response_model=SettingsListResponse)
def list_org_settings(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List all org-overridable settings with effective values and override flags."""
    _check_org_scope(db, current_user, org_id)
    items = get_all_org_settings_with_meta(db, org_id)
    return SettingsListResponse(settings=[SettingItem(**item) for item in items])


@org_router.get("/orgs/{org_id}/settings/schema", response_model=SettingsSchemaResponse)
def get_org_settings_schema(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Return registry metadata for org-overridable keys only."""
    _check_org_scope(db, current_user, org_id)
    items = []
    for d in REGISTRY:
        if d.scope != "org":
            continue
        items.append(SettingSchemaItem(
            key=d.key,
            value_type=d.value_type,
            category=d.category,
            label=d.label,
            scope=d.scope,
            reload=d.reload,
            requires_reindex=d.requires_reindex,
            description=d.description,
            min=d.min_value,
            max=d.max_value,
            choices=list(d.choices) if d.choices else None,
            secret=d.secret,
            model_picker=d.model_picker,
            api_base_ref=d.api_base_ref,
            api_key_ref=d.api_key_ref,
        ))
    return SettingsSchemaResponse(settings=items)


@org_router.put("/orgs/{org_id}/settings")
def update_org_settings(
    org_id: int,
    payload: SettingsBulkUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Bulk upsert org-level overrides. null value = clear override."""
    _check_org_scope(db, current_user, org_id)
    results = []
    for item in payload.settings:
        if not is_org_overridable(item.key):
            results.append({"key": item.key, "status": "error", "detail": "Cannot override this setting per organisation"})
            continue
        if item.value is None:
            # null = clear override
            try:
                reset_org_setting(db, org_id, item.key)
                results.append({"key": item.key, "status": "cleared"})
            except ValueError as e:
                results.append({"key": item.key, "status": "error", "detail": str(e)})
        else:
            try:
                upsert_org_setting(db, org_id, item.key, item.value, user_id=current_user.id)
                results.append({"key": item.key, "status": "ok"})
            except ValueError as e:
                results.append({"key": item.key, "status": "error", "detail": str(e)})
    return {"results": results}


@org_router.post("/orgs/{org_id}/settings/{key}")
def update_org_setting(
    org_id: int,
    key: str,
    payload: SettingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Upsert a single org-level override."""
    _check_org_scope(db, current_user, org_id)
    if payload.key != key:
        raise HTTPException(status_code=400, detail="Key in path must match key in body")
    if not is_org_overridable(key):
        raise HTTPException(status_code=403, detail="This setting cannot be overridden per organisation")
    try:
        upsert_org_setting(db, org_id, key, payload.value, user_id=current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"key": key, "status": "ok"}


@org_router.delete("/orgs/{org_id}/settings/{key}")
def delete_org_setting(
    org_id: int,
    key: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Delete an org-level override, reverting to app default."""
    _check_org_scope(db, current_user, org_id)
    try:
        reset_org_setting(db, org_id, key)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"key": key, "status": "reset"}


@org_router.delete("/orgs/{org_id}/settings")
def delete_all_org_settings(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Clear all org-level overrides for an org."""
    _check_org_scope(db, current_user, org_id)
    reset_all_org_settings(db, org_id)
    return {"status": "all_cleared", "org_id": org_id}


@org_router.get("/orgs/{org_id}/settings/models")
def fetch_org_models(
    org_id: int,
    api_base: str,
    api_key: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Fetch available models from an OpenAI-compatible endpoint (org scope)."""
    _check_org_scope(db, current_user, org_id)
    try:
        return {"models": _fetch_models_from_endpoint(api_base, api_key)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch models: {exc}")
