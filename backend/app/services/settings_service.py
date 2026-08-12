"""
settings_service.py — 2-tier settings resolution + CRUD + cache + secret encryption.

Resolution precedence (per org, per key):
  1. Org override (scope='org', org_id=<id>) — if key is org-overridable
  2. App value (scope='app', org_id=NULL)
  3. Registry default

The settings table is the single source of truth for runtime-configurable values.
config.py / .env only controls deployment infrastructure (DB, Redis, Qdrant, etc.).

Secret values (defn.secret=True) are encrypted at rest using Fernet symmetric
encryption, keyed by PBKDF2(SECRET_KEY). Encrypted values are prefixed with
'enc:' in the DB column to distinguish from plaintext.
"""
import json
import logging
import time
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.config import settings as env_settings
from app.core.settings_registry import (
    REGISTRY, REGISTRY_BY_KEY, SettingDef,
    ORG_OVERRIDABLE_KEYS, get_def, is_org_overridable,
)
from app.models.setting import Setting

logger = logging.getLogger(__name__)

_CACHE_TTL = 30  # seconds
_cache: dict[tuple[Optional[int], str], tuple[Any, float]] = {}


# ── Secret encryption ─────────────────────────────────────────────────────

_fernet = None

def _get_fernet():
    """Lazily initialise a Fernet instance derived from SECRET_KEY."""
    global _fernet
    if _fernet is None:
        from cryptography.fernet import Fernet
        import base64
        import hashlib
        key = hashlib.pbkdf2_hmac(
            "sha256",
            env_settings.SECRET_KEY.encode(),
            b"rag-webui-settings-v1",
            480000,
        )
        _fernet = Fernet(base64.urlsafe_b64encode(key))
    return _fernet


def _encrypt(value: str) -> str:
    """Encrypt a plaintext string, returning 'enc:<ciphertext>'."""
    return "enc:" + _get_fernet().encrypt(value.encode()).decode()


def _decrypt(raw: str) -> str:
    """Decrypt an 'enc:<ciphertext>' string back to plaintext."""
    if raw.startswith("enc:"):
        return _get_fernet().decrypt(raw[4:].encode()).decode()
    return raw  # plaintext fallback (for pre-encryption rows)


def _mask_secret(value: Any) -> str:
    """Mask a secret value for API display: show last 4 chars only."""
    s = str(value) if value else ""
    if len(s) <= 4:
        return "••••"
    return "••••" + s[-4:]


# ── Encoding / decoding ───────────────────────────────────────────────────

def _encode(value: Any, defn: SettingDef) -> str:
    """Encode a Python value as JSON for DB storage. Encrypts if secret."""
    encoded = json.dumps(value)
    if defn.secret and value is not None:
        encoded = _encrypt(encoded)
    return encoded


def _decode(raw: str, defn: SettingDef) -> Any:
    """Decode a stored string back to the registry type. Decrypts if secret."""
    if defn.secret:
        raw = _decrypt(raw)
    parsed = json.loads(raw)
    if defn.value_type == "int":
        return int(parsed)
    if defn.value_type == "float":
        return float(parsed)
    if defn.value_type == "bool":
        if isinstance(parsed, str):
            return parsed.lower() == "true"
        return bool(parsed)
    return parsed  # str, json, text


# ── Validation ────────────────────────────────────────────────────────────

def validate_value(key: str, value: Any) -> Any:
    """Validate a value against the registry. Returns the coerced value or raises ValueError."""
    defn = get_def(key)
    if defn is None:
        raise ValueError(f"Unknown setting key: {key}")

    # null is allowed for optional str fields (means "inherit/fallback")
    if value is None and defn.value_type in ("str", "text"):
        return None

    if defn.value_type == "int":
        try:
            v = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer, got {value!r}")
        if defn.min_value is not None and v < defn.min_value:
            raise ValueError(f"{key} must be >= {defn.min_value}, got {v}")
        if defn.max_value is not None and v > defn.max_value:
            raise ValueError(f"{key} must be <= {defn.max_value}, got {v}")
        return v

    if defn.value_type == "float":
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a float, got {value!r}")
        if defn.min_value is not None and v < defn.min_value:
            raise ValueError(f"{key} must be >= {defn.min_value}, got {v}")
        if defn.max_value is not None and v > defn.max_value:
            raise ValueError(f"{key} must be <= {defn.max_value}, got {v}")
        return v

    if defn.value_type == "bool":
        if isinstance(value, str):
            return value.lower() == "true"
        if isinstance(value, bool):
            return value
        raise ValueError(f"{key} must be a boolean, got {value!r}")

    if defn.value_type == "json":
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"{key} must be valid JSON, got {value!r}")
        return value  # already a dict/list

    if defn.choices is not None and value not in defn.choices:
        raise ValueError(f"{key} must be one of {defn.choices}, got {value!r}")

    return value  # str, text


# ── Resolution ────────────────────────────────────────────────────────────

def _registry_default(key: str) -> Any:
    """Return the registry default for a key, or None if not in the registry."""
    defn = get_def(key)
    if defn is not None:
        return defn.default
    # For non-registry keys, fall back to config.py (infrastructure settings)
    return getattr(env_settings, key, None)


def get_setting(db: Session, key: str, org_id: Optional[int] = None) -> Any:
    """Resolve a single setting with 2-tier precedence.

    1. Org override (if scope='org' and org_id is set)
    2. App value (scope='app', org_id=NULL)
    3. Registry default

    Falls back to registry default on any DB error (e.g. mock sessions in tests).
    """
    defn = get_def(key)
    if defn is None:
        # Not a registry key — read from config.py (infrastructure setting)
        return getattr(env_settings, key, None)

    # Check cache
    cache_key = (org_id if defn.scope == "org" else None, key)
    cached = _cache.get(cache_key)
    if cached is not None:
        val, ts = cached
        if time.time() - ts < _CACHE_TTL:
            return val

    try:
        # Tier 1: org override (only if scope allows and org_id is set)
        if defn.scope == "org" and org_id is not None:
            row = db.query(Setting).filter(
                Setting.scope == "org", Setting.org_id == org_id, Setting.key == key
            ).first()
            if row is not None and row.value is not None:
                val = _decode(row.value, defn)
                _cache[cache_key] = (val, time.time())
                return val

        # Tier 2: app value
        row = db.query(Setting).filter(
            Setting.scope == "app", Setting.org_id.is_(None), Setting.key == key
        ).first()
        if row is not None and row.value is not None:
            val = _decode(row.value, defn)
            _cache[cache_key] = (val, time.time())
            return val
    except Exception:
        # DB error (mock session, connection issue, etc.) — fall back to registry default
        pass

    # Tier 3: registry default
    val = defn.default
    _cache[cache_key] = (val, time.time())
    return val


def get_org_settings(db: Session, org_id: Optional[int]) -> dict[str, Any]:
    """Resolve ALL registry keys for an org into a typed dict.

    For app-only keys, returns the app value regardless of org_id.
    For org keys, applies 3-tier precedence.
    """
    out = {}
    for defn in REGISTRY:
        out[defn.key] = get_setting(db, defn.key, org_id if defn.scope == "org" else None)
    return out


# ── OrgSettings accessor ──────────────────────────────────────────────────

class OrgSettings:
    """Attribute-access wrapper over get_org_settings.

    Services receive an OrgSettings instance instead of reading the settings singleton.
    Construct once per request: OrgSettings(db, current_user.org_id)
    When org_id is None, all keys resolve to app-level values.
    """
    def __init__(self, db: Session, org_id: Optional[int] = None):
        self._db = db
        self._org_id = org_id
        self._resolved = get_org_settings(db, org_id)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        resolved = self.__dict__.get("_resolved", {})
        if key in resolved:
            return resolved[key]
        raise AttributeError(f"Unknown setting: {key}")

    # Computed properties
    @property
    def chunk_overlap(self) -> int:
        return int(self.CHUNK_SIZE * self.OVERLAP_PERCENTAGE)

    @property
    def retrieval_config_presets(self) -> dict:
        raw = self.RETRIEVAL_CONFIG_PRESETS
        if isinstance(raw, str):
            return json.loads(raw)
        return raw or {}


# ── CRUD ──────────────────────────────────────────────────────────────────

def upsert_app_setting(db: Session, key: str, value: Any, user_id: Optional[int] = None) -> None:
    """Upsert an app-level setting (scope='app'). Validates against registry.

    For secret settings, if the value looks masked (starts with '••••'), it is
    treated as a no-op — the existing encrypted value is preserved.
    """
    defn = get_def(key)
    if defn is None:
        raise ValueError(f"Unknown setting key: {key}")
    if defn.secret and isinstance(value, str) and value.startswith("••••"):
        return  # masked value = no change
    validated = validate_value(key, value)
    encoded = _encode(validated, defn)

    row = db.query(Setting).filter(
        Setting.scope == "app", Setting.org_id.is_(None), Setting.key == key
    ).first()
    if row:
        row.value = encoded
        row.updated_by = user_id
    else:
        row = Setting(scope="app", org_id=None, key=key, value=encoded, updated_by=user_id)
        db.add(row)
    db.commit()
    _invalidate_cache(key, None)


def upsert_org_setting(db: Session, org_id: int, key: str, value: Any, user_id: Optional[int] = None) -> None:
    """Upsert an org-level override. Validates against registry + scope.

    For secret settings, if the value looks masked (starts with '••••'), it is
    treated as a no-op — the existing encrypted value is preserved.
    """
    defn = get_def(key)
    if defn is None:
        raise ValueError(f"Unknown setting key: {key}")
    if not is_org_overridable(key):
        raise ValueError(f"Setting {key} cannot be overridden per organisation")
    if defn.secret and isinstance(value, str) and value.startswith("••••"):
        return  # masked value = no change
    validated = validate_value(key, value)
    encoded = _encode(validated, defn)

    row = db.query(Setting).filter(
        Setting.scope == "org", Setting.org_id == org_id, Setting.key == key
    ).first()
    if row:
        row.value = encoded
        row.updated_by = user_id
    else:
        row = Setting(scope="org", org_id=org_id, key=key, value=encoded, updated_by=user_id)
        db.add(row)
    db.commit()
    _invalidate_cache(key, org_id)


def reset_app_setting(db: Session, key: str) -> None:
    """Delete the app-level row for a key, reverting to .env/config.py default."""
    defn = get_def(key)
    if defn is None:
        raise ValueError(f"Unknown setting key: {key}")
    row = db.query(Setting).filter(
        Setting.scope == "app", Setting.org_id.is_(None), Setting.key == key
    ).first()
    if row:
        db.delete(row)
        db.commit()
    _invalidate_cache(key, None)


def reset_org_setting(db: Session, org_id: int, key: str) -> None:
    """Delete the org-level override, reverting to app-level default."""
    defn = get_def(key)
    if defn is None:
        raise ValueError(f"Unknown setting key: {key}")
    row = db.query(Setting).filter(
        Setting.scope == "org", Setting.org_id == org_id, Setting.key == key
    ).first()
    if row:
        db.delete(row)
        db.commit()
    _invalidate_cache(key, org_id)


def reset_all_org_settings(db: Session, org_id: int) -> None:
    """Delete all org-level overrides for an org."""
    db.query(Setting).filter(
        Setting.scope == "org", Setting.org_id == org_id
    ).delete()
    db.commit()
    _invalidate_all_org(org_id)


# ── Introspection (for API responses) ─────────────────────────────────────

def get_app_setting_with_meta(db: Session, key: str) -> dict:
    """Return a setting's effective value + metadata for the API."""
    defn = get_def(key)
    if defn is None:
        raise ValueError(f"Unknown setting key: {key}")

    row = db.query(Setting).filter(
        Setting.scope == "app", Setting.org_id.is_(None), Setting.key == key
    ).first()
    source = "database" if (row and row.value is not None) else "install_default"
    effective = get_setting(db, key, None)

    # Mask secrets in API responses
    display_value = _mask_secret(effective) if (defn.secret and effective) else effective

    return {
        "key": key,
        "value": display_value,
        "value_type": defn.value_type,
        "category": defn.category,
        "label": defn.label,
        "scope": defn.scope,
        "source": source,
        "reload": defn.reload,
        "requires_reindex": defn.requires_reindex,
        "description": defn.description,
        "min": defn.min_value,
        "max": defn.max_value,
        "choices": list(defn.choices) if defn.choices else None,
        "secret": defn.secret,
        "is_set": effective is not None,
        "model_picker": defn.model_picker,
        "api_base_ref": defn.api_base_ref,
        "api_key_ref": defn.api_key_ref,
    }


def get_all_app_settings_with_meta(db: Session) -> list[dict]:
    """Return all app settings with metadata."""
    return [get_app_setting_with_meta(db, d.key) for d in REGISTRY]


def get_org_setting_with_meta(db: Session, org_id: int, key: str) -> dict:
    """Return a setting's effective value + override status for an org."""
    defn = get_def(key)
    if defn is None:
        raise ValueError(f"Unknown setting key: {key}")

    org_row = db.query(Setting).filter(
        Setting.scope == "org", Setting.org_id == org_id, Setting.key == key
    ).first()
    app_row = db.query(Setting).filter(
        Setting.scope == "app", Setting.org_id.is_(None), Setting.key == key
    ).first()

    overridden = org_row is not None and org_row.value is not None
    effective = get_setting(db, key, org_id if defn.scope == "org" else None)
    app_default = get_setting(db, key, None)

    # Mask secrets in API responses
    display_effective = _mask_secret(effective) if (defn.secret and effective) else effective
    display_app_default = _mask_secret(app_default) if (defn.secret and app_default) else app_default

    return {
        "key": key,
        "value": display_effective,
        "value_type": defn.value_type,
        "category": defn.category,
        "label": defn.label,
        "scope": defn.scope,
        "overridden": overridden,
        "app_default": display_app_default,
        "effective": display_effective,
        "reload": defn.reload,
        "requires_reindex": defn.requires_reindex,
        "description": defn.description,
        "min": defn.min_value,
        "max": defn.max_value,
        "choices": list(defn.choices) if defn.choices else None,
        "secret": defn.secret,
        "is_set": effective is not None,
        "model_picker": defn.model_picker,
        "api_base_ref": defn.api_base_ref,
        "api_key_ref": defn.api_key_ref,
    }


def get_all_org_settings_with_meta(db: Session, org_id: int) -> list[dict]:
    """Return all org settings with metadata. Only org-overridable keys are included."""
    return [
        get_org_setting_with_meta(db, org_id, d.key)
        for d in REGISTRY
        if d.scope == "org"
    ]


# ── Cache management ──────────────────────────────────────────────────────

def _invalidate_cache(key: str, org_id: Optional[int]) -> None:
    """Invalidate cache entries for a key."""
    defn = get_def(key)
    if defn is None:
        return
    # Invalidate the org-specific entry
    if defn.scope == "org":
        _cache.pop((org_id, key), None)
    # Invalidate the app-level entry (affects all orgs that don't override)
    _cache.pop((None, key), None)


def _invalidate_all_org(org_id: int) -> None:
    """Invalidate all cache entries for an org."""
    keys_to_remove = [k for k in _cache if k[0] == org_id]
    for k in keys_to_remove:
        _cache.pop(k, None)


def clear_cache() -> None:
    """Clear the entire cache. Useful for tests."""
    _cache.clear()
