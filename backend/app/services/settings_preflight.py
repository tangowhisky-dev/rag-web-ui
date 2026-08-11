"""
settings_preflight.py — Post-login settings validation.

Checks all settings required for a user's role and org, following the
3-tier resolution chain (org override → app value → registry default).
Reports only settings that resolve to None — these break functionality.

Each issue names who can fix it:
  - "org_admin"   → org-scoped setting, fixable by an admin of the user's org
  - "super_admin" → app-scoped setting, fixable only by a super admin
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from sqlalchemy.orm import Session

from app.core.settings_registry import get_def, SettingDef
from app.services.settings_service import get_setting, clear_cache


@dataclass
class PreflightIssue:
    key: str
    label: str
    category: str
    severity: str           # "error" (blocks functionality)
    message: str
    who_can_fix: str        # "org_admin" | "super_admin"
    scope: str              # "org" | "app"
    is_set: bool            # False when None/unset


@dataclass
class PreflightResult:
    role: str
    org_id: Optional[int]
    ok: bool
    issues: list[PreflightIssue]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "org_id": self.org_id,
            "ok": self.ok,
            "issues": [asdict(i) for i in self.issues],
        }


# ── Setting groups ──────────────────────────────────────────────────────────

# Settings required for chat (org-overridable, resolved per user's org).
_CHAT_SETTINGS = [
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_MODEL",
]

# Settings required for retrieval/embeddings (app-only, shared infrastructure).
# EMBEDDING_API_KEY falls back to OPENAI_API_KEY at the app level.
_EMBEDDING_SETTINGS = [
    "EMBEDDING_API_KEY",
    "EMBEDDING_API_BASE",
    "DENSE_EMBEDDINGS_MODEL",
]

# Optional ingestion settings (app-only). Not blockers for chat, but
# super_admin should be aware if they're unset.
_OPTIONAL_INGESTION_SETTINGS = [
    "VISION_MODEL",
    "GRAPHRAG_LLM",
]


def _resolve_with_fallback(db: Session, key: str, fallback_key: Optional[str],
                           org_id: Optional[int], scope: str) -> tuple[bool, str]:
    """Resolve a setting, checking a fallback key if the primary is None.

    Returns (is_set, source) where source is the key that provided the value.
    """
    resolve_org = org_id if scope == "org" else None
    val = get_setting(db, key, resolve_org)
    if val is not None:
        return True, key
    if fallback_key:
        fb_scope = get_def(fallback_key)
        fb_org = org_id if (fb_scope and fb_scope.scope == "org") else None
        fb_val = get_setting(db, fallback_key, fb_org)
        if fb_val is not None:
            return True, fallback_key
    return False, ""


def _check_setting(
    db: Session,
    defn: SettingDef,
    org_id: Optional[int],
    fallback_key: Optional[str] = None,
    message: str = "",
    optional: bool = False,
) -> Optional[PreflightIssue]:
    """Check a single setting. Returns an issue if it resolves to None."""
    is_set, source = _resolve_with_fallback(
        db, defn.key, fallback_key, org_id, defn.scope
    )
    if is_set:
        return None

    who_can_fix = "org_admin" if defn.scope == "org" else "super_admin"
    severity = "warning" if optional else "error"
    msg = message or f"{defn.label} is not set."
    if fallback_key:
        msg += f" (Falls back to {fallback_key}, which is also unset.)"

    return PreflightIssue(
        key=defn.key,
        label=defn.label,
        category=defn.category,
        severity=severity,
        message=msg,
        who_can_fix=who_can_fix,
        scope=defn.scope,
        is_set=False,
    )


def check_required_settings(
    db: Session,
    role: str,
    org_id: Optional[int],
) -> PreflightResult:
    """Check all settings required for the given role and org.

    Resolution follows the 3-tier chain: org override → app value → default.
    Only settings that resolve to None are reported as issues.
    """
    clear_cache()
    issues: list[PreflightIssue] = []

    # ── Chat settings (required for all roles) ────────────────────────────
    for key in _CHAT_SETTINGS:
        defn = get_def(key)
        if defn is None:
            continue
        fallback = "OPENAI_API_KEY" if key != "OPENAI_API_KEY" else None
        issue = _check_setting(
            db, defn, org_id,
            fallback_key=fallback,
            message=f"{defn.label} is not set. Chat will not work." if key == "OPENAI_API_KEY" else "",
        )
        if issue:
            issues.append(issue)

    # ── Embedding settings (required for retrieval, app-only) ─────────────
    for key in _EMBEDDING_SETTINGS:
        defn = get_def(key)
        if defn is None:
            continue
        fallback = "OPENAI_API_KEY" if key == "EMBEDDING_API_KEY" else (
            "OPENAI_API_BASE" if key == "EMBEDDING_API_BASE" else None
        )
        issue = _check_setting(
            db, defn, org_id,
            fallback_key=fallback,
            message=f"{defn.label} is not set. Retrieval will not work." if key == "EMBEDDING_API_KEY" else "",
        )
        if issue:
            issues.append(issue)

    # ── Optional ingestion settings (super_admin only, warnings) ──────────
    if role == "super_admin":
        for key in _OPTIONAL_INGESTION_SETTINGS:
            defn = get_def(key)
            if defn is None:
                continue
            fallback = "OPENAI_MODEL" if key in ("GRAPHRAG_LLM",) else None
            issue = _check_setting(
                db, defn, org_id,
                fallback_key=fallback,
                message=f"{defn.label} is not set. {'Graph extraction' if 'GRAPHRAG' in key else 'OCR'} will be skipped during ingestion.",
                optional=True,
            )
            if issue:
                issues.append(issue)

    return PreflightResult(
        role=role,
        org_id=org_id,
        ok=len([i for i in issues if i.severity == "error"]) == 0,
        issues=issues,
    )
