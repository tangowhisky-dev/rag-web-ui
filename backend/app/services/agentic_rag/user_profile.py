"""User profile store — cross-turn understanding for the autonomous agent.

Stores and retrieves per-user preferences, communication style, and recurring
patterns so the supervisor can tailor its decisions across turns.

Designed as a lightweight in-memory + DB hybrid:
- In-memory cache: per-ChatSession dict for the current session.
- DB persistence: UserProfile model for long-term cross-session memory.
- The supervisor receives a compact "profile summary" injected into its
  system prompt so it can adapt without bloating context.

Security: profiles are scoped by org_id + user_id. A user can only see
their own profile — never another user's.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class UserPreference:
    """A single user preference captured from interaction."""
    category: str          # e.g. "chart_preference", "detail_level", "tone"
    key: str               # e.g. "always_show_charts"
    value: Any             # The preference value
    source: str            # "user_explicit" | "auto_inferred"
    confidence: float      # 0.0–1.0 how confident we are
    updated_at: float = field(default_factory=time.monotonic)


@dataclass
class UserProfile:
    """Aggregated user profile for a single user."""
    user_id: int
    org_id: Optional[int]
    preferences: Dict[str, Dict[str, UserPreference]] = field(default_factory=dict)
    query_patterns: List[dict] = field(default_factory=list)
    communication_style: Optional[str] = None  # "concise" | "detailed" | "mixed"
    domain_focus: List[str] = field(default_factory=list)
    built_at: float = field(default_factory=time.monotonic)

    def get_preference(
        self, category: str, key: str, default: Any = None
    ) -> Any:
        """Get a preference value, falling back to default."""
        cat = self.preferences.get(category, {})
        pref = cat.get(key)
        if pref and pref.confidence >= 0.5:
            return pref.value
        return default

    def to_summary(self, max_chars: int = 500) -> str:
        """
        Produce a compact text summary for injection into the supervisor prompt.
        Only includes high-confidence preferences and notable patterns.
        """
        parts = []

        # Communication style
        if self.communication_style:
            parts.append(f"Communication: {self.communication_style}")

        # Domain focus
        if self.domain_focus:
            top_domains = self.domain_focus[:5]
            parts.append(f"Domain focus: {', '.join(top_domains)}")

        # High-confidence preferences by category
        for cat, prefs in sorted(self.preferences.items()):
            high_conf = {
                k: v for k, v in prefs.items()
                if v.confidence >= 0.7
            }
            if high_conf:
                items = []
                for k, v in sorted(high_conf.items(), key=lambda x: -x[1].confidence)[:3]:
                    items.append(f"{k}={v.value}")
                parts.append(f"{cat}: {', '.join(items)}")

        # Query patterns (top 3 most recent)
        recent_patterns = self.query_patterns[-3:]
        if recent_patterns:
            pattern_strs = [p.get("description", "") for p in recent_patterns if p.get("description")]
            parts.append(f"Recent patterns: {'; '.join(pattern_strs[:3])}")

        summary = " | ".join(parts) if parts else ""
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "…"
        return summary


class UserProfileStore:
    """
    In-memory + DB-backed user profile store.

    Usage:
        store = UserProfileStore(user_id=1, org_id=5, db=session)
        profile = store.load()
        store.set_preference("chart_preference", "always_show_charts", True,
                             source="user_explicit")
        store.save()
        summary = profile.to_summary()  # for supervisor prompt injection
    """

    def __init__(self, user_id: int, org_id: Optional[int], db: Any):
        self.user_id = user_id
        self.org_id = org_id
        self.db = db
        self._profile: Optional[UserProfile] = None
        self._dirty = False

    @property
    def profile(self) -> UserProfile:
        """Lazy-load the profile."""
        if self._profile is None:
            self._profile = self._load_from_db()
        return self._profile

    def set_preference(
        self,
        category: str,
        key: str,
        value: Any,
        source: str = "user_explicit",
        confidence: float = 0.9,
    ) -> None:
        """Record a user preference."""
        profile = self.profile
        if category not in profile.preferences:
            profile.preferences[category] = {}

        old = profile.preferences[category].get(key)
        if old:
            # Update existing — keep higher confidence
            if confidence > old.confidence:
                profile.preferences[category][key] = UserPreference(
                    category=category, key=key, value=value,
                    source=source, confidence=confidence,
                )
        else:
            profile.preferences[category][key] = UserPreference(
                category=category, key=key, value=value,
                source=source, confidence=confidence,
            )
        self._dirty = True

    def get_preference(
        self, category: str, key: str, default: Any = None
    ) -> Any:
        """Get a preference value."""
        return self.profile.get_preference(category, key, default)

    def add_query_pattern(
        self, description: str, query_type: str,
        preferred_response_format: Optional[str] = None,
    ) -> None:
        """Record a recurring query pattern."""
        self.profile.query_patterns.append({
            "description": description,
            "query_type": query_type,
            "preferred_response_format": preferred_response_format,
            "recorded_at": time.monotonic(),
        })
        self._dirty = True

    def save(self) -> None:
        """Persist dirty preferences to DB."""
        if not self._dirty:
            return
        self._save_to_db()
        self._dirty = False

    def _load_from_db(self) -> UserProfile:
        """Load profile from DB (or create fresh if none exists)."""
        try:
            from app.models.user_profile import UserProfile as DBUserProfile
            record = (
                self.db.query(DBUserProfile)
                .filter(
                    DBUserProfile.user_id == self.user_id,
                    DBUserProfile.org_id == self.org_id,
                )
                .first()
            )
            if record:
                profile = UserProfile(
                    user_id=self.user_id,
                    org_id=self.org_id,
                    preferences=json.loads(record.preferences_json or "{}"),
                    query_patterns=json.loads(record.query_patterns_json or "[]"),
                    communication_style=record.communication_style,
                    domain_focus=json.loads(record.domain_focus_json or "[]"),
                )
                logger.info(
                    "[USER_PROFILE] loaded user_id=%d org_id=%d prefs=%d patterns=%d",
                    self.user_id, self.org_id,
                    sum(len(v) for v in profile.preferences.values()),
                    len(profile.query_patterns),
                )
                return profile
        except Exception as exc:
            logger.warning("[USER_PROFILE] DB load failed, using fresh profile: %s", exc)

        # Fallback: fresh profile
        return UserProfile(
            user_id=self.user_id,
            org_id=self.org_id,
        )

    def _save_to_db(self) -> None:
        """Persist profile to DB."""
        try:
            from app.models.user_profile import UserProfile as DBUserProfile

            existing = (
                self.db.query(DBUserProfile)
                .filter(
                    DBUserProfile.user_id == self.user_id,
                    DBUserProfile.org_id == self.org_id,
                )
                .first()
            )

            profile = self.profile
            prefs_json = json.dumps(profile.preferences)
            patterns_json = json.dumps(profile.query_patterns)
            domain_json = json.dumps(profile.domain_focus)

            if existing:
                existing.preferences_json = prefs_json
                existing.query_patterns_json = patterns_json
                existing.domain_focus_json = domain_json
                existing.communication_style = profile.communication_style
            else:
                record = DBUserProfile(
                    user_id=self.user_id,
                    org_id=self.org_id,
                    preferences_json=prefs_json,
                    query_patterns_json=patterns_json,
                    domain_focus_json=domain_json,
                    communication_style=profile.communication_style,
                )
                self.db.add(record)

            self.db.commit()
            logger.info(
                "[USER_PROFILE] saved user_id=%d org_id=%d",
                self.user_id, self.org_id,
            )
        except Exception as exc:
            logger.error("[USER_PROFILE] DB save failed: %s", exc)
            self.db.rollback()
