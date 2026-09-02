"""Redis-backed login rate limiting with exponential backoff.

Shared across all workers via Redis.  When Redis is unavailable (e.g. in
tests without a Redis container), falls back to in-memory tracking so the
app still functions.

Rate-limit logic:
  1. Each failed attempt INCRs a counter for the IP.
  2. When the counter reaches MAX_LOGIN_ATTEMPTS, a backoff window is set.
  3. During backoff, login attempts are rejected with 429.
  4. When backoff expires, the counter resets but the backoff LEVEL
     increments — so the next lockout is longer.  Escalation compounds:
     15s → 30s → 60s → 120s → 240s → 480s → 900s (capped).
  5. Successful login clears all keys for that IP.
  6. Keys have TTLs so stale entries expire automatically.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import redis

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_LOGIN_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 15
MAX_BACKOFF_SECONDS = 900
# Total window after which all state for an IP expires completely.
KEY_TTL = MAX_BACKOFF_SECONDS * 2  # 1800s = 30 min

_REDIS: Optional[redis.Redis] = None

# In-memory fallback (used when Redis is not reachable).
_fallback: dict[str, dict[str, Any]] = {}


def _get_redis() -> Optional[redis.Redis]:
    """Return a shared synchronous Redis client, or None if unavailable."""
    global _REDIS
    if _REDIS is not None:
        return _REDIS
    try:
        client = redis.from_url(settings.REDIS_URL, socket_timeout=2, socket_connect_timeout=2)
        client.ping()
        _REDIS = client
        return _REDIS
    except Exception as exc:
        logger.warning("[RATE_LIMIT] Redis unavailable, using in-memory fallback: %s", exc)
        return None


def _backoff_duration(level: int) -> int:
    """Compute backoff seconds for a given level (0-indexed)."""
    return min(BASE_BACKOFF_SECONDS * (2 ** level), MAX_BACKOFF_SECONDS)


def check_rate_limit(ip: str) -> tuple[bool, int]:
    """Check if IP is rate limited.

    Returns (is_limited, retry_after_seconds).
    """
    r = _get_redis()
    if r is not None:
        return _check_redis(r, ip)
    return _check_fallback(ip)


def _check_redis(r: redis.Redis, ip: str) -> tuple[bool, int]:
    backoff_key = f"login_backoff:{ip}"
    backoff_until_raw = r.get(backoff_key)
    if backoff_until_raw is not None:
        backoff_until = float(backoff_until_raw)
        now = time.time()
        if now < backoff_until:
            return True, max(0, int(backoff_until - now))
        # Backoff timestamp is in the past but key still exists — escalate.
        r.incr(f"login_backoff_level:{ip}")
        r.delete(backoff_key)
        return False, 0

    # Key is gone — could mean never set, or expired via TTL.
    # If attempts counter shows >= MAX_LOGIN_ATTEMPTS, backoff expired via TTL.
    attempts_raw = r.get(f"login_attempts:{ip}")
    if attempts_raw is not None and int(attempts_raw) >= MAX_LOGIN_ATTEMPTS:
        r.incr(f"login_backoff_level:{ip}")
        r.delete(f"login_attempts:{ip}")

    return False, 0


def _check_fallback(ip: str) -> tuple[bool, int]:
    now = time.time()
    # Clean up entries older than KEY_TTL.
    expired = [k for k, v in _fallback.items() if now - v["first_attempt_time"] > KEY_TTL]
    for k in expired:
        del _fallback[k]

    data = _fallback.get(ip)
    if not data:
        return False, 0

    backoff_until = data.get("backoff_until")
    if backoff_until and now < backoff_until:
        return True, max(0, int(backoff_until - now))

    # Backoff expired — escalate level, allow attempt.
    if backoff_until:
        data["backoff_level"] = data.get("backoff_level", 0) + 1
        data["attempts"] = 0
        data["backoff_until"] = None
        data["first_attempt_time"] = now

    return False, 0


def record_failed_attempt(ip: str) -> int:
    """Record a failed login attempt. Returns current attempt count."""
    r = _get_redis()
    if r is not None:
        return _record_redis(r, ip)
    return _record_fallback(ip)


def _record_redis(r: redis.Redis, ip: str) -> int:
    attempts_key = f"login_attempts:{ip}"
    level_key = f"login_backoff_level:{ip}"
    backoff_key = f"login_backoff:{ip}"

    pipe = r.pipeline()
    pipe.incr(attempts_key)
    pipe.expire(attempts_key, KEY_TTL)
    pipe.expire(level_key, KEY_TTL)
    attempts, _, _ = pipe.execute()
    attempts = int(attempts)

    if attempts >= MAX_LOGIN_ATTEMPTS:
        level_raw = r.get(level_key)
        level = int(level_raw) if level_raw is not None else 0
        backoff = _backoff_duration(level)
        now = time.time()
        r.setex(backoff_key, backoff, str(now + backoff))
        logger.debug(
            "[RATE_LIMIT] backoff_set ip=%s attempts=%d level=%d backoff=%ds",
            ip, attempts, level, backoff,
        )

    return attempts


def _record_fallback(ip: str) -> int:
    now = time.time()
    data = _fallback.get(ip)
    if data is None:
        data = {
            "attempts": 0,
            "first_attempt_time": now,
            "backoff_until": None,
            "backoff_level": 0,
        }
        _fallback[ip] = data

    data["attempts"] += 1

    if data["attempts"] >= MAX_LOGIN_ATTEMPTS and not data.get("backoff_until"):
        backoff = _backoff_duration(data["backoff_level"])
        data["backoff_until"] = now + backoff
        logger.debug(
            "[RATE_LIMIT] backoff_set ip=%s attempts=%d level=%d backoff=%ds",
            ip, data["attempts"], data["backoff_level"], backoff,
        )

    return data["attempts"]


def reset_failed_attempts(ip: str) -> None:
    """Clear all rate-limit state for an IP after successful login."""
    r = _get_redis()
    if r is not None:
        r.delete(f"login_attempts:{ip}", f"login_backoff:{ip}", f"login_backoff_level:{ip}")
        return
    _fallback.pop(ip, None)
