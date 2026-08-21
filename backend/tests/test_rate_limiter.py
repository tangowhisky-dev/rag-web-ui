"""Tests for the Redis-backed login rate limiter."""
import time
from unittest.mock import patch, MagicMock

from app.core.rate_limiter import (
    check_rate_limit,
    record_failed_attempt,
    reset_failed_attempts,
    _backoff_duration,
    _fallback,
    _check_fallback,
    _record_fallback,
)


def test_backoff_duration_escalates():
    """Backoff doubles each level up to the cap."""
    assert _backoff_duration(0) == 15
    assert _backoff_duration(1) == 30
    assert _backoff_duration(2) == 60
    assert _backoff_duration(3) == 120
    assert _backoff_duration(4) == 240
    assert _backoff_duration(5) == 480
    assert _backoff_duration(6) == 900
    assert _backoff_duration(7) == 900  # capped


@patch("app.core.rate_limiter._get_redis", return_value=None)
def test_fallback_allows_first_attempts(_mock):
    """First few attempts should not be rate limited."""
    _fallback.clear()
    limited, retry = check_rate_limit("1.2.3.4")
    assert limited is False
    assert retry == 0


@patch("app.core.rate_limiter._get_redis", return_value=None)
def test_fallback_locks_after_max_attempts(_mock):
    """After MAX_LOGIN_ATTEMPTS failures, IP should be locked."""
    _fallback.clear()
    ip = "10.0.0.1"
    for _ in range(3):
        record_failed_attempt(ip)
    limited, retry = check_rate_limit(ip)
    assert limited is True
    assert retry > 0
    assert retry <= 15


@patch("app.core.rate_limiter._get_redis", return_value=None)
def test_fallback_escalates_after_backoff_expiry(_mock):
    """After backoff expires, level should increment, not reset to 0."""
    _fallback.clear()
    ip = "10.0.0.2"

    # First lockout: level 0 → 15s
    for _ in range(3):
        record_failed_attempt(ip)
    limited, _ = check_rate_limit(ip)
    assert limited is True

    # Simulate backoff expiry by backdating backoff_until
    _fallback[ip]["backoff_until"] = time.time() - 1

    # Next check should allow the attempt and escalate the level
    limited, retry = check_rate_limit(ip)
    assert limited is False
    assert _fallback[ip]["backoff_level"] == 1

    # Next lockout should use level 1 → 30s
    for _ in range(3):
        record_failed_attempt(ip)
    limited, retry = check_rate_limit(ip)
    assert limited is True
    assert retry <= 30
    assert retry > 15  # longer than the first lockout


@patch("app.core.rate_limiter._get_redis", return_value=None)
def test_fallback_reset_on_success(_mock):
    """Successful login clears all state for the IP."""
    _fallback.clear()
    ip = "10.0.0.3"
    record_failed_attempt(ip)
    record_failed_attempt(ip)
    assert ip in _fallback

    reset_failed_attempts(ip)
    assert ip not in _fallback

    # Should start fresh
    limited, _ = check_rate_limit(ip)
    assert limited is False


@patch("app.core.rate_limiter._get_redis", return_value=None)
def test_fallback_correct_login_does_not_lock(_mock):
    """Failing twice then succeeding should not lock."""
    _fallback.clear()
    ip = "10.0.0.4"
    record_failed_attempt(ip)
    record_failed_attempt(ip)
    reset_failed_attempts(ip)  # simulate successful login

    limited, _ = check_rate_limit(ip)
    assert limited is False


def test_redis_path_uses_pipeline():
    """Verify the Redis path uses atomic pipeline for increment+expire."""
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    mock_redis.pipeline.return_value = mock_pipe
    mock_pipe.execute.return_value = [1, True, True]

    with patch("app.core.rate_limiter._get_redis", return_value=mock_redis):
        attempts = record_failed_attempt("5.6.7.8")

    assert attempts == 1
    mock_pipe.incr.assert_called_once_with("login_attempts:5.6.7.8")
    # expire is called for attempts key and level key
    assert mock_pipe.expire.call_count == 2


def test_redis_path_sets_backoff_on_threshold():
    """When attempts reach MAX_LOGIN_ATTEMPTS, backoff key should be set."""
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    mock_redis.pipeline.return_value = mock_pipe
    mock_pipe.execute.return_value = [3, True, True]  # 3rd attempt
    mock_redis.get.return_value = None  # no prior backoff level

    with patch("app.core.rate_limiter._get_redis", return_value=mock_redis):
        attempts = record_failed_attempt("9.10.11.12")

    assert attempts == 3
    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args
    assert args.args[0] == "login_backoff:9.10.11.12"
    assert args.args[1] == 15  # level 0 → 15s


def test_redis_path_rejects_during_backoff():
    """When backoff key exists and hasn't expired, should reject."""
    mock_redis = MagicMock()
    future = str(time.time() + 30)
    mock_redis.get.return_value = future.encode()

    with patch("app.core.rate_limiter._get_redis", return_value=mock_redis):
        limited, retry = check_rate_limit("13.14.15.16")

    assert limited is True
    assert retry > 0
    assert retry <= 30


def test_redis_path_escalates_after_expiry():
    """When backoff key expired via TTL but attempts counter remains, escalate."""
    mock_redis = MagicMock()

    # backoff key gone (TTL expired), but attempts counter shows 3
    def mock_get(key):
        if key == "login_backoff:17.18.19.20":
            return None
        if key == "login_attempts:17.18.19.20":
            return b"3"
        return None

    mock_redis.get.side_effect = mock_get

    with patch("app.core.rate_limiter._get_redis", return_value=mock_redis):
        limited, retry = check_rate_limit("17.18.19.20")

    assert limited is False
    mock_redis.incr.assert_called_once_with("login_backoff_level:17.18.19.20")
    mock_redis.delete.assert_called_once_with("login_attempts:17.18.19.20")


def test_redis_path_reset_deletes_keys():
    """reset_failed_attempts should delete all keys for the IP."""
    mock_redis = MagicMock()
    with patch("app.core.rate_limiter._get_redis", return_value=mock_redis):
        reset_failed_attempts("21.22.23.24")

    mock_redis.delete.assert_called_once_with(
        "login_attempts:21.22.23.24",
        "login_backoff:21.22.23.24",
        "login_backoff_level:21.22.23.24",
    )
