# S03 Replan

**Milestone:** M002
**Slice:** S03
**Blocker Task:** T01
**Created:** 2026-05-21T14:40:52.931Z

## Blocker Description

pytest backend/ fails with exit code 2 during collection: Pydantic Settings rejects extra env vars (relik_url, timeout_seconds, tz) as extra_forbidden. This is a pre-existing issue — S03 did not change config.py or backend tests — but the verification gate requires pytest to pass. Fix requires adding model_config = ConfigDict(extra='ignore') to Settings class in backend/app/core/config.py.

## What Changed

Added T04 to fix the pre-existing Pydantic Settings extra-fields rejection that blocks pytest collection. All T01–T03 work is intact; T04 is a one-line fix to config.py.
