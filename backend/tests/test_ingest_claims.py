"""Tests for the Redis-backed ingestion claim registry and the
chunk↔point parity helper used by scan-time healing."""
from unittest.mock import MagicMock, patch

import pytest

from app.services.infrastructure import ingest_claims


class _FakePipeline:
    def __init__(self, store):
        self._store = store
        self._keys = []

    def exists(self, key):
        self._keys.append(key)
        return self

    def execute(self):
        out = [1 if k in self._store else 0 for k in self._keys]
        self._keys.clear()
        return out


class _FakeRedis:
    """Minimal in-memory stand-in for the Redis ops ingest_claims uses."""

    def __init__(self):
        self.store = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def delete(self, *keys):
        n = 0
        for k in keys:
            if self.store.pop(k, None) is not None:
                n += 1
        return n

    def exists(self, key):
        return 1 if key in self.store else 0

    def expire(self, key, ttl):
        return key in self.store

    def scan_iter(self, pattern, count=500):
        prefix = pattern.rstrip("*")
        return [k for k in list(self.store) if k.startswith(prefix)]

    def pipeline(self):
        return _FakePipeline(self.store)


@pytest.fixture()
def fake_redis():
    r = _FakeRedis()
    with patch.object(ingest_claims, "_get_redis", return_value=r):
        yield r


class TestIngestClaims:
    def test_claim_is_exclusive(self, fake_redis):
        assert ingest_claims.claim_ingestion(1) is True
        assert ingest_claims.claim_ingestion(1) is False

    def test_release_allows_reclaim(self, fake_redis):
        ingest_claims.claim_ingestion(1)
        ingest_claims.release_ingestion_claim(1)
        assert ingest_claims.claim_ingestion(1) is True

    def test_batch_check(self, fake_redis):
        ingest_claims.claim_ingestion(1)
        ingest_claims.claim_ingestion(3)
        assert ingest_claims.get_claimed_task_ids([1, 2, 3]) == {1, 3}
        assert ingest_claims.get_claimed_task_ids([]) == set()

    def test_clear_all(self, fake_redis):
        ingest_claims.claim_ingestion(1)
        ingest_claims.claim_ingestion(2)
        assert ingest_claims.clear_ingestion_claims() == 2
        assert ingest_claims.get_claimed_task_ids([1, 2]) == set()

    def test_redis_down_fails_open(self):
        """Without Redis, claims behave as 'not claimed' — submission is
        never blocked and requeue paths stay functional."""
        with patch.object(ingest_claims, "_get_redis", return_value=None):
            assert ingest_claims.claim_ingestion(1) is True
            assert ingest_claims.get_claimed_task_ids([1]) == set()
            ingest_claims.release_ingestion_claim(1)
            ingest_claims.touch_ingestion_claim(1)
            assert ingest_claims.clear_ingestion_claims() == 0


class TestDocChunksHaveVectors:
    def test_all_points_present(self):
        from app.services.ingestion import document_qdrant
        mock_qdrant = MagicMock()
        mock_qdrant.count.return_value.count = 2
        with patch.object(document_qdrant, "get_qdrant_client", return_value=mock_qdrant):
            assert document_qdrant.doc_chunks_have_vectors("ds_1", ["c1", "c2"]) is True

    def test_missing_point_returns_false(self):
        from app.services.ingestion import document_qdrant
        mock_qdrant = MagicMock()
        mock_qdrant.count.return_value.count = 1
        with patch.object(document_qdrant, "get_qdrant_client", return_value=mock_qdrant):
            assert document_qdrant.doc_chunks_have_vectors("ds_1", ["c1", "c2"]) is False

    def test_qdrant_error_fail_open(self):
        """A Qdrant outage must not trigger mass re-ingestion."""
        from app.services.ingestion import document_qdrant
        mock_qdrant = MagicMock()
        mock_qdrant.count.side_effect = Exception("connection refused")
        with patch.object(document_qdrant, "get_qdrant_client", return_value=mock_qdrant):
            assert document_qdrant.doc_chunks_have_vectors("ds_1", ["c1"]) is True

    def test_empty_chunk_list(self):
        from app.services.ingestion import document_qdrant
        assert document_qdrant.doc_chunks_have_vectors("ds_1", []) is False
