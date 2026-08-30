"""
test_settings_phase4.py — Phase 4: retrieval + graph query + ingestion settings.

Tests:
  1. expand_docs_via_graph resolves org-overridable hops/limit/fanout.
  2. Ingestion settings (CHUNK_SIZE, OVERLAP_PERCENTAGE) are app-only.
  3. Graph ingestion settings (GRAPHRAG_ENABLED, MAX_CHUNKS, NEO4J_LLM_CONTEXT) are app-only.
"""
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.organisation import Organisation
import app.models.user  # noqa
import app.models.knowledge  # noqa
import app.models.chat  # noqa
import app.models.datastore  # noqa
import app.models.setting  # noqa

from app.services.settings_service import (
    upsert_app_setting, upsert_org_setting, clear_cache, get_setting,
)


@pytest.fixture()
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


def _create_org(db, name="TestOrg"):
    org = Organisation(name=name, parent_id=None, path="/1")
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


# ---------------------------------------------------------------------------
# 1. expand_docs_via_graph resolves org-overridable settings
# ---------------------------------------------------------------------------

def test_expand_docs_via_graph_accepts_db_and_org_id(db_session):
    """expand_docs_via_graph accepts db and org_id parameters without error."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "GRAPHRAG_RETRIEVAL_HOPS", 2)
    upsert_org_setting(db_session, org.id, "GRAPHRAG_ENTITY_FANOUT_CAP", 30)
    clear_cache()

    # With no docs, it returns [] immediately
    from app.services.graph.expand import expand_docs_via_graph
    result = expand_docs_via_graph([], [], db_session, org.id)
    assert result == []


# ---------------------------------------------------------------------------
# 2. Ingestion settings are app-only (cannot be org-overridden)
# ---------------------------------------------------------------------------

def test_chunk_size_is_app_only(db_session):
    """CHUNK_SIZE cannot be overridden at the org level."""
    org = _create_org(db_session)
    from app.services.settings_service import upsert_org_setting
    from app.core.settings_registry import is_org_overridable

    assert not is_org_overridable("CHUNK_SIZE")
    with pytest.raises(ValueError, match="cannot be overridden"):
        upsert_org_setting(db_session, org.id, "CHUNK_SIZE", 2000)


def test_overlap_percentage_is_app_only():
    """OVERLAP_PERCENTAGE cannot be overridden at the org level."""
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("OVERLAP_PERCENTAGE")


def test_graphrag_max_chunks_is_app_only():
    """GRAPHRAG_MAX_CHUNKS cannot be overridden at the org level."""
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("GRAPHRAG_MAX_CHUNKS")


def test_neo4j_llm_context_is_app_only():
    """NEO4J_LLM_CONTEXT cannot be overridden at the org level."""
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("NEO4J_LLM_CONTEXT")


def test_graphrag_enabled_is_app_only():
    """GRAPHRAG_ENABLED cannot be overridden at the org level."""
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("GRAPHRAG_ENABLED")


# ---------------------------------------------------------------------------
# 3. App-level ingestion settings are readable via settings service
# ---------------------------------------------------------------------------

def test_app_level_chunk_size_readable(db_session):
    """App-level CHUNK_SIZE is readable via the settings service."""
    upsert_app_setting(db_session, "CHUNK_SIZE", 2000)
    clear_cache()
    val = get_setting(db_session, "CHUNK_SIZE", None)
    assert val == 2000


def test_app_level_graphrag_max_chunks_readable(db_session):
    """App-level GRAPHRAG_MAX_CHUNKS is readable via the settings service."""
    upsert_app_setting(db_session, "GRAPHRAG_MAX_CHUNKS", 300)
    clear_cache()
    val = get_setting(db_session, "GRAPHRAG_MAX_CHUNKS", None)
    assert val == 300


# ---------------------------------------------------------------------------
# 4. Org-overridable retrieval settings are correctly classified
# ---------------------------------------------------------------------------

def test_retrieval_top_k_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("RETRIEVAL_TOP_K")


def test_graphrag_retrieval_hops_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("GRAPHRAG_RETRIEVAL_HOPS")


def test_reranker_enabled_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("RERANKER_ENABLED")


def test_dense_embeddings_model_is_app_only():
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("DENSE_EMBEDDINGS_MODEL")


def test_splade_model_is_app_only():
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("SPLADE_MODEL")


def test_reranker_model_is_app_only():
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("RERANKER_MODEL")
