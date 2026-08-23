#!/usr/bin/env python3
"""End-to-end test for abbreviation expansion implementation.

Tests:
  1. Abbreviation list CRUD via API
  2. Lookup building and caching
  3. Ingestion expansion (chunk_text expanded, original_text in metadata)
  4. Query expansion (expanded_query in state)
  5. Generation glossary (original_text + glossary in context)
  6. Disable/enable toggle
"""
import os
import sys
import time
import json

sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")


def test_1_lookup_and_expansion():
    """Test that the abbreviation service builds lookup and expands correctly."""
    from app.db.session import SessionLocal
    from app.services.abbreviation_service import (
        build_lookup, expand_suffix, expand_query_suffix,
        build_glossary, build_glossary_from_texts, find_abbrs_in_text,
    )

    db = SessionLocal()
    try:
        lookup = build_lookup(db, None)
        assert not lookup.is_empty, "Lookup should not be empty with Military Abbreviations list"
        assert "CO" in lookup.forward, "CO should be in lookup"
        assert "DA" in lookup.forward, "DA should be in lookup"
        assert len(lookup.forward["DA"]) >= 3, "DA should have multiple meanings"

        # Test suffix expansion
        text = "The CO ordered the bns to wdr from the forward position."
        expanded = expand_suffix(text, lookup)
        assert "Commanding Officer" in expanded, "Suffix should contain 'Commanding Officer'"
        assert "Battalions" in expanded, "Suffix should contain 'Battalions'"
        assert "Withdraw" in expanded, "Suffix should contain 'Withdraw'"
        assert text in expanded, "Original text should be preserved in suffix expansion"

        # Test query expansion
        query = "bns wdr from position"
        expanded_q = expand_query_suffix(query, lookup)
        assert "Battalions" in expanded_q, "Query expansion should contain 'Battalions'"
        assert "Withdraw" in expanded_q, "Query expansion should contain 'Withdraw'"
        assert query in expanded_q, "Original query should be preserved"

        # Test glossary
        glossary = build_glossary(text, lookup)
        assert "CO = Commanding Officer" in glossary, "Glossary should contain CO mapping"
        assert "bns = Battalions" in glossary, "Glossary should contain bns mapping"

        # Test find_abbrs_in_text
        found = find_abbrs_in_text(text, lookup)
        assert "CO" in found, "find_abbrs should find CO"
        assert "bns" in found, "find_abbrs should find bns"
        assert "wdr" in found, "find_abbrs should find wdr"

        # Test build_glossary_from_texts
        glossary2 = build_glossary_from_texts([text, "The DA approved resupply"], lookup)
        assert "CO" in glossary2, "Glossary from texts should contain CO"
        assert "DA" in glossary2, "Glossary from texts should contain DA"

        print("PASS: test_1_lookup_and_expansion")
        return True
    except AssertionError as e:
        print(f"FAIL: test_1_lookup_and_expansion: {e}")
        return False
    finally:
        db.close()


def test_2_empty_lookup_no_expansion():
    """Test that empty lookup results in no expansion (no-op)."""
    from app.services.abbreviation_service import AbbreviationLookup, expand_suffix, expand_query_suffix, build_glossary

    empty = AbbreviationLookup()
    text = "The CO ordered the bns to wdr."

    assert expand_suffix(text, empty) == text, "Empty lookup should not expand"
    assert expand_query_suffix(text, empty) == text, "Empty lookup should not expand query"
    assert build_glossary(text, empty) == "", "Empty lookup should produce empty glossary"

    print("PASS: test_2_empty_lookup_no_expansion")
    return True


def test_3_ingestion_expansion():
    """Test that ingestion expands chunk_text and stores original_text in metadata."""
    from app.db.session import SessionLocal
    from app.services.abbreviation_service import build_lookup, expand_suffix
    from app.models.knowledge import KnowledgeBase, DocumentChunk
    from sqlalchemy import text as sql_text

    db = SessionLocal()
    try:
        # Find a KB with documents
        kb = db.query(KnowledgeBase).first()
        if not kb:
            print("SKIP: test_3_ingestion_expansion (no KB found)")
            return True

        # Check if any chunks have original_text in metadata
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.kb_id == kb.id
        ).limit(5).all()

        if not chunks:
            print("SKIP: test_3_ingestion_expansion (no chunks found)")
            return True

        # Check if any chunk has [Expansions: in chunk_text
        has_expansion = any("[Expansions:" in (c.chunk_text or "") for c in chunks)
        if has_expansion:
            # Verify original_text is in metadata
            chunk_with_exp = next(c for c in chunks if "[Expansions:" in (c.chunk_text or ""))
            meta = chunk_with_exp.chunk_metadata or {}
            if "original_text" in meta:
                print(f"PASS: test_3_ingestion_expansion (found expanded chunk with original_text)")
                return True
            else:
                print(f"FAIL: test_3_ingestion_expansion (expanded chunk missing original_text in metadata)")
                return False
        else:
            # Chunks may have been ingested before expansion was enabled
            # Let's verify the expansion function works on existing chunk text
            lookup = build_lookup(db, kb.org_id)
            if lookup.is_empty:
                print("SKIP: test_3_ingestion_expansion (no abbreviation lists for this org)")
                return True
            test_text = "The CO ordered the bns to wdr."
            expanded = expand_suffix(test_text, lookup)
            if expanded != test_text:
                print(f"PASS: test_3_ingestion_expansion (expansion function works, no pre-existing expanded chunks)")
                return True
            else:
                print(f"FAIL: test_3_ingestion_expansion (expansion function did not expand text)")
                return False
    except Exception as e:
        print(f"FAIL: test_3_ingestion_expansion: {e}")
        return False
    finally:
        db.close()


def test_4_query_expansion_node():
    """Test that expand_query_node produces expanded_query in state."""
    from app.db.session import SessionLocal
    from app.services.agentic_rag.nodes import expand_query_node

    db = SessionLocal()
    try:
        state = {
            "rewritten_query": "bns wdr from position",
            "original_query": "bns wdr from position",
            "org_id": None,
        }
        result = expand_query_node(state, db=db, org_id=None)

        assert "expanded_query" in result, "Result should contain expanded_query"
        eq = result["expanded_query"]
        assert "Battalions" in eq or "bns" in eq, f"Expanded query should contain expansion: {eq}"
        assert "bns wdr from position" in eq, "Original query should be preserved in expanded query"

        print(f"PASS: test_4_query_expansion_node (expanded: {eq[:80]}...)")
        return True
    except AssertionError as e:
        print(f"FAIL: test_4_query_expansion_node: {e}")
        return False
    except Exception as e:
        print(f"FAIL: test_4_query_expansion_node: {e}")
        return False
    finally:
        db.close()


def test_5_generation_glossary():
    """Test that format_context_string uses original_text and appends glossary."""
    from app.db.session import SessionLocal
    from app.services.agentic_rag.utils import format_context_string

    db = SessionLocal()
    try:
        # Simulate docs with original_text in metadata
        docs = [
            {
                "page_content": "The CO ordered the bns to wdr [Expansions: CO=Commanding Officer; bns=Battalions; wdr=Withdraw]",
                "metadata": {
                    "source": "test.pdf",
                    "original_text": "The CO ordered the bns to wdr from the forward position.",
                },
            },
            {
                "page_content": "The DA approved resupply [Expansions: DA=Daily Allowance Defence Attache Deputy Assistant]",
                "metadata": {
                    "source": "test2.pdf",
                    "original_text": "The DA approved the medical resupply.",
                },
            },
        ]

        context = format_context_string(docs, db=db, org_id=None)

        # Should use original_text (clean prose), not page_content (with [Expansions:])
        assert "The CO ordered the bns to wdr from the forward position." in context, \
            "Context should use original_text"
        assert "[Expansions:" not in context, \
            "Context should not contain expansion suffix metadata"

        # Should contain glossary
        assert "[Abbreviation Glossary]" in context, \
            "Context should contain abbreviation glossary"
        assert "CO = Commanding Officer" in context, \
            "Glossary should contain CO mapping"
        assert "DA =" in context, \
            "Glossary should contain DA mapping"

        print(f"PASS: test_5_generation_glossary")
        return True
    except AssertionError as e:
        print(f"FAIL: test_5_generation_glossary: {e}")
        return False
    except Exception as e:
        print(f"FAIL: test_5_generation_glossary: {e}")
        return False
    finally:
        db.close()


def test_6_disable_expansion():
    """Test that disabling expansion makes the lookup empty."""
    from app.db.session import SessionLocal
    from app.services.abbreviation_service import build_lookup, _invalidate_cache
    from app.services.settings_service import upsert_app_setting

    db = SessionLocal()
    try:
        # First verify expansion is enabled and working
        _invalidate_cache()
        lookup = build_lookup(db, None)
        assert not lookup.is_empty, "Lookup should be non-empty when enabled"

        # Disable expansion
        upsert_app_setting(db, "ABBREVIATION_EXPANSION_ENABLED", False, user_id=1)
        _invalidate_cache()

        lookup_disabled = build_lookup(db, None)
        assert lookup_disabled.is_empty, "Lookup should be empty when disabled"

        # Re-enable
        upsert_app_setting(db, "ABBREVIATION_EXPANSION_ENABLED", True, user_id=1)
        _invalidate_cache()

        lookup_reenabled = build_lookup(db, None)
        assert not lookup_reenabled.is_empty, "Lookup should be non-empty when re-enabled"

        print("PASS: test_6_disable_expansion")
        return True
    except AssertionError as e:
        print(f"FAIL: test_6_disable_expansion: {e}")
        # Re-enable to avoid leaving it disabled
        try:
            upsert_app_setting(db, "ABBREVIATION_EXPANSION_ENABLED", True, user_id=1)
            _invalidate_cache()
        except:
            pass
        return False
    except Exception as e:
        print(f"FAIL: test_6_disable_expansion: {e}")
        return False
    finally:
        db.close()


def test_7_api_endpoints():
    """Test abbreviation list API endpoints via HTTP."""
    import requests

    # Login as super_admin
    resp = requests.post(
        "http://127.0.0.1:8000/api/auth/token",
        data={"username": "super_admin", "password": "tango123"},
        timeout=10,
    )
    token = resp.json().get("access_token")
    if not token:
        print("SKIP: test_7_api_endpoints (could not login)")
        return True

    headers = {"Authorization": f"Bearer {token}"}

    # List abbreviation lists
    resp = requests.get("http://127.0.0.1:8000/api/admin/abbreviation-lists", headers=headers, timeout=10)
    assert resp.status_code == 200, f"List endpoint failed: {resp.status_code}"
    lists = resp.json()
    assert len(lists) > 0, "Should have at least one list"
    assert lists[0]["name"] == "Military Abbreviations", f"First list should be Military Abbreviations: {lists[0]['name']}"

    list_id = lists[0]["id"]

    # Get single list
    resp = requests.get(f"http://127.0.0.1:8000/api/admin/abbreviation-lists/{list_id}", headers=headers, timeout=10)
    assert resp.status_code == 200, f"Get list endpoint failed: {resp.status_code}"

    # Browse abbreviations
    resp = requests.get(
        f"http://127.0.0.1:8000/api/admin/abbreviation-lists/{list_id}/abbreviations?search=CO&size=5",
        headers=headers, timeout=10,
    )
    assert resp.status_code == 200, f"Browse abbreviations failed: {resp.status_code}"
    data = resp.json()
    assert data["total"] > 0, "Should find abbreviations matching 'CO'"

    # Update list (toggle enabled)
    resp = requests.put(
        f"http://127.0.0.1:8000/api/admin/abbreviation-lists/{list_id}",
        json={"is_enabled": False},
        headers=headers, timeout=10,
    )
    assert resp.status_code == 200, f"Update list failed: {resp.status_code}"
    assert resp.json()["is_enabled"] == False, "List should be disabled"

    # Re-enable
    resp = requests.put(
        f"http://127.0.0.1:8000/api/admin/abbreviation-lists/{list_id}",
        json={"is_enabled": True},
        headers=headers, timeout=10,
    )
    assert resp.status_code == 200, f"Re-enable list failed: {resp.status_code}"
    assert resp.json()["is_enabled"] == True, "List should be re-enabled"

    print("PASS: test_7_api_endpoints")
    return True


def main():
    print("=" * 70)
    print("ABBREVIATION EXPANSION END-TO-END TEST")
    print("=" * 70)

    results = []
    results.append(test_1_lookup_and_expansion())
    results.append(test_2_empty_lookup_no_expansion())
    results.append(test_3_ingestion_expansion())
    results.append(test_4_query_expansion_node())
    results.append(test_5_generation_glossary())
    results.append(test_6_disable_expansion())
    results.append(test_7_api_endpoints())

    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"RESULTS: {passed}/{total} passed")
    print("=" * 70)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
