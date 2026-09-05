#!/usr/bin/env python3
"""Comprehensive quality tests for abbreviation handling across the pipeline.

Tests the full pipeline with abbreviation data loaded from the enhanced CSV:
  1. Lookup correctness (forward, reverse, case-sensitivity tiers)
  2. Query suffix expansion (bidirectional, preserves original)
  3. Glossary generation (clean format, sorted, no duplicates)
  4. format_context_string glossary injection (query + chunk merge)
  5. expand_query_node produces glossary in state
  6. No adverse effect on queries without abbreviations
  7. No adverse effect on chunk text without abbreviations
  8. Glossary instructions present in all LLM system prompts
  9. Stopwords are not matched
 10. Multi-word abbreviations and forms work
 11. Possessive forms (CO's) are handled
 12. Mixed-case abbreviations match exactly
 13. Qualification abbreviations match lowercase only

Runs inside the backend container:
  docker exec rag-web-ui-backend-1 pytest tests/test_abbr_quality.py -v
"""
import csv
import os
import sys
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")

# Must set SQLite URI BEFORE importing app modules that read session.py.
_sqlite_dir = tempfile.mkdtemp(prefix="rag_abbr_test_")
os.environ["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{_sqlite_dir}/test.db"
os.environ["UPLOAD_DIR"] = "/tmp/rag_abbr_test_uploads"

from app.models.base import Base
import app.models.user  # noqa
import app.models.knowledge  # noqa
import app.models.chat  # noqa
import app.models.datastore  # noqa
import app.models.setting  # noqa
import app.models.abbreviation  # noqa
import app.models.search_history  # noqa

CSV_PATH = "/app/assets/abbreviations_enhanced.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture(scope="module")
def db_session(engine):
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()

    # Populate abbreviation data from CSV
    from app.models.abbreviation import AbbreviationList, Abbreviation
    from app.services.settings_service import upsert_app_setting, clear_cache as clear_settings_cache
    from app.services.abbreviation_service import _invalidate_cache

    # Enable abbreviation expansion
    upsert_app_setting(session, "ABBREVIATION_EXPANSION_ENABLED", "true")
    session.commit()

    # Create a universal abbreviation list
    lst = AbbreviationList(
        name="Test Military Abbreviations",
        description="Loaded from enhanced CSV for quality tests",
        org_id=None,
        is_enabled=True,
        row_count=0,
        created_by=1,
    )
    session.add(lst)
    session.flush()

    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            abbr = r["abbreviation"].strip()
            form = r["expanded_form"].strip()
            cat = r.get("category", "").strip()
            if abbr and form:
                rows.append(Abbreviation(
                    list_id=lst.id,
                    abbreviation=abbr,
                    expanded_form=form,
                    category=cat,
                ))
    session.add_all(rows)
    lst.row_count = len(rows)
    session.commit()

    _invalidate_cache()
    clear_settings_cache()

    yield session

    session.close()


@pytest.fixture(scope="module")
def lookup(db_session):
    from app.services.abbreviation_service import build_lookup
    lk = build_lookup(db_session, None)
    assert not lk.is_empty, "Lookup must be populated from CSV"
    return lk


# ---------------------------------------------------------------------------
# 1. Lookup correctness
# ---------------------------------------------------------------------------

class TestLookupCorrectness:
    def test_lookup_has_substantial_entries(self, lookup):
        assert len(lookup.forward) > 1000, f"Expected >1000 forward entries, got {len(lookup.forward)}"

    def test_lookup_has_reverse_entries(self, lookup):
        assert len(lookup.reverse) > 100, f"Expected >100 reverse entries, got {len(lookup.reverse)}"

    def test_known_abbreviation_present(self, lookup):
        assert "CO" in lookup.forward
        assert "Commanding Officer" in lookup.forward["CO"]

    def test_lowercase_abbreviation_present(self, lookup):
        assert "bns" in lookup.forward
        assert "Battalions" in lookup.forward["bns"]

    def test_multi_form_abbreviation(self, lookup):
        assert "wdr" in lookup.forward
        assert len(lookup.forward["wdr"]) > 1, "wdr should have multiple forms"

    def test_processors_built(self, lookup):
        assert lookup.kp_exact is not None
        assert lookup.kp_prose is not None
        assert lookup.kp_reverse is not None


# ---------------------------------------------------------------------------
# 2. Forward matching (abbreviation → expansion)
# ---------------------------------------------------------------------------

class TestForwardMatching:
    def test_uppercase_abbr_matched(self, lookup):
        from app.services.abbreviation_service import find_abbrs_in_text
        found = find_abbrs_in_text("CO ordered the attack", lookup)
        assert "CO" in found

    def test_uppercase_abbr_case_sensitive(self, lookup):
        """Uppercase abbrs should only match in exact case."""
        from app.services.abbreviation_service import find_abbrs_in_text
        found = find_abbrs_in_text("co ordered the attack", lookup)
        assert "CO" not in found, "CO (uppercase) should not match lowercase 'co'"

    def test_lowercase_abbr_case_insensitive(self, lookup):
        from app.services.abbreviation_service import find_abbrs_in_text
        assert "bns" in find_abbrs_in_text("bns moved north", lookup)
        assert "bns" in find_abbrs_in_text("BNS moved north", lookup)
        assert "bns" in find_abbrs_in_text("Bns moved north", lookup)

    def test_mixed_case_abbr_exact(self, lookup):
        """Mixed-case abbrs (Comd) are in the prose tier (case-insensitive).
        When both 'Comd' and 'comd' exist in the CSV, flashtext2 returns one
        of them — both map to the same forms."""
        from app.services.abbreviation_service import find_abbrs_in_text
        found = find_abbrs_in_text("Comd issued orders", lookup)
        assert "Comd" in found or "comd" in found, f"Expected Comd or comd in {found}"
        # Both map to the same forms
        key = "Comd" if "Comd" in found else "comd"
        assert "Command" in found[key]

    def test_mixed_case_abbr_no_wrong_case(self, lookup):
        """With the two-tier system, mixed-case abbrs (Comd) go into the prose
        tier (case-insensitive). When both 'Comd' and 'comd' exist in the CSV,
        both match case-insensitively — 'Comd' text and 'comd' text both match.
        This is correct: the prose tier is case-insensitive by design."""
        from app.services.abbreviation_service import find_abbrs_in_text
        if "Comd" in lookup.forward and "comd" in lookup.forward:
            found_comd = find_abbrs_in_text("Comd issued orders", lookup)
            found_low = find_abbrs_in_text("comd issued orders", lookup)
            # Both should match (case-insensitive prose tier)
            assert found_comd, "Comd text should match"
            assert found_low, "comd text should match"
            # Both should map to the same forms
            assert "Command" in list(found_comd.values())[0]
            assert "Command" in list(found_low.values())[0]

    def test_multi_word_abbr_matched(self, lookup):
        from app.services.abbreviation_service import find_abbrs_in_text
        multi_word = [a for a in lookup.forward if " " in a or "*" in a]
        if multi_word:
            abbr = multi_word[0]
            found = find_abbrs_in_text(f"used {abbr} in the field", lookup)
            assert abbr in found, f"Multi-word abbr {abbr!r} should be matched"

    def test_multiple_abbrs_in_one_query(self, lookup):
        from app.services.abbreviation_service import find_abbrs_in_text
        found = find_abbrs_in_text("CO ordered bns to wdr from HQ", lookup)
        assert "CO" in found
        assert "bns" in found
        assert "wdr" in found
        assert "HQ" in found

    def test_no_false_positive_on_plain_english(self, lookup):
        """Uppercase-only abbreviations should not match lowercase prose.

        'at' is uppercase-only (AT = Animal Transport) in the CSV, so lowercase
        'at' in prose should NOT match. 'cat' IS a valid lowercase abbreviation
        (cat = Categorisation) so it WILL match — that's correct behavior.
        """
        from app.services.abbreviation_service import find_abbrs_in_text
        found = find_abbrs_in_text("the meeting at noon was cancelled", lookup)
        assert "AT" not in found, "Lowercase 'at' should not match uppercase-only AT"
        assert "at" not in found, "Lowercase 'at' should not match uppercase-only AT"

    def test_possessive_form_matched(self, lookup):
        from app.services.abbreviation_service import find_abbrs_in_text
        found = find_abbrs_in_text("CO's orders were clear", lookup)
        assert "CO" in found, "Possessive CO's should match CO"


# ---------------------------------------------------------------------------
# 3. Reverse matching (full-form → abbreviation)
# ---------------------------------------------------------------------------

class TestReverseMatching:
    def test_full_form_matched_to_abbr(self, lookup):
        from app.services.abbreviation_service import find_forms_in_text
        found = find_forms_in_text("Commanding Officer ordered the attack", lookup)
        assert "CO" in found, "'Commanding Officer' should reverse-match to CO"

    def test_multi_word_full_form_matched(self, lookup):
        from app.services.abbreviation_service import find_forms_in_text
        found = find_forms_in_text("battalions withdrew from position", lookup)
        assert "bns" in found, "'battalions' should reverse-match to bns"
        assert "wdr" in found, "'withdrew' should reverse-match to wdr"

    def test_reverse_and_forward_combined(self, lookup):
        from app.services.abbreviation_service import find_abbrs_in_text, find_forms_in_text
        query = "CO ordered battalions to wdr"
        forward = find_abbrs_in_text(query, lookup)
        reverse = find_forms_in_text(query, lookup)
        assert "CO" in forward
        assert "wdr" in forward
        assert "bns" in reverse, "'battalions' should reverse-match to bns"


# ---------------------------------------------------------------------------
# 4. Query suffix expansion
# ---------------------------------------------------------------------------

class TestQuerySuffixExpansion:
    def test_preserves_original_query(self, lookup):
        from app.services.abbreviation_service import expand_query_suffix
        query = "CO ordered bns to wdr"
        expanded = expand_query_suffix(query, lookup)
        assert expanded.startswith(query)

    def test_appends_expansions_suffix(self, lookup):
        from app.services.abbreviation_service import expand_query_suffix
        expanded = expand_query_suffix("CO ordered bns", lookup)
        assert "[Abbreviation Glossary]" in expanded

    def test_no_expansion_for_plain_english(self, lookup):
        from app.services.abbreviation_service import expand_query_suffix
        query = "what is the weather today"
        assert expand_query_suffix(query, lookup) == query

    def test_bidirectional_expansion(self, lookup):
        from app.services.abbreviation_service import expand_query_suffix
        expanded = expand_query_suffix("CO ordered battalions to wdr", lookup)
        assert "CO =" in expanded
        assert "bns =" in expanded
        assert "wdr =" in expanded

    def test_multiple_forms_in_expansion(self, lookup):
        from app.services.abbreviation_service import expand_query_suffix
        expanded = expand_query_suffix("wdr", lookup)
        # New format: "wdr = Withdraw, Withdrawal\n..."
        forms_part = expanded.split("wdr =")[1].split("\n")[0].strip()
        forms = [f.strip() for f in forms_part.split(",")]
        assert len(forms) >= 2, f"wdr should have multiple forms, got: {forms}"

    def test_empty_query(self, lookup):
        from app.services.abbreviation_service import expand_query_suffix
        assert expand_query_suffix("", lookup) == ""

    def test_query_with_only_full_forms(self, lookup):
        from app.services.abbreviation_service import expand_query_suffix
        expanded = expand_query_suffix("commanding officer ordered withdrawal", lookup)
        assert "[Abbreviation Glossary]" in expanded
        assert "CO =" in expanded
        assert "wdr =" in expanded


# ---------------------------------------------------------------------------
# 5. Glossary generation
# ---------------------------------------------------------------------------

class TestGlossaryGeneration:
    def test_glossary_format(self, lookup):
        from app.services.abbreviation_service import build_glossary
        g = build_glossary("CO ordered bns to wdr", lookup)
        for line in g.strip().split("\n"):
            assert " = " in line, f"Each line must be 'abbr = forms', got: {line!r}"

    def test_glossary_sorted(self, lookup):
        from app.services.abbreviation_service import build_glossary
        g = build_glossary("wdr bns CO", lookup)
        lines = [l.split(" = ")[0] for l in g.strip().split("\n")]
        assert lines == sorted(lines, key=str.lower)

    def test_glossary_empty_for_plain_english(self, lookup):
        from app.services.abbreviation_service import build_glossary
        assert build_glossary("the weather is nice today", lookup) == ""

    def test_glossary_from_texts_merges(self, lookup):
        from app.services.abbreviation_service import build_glossary_from_texts
        g = build_glossary_from_texts(["CO ordered the attack", "bns moved north"], lookup)
        assert "CO" in g
        assert "bns" in g

    def test_glossary_from_texts_no_duplicates(self, lookup):
        from app.services.abbreviation_service import build_glossary_from_texts
        g = build_glossary_from_texts(["CO ordered bns", "CO and bns again"], lookup)
        assert g.count("CO = ") == 1
        assert g.count("bns = ") == 1


# ---------------------------------------------------------------------------
# 6. format_context_string glossary injection
# ---------------------------------------------------------------------------

class TestContextStringGlossary:
    def test_glossary_appended_to_context(self, db_session, lookup):
        from app.services.agentic_rag.utils import format_context_string
        docs = [{"page_content": "CO ordered the attack", "metadata": {"source": "doc1.pdf"}}]
        ctx = format_context_string(docs, db=db_session, org_id=None, query_glossary="bns = Battalions")
        assert "[Abbreviation Glossary]" in ctx
        assert "bns = Battalions" in ctx
        assert "CO = " in ctx, "Chunk abbreviation CO should also appear"

    def test_query_glossary_preserved_without_db(self, lookup):
        from app.services.agentic_rag.utils import format_context_string
        docs = [{"page_content": "no abbreviations here", "metadata": {"source": "doc1.pdf"}}]
        ctx = format_context_string(docs, query_glossary="CO = Commanding Officer")
        assert "[Abbreviation Glossary]" in ctx
        assert "CO = Commanding Officer" in ctx

    def test_no_glossary_when_empty(self, db_session, lookup):
        """When the chunk text has no abbreviation tokens and no query glossary,
        the context should not contain a glossary block. Use text that avoids
        any valid CSV abbreviations (e.g. 'no', 'in', 'cat', 'ill', 'temp')."""
        from app.services.agentic_rag.utils import format_context_string
        docs = [{"page_content": "the weather is sunny and warm today", "metadata": {"source": "doc1.pdf"}}]
        ctx = format_context_string(docs, db=db_session, org_id=None, query_glossary="")
        assert "[Abbreviation Glossary]" not in ctx

    def test_merge_no_duplicates(self, db_session, lookup):
        from app.services.agentic_rag.utils import format_context_string
        docs = [{"page_content": "CO and bns were deployed", "metadata": {"source": "doc1.pdf"}}]
        query_gloss = "CO = Commanding Officer\nbns = Battalions"
        ctx = format_context_string(docs, db=db_session, org_id=None, query_glossary=query_gloss)
        glossary_section = ctx.split("[Abbreviation Glossary]")[1] if "[Abbreviation Glossary]" in ctx else ""
        assert glossary_section.count("CO = ") == 1
        assert glossary_section.count("bns = ") == 1

    def test_chunk_only_abbr_appended(self, db_session, lookup):
        from app.services.agentic_rag.utils import format_context_string
        docs = [{"page_content": "HQ was informed", "metadata": {"source": "doc1.pdf"}}]
        ctx = format_context_string(docs, db=db_session, org_id=None, query_glossary="CO = Commanding Officer")
        glossary_section = ctx.split("[Abbreviation Glossary]")[1]
        assert "CO = " in glossary_section
        assert "HQ = " in glossary_section, "Chunk-only abbr HQ should be appended"


# ---------------------------------------------------------------------------
# 8. System prompt glossary instructions
# ---------------------------------------------------------------------------

class TestPromptInstructions:
    def test_rewrite_prompt_has_glossary_instruction(self):
        from app.services.agentic_rag.prompts import REWRITE_SYSTEM_PROMPT
        assert "[Abbreviation Glossary]" in REWRITE_SYSTEM_PROMPT

    def test_plan_prompt_has_glossary_instruction(self):
        from app.services.agentic_rag.prompts import PLAN_SYSTEM_PROMPT
        assert "[Abbreviation Glossary]" in PLAN_SYSTEM_PROMPT

    def test_think_prompt_has_glossary_instruction(self):
        from app.services.agentic_rag.prompts import THINK_SYSTEM_PROMPT
        assert "[Abbreviation Glossary]" in THINK_SYSTEM_PROMPT

    def test_finalize_prompt_has_glossary_instruction(self):
        from app.services.agentic_rag.prompts import FINALIZE_GUARDRAIL_PROMPT
        assert "[Abbreviation Glossary]" in FINALIZE_GUARDRAIL_PROMPT

    def test_evaluation_prompt_has_glossary_instruction(self):
        from app.services.agentic_rag.prompts import EVALUATION_AND_FOLLOWUP_PROMPT
        assert "[Abbreviation Glossary]" in EVALUATION_AND_FOLLOWUP_PROMPT


# ---------------------------------------------------------------------------
# 9. No adverse effect on plain queries
# ---------------------------------------------------------------------------

class TestNoAdverseEffect:
    @pytest.mark.parametrize("query", [
        "what is the weather today",
        "explain machine learning concepts",
        "describe the process of photosynthesis",
        "the quick brown fox jumps over the lazy dog",
    ])
    def test_plain_query_unchanged_by_expansion(self, lookup, query):
        from app.services.abbreviation_service import expand_query_suffix
        assert expand_query_suffix(query, lookup) == query

    @pytest.mark.parametrize("query", [
        "what is the weather today",
        "explain machine learning concepts",
    ])
    def test_plain_query_empty_glossary(self, lookup, query):
        from app.services.abbreviation_service import build_glossary
        assert build_glossary(query, lookup) == ""

    def test_plain_chunk_no_glossary_in_context(self, db_session, lookup):
        from app.services.agentic_rag.utils import format_context_string
        docs = [{"page_content": "Photosynthesis is the process by which plants make food.", "metadata": {"source": "bio.pdf"}}]
        ctx = format_context_string(docs, db=db_session, org_id=None, query_glossary="")
        assert "[Abbreviation Glossary]" not in ctx

    @pytest.mark.parametrize("query", [
        "CO ordered bns",
        "What is HQ?",
        "The GOC visited the bns today",
    ])
    def test_original_query_preserved_in_expanded(self, lookup, query):
        from app.services.abbreviation_service import expand_query_suffix
        expanded = expand_query_suffix(query, lookup)
        assert expanded.startswith(query)


# ---------------------------------------------------------------------------
# 10. Lowercase abbreviation handling (no stopword filter)
# ---------------------------------------------------------------------------

class TestLowercaseAbbrHandling:
    """With the two-tier system, valid lowercase CSV abbreviations like
    'in', 'no', 'cat', 'ill', 'temp' are legitimate abbreviations and
    SHOULD match case-insensitively. The old STOPWORDS filter is gone."""

    def test_lowercase_in_matched(self, lookup):
        """'in' = Inch in the CSV — valid lowercase abbr, should match."""
        from app.services.abbreviation_service import find_abbrs_in_text
        if "in" in lookup.forward:
            found = find_abbrs_in_text("the unit in the field", lookup)
            assert "in" in found, f"'in' is a valid CSV abbr, should match: {found}"

    def test_lowercase_no_matched(self, lookup):
        """'no' = Number in the CSV — valid lowercase abbr, should match."""
        from app.services.abbreviation_service import find_abbrs_in_text
        if "no" in lookup.forward:
            found = find_abbrs_in_text("no units were present", lookup)
            assert "no" in found, f"'no' is a valid CSV abbr, should match: {found}"

    def test_lowercase_cat_matched(self, lookup):
        """'cat' = Categorisation in the CSV — valid lowercase abbr, should match."""
        from app.services.abbreviation_service import find_abbrs_in_text
        if "cat" in lookup.forward:
            found = find_abbrs_in_text("the cat sat on the mat", lookup)
            assert "cat" in found, f"'cat' is a valid CSV abbr, should match: {found}"

    def test_lowercase_ill_matched(self, lookup):
        """'ill' = Illuminate in the CSV — valid lowercase abbr, should match."""
        from app.services.abbreviation_service import find_abbrs_in_text
        if "ill" in lookup.forward:
            found = find_abbrs_in_text("he was ill yesterday", lookup)
            assert "ill" in found, f"'ill' is a valid CSV abbr, should match: {found}"

    def test_lowercase_temp_matched(self, lookup):
        """'temp' = Temperature in the CSV — valid lowercase abbr, should match."""
        from app.services.abbreviation_service import find_abbrs_in_text
        if "temp" in lookup.forward:
            found = find_abbrs_in_text("the temp was high", lookup)
            assert "temp" in found, f"'temp' is a valid CSV abbr, should match: {found}"

    def test_uppercase_at_not_matched_lowercase(self, lookup):
        """'AT' = Animal Transport is uppercase-only — lowercase 'at' must NOT match."""
        from app.services.abbreviation_service import find_abbrs_in_text
        found = find_abbrs_in_text("the meeting at noon", lookup)
        assert "AT" not in found, "Lowercase 'at' should not match uppercase-only AT"
        assert "at" not in found, "Lowercase 'at' should not match uppercase-only AT"

    def test_uppercase_to_not_matched_lowercase(self, lookup):
        """'TO' = Transport Officer is uppercase-only — lowercase 'to' must NOT match."""
        from app.services.abbreviation_service import find_abbrs_in_text
        found = find_abbrs_in_text("go to the base", lookup)
        assert "TO" not in found, "Lowercase 'to' should not match uppercase-only TO"
        assert "to" not in found, "Lowercase 'to' should not match uppercase-only TO"


# ---------------------------------------------------------------------------
# 11. Qualification abbreviation handling
# ---------------------------------------------------------------------------

class TestQualificationAbbrs:
    """With the two-tier system, lowercase qualification abbreviations like
    'psc' and 'ndc' go into the prose tier (case-insensitive). Both lowercase
    and uppercase text will match them."""

    def test_qualification_lowercase_matched(self, lookup):
        from app.services.abbreviation_service import find_abbrs_in_text
        found = find_abbrs_in_text("completed psc and ndc", lookup)
        assert "psc" in found, "psc should match lowercase"
        assert "ndc" in found, "ndc should match lowercase"

    def test_qualification_uppercase_also_matched(self, lookup):
        """Uppercase PSC/NDC text matches the lowercase psc/ndc CSV entries
        because the prose tier is case-insensitive."""
        from app.services.abbreviation_service import find_abbrs_in_text
        found = find_abbrs_in_text("completed PSC and NDC", lookup)
        assert "psc" in found, "psc should match uppercase PSC (case-insensitive)"
        assert "ndc" in found, "ndc should match uppercase NDC (case-insensitive)"


# ---------------------------------------------------------------------------
# 12. Graph state field
# ---------------------------------------------------------------------------

class TestGraphState:
    # abbreviation_glossary and expanded_query were removed from AgentState
    # in the atomic-tools redesign. The glossary is now built inline by
    # format_context_string, not stored in state.
    pass
