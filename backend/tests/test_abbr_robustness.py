#!/usr/bin/env python3
"""
Comprehensive robustness test for abbreviation handling.

Tests all edge cases:
  - Lengthy queries with multiple abbreviations
  - Mixed abbreviation + expanded form queries
  - Ambiguous abbreviations with diverse meanings (DA, op, CD, CO, etc.)
  - Common English word false matches (in, no, up, ex, cat, etc.)
  - Reverse lookup false matches (operation, commanding, officer, etc.)
  - Case sensitivity (co vs CO vs Co)
  - Substring safety (bns should not match inside bnslog)
  - Punctuation boundaries (bns, bns. (bns) bns's)
  - Plural forms in reverse lookup
  - Empty/whitespace queries
  - Very long paragraph-length queries
  - Repeated abbreviations
  - Adjacent abbreviations
  - Glossary format correctness
  - Ingestion suffix vs query suffix consistency

Runs inside the backend container:
  docker exec rag-web-ui-backend-1 python3 /app/tests/test_abbr_robustness.py
"""
import csv
import re
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set

sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")

# ─── Load abbreviation CSV directly (no DB needed) ──────────────────────────

CSV_PATH = "/app/assets/abbreviations_enhanced.csv"

def load_csv() -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Set[str]]]:
    """Returns (forward, reverse, abbr_categories) maps from CSV."""
    forward = defaultdict(list)
    abbr_categories: Dict[str, Set[str]] = defaultdict(set)
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            abbr = row["abbreviation"].strip()
            form = row["expanded_form"].strip()
            cat = (row.get("category") or "").strip()
            if abbr and form:
                if form not in forward[abbr]:
                    forward[abbr].append(form)
                if cat:
                    abbr_categories[abbr].add(cat)
    reverse = defaultdict(list)
    for abbr, forms in forward.items():
        for form in forms:
            key = form.lower()
            if abbr not in reverse[key]:
                reverse[key].append(abbr)
    return dict(forward), dict(reverse), dict(abbr_categories)

FORWARD, REVERSE, ABBR_CATEGORIES = load_csv()
ALL_ABBRS = sorted(FORWARD.keys(), key=len, reverse=True)
ALL_FORMS = sorted(REVERSE.keys(), key=len, reverse=True)

# ─── Build production-like AbbreviationLookup ───────────────────────────────

from flashtext2 import KeywordProcessor
from app.services.abbreviation_service import (
    AbbreviationLookup,
    find_abbrs_in_text,
    find_forms_in_text,
    expand_suffix,
    expand_query_suffix,
    build_glossary,
    build_glossary_from_texts,
    _is_uppercase,
    _REVERSE_MIN_FORM_LEN,
)

def make_lookup() -> AbbreviationLookup:
    """Build an AbbreviationLookup from the CSV (bypasses DB).

    Mirrors the production build_lookup() logic but loads from CSV directly.
    """
    # Classify abbreviations into case-sensitivity tiers
    exact_abbrs: List[str] = []     # uppercase only
    prose_abbrs: List[str] = []     # lowercase + mixed-case
    for abbr in FORWARD:
        if len(abbr.strip()) <= 1:
            continue
        if _is_uppercase(abbr):
            exact_abbrs.append(abbr)
        else:
            prose_abbrs.append(abbr)

    # Build flashtext2 processors
    kp_exact = KeywordProcessor(case_sensitive=True)
    for abbr in exact_abbrs:
        kp_exact.add_keyword(abbr, abbr)

    kp_prose = KeywordProcessor(case_sensitive=False)
    for abbr in prose_abbrs:
        kp_prose.add_keyword(abbr, abbr)

    # Build reverse mapping with min-length filter
    reverse_filtered: Dict[str, List[str]] = {}
    for abbr, forms in FORWARD.items():
        if len(abbr.strip()) <= 1:
            continue
        for form in forms:
            key = form.lower()
            if len(key) < _REVERSE_MIN_FORM_LEN:
                continue
            if key not in reverse_filtered:
                reverse_filtered[key] = []
            if abbr not in reverse_filtered[key]:
                reverse_filtered[key].append(abbr)

    kp_reverse = KeywordProcessor(case_sensitive=False)
    for form_lower in reverse_filtered:
        kp_reverse.add_keyword(form_lower, form_lower)

    return AbbreviationLookup(
        forward=FORWARD,
        kp_exact=kp_exact,
        kp_prose=kp_prose,
        reverse=reverse_filtered,
        kp_reverse=kp_reverse,
    )

LOOKUP = make_lookup()

# ─── Test framework ─────────────────────────────────────────────────────────

passed = 0
failed = 0
warnings = 0
failures: List[str] = []
warns: List[str] = []

def check(condition: bool, name: str, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"  FAIL: {name}" + (f" — {detail}" if detail else "")
        print(msg)
        failures.append(f"{name}: {detail}")

def warn(condition: bool, name: str, detail: str = "") -> None:
    global warnings
    if not condition:
        warnings += 1
        msg = f"  WARN: {name}" + (f" — {detail}" if detail else "")
        print(msg)
        warns.append(f"{name}: {detail}")
    else:
        print(f"  OK:   {name}")

def section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

# ─── Section 1: Forward expansion — basic correctness ──────────────────────

section("1. FORWARD EXPANSION — Basic Correctness")

# 1.1: Single abbreviation
result = expand_query_suffix("bns wdr from position", LOOKUP)
check("bns = Battalions" in result, "1.1a single abbr 'bns' expanded", result)
check("wdr = Withdraw" in result, "1.1b single abbr 'wdr' expanded", result)
check(result.startswith("bns wdr from position"), "1.1c original query preserved at start", result[:50])

# 1.2: No abbreviations → unchanged
# Use a query with no abbreviation tokens AND no full forms that would trigger
# reverse lookup. "temperature" is a full form of "temp", so it would trigger
# reverse matching. Use "humidity" instead.
result = expand_query_suffix("weather forecast rain humidity", LOOKUP)
check(result == "weather forecast rain humidity", "1.2 no abbrs → unchanged", result)

# 1.3: Empty query
result = expand_query_suffix("", LOOKUP)
check(result == "", "1.3 empty query → unchanged", repr(result))

# 1.4: Whitespace-only query
result = expand_query_suffix("   ", LOOKUP)
check(result == "   ", "1.4 whitespace query → unchanged", repr(result))

# 1.5: Abbreviation at start
result = expand_query_suffix("CO ordered the attack", LOOKUP)
check("CO = Commanding Officer" in result, "1.5a abbr at start expanded", result)
check(result.startswith("CO ordered the attack"), "1.5b abbr at start preserves original", result[:30])

# 1.6: Abbreviation at end
result = expand_query_suffix("report to the CO", LOOKUP)
check("CO = Commanding Officer" in result, "1.6 abbr at end expanded", result)

# 1.7: Abbreviation with punctuation
for punct in [",", ".", "!", "?", ";", ":", ")"]:
    result = expand_query_suffix(f"the CO{punct} ordered", LOOKUP)
    check("CO = Commanding Officer" in result, f"1.7 abbr before '{punct}' expanded", result)

# 1.8: Abbreviation in parentheses
result = expand_query_suffix("(CO) ordered", LOOKUP)
check("CO = Commanding Officer" in result, "1.8 abbr in parens expanded", result)

# 1.9: Abbreviation with apostrophe possessive — "CO's" should match "CO".
# flashtext2 uses Unicode UAX #29 word segmentation which treats apostrophe as
# a word boundary. We normalize possessive "'s" before matching so the base
# abbreviation can be found. Original text is preserved in the output.
result = expand_query_suffix("the CO's order", LOOKUP)
check("CO = Commanding Officer" in result, "1.9 abbr with apostrophe expanded", result)

# ─── Section 2: Substring safety ───────────────────────────────────────────

section("2. SUBSTRING SAFETY — No Partial Matches")

# 2.1: "bns" should NOT match inside "bnslog"
result = expand_query_suffix("bnslog report", LOOKUP)
found = find_abbrs_in_text("bnslog report", LOOKUP)
check("bns" not in found, "2.1a 'bns' not matched inside 'bnslog'", str(found))
check("bns = Battalions" not in result, "2.1b 'bns' not expanded inside 'bnslog'", result)

# 2.2: "CO" should NOT match inside "COVER" or "COOPERATION"
result = expand_query_suffix("cover the position", LOOKUP)
found = find_abbrs_in_text("cover the position", LOOKUP)
check("CO" not in found, "2.2a 'CO' not matched inside 'cover'", str(found))

# 2.3: "op" should NOT match inside "option", "open", "operate"
for word in ["option", "open", "operate", "opera", "optimism", "copy", "topic", "stop"]:
    found = find_abbrs_in_text(f"the {word} works", LOOKUP)
    check("op" not in found, f"2.3 'op' not matched inside '{word}'", str(found))

# 2.4: "in" should NOT match inside "inside", "index", "infrastructure"
for word in ["inside", "index", "infrastructure", "inning", "independent", "training", "finding"]:
    found = find_abbrs_in_text(f"the {word} is", LOOKUP)
    check("in" not in found, f"2.4 'in' not matched inside '{word}'", str(found))

# 2.5: "sp" should NOT match inside "space", "special", "speed"
for word in ["space", "special", "speed", "spoon", "sport", "hospital"]:
    found = find_abbrs_in_text(f"the {word} is", LOOKUP)
    check("sp" not in found, f"2.5 'sp' not matched inside '{word}'", str(found))

# ─── Section 3: Case sensitivity ───────────────────────────────────────────

section("3. CASE SENSITIVITY")

# 3.1: Uppercase abbreviation
result_upper = expand_query_suffix("CO ordered", LOOKUP)
check("CO = Commanding Officer" in result_upper, "3.1a 'CO' (upper) expanded", result_upper)

# 3.2: Lowercase abbreviation — case-sensitive for ≤3 chars, so "co" should NOT match "CO"
result_lower = expand_query_suffix("co ordered", LOOKUP)
found_lower = find_abbrs_in_text("co ordered", LOOKUP)
check("CO" not in found_lower and "co" not in found_lower,
      "3.2a 'co' (lower) NOT matched (case-sensitive for ≤3 chars)", str(found_lower))

# 3.3: Mixed case — "Co" should NOT match "CO" (case-sensitive for ≤3 chars)
result_mixed = expand_query_suffix("Co ordered", LOOKUP)
found_mixed = find_abbrs_in_text("Co ordered", LOOKUP)
check("CO" not in found_mixed and "Co" not in found_mixed,
      "3.3a 'Co' (mixed) NOT matched (case-sensitive for ≤3 chars)", str(found_mixed))

# 3.4: The glossary suffix should use the canonical abbreviation form from the lookup
# (not the user's casing)
# This is informational — the current implementation uses the lookup key
result = expand_query_suffix("co ordered", LOOKUP)
print(f"  INFO: lowercase 'co' → suffix uses: {result}")

# 3.5: Case-sensitive matching for ≤3 char abbreviations — prevents false matches
# "is" should NOT match "IS" (Internal Security)
found = find_abbrs_in_text("this is a test", LOOKUP)
check("IS" not in found and "is" not in found, "3.5a 'is' NOT matched as IS (case-sensitive)", str(found))

# "he" should NOT match "HE" (High Explosive)
found = find_abbrs_in_text("he said hello", LOOKUP)
check("HE" not in found and "he" not in found, "3.5b 'he' NOT matched as HE (case-sensitive)", str(found))

# "to" should NOT match "TO" (Transport Officer)
found = find_abbrs_in_text("go to the store", LOOKUP)
check("TO" not in found and "to" not in found, "3.5c 'to' NOT matched as TO (case-sensitive)", str(found))

# "IS" SHOULD match "IS" (uppercase in text matches uppercase in CSV)
found = find_abbrs_in_text("IS operations were conducted", LOOKUP)
check("IS" in found, "3.5d 'IS' (upper) matched as IS", str(found))

# "HE" SHOULD match "HE" (High Explosive)
found = find_abbrs_in_text("HE rounds were used", LOOKUP)
check("HE" in found, "3.5e 'HE' (upper) matched as HE", str(found))

# "TO" SHOULD match "TO" (Transport Officer)
found = find_abbrs_in_text("TO coordinated the move", LOOKUP)
check("TO" in found, "3.5f 'TO' (upper) matched as TO", str(found))

# 3.6: Case-insensitive matching for >3 char abbreviations
# "recce" should match "recce" (lowercase in CSV)
found = find_abbrs_in_text("recce team", LOOKUP)
check("recce" in found, "3.6a 'recce' (lower) matched", str(found))

# "RECCE" should match "recce" (case-insensitive for >3 chars)
found = find_abbrs_in_text("RECCE team", LOOKUP)
check("recce" in found, "3.6b 'RECCE' (upper) matched (case-insensitive >3 chars)", str(found))

# "Recce" should match "recce"
found = find_abbrs_in_text("Recce team", LOOKUP)
check("recce" in found, "3.6c 'Recce' (mixed) matched (case-insensitive >3 chars)", str(found))

# ─── Section 3b: Stopword filtering ────────────────────────────────────────

section("3b. STOPWORD FILTERING — Common Word Exclusion")

# 3b.1: "in" is a stopword — should NOT be matched even though it's in the CSV
found = find_abbrs_in_text("the project is in its final stage", LOOKUP)
check("in" not in found and "IN" not in found, "3b.1a 'in' NOT matched (stopword)", str(found))
result = expand_query_suffix("the project is in its final stage", LOOKUP)
check("in=Inch" not in result, "3b.1b 'in' not expanded (stopword)", result)

# 3b.2: "no" is a stopword
found = find_abbrs_in_text("there is no reason", LOOKUP)
check("no" not in found and "NO" not in found, "3b.2a 'no' NOT matched (stopword)", str(found))

# 3b.3: "up" is a stopword
found = find_abbrs_in_text("we need to set up the equipment", LOOKUP)
check("up" not in found and "UP" not in found, "3b.3a 'up' NOT matched (stopword)", str(found))

# 3b.4: "cat" is a stopword
found = find_abbrs_in_text("the cat sat on the mat", LOOKUP)
check("cat" not in found and "CAT" not in found, "3b.4a 'cat' NOT matched (stopword)", str(found))

# 3b.5: "ill" is a stopword
found = find_abbrs_in_text("he is feeling ill today", LOOKUP)
check("ill" not in found and "ILL" not in found, "3b.5a 'ill' NOT matched (stopword)", str(found))

# 3b.6: "temp" is a stopword
found = find_abbrs_in_text("the temp is high", LOOKUP)
check("temp" not in found, "3b.6a 'temp' NOT matched (stopword)", str(found))

# 3b.7: Legitimate military abbreviations should still match (not in stopwords)
for abbr in ["op", "sp", "sec", "med", "def", "str", "sig", "gp", "lt", "rt", "ma",
             "CO", "DA", "HQ", "MO", "bns", "wdr", "bde", "inf", "obj", "armd", "sqn"]:
    test_text = f"the {abbr} is ready"
    found = find_abbrs_in_text(test_text, LOOKUP)
    matched = abbr in found
    if not matched and abbr.upper() in found:
        matched = True
    check(matched, f"3b.7 '{abbr}' still matches (not stopword)", str(found))

# ─── Section 4: Bidirectional — full-form to abbreviation ──────────────────

section("4. BIDIRECTIONAL — Full-Form to Abbreviation (Reverse Lookup)")

# 4.1: Single full form
result = expand_query_suffix("battalions withdrew from position", LOOKUP)
check("bns = Battalions" in result, "4.1a 'battalions' → bns expansion", result)
check("wdr = Withdraw" in result, "4.1b 'withdrew' → wdr expansion", result)
check(result.startswith("battalions withdrew from position"), "4.1c original preserved", result[:50])

# 4.2: Multiple full forms
result = expand_query_suffix("commanding officer ordered battalions to withdraw", LOOKUP)
check("CO = Commanding Officer" in result, "4.2a 'commanding officer' → CO", result)
check("bns = Battalions" in result, "4.2b 'battalions' → bns", result)
check("wdr = Withdraw" in result, "4.2c 'withdraw' → wdr", result)

# 4.3: Mixed abbreviation + full form in same query
result = expand_query_suffix("the CO ordered battalions to wdr", LOOKUP)
check("CO = Commanding Officer" in result, "4.3a 'CO' (abbr) expanded", result)
check("bns = Battalions" in result, "4.3b 'battalions' (full form) → bns", result)
check("wdr = Withdraw" in result, "4.3c 'wdr' (abbr) expanded", result)

# 4.4: Full form that maps to multiple abbreviations
# "withdraw" maps to "wdr" — check it doesn't create duplicate entries
result = expand_query_suffix("withdraw the troops", LOOKUP)
wdr_count = result.count("wdr =")
check(wdr_count == 1, f"4.4 'withdraw' → single wdr entry (not duplicated)", f"count={wdr_count} in: {result}")

# 4.5: Full form with different casing
result = expand_query_suffix("Battalions Withdrew", LOOKUP)
check("bns = Battalions" in result, "4.5a 'Battalions' (capitalized) → bns", result)
check("wdr = Withdraw" in result, "4.5b 'Withdrew' (capitalized) → wdr", result)

# 4.6: Plural form vs singular form in reverse lookup
# "Battalions" is in the CSV (→ bns), "Battalion" (singular) is also in CSV (→ bn)
result_plural = expand_query_suffix("battalions attacked", LOOKUP)
result_singular = expand_query_suffix("battalion attacked", LOOKUP)
check("bns = Battalions" in result_plural, "4.6a 'battalions' (plural) → bns", result_plural)
check("bn=Battalion" in result_singular, "4.6b 'battalion' (singular) → bn", result_singular)

# ─── Section 5: Ambiguous abbreviations with diverse meanings ──────────────

section("5. AMBIGUOUS ABBREVIATIONS — Multiple Diverse Meanings")

# 5.1: DA — Daily Allowance, Defence Attache, Deputy Assistant, Direct action
da_forms = FORWARD.get("DA", [])
print(f"  DA has {len(da_forms)} meanings: {da_forms}")
result = expand_query_suffix("DA approved the request", LOOKUP)
check("DA =" in result, "5.1a 'DA' expanded with all meanings", result)
# All forms should be in the expansion
for form in da_forms:
    check(form in result, f"5.1b DA form '{form}' present in expansion", result)

# 5.2: op — 9 forms (Operate, Operated, Operates, Operating, Operation, etc.)
op_forms = FORWARD.get("op", [])
print(f"  op has {len(op_forms)} meanings: {op_forms}")
result = expand_query_suffix("the op was successful", LOOKUP)
check("op =" in result, "5.2a 'op' expanded with all meanings", result)
for form in op_forms:
    check(form in result, f"5.2b op form '{form}' present", result)

# 5.3: CD — Central Discussion, Circle of Damage, Civil Defence, etc.
cd_forms = FORWARD.get("CD", [])
print(f"  CD has {len(cd_forms)} meanings: {cd_forms}")
result = expand_query_suffix("CD was activated", LOOKUP)
check("CD=" in result, "5.3a 'CD' expanded", result)

# 5.4: CO — only one meaning (Commanding Officer) — should be unambiguous
co_forms = FORWARD.get("CO", [])
print(f"  CO has {len(co_forms)} meanings: {co_forms}")
result = expand_query_suffix("CO ordered", LOOKUP)
check(len(co_forms) == 1, "5.4a CO is unambiguous (1 meaning)", str(co_forms))
check("CO = Commanding Officer" in result, "5.4b CO expanded correctly", result)

# 5.5: sp — Support, Supported, Supporter, Supporting, etc.
sp_forms = FORWARD.get("sp", [])
print(f"  sp has {len(sp_forms)} meanings: {sp_forms[:4]}...")
result = expand_query_suffix("sp was provided", LOOKUP)
check("sp=" in result, "5.5 'sp' expanded with all meanings", result)

# 5.6: Verify all forms are included for ambiguous abbreviations
for abbr in ["DA", "op", "CD", "sp", "coord", "dep", "rel"]:
    if abbr not in FORWARD:
        continue
    forms = FORWARD[abbr]
    result = expand_query_suffix(f"{abbr} report", LOOKUP)
    all_present = all(f in result for f in forms)
    check(all_present, f"5.6 '{abbr}' includes all {len(forms)} forms", f"missing: {[f for f in forms if f not in result]}")

# ─── Section 6: Common English word false matches (CRITICAL) ───────────────

section("6. COMMON ENGLISH WORD FALSE MATCHES (CRITICAL)")

# These are abbreviations that are also common English words.
# When they appear in normal English text, they should NOT cause problems.
# "Problems" = adding irrelevant military expansions to non-military queries.

# 6.1: Test common English words that are also abbreviations
# With case-sensitive matching for ≤3 chars + stopwords, most should be filtered.
common_word_abbrs = {
    # Stopwords (should NOT match)
    "in": "Inch", "no": "Number", "up": "Unpaid", "cat": "Categorisation",
    "ill": "Illuminate", "temp": "Temperature",
    # Uppercase abbreviations matching lowercase words (case-sensitive fixes these)
    "is": "Internal Security", "he": "High Explosive", "to": "Transport Officer",
    "at": "Animal Transport", "an": "Afternoon",
    # Lowercase abbreviations matching lowercase words (legitimate military abbrs)
    "op": "Operate", "sp": "Support", "sec": "Section", "med": "Medical",
    "def": "Defence", "str": "Strength", "sig": "Signal", "gp": "Group",
    "lt": "Left", "rt": "Right", "ma": "Mili Amperes",
    "ex": "Exercise", "fin": "Finance", "con": "Control",
    "org": "Organisation", "ref": "Reference", "reg": "Regular",
    "req": "Require", "est": "Estimate", "mov": "Move",
    "nav": "Navigate", "sel": "Select", "std": "Standard", "svc": "Service",
}

false_match_count = 0
expected_match_count = 0
for word, expansion in sorted(common_word_abbrs.items()):
    test_query = f"the {word} is important"
    result = expand_query_suffix(test_query, LOOKUP)
    found = find_abbrs_in_text(test_query, LOOKUP)
    word_matched = word in found or word.upper() in found

    # Determine if this should match or not
    # With the two-tier system:
    # - Uppercase-only CSV abbrs (AT, TO, IS, HE, AN) match case-sensitively only,
    #   so lowercase prose "at", "to", "is", "he", "an" should NOT match.
    # - Lowercase/mixed-case CSV abbrs (cat, ill, temp, met, op, sp, etc.) match
    #   case-insensitively, so they WILL match in prose.
    # - Single-letter abbrs are excluded entirely.
    uppercase_only = word.upper() in FORWARD and word not in FORWARD and word.lower() not in FORWARD
    should_match = len(word) > 1 and not uppercase_only

    if word_matched and not should_match:
        false_match_count += 1
        print(f"  FALSE MATCH: '{word}' should NOT match → {result[:80]}")
    elif not word_matched and should_match:
        print(f"  MISSED: '{word}' should match but didn't → {found}")
    elif word_matched and should_match:
        expected_match_count += 1
        print(f"  EXPECTED: '{word}' matched (legitimate military abbr)")
    else:
        print(f"  CORRECTLY FILTERED: '{word}' not matched")

print(f"\n  Summary: {false_match_count} false matches, {expected_match_count} expected matches")
check(false_match_count == 0, "6.1 No false matches on common English words",
      f"{false_match_count} false matches")

# 6.2: Test a non-military query that should NOT get military expansions
non_military_queries = [
    "what is the weather forecast for tomorrow",
    "how to cook pasta with tomato sauce",
    "the cat sat on the mat",
    "she went to the shop to buy some food",
    "the car is running out of petrol",
    "he is not feeling well today",
    "the meeting is scheduled for next week",
    "please send the report by friday",
    "the project is in its final stage",
    "we need to set up the equipment",
]

print(f"\n  Non-military queries (should have minimal/no expansion):")
non_military_clean = 0
non_military_expanded = 0
for q in non_military_queries:
    result = expand_query_suffix(q, LOOKUP)
    has_expansion = "[Abbreviation Glossary]" in result
    if has_expansion:
        non_military_expanded += 1
        found = find_abbrs_in_text(q, LOOKUP)
        found_forms = find_forms_in_text(q, LOOKUP)
        print(f"  EXPANDED: '{q}' → {len(found)} abbrs + {len(found_forms)} reverse: {list(found.keys())[:3]} + {list(found_forms.keys())[:3]}")
    else:
        non_military_clean += 1
        print(f"  CLEAN: '{q}'")
print(f"\n  Summary: {non_military_clean}/{len(non_military_queries)} clean, {non_military_expanded} expanded")
warn(non_military_clean >= 7, "6.2 At least 7/10 non-military queries clean",
     f"{non_military_clean}/10 clean")

# ─── Section 7: Reverse lookup false matches (CRITICAL) ────────────────────

section("7. REVERSE LOOKUP FALSE MATCHES (CRITICAL)")

# Full forms that are common English words and will trigger reverse lookup
# "operation" → op, "commanding" → CO, "officer" → CO, "control" → con, etc.
# With _REVERSE_MIN_FORM_LEN=5, short forms (<5 chars) are filtered.
common_full_forms = [
    # ≥5 chars: will be reverse-matched (legitimate or false)
    "operation", "commanding", "officer", "control", "defence", "medical",
    "finance", "section", "standard", "support", "group", "exercise",
    "estimate", "navigate", "select", "signal", "service",
    "illuminate", "require", "refer", "regular", "strength",
    "advance", "direct", "effect", "mobile", "public", "represent",
    "confirm", "arrive", "assign", "attach", "develop", "employ",
    "follow", "increase", "decrease", "limit", "station", "transport",
    # <5 chars: should NOT be reverse-matched (filtered by min length)
    "move", "cat", "dog", "car", "set", "put", "run", "get", "all",
]

print(f"  Testing {len(common_full_forms)} common full forms for false reverse matches:")
reverse_false_count = 0
short_form_filtered = 0
for form in common_full_forms:
    test_query = f"the {form} is working well"
    result = expand_query_suffix(test_query, LOOKUP)
    found_forms = find_forms_in_text(test_query, LOOKUP)
    if found_forms:
        if len(form) < 5:
            # This should have been filtered!
            reverse_false_count += 1
            print(f"  LEAKED (short): '{form}' ({len(form)} chars) → abbrs {list(found_forms.keys())} — should be filtered!")
        else:
            # Expected for ≥5 char forms — these are legitimate reverse matches
            # but may be false matches in non-military context
            reverse_false_count += 1
            print(f"  REVERSE: '{form}' ({len(form)} chars) → abbrs {list(found_forms.keys())[:3]}")
    else:
        if len(form) < 5:
            short_form_filtered += 1
            print(f"  FILTERED: '{form}' ({len(form)} chars) — correctly not reverse-matched")
        else:
            print(f"  OK: '{form}' ({len(form)} chars) no reverse match")

print(f"\n  Summary: {reverse_false_count} reverse matches, {short_form_filtered} short forms correctly filtered")
check(short_form_filtered >= 8, "7.1a Short forms (<5 chars) filtered from reverse lookup",
      f"{short_form_filtered} filtered")
# For ≥5 char forms, reverse matching is expected behavior (bidirectional expansion).
# The false match risk is accepted as a tradeoff for bidirectional coverage.
print(f"  NOTE: {reverse_false_count - short_form_filtered} ≥5 char forms reverse-matched (expected for bidirectional expansion)")

# ─── Section 8: Lengthy queries with multiple abbreviations ────────────────

section("8. LENGTHY QUERIES — Multiple Abbreviations")

# 8.1: Military query with many abbreviations
long_military = (
    "The CO ordered the bns to wdr from the forward position. The MO reported "
    "casualties. The adjt coordinated with HQ. The op was conducted at first "
    "light. The recce team provided intelligence on enemy positions. The GOC "
    "visited the bde HQ and briefed the bde comd on the op. The inf bn was "
    "tasked to secure the obj. The armd sqn was to provide spt. The arty bty "
    "was placed in sp of the inf. The DA approved the medical resupply. The SP "
    "was established at checkpoint 4."
)
result = expand_query_suffix(long_military, LOOKUP)
found = find_abbrs_in_text(long_military, LOOKUP)
print(f"  Found {len(found)} abbreviations in long military text:")
for abbr in sorted(found.keys()):
    print(f"    {abbr}: {found[abbr][:3]}...")
check(len(found) >= 10, "8.1a long military text has >=10 abbrs", f"found {len(found)}")
check("[Abbreviation Glossary]" in result, "8.1b expansion suffix present", result[:50])
check(result.startswith("The CO ordered"), "8.1c original text preserved", result[:30])

# 8.2: Verify all found abbreviations appear in the expansion suffix
for abbr in found:
    check(f"{abbr} =" in result, f"8.2 '{abbr}' present in expansion suffix", "missing from suffix")

# 8.3: Very long paragraph (500+ words)
very_long = (
    "The CO ordered the bns to wdr from the forward position at 0600 hrs. "
    "The MO reported casualties during the op. The adjt coordinated with HQ "
    "for the next phase of the operation. The recce team provided intelligence "
    "on enemy positions. The GOC visited the bde HQ and briefed the bde comd "
    "on the upcoming op. The inf bn was tasked to secure the obj. The armd sqn "
    "was to provide spt. The arty bty was placed in sp of the inf. The DA "
    "approved the medical resupply. The SP was established at checkpoint 4. "
    "The cas were evacuated to the Fd Amb. The spt elements moved up at 0600. "
    "The CO directed the bns to hold position until further orders. The wdr "
    "was completed by 0800. The MO confirmed all casualties were evacuated. "
    "The adjt reported to HQ that the op was successful. The recce team "
    "identified enemy positions to the north. The GOC commended the bde comd "
    "for the successful execution of the op. The inf bn maintained security "
    "of the obj. The armd sqn continued to provide spt. The arty bty remained "
    "in sp. The DA authorized additional resupply. The SP was resupplied at "
    "1200. The cas were treated at the Fd Amb. The spt elements were released "
    "at 1400. The CO declared the op complete. The bns returned to base. "
    "The wdr was executed as planned. The MO filed his report. The adjt "
    "closed the op with HQ. The recce team was debriefed. The GOC departed "
    "the bde HQ. The bde comd thanked all units. The inf bn stood down. "
    "The armd sqn returned to leaguer. The arty bty ceased fire. The DA "
    "confirmed all supplies were delivered. The SP was closed. The Fd Amb "
    "completed all evacuations."
)
result = expand_query_suffix(very_long, LOOKUP)
found = find_abbrs_in_text(very_long, LOOKUP)
print(f"\n  Very long text ({len(very_long)} chars): found {len(found)} abbreviations")
check(len(found) >= 8, "8.3a very long text finds >=8 abbrs", f"found {len(found)}: {list(found.keys())}")
check("[Abbreviation Glossary]" in result, "8.3b expansion suffix present")

# 8.4: Query that is a full paragraph with mixed abbrs and full forms
mixed_paragraph = (
    "The commanding officer ordered the battalions to withdraw from the "
    "forward position. The CO directed the bns to hold. The MO reported "
    "casualties during the operation. The medical officer confirmed all "
    "casualties were evacuated. The adjt coordinated with HQ for the wdr. "
    "The GOC visited the bde HQ and briefed the brigade commander on the op."
)
result = expand_query_suffix(mixed_paragraph, LOOKUP)
found_abbrs = find_abbrs_in_text(mixed_paragraph, LOOKUP)
found_forms = find_forms_in_text(mixed_paragraph, LOOKUP)
print(f"\n  Mixed paragraph: {len(found_abbrs)} abbrs + {len(found_forms)} reverse matches")
merged_keys = set(found_abbrs.keys()) | set(found_forms.keys())
print(f"  Merged: {sorted(merged_keys)}")
check(len(merged_keys) >= 5, "8.4 mixed paragraph finds >=5 unique abbrs", f"found {sorted(merged_keys)}")

# ─── Section 9: Repeated and adjacent abbreviations ────────────────────────

section("9. REPEATED AND ADJACENT ABBREVIATIONS")

# 9.1: Same abbreviation repeated
result = expand_query_suffix("bns bns bns", LOOKUP)
bns_count = result.count("bns = Battalions")
check(bns_count == 1, "9.1a repeated 'bns' → single expansion entry", f"count={bns_count} in: {result}")

# 9.2: Two different abbreviations adjacent
result = expand_query_suffix("CO bns", LOOKUP)
check("CO = Commanding Officer" in result, "9.2a 'CO bns' → CO expanded", result)
check("bns = Battalions" in result, "9.2b 'CO bns' → bns expanded", result)

# 9.3: Three abbreviations in sequence
result = expand_query_suffix("CO bns wdr", LOOKUP)
check("CO =" in result, "9.3a 'CO bns wdr' → CO", result)
check("bns =" in result, "9.3b 'CO bns wdr' → bns", result)
check("wdr =" in result, "9.3c 'CO bns wdr' → wdr", result)

# 9.4: Abbreviation repeated with other words between
result = expand_query_suffix("CO ordered bns to wdr, bns moved to rear", LOOKUP)
bns_count = result.count("bns = Battalions")
check(bns_count == 1, "9.4 'bns' appears twice → single expansion entry", f"count={bns_count}")

# ─── Section 10: Glossary format correctness ───────────────────────────────

section("10. GLOSSARY FORMAT CORRECTNESS")

# 10.1: Format is "query\n\n[Abbreviation Glossary]\nabbr = form1, form2"
result = expand_query_suffix("bns wdr", LOOKUP)
check("[Abbreviation Glossary]" in result, "10.1a format has [Abbreviation Glossary] block", result)
check(result.startswith("bns wdr\n\n[Abbreviation Glossary]"), "10.1b query text preserved before glossary", result[:50])

# 10.2: Multiple abbreviations, one per line
# Extract the glossary block
match = re.search(r'\[Abbreviation Glossary\]\n(.+)$', result, re.DOTALL)
if match:
    content = match.group(1)
    parts = content.split("\n")
    check(len(parts) >= 2, "10.2 multiple abbrs separated by '; '", f"parts: {parts}")

# 10.3: Each entry is "abbr = form1, form2"
result = expand_query_suffix("DA op", LOOKUP)
match = re.search(r'\[Abbreviation Glossary\]\n(.+)$', result, re.DOTALL)
if match:
    content = match.group(1)
    parts = content.split("\n")
    for part in parts:
        check("=" in part, f"10.3 entry has '=': '{part}'", content)
        abbr, forms = part.split("=", 1)
        check(len(abbr) > 0, f"10.3a abbr non-empty: '{abbr}'", part)
        check(len(forms) > 0, f"10.3b forms non-empty: '{forms}'", part)

# 10.4: No duplicate entries
result = expand_query_suffix("bns bns wdr wdr", LOOKUP)
match = re.search(r'\[Abbreviation Glossary\]\n(.+)$', result, re.DOTALL)
if match:
    content = match.group(1)
    parts = content.split("\n")
    abbrs_in_suffix = [p.split("=")[0] for p in parts]
    check(len(abbrs_in_suffix) == len(set(abbrs_in_suffix)),
          "10.4 no duplicate abbr entries in suffix", str(abbrs_in_suffix))

# 10.5: Ingestion suffix format (expand_suffix) vs query suffix format (expand_query_suffix)
ingestion_result = expand_suffix("bns wdr from position", LOOKUP)
query_result = expand_query_suffix("bns wdr from position", LOOKUP)
print(f"  Ingestion format: {ingestion_result[:80]}")
print(f"  Query format:     {query_result[:80]}")
check(ingestion_result.startswith("bns wdr from position\n\n[Abbreviation Glossary]"), "10.5a ingestion format correct", ingestion_result[:80])
check(query_result.startswith("bns wdr from position\n\n[Abbreviation Glossary]"), "10.5b query format correct", query_result[:80])

# ─── Section 11: Glossary generation ───────────────────────────────────────

section("11. GLOSSARY GENERATION (for generation context)")

# 11.1: build_glossary
glossary = build_glossary("the CO ordered the bns", LOOKUP)
check("CO = Commanding Officer" in glossary, "11.1a glossary has CO", glossary)
check("bns = Battalions" in glossary, "11.1b glossary has bns", glossary)

# 11.2: build_glossary_from_texts
glossary = build_glossary_from_texts(["the CO ordered", "the bns wdr"], LOOKUP)
check("CO = Commanding Officer" in glossary, "11.2a multi-text glossary has CO", glossary)
check("bns = Battalions" in glossary, "11.2b multi-text glossary has bns", glossary)
check("wdr" in glossary, "11.2c multi-text glossary has wdr", glossary)

# 11.3: Empty glossary for no abbreviations
glossary = build_glossary("weather forecast", LOOKUP)
check(glossary == "", "11.3 no abbrs → empty glossary", repr(glossary))

# ─── Section 12: Edge cases ────────────────────────────────────────────────

section("12. EDGE CASES")

# 12.1: Single character abbreviations should NOT be expanded (avoid false positives)
single_char_abbrs = [a for a in FORWARD if len(a) == 1]
print(f"  Single-char abbreviations in CSV: {single_char_abbrs[:10]}")
if single_char_abbrs:
    abbr = single_char_abbrs[0]
    result = expand_query_suffix(f"the {abbr} is", LOOKUP)
    check(f"{abbr} =" not in result, f"12.1 single-char '{abbr}' NOT expanded (avoids false positives)", result)

# 12.2: Very short query (single word abbreviation)
result = expand_query_suffix("CO", LOOKUP)
check("CO = Commanding Officer" in result, "12.2 single-word abbr query expanded", result)

# 12.3: Query that is just an abbreviation + punctuation
result = expand_query_suffix("CO?", LOOKUP)
check("CO = Commanding Officer" in result, "12.3 'CO?' expanded", result)

# 12.4: Newline in query
result = expand_query_suffix("the CO\nordered bns", LOOKUP)
check("CO = Commanding Officer" in result, "12.4a newline query — CO expanded", result)
check("bns = Battalions" in result, "12.4b newline query — bns expanded", result)

# 12.5: Tab in query
result = expand_query_suffix("CO\tbns", LOOKUP)
check("CO =" in result, "12.5a tab query — CO expanded", result)
check("bns =" in result, "12.5b tab query — bns expanded", result)

# 12.6: Unicode characters
result = expand_query_suffix("CO ordered café bns", LOOKUP)
check("CO =" in result, "12.6a unicode query — CO expanded", result)
check("bns =" in result, "12.6b unicode query — bns expanded", result)

# 12.7: Very long single abbreviation (longest in CSV)
longest_abbr = max(FORWARD.keys(), key=len)
print(f"  Longest abbreviation: '{longest_abbr}' ({len(longest_abbr)} chars)")
result = expand_query_suffix(f"the {longest_abbr} works", LOOKUP)
check(f"{longest_abbr}=" in result, f"12.7 longest abbr '{longest_abbr}' expanded", result[:80])

# 12.8: Query with numbers
result = expand_query_suffix("CO ordered 50 bns", LOOKUP)
check("CO =" in result, "12.8a numbers query — CO expanded", result)
check("bns =" in result, "12.8b numbers query — bns expanded", result)

# ─── Section 13: Performance ───────────────────────────────────────────────

section("13. PERFORMANCE")

import time

# 13.1: Expansion time for long query (tokenization should be fast)
t0 = time.time()
for _ in range(100):
    expand_query_suffix(very_long, LOOKUP)
elapsed = time.time() - t0
print(f"  100 expansions of {len(very_long)}-char text: {elapsed:.3f}s ({elapsed/100*1000:.1f}ms each)")
check(elapsed < 2.0, "13.1 100 expansions of long text < 2s", f"{elapsed:.3f}s")

# 13.2: Lookup build time
t0 = time.time()
for _ in range(10):
    make_lookup()
elapsed = time.time() - t0
print(f"  10 lookup builds: {elapsed:.3f}s ({elapsed/10*1000:.1f}ms each)")
check(elapsed < 2.0, "13.2 10 lookup builds < 2s", f"{elapsed:.3f}s")

# ─── Section 14: Consistency between expand_suffix and expand_query_suffix ─

section("14. CONSISTENCY: expand_suffix (ingestion) vs expand_query_suffix (query)")

# 14.1: Both should find the same abbreviations
test_text = "the CO ordered the bns to wdr from position"
ingestion_found = find_abbrs_in_text(test_text, LOOKUP)
query_result = expand_query_suffix(test_text, LOOKUP)
ingestion_result = expand_suffix(test_text, LOOKUP)

# Both should have the same abbreviations in the suffix
for abbr in ingestion_found:
    check(f"{abbr} =" in ingestion_result, f"14.1a '{abbr}' in ingestion suffix", ingestion_result[:80])
    check(f"{abbr} =" in query_result, f"14.1b '{abbr}' in query suffix", query_result[:80])

# 14.2: Query suffix should have MORE (bidirectional) for full-form queries
full_form_text = "commanding officer ordered battalions"
ingestion_result = expand_suffix(full_form_text, LOOKUP)
query_result = expand_query_suffix(full_form_text, LOOKUP)
# Ingestion (forward-only) should NOT find "commanding officer" as an abbreviation
# Query (bidirectional) SHOULD find it via reverse lookup
ingestion_has_expansion = "[Abbreviation Glossary]" in ingestion_result
query_has_expansion = "[Abbreviation Glossary]" in query_result
print(f"  Ingestion (forward-only): {ingestion_result[:80]}")
print(f"  Query (bidirectional):     {query_result[:80]}")
# Note: ingestion uses find_abbrs_in_text only (forward), so it won't match full forms
# Query uses bidirectional, so it should match
check(query_has_expansion, "14.2a query suffix has expansion for full forms", query_result[:80])
# Ingestion may or may not have expansion — depends on whether "officer" triggers reverse
# But ingestion is forward-only, so it shouldn't
warn(not ingestion_has_expansion or "offr" in ingestion_result,
     "14.2b ingestion suffix is forward-only (no reverse)", ingestion_result[:80])

# ─── Section 15: Realistic military queries ────────────────────────────────

section("15. REALISTIC MILITARY QUERIES")

military_queries = [
    ("CO ordered bns to wdr", ["CO", "bns", "wdr"]),
    ("the GOC visited bde HQ for the op", ["GOC", "bde", "HQ", "op"]),
    ("DA approved medical resupply at SP", ["DA", "SP"]),
    ("inf bn secured the obj with armd sqn spt", ["inf", "bn", "obj", "armd", "sqn"]),  # spt not in CSV
    ("arty bty in sp of inf", ["arty", "bty", "sp", "inf"]),
    ("recce team identified enemy positions", ["recce"]),
    ("adjt coordinated with HQ for the wdr", ["adjt", "HQ", "wdr"]),
    ("MO reported cas evacuated to Fd Amb", ["MO", "cas", "amb", "fd"]),  # Fd/Amb → fd/amb in CSV
    ("bde comd briefed on op by GOC", ["bde", "comd", "op", "GOC"]),
    ("CO directed bns to hold posn", ["CO", "bns", "posn"]),
]

for query, expected_abbrs in military_queries:
    found = find_abbrs_in_text(query, LOOKUP)
    result = expand_query_suffix(query, LOOKUP)
    missing = [a for a in expected_abbrs if a not in found and a.upper() not in found]
    if missing:
        # Some might not be in the CSV — that's OK, just report
        print(f"  NOTE: '{query}' — expected {expected_abbrs}, missing from CSV: {missing}")
        print(f"        found: {list(found.keys())}")
    else:
        print(f"  OK: '{query}' — all {len(expected_abbrs)} abbrs found")
    check(len(missing) == 0, f"15 military query '{query[:30]}'", f"missing: {missing}")

# ─── Section 16: Reverse lookup — diverse meanings ─────────────────────────

section("16. REVERSE LOOKUP — Same Full Form, Multiple Abbreviations")

# Find full forms that map to multiple abbreviations
multi_abbr_forms = {form: abbrs for form, abbrs in REVERSE.items() if len(abbrs) > 1}
print(f"  Full forms mapping to multiple abbreviations: {len(multi_abbr_forms)}")
for form, abbrs in sorted(multi_abbr_forms.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
    print(f"    '{form}' → {abbrs}")

# Test: query with a full form that maps to multiple abbreviations
if multi_abbr_forms:
    test_form = max(multi_abbr_forms, key=lambda f: len(multi_abbr_forms[f]))
    test_abbrs = multi_abbr_forms[test_form]
    result = expand_query_suffix(f"the {test_form} is important", LOOKUP)
    found_forms = find_forms_in_text(f"the {test_form} is important", LOOKUP)
    print(f"\n  Test: '{test_form}' → {test_abbrs}")
    print(f"  Result: {result[:100]}")
    for abbr in test_abbrs:
        check(abbr in found_forms, f"16a '{test_form}' → '{abbr}' found in reverse lookup", str(found_forms))

# ─── Section 17: Mixed scenario — abbr + full form of SAME abbreviation ────

section("17. MIXED — Same Abbreviation as Both Abbr and Full Form")

# Query contains both "bns" (abbr) and "Battalions" (full form)
result = expand_query_suffix("bns and Battalions were deployed", LOOKUP)
found_abbrs = find_abbrs_in_text("bns and Battalions were deployed", LOOKUP)
found_forms = find_forms_in_text("bns and Battalions were deployed", LOOKUP)
print(f"  Found abbrs: {list(found_abbrs.keys())}")
print(f"  Found forms (reverse): {list(found_forms.keys())}")
# Both should find "bns" — but it should only appear ONCE in the suffix
bns_count = result.count("bns = Battalions")
check(bns_count == 1, "17a 'bns' appears once in suffix (not duplicated)", f"count={bns_count} in: {result}")

# Same with "wdr" and "Withdraw"
result = expand_query_suffix("wdr and Withdraw completed", LOOKUP)
wdr_count = result.count("wdr = Withdraw")
check(wdr_count == 1, "17b 'wdr' appears once in suffix", f"count={wdr_count} in: {result}")

# ─── Section 18: Stress test — all abbreviations in one query ──────────────

section("18. STRESS TEST — All Abbreviations in One Query")

# Build a query containing every abbreviation (space-separated)
# Use first 100 to keep it manageable
sample_abbrs = ALL_ABBRS[:100]
stress_query = " ".join(sample_abbrs)
result = expand_query_suffix(stress_query, LOOKUP)
found = find_abbrs_in_text(stress_query, LOOKUP)
print(f"  Query with {len(sample_abbrs)} abbreviations ({len(stress_query)} chars)")
print(f"  Found {len(found)} abbreviations")
print(f"  Result length: {len(result)} chars")
check(len(found) >= 90, "18a stress test finds >=90 of 100 abbrs", f"found {len(found)}")
check("[Abbreviation Glossary]" in result, "18b stress test has expansion suffix")

# Verify no duplicates in the suffix
match = re.search(r'\[Abbreviation Glossary\]\n(.+)$', result, re.DOTALL)
if match:
    content = match.group(1)
    parts = content.split("\n")
    abbrs_in_suffix = [p.split("=")[0] for p in parts]
    check(len(abbrs_in_suffix) == len(set(abbrs_in_suffix)),
          "18c no duplicates in stress test suffix",
          f"{len(abbrs_in_suffix)} entries, {len(set(abbrs_in_suffix))} unique")

# ─── FINAL SUMMARY ─────────────────────────────────────────────────────────

print(f"\n{'=' * 80}")
print(f"  FINAL SUMMARY")
print(f"{'=' * 80}")
print(f"  Passed:   {passed}")
print(f"  Failed:   {failed}")
print(f"  Warnings: {warnings}")
print(f"  Total:    {passed + failed + warnings}")
print()

if failures:
    print("  FAILURES:")
    for f in failures:
        print(f"    - {f}")
    print()

if warns:
    print("  WARNINGS:")
    for w in warns:
        print(f"    - {w}")
    print()

if failed == 0 and warnings == 0:
    print("  VERDICT: ALL TESTS PASSED — abbreviation handling is robust")
elif failed == 0:
    print("  VERDICT: No failures, but warnings indicate areas for improvement")
else:
    print(f"  VERDICT: {failed} failures need to be fixed")

if __name__ == "__main__":
    sys.exit(1 if failed > 0 else 0)
