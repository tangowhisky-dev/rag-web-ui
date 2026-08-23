#!/usr/bin/env python3
"""
Deterministic abbreviation expander for RAG pipelines.

No LLM needed. Pure CSV lookup + string matching.

Two use cases:
1. INGESTION TIME: Expand abbreviations in document chunks
   - "The CO ordered the bns to move" → "The CO (Commanding Officer) ordered the bns (Battalions) to move"
   - All possible expanded forms are appended so semantic search can find them

2. QUERY TIME: Expand abbreviations and full forms in user queries
   - User queries "increased" → also search for "inc"
   - User queries "DA" → also search for "Daily Allowance", "Defence Attache", etc.

Usage:
    from abbr_expander import AbbrExpander

    expander = AbbrExpander("abbreviations_enhanced.csv")

    # Ingestion: expand abbreviations in a text chunk
    expanded = expander.expand_ingestion("The CO ordered bns to wdr")

    # Query: expand a query string
    expanded = expander.expand_query("battalions withdrew from position")
"""
import csv
import re
from collections import defaultdict
from pathlib import Path


class AbbrExpander:
    """Deterministic abbreviation expander for RAG pipelines."""

    def __init__(self, csv_path: str):
        """Load the abbreviation CSV and build lookup indexes.

        Builds three indexes:
        - forward_map: abbreviation -> [expanded_form, ...]
        - reverse_map: expanded_form -> [abbreviation, ...]
        - all_abbreviations: sorted list of abbreviations (longest first for matching)
        """
        self.forward_map: dict[str, list[str]] = defaultdict(list)
        self.reverse_map: dict[str, list[str]] = defaultdict(list)

        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                abbr = row["abbreviation"].strip()
                form = row["expanded_form"].strip()
                if abbr and form:
                    self.forward_map[abbr].append(form)
                    self.reverse_map[form.lower()].append(abbr)

        # Sort abbreviations by length (longest first) so "Gp Capt" matches before "Gp"
        self.all_abbreviations = sorted(
            self.forward_map.keys(), key=len, reverse=True
        )

        # Build a single regex pattern for all abbreviations
        # Escape special regex chars, match word boundaries
        escaped = [re.escape(a) for a in self.all_abbreviations]
        # Use word boundary on both sides; allow the abbreviation to contain spaces
        self.abbr_pattern = re.compile(
            r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE
        )

        # Build reverse lookup: lowercase expanded_form -> list of abbreviations
        # Also build a set of all expanded forms for query expansion
        self.all_expanded_forms = set(self.reverse_map.keys())

    # ------------------------------------------------------------------
    # INGESTION TIME EXPANSION
    # ------------------------------------------------------------------
    def expand_ingestion(self, text: str, mode: str = "append") -> str:
        """Expand abbreviations in a document chunk during ingestion.

        Args:
            text: The original text chunk.
            mode: How to add expansions:
                "append" - Add all expanded forms in parentheses after the abbreviation.
                           "The CO ordered" → "The CO (Commanding Officer) ordered"
                "suffix" - Append all expansions at the end of the text.
                           "The CO ordered" → "The CO ordered [Expansions: CO=Commanding Officer]"
                "replace" - Replace abbreviation with all forms joined by space.
                           "The CO ordered" → "The Commanding Officer ordered"

        Returns:
            Text with abbreviations expanded.

        Example:
            >>> expander.expand_ingestion("The CO ordered bns to wdr")
            'The CO (Commanding Officer) ordered bns (Battalions) to wdr (Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew)'
        """
        if mode == "replace":
            return self.abbr_pattern.sub(self._replace_match, text)

        if mode == "suffix":
            return self._expand_suffix(text)

        # Default: "append" mode
        return self.abbr_pattern.sub(self._append_match, text)

    def _append_match(self, m: re.Match) -> str:
        """Replace abbreviation with itself + all expanded forms in parentheses."""
        matched = m.group(1)
        # Find the actual abbreviation key (case-insensitive)
        forms = self._lookup_forward(matched)
        if not forms:
            return matched
        forms_str = " ".join(forms)
        return f"{matched} ({forms_str})"

    def _replace_match(self, m: re.Match) -> str:
        """Replace abbreviation with all expanded forms."""
        matched = m.group(1)
        forms = self._lookup_forward(matched)
        if not forms:
            return matched
        return " ".join(forms)

    def _expand_suffix(self, text: str) -> str:
        """Append all expansion info at the end of the text."""
        expansions = []
        for m in self.abbr_pattern.finditer(text):
            matched = m.group(1)
            forms = self._lookup_forward(matched)
            if forms:
                expansions.append(f"{matched}={', '.join(forms)}")
        if expansions:
            return text + " [Expansions: " + "; ".join(expansions) + "]"
        return text

    def _lookup_forward(self, abbr: str) -> list[str]:
        """Look up expanded forms for an abbreviation (case-insensitive)."""
        # Try exact match first
        if abbr in self.forward_map:
            return self.forward_map[abbr]
        # Try case-insensitive
        for key, forms in self.forward_map.items():
            if key.lower() == abbr.lower():
                return forms
        return []

    # ------------------------------------------------------------------
    # QUERY TIME EXPANSION
    # ------------------------------------------------------------------
    def expand_query(self, query: str) -> str:
        """Expand a user query for better semantic search.

        Two things happen:
        1. If query contains an abbreviation (e.g. "bns"), append all expanded forms.
        2. If query contains a full form (e.g. "battalions"), append the abbreviation.

        This ensures both directions of matching work.

        Args:
            query: The user's search query.

        Returns:
            Expanded query string.

        Example:
            >>> expander.expand_query("bns withdrew from position")
            'bns Battalions withdrew Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew from position'
        """
        result = query

        # Step 1: Find abbreviations in query, append their expanded forms
        abbreviations_found = set()
        for m in self.abbr_pattern.finditer(query):
            abbreviations_found.add(m.group(1))

        for abbr in abbreviations_found:
            forms = self._lookup_forward(abbr)
            if forms:
                result += " " + " ".join(forms)

        # Step 2: Find full forms in query, append their abbreviations
        query_lower = query.lower()
        for form_lower, abbrs in self.reverse_map.items():
            # Use word boundary matching for the full form
            pattern = re.compile(
                r"\b" + re.escape(form_lower) + r"\b", re.IGNORECASE
            )
            if pattern.search(query_lower):
                for abbr in abbrs:
                    if abbr not in abbreviations_found:
                        result += " " + abbr

        return result

    # ------------------------------------------------------------------
    # ANALYSIS HELPERS
    # ------------------------------------------------------------------
    def get_ambiguous_abbreviations(self, min_meanings: int = 2) -> dict[str, list[str]]:
        """Return abbreviations that have multiple completely different meanings.

        This is useful for understanding which abbreviations are ambiguous.

        Args:
            min_meanings: Minimum number of different meanings to qualify.

        Returns:
            Dict of abbreviation -> list of expanded forms.
        """
        result = {}
        for abbr, forms in self.forward_map.items():
            if len(forms) >= min_meanings:
                result[abbr] = forms
        return result

    def get_derivative_abbreviations(self) -> dict[str, list[str]]:
        """Return abbreviations where all forms share the same root word.

        These are verb conjugations, noun forms, adjective forms, etc.
        """
        result = {}
        for abbr, forms in self.forward_map.items():
            if len(forms) < 2:
                continue
            # Check if all forms share the same first 4 characters (lowercase)
            roots = set()
            for f in forms:
                first_word = f.lower().split()[0].split("(")[0].strip()
                roots.add(first_word[:4])
            if len(roots) == 1:
                result[abbr] = forms
        return result


# ----------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "abbreviations_enhanced.csv")
    expander = AbbrExpander(csv_path)

    print("=" * 70)
    print("DETERMINISTIC ABBREVIATION EXPANDER - DEMO")
    print("=" * 70)

    # --- Case 1: Derivative forms (same root) ---
    print("\n--- Case 1: Derivative forms (same root) ---")
    print("Example: 'inc' expands to multiple forms of 'increase'\n")

    test_text = "The enemy inc their fire on our bns"
    print(f"Original:  {test_text!r}")
    print(f"Append:    {expander.expand_ingestion(test_text, 'append')!r}")
    print(f"Replace:   {expander.expand_ingestion(test_text, 'replace')!r}")
    print(f"Suffix:    {expander.expand_ingestion(test_text, 'suffix')!r}")

    print()
    test_query = "increased casualties in battalions"
    print(f"Query:     {test_query!r}")
    print(f"Expanded:  {expander.expand_query(test_query)!r}")

    # --- Case 2: Completely different meanings ---
    print("\n\n--- Case 2: Completely different meanings ---")
    print("Example: 'DA' has 5 unrelated meanings\n")

    da_forms = expander.forward_map["DA"]
    print(f"DA expands to: {da_forms}")

    test_text = "The DA approved the plan"
    print(f"\nOriginal:  {test_text!r}")
    print(f"Append:    {expander.expand_ingestion(test_text, 'append')!r}")

    test_query = "dispersal area"
    print(f"\nQuery:     {test_query!r}")
    print(f"Expanded:  {expander.expand_query(test_query)!r}")

    # --- Case 3: Mixed (some derivative, some different) ---
    print("\n\n--- Case 3: Mixed (partially related) ---")
    print("Example: 'dep' has both Depart* and Depot*\n")

    dep_forms = expander.forward_map["dep"]
    print(f"dep expands to: {dep_forms}")

    test_text = "The dep was located at the dep"
    print(f"\nOriginal:  {test_text!r}")
    print(f"Append:    {expander.expand_ingestion(test_text, 'append')!r}")

    # --- Statistics ---
    print("\n\n--- Statistics ---")
    ambiguous = expander.get_ambiguous_abbreviations(min_meanings=2)
    derivatives = expander.get_derivative_abbreviations()
    print(f"Total abbreviations: {len(expander.forward_map)}")
    print(f"Abbreviations with multiple forms: {len(ambiguous)}")
    print(f"  - Derivative forms (same root): {len(derivatives)}")
    print(f"  - Different meanings: {len(ambiguous) - len(derivatives)}")

    print("\n--- Most ambiguous abbreviations ---")
    sorted_ambiguous = sorted(ambiguous.items(), key=lambda x: len(x[1]), reverse=True)
    for abbr, forms in sorted_ambiguous[:10]:
        print(f"  {abbr} ({len(forms)} forms): {forms[:3]}{'...' if len(forms) > 3 else ''}")
