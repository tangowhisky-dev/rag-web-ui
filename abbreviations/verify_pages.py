#!/usr/bin/env python3
"""Verify each page of PDF against CSV output to find missing entries."""
import pdfplumber
import csv
import re
import json
from parse_abbr import (get_threshold, group_words_into_rows, is_noise_line,
                        is_note_line, is_subcategory_header, has_unclosed_paren,
                        clean_abbrev, clean_word, parse_word_field,
                        split_multi_variant, expand_derivative, is_derivative_suffix,
                        extract_ammunition_tables, reconstruct_table_entries,
                        fix_ammunition_word_split, is_section_header)

PDF_PATH = "abvns with explanation.pdf"
CSV_PATH = "abbreviations.csv"


def load_csv():
    """Load CSV entries into a set of (abbrev, expanded_form, category) tuples."""
    with open(CSV_PATH, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows


def extract_page_entries(pdf, pi, category, table_name=None):
    """Extract all entries from a single page, returning list of (word, abbrev) pairs."""
    page = pdf.pages[pi]
    words = page.extract_words()
    if not words:
        return [], None

    threshold = get_threshold(page, words)
    if threshold is None:
        return [], None

    rows = group_words_into_rows(words, threshold)

    # For ammunition tables, use the same logic as the main parser
    if table_name:
        raw_rows = []
        for r in rows:
            word = r['word'].strip()
            abbrev = r['abbrev'].strip()
            combined = (word + ' ' + abbrev).strip()

            # Detect TABLE headers
            table_match = re.match(r'^TABLE\s+([IVX]+)', word)
            if table_match:
                continue

            if is_noise_line(word, abbrev):
                continue
            if 'Nature' in word and 'Type' in word and 'Ammunition' in word:
                continue
            if word == 'Nature and Type Ammunition':
                continue
            if word == 'Nature and Type of Ammunition':
                continue
            if abbrev == 'Abbreviation' and not word:
                continue
            if combined == 'RESTRICTED':
                continue
            if re.match(r'^1\.\s*\d+$', combined):
                continue

            # Mark section headers
            is_header = is_section_header(word, abbrev)

            # Fix word/abbrev split for ammunition tables
            if word and not is_header:
                word, abbrev = fix_ammunition_word_split(word, abbrev)

            raw_rows.append({
                'page': pi,
                'row': r['row'],
                'top': r['top'],
                'word': word,
                'abbrev': abbrev,
                'category': category,
                'table': table_name,
                'is_section_header': is_header,
            })
        return raw_rows, threshold

    # Standard extraction for non-table pages
    raw_rows = []
    seen_header = False
    for r in rows:
        word = r['word'].strip()
        abbrev = r['abbrev'].strip()

        if is_noise_line(word, abbrev):
            if 'Word(s) In Full' in word or 'Word(s) In Full' in abbrev:
                seen_header = True
            continue

        if not seen_header:
            if is_note_line(word, abbrev):
                continue
            if word == 'Word(s) In Full' or 'Word(s) In Full' in (word + ' ' + abbrev):
                seen_header = True
                continue
            if word in ('GENERAL ABBREVIATIONS', 'SPECIALIZED ABBREVIATIONS',
                        'ABBREVIATIONS LAND SERVICE AMMUNITION',
                        'GENERAL TERMS AND ABBREVIATIONS'):
                continue
            if abbrev in ('Section 1', 'Section 2') and word == 'To':
                continue
            if word == 'Annex A' or word == 'Annex B':
                continue
            continue

        raw_rows.append({
            'page': pi,
            'row': r['row'],
            'top': r['top'],
            'word': word,
            'abbrev': abbrev,
            'category': category,
            'table': table_name,
        })

    return raw_rows, threshold


def reconstruct_page_entries(raw_rows):
    """Reconstruct entries from a single page's raw rows."""
    # Check if this is a table page (has is_section_header field)
    if raw_rows and 'is_section_header' in raw_rows[0]:
        return reconstruct_table_entries(raw_rows)

    entries = []
    cur_word = ""
    cur_abbrev = ""
    cur_subcat = ""

    def save():
        nonlocal cur_word, cur_abbrev
        w = cur_word.strip()
        a = cur_abbrev.strip()
        if w and a:
            entries.append((w, a, cur_subcat))
        cur_word = ""
        cur_abbrev = ""

    for i, r in enumerate(raw_rows):
        word = r['word'].strip()
        abbrev = r['abbrev'].strip()

        if r.get('table'):
            cur_subcat = r['table']

        if is_subcategory_header(word, abbrev):
            save()
            cur_subcat = word
            continue

        if is_note_line(word, abbrev):
            if abbrev and (cur_word or cur_abbrev):
                cur_abbrev += ' ' + abbrev
            continue

        if word and abbrev:
            is_continuation = False
            if cur_word:
                if re.search(r'[,;(\[]\s*$', cur_word):
                    is_continuation = True
                elif has_unclosed_paren(cur_word):
                    is_continuation = True
                elif re.match(r'^\([^)]+\)$', abbrev) and not cur_abbrev.endswith(')'):
                    is_continuation = True

            if is_continuation:
                cur_word += ' ' + word
                cur_abbrev += ' ' + abbrev
            else:
                save()
                cur_word = word
                cur_abbrev = abbrev
        elif word and not abbrev:
            is_continuation = False
            if cur_word:
                if re.search(r'[,;(\[]\s*$', cur_word):
                    is_continuation = True
                elif has_unclosed_paren(cur_word):
                    is_continuation = True
                elif word.startswith('(') and cur_abbrev:
                    is_continuation = True
                elif cur_abbrev and not word.endswith(';') and not word.startswith('('):
                    cur_has_balanced = cur_word.count('(') > 0 and cur_word.count('(') == cur_word.count(')')
                    if not cur_has_balanced:
                        next_has_both = False
                        for nr in raw_rows[i+1:i+3]:
                            if nr['word'].strip() and nr['abbrev'].strip():
                                next_has_both = True
                                break
                            if nr['word'].strip() or nr['abbrev'].strip():
                                break
                        if next_has_both:
                            is_continuation = True

            if is_continuation:
                cur_word += ' ' + word
            elif cur_abbrev:
                save()
                cur_word = word
                cur_abbrev = ""
            else:
                if cur_word:
                    cur_word += ' ' + word
                else:
                    cur_word = word
        elif abbrev and not word:
            if cur_word or cur_abbrev:
                cur_abbrev += ' ' + abbrev
            else:
                cur_abbrev = abbrev

    save()
    return entries


def expand_entry(word, abbrev):
    """Expand an entry into list of (abbrev, expanded_form) pairs."""
    abbrev = clean_abbrev(abbrev)
    if not abbrev or len(abbrev) > 60:
        return []
    if len(abbrev.split()) > 8 and not any(c.isupper() for c in abbrev[:3]):
        return []

    variants = split_multi_variant(word, abbrev)
    result = []
    for var_word, var_abbrev in variants:
        forms = parse_word_field(var_word)
        if not forms:
            forms = [var_word] if var_word else []
        for form in forms:
            if not form or len(form) > 120:
                continue
            if form.startswith(('a.', 'b.', 'c.')):
                continue
            if var_abbrev == 'Abbreviation' or form.startswith('Nature and Type'):
                continue
            result.append((var_abbrev, form))
    return result


def verify_page(pdf, pi, category, csv_rows, table_name=None):
    """Verify a single page. Returns list of missing entries."""
    raw_rows, threshold = extract_page_entries(pdf, pi, category, table_name)
    if not raw_rows:
        return [], []

    entries = reconstruct_page_entries(raw_rows)

    # Build set of ALL CSV entries (regardless of category) for cross-checking
    csv_all = set()
    for r in csv_rows:
        csv_all.add((r['abbreviation'], r['expanded_form']))

    # Build map of abbreviation -> list of expanded forms for fuzzy matching
    csv_by_abbr = {}
    for r in csv_rows:
        key = r['abbreviation']
        if key not in csv_by_abbr:
            csv_by_abbr[key] = []
        csv_by_abbr[key].append(r['expanded_form'])

    # Expand page entries and check against CSV
    page_expanded = []
    missing = []
    for word, abbrev, subcat in entries:
        expanded = expand_entry(word, abbrev)
        for abbr, form in expanded:
            page_expanded.append((abbr, form, word, abbrev))
            # Check if in any category (true missing)
            if (abbr, form) not in csv_all:
                # Check if it's a prefix of a CSV entry (multi-line entry spanning pages)
                csv_forms = csv_by_abbr.get(abbr, [])
                is_prefix = any(csv_f.startswith(form) for csv_f in csv_forms)
                if not is_prefix:
                    # Check if abbreviation is a prefix of any CSV abbreviation
                    # (continuation merged into full abbreviation)
                    is_abbr_prefix = False
                    for csv_a, csv_fs in csv_by_abbr.items():
                        if csv_a.startswith(abbr) and any(csv_f.startswith(form) for csv_f in csv_fs):
                            is_abbr_prefix = True
                            break
                    if not is_abbr_prefix:
                        # Check if form is a suffix/substring of any CSV entry
                        # (continuation text that was merged into parent entry)
                        is_substring = False
                        for csv_a, csv_fs in csv_by_abbr.items():
                            for csv_f in csv_fs:
                                if form in csv_f or csv_f.endswith(form):
                                    is_substring = True
                                    break
                            if is_substring:
                                break
                        if not is_substring:
                            # Check if abbreviation is a comma-separated multi-variant
                            # that was split by post-processing
                            is_multi_variant = False
                            if ',' in abbr:
                                parts = [p.strip() for p in abbr.split(',') if p.strip()]
                                # Check if all parts are in CSV, trying with Q prefix
                                all_found = True
                                for p in parts:
                                    if p in csv_by_abbr:
                                        continue
                                    # Try prepending "Q " for Quartermaster variants
                                    if f'Q {p}' in csv_by_abbr:
                                        continue
                                    all_found = False
                                    break
                                if all_found:
                                    is_multi_variant = True
                            if not is_multi_variant:
                                missing.append((abbr, form, word, abbrev))

    return page_expanded, missing


def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        csv_rows = load_csv()

        all_missing = []
        all_page_entries = []

        # Section 1 Annex A: General (pages 6-48)
        print("=" * 80)
        print("SECTION 1, ANNEX A - GENERAL ABBREVIATIONS (pages 6-48)")
        print("=" * 80)
        for pi in range(6, 49):
            page_entries, missing = verify_page(pdf, pi, "general", csv_rows)
            if page_entries or missing:
                print(f"\nPage {pi}: {len(page_entries)} entries, {len(missing)} missing")
                for abbr, form, raw_word, raw_abbrev in missing:
                    print(f"  MISSING: {abbr!r} -> {form!r} (from {raw_word!r} -> {raw_abbrev!r})")
                    all_missing.append(('general', pi, abbr, form, raw_word, raw_abbrev))
                all_page_entries.extend([(pi, a, f) for a, f, _, _ in page_entries])

        # Section 1 Annex B: Specialized (pages 49-68)
        print("\n" + "=" * 80)
        print("SECTION 1, ANNEX B - SPECIALIZED ABBREVIATIONS (pages 49-68)")
        print("=" * 80)
        for pi in range(49, 69):
            page_entries, missing = verify_page(pdf, pi, "specialized", csv_rows)
            if page_entries or missing:
                print(f"\nPage {pi}: {len(page_entries)} entries, {len(missing)} missing")
                for abbr, form, raw_word, raw_abbrev in missing:
                    print(f"  MISSING: {abbr!r} -> {form!r} (from {raw_word!r} -> {raw_abbrev!r})")
                    all_missing.append(('specialized', pi, abbr, form, raw_word, raw_abbrev))
                all_page_entries.extend([(pi, a, f) for a, f, _, _ in page_entries])

        # Section 2 Annex A: Ammunition general (pages 70-76)
        print("\n" + "=" * 80)
        print("SECTION 2, ANNEX A - AMMUNITION GENERAL (pages 70-76)")
        print("=" * 80)
        for pi in range(70, 77):
            page_entries, missing = verify_page(pdf, pi, "ammunition", csv_rows)
            if page_entries or missing:
                print(f"\nPage {pi}: {len(page_entries)} entries, {len(missing)} missing")
                for abbr, form, raw_word, raw_abbrev in missing:
                    print(f"  MISSING: {abbr!r} -> {form!r} (from {raw_word!r} -> {raw_abbrev!r})")
                    all_missing.append(('ammunition', pi, abbr, form, raw_word, raw_abbrev))
                all_page_entries.extend([(pi, a, f) for a, f, _, _ in page_entries])

        # Section 2 Tables I-X (pages 77-91)
        print("\n" + "=" * 80)
        print("SECTION 2, TABLES I-X (pages 77-91)")
        print("=" * 80)

        # Detect table names per page
        table_pages = {}
        current_table = None
        for pi in range(77, 92):
            page = pdf.pages[pi]
            txt = page.extract_text() or ""
            for line in txt.split('\n'):
                m = re.match(r'^TABLE\s+([IVX]+)\s*[–—-]\s*(.+)', line.strip())
                if m:
                    current_table = line.strip()
                    break
            table_pages[pi] = current_table

        for pi in range(77, 92):
            table_name = table_pages.get(pi)
            page_entries, missing = verify_page(pdf, pi, "ammunition", csv_rows, table_name=table_name)
            if page_entries or missing:
                print(f"\nPage {pi} [{table_name}]: {len(page_entries)} entries, {len(missing)} missing")
                for abbr, form, raw_word, raw_abbrev in missing:
                    print(f"  MISSING: {abbr!r} -> {form!r} (from {raw_word!r} -> {raw_abbrev!r})")
                    all_missing.append((table_name or 'ammunition', pi, abbr, form, raw_word, raw_abbrev))
                all_page_entries.extend([(pi, a, f) for a, f, _, _ in page_entries])

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total page entries: {len(all_page_entries)}")
        print(f"Total missing entries: {len(all_missing)}")

        if all_missing:
            print("\nAll missing entries:")
            for cat, pi, abbr, form, raw_word, raw_abbrev in all_missing:
                print(f"  p{pi} [{cat}] {abbr!r} -> {form!r}")
                print(f"    raw: {raw_word!r} -> {raw_abbrev!r}")


if __name__ == "__main__":
    main()
