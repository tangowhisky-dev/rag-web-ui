#!/usr/bin/env python3
"""Parse abbreviations from PDF using word positions."""
import pdfplumber
import re
import json
import csv

PDF_PATH = "abvns with explanation.pdf"

# Section page ranges (0-indexed)
SECTIONS = {
    "general": list(range(6, 49)),       # Annex A Section 1
    "specialized": list(range(49, 69)),  # Annex B Section 1
    "ammunition": list(range(70, 77)),   # Annex A Section 2 general terms
}


def get_threshold(page, words):
    """Detect abbreviation column x-threshold.

    Primary: use 'Abbreviation' header position with small offset.
    Fallback: find the largest gap in x0 values between word and abbrev columns.
    """
    # Primary: use header position
    for w in words:
        if w['text'] == 'Abbreviation' and w['top'] < 260:
            # Use header_x - 1 to include abbrev values that may start
            # slightly left of the header position
            return w['x0'] - 1
    # Fallback: gap detection
    xs = sorted(set(round(w['x0']) for w in words if w['top'] > 200))
    gaps = []
    for i in range(len(xs) - 1):
        if 200 <= xs[i] <= 450:
            gap = xs[i + 1] - xs[i]
            if gap > 10:
                gaps.append((gap, xs[i], xs[i + 1]))
    gaps.sort(reverse=True)
    if gaps:
        g, lo, hi = gaps[0]
        return (lo + hi) / 2
    return None


def group_words_into_rows(words, threshold):
    """Group words into rows by vertical position. Split into word/abbrev columns.

    Two-pass approach:
    1. Cluster word-column words (x0 < threshold) to establish row anchors.
    2. Assign each abbrev-column word to the nearest row anchor.
    """
    word_words = sorted([w for w in words if w['x0'] < threshold], key=lambda w: (w['top'], w['x0']))
    abbrev_words_all = sorted([w for w in words if w['x0'] >= threshold], key=lambda w: (w['top'], w['x0']))

    # Cluster word-column words by top position (tolerance 5)
    word_clusters = []
    for w in word_words:
        if word_clusters and abs(w['top'] - word_clusters[-1]['anchor']) <= 5:
            word_clusters[-1]['words'].append(w)
        else:
            word_clusters.append({'anchor': w['top'], 'words': [w]})

    # If no word-column words, fall back to clustering all words
    if not word_clusters:
        all_sorted = sorted(words, key=lambda w: (w['top'], w['x0']))
        clusters = []
        for w in all_sorted:
            if clusters and abs(w['top'] - clusters[-1]['anchor']) <= 5:
                clusters[-1]['words'].append(w)
            else:
                clusters.append({'anchor': w['top'], 'words': [w]})
        result = []
        for ci, cluster in enumerate(clusters):
            ww = sorted([w for w in cluster['words'] if w['x0'] < threshold], key=lambda w: (w['top'], w['x0']))
            aw = sorted([w for w in cluster['words'] if w['x0'] >= threshold], key=lambda w: (w['top'], w['x0']))
            result.append({
                'row': ci,
                'top': cluster['anchor'],
                'word': ' '.join(w['text'] for w in ww).strip(),
                'abbrev': ' '.join(w['text'] for w in aw).strip(),
            })
        return result

    # Create row data from word clusters
    row_data = {i: {'word': c['words'], 'abbrev': [], 'top': c['anchor']}
                for i, c in enumerate(word_clusters)}
    row_anchors = [c['anchor'] for c in word_clusters]

    # Assign abbrev words to nearest preceding-or-current row anchor.
    # This ensures continuation abbrevs (which fall between word rows)
    # are assigned to the entry they belong to, not the next one.
    for aw in abbrev_words_all:
        aw_top = aw['top']
        # Find nearest anchor at or before the abbrev's top (with small tolerance)
        best_idx = None
        best_dist = float('inf')
        for i, anchor in enumerate(row_anchors):
            if anchor <= aw_top + 3:
                dist = aw_top - anchor
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        # If no preceding anchor, use nearest following
        if best_idx is None:
            best_idx = min(range(len(row_anchors)), key=lambda i: abs(aw_top - row_anchors[i]))
        row_data[best_idx]['abbrev'].append(aw)

    result = []
    for i in sorted(row_data.keys()):
        r = row_data[i]
        ww = sorted(r['word'], key=lambda w: (w['top'], w['x0']))
        aw = sorted(r['abbrev'], key=lambda w: (w['top'], w['x0']))
        result.append({
            'row': i,
            'top': r['top'],
            'word': ' '.join(w['text'] for w in ww).strip(),
            'abbrev': ' '.join(w['text'] for w in aw).strip(),
        })
    return result


def is_noise_line(word, abbrev):
    """Check if a line is page noise."""
    t = (word + ' ' + abbrev).strip()
    if not t:
        return True
    if t == 'RESTRICTED':
        return True
    if re.match(r'^1\.\s*\d+$', t):
        return True
    # Page number patterns like "1. 1 86", "1. 181", etc.
    if re.match(r'^1\.\s*\d+\s+\d+$', t):
        return True
    if re.match(r'^\d+\s*\d+\s*\d*$', t) and len(t) <= 10:
        return True
    if 'Word(s) In Full' in t and 'Abbreviation' in t:
        return True
    if word.strip() == 'Word(s) In Full':
        return True
    if abbrev.strip() == 'Abbreviation' and not word.strip():
        return True
    # Alphabet dividers: single capital letter
    if re.match(r'^[A-Z]$', word.strip()) and not abbrev.strip():
        return True
    return False


def is_note_line(word, abbrev):
    """Detect note/explanation lines that aren't real entries."""
    t = word.strip()
    if t.startswith('Notes:-') or t.startswith('Notes:'):
        return True
    if re.match(r'^[a-z]\.\s', t):  # "a. This list..." note items
        return True
    if re.match(r'^\d+\.\s', t) and not abbrev.strip():  # numbered notes
        return True
    # "This may be qualified e.g." and similar explanatory text
    if re.match(r'^This may be qualified', t):
        return True
    if re.match(r'^e\.g\.', t) and not abbrev.strip():
        return True
    # Note text about course precedence
    if 'over foreign courses' in t or 'take precedence' in t:
        return True
    if 'first letter of the country' in t:
        return True
    if 'Word "gsc" in small letters' in t or 'Word \u201cgsc\u201d in small letters' in t:
        return True
    return False


def is_subcategory_header(word, abbrev):
    """Detect subcategory headers in Annex B (all-caps, no abbrev)."""
    t = word.strip()
    if not t:
        return False
    # All-caps headers like "HIGHER DEF ORG STAFF AND APPTS"
    # Also allow parentheses for headers like "APPOINTMENTS (LESS THOSE..."
    if abbrev.strip():
        # Special case: "ORG)" is part of header, not abbrev
        if re.match(r'^[A-Z]+\)', abbrev.strip()) and re.match(r'^[A-Z]', t):
            return True
        return False
    # All-caps text (allow parentheses, spaces, &, /, -, commas)
    if re.match(r'^[A-Z][A-Z\s&/,\-()]+$', t) and len(t) > 3:
        return True
    return False


def is_section_header(word, abbrev):
    """Detect section headers in ammunition tables (e.g. 'Cartridges QF 2 Pr')."""
    if not word or abbrev.strip():
        return False
    # Section headers start with known patterns
    patterns = [
        r'^Cartridges\s',
        r'^Shell\s',
        r'^Bomb\s',
        r'^Projectile\s',
        r'^Bombs\s',
        r'^Rockets?\s',
        r'^Grenades?\s',
        r'^Mines?\s',
        r'^Booster\s',
        r'^Fuze\s',
        r'^Kit\s',
        r'^Lighter\s',
        r'^Igniter\s',
        r'^Napalm\s',
    ]
    for p in patterns:
        if re.match(p, word):
            return True
    return False


def fix_ammunition_word_split(word, abbrev):
    """Fix word/abbrev split where parts of abbreviation ended up in word column.

    Handles multiple cases:
    1. Leading numbers: word='HE Shell Fuze No* 2' abbrev='PR HE * FUZ'
       -> word='HE Shell Fuze No*' abbrev='2 PR HE * FUZ'
    2. Leading caliber: word='Cartridges QF 37 MM 37MM' abbrev='HE (CHE)'
       -> word='Cartridges QF 37 MM' abbrev='37MM HE (CHE)'
    3. Trailing abbrev text: word='Type* (Chinese) FUZ' abbrev='(CHE)'
       -> word='Type* (Chinese)' abbrev='FUZ (CHE)'
    """
    if not word or not abbrev:
        return word, abbrev

    # Case 1: word ends with a number that should be start of abbreviation
    # Only move if the abbreviation starts with a unit pattern (PR, MM, IN, CM, etc.)
    # that would follow a caliber number
    m = re.search(r'^(.*?)(\s+(\d+(?:\.\d+)?))$', word)
    if m and not re.match(r'^\d', abbrev):
        prefix = m.group(1).strip()
        number = m.group(3)
        # Only move if abbrev starts with a known unit pattern
        if re.match(r'^(PR|MM|IN|CM|MM/|MM\b)', abbrev):
            return prefix, number + ' ' + abbrev

    # Case 2: word ends with a caliber pattern like "37MM", "75MM", "122MM"
    m_cal = re.search(r'^(.*?\S)\s+(\d+(?:\.\d+)?\s*MM)$', word, re.IGNORECASE)
    if m_cal and not re.match(r'^\d', abbrev):
        prefix = m_cal.group(1).strip()
        caliber = m_cal.group(2).strip()
        return prefix, caliber + ' ' + abbrev

    # Case 3: word ends with all-caps abbrev text (like "FUZ", "DEC")
    # that should be part of the abbreviation
    # Only move very short all-caps text (<= 4 chars) that is clearly abbrev text
    # Exclude common unit designations that appear in word column
    EXCLUDE_CAPS = {'MM', 'IN', 'CM', 'KG', 'GM', 'ML', 'MM/', 'BESA', 'BERNO'}
    m2 = re.search(r'^(.*?\S)\s+([A-Z][A-Z*]+)$', word)
    if m2:
        prefix = m2.group(1).strip()
        caps_text = m2.group(2).strip()
        if len(caps_text) <= 4 and re.match(r'^[A-Z][A-Z*]+$', caps_text) and caps_text not in EXCLUDE_CAPS:
            return prefix, caps_text + ' ' + abbrev

    return word, abbrev


def extract_ammunition_tables(pdf, pages):
    """Extract entries from ammunition Tables I-X.

    Ammunition tables have a different structure:
    - Section headers (e.g. "Cartridges QF 2 Pr") with no abbreviation
    - Entry rows with both word and abbreviation
    - Continuation rows where word or abbreviation wraps to next line
    """
    raw_rows = []
    current_table = ""

    for pi in pages:
        page = pdf.pages[pi]
        words = page.extract_words()
        if not words:
            continue
        threshold = get_threshold(page, words)
        if threshold is None:
            continue

        rows = group_words_into_rows(words, threshold)

        for r in rows:
            word = r['word'].strip()
            abbrev = r['abbrev'].strip()
            combined = (word + ' ' + abbrev).strip()

            # Detect TABLE headers
            table_match = re.match(r'^TABLE\s+([IVX]+)', word)
            if table_match:
                current_table = combined
                continue

            # Skip noise
            if is_noise_line(word, abbrev):
                continue

            # Skip column headers
            if 'Nature' in word and 'Type' in word and 'Ammunition' in word:
                continue
            if word == 'Nature and Type Ammunition':
                continue
            if word == 'Nature and Type of Ammunition':
                continue
            if abbrev == 'Abbreviation' and not word:
                continue

            # Skip page-level noise
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
                'category': 'ammunition',
                'table': current_table,
                'is_section_header': is_header,
            })

    return raw_rows


def reconstruct_table_entries(raw_rows):
    """Reconstruct ammunition table entries, handling section headers and continuations."""
    entries = []
    cur_word = ""
    cur_abbrev = ""
    cur_table = ""

    def save():
        nonlocal cur_word, cur_abbrev
        w = cur_word.strip()
        a = cur_abbrev.strip()
        if w and a:
            entries.append((w, a, cur_table))
        cur_word = ""
        cur_abbrev = ""

    for i, r in enumerate(raw_rows):
        word = r['word'].strip()
        abbrev = r['abbrev'].strip()

        # Save current entry at page boundaries, UNLESS the current row
        # is a short continuation (e.g. "(Bofors)" / "(BOFORS)")
        if i > 0 and r.get('page') != raw_rows[i-1].get('page'):
            is_short_cont = (
                (word and len(word) <= 15 and re.match(r'^\(', word)) or
                (abbrev and len(abbrev) <= 15 and re.match(r'^\(', abbrev))
            )
            if not is_short_cont:
                save()

        if r.get('table'):
            cur_table = r['table']

        # Section headers: save current entry, skip header
        if r.get('is_section_header'):
            save()
            continue

        if word and abbrev:
            # Both present - could be new entry or continuation
            # Check if this is a continuation of the previous entry
            is_continuation = False
            if cur_abbrev:
                # Continuation if current abbrev is a short parenthetical qualifier
                # (like "(CHE)", "(BRDT)", "(BELG)")
                if len(abbrev) <= 10 and re.match(r'^\([^)]+\)$', abbrev):
                    is_continuation = True

            if is_continuation:
                cur_word += ' ' + word
                cur_abbrev += ' ' + abbrev
            else:
                save()
                cur_word = word
                cur_abbrev = abbrev
        elif word and not abbrev:
            # Word continuation
            # Don't move all-caps words to abbrev - they could be legitimate
            # word-column values like "APCBC", "APDS"
            if cur_abbrev:
                cur_word += ' ' + word
            else:
                if cur_word:
                    cur_word += ' ' + word
                else:
                    cur_word = word
        elif abbrev and not word:
            # Abbrev continuation
            if cur_word or cur_abbrev:
                cur_abbrev += ' ' + abbrev
            else:
                cur_abbrev = abbrev
        # both empty: skip

    save()
    return entries


def extract_section(pdf, pages, category):
    """Extract entries from a section."""
    raw_rows = []
    threshold = None
    seen_header = False

    for pi in pages:
        page = pdf.pages[pi]
        words = page.extract_words()
        if not words:
            continue
        new_thresh = get_threshold(page, words)
        if new_thresh:
            threshold = new_thresh
            seen_header = False  # reset for new page
        if threshold is None:
            continue

        rows = group_words_into_rows(words, threshold)

        for r in rows:
            if is_noise_line(r['word'], r['abbrev']):
                # Mark header seen
                if 'Word(s) In Full' in r['word'] or 'Word(s) In Full' in r['abbrev']:
                    seen_header = True
                continue
            if not seen_header:
                # Skip notes before first header on page
                if is_note_line(r['word'], r['abbrev']):
                    continue
                # Check for header
                if r['word'].strip() == 'Word(s) In Full' or 'Word(s) In Full' in (r['word'] + ' ' + r['abbrev']):
                    seen_header = True
                    continue
                # Skip section title lines
                if r['word'] in ('GENERAL ABBREVIATIONS', 'SPECIALIZED ABBREVIATIONS',
                                 'ABBREVIATIONS LAND SERVICE AMMUNITION',
                                 'GENERAL TERMS AND ABBREVIATIONS'):
                    continue
                if r['abbrev'] in ('Section 1', 'Section 2') and r['word'] == 'To':
                    continue
                if r['word'] == 'Annex A' or r['word'] == 'Annex B':
                    continue
                continue  # skip everything before header

            raw_rows.append({
                'page': pi,
                'row': r['row'],
                'top': r['top'],
                'word': r['word'],
                'abbrev': r['abbrev'],
                'category': category,
            })

    return raw_rows


def has_unclosed_paren(s):
    """Check if string has unclosed parenthesis."""
    return s.count('(') > s.count(')')


def reconstruct_entries(raw_rows):
    """Reconstruct multi-line entries into (word, abbrev, subcategory) tuples."""
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

        # Save current entry at page boundaries
        if i > 0 and r.get('page') != raw_rows[i-1].get('page'):
            save()

        # Use table field as subcategory if available (ammunition tables)
        if r.get('table'):
            cur_subcat = r['table']

        # Check for subcategory header (Annex B)
        if is_subcategory_header(word, abbrev):
            # Check if this is a continuation of a multi-line header
            # (previous header has unclosed parenthesis)
            if cur_subcat and has_unclosed_paren(cur_subcat):
                cur_subcat += ' ' + word
                if abbrev:
                    cur_subcat += ' ' + abbrev
                continue
            save()
            cur_subcat = word
            if abbrev and re.match(r'^[A-Z]+\)', abbrev):
                # Abbrev is part of header (e.g. "ORG)")
                cur_subcat += ' ' + abbrev
            continue

        # Handle note lines: if the word is note text but has an abbrev,
        # append the abbrev to the current entry (the abbrev is a continuation)
        if is_note_line(word, abbrev):
            if abbrev and (cur_word or cur_abbrev):
                cur_abbrev += ' ' + abbrev
            continue

        if word and abbrev:
            # Both present - could be new entry or continuation
            is_continuation = False
            if cur_word:
                # Continuation if current word ends with comma/semicolon/open-bracket
                if re.search(r'[,;(\[]\s*$', cur_word):
                    is_continuation = True
                # Continuation if current word has unclosed parenthesis
                elif has_unclosed_paren(cur_word):
                    is_continuation = True
                # Continuation if abbrev is just a parenthetical qualifier
                # and current abbrev doesn't end with closing paren
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
                # Parenthetical continuation (e.g. "(Budget & Maintenance)")
                elif word.startswith('(') and cur_abbrev:
                    is_continuation = True
                # Look-ahead: if cur_abbrev is set, word doesn't end with ";",
                # cur_word has no balanced parens (entry seems incomplete),
                # and the next non-empty row has both word+abbrev, treat as continuation
                elif cur_abbrev and not word.endswith(';') and not word.startswith('('):
                    # Only use look-ahead if current entry seems incomplete
                    # (no balanced parentheses in cur_word)
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
                # Current entry is complete, start new (waiting for abbrev)
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
        # both empty: skip

    save()
    return entries


def clean_word(w):
    """Clean up word field."""
    # Remove note text that got merged
    w = re.sub(r'\s*This may be qualified e\.g\.\s*.*$', '', w).strip()
    # Remove leading section numbers (e.g. "1 Ministry of Defence" -> "Ministry of Defence")
    w = re.sub(r'^\d+\s+', '', w).strip()
    # Collapse multiple spaces
    w = re.sub(r'\s+', ' ', w).strip()
    return w


def clean_abbrev(a):
    """Clean up abbrev field."""
    # Fix PDF font encoding issue: cid:31 = 'C' (only occurrence in document)
    a = a.replace('(cid:31)', 'C')
    # Fix spacing after commas
    a = re.sub(r',(?!\s)', ', ', a)
    # Fix missing commas between variants: "Q (Log) Q (Qtg)" -> "Q (Log), Q (Qtg)"
    a = re.sub(r'(\))\s+([A-Z]\s*\()', r'\1, \2', a)
    a = re.sub(r'\s+', ' ', a).strip()
    return a


def split_multi_variant(word, abbrev):
    """Split entries with multiple abbreviation variants.

    E.g. "Quartermaster General (Branch), (Matters), (Staff)" ->
         "Q (Br), Q (Matters), Q (Staff)"
    becomes:
         ("Quartermaster General (Branch)", "Q (Br)")
         ("Quartermaster General (Matters)", "Q (Matters)")
         ("Quartermaster General (Staff)", "Q (Staff)")
    """
    # Check if abbrev has multiple variants separated by commas
    abbrev_parts = [p.strip() for p in abbrev.split(',') if p.strip()]
    if len(abbrev_parts) <= 1:
        return [(word, abbrev)]

    # Check if all abbrev parts share a common prefix (like "Q", "A", "G")
    # Extract the base letter from each part
    bases = []
    for p in abbrev_parts:
        m = re.match(r'^([A-Z])\s*\(', p)
        if m:
            bases.append(m.group(1))
        else:
            return [(word, abbrev)]  # not a multi-variant pattern

    if len(set(bases)) != 1:
        return [(word, abbrev)]

    base = bases[0]

    # Parse the word to extract the root and variants
    # Pattern: "Root (variant1), (variant2), (variant3)"
    word_parts = [p.strip() for p in word.split(',') if p.strip()]
    if len(word_parts) != len(abbrev_parts):
        return [(word, abbrev)]

    # First word part should be "Root (variant1)"
    # Extract root and first variant
    m = re.match(r'^(.+?)\s*\(([^)]+)\)\s*$', word_parts[0])
    if not m:
        return [(word, abbrev)]

    root = m.group(1).strip()

    # Extract variant from each word part
    result = []
    for wp, ap in zip(word_parts, abbrev_parts):
        vm = re.search(r'\(([^)]+)\)\s*$', wp)
        if vm:
            variant = vm.group(1)
            result.append((f"{root} ({variant})", ap))
        else:
            # Try to extract variant from abbrev part (e.g. "A (Staff)")
            am = re.search(r'\(([^)]+)\)\s*$', ap)
            if am:
                variant = am.group(1)
                result.append((f"{root} ({variant})", ap))
            else:
                result.append((wp, ap))

    return result


# Derivative suffix expansion
DERIVATIVE_SUFFIXES = {
    's', 'd', 'ed', 'ing', 'ly', 'ment', 'al', 'ion', 'ation', 'er', 'y',
    'or', 'led', 'ure', 'ity', 'ify', 'ified', 'ic', 'ary', 'ant', 'ance',
    'ally', 'ter', 'ted', 'red', 'r', 'ous', 'n', 'men', 'man', 'ling',
    'ler', 'ized', 'iveness', 'ively', 'ive', 'ised', 'ise', 'ile',
    'ification', 'ened', 'ence', 'en', 'ature', 'ative', 'ability', 'ting',
    'lers', 'ation', 'atory', 'able', 'ations',
}

# Suffixes with special e-dropping rules
E_DROP_SUFFIXES = {'d', 'ed', 'ing', 'ion', 'ation', 'ative', 'ative'}


def expand_derivative(root, suffix):
    """Expand a root + parenthetical suffix into the full word."""
    root = root.strip()
    suffix = suffix.lower()
    if suffix in ('d', 'ed'):
        if root.endswith('e'):
            return root + 'd'
        else:
            return root + 'ed'
    elif suffix == 'ing':
        if root.endswith('e'):
            return root[:-1] + 'ing'
        else:
            return root + 'ing'
    elif suffix == 'ion':
        if root.endswith('e'):
            return root[:-1] + 'ion'
        else:
            return root + 'ion'
    elif suffix == 'ation':
        if root.endswith('e'):
            return root[:-1] + 'ation'
        else:
            return root + 'ation'
    else:
        return root + suffix


def is_derivative_suffix(s):
    """Check if a parenthetical content is a derivative suffix."""
    s = s.lower()
    return s in DERIVATIVE_SUFFIXES


def parse_word_field(word):
    """Parse a word field into list of expanded forms.

    Handles:
    - Semicolon-separated distinct terms
    - Comma-separated derivative suffixes in parentheses
    """
    word = clean_word(word)
    if not word:
        return []

    # Split on semicolons to get distinct terms
    terms = [t.strip() for t in word.split(';') if t.strip()]

    expanded_forms = []

    for term in terms:
        # Parse comma-separated parts
        parts = [p.strip() for p in term.split(',') if p.strip()]
        if not parts:
            continue

        # The first part is the root
        root = parts[0]

        # Check if remaining parts are derivative suffixes
        derivatives = []
        non_derivative_parts = []

        for p in parts[1:]:
            # Check if it's a parenthetical suffix
            m = re.match(r'^\(([^)]+)\)$', p)
            if m and is_derivative_suffix(m.group(1)):
                derivatives.append(m.group(1))
            else:
                non_derivative_parts.append(p)

        if derivatives:
            # Root is a word, expand derivatives
            expanded_forms.append(root)
            for deriv in derivatives:
                expanded_forms.append(expand_derivative(root, deriv))
        else:
            # No derivatives - join non-derivative parts back with commas
            if non_derivative_parts:
                expanded_forms.append(root + ', ' + ', '.join(non_derivative_parts))
            else:
                expanded_forms.append(root)

    return expanded_forms


def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        all_entries = []

        for category, pages in SECTIONS.items():
            print(f"\n=== {category} (pages {pages[0]}-{pages[-1]}) ===")
            raw = extract_section(pdf, pages, category)
            print(f"  Raw rows: {len(raw)}")
            entries = reconstruct_entries(raw)
            print(f"  Reconstructed entries: {len(entries)}")

            for word, abbrev, subcat in entries[:5]:
                print(f"    {word!r} -> {abbrev!r} [{subcat}]")
            if len(entries) > 10:
                print(f"    ...")
            for word, abbrev, subcat in entries[-3:]:
                print(f"    {word!r} -> {abbrev!r} [{subcat}]")

            all_entries.extend([(category, word, abbrev, subcat) for word, abbrev, subcat in entries])

        # Extract ammunition tables (I-X)
        print(f"\n=== ammunition tables (pages 77-91) ===")
        table_raw = extract_ammunition_tables(pdf, list(range(77, 92)))
        print(f"  Raw rows: {len(table_raw)}")
        # Reconstruct with table-specific reconstruction
        table_entries = reconstruct_table_entries(table_raw)
        print(f"  Reconstructed entries: {len(table_entries)}")
        for word, abbrev, subcat in table_entries[:5]:
            print(f"    {word!r} -> {abbrev!r} [{subcat}]")
        if len(table_entries) > 10:
            print(f"    ...")
        for word, abbrev, subcat in table_entries[-3:]:
            print(f"    {word!r} -> {abbrev!r} [{subcat}]")
        all_entries.extend([("ammunition", word, abbrev, subcat) for word, abbrev, subcat in table_entries])

        # Save raw reconstructed entries
        with open("raw_entries.json", "w") as f:
            json.dump(all_entries, f, indent=2, ensure_ascii=False)

        print(f"\nTotal raw entries: {len(all_entries)}")

        # Now expand into final CSV rows
        csv_rows = []
        for category, word, abbrev, subcat in all_entries:
            abbrev = clean_abbrev(abbrev)

            # Skip entries that are clearly note/explanation text
            if not abbrev or len(abbrev) > 60:
                continue
            # Skip if abbreviation looks like a sentence (note text leaked)
            if len(abbrev.split()) > 8 and not any(c.isupper() for c in abbrev[:3]):
                continue

            # Try to split multi-variant abbreviations
            variants = split_multi_variant(word, abbrev)

            for var_word, var_abbrev in variants:
                forms = parse_word_field(var_word)
                if not forms:
                    forms = [var_word] if var_word else []

                for form in forms:
                    # Skip empty or note-like forms
                    if not form or len(form) > 120:
                        continue
                    if form.startswith(('a.', 'b.', 'c.')):
                        continue
                    # Skip leaked column headers
                    if var_abbrev == 'Abbreviation' or form.startswith('Nature and Type'):
                        continue
                    csv_rows.append({
                        'abbreviation': var_abbrev,
                        'expanded_form': form,
                        'category': subcat if subcat else category,
                    })

        print(f"Total CSV rows (before dedup): {len(csv_rows)}")

        # Post-processing fixes
        # 1. Fix General Staff and Quartermaster General multi-variant entries
        fixed_rows = []
        for r in csv_rows:
            if r['abbreviation'] == 'G (Br), G (Matters), G (Staff), G (Int)':
                fixed_rows.append({'abbreviation': 'G (Br)', 'expanded_form': 'General Staff (Branch)', 'category': 'general'})
                fixed_rows.append({'abbreviation': 'G (Matters)', 'expanded_form': 'General Staff (Matters)', 'category': 'general'})
                fixed_rows.append({'abbreviation': 'G (Staff)', 'expanded_form': 'General Staff (Staff)', 'category': 'general'})
                fixed_rows.append({'abbreviation': 'G (Int)', 'expanded_form': 'General Staff (Intelligence)', 'category': 'general'})
            elif r['abbreviation'] == 'G (Ops)' and r['expanded_form'] == 'Operations':
                fixed_rows.append({'abbreviation': 'G (Ops)', 'expanded_form': 'General Staff (Operations)', 'category': 'general'})
            # Skip raw Quartermaster General fragment entries (handled by post-processing)
            elif r['abbreviation'] == 'Q (Qtg)' and r['expanded_form'] == '(Quartering)':
                continue
            elif 'Quartermaster General' in r['expanded_form'] and ',' in r['abbreviation'] and 'Q (' in r['abbreviation']:
                # Split Quartermaster General multi-variant entry
                # Hardcode the known variants since the raw entry is garbled
                q_variants = [
                    ('Q (Br)', 'Quartermaster General (Branch)'),
                    ('Q (Matters)', 'Quartermaster General (Matters)'),
                    ('Q (Staff)', 'Quartermaster General (Staff)'),
                    ('Q (Log)', 'Quartermaster General (Logistics)'),
                    ('Q (Qtg)', 'Quartermaster General (Quartering)'),
                ]
                for abbr, form in q_variants:
                    fixed_rows.append({
                        'abbreviation': abbr,
                        'expanded_form': form,
                        'category': 'general'
                    })
            else:
                fixed_rows.append(r)
        csv_rows = fixed_rows

        # Remove exact duplicates
        seen = set()
        deduped = []
        for r in csv_rows:
            key = (r['abbreviation'], r['expanded_form'], r['category'])
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        print(f"Total CSV rows (after dedup): {len(deduped)}")

        # Sort by abbreviation (case-insensitive), then expanded form
        deduped.sort(key=lambda r: (r['abbreviation'].lower(), r['expanded_form'].lower()))

        # Write CSV
        with open("abbreviations.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['abbreviation', 'expanded_form', 'category'])
            writer.writeheader()
            for r in deduped:
                writer.writerow(r)

        print(f"\nWrote {len(deduped)} rows to abbreviations.csv")


if __name__ == "__main__":
    main()
