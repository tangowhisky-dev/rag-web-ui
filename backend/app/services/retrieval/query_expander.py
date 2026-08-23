"""Legacy query expander — thin wrapper around abbreviation_service.

The expand() function is kept for backward compatibility but now delegates
to the abbreviation_service module.
"""
import re
from typing import Dict, List


def expand(query: str, abbreviations: Dict[str, List[str]]) -> str:
    """Append all expanded forms for abbreviations found in the query.

    Deprecated: use app.services.abbreviation_service.expand_query_suffix instead.
    """
    if not abbreviations:
        return query
    for short in sorted(abbreviations, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(short) + r'\b', re.IGNORECASE)
        if pattern.search(query):
            forms = abbreviations[short]
            if isinstance(forms, list):
                query += " " + " ".join(forms)
            else:
                query += " " + str(forms)
    return query


def load_org_abbreviations(org_id: int, db) -> Dict[str, List[str]]:
    """Return {short: [form1, form2, ...]} dict for the given org.

    Deprecated: use app.services.abbreviation_service.build_lookup instead.
    """
    if org_id is None:
        return {}
    from app.services.abbreviation_service import build_lookup
    lookup = build_lookup(db, org_id)
    return lookup.forward
