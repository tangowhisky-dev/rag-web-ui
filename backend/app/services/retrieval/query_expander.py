import re
from typing import Dict


def expand(query: str, abbreviations: Dict[str, str]) -> str:
    """Replace each abbreviation key with its expansion using word-boundary regex.
    Matching is case-insensitive; original casing of surrounding text is preserved.
    Longer keys are replaced first to avoid partial collisions."""
    if not abbreviations:
        return query
    for short in sorted(abbreviations, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(short) + r'\b', re.IGNORECASE)
        query = pattern.sub(abbreviations[short], query)
    return query


def load_org_abbreviations(org_id: int, db) -> Dict[str, str]:
    """Return {short: expansion} dict for the given org. Returns {} if org_id is None."""
    if org_id is None:
        return {}
    from app.models.organisation import OrgAbbreviation
    rows = db.query(OrgAbbreviation).filter(OrgAbbreviation.org_id == org_id).all()
    return {r.short: r.expansion for r in rows}
