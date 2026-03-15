"""Normalization and deduplication utilities for ESG entities and relations."""

from __future__ import annotations

import re
from typing import List, Set

from .models import Entity, Relation


# ---------------------------------------------------------------------------
# Canonical name normalization
# ---------------------------------------------------------------------------

# Common irregular plurals → singular mappings relevant to ESG discourse.
_IRREGULAR_PLURALS: dict[str, str] = {
    "emissions": "emission",
    "resources": "resource",
    "policies": "policy",
    "subsidies": "subsidy",
    "companies": "company",
    "authorities": "authority",
    "communities": "community",
    "liabilities": "liability",
    "responsibilities": "responsibility",
    "initiatives": "initiative",
    "activities": "activity",
    "disclosures": "disclosure",
    "practices": "practice",
    "processes": "process",
    "risks": "risk",
    "rights": "right",
    "standards": "standard",
    "targets": "target",
    "criteria": "criterion",
    "indices": "index",
}


def normalize_canonical_name(name: str) -> str:
    """Return a normalized canonical name.

    Steps applied:
    1. Strip leading/trailing whitespace.
    2. Collapse internal whitespace runs to a single space.
    3. Convert to lowercase.
    4. Apply irregular-plural lookup.
    5. Strip a trailing 's' for simple regular plurals (only when the
       resulting stem is at least 4 characters, to avoid false positives
       on words like "gas" or "bus").

    The LLM is expected to deliver singular forms; this function acts as
    a lightweight safety net.
    """
    if not name:
        return name

    # Step 1–3: clean whitespace and lowercase
    normalized = re.sub(r"\s+", " ", name.strip()).lower()

    # Step 4: irregular plurals
    if normalized in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[normalized]

    # Step 5: simple regular plural ending in 's'
    # Only attempt de-pluralisation when the *last word* of the phrase is
    # longer than 3 characters.  This prevents "gas" → "ga" and similar
    # false positives while still handling "risks" → "risk" etc.
    last_word = normalized.rsplit(" ", 1)[-1]
    if normalized.endswith("s") and len(last_word) > 3:
        # Check exclusions before stripping:
        # - words ending in 'ss' (e.g. 'business', 'access')
        # - words ending in 'ous' (e.g. 'hazardous')
        if not normalized.endswith("ss") and not normalized.endswith("ous"):
            return normalized[:-1]

    return normalized


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate_entities(entities: List[Entity]) -> List[Entity]:
    """Remove duplicate entities, keeping the first occurrence per canonical name.

    Two entities are considered duplicates when their normalized canonical
    names are identical.
    """
    seen: Set[str] = set()
    unique: List[Entity] = []
    for entity in entities:
        key = normalize_canonical_name(entity.canonical)
        if key not in seen:
            seen.add(key)
            # Return entity with normalized canonical name
            unique.append(entity.model_copy(update={"canonical": key}))
    return unique


# ---------------------------------------------------------------------------
# Relation cleaning
# ---------------------------------------------------------------------------


def clean_relations(
    relations: List[Relation],
    canonical_names: Set[str],
) -> List[Relation]:
    """Remove relations whose source or target is not in *canonical_names*.

    Also deduplicates relations that share the same (source, target, relation)
    triple, keeping the one with the highest confidence score.
    """
    # Filter relations that reference known entities
    valid: List[Relation] = [
        r
        for r in relations
        if normalize_canonical_name(r.source) in canonical_names
        and normalize_canonical_name(r.target) in canonical_names
    ]

    # Deduplicate by (source, target, relation) — keep highest confidence
    best: dict[tuple[str, str, str], Relation] = {}
    for rel in valid:
        key = (
            normalize_canonical_name(rel.source),
            normalize_canonical_name(rel.target),
            rel.relation,
        )
        if key not in best or rel.confidence > best[key].confidence:
            best[key] = rel.model_copy(
                update={
                    "source": normalize_canonical_name(rel.source),
                    "target": normalize_canonical_name(rel.target),
                }
            )

    return list(best.values())
