"""Tests for esg_kg.normalize."""

import pytest

from esg_kg.models import Entity, Relation
from esg_kg.normalize import (
    clean_relations,
    deduplicate_entities,
    normalize_canonical_name,
)


# ---------------------------------------------------------------------------
# normalize_canonical_name
# ---------------------------------------------------------------------------


class TestNormalizeCanonicalName:
    def test_empty_string_returns_empty(self):
        assert normalize_canonical_name("") == ""

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_canonical_name("  carbon  ") == "carbon"

    def test_collapses_internal_whitespace(self):
        assert normalize_canonical_name("carbon  dioxide") == "carbon dioxide"

    def test_lowercases(self):
        assert normalize_canonical_name("Carbon Dioxide") == "carbon dioxide"

    def test_irregular_plural_emissions(self):
        assert normalize_canonical_name("emissions") == "emission"

    def test_irregular_plural_resources(self):
        assert normalize_canonical_name("resources") == "resource"

    def test_irregular_plural_policies(self):
        assert normalize_canonical_name("policies") == "policy"

    def test_irregular_plural_subsidies(self):
        assert normalize_canonical_name("subsidies") == "subsidy"

    def test_irregular_plural_companies(self):
        assert normalize_canonical_name("companies") == "company"

    def test_irregular_plural_criteria(self):
        assert normalize_canonical_name("criteria") == "criterion"

    def test_irregular_plural_indices(self):
        assert normalize_canonical_name("indices") == "index"

    def test_irregular_plural_disclosures(self):
        assert normalize_canonical_name("disclosures") == "disclosure"

    def test_irregular_plural_practices(self):
        assert normalize_canonical_name("practices") == "practice"

    def test_regular_plural_stripped(self):
        # "risks" → "risk" (last word 5 chars, ends with s, not ss or ous)
        assert normalize_canonical_name("risks") == "risk"

    def test_regular_plural_multi_word(self):
        # "carbon credits" → "carbon credit"
        assert normalize_canonical_name("carbon credits") == "carbon credit"

    def test_no_strip_when_last_word_too_short(self):
        # "gas" has only 3 chars after potential strip — should not strip
        assert normalize_canonical_name("gas") == "gas"

    def test_no_strip_double_s_ending(self):
        # "business" ends with "ss" — should NOT be de-pluralised
        assert normalize_canonical_name("business") == "business"

    def test_no_strip_ous_ending(self):
        # "hazardous" ends with "ous" — should NOT be de-pluralised
        assert normalize_canonical_name("hazardous") == "hazardous"

    def test_already_singular(self):
        assert normalize_canonical_name("emission") == "emission"

    def test_whitespace_tabs_normalized(self):
        assert normalize_canonical_name("carbon\t dioxide") == "carbon dioxide"

    def test_newline_in_name_normalized(self):
        assert normalize_canonical_name("carbon\ndioxide") == "carbon dioxide"


# ---------------------------------------------------------------------------
# deduplicate_entities
# ---------------------------------------------------------------------------


def _make_entity(canonical: str, surface: str = "surface", esg: str = "E") -> Entity:
    return Entity(surface=surface, canonical=canonical, type="emission", esg=esg)


class TestDeduplicateEntities:
    def test_empty_list(self):
        assert deduplicate_entities([]) == []

    def test_no_duplicates_unchanged(self):
        entities = [
            _make_entity("carbon dioxide"),
            _make_entity("methane"),
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 2

    def test_duplicate_canonical_names_deduplicated(self):
        entities = [
            _make_entity("carbon dioxide", surface="Carbon Dioxide"),
            _make_entity("carbon dioxide", surface="CO2"),
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1
        # First occurrence kept
        assert result[0].surface == "Carbon Dioxide"

    def test_canonical_name_normalized_on_output(self):
        entities = [_make_entity("Emissions")]
        result = deduplicate_entities(entities)
        # "Emissions" → normalize → "emission" (irregular plural)
        assert result[0].canonical == "emission"

    def test_irregular_plural_deduplication(self):
        # "risks" and "risk" should be treated as the same canonical name
        entities = [
            _make_entity("risks", surface="Risks"),
            _make_entity("risk", surface="Risk"),
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1

    def test_case_insensitive_deduplication(self):
        entities = [
            _make_entity("Carbon Dioxide"),
            _make_entity("carbon dioxide"),
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1

    def test_order_preserved(self):
        entities = [
            _make_entity("methane"),
            _make_entity("carbon dioxide"),
            _make_entity("water"),
        ]
        result = deduplicate_entities(entities)
        assert [e.canonical for e in result] == ["methane", "carbon dioxide", "water"]


# ---------------------------------------------------------------------------
# clean_relations
# ---------------------------------------------------------------------------


def _make_relation(
    source: str,
    target: str,
    relation: str = "CAUSES",
    confidence: float = 0.8,
) -> Relation:
    return Relation(
        source=source,
        target=target,
        relation=relation,
        measurement_type="qualitative",
        polarity="negative",
        evidence="test evidence",
        confidence=confidence,
    )


class TestCleanRelations:
    def test_empty_relations_empty_result(self):
        assert clean_relations([], set()) == []

    def test_valid_relations_kept(self):
        canonical = {"carbon emission", "climate change"}
        relations = [_make_relation("carbon emission", "climate change")]
        result = clean_relations(relations, canonical)
        assert len(result) == 1

    def test_unknown_source_filtered_out(self):
        canonical = {"climate change"}
        relations = [_make_relation("unknown entity", "climate change")]
        result = clean_relations(relations, canonical)
        assert result == []

    def test_unknown_target_filtered_out(self):
        canonical = {"carbon emission"}
        relations = [_make_relation("carbon emission", "unknown entity")]
        result = clean_relations(relations, canonical)
        assert result == []

    def test_both_unknown_filtered_out(self):
        canonical = {"some entity"}
        relations = [_make_relation("unknown a", "unknown b")]
        result = clean_relations(relations, canonical)
        assert result == []

    def test_deduplication_keeps_highest_confidence(self):
        canonical = {"carbon emission", "climate change"}
        relations = [
            _make_relation("carbon emission", "climate change", confidence=0.6),
            _make_relation("carbon emission", "climate change", confidence=0.9),
        ]
        result = clean_relations(relations, canonical)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_different_relation_types_kept_separately(self):
        canonical = {"carbon emission", "climate change"}
        relations = [
            _make_relation("carbon emission", "climate change", relation="CAUSES"),
            _make_relation("carbon emission", "climate change", relation="PROMOTES"),
        ]
        result = clean_relations(relations, canonical)
        assert len(result) == 2

    def test_relation_source_target_normalized(self):
        # "Carbon Emission" and "carbon emission" should match after normalization
        canonical = {"carbon emission", "climate change"}
        relations = [_make_relation("Carbon Emission", "climate change")]
        result = clean_relations(relations, canonical)
        assert len(result) == 1
        assert result[0].source == "carbon emission"

    def test_irregular_plural_source_normalized(self):
        canonical = {"emission", "climate change"}
        relations = [_make_relation("emissions", "climate change")]
        result = clean_relations(relations, canonical)
        assert len(result) == 1
        assert result[0].source == "emission"

    def test_mixed_valid_and_invalid_relations(self):
        canonical = {"carbon emission", "climate change"}
        relations = [
            _make_relation("carbon emission", "climate change"),
            _make_relation("unknown", "climate change"),
            _make_relation("carbon emission", "unknown"),
        ]
        result = clean_relations(relations, canonical)
        assert len(result) == 1
