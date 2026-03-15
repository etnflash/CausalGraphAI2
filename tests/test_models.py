"""Tests for esg_kg.models."""

import pytest
from pydantic import ValidationError

from esg_kg.models import Entity, ExtractionResult, Relation


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class TestEntity:
    def _valid_entity(self, **overrides):
        data = {
            "surface": "Scope 1 emissions",
            "canonical": "scope 1 emission",
            "type": "emission",
            "esg": "E",
        }
        data.update(overrides)
        return data

    def test_valid_entity(self):
        entity = Entity(**self._valid_entity())
        assert entity.surface == "Scope 1 emissions"
        assert entity.canonical == "scope 1 emission"
        assert entity.type == "emission"
        assert entity.esg == "E"

    def test_all_esg_values(self):
        for esg in ("E", "S", "G"):
            entity = Entity(**self._valid_entity(esg=esg))
            assert entity.esg == esg

    def test_invalid_esg_raises(self):
        with pytest.raises(ValidationError):
            Entity(**self._valid_entity(esg="X"))

    def test_missing_required_field_raises(self):
        data = self._valid_entity()
        del data["surface"]
        with pytest.raises(ValidationError):
            Entity(**data)


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------


class TestRelation:
    def _valid_relation(self, **overrides):
        data = {
            "source": "scope 1 emission",
            "target": "paris agreement",
            "relation": "COMPLIES_WITH",
            "measurement_type": "quantitative",
            "polarity": "positive",
            "evidence": "cut Scope 1 emissions by 40%",
            "confidence": 0.95,
            "value": 40.0,
            "unit": "%",
        }
        data.update(overrides)
        return data

    def test_valid_relation(self):
        rel = Relation(**self._valid_relation())
        assert rel.source == "scope 1 emission"
        assert rel.relation == "COMPLIES_WITH"
        assert rel.confidence == 0.95
        assert rel.value == 40.0
        assert rel.unit == "%"

    def test_all_relation_types(self):
        valid_types = [
            "CAUSES",
            "PROMOTES",
            "INHIBITS",
            "ASSOCIATED_WITH",
            "PART_OF",
            "IS_A",
            "INCREASES",
            "DECREASES",
            "VIOLATES",
            "COMPLIES_WITH",
        ]
        for rtype in valid_types:
            rel = Relation(**self._valid_relation(relation=rtype))
            assert rel.relation == rtype

    def test_invalid_relation_type_raises(self):
        with pytest.raises(ValidationError):
            Relation(**self._valid_relation(relation="UNKNOWN"))

    def test_invalid_measurement_type_raises(self):
        with pytest.raises(ValidationError):
            Relation(**self._valid_relation(measurement_type="mixed"))

    def test_invalid_polarity_raises(self):
        with pytest.raises(ValidationError):
            Relation(**self._valid_relation(polarity="bad"))

    def test_confidence_boundary_values(self):
        for conf in (0.0, 0.5, 1.0):
            rel = Relation(**self._valid_relation(confidence=conf))
            assert rel.confidence == conf

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            Relation(**self._valid_relation(confidence=-0.01))

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            Relation(**self._valid_relation(confidence=1.01))

    def test_optional_value_unit_default_none(self):
        rel = Relation(**self._valid_relation(value=None, unit=None))
        assert rel.value is None
        assert rel.unit is None

    def test_qualitative_relation_no_value(self):
        rel = Relation(
            **self._valid_relation(
                measurement_type="qualitative",
                value=None,
                unit=None,
            )
        )
        assert rel.measurement_type == "qualitative"
        assert rel.value is None

    def test_all_polarity_values(self):
        for polarity in ("positive", "negative", "neutral"):
            rel = Relation(**self._valid_relation(polarity=polarity))
            assert rel.polarity == polarity


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------


class TestExtractionResult:
    def test_empty_result(self):
        result = ExtractionResult()
        assert result.entities == []
        assert result.relations == []

    def test_result_with_entities_and_relations(self):
        entity = Entity(
            surface="Scope 1 emissions",
            canonical="scope 1 emission",
            type="emission",
            esg="E",
        )
        relation = Relation(
            source="scope 1 emission",
            target="paris agreement",
            relation="COMPLIES_WITH",
            measurement_type="quantitative",
            polarity="positive",
            evidence="cut Scope 1 emissions by 40%",
            confidence=0.9,
            value=40.0,
            unit="%",
        )
        result = ExtractionResult(entities=[entity], relations=[relation])
        assert len(result.entities) == 1
        assert len(result.relations) == 1

    def test_model_dump_json_roundtrip(self):
        entity = Entity(
            surface="carbon emissions",
            canonical="carbon emission",
            type="emission",
            esg="E",
        )
        result = ExtractionResult(entities=[entity])
        json_str = result.model_dump_json()
        restored = ExtractionResult.model_validate_json(json_str)
        assert restored.entities[0].canonical == "carbon emission"
