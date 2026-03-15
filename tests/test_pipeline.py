"""Tests for esg_kg.pipeline."""

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from esg_kg.models import ExtractionResult
from esg_kg.pipeline import extract_kg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_json(entities=None, relations=None) -> str:
    return json.dumps(
        {
            "entities": entities or [],
            "relations": relations or [],
        }
    )


_VALID_ENTITY = {
    "surface": "Scope 1 emissions",
    "canonical": "scope 1 emission",
    "type": "emission",
    "esg": "E",
}

_VALID_RELATION = {
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

_VALID_ENTITY_2 = {
    "surface": "Paris Agreement",
    "canonical": "paris agreement",
    "type": "standard",
    "esg": "E",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractKg:
    def _patch_model(self, raw_json: str):
        return patch("esg_kg.pipeline.call_vertex_model", return_value=raw_json)

    def test_returns_extraction_result(self):
        raw = _make_raw_json()
        with self._patch_model(raw):
            result = extract_kg("some text")
        assert isinstance(result, ExtractionResult)

    def test_empty_response_returns_empty_result(self):
        raw = _make_raw_json()
        with self._patch_model(raw):
            result = extract_kg("text")
        assert result.entities == []
        assert result.relations == []

    def test_entities_extracted_and_normalized(self):
        raw = _make_raw_json(entities=[_VALID_ENTITY])
        with self._patch_model(raw):
            result = extract_kg("text")
        assert len(result.entities) == 1
        assert result.entities[0].canonical == "scope 1 emission"

    def test_relations_extracted(self):
        raw = _make_raw_json(
            entities=[_VALID_ENTITY, _VALID_ENTITY_2],
            relations=[_VALID_RELATION],
        )
        with self._patch_model(raw):
            result = extract_kg("text")
        assert len(result.relations) == 1
        assert result.relations[0].relation == "COMPLIES_WITH"

    def test_duplicate_entities_deduplicated(self):
        duplicate = dict(_VALID_ENTITY)
        raw = _make_raw_json(entities=[_VALID_ENTITY, duplicate])
        with self._patch_model(raw):
            result = extract_kg("text")
        assert len(result.entities) == 1

    def test_relation_with_unknown_entity_filtered(self):
        # Relation references "unknown entity" which is not in entities
        raw = _make_raw_json(
            entities=[_VALID_ENTITY],
            relations=[
                dict(_VALID_RELATION, target="unknown entity")
            ],
        )
        with self._patch_model(raw):
            result = extract_kg("text")
        assert result.relations == []

    def test_invalid_json_raises_value_error(self):
        with self._patch_model("this is not json"):
            with pytest.raises(ValueError, match="Failed to parse"):
                extract_kg("text")

    def test_schema_validation_error_raises_value_error(self):
        # Provide valid JSON but with an invalid entity (bad esg value)
        bad_entity = dict(_VALID_ENTITY, esg="X")
        raw = _make_raw_json(entities=[bad_entity])
        with self._patch_model(raw):
            with pytest.raises(ValueError, match="schema validation"):
                extract_kg("text")

    def test_kwargs_forwarded_to_vertex_model(self):
        raw = _make_raw_json()
        with patch("esg_kg.pipeline.call_vertex_model", return_value=raw) as mock_call:
            extract_kg(
                "text",
                project="my-proj",
                location="eu",
                model_name="gemini-2.0",
            )
        mock_call.assert_called_once_with(
            "text",
            project="my-proj",
            location="eu",
            model_name="gemini-2.0",
        )

    def test_irregular_plural_in_entity_normalized(self):
        entity = dict(_VALID_ENTITY, canonical="emissions")
        raw = _make_raw_json(entities=[entity])
        with self._patch_model(raw):
            result = extract_kg("text")
        # "emissions" should be normalized to "emission"
        assert result.entities[0].canonical == "emission"

    def test_relation_source_target_normalized(self):
        entity1 = dict(_VALID_ENTITY, canonical="emissions")
        entity2 = dict(_VALID_ENTITY_2)
        relation = dict(
            _VALID_RELATION,
            source="emissions",
            target="paris agreement",
        )
        raw = _make_raw_json(entities=[entity1, entity2], relations=[relation])
        with self._patch_model(raw):
            result = extract_kg("text")
        assert len(result.relations) == 1
        assert result.relations[0].source == "emission"
