"""Tests for esg_kg.neo4j_writer."""

from unittest.mock import MagicMock, call, patch

import pytest

from esg_kg.models import Entity, ExtractionResult, Relation
from esg_kg.neo4j_writer import Neo4jWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(canonical: str = "carbon emission", esg: str = "E") -> Entity:
    return Entity(
        surface="Carbon emissions",
        canonical=canonical,
        type="emission",
        esg=esg,
    )


def _make_relation(
    source: str = "carbon emission",
    target: str = "climate change",
) -> Relation:
    return Relation(
        source=source,
        target=target,
        relation="CAUSES",
        measurement_type="qualitative",
        polarity="negative",
        evidence="Carbon emissions cause climate change.",
        confidence=0.9,
    )


def _make_writer_with_mock_driver():
    """Return a (writer, mock_driver) pair with the Neo4j driver mocked out."""
    with patch("esg_kg.neo4j_writer.GraphDatabase.driver") as mock_driver_factory:
        mock_driver = MagicMock()
        mock_driver_factory.return_value = mock_driver
        writer = Neo4jWriter("bolt://localhost:7687", "neo4j", "password")
    return writer, mock_driver


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestNeo4jWriterInit:
    def test_driver_created_with_credentials(self):
        with patch(
            "esg_kg.neo4j_writer.GraphDatabase.driver"
        ) as mock_driver_factory:
            Neo4jWriter("bolt://localhost:7687", "user", "pass")
            mock_driver_factory.assert_called_once_with(
                "bolt://localhost:7687", auth=("user", "pass")
            )

    def test_default_database_is_neo4j(self):
        with patch("esg_kg.neo4j_writer.GraphDatabase.driver"):
            writer = Neo4jWriter("bolt://localhost:7687", "user", "pass")
        assert writer._database == "neo4j"

    def test_custom_database_stored(self):
        with patch("esg_kg.neo4j_writer.GraphDatabase.driver"):
            writer = Neo4jWriter(
                "bolt://localhost:7687", "user", "pass", database="esg_db"
            )
        assert writer._database == "esg_db"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestNeo4jWriterContextManager:
    def test_enter_returns_self(self):
        writer, _ = _make_writer_with_mock_driver()
        result = writer.__enter__()
        assert result is writer

    def test_exit_calls_close(self):
        writer, mock_driver = _make_writer_with_mock_driver()
        writer.__exit__(None, None, None)
        mock_driver.close.assert_called_once()

    def test_with_statement_closes_on_exit(self):
        with patch(
            "esg_kg.neo4j_writer.GraphDatabase.driver"
        ) as mock_driver_factory:
            mock_driver = MagicMock()
            mock_driver_factory.return_value = mock_driver
            mock_session = MagicMock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = False

            with Neo4jWriter("bolt://localhost:7687", "u", "p"):
                pass

        mock_driver.close.assert_called_once()


# ---------------------------------------------------------------------------
# write_extraction
# ---------------------------------------------------------------------------


class TestWriteExtraction:
    def _setup_mock_session(self, mock_driver):
        """Configure mock_driver to return a usable mock session."""
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = False
        return mock_session

    def test_empty_result_no_writes(self):
        writer, mock_driver = _make_writer_with_mock_driver()
        mock_session = self._setup_mock_session(mock_driver)

        writer.write_extraction(ExtractionResult())
        mock_session.execute_write.assert_not_called()

    def test_entity_written_via_execute_write(self):
        writer, mock_driver = _make_writer_with_mock_driver()
        mock_session = self._setup_mock_session(mock_driver)

        result = ExtractionResult(entities=[_make_entity()])
        writer.write_extraction(result)

        assert mock_session.execute_write.call_count == 1

    def test_relation_written_via_execute_write(self):
        writer, mock_driver = _make_writer_with_mock_driver()
        mock_session = self._setup_mock_session(mock_driver)

        result = ExtractionResult(
            entities=[_make_entity("carbon emission"), _make_entity("climate change")],
            relations=[_make_relation()],
        )
        writer.write_extraction(result)

        # 2 entities + 1 relation = 3 execute_write calls
        assert mock_session.execute_write.call_count == 3

    def test_entity_written_with_correct_kwargs(self):
        writer, mock_driver = _make_writer_with_mock_driver()
        mock_session = self._setup_mock_session(mock_driver)

        entity = _make_entity("carbon emission")
        result = ExtractionResult(entities=[entity])
        writer.write_extraction(result)

        call_kwargs = mock_session.execute_write.call_args[1]
        assert call_kwargs["canonical"] == "carbon emission"
        assert call_kwargs["surface"] == "Carbon emissions"
        assert call_kwargs["esg"] == "E"

    def test_relation_written_with_correct_kwargs(self):
        writer, mock_driver = _make_writer_with_mock_driver()
        mock_session = self._setup_mock_session(mock_driver)

        rel = _make_relation()
        result = ExtractionResult(
            entities=[_make_entity("carbon emission"), _make_entity("climate change")],
            relations=[rel],
        )
        writer.write_extraction(result)

        # The last execute_write call is for the relation
        rel_call_kwargs = mock_session.execute_write.call_args_list[-1][1]
        assert rel_call_kwargs["source"] == "carbon emission"
        assert rel_call_kwargs["target"] == "climate change"
        assert rel_call_kwargs["relation"] == "CAUSES"
        assert rel_call_kwargs["confidence"] == 0.9

    def test_neo4j_error_on_relation_is_logged_not_raised(self):
        """A Neo4jError during relation write should be caught and logged."""
        from neo4j.exceptions import Neo4jError

        writer, mock_driver = _make_writer_with_mock_driver()
        mock_session = self._setup_mock_session(mock_driver)

        entity_write_count = 0

        def execute_write_side_effect(fn, **kwargs):
            nonlocal entity_write_count
            # First two calls are entities — succeed; third call is relation — fail
            if entity_write_count < 2:
                entity_write_count += 1
            else:
                raise Neo4jError("Node not found")

        mock_session.execute_write.side_effect = execute_write_side_effect

        result = ExtractionResult(
            entities=[_make_entity("carbon emission"), _make_entity("climate change")],
            relations=[_make_relation()],
        )
        # Should not raise
        writer.write_extraction(result)

    def test_session_opened_with_correct_database(self):
        with patch(
            "esg_kg.neo4j_writer.GraphDatabase.driver"
        ) as mock_driver_factory:
            mock_driver = MagicMock()
            mock_driver_factory.return_value = mock_driver
            mock_session = MagicMock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = False

            writer = Neo4jWriter(
                "bolt://localhost:7687", "u", "p", database="custom_db"
            )
            writer.write_extraction(ExtractionResult())

        mock_driver.session.assert_called_with(database="custom_db")
