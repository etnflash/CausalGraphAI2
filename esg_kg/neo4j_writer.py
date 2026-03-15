"""Neo4j writer for ESG Knowledge Graph results."""

from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError

from .models import ExtractionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

_MERGE_ENTITY_QUERY = """
MERGE (e:Entity {canonical: $canonical})
SET   e.surface  = $surface,
      e.type     = $type,
      e.esg      = $esg
"""

_MERGE_RELATION_QUERY = """
MATCH (src:Entity {canonical: $source})
MATCH (tgt:Entity {canonical: $target})
MERGE (src)-[r:RELATION {type: $relation, source: $source, target: $target}]->(tgt)
SET   r.measurement_type = $measurement_type,
      r.polarity         = $polarity,
      r.confidence       = $confidence,
      r.evidence         = $evidence,
      r.value            = $value,
      r.unit             = $unit
"""


# ---------------------------------------------------------------------------
# Writer class
# ---------------------------------------------------------------------------


class Neo4jWriter:
    """Writes ESG extraction results to a Neo4j database.

    Parameters
    ----------
    uri:
        Bolt or neo4j URI, e.g. ``"bolt://localhost:7687"``.
    user:
        Database username.
    password:
        Database password.
    database:
        Target database name. Defaults to ``"neo4j"``.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(
            uri, auth=(user, password)
        )
        self._database = database

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Neo4jWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def write_extraction(self, result: ExtractionResult) -> None:
        """Persist all entities and relations from *result* into Neo4j.

        Nodes and relationships are written with ``MERGE`` so the method
        is idempotent — repeated calls with the same data will not create
        duplicates.

        Parameters
        ----------
        result:
            A validated :class:`~esg_kg.models.ExtractionResult` instance.
        """
        with self._driver.session(database=self._database) as session:
            # Write entities first so relation MATCHes can find them
            for entity in result.entities:
                session.execute_write(
                    self._write_entity,
                    canonical=entity.canonical,
                    surface=entity.surface,
                    type_=entity.type,
                    esg=entity.esg,
                )
                logger.debug("Merged entity: %s (%s)", entity.canonical, entity.esg)

            # Write relations
            skipped = 0
            for rel in result.relations:
                try:
                    session.execute_write(
                        self._write_relation,
                        source=rel.source,
                        target=rel.target,
                        relation=rel.relation,
                        measurement_type=rel.measurement_type,
                        polarity=rel.polarity,
                        confidence=rel.confidence,
                        evidence=rel.evidence,
                        value=rel.value,
                        unit=rel.unit,
                    )
                    logger.debug(
                        "Merged relation: %s -[%s]-> %s",
                        rel.source,
                        rel.relation,
                        rel.target,
                    )
                except Neo4jError as exc:
                    logger.warning(
                        "Could not write relation %s -[%s]-> %s: %s",
                        rel.source,
                        rel.relation,
                        rel.target,
                        exc,
                    )
                    skipped += 1

        logger.info(
            "Wrote %d entities and %d relations to Neo4j (%d relation(s) skipped).",
            len(result.entities),
            len(result.relations) - skipped,
            skipped,
        )

    # ------------------------------------------------------------------
    # Internal transaction functions
    # ------------------------------------------------------------------

    @staticmethod
    def _write_entity(
        tx: Any,
        *,
        canonical: str,
        surface: str,
        type_: str,
        esg: str,
    ) -> None:
        tx.run(
            _MERGE_ENTITY_QUERY,
            canonical=canonical,
            surface=surface,
            type=type_,
            esg=esg,
        )

    @staticmethod
    def _write_relation(
        tx: Any,
        *,
        source: str,
        target: str,
        relation: str,
        measurement_type: str,
        polarity: str,
        confidence: float,
        evidence: str,
        value: float | None,
        unit: str | None,
    ) -> None:
        tx.run(
            _MERGE_RELATION_QUERY,
            source=source,
            target=target,
            relation=relation,
            measurement_type=measurement_type,
            polarity=polarity,
            confidence=confidence,
            evidence=evidence,
            value=value,
            unit=unit,
        )
