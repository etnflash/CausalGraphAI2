"""ESG Knowledge Graph extraction pipeline.

Public API
----------
>>> from esg_kg import extract_kg, Neo4jWriter
>>> from esg_kg.models import Entity, Relation, ExtractionResult
"""

from .models import Entity, ExtractionResult, Relation
from .neo4j_writer import Neo4jWriter
from .normalize import clean_relations, deduplicate_entities, normalize_canonical_name
from .pipeline import extract_kg
from .vertex_ai import call_vertex_model

__all__ = [
    # models
    "Entity",
    "Relation",
    "ExtractionResult",
    # normalization
    "normalize_canonical_name",
    "deduplicate_entities",
    "clean_relations",
    # vertex ai
    "call_vertex_model",
    # pipeline
    "extract_kg",
    # neo4j
    "Neo4jWriter",
]
