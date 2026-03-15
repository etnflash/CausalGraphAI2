"""Pydantic models for the ESG Knowledge Graph extraction pipeline."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Allowed relation types
# ---------------------------------------------------------------------------

RelationType = Literal[
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


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    """A single ESG-related entity extracted from text."""

    surface: str = Field(
        ...,
        description="Original text span as it appears in the source text.",
    )
    canonical: str = Field(
        ...,
        description="Normalized entity name, preferring singular form.",
    )
    type: str = Field(
        ...,
        description="Short semantic type label (e.g. 'emission', 'policy', 'company').",
    )
    esg: Literal["E", "S", "G"] = Field(
        ...,
        description=(
            "ESG category: E=Environmental, S=Social, G=Governance."
        ),
    )


class Relation(BaseModel):
    """A directed relationship between two ESG entities."""

    source: str = Field(
        ...,
        description="Canonical name of the source entity.",
    )
    target: str = Field(
        ...,
        description="Canonical name of the target entity.",
    )
    relation: RelationType = Field(
        ...,
        description="Semantic relation type.",
    )
    measurement_type: Literal["qualitative", "quantitative"] = Field(
        ...,
        description=(
            "'quantitative' when the evidence contains a numeric measurement "
            "(percentage, count, score, amount); otherwise 'qualitative'."
        ),
    )
    polarity: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="Whether this relation has a positive, negative, or neutral impact.",
    )
    evidence: str = Field(
        ...,
        description="Verbatim text span that supports this relation.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0 and 1.",
    )
    value: Optional[float] = Field(
        default=None,
        description="Numeric value for quantitative relations.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement for quantitative relations.",
    )


class ExtractionResult(BaseModel):
    """Full extraction result containing entities and relations."""

    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
