"""ESG Knowledge Graph extraction pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from .models import ExtractionResult
from .normalize import clean_relations, deduplicate_entities, normalize_canonical_name
from .vertex_ai import call_vertex_model

logger = logging.getLogger(__name__)


def extract_kg(
    text: str,
    *,
    project: str | None = None,
    location: str | None = None,
    model_name: str | None = None,
) -> ExtractionResult:
    """Run the full ESG knowledge-graph extraction pipeline on *text*.

    Pipeline steps
    --------------
    1. Call the Vertex AI Gemini model to obtain a structured JSON response.
    2. Parse the raw JSON string.
    3. Validate the parsed data with Pydantic.
    4. Normalise entity canonical names.
    5. Deduplicate entities (by canonical name).
    6. Clean and deduplicate relations (remove references to unknown entities).

    Parameters
    ----------
    text:
        Source document text to analyse.
    project:
        Google Cloud project ID (forwarded to :func:`call_vertex_model`).
    location:
        Google Cloud region (forwarded to :func:`call_vertex_model`).
    model_name:
        Gemini model name (forwarded to :func:`call_vertex_model`).

    Returns
    -------
    ExtractionResult
        Validated, normalised, and deduplicated extraction result.
    """
    # ------------------------------------------------------------------ #
    # Step 1 — call the model                                             #
    # ------------------------------------------------------------------ #
    logger.info("Calling Vertex AI model …")
    raw_json = call_vertex_model(
        text,
        project=project,
        location=location,
        model_name=model_name,
    )

    # ------------------------------------------------------------------ #
    # Step 2 — parse raw JSON                                             #
    # ------------------------------------------------------------------ #
    try:
        data: Any = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse model JSON response: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Step 3 — validate with Pydantic                                     #
    # ------------------------------------------------------------------ #
    try:
        result = ExtractionResult.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Model response failed schema validation:\n{exc}") from exc

    logger.info(
        "Extracted %d entities and %d relations before deduplication.",
        len(result.entities),
        len(result.relations),
    )

    # ------------------------------------------------------------------ #
    # Step 4–5 — normalise canonical names and deduplicate entities       #
    # ------------------------------------------------------------------ #
    entities = deduplicate_entities(result.entities)

    # Build the set of valid canonical names after normalisation
    canonical_names = {normalize_canonical_name(e.canonical) for e in entities}

    # ------------------------------------------------------------------ #
    # Step 6 — clean and deduplicate relations                            #
    # ------------------------------------------------------------------ #
    relations = clean_relations(result.relations, canonical_names)

    logger.info(
        "After deduplication: %d entities and %d relations.",
        len(entities),
        len(relations),
    )

    return ExtractionResult(entities=entities, relations=relations)
