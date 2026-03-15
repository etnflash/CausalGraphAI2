"""Vertex AI / Gemini model call for ESG entity and relation extraction."""

from __future__ import annotations

import json
import os
import re

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

_DEFAULT_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
_DEFAULT_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
_DEFAULT_MODEL = os.environ.get("VERTEX_MODEL", "gemini-1.5-pro")

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert ESG (Environmental, Social, and Governance) knowledge-graph extraction assistant.

Your task is to read the provided text and extract:
1. All ESG-relevant entities.
2. Directed relationships between those entities.

=== ENTITY SCHEMA ===
Each entity must have:
  - surface:    exact text span from the source text
  - canonical:  normalised entity name (singular form when possible, lower-case)
  - type:       short semantic label (e.g. "emission", "policy", "company", "standard")
  - esg:        one of ["E", "S", "G"]
                  E = Environmental (climate, emissions, pollution, energy, water,
                                     waste, biodiversity, natural resources)
                  S = Social        (labor, health, safety, human rights, diversity,
                                     inclusion, community, education)
                  G = Governance    (board structure, corporate governance, corruption,
                                     compliance, regulation, executive compensation, audit)

IMPORTANT: Extract ONLY entities that genuinely belong to an ESG category.
Do NOT extract general business, financial, or geographic entities unless
they are directly relevant to an ESG topic.

=== RELATION SCHEMA ===
Each relation must have:
  - source:            canonical name of the source entity (must exist in entities list)
  - target:            canonical name of the target entity (must exist in entities list)
  - relation:          one of ["CAUSES","PROMOTES","INHIBITS","ASSOCIATED_WITH",
                                "PART_OF","IS_A","INCREASES","DECREASES",
                                "VIOLATES","COMPLIES_WITH"]
  - measurement_type:  "quantitative" if the evidence contains a numeric measurement
                       (percentage, count, score, amount); otherwise "qualitative"
  - polarity:          "positive" | "negative" | "neutral"
  - evidence:          verbatim text span that supports this relation
  - confidence:        float between 0.0 and 1.0
  - value:             numeric value (only for quantitative; null otherwise)
  - unit:              unit string  (only for quantitative; null otherwise)

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):

{
  "entities": [
    {
      "surface": "...",
      "canonical": "...",
      "type": "...",
      "esg": "E" | "S" | "G"
    }
  ],
  "relations": [
    {
      "source": "...",
      "target": "...",
      "relation": "...",
      "measurement_type": "qualitative" | "quantitative",
      "polarity": "positive" | "negative" | "neutral",
      "evidence": "...",
      "confidence": 0.0,
      "value": null,
      "unit": null
    }
  ]
}
"""


def _build_prompt(text: str) -> str:
    """Combine the system prompt with the user-supplied text."""
    return f"{_SYSTEM_PROMPT}\n\n=== TEXT TO ANALYSE ===\n{text}"


def _strip_markdown_fences(raw: str) -> str:
    """Remove optional ```json ... ``` or ``` ... ``` fences from the response."""
    raw = raw.strip()
    # Remove leading fence (with optional language tag)
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    # Remove trailing fence
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_vertex_model(
    text: str,
    *,
    project: str | None = None,
    location: str | None = None,
    model_name: str | None = None,
) -> str:
    """Send *text* to a Vertex AI Gemini model and return raw JSON text.

    Parameters
    ----------
    text:
        The source document text to analyse.
    project:
        Google Cloud project ID. Defaults to the ``GOOGLE_CLOUD_PROJECT``
        environment variable.
    location:
        Google Cloud region. Defaults to the ``GOOGLE_CLOUD_LOCATION``
        environment variable or ``"us-central1"``.
    model_name:
        Gemini model name. Defaults to the ``VERTEX_MODEL`` environment
        variable or ``"gemini-1.5-pro"``.

    Returns
    -------
    str
        Raw JSON string returned by the model (markdown fences stripped).
    """
    resolved_project = project or _DEFAULT_PROJECT
    resolved_location = location or _DEFAULT_LOCATION
    resolved_model = model_name or _DEFAULT_MODEL

    vertexai.init(project=resolved_project, location=resolved_location)

    model = GenerativeModel(
        resolved_model,
        generation_config=GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=8192,
        ),
    )

    prompt = _build_prompt(text)
    response = model.generate_content(prompt)
    raw = response.text

    # Validate that the response is parseable JSON before returning
    cleaned = _strip_markdown_fences(raw)
    try:
        json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Vertex AI model returned non-JSON output: {exc}\n\nRaw output:\n{raw}"
        ) from exc

    return cleaned
