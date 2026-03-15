"""Example entry-point demonstrating the end-to-end ESG KG pipeline.

Usage
-----
Set the following environment variables before running::

    export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
    export GOOGLE_CLOUD_LOCATION="us-central1"   # optional, default shown
    export VERTEX_MODEL="gemini-1.5-pro"         # optional, default shown
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="your-password"

Then execute::

    python -m esg_kg.main
"""

from __future__ import annotations

import json
import logging
import os

from .neo4j_writer import Neo4jWriter
from .pipeline import extract_kg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Sample text
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
Acme Corp has committed to reducing its carbon emissions by 40% before 2030,
aligning with the Paris Agreement targets. The company currently discloses
its Scope 1 and Scope 2 greenhouse gas emissions annually in accordance with
the GHG Protocol standard.

Water consumption at the main manufacturing plant rose by 12% last year due to
increased production volumes, raising concerns about long-term water scarcity
risks in the region.

The board of directors approved a new diversity and inclusion policy that sets
a target of 40% female representation at the senior management level by 2025.
The independent audit committee reviewed executive compensation packages to
ensure alignment with long-term sustainability goals.

A recent environmental impact assessment found that the company's supply chain
activities contribute to deforestation in Southeast Asia, violating its own
supplier code of conduct on biodiversity protection.
"""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def main() -> None:
    """Run ESG extraction on sample text, write to Neo4j, and print results."""
    # -- 1. Extract knowledge graph ----------------------------------------
    print("Running ESG knowledge graph extraction …\n")
    result = extract_kg(
        SAMPLE_TEXT,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
        model_name=os.environ.get("VERTEX_MODEL"),
    )

    # -- 2. Print structured result ----------------------------------------
    print("=== Extraction Result ===\n")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print()

    # -- 3. Write to Neo4j --------------------------------------------------
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    print(f"Writing results to Neo4j at {neo4j_uri} …")
    with Neo4jWriter(neo4j_uri, neo4j_user, neo4j_password) as writer:
        writer.write_extraction(result)
    print("Done.")


if __name__ == "__main__":
    main()
