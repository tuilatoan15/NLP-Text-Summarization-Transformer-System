"""CLI wrapper for document ingest.

Run:
    python -m scripts.ingest_document data/sample.pdf --output storage/ingest/sample.json --pretty
"""

from __future__ import annotations

from pipeline.ingest_pipeline import main


if __name__ == "__main__":
    main()
